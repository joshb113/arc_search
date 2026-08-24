"""Fetch-layer tests.

Split in two. The header/magic-byte screening is pure and gets exhaustive
parametrized coverage. The client behaviour -- retries, the streaming size cap,
the robots gate -- is exercised against respx so it runs in CI with no network.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from arc_search.config import CrawlSettings
from arc_search.crawler.fetch import (
    Fetched,
    Fetcher,
    FetchError,
    Skipped,
    backoff_delay,
    is_html,
    normalize_content_type,
    screen_image_headers,
    sniff_image_type,
)
from arc_search.crawler.politeness import Politeness
from imagefixtures import make_image

# Real encoded images. get_image() reads headers now, so a hand-written magic
# prefix would be rejected as unreadable_header. See conftest.
JPEG = make_image(0, "JPEG")
PNG = make_image(1, "PNG")

# Deliberately NOT decodable: magic bytes only. Used where the test asserts on
# behaviour that must fire before the header is ever parsed.
FAKE_JPEG_PREFIX = b"\xff\xd8\xff\xe0" + b"\x00" * 20_000


def cfg(**kw) -> CrawlSettings:
    base = {"respect_robots": False, "max_retries": 3, "backoff_base_s": 0.0}
    return CrawlSettings(**{**base, **kw})


# --- pure: content type ----------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("image/JPEG; charset=binary", "image/jpeg"),
        ("  text/html ", "text/html"),
        ("text/html;charset=UTF-8", "text/html"),
        (None, ""),
        ("", ""),
    ],
)
def test_normalize_content_type(raw, expected):
    assert normalize_content_type(raw) == expected


def test_is_html_accepts_xhtml():
    assert is_html("text/html; charset=utf-8")
    assert is_html("application/xhtml+xml")
    assert not is_html("image/jpeg")
    assert not is_html(None)


# --- pure: magic bytes -----------------------------------------------------


@pytest.mark.parametrize(
    ("head", "expected"),
    [
        (b"\xff\xd8\xff\xe0\x00\x10JFIF", "image/jpeg"),
        (b"\x89PNG\r\n\x1a\n\x00\x00\x00\x0d", "image/png"),
        (b"RIFF\x24\x00\x00\x00WEBPVP8 ", "image/webp"),
        (b"\x00\x00\x00\x20ftypavif\x00\x00\x00\x00", "image/avif"),
        (b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00", "image/heic"),
        (b"GIF89a\x00\x00", "image/gif"),
        (b"<!DOCTYPE html>", None),
        (b"", None),
        # RIFF but not WEBP -- a .wav. Must not be claimed as an image.
        (b"RIFF\x24\x00\x00\x00WAVEfmt ", None),
    ],
)
def test_sniff_image_type(head, expected):
    assert sniff_image_type(head) == expected


# --- pure: header screening ------------------------------------------------


@pytest.mark.parametrize(
    ("ctype", "length", "ok", "reason_prefix"),
    [
        ("image/jpeg", 50_000, True, "ok"),
        ("image/png", None, True, "ok"),  # chunked: unknown length is fine
        ("image/webp", 50_000, True, "ok"),
        # octet-stream is useless, not disqualifying -- sniffing decides later.
        ("application/octet-stream", 50_000, True, "ok"),
        (None, 50_000, True, "ok"),
        # positive evidence of uselessness
        ("image/svg+xml", 50_000, False, "image_subtype_unusable"),
        ("image/gif", 50_000, False, "image_subtype_unusable"),
        ("text/html", 50_000, False, "not_an_image"),
        ("application/pdf", 50_000, False, "not_an_image"),
        # size gates
        ("image/jpeg", 100, False, "too_small"),
        ("image/jpeg", 500_000_000, False, "too_large"),
    ],
)
def test_screen_image_headers(ctype, length, ok, reason_prefix):
    v = screen_image_headers(ctype, length, cfg())
    assert v.ok is ok
    assert v.reason.split(":")[0] == reason_prefix


def test_backoff_is_bounded_and_jittered():
    c = cfg(backoff_base_s=1.0, backoff_max_s=10.0)
    import random

    rnd = random.Random(0)
    seen = {backoff_delay(3, c, rand=rnd) for _ in range(50)}
    assert len(seen) > 1, "unjittered backoff marches retries back in lockstep"
    assert all(0.0 <= d <= 10.0 for d in seen)
    # And the cap actually binds at high attempt counts.
    assert all(backoff_delay(30, c, rand=rnd) <= 10.0 for _ in range(20))


# --- client behaviour ------------------------------------------------------


async def _fetcher(client: httpx.AsyncClient, c: CrawlSettings) -> Fetcher:
    async def no_sleep(_seconds: float) -> None:
        return None

    return Fetcher(c, client, Politeness(c, client), sleep=no_sleep)


@pytest.mark.asyncio
@respx.mock
async def test_transient_5xx_is_retried_then_succeeds():
    route = respx.get("https://h.test/p").mock(
        side_effect=[
            httpx.Response(503),
            httpx.Response(503),
            httpx.Response(200, html="<html><title>ok</title></html>"),
        ]
    )
    async with httpx.AsyncClient() as client:
        f = await _fetcher(client, cfg())
        page = await f.get_page("https://h.test/p")
    assert route.call_count == 3
    assert "ok" in page.text


@pytest.mark.asyncio
@respx.mock
async def test_404_is_not_retried():
    """Retrying a 404 burns the host's rate budget to learn nothing."""
    route = respx.get("https://h.test/gone").mock(return_value=httpx.Response(404))
    async with httpx.AsyncClient() as client:
        f = await _fetcher(client, cfg())
        with pytest.raises(FetchError) as ei:
            await f.get_page("https://h.test/gone")
    assert route.call_count == 1
    assert ei.value.retryable is False


@pytest.mark.asyncio
@respx.mock
async def test_retries_are_exhausted_then_reported_retryable():
    respx.get("https://h.test/p").mock(return_value=httpx.Response(503))
    async with httpx.AsyncClient() as client:
        f = await _fetcher(client, cfg(max_retries=3))
        with pytest.raises(FetchError) as ei:
            await f.get_page("https://h.test/p")
    assert ei.value.retryable is True
    assert "exhausted_retries" in ei.value.reason


@pytest.mark.asyncio
@respx.mock
async def test_non_html_page_is_a_skip_not_an_error():
    """A skip is a success: the URL was evaluated and correctly rejected."""
    respx.get("https://h.test/x").mock(
        return_value=httpx.Response(200, content=JPEG, headers={"content-type": "image/jpeg"})
    )
    async with httpx.AsyncClient() as client:
        f = await _fetcher(client, cfg())
        with pytest.raises(Skipped, match="not_html"):
            await f.get_page("https://h.test/x")


@pytest.mark.asyncio
@respx.mock
async def test_image_declared_too_large_is_never_downloaded():
    respx.get("https://h.test/big.jpg").mock(
        return_value=httpx.Response(
            200,
            content=JPEG,
            headers={"content-type": "image/jpeg", "content-length": "999999999"},
        )
    )
    async with httpx.AsyncClient() as client:
        f = await _fetcher(client, cfg())
        with pytest.raises(Skipped, match="too_large"):
            await f.get_image("https://h.test/big.jpg")


@pytest.mark.asyncio
@respx.mock
async def test_lying_content_length_is_caught_mid_stream():
    """A header is a suggestion. The cap has to be enforced on real bytes."""
    respx.get("https://h.test/liar.jpg").mock(
        return_value=httpx.Response(
            200,
            content=b"\xff\xd8\xff" + b"\x00" * 200_000,
            headers={"content-type": "image/jpeg", "content-length": "20000"},
        )
    )
    async with httpx.AsyncClient() as client:
        f = await _fetcher(client, cfg(max_image_bytes=50_000))
        with pytest.raises(Skipped, match="stream_exceeded_cap"):
            await f.get_image("https://h.test/liar.jpg")


@pytest.mark.asyncio
@respx.mock
async def test_octet_stream_jpeg_is_recovered_by_sniffing():
    """Servers mislabel real images constantly; trusting the header loses them."""
    respx.get("https://h.test/mystery").mock(
        return_value=httpx.Response(
            200, content=JPEG, headers={"content-type": "application/octet-stream"}
        )
    )
    async with httpx.AsyncClient() as client:
        f = await _fetcher(client, cfg())
        got = await f.get_image("https://h.test/mystery")
    assert got.content_type == "image/jpeg"


@pytest.mark.asyncio
@respx.mock
async def test_html_served_as_an_image_is_rejected_by_sniffing():
    respx.get("https://h.test/notreally.jpg").mock(
        return_value=httpx.Response(
            200,
            content=b"<!DOCTYPE html>" + b" " * 20_000,
            headers={"content-type": "image/jpeg"},
        )
    )
    async with httpx.AsyncClient() as client:
        f = await _fetcher(client, cfg())
        with pytest.raises(Skipped, match="not_image_bytes"):
            await f.get_image("https://h.test/notreally.jpg")


@pytest.mark.asyncio
@respx.mock
async def test_gif_passes_headers_but_is_rejected_on_sniff():
    """content-type lies in the useful direction too."""
    respx.get("https://h.test/a.gif").mock(
        return_value=httpx.Response(
            200,
            content=b"GIF89a" + b"\x00" * 20_000,
            headers={"content-type": "application/octet-stream"},
        )
    )
    async with httpx.AsyncClient() as client:
        f = await _fetcher(client, cfg())
        with pytest.raises(Skipped, match="image_subtype_unusable"):
            await f.get_image("https://h.test/a.gif")


# --- the choke point -------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_robots_disallow_blocks_the_fetch_entirely():
    """Non-negotiable #6. There must be no path to bytes that skips this."""
    respx.get("https://h.test/robots.txt").mock(
        return_value=httpx.Response(200, text="User-agent: *\nDisallow: /private/\n")
    )
    page = respx.get("https://h.test/private/x").mock(
        return_value=httpx.Response(200, html="<html></html>")
    )
    async with httpx.AsyncClient() as client:
        f = await _fetcher(client, CrawlSettings(respect_robots=True, backoff_base_s=0.0))
        with pytest.raises(Skipped, match="robots_disallow"):
            await f.get_page("https://h.test/private/x")
    assert page.call_count == 0, "robots check must happen before the request"


@pytest.mark.asyncio
@respx.mock
async def test_missing_robots_txt_means_allow_all():
    """archive.fosdem.org has no robots.txt. A 404 is not a disallow."""
    respx.get("https://h.test/robots.txt").mock(return_value=httpx.Response(404))
    respx.get("https://h.test/p").mock(return_value=httpx.Response(200, html="<html>hi</html>"))
    async with httpx.AsyncClient() as client:
        f = await _fetcher(client, CrawlSettings(respect_robots=True, backoff_base_s=0.0))
        got = await f.get_page("https://h.test/p")
    assert got.status == 200


@pytest.mark.asyncio
@respx.mock
async def test_forbidden_robots_txt_fails_closed():
    """401/403 on robots.txt is an explicit 'not for you'."""
    respx.get("https://h.test/robots.txt").mock(return_value=httpx.Response(403))
    async with httpx.AsyncClient() as client:
        f = await _fetcher(client, CrawlSettings(respect_robots=True, backoff_base_s=0.0))
        with pytest.raises(Skipped, match="robots_disallow"):
            await f.get_page("https://h.test/p")


def test_fetched_text_survives_bad_encoding():
    f = Fetched(
        url="u",
        final_url="u",
        status=200,
        content_type="text/html",
        kind="html",
        body=b"caf\xe9",  # latin-1 in a utf-8 world
    )
    assert "caf" in f.text  # replaced, not raised
