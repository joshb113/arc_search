"""Crawl-loop tests against a fake three-page site.

These are the tests that would have caught eye_of_web's actual crawl failures:
state lost on restart, scope not enforced on discovered links, and a queue that
cannot tell "finished" from "momentarily empty".

No network. respx serves the site; the frontiers are real SQLite files in tmp.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from arc_search.config import CrawlSettings
from arc_search.crawler.fetch import Fetcher
from arc_search.crawler.frontier import Frontier
from arc_search.crawler.politeness import Politeness
from arc_search.crawler.run import Crawler, MetadataSink, all_loopback, check_user_agent
from arc_search.crawler.seeds import SeedConfig, Vertical
from arc_search.index.dedup import Deduper
from imagefixtures import make_image

# Real encoded JPEGs. The fetch path reads image headers now, so a hand-written
# magic-byte prefix is no longer a usable fixture. See conftest.
JPEG_A = make_image(1, "JPEG")
JPEG_B = make_image(2, "JPEG")

INDEX = """
<html><head><title>Speakers</title></head><body>
  <a href="/speaker/ada/">Ada</a>
  <a href="/speaker/grace/">Grace</a>
  <a href="https://offsite.test/elsewhere">offsite</a>
  <a href="/export.pdf">slides</a>
</body></html>
"""

SPEAKER = """
<html><head><title>{name}</title></head><body>
  <img src="{img}" class="speaker-photo" alt="Photo of {name}">
</body></html>
"""


def _site() -> None:
    respx.get("https://conf.test/robots.txt").mock(return_value=httpx.Response(404))
    respx.get("https://conf.test/speakers/").mock(return_value=httpx.Response(200, html=INDEX))
    respx.get("https://conf.test/speaker/ada/").mock(
        return_value=httpx.Response(200, html=SPEAKER.format(name="Ada", img="/i/ada.jpg"))
    )
    respx.get("https://conf.test/speaker/grace/").mock(
        return_value=httpx.Response(200, html=SPEAKER.format(name="Grace", img="/i/grace.jpg"))
    )
    for path, blob in (("ada", JPEG_A), ("grace", JPEG_B)):
        respx.get(f"https://conf.test/i/{path}.jpg").mock(
            return_value=httpx.Response(200, content=blob, headers={"content-type": "image/jpeg"})
        )


def _seeds() -> SeedConfig:
    return SeedConfig(
        verticals=[
            Vertical(
                name="conf",
                enabled=True,
                seeds=["https://conf.test/speakers/"],
                allow_hosts=["conf.test"],
                deny_patterns=[r"\.pdf$"],
                max_depth=3,
            )
        ],
        global_deny_hosts=["facebook.com"],
    )


def _cfg(**kw) -> CrawlSettings:
    base = {
        "respect_robots": True,  # the robots GATE stays on; these tests rely on it
        "backoff_base_s": 0.0,
        "concurrency": 4,
        "max_retries": 2,
        # The rate LIMITER is turned up out of the way. At the 0.5 rps default
        # these tests spend ~6.5s each waiting on a token bucket against a fake
        # host, which is 40s of a CI budget that has to stay under a minute.
        # Politeness timing has its own coverage in test_fetch.py; what is under
        # test here is the loop.
        "per_host_rps": 1000.0,
        "per_host_burst": 100,
    }
    return CrawlSettings(**{**base, **kw})


async def _build(tmp_path, cfg, client, seeds=None):
    pages = Frontier(tmp_path / "f.sqlite")
    images = Frontier(tmp_path / "f-images.sqlite")
    sink = MetadataSink(tmp_path / "images.jsonl", Deduper())
    crawler = Crawler(
        cfg,
        seeds or _seeds(),
        Fetcher(cfg, client, Politeness(cfg, client)),
        pages,
        images,
        sink,
    )
    return crawler, pages, images, sink


def _rows(tmp_path):
    p = tmp_path / "images.jsonl"
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_full_crawl_discovers_pages_and_images(tmp_path):
    _site()
    async with httpx.AsyncClient() as client:
        crawler, pages, images, sink = await _build(tmp_path, _cfg(), client)
        stats = await crawler.run()
        sink.close()

    assert stats.pages_fetched == 3  # index + two speakers
    assert stats.images_fetched == 2
    rows = _rows(tmp_path)
    assert {r["url"] for r in rows} == {
        "https://conf.test/i/ada.jpg",
        "https://conf.test/i/grace.jpg",
    }
    # The loop terminated on its own rather than spinning.
    assert pages.stats()["pending"] == 0
    assert images.stats()["pending"] == 0


@pytest.mark.asyncio
@respx.mock
async def test_alt_text_is_carried_through_to_the_record(tmp_path):
    """FOSDEM's `alt="Photo of NAME"` is a free weak label for eval/calibrate.

    Dropping it at crawl time means a full recrawl to get it back.
    """
    _site()
    async with httpx.AsyncClient() as client:
        crawler, _, _, sink = await _build(tmp_path, _cfg(), client)
        await crawler.run()
        sink.close()

    by_url = {r["url"]: r for r in _rows(tmp_path)}
    assert by_url["https://conf.test/i/ada.jpg"]["alt"] == "Photo of Ada"
    assert by_url["https://conf.test/i/ada.jpg"]["page_url"] == "https://conf.test/speaker/ada/"
    assert by_url["https://conf.test/i/ada.jpg"]["page_title"] == "Ada"


@pytest.mark.asyncio
@respx.mock
async def test_offsite_link_is_never_fetched(tmp_path):
    """Scope is enforced at enqueue, not at fetch. Nothing may leave the vertical."""
    _site()
    offsite = respx.get("https://offsite.test/elsewhere").mock(
        return_value=httpx.Response(200, html="<html></html>")
    )
    async with httpx.AsyncClient() as client:
        crawler, _, _, sink = await _build(tmp_path, _cfg(), client)
        stats = await crawler.run()
        sink.close()

    assert offsite.call_count == 0
    assert stats.skips["off_vertical"] >= 1


@pytest.mark.asyncio
@respx.mock
async def test_deny_pattern_is_applied_to_discovered_links(tmp_path):
    _site()
    pdf = respx.get("https://conf.test/export.pdf").mock(return_value=httpx.Response(200))
    async with httpx.AsyncClient() as client:
        crawler, _, _, sink = await _build(tmp_path, _cfg(), client)
        stats = await crawler.run()
        sink.close()

    assert pdf.call_count == 0
    assert stats.skips["deny_pattern"] >= 1


@pytest.mark.asyncio
@respx.mock
async def test_identical_bytes_at_two_urls_are_deduped_before_the_record(tmp_path):
    """Non-negotiable #2: the SHA1 gate runs before anything expensive."""
    respx.get("https://conf.test/robots.txt").mock(return_value=httpx.Response(404))
    respx.get("https://conf.test/speakers/").mock(
        return_value=httpx.Response(
            200,
            html='<html><body><img src="/a.jpg" alt="x"><img src="/b.jpg" alt="y"></body></html>',
        )
    )
    for p in ("a", "b"):
        respx.get(f"https://conf.test/{p}.jpg").mock(
            return_value=httpx.Response(200, content=JPEG_A, headers={"content-type": "image/jpeg"})
        )

    async with httpx.AsyncClient() as client:
        crawler, _, _, sink = await _build(tmp_path, _cfg(concurrency=1), client)
        stats = await crawler.run()
        sink.close()

    assert stats.images_fetched == 2
    assert stats.verdicts["new"] == 1
    assert stats.verdicts["exact_dup"] == 1
    assert len(_rows(tmp_path)) == 1, "a duplicate must not produce a second record"


@pytest.mark.asyncio
@respx.mock
async def test_crawl_resumes_after_a_simulated_kill(tmp_path):
    """The week-1 exit criterion: survive a kill -9 and resume.

    Round one is stopped after the seed page. Round two opens the same SQLite
    files in a fresh process-equivalent and must finish the job -- without
    re-fetching what round one completed.
    """
    _site()
    cfg = _cfg(concurrency=1)

    async with httpx.AsyncClient() as client:
        crawler, pages, images, sink = await _build(tmp_path, cfg, client)
        crawler.seed()
        await crawler._do_page("https://conf.test/speakers/", 0)
        # Leave a row in-flight, exactly as a kill mid-request would.
        leased = pages.lease(1)
        assert leased
        sink.close()
        pages.close()
        images.close()

    assert leased[0].url in (
        "https://conf.test/speaker/ada/",
        "https://conf.test/speaker/grace/",
    )

    async with httpx.AsyncClient() as client:
        crawler2, pages2, _, sink2 = await _build(tmp_path, cfg, client)
        stats = await crawler2.run()
        sink2.close()

    # recover_inflight() put the abandoned URL back; both speakers got crawled.
    assert stats.pages_fetched == 2, "the in-flight page must be recovered, not lost"
    assert pages2.stats()["pending"] == 0
    assert stats.images_fetched == 2


@pytest.mark.asyncio
@respx.mock
async def test_a_failing_host_does_not_kill_the_crawl(tmp_path):
    """One dead image must not take the loop down or block the other."""
    _site()
    respx.get("https://conf.test/i/ada.jpg").mock(return_value=httpx.Response(500))

    async with httpx.AsyncClient() as client:
        crawler, _, images, sink = await _build(tmp_path, _cfg(max_retries=2), client)
        stats = await crawler.run()
        sink.close()

    assert stats.images_fetched == 1  # grace survived
    assert images.stats()["pending"] == 0  # the loop still reached quiescence
    # Regression: Frontier.fail() marks the row FAILED once retries run out.
    # An earlier version of _do_image then called complete() on the same row,
    # overwriting FAILED with DONE -- so the failed count read zero forever and
    # a crawl bleeding images to a dead host reported a clean run.
    assert images.stats()["failed"] == 1
    assert images.stats()["done"] == 1


@pytest.mark.asyncio
@respx.mock
async def test_a_404_is_done_not_failed(tmp_path):
    """A missing image is not a crawl failure -- nothing is wrong and there is
    nothing to retry. Only exhausted retries count as FAILED, or the stat stops
    meaning anything."""
    _site()
    respx.get("https://conf.test/i/ada.jpg").mock(return_value=httpx.Response(404))

    async with httpx.AsyncClient() as client:
        crawler, _, images, sink = await _build(tmp_path, _cfg(), client)
        stats = await crawler.run()
        sink.close()

    assert stats.images_fetched == 1
    assert images.stats()["failed"] == 0
    assert images.stats()["done"] == 2


@pytest.mark.asyncio
@respx.mock
async def test_vertical_page_budget_is_enforced(tmp_path):
    _site()
    seeds = _seeds()
    seeds.verticals[0].max_pages = 1

    async with httpx.AsyncClient() as client:
        crawler, _, _, sink = await _build(tmp_path, _cfg(concurrency=1), client, seeds)
        stats = await crawler.run()
        sink.close()

    assert stats.pages_fetched == 1
    assert stats.skips["vertical_budget"] >= 1


@pytest.mark.asyncio
@respx.mock
async def test_images_are_not_dropped_by_the_link_depth_ceiling(tmp_path):
    """An image is a leaf. The ceiling that stops link-following must not also
    discard the photographs on the deepest legitimate page."""
    _site()
    seeds = _seeds()
    seeds.verticals[0].max_depth = 1  # speakers/ is depth 0, speaker pages are 1

    async with httpx.AsyncClient() as client:
        crawler, _, _, sink = await _build(tmp_path, _cfg(), client, seeds)
        stats = await crawler.run()
        sink.close()

    # Speaker pages sit exactly at the ceiling; their images are at depth 2.
    assert stats.pages_fetched == 3
    assert stats.images_fetched == 2


@pytest.mark.asyncio
@respx.mock
async def test_report_renders_without_blowing_up_on_an_empty_crawl(tmp_path):
    """The report is the week-1 deliverable. It must not raise on zero rows."""
    respx.get("https://conf.test/robots.txt").mock(return_value=httpx.Response(404))
    respx.get("https://conf.test/speakers/").mock(return_value=httpx.Response(404))

    async with httpx.AsyncClient() as client:
        crawler, _, _, sink = await _build(tmp_path, _cfg(), client)
        stats = await crawler.run()
        sink.close()

    text = stats.report()
    assert "robots exclusion" in text
    assert "images per host" in text


# --- identity (non-negotiable #6) ------------------------------------------


@pytest.mark.parametrize(
    "ua",
    [
        # The shipped default, and the two half-edited versions of it.
        "arc_search/0.1 (+https://github.com/YOURNAME/arc_search; contact: YOUR_EMAIL_HERE)",
        "arc_search/0.1 (+https://github.com/YOURNAME/arc_search; contact: me@real.org)",
        "arc_search/0.1 (+https://github.com/jb/arc_search; contact: YOUR_EMAIL_HERE)",
        "arc_search/0.1 (+https://example.com/bot)",
        "arc_search/0.1 (+https://github.com/jb/arc_search; contact: changeme)",
    ],
)
def test_placeholder_user_agents_are_refused(ua):
    """A half-edited template is the likely failure, not a fully blank one.

    The original check grepped for YOUR_EMAIL_HERE only, so filling in just the
    email let YOURNAME through onto the open internet.
    """
    problem = check_user_agent(ua)
    assert problem is not None
    assert "placeholder" in problem


def test_a_repo_url_alone_is_sufficient():
    """No email required. Demanding one just teaches people to invent one."""
    assert check_user_agent("arc_search/0.1 (+https://github.com/jb/arc_search)") is None


def test_an_email_alone_is_sufficient():
    assert check_user_agent("arc_search/0.1 (contact: crawler@some-domain.org)") is None


def test_user_agent_must_name_the_project():
    """eye_of_web rotated Opera strings and kept a Googlebot-Image constant."""
    problem = check_user_agent("Mozilla/5.0 (+https://github.com/jb/thing)")
    assert problem is not None
    assert "name the project" in problem


def test_user_agent_must_carry_a_contact_route():
    problem = check_user_agent("arc_search/0.1")
    assert problem is not None
    assert "contact route" in problem


def test_loopback_only_crawl_skips_the_identity_check():
    """Crawling your own fixture server is not an act anyone can complain about.

    Forcing a contact URL here would only teach people to set a fake one, which
    is strictly worse than the check not existing.
    """
    bad = "arc_search/0.1 (+https://github.com/YOURNAME/arc_search)"
    assert check_user_agent(bad, loopback_only=True) is None


def test_all_loopback_detects_a_fixture_config():
    local = SeedConfig(
        verticals=[
            Vertical(
                name="fixture",
                enabled=True,
                seeds=["http://127.0.0.1:8111/speakers/"],
                allow_hosts=["127.0.0.1"],
            )
        ]
    )
    assert all_loopback(local) is True


def test_all_loopback_is_false_when_any_host_is_public():
    """One public allow_host must disarm the bypass for the whole run."""
    mixed = SeedConfig(
        verticals=[
            Vertical(
                name="mixed",
                enabled=True,
                seeds=["http://127.0.0.1:8111/x"],
                allow_hosts=["127.0.0.1", "conf.test"],
            )
        ]
    )
    assert all_loopback(mixed) is False
    assert all_loopback(_seeds()) is False


def test_all_loopback_is_false_when_nothing_is_enabled():
    """An empty config must not read as 'safe to crawl anonymously'."""
    assert all_loopback(SeedConfig()) is False
