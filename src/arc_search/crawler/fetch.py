"""HTTP fetching: the one place bytes enter the system.

Every outbound request in arc_search goes through ``Fetcher``. That is a
deliberate choke point. ``Politeness`` is consulted *inside* it, so there is no
code path that can fetch a URL without first checking robots.txt and taking a
token from that host's bucket. eye_of_web's equivalent logic was inlined at each
of its call sites, which is how it ended up with some paths that rate-limited
and some that did not.

Three things this module gets right that a naive version does not:

**Screen before you download.** Response headers are inspected and the body is
abandoned if the content-type or Content-Length disqualifies it. A 40 MB TIFF
and a 60-byte tracking pixel both cost us headers only.

**Enforce the size cap while streaming, not after.** A server can advertise
``Content-Length: 5000`` and then send you 2 GB. Checking the header is not a
limit, it is a suggestion. We count bytes as they arrive and abort.

**Sniff the magic bytes.** ``Content-Type: application/octet-stream`` on a
perfectly good JPEG is common enough that trusting the header alone silently
loses real images.

Retry policy is split on purpose:

  * *In-request* transients (connect resets, 429, 5xx) retry here, a few times,
    with jittered backoff. The connection is already warm; giving up would be
    wasteful.
  * *Durable* failures go back to ``Frontier.fail``, which requeues across
    process restarts. A host that is down for an hour is not a retry loop
    problem, it is a scheduling problem.

Mixing those two -- as eye_of_web did, by having neither -- gives you either a
crawl that hammers a struggling host or one that permanently drops pages on a
single blip.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

import httpx
import structlog

from arc_search.config import CrawlSettings
from arc_search.crawler.politeness import Politeness

log = structlog.get_logger(__name__)

# Worth trying again on the same connection. Note 429 is here: it means "slow
# down", not "go away", and our backoff is the correct response.
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

# Raster formats a face detector can actually use.
#   - svg+xml is vector; there is no face in it.
#   - gif is almost always UI chrome, and animated gifs decode to a surprise.
#   - x-icon is a favicon.
IMAGE_TYPES = frozenset(
    {
        "image/jpeg",
        "image/jpg",  # nonstandard but widely emitted
        "image/png",
        "image/webp",
        "image/avif",
        "image/heic",
        "image/heif",
        "image/tiff",
        "image/bmp",
    }
)

# Magic-number prefixes, for when the server's content-type is useless.
_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
    (b"II*\x00", "image/tiff"),
    (b"MM\x00*", "image/tiff"),
)


def sniff_image_type(head: bytes) -> str | None:
    """Identify an image from its leading bytes. Returns a MIME type or None."""
    for magic, mime in _MAGIC:
        if head.startswith(magic):
            return mime
    # RIFF container: bytes 0-3 "RIFF", 8-11 the form type.
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    # ISO-BMFF: 4-7 "ftyp", then a brand.
    if head[4:8] == b"ftyp":
        brand = head[8:12]
        if brand in (b"avif", b"avis"):
            return "image/avif"
        if brand in (b"heic", b"heix", b"hevc", b"mif1", b"msf1"):
            return "image/heic"
    return None


def normalize_content_type(raw: str | None) -> str:
    """``"image/JPEG; charset=binary"`` -> ``"image/jpeg"``."""
    if not raw:
        return ""
    return raw.split(";", 1)[0].strip().lower()


def is_html(content_type: str | None) -> bool:
    return normalize_content_type(content_type) in ("text/html", "application/xhtml+xml")


@dataclass(frozen=True)
class Screen:
    """Verdict on a response we have headers for but have not downloaded."""

    ok: bool
    reason: str


def screen_image_headers(
    content_type: str | None,
    content_length: int | None,
    cfg: CrawlSettings,
) -> Screen:
    """Decide from headers alone whether an image body is worth downloading.

    Pure. No I/O. This is the function that has to be right, so it is the one
    that is unit-tested exhaustively.

    An unknown or absent Content-Length is *not* a rejection -- chunked
    responses are normal and the streaming cap catches oversize bodies anyway.
    An unknown content-type is also not a rejection, because magic-byte sniffing
    gets a second opinion once bytes arrive. We only reject on positive evidence.
    """
    ctype = normalize_content_type(content_type)

    if ctype and ctype not in IMAGE_TYPES:
        if ctype.startswith("image/"):
            return Screen(False, f"image_subtype_unusable:{ctype}")
        if ctype in ("application/octet-stream", "binary/octet-stream"):
            pass  # useless but not disqualifying; sniff will decide
        else:
            return Screen(False, f"not_an_image:{ctype}")

    if content_length is not None:
        if content_length < cfg.min_image_bytes:
            return Screen(False, f"too_small:{content_length}B")
        if content_length > cfg.max_image_bytes:
            return Screen(False, f"too_large:{content_length}B")

    return Screen(True, "ok")


def backoff_delay(attempt: int, cfg: CrawlSettings, *, rand: random.Random | None = None) -> float:
    """Exponential backoff with full jitter, capped.

    Full jitter (uniform over ``[0, computed]``) rather than the more obvious
    ``computed +/- noise``: when a host 503s, every in-flight request for it
    fails at once, and unjittered backoff marches them all back in lockstep.
    """
    rnd = rand or random
    ceiling = min(cfg.backoff_max_s, cfg.backoff_base_s * (2**attempt))
    return rnd.uniform(0.0, ceiling)


@dataclass(frozen=True)
class Fetched:
    url: str
    final_url: str
    status: int
    content_type: str
    kind: Literal["html", "image", "other"]
    body: bytes
    from_cache: bool = False

    @property
    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")


class FetchError(Exception):
    """Raised for a failure the caller should hand to ``Frontier.fail``."""

    def __init__(self, url: str, reason: str, *, retryable: bool = True) -> None:
        super().__init__(f"{url}: {reason}")
        self.url = url
        self.reason = reason
        self.retryable = retryable


class Skipped(Exception):
    """Raised when a URL was deliberately not downloaded.

    Distinct from ``FetchError`` because a skip is a *success* -- the URL was
    evaluated and correctly rejected. Conflating the two is how a crawler ends
    up retrying a 40 MB TIFF four times.
    """

    def __init__(self, url: str, reason: str) -> None:
        super().__init__(f"{url}: {reason}")
        self.url = url
        self.reason = reason


class Fetcher:
    """Politeness-enforcing HTTP client.

    ::

        async with httpx.AsyncClient(http2=True, follow_redirects=True) as c:
            f = Fetcher(cfg, c, Politeness(cfg, c))
            page = await f.get_page("https://archive.fosdem.org/2025/schedule/speakers/")
    """

    def __init__(
        self,
        cfg: CrawlSettings,
        client: httpx.AsyncClient,
        politeness: Politeness,
        *,
        sleep: object | None = None,
    ) -> None:
        self._cfg = cfg
        self._client = client
        self._pol = politeness
        # Injectable so tests do not actually wait out a backoff.
        import asyncio

        self._sleep = sleep or asyncio.sleep

    @property
    def headers(self) -> dict[str, str]:
        return {
            "User-Agent": self._cfg.user_agent,
            "Accept-Encoding": "gzip, deflate, br",
        }

    async def _gate(self, url: str) -> None:
        """robots.txt then rate limit. Both, always, before any request."""
        if not await self._pol.allowed(url):
            raise Skipped(url, "robots_disallow")
        await self._pol.wait(url)

    async def get_page(self, url: str) -> Fetched:
        """Fetch an HTML page. Non-HTML responses are a ``Skipped``, not an error."""
        await self._gate(url)
        resp = await self._request("GET", url, headers={**self.headers, "Accept": "text/html"})
        ctype = normalize_content_type(resp.headers.get("content-type"))
        if not is_html(ctype):
            await resp.aclose()
            raise Skipped(url, f"not_html:{ctype or 'unknown'}")
        body = await resp.aread()
        await resp.aclose()
        return Fetched(
            url=url,
            final_url=str(resp.url),
            status=resp.status_code,
            content_type=ctype,
            kind="html",
            body=body,
        )

    async def get_image(self, url: str) -> Fetched:
        """Fetch an image, screening on headers and capping the body mid-stream.

        Deliberately NOT a HEAD pre-flight. The plan called for one, but a HEAD
        costs a full extra round trip and a meaningful minority of servers
        either 405 it or answer it with headers that differ from the GET. A
        streaming GET that abandons the body after reading headers achieves the
        same saving -- we never pull the payload -- in one round trip, and it
        works everywhere. The bandwidth we avoid is identical; the latency we
        avoid is real.
        """
        await self._gate(url)
        resp = await self._request("GET", url, headers={**self.headers, "Accept": "image/*"})

        declared = resp.headers.get("content-type")
        try:
            length = int(resp.headers.get("content-length", ""))
        except ValueError:
            length = None  # absent or malformed; the stream cap covers us

        verdict = screen_image_headers(declared, length, self._cfg)
        if not verdict.ok:
            await resp.aclose()
            raise Skipped(url, verdict.reason)

        cap = self._cfg.max_image_bytes
        chunks: list[bytes] = []
        total = 0
        try:
            async for chunk in resp.aiter_bytes():
                total += len(chunk)
                if total > cap:
                    # The server lied about Content-Length, or never declared
                    # one. Either way we stop paying for it now.
                    raise Skipped(url, f"stream_exceeded_cap:{cap}B")
                chunks.append(chunk)
        finally:
            await resp.aclose()

        body = b"".join(chunks)

        if total < self._cfg.min_image_bytes:
            raise Skipped(url, f"too_small:{total}B")

        sniffed = sniff_image_type(body[:16])
        if sniffed is None:
            raise Skipped(url, f"not_image_bytes:{normalize_content_type(declared) or 'unknown'}")
        if sniffed not in IMAGE_TYPES:
            raise Skipped(url, f"image_subtype_unusable:{sniffed}")

        return Fetched(
            url=url,
            final_url=str(resp.url),
            status=resp.status_code,
            content_type=sniffed,
            kind="image",
            body=body,
        )

    async def _request(self, method: str, url: str, **kw: object) -> httpx.Response:
        """Send with in-request retry. Returns an UNREAD streaming response."""
        last = "unknown"
        for attempt in range(self._cfg.max_retries):
            try:
                req = self._client.build_request(method, url, timeout=self._cfg.timeout_s, **kw)
                resp = await self._client.send(req, stream=True)
            except httpx.HTTPError as exc:
                last = f"{type(exc).__name__}: {exc}"
                log.debug("fetch.transport_error", url=url, attempt=attempt, error=last)
            else:
                if resp.status_code < 400:
                    return resp
                await resp.aclose()
                last = f"http_{resp.status_code}"
                if resp.status_code not in RETRYABLE_STATUS:
                    # 404, 403, 410: retrying changes nothing. Fail durably so
                    # the frontier marks it done rather than requeuing it.
                    raise FetchError(url, last, retryable=False)

            if attempt + 1 < self._cfg.max_retries:
                await self._sleep(backoff_delay(attempt, self._cfg))  # type: ignore[operator]

        raise FetchError(url, f"exhausted_retries:{last}", retryable=True)
