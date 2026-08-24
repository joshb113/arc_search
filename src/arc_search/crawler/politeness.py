"""Per-host politeness: robots.txt and rate limiting.

eye_of_web had neither. Zero repo-wide matches for ``robots``/``Crawl-delay``,
and four worker threads times two image sub-threads hit a single host as fast as
it would answer. The only pause in the system was five seconds *between whole
domains*.

Two components here:

``RobotsCache``  fetches and caches robots.txt per host, honours Crawl-delay.
``HostLimiter``  an async token bucket, one per host, sized from robots.txt.

Both are keyed on host, never global. A global rate limit either starves a large
crawl or hammers a small host; neither is what you want.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import httpx
import structlog
from protego import Protego

from arc_search.config import CrawlSettings
from arc_search.crawler.seeds import host_matches

log = structlog.get_logger(__name__)


class TokenBucket:
    """Async token bucket. ``await acquire()`` blocks until a token is free."""

    def __init__(self, rate: float, burst: int) -> None:
        if rate <= 0:
            raise ValueError("rate must be positive")
        self._rate = rate
        self._capacity = max(1, burst)
        self._tokens = float(self._capacity)
        self._updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            while True:
                now = time.monotonic()
                self._tokens = min(
                    self._capacity, self._tokens + (now - self._updated) * self._rate
                )
                self._updated = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                deficit = (1.0 - self._tokens) / self._rate
                await asyncio.sleep(deficit)


@dataclass
class _HostState:
    robots: Protego | None = None
    limiter: TokenBucket | None = None
    fetched: bool = False
    blocked: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class Politeness:
    """Gatekeeper for every outbound request.

    Usage::

        pol = Politeness(cfg, client)
        if await pol.allowed(url):
            await pol.wait(url)
            resp = await client.get(url)
    """

    def __init__(
        self,
        cfg: CrawlSettings,
        client: httpx.AsyncClient,
        host_rps: Mapping[str, float] | None = None,
    ) -> None:
        self._cfg = cfg
        self._client = client
        self._hosts: dict[str, _HostState] = {}
        # Per-host overrides, from Vertical.per_host_rps. Matched by suffix, so
        # an entry for example.com also governs static.example.com.
        self._host_rps = dict(host_rps or {})

    def configured_rate(self, host: str) -> float:
        """Requests per second for this host, before robots.txt is consulted.

        seeds.yaml documents a per-vertical ``per_host_rps`` and says "only
        ever lower this". It was parsed, validated, and then read by nothing --
        every host ran at the global default. Harmless when the override was
        faster; a broken promise when someone set 0.1 for a fragile host and
        was quietly given 0.5.

        When several entries match, the slowest wins.
        """
        rates = [
            rps for entry, rps in self._host_rps.items() if host_matches(host, entry) and rps > 0
        ]
        return min([self._cfg.per_host_rps, *rates])

    def _state(self, host: str) -> _HostState:
        st = self._hosts.get(host)
        if st is None:
            st = _HostState()
            self._hosts[host] = st
        return st

    async def _ensure_robots(self, host: str, scheme: str) -> _HostState:
        st = self._state(host)
        if st.fetched:
            return st

        async with st.lock:
            if st.fetched:  # another coroutine won the race
                return st

            rate = self.configured_rate(host)
            if not self._cfg.respect_robots:
                st.robots = None
            else:
                url = f"{scheme}://{host}/robots.txt"
                try:
                    resp = await self._client.get(
                        url,
                        timeout=self._cfg.timeout_s,
                        headers={"User-Agent": self._cfg.user_agent},
                    )
                    if resp.status_code == 200:
                        st.robots = Protego.parse(resp.text)
                    elif resp.status_code in (401, 403):
                        # Explicitly restricted. Treat as full disallow.
                        st.blocked = True
                        log.info("robots.forbidden", host=host, status=resp.status_code)
                    else:
                        st.robots = None  # 404 and friends => no restrictions
                except httpx.HTTPError as exc:
                    # Fail closed on the FIRST fetch: we do not know the rules yet.
                    st.blocked = True
                    log.warning("robots.fetch_failed", host=host, error=str(exc))

            if st.robots is not None:
                delay = st.robots.crawl_delay(self._cfg.user_agent)
                if delay:
                    if delay > self._cfg.max_crawl_delay:
                        st.blocked = True
                        log.info("robots.delay_too_long", host=host, delay=delay)
                    else:
                        # The SLOWER of the two wins, always. This used to be a
                        # plain assignment, which meant a site publishing
                        # Crawl-delay: 1 would SPEED UP a crawler deliberately
                        # configured to 0.1 rps. Being gentler than robots.txt
                        # requires is never something to override.
                        rate = min(rate, 1.0 / float(delay))

            st.limiter = TokenBucket(rate, self._cfg.per_host_burst)
            st.fetched = True
            return st

    async def allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        if not parts.hostname:
            return False
        st = await self._ensure_robots(parts.hostname, parts.scheme or "https")
        if st.blocked:
            return False
        if st.robots is None:
            return True
        return bool(st.robots.can_fetch(url, self._cfg.user_agent))

    async def wait(self, url: str) -> None:
        """Block until this host's bucket allows another request."""
        parts = urlsplit(url)
        if not parts.hostname:
            return
        st = await self._ensure_robots(parts.hostname, parts.scheme or "https")
        if st.limiter is not None:
            await st.limiter.acquire()

    def sitemaps(self, host: str) -> list[str]:
        """Sitemaps declared in robots.txt -- a far better frontier seed than
        blind link-following. eye_of_web never looked at them."""
        st = self._hosts.get(host)
        if st is None or st.robots is None:
            return []
        return list(st.robots.sitemaps)
