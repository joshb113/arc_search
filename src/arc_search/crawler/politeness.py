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


_DEFAULT_PORTS = {"http": 80, "https": 443}


def authority_of(url: str) -> tuple[str, str, str]:
    """``(authority, scheme, hostname)`` for a URL.

    ``authority`` carries a non-default port; ``hostname`` never does. robots.txt
    is defined per (scheme, host, port) by RFC 9309, so the authority is what
    identifies a robots file -- while the rate budget belongs to the hostname,
    because that is what identifies the machine being asked to do the work.
    """
    parts = urlsplit(url)
    scheme = (parts.scheme or "https").lower()
    hostname = (parts.hostname or "").lower()
    port = parts.port
    authority = f"{hostname}:{port}" if port and port != _DEFAULT_PORTS.get(scheme) else hostname
    return authority, scheme, hostname


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

    def tighten(self, rate: float) -> None:
        """Lower the rate if ``rate`` is slower. Never raises it.

        One hostname can be reached at several authorities (ports, schemes),
        each with its own robots.txt. They share this bucket, and the strictest
        one governs -- a second authority must not be able to relax a limit the
        first one established.
        """
        if rate > 0:
            self._rate = min(self._rate, rate)

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
        # robots state per "scheme://authority" (RFC 9309); rate budget per
        # bare hostname, because one machine answers all of its ports.
        self._hosts: dict[str, _HostState] = {}
        self._buckets: dict[str, TokenBucket] = {}
        # Per-host overrides, from Vertical.per_host_rps. Matched by suffix, so
        # an entry for example.com also governs static.example.com.
        self._host_rps = dict(host_rps or {})
        # Every outbound request passes through wait() exactly once, so this is
        # the only exact count of requests the crawler makes. DB rows are not a
        # proxy for it: skipped and duplicate image fetches spend a token and
        # write nothing, which is what made a throughput shortfall unreadable.
        self.requests_made = 0

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

    def effective_rate(self, url: str) -> float | None:
        """The rate actually in force for this URL, after robots.txt.

        None until the first request to that host has resolved robots. Exists
        so callers and tests can ask what the crawler is really doing without
        reaching into private state -- the answer depends on the config, the
        per-vertical override, and the site's Crawl-delay all at once, which is
        exactly the kind of thing that should be observable.
        """
        authority, scheme, _ = authority_of(url)
        st = self._hosts.get(f"{scheme}://{authority}")
        return st.limiter._rate if st and st.limiter else None

    def _state(self, key: str) -> _HostState:
        st = self._hosts.get(key)
        if st is None:
            st = _HostState()
            self._hosts[key] = st
        return st

    def _bucket(self, hostname: str, rate: float) -> TokenBucket:
        """One rate budget per HOSTNAME, even across ports and schemes.

        robots.txt is per (scheme, host, port) -- but the machine answering
        those requests is one machine. Giving http://h:80 and https://h:8443
        a bucket each would hand a single server double the traffic we promised
        it. When two authorities on one host disagree, the slower rate wins.
        """
        bucket = self._buckets.get(hostname)
        if bucket is None:
            bucket = TokenBucket(rate, self._cfg.per_host_burst)
            self._buckets[hostname] = bucket
        else:
            bucket.tighten(rate)
        return bucket

    async def _ensure_robots(self, authority: str, scheme: str, hostname: str) -> _HostState:
        key = f"{scheme}://{authority}"
        st = self._state(key)
        if st.fetched:
            return st

        async with st.lock:
            if st.fetched:  # another coroutine won the race
                return st

            rate = self.configured_rate(hostname)
            if not self._cfg.respect_robots:
                st.robots = None
            else:
                # The AUTHORITY, not the bare hostname. Dropping the port here
                # meant a host on :8080 had its robots.txt fetched from :80,
                # which fails, which fails closed -- so the entire host was
                # silently skipped and reported only as robots_disallow.
                url = f"{scheme}://{authority}/robots.txt"
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
                        log.info("robots.forbidden", host=authority, status=resp.status_code)
                    else:
                        st.robots = None  # 404 and friends => no restrictions
                except httpx.HTTPError as exc:
                    # Fail closed on the FIRST fetch: we do not know the rules yet.
                    st.blocked = True
                    log.warning("robots.fetch_failed", host=authority, error=str(exc))

            if st.robots is not None:
                delay = st.robots.crawl_delay(self._cfg.user_agent)
                if delay:
                    if delay > self._cfg.max_crawl_delay:
                        st.blocked = True
                        log.info("robots.delay_too_long", host=authority, delay=delay)
                    else:
                        # The SLOWER of the two wins, always. This used to be a
                        # plain assignment, which meant a site publishing
                        # Crawl-delay: 1 would SPEED UP a crawler deliberately
                        # configured to 0.1 rps. Being gentler than robots.txt
                        # requires is never something to override.
                        rate = min(rate, 1.0 / float(delay))

            st.limiter = self._bucket(hostname, rate)
            st.fetched = True
            return st

    async def allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        if not parts.hostname:
            return False
        st = await self._ensure_robots(*authority_of(url))
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
        st = await self._ensure_robots(*authority_of(url))
        if st.limiter is not None:
            await st.limiter.acquire()
        self.requests_made += 1

    def sitemaps(self, url_or_key: str) -> list[str]:
        """Sitemaps declared in robots.txt -- a far better frontier seed than
        blind link-following. eye_of_web never looked at them.

        Takes a URL or a ``scheme://authority`` key, since state is now keyed
        per authority rather than per bare hostname.
        """
        key = url_or_key
        if "//" in url_or_key and urlsplit(url_or_key).hostname:
            authority, scheme, _ = authority_of(url_or_key)
            key = f"{scheme}://{authority}"
        st = self._hosts.get(key)
        if st is None or st.robots is None:
            return []
        return list(st.robots.sitemaps)
