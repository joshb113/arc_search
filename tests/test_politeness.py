"""Politeness tests.

Both bugs covered here were live in a real crawl: a per-vertical rate limit
that nothing read, and robots.txt handling that could make the crawler go
FASTER than it was configured to.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from arc_search.config import CrawlSettings
from arc_search.crawler.politeness import Politeness, TokenBucket, authority_of
from arc_search.crawler.seeds import SeedConfig, Vertical


def cfg(**kw) -> CrawlSettings:
    return CrawlSettings(**{"per_host_rps": 0.5, **kw})


# --- the configured rate ---------------------------------------------------


def test_no_override_uses_the_global_default():
    p = Politeness(cfg(), client=None)  # type: ignore[arg-type]
    assert p.configured_rate("h.test") == 0.5


def test_a_slower_override_is_honoured():
    """seeds.yaml says 'only ever lower this'. It has to mean something.

    This is the case that matters: someone sets 0.1 for a fragile host. The
    override was parsed and then read by nothing, so they silently got 0.5.
    """
    p = Politeness(cfg(), None, {"h.test": 0.1})  # type: ignore[arg-type]
    assert p.configured_rate("h.test") == 0.1


def test_an_override_matches_subdomains():
    """Image CDNs are almost always subdomains of the host you configured."""
    p = Politeness(cfg(), None, {"example.com": 0.2})  # type: ignore[arg-type]
    assert p.configured_rate("static.example.com") == 0.2
    assert p.configured_rate("notexample.com") == 0.5  # label-wise, not endswith


def test_the_slowest_matching_override_wins():
    p = Politeness(cfg(), None, {"example.com": 0.4, "cdn.example.com": 0.05})  # type: ignore[arg-type]
    assert p.configured_rate("cdn.example.com") == 0.05


def test_an_unrelated_host_is_unaffected():
    p = Politeness(cfg(), None, {"other.test": 0.01})  # type: ignore[arg-type]
    assert p.configured_rate("h.test") == 0.5


# --- robots.txt interaction ------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_crawl_delay_slower_than_config_wins():
    respx.get("https://h.test/robots.txt").mock(
        return_value=httpx.Response(200, text="User-agent: *\nCrawl-delay: 10\n")
    )
    async with httpx.AsyncClient() as client:
        p = Politeness(cfg(per_host_rps=2.0), client)
        await p.allowed("https://h.test/x")
        assert p.effective_rate("https://h.test/x") == pytest.approx(0.1)


@pytest.mark.asyncio
@respx.mock
async def test_crawl_delay_never_speeds_us_up():
    """The bug: `rate = 1/delay` was an assignment, not a minimum.

    A host publishing Crawl-delay: 1 would override a crawler deliberately
    configured to 0.1 rps and pull it up to 1 rps -- ten times faster than the
    operator asked for, in the name of respecting robots.txt.
    """
    respx.get("https://h.test/robots.txt").mock(
        return_value=httpx.Response(200, text="User-agent: *\nCrawl-delay: 1\n")
    )
    async with httpx.AsyncClient() as client:
        p = Politeness(cfg(), client, {"h.test": 0.1})
        await p.allowed("https://h.test/x")
        assert p.effective_rate("https://h.test/x") == pytest.approx(0.1)


@pytest.mark.asyncio
@respx.mock
async def test_no_crawl_delay_leaves_the_override_intact():
    respx.get("https://h.test/robots.txt").mock(return_value=httpx.Response(404))
    async with httpx.AsyncClient() as client:
        p = Politeness(cfg(), client, {"h.test": 0.25})
        await p.allowed("https://h.test/x")
        assert p.effective_rate("https://h.test/x") == pytest.approx(0.25)


# --- the wiring ------------------------------------------------------------


def test_seed_config_exports_its_rate_limits():
    s = SeedConfig(
        verticals=[
            Vertical(name="a", enabled=True, seeds=["https://a.test/"], per_host_rps=1.0),
            Vertical(name="off", enabled=False, seeds=["https://b.test/"], per_host_rps=9.0),
            Vertical(name="c", enabled=True, seeds=["https://c.test/"]),  # unset
        ]
    )
    assert s.host_rate_limits() == {"a.test": 1.0}


def test_two_verticals_on_one_host_take_the_slower_rate():
    """A promise to be gentle is not cancelled by a second vertical in a hurry."""
    s = SeedConfig(
        verticals=[
            Vertical(
                name="fast",
                enabled=True,
                seeds=["https://x.test/a"],
                allow_hosts=["x.test"],
                per_host_rps=2.0,
            ),
            Vertical(
                name="slow",
                enabled=True,
                seeds=["https://x.test/b"],
                allow_hosts=["x.test"],
                per_host_rps=0.2,
            ),
        ]
    )
    assert s.host_rate_limits() == {"x.test": 0.2}


def test_token_bucket_rejects_a_nonsense_rate():
    with pytest.raises(ValueError):
        TokenBucket(0, 1)


# --- authority vs hostname -------------------------------------------------


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://h.test/x", ("h.test", "https", "h.test")),
        ("https://h.test:443/x", ("h.test", "https", "h.test")),  # default port elided
        ("http://h.test:80/x", ("h.test", "http", "h.test")),
        ("http://h.test:8080/x", ("h.test:8080", "http", "h.test")),
        ("https://H.TEST:8443/x", ("h.test:8443", "https", "h.test")),
    ],
)
def test_authority_of(url, expected):
    assert authority_of(url) == expected


@pytest.mark.asyncio
@respx.mock
async def test_robots_is_fetched_from_the_right_port():
    """The bug a loopback profiler found in about four seconds.

    The robots URL was built from the bare hostname, so a host on :8080 had its
    robots.txt requested from :80. That connection fails, Politeness fails
    closed, and every URL on the host is silently skipped -- reported only as
    robots_disallow, which reads like the site said no.
    """
    # ORDER MATTERS, and not for an obvious reason: a respx pattern written
    # without a port matches requests on ANY port, and the first matching route
    # wins. Register the portless one first and it shadows the ported one, so
    # this test passes or fails for reasons unrelated to the code under test.
    # The ported route has to come first for the two to be distinguishable.
    right = respx.get("http://h.test:8080/robots.txt").mock(
        return_value=httpx.Response(200, text="User-agent: *\nDisallow: /private/\n")
    )
    wrong = respx.get("http://h.test/robots.txt").mock(return_value=httpx.Response(404))
    async with httpx.AsyncClient() as client:
        p = Politeness(cfg(), client)
        assert await p.allowed("http://h.test:8080/ok") is True
        assert await p.allowed("http://h.test:8080/private/x") is False

    assert right.call_count == 1
    assert wrong.call_count == 0, "robots.txt must be fetched from the URL's own port"


@pytest.mark.asyncio
@respx.mock
async def test_two_ports_on_one_host_share_one_rate_budget():
    """robots.txt is per authority, but the machine is per hostname.

    A bucket each would hand one server double the traffic we promised it.
    """
    respx.get("http://h.test:8080/robots.txt").mock(return_value=httpx.Response(404))
    respx.get("http://h.test:9090/robots.txt").mock(return_value=httpx.Response(404))
    async with httpx.AsyncClient() as client:
        p = Politeness(cfg(), client, {"h.test": 0.5})
        await p.allowed("http://h.test:8080/a")
        await p.allowed("http://h.test:9090/b")
        assert len(p._buckets) == 1
        assert p._buckets["h.test"]._rate == pytest.approx(0.5)


@pytest.mark.asyncio
@respx.mock
async def test_the_strictest_authority_governs_the_shared_budget():
    """A second port must not be able to relax a limit the first established."""
    respx.get("http://h.test:8080/robots.txt").mock(
        return_value=httpx.Response(200, text="User-agent: *\nCrawl-delay: 20\n")
    )
    respx.get("http://h.test:9090/robots.txt").mock(return_value=httpx.Response(404))
    async with httpx.AsyncClient() as client:
        p = Politeness(cfg(per_host_rps=1.0, max_crawl_delay=60), client)
        await p.allowed("http://h.test:8080/a")  # 0.05/s from Crawl-delay: 20
        await p.allowed("http://h.test:9090/b")  # would be 1.0/s alone
        assert p._buckets["h.test"]._rate == pytest.approx(0.05)


def test_tighten_never_raises_the_rate():
    b = TokenBucket(0.1, 2)
    b.tighten(5.0)
    assert b._rate == pytest.approx(0.1)
    b.tighten(0.01)
    assert b._rate == pytest.approx(0.01)


def test_an_override_faster_than_the_global_is_clamped():
    """per_host_rps may only lower the rate. This is the documented contract.

    Worth pinning because the failure is invisible: seeds.yaml asking for 1.0
    against a 0.5 global runs at 0.5, which looked for a while like a two-fold
    throughput bug in the crawl loop. run.py now logs the effective rate and
    warns that the override was ignored.
    """
    p = Politeness(cfg(per_host_rps=0.5), None, {"h.test": 5.0})  # type: ignore[arg-type]
    assert p.configured_rate("h.test") == 0.5


def test_raising_the_global_is_how_you_actually_go_faster():
    p = Politeness(cfg(per_host_rps=2.0), None, {"h.test": 5.0})  # type: ignore[arg-type]
    assert p.configured_rate("h.test") == 2.0


def test_requests_are_counted_at_the_rate_limiter():
    """Row counts are not a proxy for requests -- skipped and duplicate image
    fetches spend a token and write nothing, which is what made the throughput
    question unreadable in the first place."""
    p = Politeness(cfg(), None)  # type: ignore[arg-type]
    assert p.requests_made == 0
