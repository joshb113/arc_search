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
from arc_search.crawler.politeness import Politeness, TokenBucket
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
        assert p._hosts["h.test"].limiter._rate == pytest.approx(0.1)


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
        assert p._hosts["h.test"].limiter._rate == pytest.approx(0.1)


@pytest.mark.asyncio
@respx.mock
async def test_no_crawl_delay_leaves_the_override_intact():
    respx.get("https://h.test/robots.txt").mock(return_value=httpx.Response(404))
    async with httpx.AsyncClient() as client:
        p = Politeness(cfg(), client, {"h.test": 0.25})
        await p.allowed("https://h.test/x")
        assert p._hosts["h.test"].limiter._rate == pytest.approx(0.25)


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
