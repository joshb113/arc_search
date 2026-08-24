"""Frontier: URL normalization, persistence, retry, depth recording.

Every test here corresponds to a specific eye_of_web defect. They exist so that
if someone later "simplifies" this module, the regression is loud.
"""

from __future__ import annotations

import pytest

from arc_search.crawler.frontier import Frontier, normalize, url_key


class TestNormalize:
    """eye_of_web deduped on the raw URL string with only the fragment stripped,
    so all four of these were separate crawl targets."""

    def test_trailing_slash_collapses(self):
        assert normalize("https://x.com/a") == normalize("https://x.com/a/")

    def test_query_order_is_canonical(self):
        assert normalize("https://x.com/a?b=1&c=2") == normalize("https://x.com/a?c=2&b=1")

    def test_host_case_and_www_collapse(self):
        assert normalize("https://WWW.X.com/a") == normalize("https://x.com/a")

    def test_fragment_dropped(self):
        assert normalize("https://x.com/a#section") == normalize("https://x.com/a")

    def test_default_port_dropped(self):
        assert normalize("https://x.com:443/a") == normalize("https://x.com/a")
        assert normalize("http://x.com:80/a") == normalize("http://x.com/a")

    def test_nondefault_port_kept(self):
        assert normalize("https://x.com:8443/a") != normalize("https://x.com/a")

    def test_tracking_params_stripped(self):
        assert normalize("https://x.com/a?utm_source=t&id=5") == normalize("https://x.com/a?id=5")
        assert normalize("https://x.com/a?fbclid=xyz") == normalize("https://x.com/a")

    def test_meaningful_params_kept(self):
        assert normalize("https://x.com/a?id=5") != normalize("https://x.com/a?id=6")

    def test_duplicate_slashes_collapse(self):
        assert normalize("https://x.com//a//b") == normalize("https://x.com/a/b")

    def test_scheme_distinguishes(self):
        assert normalize("http://x.com/a") != normalize("https://x.com/a")

    def test_key_is_stable(self):
        assert url_key("https://x.com/a/") == url_key("https://X.com/a#frag")


@pytest.fixture
def frontier(tmp_path):
    f = Frontier(tmp_path / "f.sqlite")
    yield f
    f.close()


class TestFrontier:
    def test_add_and_lease(self, frontier):
        assert frontier.add("https://x.com/a", depth=0, max_depth=3)
        tasks = frontier.lease(10)
        assert [t.url for t in tasks] == ["https://x.com/a"]

    def test_leased_urls_are_not_released_twice(self, frontier):
        frontier.add("https://x.com/a", 0, 3)
        assert len(frontier.lease(10)) == 1
        assert frontier.lease(10) == []

    def test_duplicate_add_is_rejected(self, frontier):
        assert frontier.add("https://x.com/a", 0, 3)
        assert not frontier.add("https://x.com/a/", 0, 3)  # normalizes identically

    def test_over_depth_is_recorded_not_dropped(self, frontier):
        """eye_of_web returned before adding to visited_urls, so every
        rediscovery at the depth boundary redid the work and re-logged it."""
        assert not frontier.add("https://x.com/deep", depth=9, max_depth=3)
        assert frontier.stats()["done"] == 1  # recorded, not forgotten
        assert not frontier.add("https://x.com/deep", depth=9, max_depth=3)
        assert frontier.stats()["done"] == 1  # still one row, no churn

    def test_failure_requeues_until_max_retries(self, frontier):
        """A transient 503 must not permanently drop a page."""
        frontier.add("https://x.com/a", 0, 3)
        frontier.lease(1)
        for _ in range(3):
            assert frontier.fail("https://x.com/a", max_retries=3) is True
            assert len(frontier.lease(1)) == 1  # came back
        assert frontier.fail("https://x.com/a", max_retries=3) is False
        assert frontier.lease(1) == []
        assert frontier.stats()["failed"] == 1

    def test_state_survives_reopen(self, tmp_path):
        """A crash must not lose the crawl. eye_of_web kept everything in
        memory and re-crawled from zero on restart."""
        path = tmp_path / "f.sqlite"
        f1 = Frontier(path)
        f1.add("https://x.com/a", 0, 3)
        f1.add("https://x.com/b", 0, 3)
        f1.lease(1)
        f1.complete("https://x.com/a")
        f1.close()

        f2 = Frontier(path)
        assert f2.stats()["done"] == 1
        assert f2.stats()["pending"] == 1
        f2.close()

    def test_recover_inflight_after_crash(self, tmp_path):
        path = tmp_path / "f.sqlite"
        f1 = Frontier(path)
        f1.add("https://x.com/a", 0, 3)
        f1.lease(1)  # leased, then "crash"
        f1.close()

        f2 = Frontier(path)
        assert f2.stats()["inflight"] == 1
        assert f2.recover_inflight() == 1
        assert f2.stats()["pending"] == 1
        assert len(f2.lease(1)) == 1
        f2.close()

    def test_lease_prefers_shallower_depth(self, frontier):
        frontier.add("https://x.com/deep", depth=2, max_depth=5)
        frontier.add("https://x.com/shallow", depth=0, max_depth=5)
        tasks = frontier.lease(2)
        assert tasks[0].url.endswith("/shallow")
