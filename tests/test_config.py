"""Config-loading tests.

The bug these exist for: all four settings groups read the SAME .env, and
pydantic-settings defaults a dotenv source to extra="forbid". So a .env holding
keys for more than one group raised ValidationError on import -- meaning the
first real run would have died before fetching anything, on a file the test
suite never touched because tests construct settings with explicit kwargs.
"""

from __future__ import annotations

import pytest

from arc_search.config import (
    CrawlSettings,
    FaceSettings,
    IndexSettings,
    SearchSettings,
    unknown_settings,
)

ALL_GROUPS = [CrawlSettings, FaceSettings, IndexSettings, SearchSettings]


@pytest.mark.parametrize("cls", ALL_GROUPS)
def test_every_group_tolerates_another_groups_env_vars(cls, monkeypatch):
    """One shared .env must not blow up three of the four groups."""
    monkeypatch.setenv("ARC_CRAWL_USER_AGENT", "arc_search/0.1 (+https://example.org/x)")
    monkeypatch.setenv("ARC_FACE_MIN_FACE_PX", "80")
    monkeypatch.setenv("ARC_INDEX_COLLECTION", "faces_v2")
    monkeypatch.setenv("ARC_SEARCH_RETRIEVE_K", "250")
    cls()  # must not raise


def test_each_group_still_reads_its_own_prefix(monkeypatch):
    monkeypatch.setenv("ARC_CRAWL_PER_HOST_RPS", "0.25")
    monkeypatch.setenv("ARC_FACE_MIN_FACE_PX", "80")
    assert CrawlSettings().per_host_rps == 0.25
    assert FaceSettings().min_face_px == 80


# --- the compensating check ------------------------------------------------


def test_typo_in_a_known_prefix_is_reported():
    """extra='ignore' means ARC_CRAWL_PER_HOST_RPZ silently does nothing.

    Without this check you get the default 0.5 rps and no complaint, then spend
    an afternoon wondering why the crawl is slow.
    """
    stray = unknown_settings({"ARC_CRAWL_PER_HOST_RPZ": "5"})
    assert stray == ["ARC_CRAWL_PER_HOST_RPZ"]


def test_valid_vars_across_all_groups_are_not_reported():
    assert (
        unknown_settings(
            {
                "ARC_CRAWL_USER_AGENT": "x",
                "ARC_FACE_MIN_FACE_PX": "80",
                "ARC_INDEX_QDRANT_URL": "http://x",
                "ARC_SEARCH_RETRIEVE_K": "10",
            }
        )
        == []
    )


def test_unrelated_env_vars_are_ignored():
    assert unknown_settings({"PATH": "/usr/bin", "PGPASSWORD": "hunter2"}) == []


def test_an_unknown_prefix_is_reported():
    """ARC_CRAWLER_* is a plausible and completely inert misspelling."""
    assert unknown_settings({"ARC_CRAWLER_CONCURRENCY": "8"}) == ["ARC_CRAWLER_CONCURRENCY"]


def test_the_shipped_env_example_has_no_typos():
    """.env.example is what people copy. Every ARC_ key in it must be real."""
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / ".env.example").read_text(encoding="utf-8")
    keys = {}
    for line in text.splitlines():
        line = line.strip().lstrip("#").strip()
        if line.startswith("ARC_") and "=" in line:
            k, v = line.split("=", 1)
            keys[k.strip()] = v.strip()
    assert keys, "no ARC_ keys found -- did the file move?"
    assert unknown_settings(keys) == []
