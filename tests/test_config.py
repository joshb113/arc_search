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


# --- the crawl-time size gate ----------------------------------------------


def test_min_image_dim_is_no_longer_derived_from_the_face_gate():
    """⚠️ THE OLD DERIVATION IS VOID, and this test replaced it.

    It used to assert ``min_image_dim <= min_face_px``, justified by: an image
    shorter than the smallest acceptable FACE cannot contain one. ADR-005 made
    image search primary, and that argument says nothing whatever about the
    smallest image worth indexing for *visual* search -- a 60px logo has no
    face in it and is corpus.

    Keeping the old assertion would have been worse than deleting it: a green
    test enforcing reasoning nobody believes any more, which is how a dead
    argument survives a change of goal.

    What replaces it is a measurement (vault/research/image-size-gate.md):
    scene-embedding self-similarity under downscaling, on real corpus images.

        short side   128    96     64     48     32     16
        scene mean   0.988  0.965  0.938  0.906  0.840  0.634
        text  mean   0.988  0.976  0.952  0.936  0.909  0.813

    Two things follow, and neither is "raise the gate":

    1. **Text degrades far more slowly than scene.** Text is the primary mode
       under ADR-005 and is still 0.909 at 32px, where scene has fallen to
       0.840. A gate tuned for scene would throw away images text search can
       still use.
    2. **Exclusion at crawl time is IRREVERSIBLE.** Nothing stores scene pixels,
       so an image the gate rejects needs a whole recrawl to recover, while an
       image admitted and later judged useless costs a filter at query time.
       That is ADR-004's asymmetry exactly, and it points at a LOW gate.

    So the number stays 48 and is now justified on its own terms: a floor
    against tracking pixels and spacers, not a quality judgement. Measured on
    this corpus it excludes **0 of 4,753 images** -- it is not currently binding
    at all, which is the right place for a gate nobody has calibrated.
    """
    cfg = CrawlSettings()
    assert cfg.min_image_dim == 48

    # The point is the absence of coupling. This must NOT be re-derived from the
    # face gate; min_face_px can move for face reasons without dragging the
    # crawl gate with it.
    assert cfg.min_image_dim < 64, (
        "the crawl gate should stay well below where scene similarity degrades "
        "(min 0.86 at 96px), because admitting a marginal image is reversible "
        "and excluding it at crawl time is not"
    )


def test_the_size_gate_does_not_encode_a_quality_judgement():
    """Quality is decided after retrieval, where it is reversible.

    The gate's job is to skip tracking pixels and spacer GIFs. Anything larger
    is a question for ranking, not for admission -- and ranking can be changed
    without a recrawl.
    """
    assert CrawlSettings().min_image_dim <= 64


def test_min_image_bytes_stays_a_bandwidth_filter_not_a_quality_gate():
    """Real portraits are small. The smallest FOSDEM speaker photo in a 9-image
    sample was 5,399 bytes; at the old 8,000 floor, 4 of 9 were discarded.

    Tracking pixels and spacers are well under 1 KB, so anything at or below
    ~2 KB does the intended job. Above that this silently becomes a quality
    filter, which is not what it is for and not where quality is decided.
    """
    assert CrawlSettings().min_image_bytes <= 2_000
