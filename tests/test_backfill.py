"""Backfill runner tests.

The orchestration is what matters here, not the pieces -- the store, the vector
store and the extractor all have their own suites. What this file pins is the
sequencing and the failure handling, because those are where a long unattended
drain goes quietly wrong:

  * an empty result must not tombstone while thresholds are uncalibrated
  * a failed image must not park the cursor and stall the whole queue
  * the two stores must be written in an order that survives a crash
  * one bad URL must not end the run

The fetcher and extractor are faked. A test that needs the network or a GPU to
tell you the cursor advanced is a test nobody runs.
"""

from __future__ import annotations

import uuid

import numpy as np
import pytest

from arc_search.crawler.fetch import Fetched, FetchError, Skipped
from arc_search.index.backfill import Backfill, BackfillStats
from pgfixtures import ctx as _ctx
from pgfixtures import img as _img
from pgfixtures import needs_pg

# --- fakes -----------------------------------------------------------------


class FakeFace:
    """Enough of index.faces.Face for the writer and the vector store."""

    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(512).astype(np.float32)
        self.qdrant_id = uuid.uuid4()
        self.embedding = v / np.linalg.norm(v)
        self.bbox = (1.0, 2.0, 60.0, 70.0)
        self.landmarks = np.zeros((5, 2), dtype=np.float32)
        self.src_width, self.src_height = 220, 180
        self.det_score, self.blur_var, self.yaw = 0.9, 120.0, 0.0
        self.quality, self.age_est = 0.7, 30


class FakeRejects:
    def __init__(self, **kw):
        self.too_small = kw.get("too_small", 0)
        self.low_score = kw.get("low_score", 0)
        self.too_blurry = kw.get("too_blurry", 0)
        self.bad_pose = kw.get("bad_pose", 0)
        self.underage = kw.get("underage", 0)


class FakeExtractor:
    """Returns a scripted number of faces per call. Never loads a model."""

    def __init__(self, per_image, rejects=None):
        self._per = list(per_image)
        self._rejects = rejects or FakeRejects()
        self.crops_written = []

    def extract(self, img):
        n = self._per.pop(0) if self._per else 0
        return [FakeFace(i + 1) for i in range(n)], self._rejects

    def write_crop(self, face, root):
        rel = f"{face.qdrant_id.hex[:2]}/{face.qdrant_id.hex[2:4]}/{face.qdrant_id.hex}.webp"
        self.crops_written.append(rel)
        return rel


class FakeFetcher:
    """Serves bytes, or raises per a scripted map of url -> exception."""

    def __init__(self, body: bytes, failures=None):
        self._body = body
        self._failures = failures or {}
        self.requested = []
        self.requests_made = 0

    async def get_image(self, url: str) -> Fetched:
        self.requested.append(url)
        self.requests_made += 1
        if url in self._failures:
            raise self._failures[url]
        return Fetched(
            url=url,
            final_url=url,
            status=200,
            content_type="image/png",
            kind="image",
            body=self._body,
            width=220,
            height=180,
        )


def _seed(writer, n: int, png: bytes) -> list[int]:
    """Put n distinct images in the corpus at face_count = -1."""
    from imagefixtures import make_image

    ids = []
    for i in range(n):
        body = make_image(100 + i, "PNG")
        writer.handle(_img(body, url=f"https://conf.test/i/{i}.png"), _ctx())
        ids.append(writer._exec("SELECT id FROM image ORDER BY id DESC LIMIT 1").fetchone()[0])
    return ids


@pytest.fixture
def vectors():
    """A disposable Qdrant collection, mirroring test_vectors.py's guard."""
    from qdrant_client import QdrantClient

    from arc_search.config import IndexSettings
    from arc_search.index.vectors import VectorStore
    from test_vectors import QDRANT_URL, require_test_collection

    name = require_test_collection("faces_test")
    client = QdrantClient(url=QDRANT_URL)
    if client.collection_exists(name):
        client.delete_collection(name)
    vs = VectorStore(IndexSettings(qdrant_url=QDRANT_URL, collection=name), client=client)
    yield vs
    if client.collection_exists(name):
        client.delete_collection(name)
    client.close()


def _job(writer, vectors, extractor, fetcher, tmp_path, **kw):
    return Backfill(writer, vectors, extractor, fetcher, crop_root=tmp_path, **kw)


# --- pure ------------------------------------------------------------------


def test_the_report_never_double_counts_an_image():
    """Every image lands in exactly one outcome bucket. A report whose parts do
    not sum to the whole is worse than no report -- it looks authoritative."""
    s = BackfillStats(
        with_faces=3, provisional_empty=2, barren=1, fetch_failed=1, skipped=1, decode_failed=1
    )
    assert "images processed                       9" in s.report()


def test_the_reject_breakdown_is_reported_when_present():
    """plan-002 Phase 3 asks for this explicitly. If too_small dominates you are
    crawling thumbnail galleries; if too_blurry does, the vertical cannot support
    face search at all."""
    s = BackfillStats()
    s.add_rejects(FakeRejects(too_small=4, low_score=2))
    out = s.report()
    assert "too_small" in out and "low_score" in out
    # Sorted by count, worst first.
    assert out.index("too_small") < out.index("low_score")


def test_zero_rejects_are_not_printed_as_noise():
    s = BackfillStats()
    s.add_rejects(FakeRejects())
    assert "too_small" not in s.report()


# --- integration -----------------------------------------------------------


@pytest.mark.integration
@needs_pg
@pytest.mark.asyncio
async def test_a_full_pass_writes_both_stores_and_commits(writer, vectors, png, tmp_path):
    ids = _seed(writer, 3, png)
    ex = FakeExtractor(per_image=[2, 1, 0])
    job = _job(writer, vectors, ex, FakeFetcher(png), tmp_path)

    stats = await job.run()

    assert stats.with_faces == 2
    assert stats.faces_indexed == 3
    assert writer._exec("SELECT count(*) FROM face").fetchone()[0] == 3
    assert vectors.count() == 3
    assert len(ex.crops_written) == 3

    counts = writer.face_counts()
    assert counts["unexamined"] == 0
    assert counts["with_faces"] == 2
    assert counts["provisional"] == 1  # the 0-face image
    assert counts["barren"] == 0
    assert ids  # seeded


@pytest.mark.integration
@needs_pg
@pytest.mark.asyncio
async def test_an_empty_result_is_never_tombstoned_while_uncalibrated(
    writer, vectors, png, tmp_path
):
    """ADR-004, enforced at the runner level. Had this been wrong, a full drain
    would have permanently retired every image gated out by an uncalibrated
    threshold -- and the only symptom would be a corpus that looked small."""
    _seed(writer, 2, png)
    job = _job(writer, vectors, FakeExtractor([0, 0]), FakeFetcher(png), tmp_path)

    stats = await job.run()

    assert stats.provisional_empty == 2
    assert stats.barren == 0
    assert writer.provisional_count() == 2
    assert writer.face_counts()["barren"] == 0
    assert all(r[0] == -2 for r in writer._exec("SELECT face_count FROM image").fetchall()), (
        "an uncalibrated empty result must be re-examinable"
    )


@pytest.mark.integration
@needs_pg
@pytest.mark.asyncio
async def test_calibrated_runs_do_tombstone(writer, vectors, png, tmp_path):
    _seed(writer, 1, png)
    job = _job(writer, vectors, FakeExtractor([0]), FakeFetcher(png), tmp_path, calibrated=True)

    stats = await job.run()

    assert (stats.barren, stats.provisional_empty) == (1, 0)
    assert writer.face_counts()["barren"] == 1


@pytest.mark.integration
@needs_pg
@pytest.mark.asyncio
async def test_a_failed_fetch_does_not_stall_the_queue(writer, vectors, png, tmp_path):
    """THE failure mode for an unattended drain.

    A failed image keeps face_count = -1, so it is still first in the queue. If
    the cursor did not advance past it, the very next query would hand back the
    same image forever and the run would spin on one dead URL, making no
    progress while looking busy.
    """
    _seed(writer, 3, png)
    fetcher = FakeFetcher(png, failures={"https://conf.test/i/0.png": FetchError("u", "boom")})
    job = _job(writer, vectors, FakeExtractor([1, 1]), fetcher, tmp_path)

    stats = await job.run()

    assert stats.fetch_failed == 1
    assert stats.examined == 3, "the run must reach every image exactly once"
    assert len(fetcher.requested) == 3, "the dead URL must be tried once, not repeatedly"
    # The failed one stays on the queue for the NEXT run -- a transient 503
    # should be retried, just not in an infinite loop.
    assert writer.unexamined_count() == 1


@pytest.mark.integration
@needs_pg
@pytest.mark.asyncio
async def test_an_unexpected_exception_does_not_end_the_run(writer, vectors, png, tmp_path):
    """A 20k-image drain finds malformed URLs and hosts httpx did not expect.
    None of those are worth losing the other 19,999 images over."""
    _seed(writer, 3, png)
    fetcher = FakeFetcher(png, failures={"https://conf.test/i/1.png": RuntimeError("tls exploded")})
    job = _job(writer, vectors, FakeExtractor([1, 1]), fetcher, tmp_path)

    stats = await job.run()

    assert stats.fetch_failed == 1
    assert stats.with_faces == 2
    assert stats.examined == 3


@pytest.mark.integration
@needs_pg
@pytest.mark.asyncio
async def test_a_robots_skip_is_not_counted_as_a_failure(writer, vectors, png, tmp_path):
    """Skipped and failed mean different things. Conflating them would make a
    politeness decision look like an outage in the report."""
    _seed(writer, 2, png)
    fetcher = FakeFetcher(
        png, failures={"https://conf.test/i/0.png": Skipped("u", "robots_disallow")}
    )
    job = _job(writer, vectors, FakeExtractor([1]), fetcher, tmp_path)

    stats = await job.run()

    assert (stats.skipped, stats.fetch_failed) == (1, 0)


@pytest.mark.integration
@needs_pg
@pytest.mark.asyncio
async def test_limit_stops_early_and_leaves_the_rest_queued(writer, vectors, png, tmp_path):
    _seed(writer, 5, png)
    job = _job(writer, vectors, FakeExtractor([1] * 5), FakeFetcher(png), tmp_path)

    stats = await job.run(limit=2)

    assert stats.examined == 2
    assert writer.unexamined_count() == 3


@pytest.mark.integration
@needs_pg
@pytest.mark.asyncio
async def test_a_rerun_is_idempotent(writer, vectors, png, tmp_path):
    """Resume after a kill. The second pass must converge on the same corpus,
    not double it -- both stores clear before rewriting."""
    _seed(writer, 2, png)
    ex = FakeExtractor([1, 1])
    await _job(writer, vectors, ex, FakeFetcher(png), tmp_path).run()

    faces_after_first = writer._exec("SELECT count(*) FROM face").fetchone()[0]
    vectors_after_first = vectors.count()

    # Everything is committed, so a second run has nothing to do.
    stats = await _job(writer, vectors, FakeExtractor([1, 1]), FakeFetcher(png), tmp_path).run()

    assert stats.examined == 0
    assert writer._exec("SELECT count(*) FROM face").fetchone()[0] == faces_after_first
    assert vectors.count() == vectors_after_first


@pytest.mark.integration
@needs_pg
@pytest.mark.asyncio
async def test_stop_finishes_the_image_in_flight(writer, vectors, png, tmp_path):
    """Ctrl-C must not leave an image half-written across two stores."""
    _seed(writer, 4, png)
    job = _job(writer, vectors, FakeExtractor([1] * 4), FakeFetcher(png), tmp_path)
    job.stop()

    stats = await job.run()

    assert stats.examined == 0
    # Nothing partially committed: no face rows without a committed face_count.
    orphans = writer._exec(
        "SELECT count(*) FROM face f JOIN image i ON i.id = f.image_id WHERE i.face_count < 0"
    ).fetchone()[0]
    assert orphans == 0


@pytest.mark.integration
@needs_pg
@pytest.mark.asyncio
async def test_every_indexed_vector_joins_back_to_postgres(writer, vectors, png, tmp_path):
    """The invariant the query UI depends on. A vector with no row is a hit the
    interface can neither explain nor attribute."""
    _seed(writer, 3, png)
    await _job(writer, vectors, FakeExtractor([2, 1, 1]), FakeFetcher(png), tmp_path).run()

    pg_ids = {r[0] for r in writer._exec("SELECT id FROM face").fetchall()}
    for point in vectors.client.scroll(vectors.name, limit=100, with_payload=True)[0]:
        assert point.payload["face_id"] in pg_ids
    assert vectors.count() == len(pg_ids) == 4
