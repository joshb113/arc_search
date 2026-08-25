"""EmbeddingSink: whole-image embedding inside the crawl loop.

The point of this file is the failure isolation. Putting a GPU in a five-hour
crawl loop is only defensible if the crawl is never worse off for it, so every
way embedding can fail gets a test that proves the crawl survived and the image
stayed on the work queue.

Models are faked. A test that needs 1.5 GB of weights to prove a cursor advanced
is a test nobody runs.
"""

from __future__ import annotations

import numpy as np
import pytest

from arc_search.config import EmbedSettings, IndexSettings
from arc_search.index.embed_sink import EmbeddingSink
from imagefixtures import make_image
from pgfixtures import ctx as _ctx
from pgfixtures import img as _img
from pgfixtures import needs_pg

# --- fakes -----------------------------------------------------------------


class FakeVectors:
    def __init__(self, named=(("scene", 768), ("text", 768)), fail=False):
        from arc_search.config import CollectionSpec

        self.spec = CollectionSpec(name="images_test", dim=768, named=named)
        self.name = "images_test"
        self.written: dict[int, dict] = {}
        self.fail = fail

    def ensure_collection(self):
        return True

    def verify(self):
        return []

    def upsert_image(self, image_id, vectors):
        if self.fail:
            raise RuntimeError("qdrant is on fire")
        self.written[image_id] = vectors


class FakeVec:
    def __init__(self):
        self.scene = np.zeros(768, dtype=np.float32)
        self.text = np.zeros(768, dtype=np.float32)


class FakeEmbedder:
    def __init__(self, dims=(768, 768), fail=False, device="cuda:0"):
        self._dims = dims
        self.fail = fail
        self._device = device
        self.calls = 0
        self.batch_sizes: list[int] = []

    def dims(self):
        return self._dims

    def effective_device(self):
        return self._device

    def embed_images(self, images):
        self.calls += 1
        self.batch_sizes.append(len(images))
        if self.fail:
            raise RuntimeError("CUDA out of memory")
        return [FakeVec() for _ in images]


def _sink(writer, *, vectors=None, embedder=None, batch_size=4):
    return EmbeddingSink(
        writer,
        writer=writer,
        vectors=vectors if vectors is not None else FakeVectors(),
        embedder=embedder if embedder is not None else FakeEmbedder(),
        cfg=EmbedSettings(batch_size=batch_size),
    )


def _crawl(sink, n, png=None):
    """Push n distinct images through the sink the way the crawler does."""
    for i in range(n):
        body = png if png is not None else make_image(200 + i, "PNG")
        sink.handle(_img(body, url=f"https://conf.test/w/{i}.png"), _ctx())


# --- the happy path --------------------------------------------------------


@pytest.mark.integration
@needs_pg
def test_images_are_embedded_and_committed(writer):
    v, e = FakeVectors(), FakeEmbedder()
    sink = _sink(writer, vectors=v, embedder=e, batch_size=4)
    assert sink.prepare() is True

    _crawl(sink, 4)

    assert sink.embedded == 4
    assert len(v.written) == 4
    assert writer.embed_counts() == {"unembedded": 0, "embedded": 4, "provisional": 0}


@pytest.mark.integration
@needs_pg
def test_both_named_vectors_are_written(writer):
    v = FakeVectors()
    sink = _sink(writer, vectors=v, batch_size=1)
    sink.prepare()
    _crawl(sink, 1)
    (written,) = v.written.values()
    assert set(written) == {"scene", "text"}


@pytest.mark.integration
@needs_pg
def test_embedding_is_batched(writer):
    """The GPU wants batches -- 70 img/s at batch 1 against 179 at batch 7 --
    and the crawl yields one image at a time."""
    e = FakeEmbedder()
    sink = _sink(writer, embedder=e, batch_size=4)
    sink.prepare()
    _crawl(sink, 8)
    assert e.calls == 2
    assert e.batch_sizes == [4, 4]


@pytest.mark.integration
@needs_pg
def test_the_tail_is_flushed_on_close(writer):
    """Without this, up to batch_size-1 images per run are fetched, recorded and
    never embedded -- a slow leak showing up only as a queue that never empties."""
    e = FakeEmbedder()
    sink = _sink(writer, embedder=e, batch_size=10)
    sink.prepare()
    _crawl(sink, 3)

    assert sink.embedded == 0, "not full yet, so nothing should have flushed"
    assert writer.unembedded_count() == 3

    sink.flush()
    assert sink.embedded == 3
    assert writer.unembedded_count() == 0


# --- failure isolation: the reason this design is acceptable ---------------


@pytest.mark.integration
@needs_pg
def test_a_model_failure_does_not_break_the_crawl(writer):
    """A CUDA OOM must cost the batch, not the five-hour crawl."""
    e = FakeEmbedder(fail=True)
    sink = _sink(writer, embedder=e, batch_size=2)
    sink.prepare()

    _crawl(sink, 4)  # must not raise

    assert sink.failed == 4
    assert sink.embedded == 0
    # ...and every image is still recorded, and still on the work queue.
    assert writer._exec("SELECT count(*) FROM image").fetchone()[0] == 4
    assert writer.unembedded_count() == 4


@pytest.mark.integration
@needs_pg
def test_a_vector_store_failure_leaves_the_image_queued(writer):
    """Qdrant being down must not mark anything embedded -- the vectors are the
    thing embed_state is claiming exist."""
    sink = _sink(writer, vectors=FakeVectors(fail=True), batch_size=2)
    sink.prepare()
    _crawl(sink, 2)

    assert sink.failed == 2
    assert writer.unembedded_count() == 2
    assert writer.embed_counts()["embedded"] == 0


@pytest.mark.integration
@needs_pg
def test_an_undecodable_image_does_not_stop_the_batch(writer):
    """One corrupt file in a 30M-image crawl must not end the run."""
    sink = _sink(writer, batch_size=2)
    sink.prepare()

    sink.handle(_img(b"\x89PNG\r\n\x1a\nnot-an-image", url="https://conf.test/bad.png"), _ctx())
    _crawl(sink, 2)
    sink.flush()

    assert sink.failed == 1
    assert sink.embedded == 2, "the good images still went through"


@pytest.mark.integration
@needs_pg
def test_a_failed_prepare_degrades_instead_of_raising(writer):
    """If the model stack is unavailable the crawl must run exactly as it did
    before ADR-005, with images left for the backfill."""

    class Broken(FakeEmbedder):
        def dims(self):
            raise RuntimeError("no CUDA device")

    sink = _sink(writer, embedder=Broken())
    assert sink.prepare() is False

    _crawl(sink, 3)  # must not raise
    sink.flush()

    assert sink.embedded == 0
    assert writer._exec("SELECT count(*) FROM image").fetchone()[0] == 3
    assert writer.unembedded_count() == 3, "left for the backfill, not lost"


@pytest.mark.integration
@needs_pg
def test_a_dimension_mismatch_is_refused_at_startup(writer):
    """A model swap that disagrees with the collection would write vectors
    nobody can search. DINOv3 ViT-L is 1024-d, so this is a live risk."""
    sink = _sink(writer, embedder=FakeEmbedder(dims=(1024, 768)))
    assert sink.prepare() is False
    _crawl(sink, 2)
    assert sink.embedded == 0
    assert writer.unembedded_count() == 2


# --- pass-through behaviour ------------------------------------------------


@pytest.mark.integration
@needs_pg
def test_the_wrapped_sink_still_records_everything(writer, png):
    """The decorator must be transparent -- provenance, dedup and verdicts all
    keep working, because the crawl tier does not know it is wrapped."""
    sink = _sink(writer, batch_size=1)
    sink.prepare()

    assert sink.handle(_img(png), _ctx()) == "new"
    assert sink.handle(_img(png), _ctx()) == "exact_dup"
    assert writer._exec("SELECT count(*) FROM image").fetchone()[0] == 1
    assert writer._exec("SELECT count(*) FROM image_source").fetchone()[0] == 1


@pytest.mark.integration
@needs_pg
def test_duplicates_are_not_re_embedded(writer, png):
    """A duplicate's bytes are already represented by a row that either has
    vectors or is on the backfill queue. Re-embedding would be wasted GPU."""
    e = FakeEmbedder()
    sink = _sink(writer, embedder=e, batch_size=1)
    sink.prepare()

    sink.handle(_img(png), _ctx())
    sink.handle(_img(png), _ctx(page_url="https://conf.test/other/"))

    assert e.calls == 1
    assert sink.embedded == 1


@pytest.mark.integration
@needs_pg
def test_record_page_passes_through(writer):
    sink = _sink(writer)
    page_id = sink.record_page("https://conf.test/a/", "A", 200)
    assert isinstance(page_id, int)
    assert writer._exec("SELECT count(*) FROM page").fetchone()[0] == 1


def test_the_sink_builds_its_own_collaborators_by_default():
    """run.py constructs it with two arguments; everything else is defaulted so
    the wiring in the crawl loop stays a single line."""
    cfg = IndexSettings()
    assert dict(cfg.image_spec().named) == {"scene": 768, "text": 768}


def test_close_flushes_then_closes_the_wrapped_sink():
    """run.py calls sink.close() exactly once, so the decorator owns delegating
    it. Verified against a fake inner rather than the shared writer, because
    closing the real connection tears the fixture down mid-test."""

    class Inner:
        def __init__(self):
            self.closed = False

        def handle(self, fetched, context):
            return "new"

        def close(self):
            self.closed = True

    class Writer:
        def image_id_for_sha1(self, digest):
            return 1

        def mark_embedded(self, image_id):
            pass

    inner, e = Inner(), FakeEmbedder()
    sink = EmbeddingSink(
        inner,
        writer=Writer(),
        vectors=FakeVectors(),
        embedder=e,
        cfg=EmbedSettings(batch_size=100),
    )
    sink.prepare()
    sink.handle(_img(make_image(1, "PNG")), _ctx())

    assert sink.embedded == 0, "buffered, not yet flushed"
    sink.close()
    assert sink.embedded == 1, "close must flush the tail"
    assert inner.closed is True, "close must delegate"
