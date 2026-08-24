"""VectorStore tests.

Two tiers, matching test_store.py. Shape and validation logic is pure and always
runs; anything that talks to Qdrant is marked ``integration`` and skipped unless
a server is reachable.

THESE TESTS DELETE THE COLLECTION THEY POINT AT.

As with ARC_TEST_PG_DSN, the guard is on the name rather than on remembering to
be careful: the fixture refuses any collection whose name does not end in
``_test``, so pointing it at the real ``faces`` collection fails loudly instead
of destroying an indexing run.

    docker compose up -d qdrant
    pytest -m integration
"""

from __future__ import annotations

import os
import uuid

import numpy as np
import pytest

from arc_search.config import IndexSettings
from arc_search.index.vectors import Hit, VectorRecord, VectorStore

QDRANT_URL = os.environ.get("ARC_TEST_QDRANT_URL", "http://127.0.0.1:6333")

TEST_COLLECTION_SUFFIX = "_test"


def require_test_collection(name: str) -> str:
    """Refuse any collection that is not clearly disposable."""
    if not name.endswith(TEST_COLLECTION_SUFFIX):
        raise ValueError(
            f"refusing to run: collection {name!r} does not end in "
            f"{TEST_COLLECTION_SUFFIX!r}. These tests DELETE the collection."
        )
    return name


def _qdrant_up() -> bool:
    try:
        import httpx

        return httpx.get(QDRANT_URL, timeout=2.0).status_code == 200
    except Exception:
        return False


needs_qdrant = pytest.mark.skipif(not _qdrant_up(), reason=f"no qdrant at {QDRANT_URL}")


def _unit(seed: int, dim: int = 512) -> np.ndarray:
    """An L2-normalized vector, like the ones ArcFace emits."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _record(seed: int, image_id: int = 1, face_id: int | None = None) -> VectorRecord:
    return VectorRecord(
        qdrant_id=uuid.uuid4(),
        embedding=_unit(seed),
        image_id=image_id,
        face_id=seed if face_id is None else face_id,
        quality=0.8,
    )


# --- pure ------------------------------------------------------------------


def test_require_test_collection_rejects_the_real_one():
    """The production collection is named `faces`. It must not be deletable."""
    with pytest.raises(ValueError, match="refusing to run"):
        require_test_collection("faces")
    assert require_test_collection("faces_test") == "faces_test"


def test_wrong_embedding_dimension_raises_rather_than_skipping():
    """A 512-d config that receives a 128-d vector means the model pack is not
    what config.py claims. Every vector already written is then suspect, so this
    has to be loud -- a skipped row would leave a half-wrong index behind."""
    store = VectorStore(IndexSettings(collection="faces_test"), client=object())  # type: ignore[arg-type]
    bad = VectorRecord(
        qdrant_id=uuid.uuid4(),
        embedding=np.zeros(128, dtype=np.float32),
        image_id=1,
        face_id=1,
        quality=0.5,
    )
    with pytest.raises(ValueError, match=r"shape \(128,\), expected \(512,\)"):
        store._point(bad)


def test_search_rejects_a_wrong_dimension_query():
    store = VectorStore(IndexSettings(collection="faces_test"), client=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="query embedding"):
        store.search(np.zeros(256, dtype=np.float32))


def test_upsert_of_nothing_is_not_a_round_trip():
    """An image with no qualifying face must not call Qdrant at all. The client
    here would raise on any attribute access."""

    class Explode:
        def __getattr__(self, name):  # pragma: no cover - must never be reached
            raise AssertionError(f"VectorStore touched the client: .{name}")

    store = VectorStore(IndexSettings(collection="faces_test"), client=Explode())  # type: ignore[arg-type]
    assert store.upsert([]) == 0


# --- integration -----------------------------------------------------------


@pytest.fixture
def store():
    """A VectorStore over a freshly created, empty test collection.

    Dropped BEFORE as well as after, for the same reason PostgresWriter's
    fixture wipes on both sides: a test that needs an empty collection has to
    make one, not hope the previous run cleaned up after itself.
    """
    from qdrant_client import QdrantClient

    name = require_test_collection("faces_test")
    cfg = IndexSettings(qdrant_url=QDRANT_URL, collection=name)
    client = QdrantClient(url=QDRANT_URL)

    if client.collection_exists(name):
        client.delete_collection(name)

    vs = VectorStore(cfg, client=client)
    vs.ensure_collection()
    yield vs

    if client.collection_exists(name):
        client.delete_collection(name)
    client.close()


@needs_qdrant
@pytest.mark.integration
def test_ensure_collection_is_idempotent(store):
    """Startup runs this every time. The second call must be a no-op, not an
    error and not a silent rebuild that discards the index."""
    assert store.ensure_collection() is False
    assert store.verify() == []


@needs_qdrant
@pytest.mark.integration
def test_the_collection_is_actually_configured_as_specified(store):
    """int8 quantization and cosine distance are not decoration -- they are the
    difference between the recall this engine needs and the recall it gets. If a
    settings change silently fails to apply, verify() is what says so."""
    info = store.client.get_collection(store.name)
    params = info.config.params.vectors
    assert params.size == 512
    assert params.distance.value == "Cosine"

    quant = info.config.quantization_config
    assert quant is not None, "quantization silently absent"
    assert quant.scalar.type.value == "int8"

    assert info.config.hnsw_config.m == 16
    assert info.config.hnsw_config.ef_construct == 200


@needs_qdrant
@pytest.mark.integration
def test_search_finds_the_vector_it_was_given(store):
    """The end-to-end smoke test that the version skew would have failed: writes
    succeeded against a 1.9.2 server and only the query 404'd."""
    recs = [_record(i, image_id=i) for i in range(1, 6)]
    assert store.upsert(recs) == 5

    hits = store.search(recs[2].embedding, limit=3)
    assert hits, "search returned nothing"
    assert hits[0].qdrant_id == recs[2].qdrant_id
    assert hits[0].score == pytest.approx(1.0, abs=1e-3)
    assert hits[0].image_id == 3
    assert hits[0].face_id == 3


@needs_qdrant
@pytest.mark.integration
def test_payload_survives_the_round_trip(store):
    """image_id and face_id are the join back to Postgres. A hit that loses them
    is a result the UI cannot attribute to a page, which makes it useless."""
    rec = _record(7, image_id=42, face_id=99)
    store.upsert([rec])
    (hit,) = store.search(rec.embedding, limit=1)
    assert isinstance(hit, Hit)
    assert (hit.image_id, hit.face_id) == (42, 99)
    assert hit.quality == pytest.approx(0.8)


@needs_qdrant
@pytest.mark.integration
def test_delete_image_removes_only_that_image(store):
    """Re-indexing one image must not disturb its neighbours."""
    store.upsert([_record(1, image_id=1), _record(2, image_id=1), _record(3, image_id=2)])
    assert store.count() == 3

    store.delete_image(1)
    assert store.count() == 1
    remaining = store.search(_unit(3), limit=5)
    assert [h.image_id for h in remaining] == [2]


@needs_qdrant
@pytest.mark.integration
def test_reindexing_an_image_leaves_no_orphans(store):
    """The reprocessing path, exactly as the crash-recovery contract runs it.

    An image re-examined after a crash (or by a better model) may yield a
    different number of faces. Delete-then-upsert has to converge on the new
    set -- an orphaned vector would score in queries with no Postgres row to
    join, producing a hit the UI can neither explain nor attribute.
    """
    store.upsert([_record(i, image_id=1) for i in (1, 2, 3)])
    assert store.count() == 3

    # Second pass finds only one face.
    store.delete_image(1)
    survivor = _record(9, image_id=1)
    store.upsert([survivor])

    assert store.count() == 1
    (hit,) = store.search(survivor.embedding, limit=5)
    assert hit.qdrant_id == survivor.qdrant_id


@needs_qdrant
@pytest.mark.integration
def test_excluded_vectors_never_reach_the_caller(store):
    """The opt-out list is a legal obligation, not a ranking preference.

    Applied server-side as must_not, so an excluded face does not consume one
    of the k slots on its way to being dropped -- filtering after retrieval
    would silently shrink the result set.
    """
    recs = [_record(i, image_id=i) for i in range(1, 5)]
    store.upsert(recs)
    target = recs[0]

    assert store.search(target.embedding, limit=4)[0].qdrant_id == target.qdrant_id

    hits = store.search(target.embedding, limit=4, exclude=[target.qdrant_id])
    assert target.qdrant_id not in {h.qdrant_id for h in hits}
    assert len(hits) == 3, "exclusion must not cost a result slot"


@needs_qdrant
@pytest.mark.integration
def test_upsert_is_replayable(store):
    """Same ids written twice is one row, not two. A retried batch after a
    timeout must not double-count the corpus."""
    recs = [_record(i, image_id=1) for i in (1, 2)]
    store.upsert(recs)
    store.upsert(recs)
    assert store.count() == 2


@needs_qdrant
@pytest.mark.integration
def test_verify_reports_a_dimension_mismatch_instead_of_writing(store):
    """Changing vector_dim under a live collection is a migration, not a config
    edit. ensure_collection must not quietly rebuild, and verify must say so."""
    mismatched = VectorStore(
        IndexSettings(qdrant_url=QDRANT_URL, collection=store.name, vector_dim=256),
        client=store.client,
    )
    assert mismatched.ensure_collection() is False  # did NOT rebuild
    problems = mismatched.verify()
    assert any("vector_dim" in p for p in problems), problems


@needs_qdrant
@pytest.mark.integration
def test_count_on_a_missing_collection_is_zero_not_an_error(store):
    """Startup reporting runs before the first index pass."""
    absent = VectorStore(
        IndexSettings(qdrant_url=QDRANT_URL, collection="definitely_absent_test"),
        client=store.client,
    )
    assert absent.count() == 0
    assert absent.verify() == ["collection_missing"]
