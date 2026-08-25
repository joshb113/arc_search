"""Multi-collection support and the whole-image embedding queue (ADR-005).

Phase 1 of plan-005: schema and collections, no model in the loop. What matters
here is the two structural facts that the rest of the plan rests on:

  * faces and whole-images cannot share a collection, because they do not share
    a point identity
  * `embed_state` is a SEPARATE column from `face_count`, because an image can
    be face-examined without being embedded, or the reverse
"""

from __future__ import annotations

import pytest

from arc_search.config import CollectionSpec, IndexSettings
from arc_search.index.store import EMBED_PROVISIONAL, EMBEDDED, UNEMBEDDED
from pgfixtures import ctx as _ctx
from pgfixtures import img as _img
from pgfixtures import needs_pg

# --- the collection split --------------------------------------------------


def test_face_and_image_collections_are_distinct():
    """They cannot be merged. Named vectors share a POINT, and a point needs one
    identity -- but a crowd photo yields several face points and exactly one
    image point. Measured on the live corpus: 2,828 face points, 4,712 images."""
    cfg = IndexSettings()
    face, image = cfg.face_spec(), cfg.image_spec()
    assert face.name != image.name
    assert face.point_id == "uuid"
    assert image.point_id == "image_id"


def test_the_image_collection_is_keyed_by_image_id():
    """This is what removes a whole class of bug. For faces, a vector can be
    orphaned from its Postgres row and `hydrate` has to drop it. For images the
    point id IS the row id, so an orphan is not representable."""
    assert IndexSettings().image_spec().point_id == "image_id"


def test_the_whole_image_dimension_is_the_measured_one():
    """768 is measured -- dinov2-base and siglip2-base are both 768-d. If this
    ever silently becomes 512 or 1024, the 280 GB scale target moves with it."""
    cfg = IndexSettings()
    assert cfg.scene_dim == 768
    assert cfg.text_dim == 768
    assert cfg.image_spec().dim == 768


def test_scene_and_text_dims_are_separate_settings():
    """They are equal today and writing that as one number would hide the day
    they diverge -- which happens the moment either model is swapped."""
    cfg = IndexSettings(scene_dim=1024)
    assert cfg.scene_dim == 1024
    assert cfg.text_dim == 768


def test_the_face_collection_still_defaults_unchanged():
    """ADR-005 must not disturb the live face index. `collection` and
    `vector_dim` stay bound to ARC_INDEX_COLLECTION / ARC_INDEX_VECTOR_DIM."""
    cfg = IndexSettings()
    assert (cfg.collection, cfg.vector_dim) == ("faces", 512)
    assert cfg.face_spec() == CollectionSpec(
        name="faces", dim=512, hnsw_m=16, hnsw_ef_construct=200, search_ef=256, point_id="uuid"
    )


def test_a_vector_store_defaults_to_the_face_collection():
    """Every pre-ADR-005 call site passes no spec and must keep working."""
    from arc_search.index.vectors import VectorStore

    store = VectorStore(IndexSettings(), client=object())  # type: ignore[arg-type]
    assert store.name == "faces"
    assert store.dim == 512


def test_a_vector_store_takes_an_explicit_spec():
    from arc_search.index.vectors import VectorStore

    cfg = IndexSettings()
    store = VectorStore(cfg, client=object(), spec=cfg.image_spec())  # type: ignore[arg-type]
    assert store.name == "images"
    assert store.dim == 768


def test_dimension_validation_follows_the_spec_not_the_global_config():
    """The bug this prevents: a 768-d scene vector being accepted because the
    store validated against the FACE collection's 512."""
    import uuid

    import numpy as np

    from arc_search.index.vectors import VectorRecord, VectorStore

    cfg = IndexSettings()
    store = VectorStore(cfg, client=object(), spec=cfg.image_spec())  # type: ignore[arg-type]

    ok = VectorRecord(uuid.uuid4(), np.zeros(768, dtype=np.float32), 1, 1, 0.5)
    store._point(ok)  # must not raise

    wrong = VectorRecord(uuid.uuid4(), np.zeros(512, dtype=np.float32), 1, 1, 0.5)
    with pytest.raises(ValueError, match=r"expected \(768,\)"):
        store._point(wrong)


# --- embed_state -----------------------------------------------------------


@pytest.mark.integration
@needs_pg
def test_the_crawl_tier_leaves_images_unembedded(writer, png):
    writer.handle(_img(png), _ctx())
    assert writer.unembedded_count() == 1
    (queued,) = writer.unembedded_images()
    assert queued.url == "https://conf.test/i/ada.png"


@pytest.mark.integration
@needs_pg
def test_embed_state_is_independent_of_face_count(writer, png):
    """The reason it is a separate column. Examining faces must not mark an
    image embedded, and embedding must not mark it face-examined -- they are
    different questions and will drain at different times."""
    writer.handle(_img(png), _ctx())
    image_id = writer._exec("SELECT id FROM image ORDER BY id DESC LIMIT 1").fetchone()[0]

    writer.mark_examined(image_id, 0)  # face pass finishes
    assert writer.unembedded_count() == 1, "a face pass must not mark it embedded"

    writer.mark_embedded(image_id)
    assert writer.unembedded_count() == 0
    # ...and the face verdict is untouched.
    assert writer.face_counts()["provisional"] == 1


@pytest.mark.integration
@needs_pg
def test_mark_embedded_commits_and_leaves_the_queue(writer, png):
    writer.handle(_img(png), _ctx())
    image_id = writer._exec("SELECT id FROM image ORDER BY id DESC LIMIT 1").fetchone()[0]
    writer.mark_embedded(image_id)

    state = writer._exec("SELECT embed_state FROM image WHERE id = %s", (image_id,)).fetchone()[0]
    assert state == EMBEDDED
    assert writer.unembedded_images() == []
    assert writer.embed_counts() == {"unembedded": 0, "embedded": 1, "provisional": 0}


@pytest.mark.integration
@needs_pg
def test_the_embed_queue_pages_by_keyset(writer):
    """Same rule as the face queue: rows leave as they are processed, so an
    OFFSET walk over a shrinking set skips work silently."""
    from imagefixtures import make_image

    ids = []
    for seed in range(2, 7):
        writer.handle(_img(make_image(seed, "PNG"), url=f"https://conf.test/e/{seed}.png"), _ctx())
        ids.append(writer._exec("SELECT id FROM image ORDER BY id DESC LIMIT 1").fetchone()[0])

    first = writer.unembedded_images(limit=2)
    assert [q.image_id for q in first] == ids[:2]
    for q in first:
        writer.mark_embedded(q.image_id)

    second = writer.unembedded_images(limit=2, after_id=first[-1].image_id)
    assert [q.image_id for q in second] == ids[2:4]


@pytest.mark.integration
@needs_pg
def test_the_reserved_provisional_state_is_allowed_by_the_constraint(writer, png):
    """Nothing writes -2 yet. The CHECK permits it now because plan-005 Phase 3
    re-measures min_image_dim for visual search, that gate will be uncalibrated,
    and ADR-004 forbids an uncalibrated gate writing an irreversible verdict.
    Widening a CHECK on a 30M-row table later is expensive; allowing it is free."""
    writer.handle(_img(png), _ctx())
    image_id = writer._exec("SELECT id FROM image ORDER BY id DESC LIMIT 1").fetchone()[0]

    writer._exec("UPDATE image SET embed_state = %s WHERE id = %s", (EMBED_PROVISIONAL, image_id))
    assert writer.embed_counts()["provisional"] == 1
    # ...and it is NOT on the work queue, so it cannot be re-fetched forever.
    assert writer.unembedded_count() == 0


@pytest.mark.integration
@needs_pg
def test_embed_state_rejects_undefined_values(writer, png):
    """0 is deliberately not a valid state. face_count uses 0 for 'examined,
    found nothing', and a reader who just learned that would read 0 here as
    failure rather than success."""
    import psycopg

    writer.handle(_img(png), _ctx())
    image_id = writer._exec("SELECT id FROM image ORDER BY id DESC LIMIT 1").fetchone()[0]

    for bad in (0, 2, -3):
        with pytest.raises(psycopg.Error):
            writer._exec("UPDATE image SET embed_state = %s WHERE id = %s", (bad, image_id))


@pytest.mark.integration
@needs_pg
def test_the_default_state_is_unembedded(writer, png):
    """The crawl tier writes nothing to this column; the default carries it.
    face_count defaulted wrong once and would have skipped the whole corpus."""
    writer.handle(_img(png), _ctx())
    state = writer._exec("SELECT embed_state FROM image").fetchone()[0]
    assert state == UNEMBEDDED
