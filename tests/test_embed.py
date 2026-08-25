"""Whole-image embedding and the named-vector collection (plan-005 Phase 2).

The model itself is behind the ``gpu`` marker -- loading DINOv2 and SigLIP costs
~30 s and ~1.5 GB, and a test suite nobody runs is worth nothing. What runs
everywhere is the wiring: config, output shapes, guards, and the named-vector
write path against a live Qdrant.
"""

from __future__ import annotations

import numpy as np
import pytest

from arc_search.config import EmbedSettings, IndexSettings

# --- config ----------------------------------------------------------------


def test_models_are_swappable_by_env(monkeypatch):
    """DINOv3 is gated=manual, so DINOv2 is the default and the swap must be a
    config change plus a re-embed, not a rewrite. Hardcoding either would decide
    an open question by accident."""
    monkeypatch.setenv("ARC_EMBED_SCENE_MODEL", "facebook/dinov3-vitb16-pretrain-lvd1689m")
    assert EmbedSettings().scene_model.startswith("facebook/dinov3")
    assert EmbedSettings().text_model.startswith("google/siglip")


def test_the_default_scene_model_is_ungated():
    """If this ever defaults to a gated repo, a fresh clone fails at first run
    with a 401 that reads like a network problem."""
    assert EmbedSettings().scene_model == "facebook/dinov2-base"


def test_the_sigmoid_scale_flag_is_on_by_default():
    """SigLIP raw cosine is not the model's scale. Read raw, a working model
    looks undiscriminating -- it cost a false alarm during bring-up."""
    assert EmbedSettings().text_uses_sigmoid_scale is True


# --- the named-vector spec -------------------------------------------------


def test_the_image_collection_declares_both_named_vectors():
    named = dict(IndexSettings().image_spec().named)
    assert named == {"scene": 768, "text": 768}


def test_the_face_collection_has_no_named_vectors():
    """Faces are per-face and cannot share a point with per-image vectors.
    An empty `named` is what keeps the existing collection single-vector."""
    assert IndexSettings().face_spec().named == ()


def test_named_vector_dims_follow_their_own_settings():
    cfg = IndexSettings(text_dim=1024)
    assert dict(cfg.image_spec().named) == {"scene": 768, "text": 1024}


# --- guards on the write path ----------------------------------------------


def _store(**kw):
    from arc_search.index.vectors import VectorStore

    cfg = IndexSettings(**kw)
    return VectorStore(cfg, client=object(), spec=cfg.image_spec())  # type: ignore[arg-type]


def test_writing_one_vector_of_a_pair_is_refused():
    """A half-written point would make an image findable by scene and invisible
    to text -- the kind of half-state that surfaces months later as 'search is
    broken' rather than as an error."""
    store = _store()
    with pytest.raises(ValueError, match=r"missing named vectors.*text"):
        store.upsert_image(1, {"scene": np.zeros(768, dtype=np.float32)})


def test_a_wrong_sized_named_vector_is_refused():
    store = _store()
    with pytest.raises(ValueError, match=r"'scene'.*expected \(768,\)"):
        store.upsert_image(
            1,
            {"scene": np.zeros(512, dtype=np.float32), "text": np.zeros(768, dtype=np.float32)},
        )


def test_searching_an_unknown_vector_name_is_refused():
    """scene and text are different spaces; mixing them returns nonsense. The
    caller names which mode it means rather than a default guessing."""
    store = _store()
    with pytest.raises(ValueError, match=r"no vector 'nope'"):
        store.search_named("nope", np.zeros(768, dtype=np.float32))


def test_a_wrong_sized_query_is_refused():
    store = _store()
    with pytest.raises(ValueError, match=r"expected \(768,\)"):
        store.search_named("scene", np.zeros(512, dtype=np.float32))


def test_upsert_image_refuses_a_collection_without_named_vectors():
    """Calling the per-image path against the faces collection is a category
    error, not something to paper over."""
    from arc_search.index.vectors import VectorStore

    cfg = IndexSettings()
    faces = VectorStore(cfg, client=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="no named vectors"):
        faces.upsert_image(1, {"scene": np.zeros(768, dtype=np.float32)})


# --- integration: a real Qdrant --------------------------------------------


@pytest.fixture
def images_store():
    from qdrant_client import QdrantClient

    from arc_search.index.vectors import VectorStore
    from test_vectors import QDRANT_URL

    name = "images_test"
    assert name.endswith("_test")  # same guard as the other collections
    cfg = IndexSettings(qdrant_url=QDRANT_URL, image_collection=name)
    client = QdrantClient(url=QDRANT_URL)
    if client.collection_exists(name):
        client.delete_collection(name)
    vs = VectorStore(cfg, client=client, spec=cfg.image_spec())
    vs.ensure_collection()
    yield vs
    if client.collection_exists(name):
        client.delete_collection(name)
    client.close()


def _unit(seed: int, dim: int = 768) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.mark.integration
def test_the_collection_is_created_with_two_named_vectors(images_store):
    info = images_store.client.get_collection(images_store.name)
    assert {k: v.size for k, v in info.config.params.vectors.items()} == {
        "scene": 768,
        "text": 768,
    }
    assert images_store.verify() == []


@pytest.mark.integration
def test_the_point_id_is_the_image_id(images_store):
    """The property that makes a mapping table unnecessary and an orphaned
    vector structurally impossible."""
    target = _unit(1)
    images_store.upsert_image(4712, {"scene": target, "text": _unit(2)})
    (hit,) = images_store.search_named("scene", target, limit=1)
    assert hit[0] == 4712


@pytest.mark.integration
def test_scene_and_text_are_independent_spaces(images_store):
    """Both vectors live on one point, but a scene query must not be answered
    from the text vector or the modes would silently bleed into each other."""
    scene, text = _unit(10), _unit(99)
    images_store.upsert_image(1, {"scene": scene, "text": text})

    assert images_store.search_named("scene", scene, limit=1)[0][1] == pytest.approx(1.0, abs=1e-3)
    assert images_store.search_named("text", text, limit=1)[0][1] == pytest.approx(1.0, abs=1e-3)
    # Querying the text space with the scene vector must NOT return ~1.0.
    assert images_store.search_named("text", scene, limit=1)[0][1] < 0.5


@pytest.mark.integration
def test_re_embedding_overwrites_rather_than_duplicating(images_store):
    """The point id is the row id, so a re-run after a crash converges. This is
    what lets `mark_embedded` be the commit marker."""
    images_store.upsert_image(7, {"scene": _unit(1), "text": _unit(2)})
    images_store.upsert_image(7, {"scene": _unit(3), "text": _unit(4)})
    assert images_store.count() == 1

    (hit,) = images_store.search_named("scene", _unit(3), limit=1)
    assert hit[0] == 7 and hit[1] == pytest.approx(1.0, abs=1e-3)


@pytest.mark.integration
def test_exclusions_are_applied_server_side(images_store):
    """Same rule as the face path: a suppressed image must not consume one of
    the k slots on its way to being dropped."""
    target = _unit(1)
    for i in (1, 2, 3):
        images_store.upsert_image(i, {"scene": target if i == 1 else _unit(i), "text": _unit(i)})

    assert images_store.search_named("scene", target, limit=3)[0][0] == 1
    hits = images_store.search_named("scene", target, limit=3, exclude=[1])
    assert 1 not in [h[0] for h in hits]
    assert len(hits) == 2


@pytest.mark.integration
def test_verify_reports_a_named_dimension_mismatch(images_store):
    """A named-vector collection returns a DICT of params, not one object.
    Reading `.size` off it would raise; reporting a mismatch against None would
    be worse, because verify() exists to be believed."""
    from arc_search.index.vectors import VectorStore

    wrong = IndexSettings(
        qdrant_url=images_store._cfg.qdrant_url,
        image_collection=images_store.name,
        text_dim=1024,
    )
    problems = VectorStore(wrong, client=images_store.client, spec=wrong.image_spec()).verify()
    assert problems == ["text: config=1024 live=768"]


# --- the model itself ------------------------------------------------------


@pytest.mark.gpu
def test_the_embedder_reports_its_real_device_and_dims():
    """Not what config asked for. This project has shipped the other version
    twice -- the crawler logging its requested rate, and onnxruntime silently
    running on CPU at 1/12th speed."""
    from arc_search.index.embed import ImageEmbedder

    e = ImageEmbedder()
    assert e.effective_device() == "unloaded"
    scene_dim, text_dim = e.dims()
    assert (scene_dim, text_dim) == (768, 768)
    assert e.effective_device().startswith("cuda")


@pytest.mark.gpu
def test_embedding_produces_normalized_vectors_of_the_declared_size():
    from PIL import Image

    from arc_search.index.embed import ImageEmbedder

    e = ImageEmbedder()
    imgs = [
        Image.fromarray(np.random.default_rng(i).integers(0, 255, (400, 300, 3), dtype=np.uint8))
        for i in range(4)
    ]
    out = e.embed_images(imgs)
    assert len(out) == 4
    for v in out:
        assert v.scene.shape == (768,)
        assert v.text.shape == (768,)
        assert np.linalg.norm(v.scene) == pytest.approx(1.0, abs=1e-3)
        assert np.linalg.norm(v.text) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.gpu
def test_an_empty_batch_does_not_touch_the_gpu():
    from arc_search.index.embed import ImageEmbedder

    assert ImageEmbedder().embed_images([]) == []


@pytest.mark.gpu
def test_the_sigmoid_parameters_are_read_off_the_model():
    """Measured on siglip2-base: scale 112.85, bias -16.77. Exposed rather than
    left for a caller to rediscover the hard way."""
    from arc_search.index.embed import ImageEmbedder

    scale, bias = ImageEmbedder().text_logit_params()
    assert 50 < scale < 200
    assert bias < 0
