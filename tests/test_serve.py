"""Query API tests.

Everything is injected, so none of this needs Postgres, Qdrant or a GPU. What is
pinned here is the behaviour that would be embarrassing to get wrong on a tool
that answers questions about people:

  * no verdict is rendered while the thresholds are uncalibrated
  * the opt-out list is consulted on EVERY search
  * the uploaded photo is never written to disk
  * a crop path out of the database cannot escape the crop directory
  * a face with no Postgres row never reaches the page

The database-backed hydration query has its own integration test at the bottom.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import pytest

from imagefixtures import make_image
from pgfixtures import needs_pg

pytest.importorskip("fastapi", reason="serve extra not installed")
pytest.importorskip("cv2", reason="fast CI installs no opencv; the gpu job does")

from fastapi.testclient import TestClient

from arc_search.config import FaceSettings, SearchSettings
from arc_search.serve.app import Deps, create_app
from arc_search.serve.repo import FaceHit

# --- fakes -----------------------------------------------------------------


class FakeQueryFace:
    def __init__(self, quality=0.8):
        rng = np.random.default_rng(1)
        v = rng.standard_normal(512).astype(np.float32)
        self.embedding = v / np.linalg.norm(v)
        self.quality = quality
        self.det_score = 0.91
        self.age_est = 33


class FakeRejects:
    def __init__(self, **kw):
        self.too_small = kw.get("too_small", 0)
        self.low_score = kw.get("low_score", 0)
        self.too_blurry = kw.get("too_blurry", 0)
        self.bad_pose = kw.get("bad_pose", 0)
        self.underage = kw.get("underage", 0)


class FakeExtractor:
    def __init__(self, faces=None, rejects=None):
        self._faces = faces if faces is not None else [FakeQueryFace()]
        self._rejects = rejects or FakeRejects()
        self.calls = 0

    def extract(self, img):
        self.calls += 1
        return list(self._faces), self._rejects


class FakeVectors:
    def __init__(self, hits=None):
        self._hits = hits or []
        self.last_exclude = None
        self.searches = 0

    def search(self, embedding, limit=None, exclude=()):
        self.searches += 1
        self.last_exclude = list(exclude)
        return self._hits

    def count(self):
        return len(self._hits)


class FakeRepo:
    def __init__(self, hits=None, exclusions=(), crop_rows=None):
        self._hits = hits or []
        self._exclusions = list(exclusions)
        self._crop_rows = crop_rows or {}
        self.exclusion_calls = 0

    def exclusions(self):
        self.exclusion_calls += 1
        return self._exclusions

    def hydrate(self, hits):
        return self._hits

    def stats(self):
        return {
            "faces": 1300,
            "images_with_faces": 1298,
            "images": 2311,
            "pages": 5381,
            "exclusions": len(self._exclusions),
        }

    def _exec(self, sql, params=None):
        class R:
            def __init__(self, row):
                self._row = row

            def fetchone(self):
                return self._row

        return R(self._crop_rows.get(params[0]) if params else None)


def _hit(face_id=1, score=0.87, label="Ada Lovelace", crop="ab/cd/x.webp"):
    return FaceHit(
        face_id=face_id,
        qdrant_id=uuid.uuid4(),
        score=score,
        quality=0.7,
        crop_path=crop,
        label=label,
        image_url="https://conf.test/i/ada.png",
        src_width=220,
        src_height=180,
        det_score=0.9,
        age_est=33,
        pages=["https://conf.test/speaker/ada/"],
    )


def _client(
    *,
    hits=None,
    exclusions=(),
    extractor=None,
    calibrated=False,
    crop_root=None,
    crop_rows=None,
    vectors=None,
):
    repo = FakeRepo(hits=hits, exclusions=exclusions, crop_rows=crop_rows)
    deps = Deps(
        repo=repo,
        vectors=vectors or FakeVectors(hits=hits or []),
        extractor=extractor or FakeExtractor(),
        face_cfg=FaceSettings(),
        search_cfg=SearchSettings(calibrated=calibrated),
        crop_root=crop_root or Path("data/crops"),
    )
    return TestClient(create_app(deps)), deps


def _upload(client, body=None, **params):
    return client.post(
        "/search",
        files={"photo": ("q.png", body or make_image(7, "PNG"), "image/png")},
        params=params,
    )


# --- no verdicts -----------------------------------------------------------


def test_the_uncalibrated_banner_is_shown():
    """Non-negotiable #5 made visible. A number with no calibration behind it
    must not be presented as if it meant something."""
    client, _ = _client(hits=[_hit()])
    body = _upload(client).text
    assert "UNCALIBRATED" in body
    assert "raw cosine similarity" in body


def test_no_match_verdict_language_appears_anywhere():
    """The placeholders would render 0.87 as 'near certain'. One measured
    impostor pair scores 0.651, so that word would be a lie about a person."""
    client, _ = _client(hits=[_hit(score=0.87)])
    body = _upload(client).text.lower()
    for word in ("near certain", "strong match", "confident", "verified", "identified as"):
        assert word not in body, f"rendered a verdict: {word!r}"
    assert "0.8700" in body  # the raw score IS shown


def test_json_labels_the_score_as_raw_cosine():
    """A client reading this must not mistake it for a probability."""
    client, _ = _client(hits=[_hit(score=0.87)])
    data = _upload(client, format="json").json()
    assert data["score_type"] == "raw_cosine"
    assert data["calibrated"] is False
    assert data["results"][0]["cosine"] == 0.87


def test_the_banner_disappears_once_calibrated():
    """The banner and the meaning of the number change together."""
    client, _ = _client(hits=[_hit()], calibrated=True)
    assert "UNCALIBRATED" not in _upload(client).text


# --- the opt-out list ------------------------------------------------------


def test_the_exclusion_list_is_consulted_on_every_search():
    """A legal obligation, not a ranking preference. It must not be cached into
    irrelevance or skipped on a fast path."""
    excluded = [uuid.uuid4(), uuid.uuid4()]
    client, deps = _client(hits=[_hit()], exclusions=excluded)

    _upload(client)
    _upload(client)

    assert deps.repo.exclusion_calls == 2
    assert deps.vectors.last_exclude == excluded


def test_exclusions_are_pushed_into_the_vector_search_not_filtered_after():
    """Server-side must_not, so a suppressed face never consumes one of the k
    slots. Post-filtering would silently shrink the result set."""
    excluded = [uuid.uuid4()]
    client, deps = _client(hits=[_hit()], exclusions=excluded)
    _upload(client)
    assert deps.vectors.last_exclude == excluded, "exclusions never reached the vector store"


# --- the uploaded photo ----------------------------------------------------


def test_the_uploaded_photo_is_never_written_to_disk(tmp_path, monkeypatch):
    """The premise of the whole project is that the query stays on your machine.
    A query image accumulating in a temp directory would break that quietly."""
    monkeypatch.chdir(tmp_path)
    before = set(tmp_path.rglob("*"))

    client, _ = _client(hits=[_hit()], crop_root=tmp_path / "crops")
    assert _upload(client).status_code == 200

    assert set(tmp_path.rglob("*")) == before, "something persisted the upload"


def test_an_undecodable_upload_is_a_400_not_a_crash():
    client, _ = _client()
    r = client.post("/search", files={"photo": ("x.txt", b"not an image", "text/plain")})
    assert r.status_code == 400


def test_an_empty_upload_is_rejected():
    client, _ = _client()
    r = client.post("/search", files={"photo": ("x.png", b"", "image/png")})
    assert r.status_code == 400


def test_an_oversized_upload_is_rejected_before_decoding():
    """A loopback DoS bound. Decoding a 200 MB "image" first would defeat it."""
    from arc_search.serve.app import MAX_UPLOAD_BYTES

    client, deps = _client()
    huge = b"\x89PNG\r\n\x1a\n" + b"\x00" * MAX_UPLOAD_BYTES
    r = client.post("/search", files={"photo": ("big.png", huge, "image/png")})
    assert r.status_code == 413
    assert deps.extractor.calls == 0, "decoded before checking the size"


# --- when there is no face -------------------------------------------------


def test_a_rejected_face_reports_why():
    """'No faces found' with no reason is what makes a search engine feel
    broken. The gate that rejected it is the actionable part."""
    client, _ = _client(extractor=FakeExtractor(faces=[], rejects=FakeRejects(too_small=2)))
    body = _upload(client).text
    assert "No usable face" in body
    assert "too_small" in body


def test_a_rejected_face_does_not_hit_the_index():
    """No face means no query vector. Searching anyway would be meaningless."""
    vectors = FakeVectors()
    client, _ = _client(extractor=FakeExtractor(faces=[]), vectors=vectors)
    _upload(client)
    assert vectors.searches == 0


def test_the_json_shape_is_stable_when_nothing_is_detected():
    client, _ = _client(extractor=FakeExtractor(faces=[], rejects=FakeRejects(bad_pose=1)))
    data = _upload(client, format="json").json()
    assert data["detected"] is False
    assert data["results"] == []
    assert data["rejected"] == {"bad_pose": 1}


def test_the_highest_quality_face_is_used_as_the_query():
    """A group photo has several faces. Picking the first one found would make
    the result depend on detector ordering rather than on image quality."""
    best = FakeQueryFace(quality=0.9)
    client, deps = _client(
        extractor=FakeExtractor(faces=[FakeQueryFace(quality=0.2), best, FakeQueryFace(0.5)]),
        hits=[_hit()],
    )
    _upload(client)
    assert deps.vectors.searches == 1
    assert "quality 0.900" in _upload(client).text


# --- crops -----------------------------------------------------------------


def test_a_crop_path_cannot_escape_the_crop_directory(tmp_path):
    """The path comes from our own database, which is exactly the assumption
    that turns into a traversal the day something else writes that column."""
    root = tmp_path / "crops"
    root.mkdir()
    (tmp_path / "secret.txt").write_bytes(b"not a crop")

    client, _ = _client(crop_root=root, crop_rows={1: ("../secret.txt",)})
    assert client.get("/crop/1").status_code == 404


def test_a_missing_crop_file_is_a_404_not_a_500(tmp_path):
    root = tmp_path / "crops"
    root.mkdir()
    client, _ = _client(crop_root=root, crop_rows={1: ("ab/cd/gone.webp",)})
    assert client.get("/crop/1").status_code == 404


def test_a_real_crop_is_served_as_webp(tmp_path):
    root = tmp_path / "crops"
    (root / "ab" / "cd").mkdir(parents=True)
    (root / "ab" / "cd" / "x.webp").write_bytes(b"RIFF0000WEBPfake")

    client, _ = _client(crop_root=root, crop_rows={1: ("ab/cd/x.webp",)})
    r = client.get("/crop/1")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/webp"


def test_a_face_with_no_crop_is_a_404():
    client, _ = _client(crop_rows={1: (None,)})
    assert client.get("/crop/1").status_code == 404


# --- pages -----------------------------------------------------------------


def test_results_link_back_to_the_page_they_were_seen_on():
    """A hit that cannot be traced to a page is a number, not a search result."""
    client, _ = _client(hits=[_hit()])
    body = _upload(client).text
    assert "https://conf.test/speaker/ada/" in body
    assert "Ada Lovelace" in body


def test_an_unlabelled_result_says_so_rather_than_looking_broken():
    client, _ = _client(hits=[_hit(label=None)])
    assert "no weak label" in _upload(client).text


def test_labels_are_html_escaped():
    """Weak labels come from crawled alt text -- attacker-controlled by
    definition, since anyone can put a page on the open web."""
    client, _ = _client(hits=[_hit(label="<script>alert(1)</script>")])
    body = _upload(client).text
    assert "<script>alert(1)</script>" not in body
    assert "&lt;script&gt;" in body


def test_health_reports_the_calibration_state():
    client, _ = _client(hits=[_hit()])
    data = client.get("/health").json()
    assert data["ok"] is True
    assert data["calibrated"] is False
    assert data["faces"] == 1300


def test_the_landing_page_renders_without_a_query():
    client, _ = _client()
    r = client.get("/")
    assert r.status_code == 200
    assert "1,300 faces indexed" in r.text


# --- integration: the hydration query --------------------------------------


@pytest.mark.integration
@needs_pg
def test_hydrate_preserves_rank_order_and_drops_orphans(writer, png):
    """Rank order belongs to Qdrant, not to Postgres.

    A vector with no row is an orphan the write order exists to prevent, and it
    must never reach the page -- the UI could not attribute it to anything.
    """
    import uuid as _uuid
    from dataclasses import dataclass as _dc

    from arc_search.index.store import FaceRecord
    from arc_search.serve.repo import SearchRepo
    from pgfixtures import DSN, ctx, img

    writer.handle(img(png), ctx())
    image_id = writer._exec("SELECT id FROM image ORDER BY id DESC LIMIT 1").fetchone()[0]
    recs = [
        FaceRecord(
            qdrant_id=_uuid.uuid4(),
            bbox=(1.0, 2.0, 3.0, 4.0),
            landmarks=[0.0] * 10,
            src_width=220,
            src_height=180,
            det_score=0.9,
            blur_var=100.0,
            yaw=None,
            quality=0.5,
            age_est=30,
            crop_path=f"aa/bb/{i}.webp",
        )
        for i in range(2)
    ]
    ids = writer.record_faces(image_id, recs)
    writer.mark_examined(image_id, 2)

    @_dc
    class H:
        face_id: int
        score: float

    repo = SearchRepo(DSN)
    try:
        # Reverse order, plus an id that does not exist.
        hits = [H(ids[1], 0.9), H(999_999, 0.8), H(ids[0], 0.7)]
        out = repo.hydrate(hits)

        assert [h.face_id for h in out] == [ids[1], ids[0]], "rank order not preserved"
        assert all(h.face_id != 999_999 for h in out), "orphan vector reached the UI"
        assert out[0].label == "Ada"  # "Photo of Ada" with the prefix stripped
        assert out[0].pages == ["https://conf.test/speaker/ada/"]
        assert out[0].image_url == "https://conf.test/i/ada.png"
    finally:
        repo.close()


# --- whole-image search (ADR-005) ------------------------------------------


class FakeImageVectors:
    def __init__(self, hits=None):
        self.hits = hits if hits is not None else [(1, 0.42), (2, 0.31)]
        self.last_mode = None

    def search_named(self, vector_name, embedding, limit=50, exclude=()):
        self.last_mode = vector_name
        return self.hits[:limit]


class FakeImageEmbedder:
    def __init__(self, logit_scale=112.85, logit_bias=-16.77):
        self.texts = []
        self._scale = logit_scale
        self._bias = logit_bias

    def text_logit_params(self):
        return (self._scale, self._bias)

    def embed_text(self, texts):
        self.texts.extend(texts)
        return np.zeros((len(texts), 768), dtype=np.float32)

    def embed_images(self, images):
        class V:
            scene = np.zeros(768, dtype=np.float32)
            text = np.zeros(768, dtype=np.float32)

        return [V() for _ in images]


def _ihit(image_id=1, score=0.42, alt="Ada Lovelace", faces=1):
    from arc_search.serve.repo import ImageHit

    return ImageHit(
        image_id=image_id,
        score=score,
        url="https://conf.test/i/x.png",
        width=220,
        height=180,
        face_count=faces,
        alt=alt,
        pages=["https://conf.test/p/"],
    )


class ImageRepo(FakeRepo):
    def __init__(self, image_hits=None, **kw):
        super().__init__(**kw)
        self._image_hits = image_hits if image_hits is not None else [_ihit()]
        self.urls = {1: "https://conf.test/i/x.png"}
        self.collapsed = False

    def hydrate_images(self, hits):
        return self._image_hits

    def collapse_near_duplicates(self, hits, threshold=31):
        self.collapsed = True
        return hits

    def image_url(self, image_id):
        return self.urls.get(image_id)


def _iclient(*, image_hits=None, hits=None, fetch=None, thumbs=None, wired=True):
    from arc_search.serve.app import ThumbnailCache

    repo = ImageRepo(image_hits=image_hits)
    iv = FakeImageVectors(hits=hits)
    deps = Deps(
        repo=repo,
        vectors=FakeVectors(),
        extractor=FakeExtractor(),
        face_cfg=FaceSettings(),
        search_cfg=SearchSettings(),
        crop_root=Path("data/crops"),
        image_vectors=iv if wired else None,
        embedder=FakeImageEmbedder() if wired else None,
        thumbs=thumbs if thumbs is not None else ThumbnailCache(),
        fetch_image=fetch,
    )
    return TestClient(create_app(deps)), deps


def test_text_search_uses_the_text_vector_not_the_scene_one():
    """They are different spaces. Querying scene with a text embedding returns
    nonsense, and nothing in the response would say so."""
    client, deps = _iclient()
    client.get("/text", params={"q": "a fish", "format": "json"})
    assert deps.image_vectors.last_mode == "text"
    assert deps.embedder.texts == ["a fish"]


def test_similar_uses_the_scene_vector():
    client, deps = _iclient()
    client.post(
        "/similar",
        files={"photo": ("q.png", make_image(3, "PNG"), "image/png")},
        params={"format": "json"},
    )
    assert deps.image_vectors.last_mode == "scene"


def test_text_results_are_labelled_raw_cosine():
    """SigLIP is sigmoid-loss, so raw cosine is not even the model's own scale,
    let alone a probability."""
    client, _ = _iclient()
    data = client.get("/text", params={"q": "x", "format": "json"}).json()
    assert data["score_type"] == "raw_cosine"
    assert data["calibrated"] is False
    assert data["mode"] == "text"


def test_the_image_grid_renders_no_verdict():
    client, _ = _iclient()
    body = client.get("/text", params={"q": "a conference logo"}).text
    assert "UNCALIBRATED" in body
    for word in ("near certain", "strong match", "confident", "verified"):
        assert word not in body.lower()


def test_face_less_results_are_first_class():
    """The whole point of ADR-005. An image with no faces must render normally,
    not be filtered or badged as deficient."""
    client, _ = _iclient(image_hits=[_ihit(alt=None, faces=0)])
    body = client.get("/text", params={"q": "a logo"}).text
    assert "/thumb/1" in body
    assert "face(s)" not in body, "a face count must not be shown when there are none"


def test_the_grid_offers_more_like_this():
    client, _ = _iclient()
    assert "/similar/1" in client.get("/text", params={"q": "x"}).text


def test_whole_image_modes_degrade_instead_of_erroring():
    """If the collection or models are unavailable the UI must still serve face
    search. A UI that will not start is worse than one mode short."""
    client, _ = _iclient(wired=False)
    assert client.get("/").status_code == 200
    assert client.get("/text", params={"q": "x"}).status_code == 503
    assert "not configured" in client.get("/").text


def test_the_home_page_offers_all_three_modes():
    client, _ = _iclient()
    body = client.get("/").text
    assert "/text" in body and "/similar" in body and "/search" in body


# --- the thumbnail proxy ---------------------------------------------------


class FakeFetched:
    def __init__(self, body=b"\x89PNG-bytes", content_type="image/png"):
        self.body = body
        self.content_type = content_type


def test_thumbnails_are_fetched_by_the_server_not_the_browser():
    """🔴 The privacy property. Rendering <img src='https://theirsite/...'> would
    have the USER's browser hit the source host, revealing exactly which results
    they looked at. For a face search engine that inverts the whole premise."""
    calls = []

    async def fetch(url):
        calls.append(url)
        return FakeFetched()

    client, _ = _iclient(fetch=fetch)
    body = client.get("/text", params={"q": "x"}).text
    assert "https://conf.test/i/x.png" not in body.split("source page")[0], (
        "the grid must not point <img> at the source host"
    )

    r = client.get("/thumb/1")
    assert r.status_code == 200
    assert calls == ["https://conf.test/i/x.png"]


def test_a_cached_thumbnail_costs_no_fetch():
    """A politeness token per view would make browsing the index expensive."""
    calls = []

    async def fetch(url):
        calls.append(url)
        return FakeFetched()

    client, _ = _iclient(fetch=fetch)
    assert client.get("/thumb/1").headers["x-cache"] == "miss"
    assert client.get("/thumb/1").headers["x-cache"] == "hit"
    assert len(calls) == 1


def test_a_dead_source_image_is_a_404_not_a_500():
    """Link rot is ~15%/yr; a gone image must not break the page."""

    async def fetch(url):
        raise RuntimeError("410 gone")

    client, _ = _iclient(fetch=fetch)
    assert client.get("/thumb/1").status_code == 404


def test_an_unknown_image_id_is_a_404():
    client, _ = _iclient(fetch=None)
    assert client.get("/thumb/999").status_code == 404


def test_the_thumbnail_cache_is_bounded():
    """⚠️ ADR-001 forbids persisting scene images. An unbounded cache would
    become exactly the store non-negotiable #1 exists to prevent."""
    from arc_search.serve.app import ThumbnailCache

    c = ThumbnailCache(max_bytes=100)
    c.put(1, b"a" * 40, "image/png")
    c.put(2, b"b" * 40, "image/png")
    c.put(3, b"c" * 40, "image/png")  # evicts 1

    assert c.get(1) is None
    assert c.get(3) is not None
    assert c.stats["bytes"] <= 100


def test_an_oversized_image_does_not_evict_the_whole_cache():
    from arc_search.serve.app import ThumbnailCache

    c = ThumbnailCache(max_bytes=100)
    c.put(1, b"a" * 40, "image/png")
    c.put(2, b"x" * 500, "image/png")  # larger than the whole budget

    assert c.get(1) is not None, "a single huge image must not flush everything"
    assert c.get(2) is None


# --- near-duplicate collapse (PDQ) -----------------------------------------


def test_every_whole_image_mode_collapses_duplicates():
    """Measured need: 66% of labelled genuine pairs in this corpus are the same
    photo republished, and 223 of 225 have a different sha1. Without collapse the
    first page fills with the same sponsor logo."""
    client, deps = _iclient()
    client.get("/text", params={"q": "x"})
    assert deps.repo.collapsed, "text mode did not collapse"

    deps.repo.collapsed = False
    client.post(
        "/similar",
        files={"photo": ("q.png", make_image(3, "PNG"), "image/png")},
    )
    assert deps.repo.collapsed, "scene mode did not collapse"


def test_the_fold_count_is_shown_not_hidden():
    """'8 results' that were really 40 is the kind of quiet lie that makes an
    index untrustworthy."""
    from arc_search.serve.repo import ImageHit

    hit = ImageHit(
        image_id=1,
        score=0.4,
        url="https://c.test/x.png",
        width=10,
        height=10,
        face_count=0,
        alt=None,
        pages=[],
        duplicates=3,
    )
    client, _ = _iclient(image_hits=[hit])
    assert "+3 dup" in client.get("/text", params={"q": "x"}).text


@pytest.mark.integration
@needs_pg
def test_collapse_keeps_the_best_scoring_copy(writer, png):
    """Rank-preserving and greedy: the survivor is always the highest-scoring
    member of its group, and the rest are counted onto it."""
    from arc_search.serve.repo import ImageHit, SearchRepo
    from imagefixtures import make_image as mk
    from pgfixtures import DSN, ctx, img

    # Two images with IDENTICAL pixels re-encoded, plus one genuinely different.
    same = mk(11, "PNG")
    ids = []
    for i, body in enumerate([same, same, mk(12, "PNG")]):
        writer.handle(img(body, url=f"https://conf.test/d/{i}.png"), ctx())
        ids.append(writer._exec("SELECT id FROM image ORDER BY id DESC LIMIT 1").fetchone()[0])
    # The first two are byte-identical, so sha1 dedup collapses them to one row.
    # Force the interesting case: a distinct row whose PDQ matches an existing one.
    pdq = writer._exec("SELECT pdq FROM image WHERE id = %s", (ids[0],)).fetchone()[0]
    assert pdq is not None, "PDQ was not computed on insert"
    writer._exec(
        "INSERT INTO image (sha1, pdq, width, height, byte_size, domain_id, url_path) "
        "VALUES (%s, %s::bit(256), 10, 10, 99, %s, %s)",
        (b"\x11" * 20, pdq, writer.domain_id("conf.test"), "/d/clone.png"),
    )
    clone = writer._exec("SELECT id FROM image ORDER BY id DESC LIMIT 1").fetchone()[0]

    def h(image_id, score):
        return ImageHit(
            image_id=image_id,
            score=score,
            url="u",
            width=10,
            height=10,
            face_count=0,
            alt=None,
            pages=[],
        )

    repo = SearchRepo(DSN)
    try:
        out = repo.collapse_near_duplicates([h(ids[0], 0.9), h(clone, 0.8), h(ids[-1], 0.7)])
        assert [r.image_id for r in out] == [ids[0], ids[-1]], "clone was not folded"
        assert out[0].duplicates == 1
        assert out[1].duplicates == 0
    finally:
        repo.close()


@pytest.mark.integration
@needs_pg
def test_an_image_without_a_hash_is_kept_not_guessed(writer, png):
    """A missing PDQ is not evidence of duplication. Undecodable or low-quality
    images must still appear rather than being folded into something arbitrary."""
    from arc_search.serve.repo import ImageHit, SearchRepo
    from pgfixtures import DSN, ctx, img

    writer.handle(img(png), ctx())
    real = writer._exec("SELECT id FROM image ORDER BY id DESC LIMIT 1").fetchone()[0]
    writer._exec(
        "INSERT INTO image (sha1, width, height, byte_size, domain_id, url_path) "
        "VALUES (%s, 10, 10, 99, %s, %s)",
        (b"\x22" * 20, writer.domain_id("conf.test"), "/nohash.png"),
    )
    nohash = writer._exec("SELECT id FROM image ORDER BY id DESC LIMIT 1").fetchone()[0]

    def h(i, s):
        return ImageHit(
            image_id=i, score=s, url="u", width=10, height=10, face_count=0, alt=None, pages=[]
        )

    repo = SearchRepo(DSN)
    try:
        out = repo.collapse_near_duplicates([h(real, 0.9), h(nohash, 0.8)])
        assert [r.image_id for r in out] == [real, nohash]
    finally:
        repo.close()


# --- "no match" must not look like "here are your results" -----------------
#
# 🔴 Nearest-neighbour search ALWAYS returns k results. It has no concept of a
# non-match -- it returns the closest vectors however far away they are. So a
# corpus that simply does not contain the thing renders identically to a corpus
# that does, unless the UI says otherwise.
#
# Raw cosine cannot say it. Measured on the live index: "xyzzy plugh nonsense"
# scores a HIGHER cosine (0.110) than "ballerina" (0.087). Cosine is not
# comparable across queries. SigLIP's own sigmoid is.


def test_a_no_match_query_says_so():
    """The bug this fixes: searching 'ballerina' against a conference archive
    returned 24 confident-looking rows of speaker headshots."""
    client, _ = _iclient(image_hits=[_ihit(score=0.0865)])
    body = client.get("/text", params={"q": "ballerina"}).text
    assert "No match found" in body
    assert "does not consider any of these a match" in body


def test_a_real_match_does_not_get_the_warning():
    """A cosine of 0.153 is p=0.62 on SigLIP's scale -- a genuine match, and it
    must not be labelled a non-match."""
    client, _ = _iclient(image_hits=[_ihit(score=0.1530)])
    assert "No match found" not in client.get("/text", params={"q": "a beard"}).text


def test_the_model_probability_is_shown_not_just_cosine():
    """Cosine is what the model computes; p is what it means. Showing only the
    former hands the reader a number that is not comparable across queries."""
    client, _ = _iclient(image_hits=[_ihit(score=0.1530)])
    body = client.get("/text", params={"q": "a beard"}).text
    assert "p 0.6" in body
    assert "cos 0.153" in body, "raw cosine should stay visible, just secondary"


def test_json_reports_whether_anything_matched():
    """A client must be able to tell 'no results' from 'k nearest, none of them
    it' without re-deriving the model's calibration itself."""
    client, _ = _iclient(image_hits=[_ihit(score=0.0865)])
    data = client.get("/text", params={"q": "ballerina", "format": "json"}).json()
    assert data["matched"] is False
    assert data["best_match_probability"] < 0.01
    assert data["results"][0]["probability"] < 0.01

    client, _ = _iclient(image_hits=[_ihit(score=0.1530)])
    data = client.get("/text", params={"q": "a beard", "format": "json"}).json()
    assert data["matched"] is True


def test_scene_mode_shows_no_probability():
    """DINOv2 has no sigmoid calibration -- its cosine is a plain similarity.
    Inventing a probability for it would be exactly the assertion-over-
    measurement that non-negotiable #5 forbids."""
    client, _ = _iclient(image_hits=[_ihit(score=0.9)])
    body = client.post(
        "/similar", files={"photo": ("q.png", make_image(3, "PNG"), "image/png")}
    ).text
    assert "0.9000" in body
    assert "No match found" not in body
