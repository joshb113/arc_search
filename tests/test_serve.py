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
