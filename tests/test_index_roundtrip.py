"""End-to-end: crawl row -> faces -> vectors -> query -> back to provenance.

This is the test that says Phase 1 works. Everything else checks one store in
isolation; this one runs the real write order against a live Postgres AND a live
Qdrant, then does the exact join the query UI will do:

    query vector -> Qdrant hit -> face_id -> face -> image -> image_source
                 -> page URL and the alt text the weak label lives in

A hit the UI cannot trace back to a page is useless -- it is a face with no
answer to "where did you see this". eye_of_web could not answer that question
either, which is why its results were unauditable.

Needs both services:

    docker compose up -d postgres qdrant
    ARC_TEST_PG_DSN=postgresql://arc@127.0.0.1:5432/arc_search_test pytest -m integration
"""

from __future__ import annotations

import dataclasses
import uuid

import numpy as np
import pytest

from arc_search.config import IndexSettings
from arc_search.index.store import FaceRecord
from arc_search.index.vectors import VectorRecord, VectorStore
from imagefixtures import make_image
from pgfixtures import ctx as _ctx
from pgfixtures import img as _img
from pgfixtures import needs_pg
from test_vectors import needs_qdrant, require_test_collection

QDRANT_URL = "http://127.0.0.1:6333"


def _unit(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(512).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def vectors():
    from qdrant_client import QdrantClient

    name = require_test_collection("faces_test")
    client = QdrantClient(url=QDRANT_URL)
    if client.collection_exists(name):
        client.delete_collection(name)

    vs = VectorStore(IndexSettings(qdrant_url=QDRANT_URL, collection=name), client=client)
    vs.ensure_collection()
    yield vs

    if client.collection_exists(name):
        client.delete_collection(name)
    client.close()


def _index_one_image(writer, vectors, body: bytes, embeddings: list[np.ndarray], page_url: str):
    """The real write order, exactly as the backfill will run it.

        record_faces() -> delete_image()/upsert() -> mark_examined()

    Returns (image_id, face_ids).
    """
    writer.handle(_img(body, url=f"{page_url}photo.png"), _ctx(page_url=page_url))
    image_id = writer._exec("SELECT id FROM image ORDER BY id DESC LIMIT 1").fetchone()[0]

    records = [
        FaceRecord(
            qdrant_id=uuid.uuid4(),
            bbox=(10.0, 20.0, 110.0, 140.0),
            landmarks=[float(i) for i in range(10)],
            src_width=400,
            src_height=400,
            det_score=0.9,
            blur_var=100.0,
            yaw=0.0,
            quality=0.8,
            age_est=30,
            crop_path=f"aa/bb/{i}.webp",
        )
        for i in range(len(embeddings))
    ]

    # 1. Postgres face rows. face_count deliberately still -1.
    face_ids = writer.record_faces(image_id, records)

    # 2. Qdrant, cleared first so a reprocess cannot orphan.
    vectors.delete_image(image_id)
    vectors.upsert(
        [
            VectorRecord(
                qdrant_id=rec.qdrant_id,
                embedding=emb,
                image_id=image_id,
                face_id=fid,
                quality=rec.quality,
            )
            for rec, emb, fid in zip(records, embeddings, face_ids, strict=True)
        ]
    )

    # 3. Commit the pair.
    writer.mark_examined(image_id, len(records))
    return image_id, face_ids


@pytest.mark.integration
@needs_pg
@needs_qdrant
def test_a_hit_traces_all_the_way_back_to_the_page_it_came_from(writer, vectors):
    """The query the UI is built on, end to end across both stores."""
    target = _unit(1)
    image_id, face_ids = _index_one_image(
        writer,
        vectors,
        make_image(3, "PNG"),
        [target, _unit(2)],
        "https://conf.test/speaker/ada/",
    )

    hits = vectors.search(target, limit=5)
    assert hits, "nothing came back"
    top = hits[0]
    assert top.face_id == face_ids[0]
    assert top.image_id == image_id
    assert top.score == pytest.approx(1.0, abs=1e-3)

    # The join the result page does: vector -> face -> image -> page + alt text.
    row = writer._exec(
        """
        SELECT d.host, u.path, t.body, f.crop_path, f.src_width, f.src_height
        FROM face f
        JOIN image i        ON i.id = f.image_id
        JOIN image_source s ON s.image_id = i.id
        JOIN page p         ON p.id = s.page_id
        JOIN domain d       ON d.id = p.domain_id
        JOIN url_path u     ON u.id = p.url_path_id
        LEFT JOIN text_blob t ON t.id = s.alt_text_id
        WHERE f.id = %s
        """,
        (top.face_id,),
    ).fetchone()
    assert row is not None, "hit could not be traced back to a page"
    assert row[0] == "conf.test"
    assert row[1] == "/speaker/ada/"
    assert row[2] == "Photo of Ada"  # the weak label plan-003 calibrates against
    assert row[3] == "aa/bb/0.webp"
    assert (row[4], row[5]) == (400, 400)


@pytest.mark.integration
@needs_pg
@needs_qdrant
def test_every_vector_in_qdrant_has_a_postgres_row(writer, vectors):
    """The invariant that makes results explainable.

    An orphaned vector still scores in a query, but the UI has no page, no alt
    text and no crop for it. Rather than trusting the write order in the
    abstract, index two images, reprocess one, and check the two stores agree
    afterwards -- which is the state a crashed and resumed backfill produces.
    """

    id_a, _ = _index_one_image(
        writer,
        vectors,
        make_image(4, "PNG"),
        [_unit(10), _unit(11)],
        "https://conf.test/a/",
    )
    id_b, _ = _index_one_image(
        writer,
        vectors,
        make_image(5, "PNG"),
        [_unit(12)],
        "https://conf.test/b/",
    )
    assert vectors.count() == 3

    # Reprocess A, this time finding one face instead of two.
    rec = FaceRecord(
        qdrant_id=uuid.uuid4(),
        bbox=(1.0, 2.0, 3.0, 4.0),
        landmarks=[0.0] * 10,
        src_width=400,
        src_height=400,
        det_score=0.9,
        blur_var=100.0,
        yaw=None,
        quality=0.5,
        age_est=None,
        crop_path="cc/dd/0.webp",
    )
    (new_face_id,) = writer.record_faces(id_a, [rec])
    vectors.delete_image(id_a)
    vectors.upsert([VectorRecord(rec.qdrant_id, _unit(13), id_a, new_face_id, rec.quality)])
    writer.mark_examined(id_a, 1)

    # Both stores now hold exactly two faces, and they are the SAME two.
    assert vectors.count() == 2
    pg_face_ids = {r[0] for r in writer._exec("SELECT id FROM face ORDER BY id").fetchall()}
    assert len(pg_face_ids) == 2

    # Sweep every vector and confirm it joins. A vector whose face_id is gone is
    # precisely the orphan this ordering exists to prevent.
    for point in vectors.client.scroll(vectors.name, limit=100, with_payload=True)[0]:
        fid = point.payload["face_id"]
        assert fid in pg_face_ids, f"orphaned vector: face_id {fid} has no Postgres row"

    assert writer.face_counts() == {
        "unexamined": 0,
        "provisional": 0,
        "barren": 0,
        "with_faces": 2,
        "faces_total": 2,
    }
    assert id_b in {
        r[0] for r in writer._exec("SELECT id FROM image WHERE face_count > 0").fetchall()
    }


@pytest.mark.integration
@needs_pg
@needs_qdrant
def test_a_crash_between_the_stores_leaves_the_image_reprocessable(writer, vectors):
    """Simulates the crash the write order is designed around.

    Face rows land in Postgres, then the process dies before Qdrant. The image
    must still be on the work queue, and re-running must converge rather than
    duplicate -- if face_count had been set in step 1, those faces would be
    permanently unsearchable and nothing would ever report it.
    """

    body = make_image(6, "PNG")
    writer.handle(_img(body), _ctx())
    image_id = writer._exec("SELECT id FROM image ORDER BY id DESC LIMIT 1").fetchone()[0]

    rec = FaceRecord(
        qdrant_id=uuid.uuid4(),
        bbox=(1.0, 2.0, 3.0, 4.0),
        landmarks=[0.0] * 10,
        src_width=400,
        src_height=400,
        det_score=0.9,
        blur_var=100.0,
        yaw=None,
        quality=0.5,
        age_est=None,
        crop_path=None,
    )
    writer.record_faces(image_id, [rec])
    # <-- crash here: nothing written to Qdrant, face_count untouched.

    assert vectors.count() == 0
    assert writer.unexamined_count() == 1, "a crashed image must stay on the work queue"

    # Resume: the queue still offers it, and the second pass converges.
    (queued,) = writer.unexamined_images()
    assert queued.image_id == image_id

    retry = dataclasses.replace(rec, qdrant_id=uuid.uuid4())
    (face_id,) = writer.record_faces(image_id, [retry])
    vectors.delete_image(image_id)
    vectors.upsert([VectorRecord(retry.qdrant_id, _unit(20), image_id, face_id, 0.5)])
    writer.mark_examined(image_id, 1)

    assert writer.unexamined_count() == 0
    assert vectors.count() == 1
    assert writer._exec("SELECT count(*) FROM face").fetchone()[0] == 1
