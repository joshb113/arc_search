"""Face-writer tests for PostgresWriter.

Split from test_store.py, which covers the crawl tier's half of the same class.
The shared database fixtures are imported rather than duplicated -- `writer`
already encodes the TRUNCATE guard and the wipe-on-both-sides rule, and a second
copy would drift.

THE WRITE ORDER THESE TESTS EXIST TO PIN
----------------------------------------

    record_faces()  ->  VectorStore.delete_image()/upsert()  ->  mark_examined()

``image.face_count`` is the commit marker for Postgres AND Qdrant together. It
must not flip until both stores are complete, because it is also the work queue:

    face_count > -1  =>  both stores are complete and agree for that image
    face_count == -1 =>  either store may hold partial state; it is disposable

Everything below is here to stop that ordering being "tidied up" later by
someone who reasonably assumes record_faces should set the count it just wrote.
"""

from __future__ import annotations

import dataclasses
import uuid

import pytest

from arc_search.index.store import FaceRecord
from imagefixtures import make_image
from pgfixtures import ctx as _ctx
from pgfixtures import img as _img
from pgfixtures import needs_pg

# The `writer` fixture is resolved from conftest.py by name -- see the note
# there on why it does not live in a test module.


def _face(seed: int = 0, *, crop_path: str | None = "ab/cd/abcd.webp") -> FaceRecord:
    return FaceRecord(
        qdrant_id=uuid.uuid4(),
        bbox=(10.0 + seed, 20.0 + seed, 110.0 + seed, 140.0 + seed),
        landmarks=[float(seed + i) for i in range(10)],
        src_width=800,
        src_height=600,
        det_score=0.91,
        blur_var=120.5,
        yaw=-12.5,
        quality=0.77,
        age_est=34,
        crop_path=crop_path,
    )


def _stored_image(writer, body: bytes) -> int:
    """Put one image in the corpus the way the crawl tier does; return its id."""
    writer.handle(_img(body), _ctx())
    return writer._exec("SELECT id FROM image ORDER BY id DESC LIMIT 1").fetchone()[0]


# --- the handoff between the two tiers -------------------------------------


@pytest.mark.integration
@needs_pg
def test_the_crawl_tier_leaves_every_image_on_the_work_queue(writer, png):
    """If the crawl tier ever writes 0 here instead of -1, the whole corpus
    becomes invisible to indexing and nothing reports an error. That is the
    failure the tri-state exists to prevent, and it already happened once."""
    _stored_image(writer, png)
    assert writer.unexamined_count() == 1
    (queued,) = writer.unexamined_images()
    assert queued.url == "https://conf.test/i/ada.png"
    assert (queued.width, queued.height) == (400, 400)


@pytest.mark.integration
@needs_pg
def test_the_work_queue_pages_by_keyset_not_offset(writer):
    """Rows leave the queue as they are processed, so an OFFSET walk over a
    shrinking result set skips work -- silently, which for a backfill is the
    worst available failure. Keyset on id stays correct under a live drain."""
    ids = []
    for seed in range(2, 7):
        body = make_image(seed, "PNG")
        writer.handle(_img(body, url=f"https://conf.test/i/{seed}.png"), _ctx())
        ids.append(writer._exec("SELECT id FROM image ORDER BY id DESC LIMIT 1").fetchone()[0])

    first = writer.unexamined_images(limit=2)
    assert [q.image_id for q in first] == ids[:2]

    # Drain page one, exactly as a backfill would, THEN ask for page two.
    for q in first:
        writer.mark_examined(q.image_id, 0)

    second = writer.unexamined_images(limit=2, after_id=first[-1].image_id)
    assert [q.image_id for q in second] == ids[2:4]


# --- geometry --------------------------------------------------------------


@pytest.mark.integration
@needs_pg
def test_record_faces_stores_geometry_at_original_resolution(writer, png):
    """Non-negotiable #3, and eye_of_web's exact mistake: it pre-multiplied
    landmarks by 0.5 to line them up with a lossy half-scale thumbnail. Once
    that is done the original coordinates are unrecoverable, because the scale
    factor is not stored anywhere. src_width/src_height record which resolution
    the numbers belong to; the crop is 128px regardless."""
    image_id = _stored_image(writer, png)
    rec = _face(1)
    (face_id,) = writer.record_faces(image_id, [rec])

    row = writer._exec(
        "SELECT bbox, landmarks, src_width, src_height, crop_path FROM face WHERE id = %s",
        (face_id,),
    ).fetchone()
    assert row[0] == pytest.approx(list(rec.bbox))
    assert row[1] == pytest.approx(list(rec.landmarks))
    # The SOURCE image, 800x600 -- not the 128px crop the pixels went to.
    assert (row[2], row[3]) == (800, 600)
    assert row[4] == "ab/cd/abcd.webp"


@pytest.mark.integration
@needs_pg
def test_landmarks_must_be_five_points(writer, png):
    """A 5-point kps array flattens to exactly 10 floats. Anything else means
    the model pack is not what config.py claims, and a short array would
    otherwise store happily and misalign every future re-crop."""
    image_id = _stored_image(writer, png)
    short = dataclasses.replace(_face(), landmarks=[1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="5 flattened"):
        writer.record_faces(image_id, [short])
    assert writer._exec("SELECT count(*) FROM face").fetchone()[0] == 0


# --- the commit contract ---------------------------------------------------


@pytest.mark.integration
@needs_pg
def test_record_faces_does_not_mark_the_image_examined(writer, png):
    """The load-bearing half of the crash contract. After record_faces the
    faces are in Postgres but the vectors are NOT in Qdrant, so the image must
    still look like work. If record_faces ever sets face_count itself, a crash
    between the two stores leaves faces no query can return, and nothing says
    so -- the corpus would simply be quietly short."""
    image_id = _stored_image(writer, png)
    writer.record_faces(image_id, [_face(1), _face(2)])

    count = writer._exec("SELECT face_count FROM image WHERE id = %s", (image_id,)).fetchone()[0]
    assert count == -1
    assert writer.unexamined_count() == 1, "image left the work queue before its vectors landed"


@pytest.mark.integration
@needs_pg
def test_mark_examined_commits_the_pair(writer, png):
    image_id = _stored_image(writer, png)
    writer.record_faces(image_id, [_face(1), _face(2)])
    writer.mark_examined(image_id, 2)

    count = writer._exec("SELECT face_count FROM image WHERE id = %s", (image_id,)).fetchone()[0]
    assert count == 2
    assert writer.unexamined_count() == 0
    assert writer.face_counts() == {
        "unexamined": 0,
        "provisional": 0,
        "barren": 0,
        "with_faces": 1,
        "faces_total": 2,
    }


@pytest.mark.integration
@needs_pg
def test_a_failed_insert_leaves_no_partial_face_set(writer, png):
    """record_faces is one transaction. Two good faces and a bad third must
    store none of them: a partial set that later got marked examined would be
    silent, permanent under-indexing of that image."""
    image_id = _stored_image(writer, png)
    batch = [_face(1), _face(2), dataclasses.replace(_face(3), landmarks=[0.0])]

    with pytest.raises(ValueError):
        writer.record_faces(image_id, batch)

    assert writer._exec("SELECT count(*) FROM face").fetchone()[0] == 0
    assert writer.unexamined_count() == 1


@pytest.mark.integration
@needs_pg
def test_reprocessing_replaces_faces_rather_than_accumulating(writer, png):
    """A re-run after a crash, or under a better model, may find a DIFFERENT
    number of faces. Delete-then-insert converges; an upsert would strand the
    surplus rows, whose vectors delete_image() has already dropped from Qdrant
    -- leaving rows that point at nothing."""
    image_id = _stored_image(writer, png)
    writer.record_faces(image_id, [_face(1), _face(2), _face(3)])
    assert writer._exec("SELECT count(*) FROM face").fetchone()[0] == 3

    survivor = _face(9)
    (new_id,) = writer.record_faces(image_id, [survivor])

    rows = writer._exec("SELECT id, qdrant_id FROM face").fetchall()
    assert len(rows) == 1
    assert rows[0][0] == new_id
    assert str(rows[0][1]) == str(survivor.qdrant_id)


# --- the tri-state ---------------------------------------------------------


@pytest.mark.integration
@needs_pg
def test_a_barren_image_is_examined_not_unexamined(writer, png):
    """0 and -1 are different verdicts and the difference is permanent: 0 means
    'looked, found nothing, never look again'. An image with no qualifying face
    still has to leave the work queue, or a backfill re-fetches it forever."""
    image_id = _stored_image(writer, png)
    writer.record_faces(image_id, [])
    writer.mark_examined(image_id, 0, calibrated=True)

    assert writer.unexamined_count() == 0
    assert writer.face_counts()["barren"] == 1


# --- the uncalibrated-empty escape hatch -----------------------------------
#
# 0 is a TOMBSTONE. Deduper reads it as never-look-again, permanently. But every
# gate that can produce an empty result -- min_face_px, min_det_score,
# min_blur_var, max_abs_yaw -- is still an uncalibrated placeholder, and
# min_face_px alone was measured discarding 40% of real detections at its old
# value of 64. Writing 0 before calibration makes today's unjustified numbers
# irreversible. -2 costs one re-fetch and keeps the option open.


@pytest.mark.integration
@needs_pg
def test_an_empty_result_is_provisional_until_calibration(writer, png):
    """The default path, and the one that matters: nothing is permanently
    retired on the strength of a threshold nobody has justified."""
    image_id = _stored_image(writer, png)
    writer.record_faces(image_id, [])
    writer.mark_examined(image_id, 0)  # calibrated defaults to False

    stored = writer._exec("SELECT face_count FROM image WHERE id = %s", (image_id,)).fetchone()[0]
    assert stored == -2, "an uncalibrated empty result must not be tombstoned"
    assert writer.face_counts()["barren"] == 0
    assert writer.face_counts()["provisional"] == 1


@pytest.mark.integration
@needs_pg
def test_provisional_images_leave_the_backfill_queue_but_join_the_recheck_queue(writer, png):
    """Both halves matter. Staying on the backfill queue would make every run
    re-fetch the whole provisional set forever; vanishing entirely would lose
    the images the recalibration pass exists to revisit."""
    image_id = _stored_image(writer, png)
    writer.mark_examined(image_id, 0)

    assert writer.unexamined_count() == 0, "would be re-fetched forever"
    assert writer.unexamined_images() == []
    assert writer.provisional_count() == 1

    (queued,) = writer.provisional_images()
    assert queued.image_id == image_id
    assert queued.url == "https://conf.test/i/ada.png"


@pytest.mark.integration
@needs_pg
def test_a_provisional_image_is_not_barren_to_the_deduper(writer, png):
    """The in-memory half. If PROVISIONAL_EMPTY reached `_barren`, check_bytes()
    would say skip-forever and defeat the entire point of the state."""
    from arc_search.index.dedup import Verdict, sha1_bytes

    image_id = _stored_image(writer, png)
    writer.mark_examined(image_id, 0)

    assert sha1_bytes(png) not in writer.dedup._barren
    result = writer.dedup.check_bytes(png)
    assert result is not None
    assert result.verdict is not Verdict.BARREN


@pytest.mark.integration
@needs_pg
def test_calibrated_true_writes_a_real_tombstone(writer, png):
    """Once the thresholds are derived, 0 becomes honest and the image is
    genuinely retired -- in Postgres and in the dedup state."""
    from arc_search.index.dedup import sha1_bytes

    image_id = _stored_image(writer, png)
    writer.mark_examined(image_id, 0, calibrated=True)

    stored = writer._exec("SELECT face_count FROM image WHERE id = %s", (image_id,)).fetchone()[0]
    assert stored == 0
    assert writer.provisional_count() == 0
    assert sha1_bytes(png) in writer.dedup._barren


@pytest.mark.integration
@needs_pg
def test_the_safe_default_is_the_cheap_mistake(writer, png):
    """Forgetting the flag must cost a re-fetch, never a permanent loss.

    `calibrated` defaults to False precisely so that the failure mode of
    omitting it is recoverable. This pins that default; flipping it would make
    the easy mistake the expensive one.
    """
    image_id = _stored_image(writer, png)
    writer.mark_examined(image_id, 0)
    assert writer.provisional_count() == 1
    assert writer.face_counts()["barren"] == 0


@pytest.mark.integration
@needs_pg
def test_a_positive_count_ignores_the_calibration_flag(writer, png):
    """Faces that WERE found are real wherever the gate sits. Only the empty
    result is contingent on an uncalibrated threshold."""
    image_id = _stored_image(writer, png)
    writer.record_faces(image_id, [_face(1)])
    writer.mark_examined(image_id, 1)  # uncalibrated

    stored = writer._exec("SELECT face_count FROM image WHERE id = %s", (image_id,)).fetchone()[0]
    assert stored == 1
    assert writer.provisional_count() == 0


@pytest.mark.integration
@needs_pg
def test_minus_two_is_chosen_by_the_writer_never_passed_in(writer, png):
    """-2 is an internal encoding, not part of the caller's vocabulary. Letting
    it be passed would reintroduce the ambiguity the state exists to remove."""
    image_id = _stored_image(writer, png)
    with pytest.raises(ValueError, match="never passed to it"):
        writer.mark_examined(image_id, -2)


@pytest.mark.integration
@needs_pg
def test_mark_examined_refuses_to_write_unexamined(writer, png):
    """-1 is the crawl tier's to write, once. Letting the index tier hand back
    -1 to mean 'I could not do this one' would make queue state and fetch
    failure indistinguishable, and the image would spin forever."""
    image_id = _stored_image(writer, png)
    with pytest.raises(ValueError, match="unexamined"):
        writer.mark_examined(image_id, -1)


@pytest.mark.integration
@needs_pg
def test_a_barren_verdict_reaches_the_in_memory_deduper(writer, png):
    """`Deduper._barren` is keyed by sha1 and is what makes check_bytes() say
    'skip forever'. It is only seeded at startup, so a process that crawls and
    indexes in one loop would otherwise keep re-examining images it had already
    rejected during that same run.

    Uses calibrated=True: only a real tombstone belongs in the barren set. The
    provisional case is pinned separately by
    test_a_provisional_image_is_not_barren_to_the_deduper.
    """
    from arc_search.index.dedup import Verdict, sha1_bytes

    image_id = _stored_image(writer, png)
    writer.mark_examined(image_id, 0, calibrated=True)

    result = writer.dedup.check_bytes(png)
    assert result is not None
    assert result.verdict is Verdict.BARREN
    assert sha1_bytes(png) in writer.dedup._barren


# --- referential integrity -------------------------------------------------


@pytest.mark.integration
@needs_pg
def test_deleting_an_image_takes_its_faces_with_it(writer, png):
    """ON DELETE CASCADE, pinned. A face row outliving its image would point at
    a missing provenance chain while its vector sat in Qdrant unreferenced."""
    image_id = _stored_image(writer, png)
    writer.record_faces(image_id, [_face(1)])
    writer._exec("DELETE FROM image WHERE id = %s", (image_id,))
    assert writer._exec("SELECT count(*) FROM face").fetchone()[0] == 0


@pytest.mark.integration
@needs_pg
def test_the_same_vector_cannot_be_claimed_by_two_faces(writer, png):
    """qdrant_id is UNIQUE. It is the join key between the two stores, so a
    duplicate would make a search hit ambiguous about which face produced it."""
    import psycopg

    image_id = _stored_image(writer, png)
    rec = _face(1)
    writer.record_faces(image_id, [rec])

    clash = dataclasses.replace(_face(2), qdrant_id=rec.qdrant_id)
    with pytest.raises(psycopg.Error):
        writer.record_faces(image_id, [rec, clash])
