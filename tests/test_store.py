"""PostgresWriter tests.

Two tiers. ``path_of`` and the dimension probe are pure and always run. The
writer itself needs a live database, so those are marked ``integration`` and
skipped unless ARC_TEST_PG_DSN points at one:

    docker compose up -d postgres
    ARC_TEST_PG_DSN=postgresql://arc@127.0.0.1:5432/arc_search pytest -m integration

The integration tests each work in a savepoint-free scratch schema and clean up
after themselves, so they can run against a database that already has data.
"""

from __future__ import annotations

import os

import pytest

from arc_search.crawler.fetch import image_dimensions
from arc_search.index.store import path_of

DSN = os.environ.get("ARC_TEST_PG_DSN")
needs_pg = pytest.mark.skipif(not DSN, reason="set ARC_TEST_PG_DSN to run")


# --- pure ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://h.test/a/b", "/a/b"),
        ("https://h.test/a/b?x=1", "/a/b?x=1"),
        ("https://h.test/", "/"),
        # The fragment is never part of identity -- frontier.normalize drops it
        # too, and storing it would split one page into many rows.
        ("https://h.test/a#frag", "/a"),
    ],
)
def test_path_of(url, expected):
    assert path_of(url) == expected


def _png(w: int, h: int) -> bytes:
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (w, h), (128, 0, 0)).save(buf, "PNG")
    return buf.getvalue()


def test_image_dimensions_reads_the_header():
    assert image_dimensions(_png(321, 123)) == (321, 123)


def test_image_dimensions_returns_none_on_garbage():
    """A corrupt header is a skip, not a crawl failure."""
    assert image_dimensions(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100) is None
    assert image_dimensions(b"") is None
    assert image_dimensions(b"<!DOCTYPE html>") is None


def test_image_dimensions_survives_a_truncated_jpeg():
    """Half a download must not raise out of the fetch path."""
    full = _png(400, 400)
    assert image_dimensions(full[: len(full) // 3]) in (None, (400, 400))


# --- integration -----------------------------------------------------------


@pytest.fixture
def writer():
    import psycopg

    from arc_search.index.store import PostgresWriter

    w = PostgresWriter(DSN)
    yield w
    # Order matters: image_source references both sides.
    w._conn.execute("DELETE FROM image_source")
    w._conn.execute("DELETE FROM image")
    w._conn.execute("DELETE FROM page")
    w._conn.execute("DELETE FROM text_blob")
    w._conn.execute("DELETE FROM url_path")
    w._conn.execute("DELETE FROM domain")
    w.close()
    assert psycopg  # imported for the skip check


def _ctx(page_url="https://conf.test/speaker/ada/", alt="Photo of Ada"):
    from arc_search.crawler.run import ImageContext

    return ImageContext(
        page_url=page_url,
        page_title="Ada",
        vertical="conf",
        depth=2,
        alt=alt,
        extractor="img-src",
        width_hint=None,
    )


def _img(body: bytes):
    from arc_search.crawler.fetch import Fetched

    return Fetched(
        url="https://conf.test/i/ada.png",
        final_url="https://conf.test/i/ada.png",
        status=200,
        content_type="image/png",
        kind="image",
        body=body,
        width=400,
        height=400,
    )


@pytest.mark.integration
@needs_pg
def test_interning_is_idempotent(writer):
    a = writer.domain_id("conf.test")
    b = writer.domain_id("CONF.TEST")  # case is normalized
    assert a == b
    assert writer._conn.execute("SELECT count(*) FROM domain").fetchone()[0] == 1


@pytest.mark.integration
@needs_pg
def test_empty_alt_interns_to_null_not_an_empty_row(writer):
    assert writer.text_id("") is None
    assert writer.text_id("   ") is None
    assert writer.text_id(None) is None
    assert writer._conn.execute("SELECT count(*) FROM text_blob").fetchone()[0] == 0


@pytest.mark.integration
@needs_pg
def test_records_an_image_with_provenance(writer):
    writer.record_page("https://conf.test/speaker/ada/", "Ada", 200)
    assert writer.handle(_img(_png(400, 400)), _ctx()) == "new"

    row = writer._conn.execute(
        """
        SELECT d.host, u.path, t.body, i.width, i.height, i.byte_size, i.face_count
        FROM image_source s
        JOIN image i ON i.id = s.image_id
        JOIN page p ON p.id = s.page_id
        JOIN domain d ON d.id = p.domain_id
        JOIN url_path u ON u.id = p.url_path_id
        JOIN text_blob t ON t.id = s.alt_text_id
        """
    ).fetchone()
    assert row[0] == "conf.test"
    assert row[1] == "/speaker/ada/"
    assert row[2] == "Photo of Ada"  # the weak label survives the round trip
    assert (row[3], row[4]) == (400, 400)
    assert row[6] == 0  # schema default; -1 lives only in the in-memory deduper


@pytest.mark.integration
@needs_pg
def test_same_bytes_twice_is_one_image_row(writer):
    body = _png(400, 400)
    assert writer.handle(_img(body), _ctx()) == "new"
    assert writer.handle(_img(body), _ctx()) == "exact_dup"
    assert writer._conn.execute("SELECT count(*) FROM image").fetchone()[0] == 1


@pytest.mark.integration
@needs_pg
def test_same_image_on_two_pages_keeps_both_provenance_edges(writer):
    """This is the table that answers 'where else does this face appear'."""
    body = _png(400, 400)
    writer.handle(_img(body), _ctx(page_url="https://conf.test/a/"))
    writer.handle(_img(body), _ctx(page_url="https://conf.test/b/"))
    assert writer._conn.execute("SELECT count(*) FROM image").fetchone()[0] == 1
    assert writer._conn.execute("SELECT count(*) FROM image_source").fetchone()[0] == 2


@pytest.mark.integration
@needs_pg
def test_resume_seeds_dedup_from_the_table(writer):
    """The whole reason for going to Postgres before a five-hour run.

    MetadataSink started from an empty Deduper every time, so a resumed crawl
    re-recorded everything it already had.
    """
    from arc_search.index.store import PostgresWriter

    body = _png(400, 400)
    assert writer.handle(_img(body), _ctx()) == "new"

    second = PostgresWriter(DSN)  # a fresh process, same database
    try:
        assert second.loaded == 1, "dedup was not seeded from the image table"
        assert second.handle(_img(body), _ctx()) == "exact_dup"
        assert second._conn.execute("SELECT count(*) FROM image").fetchone()[0] == 1
    finally:
        second.close()


@pytest.mark.integration
@needs_pg
def test_recording_a_page_twice_updates_rather_than_duplicates(writer):
    a = writer.record_page("https://conf.test/x/", "First", 200)
    writer._pages.clear()  # force the DB path rather than the cache
    b = writer.record_page("https://conf.test/x/", "First", 200)
    assert a == b
    assert writer._conn.execute("SELECT count(*) FROM page").fetchone()[0] == 1


@pytest.mark.integration
@needs_pg
def test_a_page_with_no_images_is_still_recorded(writer):
    """Otherwise `page` under-reports the crawl and the reaper has nothing to
    sweep."""
    writer.record_page("https://conf.test/empty/", "Nothing here", 200)
    assert writer._conn.execute("SELECT count(*) FROM page").fetchone()[0] == 1
    assert writer._conn.execute("SELECT count(*) FROM image").fetchone()[0] == 0
