"""PostgresWriter tests.

Two tiers. ``path_of`` and the dimension probe are pure and always run. The
writer itself needs a live database, so those are marked ``integration`` and
skipped unless ARC_TEST_PG_DSN points at one.

THESE TESTS TRUNCATE THE DATABASE THEY POINT AT.

That is not negotiable -- the writer seeds its dedup state from ``image`` at
construction, so a test asserting on counts has to start from a known-empty
table. What IS negotiable is which database gets truncated, and an earlier
version of this file cheerfully documented pointing it at ``arc_search``, the
production corpus. Running the suite during a crawl would have silently
destroyed hours of work. ``require_test_database`` now refuses anything whose
database name does not end in ``_test``.

Setup:

    docker compose up -d postgres
    docker exec arc_search-postgres-1 createdb -U arc arc_search_test
    docker exec -i arc_search-postgres-1 psql -U arc -d arc_search_test < sql/schema.sql
    ARC_TEST_PG_DSN=postgresql://arc@127.0.0.1:5432/arc_search_test pytest -m integration
"""

from __future__ import annotations

import os
from urllib.parse import urlsplit

import pytest

from arc_search.crawler.fetch import image_dimensions
from arc_search.index.store import path_of

DSN = os.environ.get("ARC_TEST_PG_DSN")
needs_pg = pytest.mark.skipif(not DSN, reason="set ARC_TEST_PG_DSN to run")

TEST_DB_SUFFIX = "_test"


def require_test_database(dsn: str) -> str:
    """Return the database name, or raise if it is not clearly a test database.

    A guard, not a convenience. The fixture TRUNCATEs, and the cost of pointing
    it one character wrong is the whole corpus.
    """
    name = urlsplit(dsn).path.lstrip("/")
    if not name:
        raise ValueError(f"ARC_TEST_PG_DSN names no database: {dsn!r}")
    if not name.endswith(TEST_DB_SUFFIX):
        raise ValueError(
            f"refusing to run: ARC_TEST_PG_DSN points at {name!r}, which does not end "
            f"in {TEST_DB_SUFFIX!r}. These tests TRUNCATE every table. Create a "
            f"throwaway database instead:\n"
            f"  docker exec arc_search-postgres-1 createdb -U arc {name}{TEST_DB_SUFFIX}\n"
            f"  docker exec -i arc_search-postgres-1 psql -U arc "
            f"-d {name}{TEST_DB_SUFFIX} < sql/schema.sql"
        )
    return name


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


def _wipe(conn) -> None:
    """CASCADE from image_source down; face and eval_pair follow image."""
    conn.execute(
        "TRUNCATE image_source, image, page, text_blob, url_path, domain RESTART IDENTITY CASCADE"
    )


@pytest.fixture
def writer():
    """A PostgresWriter over an empty database.

    Wipes BEFORE as well as after. Cleaning up only on teardown makes every
    test in the file depend on nothing else having touched the database first
    -- which broke the moment a real crawl was run against it by hand, since
    PostgresWriter seeds its dedup state from `image` at construction. A test
    that needs an empty database has to make one, not hope for one.
    """
    import psycopg

    from arc_search.index.store import PostgresWriter

    require_test_database(DSN)  # never truncate the production corpus

    scratch = psycopg.connect(DSN, autocommit=True)
    _wipe(scratch)
    scratch.close()

    w = PostgresWriter(DSN)
    yield w
    _wipe(w._conn)
    w.close()


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
    # -1, NOT 0. The crawl tier has no detector, so "never examined" is the
    # only honest value. 0 means "examined, found nothing" and is a tombstone.
    assert row[6] == -1


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
def test_resume_does_not_mark_unexamined_images_barren(writer):
    """The bug this suite caught on its first run against a real database.

    Deduper treats face_count == 0 as BARREN -- "known to contain no qualifying
    face, never look again". image.face_count defaulted to 0, and the crawl
    tier runs with no detector, so every row it wrote claimed to have been
    examined and found empty. On the next startup the whole corpus loaded as
    barren and week 2 would have skipped all of it: no faces indexed, no error
    anywhere, and a failure that reads like a bad model.

    The verdict must be EXACT_DUP (seen before, still needs a detector), never
    BARREN (seen before, already ruled out).
    """
    from arc_search.index.dedup import Verdict
    from arc_search.index.store import PostgresWriter

    body = _png(400, 400)
    writer.handle(_img(body), _ctx())
    assert writer._conn.execute("SELECT face_count FROM image").fetchone()[0] == -1

    fresh = PostgresWriter(DSN)
    try:
        result = fresh.dedup.check_bytes(body)
        assert result is not None
        assert result.verdict == Verdict.EXACT_DUP
        assert result.verdict != Verdict.BARREN
    finally:
        fresh.close()


@pytest.mark.integration
@needs_pg
def test_a_genuinely_barren_image_is_still_recorded_as_barren(writer):
    """The tri-state has to work in both directions, or the -1 fix just breaks
    the optimization it was protecting. 0 must still mean 'do not re-examine'."""
    from arc_search.index.dedup import Verdict
    from arc_search.index.store import PostgresWriter

    body = _png(400, 400)
    writer.handle(_img(body), _ctx())
    # Simulate week 2 having examined it and found no qualifying face.
    writer._conn.execute("UPDATE image SET face_count = 0")

    fresh = PostgresWriter(DSN)
    try:
        result = fresh.dedup.check_bytes(body)
        assert result is not None
        assert result.verdict == Verdict.BARREN
    finally:
        fresh.close()


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


# --- the guard itself (pure, always runs) ----------------------------------


@pytest.mark.parametrize(
    "dsn",
    [
        "postgresql://arc@127.0.0.1:5432/arc_search",  # the production corpus
        "postgresql://arc@127.0.0.1:5432/postgres",
        "postgresql://arc@127.0.0.1:5432/arc_search_prod",
        "postgresql://arc@127.0.0.1:5432/testing",  # 'test' prefix is not a suffix
        "postgresql://arc@127.0.0.1:5432/",  # no database at all
    ],
)
def test_guard_refuses_databases_that_are_not_clearly_disposable(dsn):
    """These tests TRUNCATE. Pointing them at the crawl corpus destroys it.

    This is not hypothetical: the fixture wiped the production database
    mid-session, during a live crawl, because the module docstring told you to
    set ARC_TEST_PG_DSN to arc_search.
    """
    with pytest.raises(ValueError):
        require_test_database(dsn)


@pytest.mark.parametrize(
    "dsn",
    [
        "postgresql://arc@127.0.0.1:5432/arc_search_test",
        "postgresql://arc:pw@db.internal:5432/anything_test",
    ],
)
def test_guard_allows_a_test_database(dsn):
    assert require_test_database(dsn).endswith("_test")


def test_guard_error_names_the_fix():
    """An error that only says 'no' costs someone twenty minutes."""
    with pytest.raises(ValueError, match="createdb"):
        require_test_database("postgresql://arc@127.0.0.1:5432/arc_search")


# --- connection resilience -------------------------------------------------


@pytest.mark.integration
@needs_pg
def test_a_dropped_connection_is_replaced_transparently(writer):
    """A five-hour crawl cannot die because the database blinked.

    An archive run died at startup with 'lost synchronization with server' --
    a corrupted wire stream, transient (the same query then ran 12/12 by hand),
    and fatal because nothing reconnected. Worse: a mid-crawl blip would have
    left every subsequent write failing while the crawler kept fetching pages
    and storing none of them, looking healthy the whole time.
    """
    writer.record_page("https://conf.test/a/", "A", 200)
    assert writer.reconnects == 0

    writer._conn.close()  # simulate the connection going away

    writer.record_page("https://conf.test/b/", "B", 200)
    assert writer.reconnects == 1
    assert writer._conn.execute("SELECT count(*) FROM page").fetchone()[0] == 2


@pytest.mark.integration
@needs_pg
def test_a_real_sql_error_is_raised_not_swallowed_by_a_reconnect(writer):
    """The retry must distinguish 'connection is gone' from 'query is wrong'.

    Reconnecting on a constraint violation would hide genuine bugs and retry
    them forever.
    """
    import psycopg

    with pytest.raises(psycopg.Error):
        writer._exec("SELECT * FROM a_table_that_does_not_exist")
    assert writer.reconnects == 0, "a bad query must not trigger a reconnect"
    # ...and the writer is still usable afterwards.
    writer.record_page("https://conf.test/c/", "C", 200)


@pytest.mark.integration
@needs_pg
def test_interning_caches_survive_a_reconnect(writer):
    """Cached ids are committed rows; they stay valid on any connection."""
    d1 = writer.domain_id("conf.test")
    writer._conn.close()

    # A cache HIT must not need the database at all, so it must not reconnect.
    assert writer.domain_id("conf.test") == d1
    assert writer.reconnects == 0

    # A cache MISS has to go to the database, and that is what triggers the
    # reconnect -- transparently, with the earlier cached id still correct.
    d2 = writer.domain_id("other.test")
    assert writer.reconnects == 1
    assert d2 != d1
    assert writer._exec("SELECT count(*) FROM domain").fetchone()[0] == 2
