"""Postgres writer for the crawl tier.

Replaces ``MetadataSink``. The difference that matters is not the storage
engine, it is that **the database is the source of truth for what has already
been crawled**. ``MetadataSink`` numbered images from zero and started with an
empty ``Deduper`` on every run, so resuming an interrupted crawl re-recorded
everything it already had, under colliding ids. That is fine for a 40-page
smoke test and wrong for an overnight one.

Here, ``sha1`` carries a UNIQUE constraint and ``Deduper`` is seeded from the
``image`` table at startup. Resume is correct by construction rather than by
remembering to do something.

WHAT IS AND IS NOT STORED
-------------------------
No pixels. ``image`` holds hashes, dimensions, and byte size; the bytes are
gone by the time this returns. Non-negotiable #1 and ADR-001.

INTERNING
---------
``domain``, ``url_path`` and ``text_blob`` are id->string tables, which is the
one piece of eye_of_web's design worth keeping -- it is what holds metadata to
~200 B/row at 30M images. Each is cached in-process, so a get-or-create costs a
dict lookup after the first sighting of a host, path, or string.

CONCURRENCY
-----------
``handle()`` is synchronous and called from an async worker, so every statement
here blocks the event loop. That is a deliberate trade: a localhost INSERT is
about a millisecond, the crawl is rate-limited to single-digit requests per
second by politeness, and per-image commits mean a kill -9 loses nothing.
Batching is the optimization if throughput ever becomes the constraint; it is
not the constraint at 1 rps.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

import psycopg
import structlog

from arc_search.index.dedup import Deduper, sha1_bytes

if TYPE_CHECKING:  # pragma: no cover
    from arc_search.crawler.fetch import Fetched
    from arc_search.crawler.run import ImageContext

log = structlog.get_logger(__name__)

# image.face_count is tri-state. This is the one the crawl tier writes: no
# detector has looked at this image yet. It must NOT be 0 -- Deduper treats 0
# as "examined and barren" and will skip the image forever. See the column
# comment in sql/schema.sql.
UNEXAMINED = -1


def path_of(url: str) -> str:
    """Everything after the host: path plus query. Never the fragment."""
    parts = urlsplit(url)
    return parts.path + (f"?{parts.query}" if parts.query else "") or "/"


def _hash(text: str) -> bytes:
    return hashlib.sha1(text.encode("utf-8")).digest()


class PostgresWriter:
    """Crawl sink backed by Postgres. Implements the ``CrawlSink`` protocol."""

    def __init__(self, dsn: str, deduper: Deduper | None = None) -> None:
        self._dsn = dsn
        self._conn = psycopg.connect(dsn, autocommit=True)
        self.reconnects = 0
        self.dedup = deduper if deduper is not None else Deduper()

        # Interning caches. Unbounded on purpose: a bounded vertical crawl sees
        # a few dozen hosts and a few hundred thousand paths, and the memory is
        # trivial next to re-querying. Revisit if this ever runs unbounded.
        self._domains: dict[str, int] = {}
        self._paths: dict[bytes, int] = {}
        self._texts: dict[bytes, int] = {}
        self._pages: dict[str, int] = {}

        self.loaded = self._load_dedup()

    # -- connection resilience ---------------------------------------------

    def _healthy(self) -> bool:
        """Is the connection still usable? Distinguishes the two failure kinds.

        A constraint violation leaves the connection perfectly fine and must be
        re-raised. A dropped or desynchronised connection must be replaced. The
        cheapest reliable test is to ask it to do something trivial.
        """
        if self._conn.closed:
            return False
        try:
            self._conn.execute("SELECT 1")
        except psycopg.Error:
            return False
        return True

    def _exec(self, sql: str, params: tuple | None = None, *, retry: bool = True):
        """Execute, replacing the connection once if it has gone bad.

        A five-hour unattended crawl cannot die because the database blinked.
        This is not hypothetical: an archive run died at startup with

            psycopg.DatabaseError: insufficient data in "D" message
            lost synchronization with server: got message type "f"

        -- a corrupted wire stream, transient (the same query then succeeded
        12/12 by hand), and fatal because nothing reconnected. Worse than
        fatal, actually: without this, a mid-crawl blip would leave every
        subsequent write failing while the crawler carried on looking busy,
        fetching pages and storing none of them.
        """
        try:
            return self._conn.execute(sql, params)
        except psycopg.Error:
            if not retry or self._healthy():
                raise  # a real error -- constraint violation, bad SQL
            log.warning("store.reconnecting", dsn_db=self._dsn.rsplit("/", 1)[-1])
            self._conn = psycopg.connect(self._dsn, autocommit=True)
            self.reconnects += 1
            # Interning caches survive deliberately: the ids they hold are
            # committed rows, still valid on any connection.
            return self._exec(sql, params, retry=False)

    # -- resume ------------------------------------------------------------

    def _load_dedup(self) -> int:
        """Seed the in-memory dedup state from what is already in the table.

        This is the whole reason to be on Postgres before a long run. Without
        it, a resumed crawl re-downloads and re-records every image it already
        has.

        Scale note: this is fine to low millions. At 10M the sha1 dict is on
        the order of a gigabyte and the BK-tree rebuild is slower still, at
        which point dedup wants to become a query rather than a preload. That
        is a week-5 problem, not a week-1 one.
        """
        rows = self._exec("SELECT id, sha1, pdq, face_count FROM image ORDER BY id").fetchall()
        if rows:
            self.dedup.load([(r[0], bytes(r[1]), None, r[3]) for r in rows])
        log.info("store.resumed", images=len(rows))
        return len(rows)

    # -- interning ---------------------------------------------------------

    def _intern(self, cache: dict, key, table: str, cols: str, values: tuple, conflict: str) -> int:
        """Get-or-create against an id->value table.

        SELECT first, then INSERT .. ON CONFLICT DO NOTHING, then SELECT again.
        The obvious one-liner is ``ON CONFLICT DO UPDATE .. RETURNING id``, but
        that writes a dead tuple and burns a BIGSERIAL value on every duplicate
        -- and duplicates are the overwhelmingly common case here. The extra
        round trip happens once per distinct value, then the cache absorbs it.
        """
        if (hit := cache.get(key)) is not None:
            return hit

        sel = f"SELECT id FROM {table} WHERE {conflict} = %s"
        row = self._exec(sel, (key,)).fetchone()
        if row is None:
            self._exec(
                f"INSERT INTO {table} ({cols}) VALUES ({', '.join(['%s'] * len(values))}) "
                f"ON CONFLICT DO NOTHING",
                values,
            )
            row = self._exec(sel, (key,)).fetchone()
            if row is None:  # pragma: no cover - only on a concurrent DELETE
                raise RuntimeError(f"{table}: row vanished between insert and select")

        cache[key] = row[0]
        return row[0]

    def domain_id(self, host: str) -> int:
        host = host.lower()
        return self._intern(self._domains, host, "domain", "host", (host,), "host")

    def path_id(self, path: str) -> int:
        h = _hash(path)
        return self._intern(self._paths, h, "url_path", "path, path_hash", (path, h), "path_hash")

    def text_id(self, body: str | None) -> int | None:
        """Intern a title or alt text. Empty and None both mean 'no text'."""
        if not body or not body.strip():
            return None
        body = body.strip()
        h = _hash(body)
        return self._intern(self._texts, h, "text_blob", "body, body_hash", (body, h), "body_hash")

    # -- pages -------------------------------------------------------------

    def record_page(self, url: str, title: str | None, status: int) -> int:
        """Upsert a crawled page and return its id.

        Called for every fetched page, not only pages that yielded images --
        otherwise ``page`` under-reports the crawl and the recrawl reaper has
        nothing to sweep.
        """
        if (hit := self._pages.get(url)) is not None:
            # Already seen this run. Refresh last_seen so the TTL reaper has an
            # accurate picture, but skip the interning round trips.
            self._exec("UPDATE page SET last_seen = now() WHERE id = %s", (hit,))
            return hit

        host = (urlsplit(url).hostname or "").lower()
        did, pid = self.domain_id(host), self.path_id(path_of(url))
        tid = self.text_id(title)

        row = self._exec(
            """
            INSERT INTO page (domain_id, url_path_id, title_id, http_status)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (domain_id, url_path_id) DO UPDATE
                SET last_seen = now(),
                    http_status = EXCLUDED.http_status,
                    title_id = COALESCE(EXCLUDED.title_id, page.title_id)
            RETURNING id
            """,
            (did, pid, tid, status),
        ).fetchone()
        assert row is not None  # DO UPDATE always returns a row
        self._pages[url] = row[0]
        return row[0]

    # -- images ------------------------------------------------------------

    def handle(self, fetched: Fetched, context: ImageContext) -> str:
        """Record one downloaded image. Returns a verdict for the stats counter.

        The bytes are not retained. They are hashed, measured, and dropped.
        """
        if (hit := self.dedup.check_bytes(fetched.body)) is not None:
            # Already known. Still link it to this page -- the same photo on a
            # second page is exactly the provenance edge the engine exists to
            # answer questions about, and it costs one narrow row.
            if hit.matched_image_id is not None:
                self._link(hit.matched_image_id, context)
            return str(hit.verdict)

        digest = sha1_bytes(fetched.body)
        # face_count is written EXPLICITLY as -1 rather than left to the column
        # default. The convention only works if both halves agree, and relying
        # on a default to carry a semantic this important is how they came to
        # disagree in the first place: -1 was passed to Deduper.register() but
        # never to the INSERT, so a resumed crawl read every row back as barren.
        row = self._exec(
            """
            INSERT INTO image (sha1, width, height, byte_size, face_count)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (sha1) DO NOTHING
            RETURNING id
            """,
            (digest, fetched.width, fetched.height, len(fetched.body), UNEXAMINED),
        ).fetchone()

        if row is None:
            # Lost a race, or the row predates this process's dedup snapshot.
            # The UNIQUE constraint is the authority, not our in-memory set.
            row = self._exec("SELECT id FROM image WHERE sha1 = %s", (digest,)).fetchone()
            assert row is not None
            image_id, verdict = row[0], "exact_dup"
        else:
            image_id, verdict = row[0], "new"

        self.dedup.register(digest, None, image_id, face_count=UNEXAMINED)
        self._link(image_id, context)
        return verdict

    def _link(self, image_id: int, context: ImageContext) -> None:
        """Record that this image appeared on this page, with its alt text."""
        if not context.page_url:
            return  # resumed image with no surviving context
        page_id = self._pages.get(context.page_url)
        if page_id is None:
            page_id = self.record_page(context.page_url, context.page_title, 200)
        self._exec(
            """
            INSERT INTO image_source (image_id, page_id, alt_text_id)
            VALUES (%s, %s, %s)
            ON CONFLICT (image_id, page_id) DO NOTHING
            """,
            (image_id, page_id, self.text_id(context.alt)),
        )

    # -- reporting ---------------------------------------------------------

    def counts(self) -> dict[str, int]:
        out = {}
        for table in ("domain", "page", "image", "image_source", "text_blob"):
            row = self._exec(f"SELECT count(*) FROM {table}").fetchone()
            out[table] = row[0] if row else 0
        return out

    def close(self) -> None:
        self._conn.close()
