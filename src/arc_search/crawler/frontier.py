"""Persistent URL frontier.

eye_of_web kept its queue, visited set, and depth map as plain Python objects.
A crash lost the entire crawl. It wrote ``is_crawled`` rows to Postgres and never
read them back, so a restart re-crawled from zero. Its dedup key was the raw URL
string with only the fragment stripped, so ``/a``, ``/a/``, ``?b=1&c=2`` and
``?c=2&b=1`` were four separate crawl targets.

This module fixes both: SQLite-backed state that survives restart, and real URL
normalization before the dedup check.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

# Tracking parameters that never change page content. Stripping these collapses
# a surprising fraction of a real crawl.
_TRACKING_PARAMS = re.compile(
    r"^(utm_|fbclid$|gclid$|mc_[ce]id$|_ga$|ref$|ref_src$|igshid$|si$)", re.I
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS frontier (
    url_key   TEXT PRIMARY KEY,
    url       TEXT NOT NULL,
    host      TEXT NOT NULL,
    depth     INTEGER NOT NULL,
    state     INTEGER NOT NULL DEFAULT 0,   -- 0 pending, 1 in-flight, 2 done, 3 failed
    attempts  INTEGER NOT NULL DEFAULT 0,
    added_at  REAL NOT NULL,

    -- Opaque caller payload, carried with the queued URL. The frontier does not
    -- interpret it; the crawler stores an image's provenance here (the page it
    -- was found on, its title, and its alt text).
    --
    -- This column exists because that context used to live in an in-memory dict
    -- beside a durable queue. The queue survived restarts and the context did
    -- not, so every image already queued at restart was recorded with no page
    -- link and NO ALT TEXT -- silently discarding the weak labels the whole
    -- calibration plan is built on, while the crawl reported perfect health.
    meta      TEXT
);
CREATE INDEX IF NOT EXISTS frontier_pending ON frontier (state, depth, added_at);
CREATE INDEX IF NOT EXISTS frontier_host    ON frontier (host, state);
"""

PENDING, INFLIGHT, DONE, FAILED = 0, 1, 2, 3


def normalize(url: str) -> str:
    """Canonical form used for dedup. Not for fetching -- fetch the original."""
    parts = urlsplit(url.strip())
    scheme = (parts.scheme or "https").lower()
    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]

    port = ""
    if parts.port and not (
        (scheme == "https" and parts.port == 443) or (scheme == "http" and parts.port == 80)
    ):
        port = f":{parts.port}"

    path = parts.path or "/"
    # Collapse duplicate slashes, drop a single trailing slash (but keep root).
    path = re.sub(r"//+", "/", path)
    if len(path) > 1 and path.endswith("/"):
        path = path[:-1]

    kept = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not _TRACKING_PARAMS.match(k)
    ]
    query = urlencode(sorted(kept))

    return urlunsplit((scheme, host + port, path, query, ""))  # fragment dropped


def url_key(url: str) -> str:
    return hashlib.sha1(normalize(url).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Task:
    url: str
    host: str
    depth: int
    attempts: int
    meta: str | None = None


class Frontier:
    """SQLite frontier. Single-writer; fine to ~5M URLs.

    Promote to Redis (``ARC_CRAWL_FRONTIER_BACKEND=redis``) when one crawler
    process stops being enough -- the interface here is the contract.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.executescript(_SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        """Bring an existing frontier file up to the current schema.

        CREATE TABLE IF NOT EXISTS does nothing to a table that already exists,
        so a frontier written before `meta` would keep working right up until
        the first query naming that column.
        """
        cols = {r[1] for r in self._db.execute("PRAGMA table_info(frontier)")}
        if "meta" not in cols:
            self._db.execute("ALTER TABLE frontier ADD COLUMN meta TEXT")

    def add(self, url: str, depth: int, max_depth: int, meta: str | None = None) -> bool:
        """Enqueue. Returns False if already known or too deep.

        Note the ordering: over-depth URLs are still RECORDED (as DONE), so a
        rediscovery does not re-evaluate them. eye_of_web returned before adding
        to its visited set, so every rediscovery at the depth boundary redid the
        work and re-logged it, forever.
        """
        key = url_key(url)
        host = urlsplit(url).hostname or ""
        state = DONE if depth > max_depth else PENDING
        cur = self._db.execute(
            "INSERT OR IGNORE INTO frontier (url_key, url, host, depth, state, added_at, meta) "
            "VALUES (?, ?, ?, ?, ?, strftime('%s','now'), ?)",
            (key, url, host, depth, state, meta),
        )
        return cur.rowcount > 0 and state == PENDING

    def lease(self, limit: int = 1) -> list[Task]:
        """Claim pending URLs, marking them in-flight so a concurrent worker or a
        restart does not double-fetch."""
        rows = self._db.execute(
            "SELECT url_key, url, host, depth, attempts, meta FROM frontier "
            "WHERE state = ? ORDER BY depth, added_at LIMIT ?",
            (PENDING, limit),
        ).fetchall()
        if not rows:
            return []
        keys = [r[0] for r in rows]
        self._db.execute(
            f"UPDATE frontier SET state = {INFLIGHT} "
            f"WHERE url_key IN ({','.join('?' * len(keys))})",
            keys,
        )
        return [Task(url=r[1], host=r[2], depth=r[3], attempts=r[4], meta=r[5]) for r in rows]

    def complete(self, url: str) -> None:
        self._db.execute("UPDATE frontier SET state = ? WHERE url_key = ?", (DONE, url_key(url)))

    def fail(self, url: str, max_retries: int) -> bool:
        """Record a failure. Returns True if the URL was requeued for retry.

        A transient 503 must not permanently drop a page. eye_of_web logged and
        returned, leaving the URL in its visited set -- one blip and that page was
        gone for the life of the crawl.
        """
        key = url_key(url)
        row = self._db.execute("SELECT attempts FROM frontier WHERE url_key = ?", (key,)).fetchone()
        attempts = (row[0] if row else 0) + 1
        requeue = attempts <= max_retries
        self._db.execute(
            "UPDATE frontier SET attempts = ?, state = ? WHERE url_key = ?",
            (attempts, PENDING if requeue else FAILED, key),
        )
        return requeue

    def release(self, url: str) -> None:
        """Return a leased URL to PENDING without counting an attempt.

        Distinct from both ``complete`` and ``fail``: the URL was never judged
        on its merits, the run simply stopped being allowed to fetch it. It
        must stay claimable by the next run.

        This exists because the page-budget check used to call ``complete()``.
        A 20-page smoke run marked 1,347 URLs DONE, of which 1,327 had never
        been fetched -- so ``--max-pages`` silently consumed the frontier and
        "crawl some tonight, more tomorrow" did nothing on the second night.
        """
        self._db.execute("UPDATE frontier SET state = ? WHERE url_key = ?", (PENDING, url_key(url)))

    def recover_inflight(self) -> int:
        """Reset rows left in-flight by a crash. Call once at startup."""
        cur = self._db.execute("UPDATE frontier SET state = ? WHERE state = ?", (PENDING, INFLIGHT))
        return cur.rowcount

    def stats(self) -> dict[str, int]:
        rows = self._db.execute("SELECT state, COUNT(*) FROM frontier GROUP BY state").fetchall()
        names = {PENDING: "pending", INFLIGHT: "inflight", DONE: "done", FAILED: "failed"}
        out = dict.fromkeys(names.values(), 0)
        for state, n in rows:
            out[names.get(state, str(state))] = n
        return out

    def close(self) -> None:
        self._db.close()
