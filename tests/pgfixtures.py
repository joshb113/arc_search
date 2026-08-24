"""Shared Postgres test helpers.

A plain module, not conftest.py, for the same reason ``imagefixtures.py`` is one:
a module imported *from* conftest only resolves when the repo root happens to be
on sys.path, which is true under ``python -m pytest`` and false under the bare
``pytest`` that CI runs. ``pythonpath = ["src", "tests"]`` in pyproject makes a
sibling module resolve identically under both.

The ``writer`` fixture itself lives in conftest.py, because pytest discovers
fixtures there without an import -- and importing a fixture into a test module
then shadowing it with a parameter of the same name is an F811 redefinition.

THESE HELPERS TRUNCATE THE DATABASE THEY POINT AT. ``require_test_database`` is
the guard, and it is on the NAME rather than on anyone remembering to be careful.
"""

from __future__ import annotations

import os
from urllib.parse import urlsplit

import pytest

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


def wipe(conn) -> None:
    """CASCADE from image_source down; face and eval_pair follow image."""
    conn.execute(
        "TRUNCATE image_source, image, page, text_blob, url_path, domain RESTART IDENTITY CASCADE"
    )


def ctx(page_url: str = "https://conf.test/speaker/ada/", alt: str = "Photo of Ada"):
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


def img(body: bytes, url: str = "https://conf.test/i/ada.png"):
    from arc_search.crawler.fetch import Fetched

    return Fetched(
        url=url,
        final_url=url,
        status=200,
        content_type="image/png",
        kind="image",
        body=body,
        width=400,
        height=400,
    )
