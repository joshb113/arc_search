"""Shared pytest fixtures.

The image helpers live in ``imagefixtures.py`` rather than here, because a
module imported *from* conftest is only resolvable when the repo root is on
sys.path -- true under ``python -m pytest``, false under a bare ``pytest``.
See that module's docstring.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from imagefixtures import make_image
from pgfixtures import DSN, require_test_database, wipe

# HuggingFace caches models under ~/.cache/huggingface by default, which on this
# machine is a C: drive sitting at ~97% full -- and DINOv2 + SigLIP2 are ~1.5 GB.
# A gpu-marked test run without HF_HOME set silently re-downloaded them there and
# took 700 MB off the system disk.
#
# Set here rather than documented, because "remember to export HF_HOME" is a rule
# that gets forgotten exactly once and then costs a disk. Anything already in the
# environment wins, so CI and other machines are unaffected.
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / "data" / "hf-cache"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


@pytest.fixture
def jpeg() -> bytes:
    return make_image(0, "JPEG")


@pytest.fixture
def png() -> bytes:
    return make_image(1, "PNG")


@pytest.fixture
def writer():
    """A PostgresWriter over an empty database.

    Lives here rather than in a test module so that both halves of the
    PostgresWriter suite -- the crawl tier in test_store.py and the face writer
    in test_store_faces.py -- get it by name, with no cross-module import. The
    alternative, importing the fixture, shadows it with the test's own
    parameter and trips F811.

    Wipes BEFORE as well as after. Cleaning up only on teardown makes every test
    depend on nothing else having touched the database first -- which broke the
    moment a real crawl was run against it by hand, since PostgresWriter seeds
    its dedup state from `image` at construction. A test that needs an empty
    database has to make one, not hope for one.
    """
    import psycopg

    from arc_search.index.store import PostgresWriter

    require_test_database(DSN)  # never truncate the production corpus

    scratch = psycopg.connect(DSN, autocommit=True)
    wipe(scratch)
    scratch.close()

    w = PostgresWriter(DSN)
    yield w
    wipe(w._conn)
    w.close()
