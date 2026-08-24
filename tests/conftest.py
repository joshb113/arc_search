"""Shared pytest fixtures.

The image helpers live in ``imagefixtures.py`` rather than here, because a
module imported *from* conftest is only resolvable when the repo root is on
sys.path -- true under ``python -m pytest``, false under a bare ``pytest``.
See that module's docstring.
"""

from __future__ import annotations

import pytest

from imagefixtures import make_image


@pytest.fixture
def jpeg() -> bytes:
    return make_image(0, "JPEG")


@pytest.fixture
def png() -> bytes:
    return make_image(1, "PNG")
