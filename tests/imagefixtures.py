"""Real encoded images for tests.

Deliberately NOT in conftest.py. conftest is auto-loaded by pytest under the
module name ``conftest``, not ``tests.conftest``, so importing it by path only
works when the repository root happens to be on ``sys.path`` -- which it is
under ``python -m pytest`` (which prepends the CWD) and is not under a bare
``pytest``. CI runs the latter. A plain module plus ``pythonpath = ["src",
"tests"]`` in pyproject resolves identically under both.

Why real images: the original fixtures were ``b"\\xff\\xd8\\xff\\xe0" + b"A" *
20000`` -- enough to satisfy ``sniff_image_type`` and nothing else. That held up
until the fetch path started reading image headers, at which point every crawl
test failed with ``unreadable_header``. The fixtures were the weak part, not
the code: a corpus no decoder would accept cannot tell you whether the pipeline
handles real images.

Noise rather than flat colour, because a solid 240x240 PNG compresses to well
under the 8 KB ``min_image_bytes`` floor and gets skipped as a tracking pixel.
"""

from __future__ import annotations

import random
from functools import lru_cache
from io import BytesIO


@lru_cache(maxsize=32)
def make_image(seed: int = 0, fmt: str = "JPEG", size: tuple[int, int] = (240, 240)) -> bytes:
    """A real, decodable image. Cached -- encoding noise is not free.

    Defaults clear both gates in CrawlSettings: 240x240 is over min_image_dim
    (200), and noise at q90 encodes to comfortably over min_image_bytes (8000).
    """
    from PIL import Image

    rnd = random.Random(seed)
    w, h = size
    im = Image.new("RGB", (w, h))
    im.putdata([(rnd.randrange(256), rnd.randrange(256), rnd.randrange(256)) for _ in range(w * h)])
    buf = BytesIO()
    if fmt == "JPEG":
        im.save(buf, fmt, quality=90)
    else:
        im.save(buf, fmt)
    return buf.getvalue()
