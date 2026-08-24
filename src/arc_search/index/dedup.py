"""Deduplication, in the order that actually saves work.

The ordering here is the whole point. eye_of_web ran its SHA1 check *after* the
blob insert, so a known-duplicate image was still downloaded, face-detected,
resized, WebP-encoded, zlib-compressed, and dragged through an unindexed
full-blob equality scan before the cheap hash lookup got a chance to
short-circuit anything.

Correct order, cheapest gate first:

    1. URL seen?          -- frontier, free
    2. SHA1 of raw bytes  -- one hash, one indexed lookup
    3. PDQ perceptual     -- one DCT, one BK-tree probe
    4. GPU detection      -- expensive, runs only on survivors

Step 3 is what eye_of_web lacked entirely (zero repo-wide matches for
imagehash/phash/pdq). With SHA1-only dedup, one re-encode or a single-pixel crop
defeats it, so the same photo circulating across fifty sites became fifty blobs
and fifty near-identical index rows -- which then inflated match counts in a way
that reads like corroborating evidence but is one image counted fifty times.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import structlog

log = structlog.get_logger(__name__)

try:
    import pdqhash

    _HAVE_PDQ = True
except ImportError:  # pragma: no cover - optional at scaffold time
    _HAVE_PDQ = False
    log.warning("pdq.unavailable", hint="pip install pdqhash; falling back to dHash")


class Verdict(StrEnum):
    NEW = "new"
    EXACT_DUP = "exact_dup"  # identical bytes, already indexed
    NEAR_DUP = "near_dup"  # perceptually identical, already indexed
    BARREN = "barren"  # known to contain no qualifying face


@dataclass(frozen=True)
class DedupResult:
    verdict: Verdict
    sha1: bytes
    pdq: np.ndarray | None = None
    matched_image_id: int | None = None


def sha1_bytes(raw: bytes) -> bytes:
    return hashlib.sha1(raw).digest()


def _dhash_fallback(img_bgr: np.ndarray) -> np.ndarray:
    """64-bit difference hash. Weaker than PDQ but dependency-free.

    Only used if pdqhash is unavailable. Do not ship on this path -- PDQ is
    256-bit and purpose-built for exactly this job.
    """
    import cv2

    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return np.packbits(bits)


def perceptual_hash(img_bgr: np.ndarray) -> tuple[np.ndarray, int]:
    """Return (hash_bits, quality). Quality is PDQ's own 0-100 confidence."""
    if _HAVE_PDQ:
        import cv2

        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        vector, quality = pdqhash.compute(rgb)
        return np.asarray(vector, dtype=np.uint8), int(quality)
    return _dhash_fallback(img_bgr), 100


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.unpackbits(np.bitwise_xor(a, b)).sum())


class BKTree:
    """BK-tree over Hamming distance for near-duplicate probes.

    Seeded from the ``image.pdq`` column at startup. For PDQ's 256-bit space a
    threshold of ~31 is the conventional "same image" boundary; tune it on your
    own corpus rather than trusting that number blindly.
    """

    __slots__ = ("_root", "_size")

    def __init__(self) -> None:
        self._root: tuple[np.ndarray, int, dict[int, object]] | None = None
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(self, key: np.ndarray, value: int) -> None:
        if self._root is None:
            self._root = (key, value, {})
            self._size = 1
            return
        node = self._root
        while True:
            d = hamming(key, node[0])
            if d == 0:
                return  # already present
            child = node[2].get(d)
            if child is None:
                node[2][d] = (key, value, {})
                self._size += 1
                return
            node = child  # type: ignore[assignment]

    def find(self, key: np.ndarray, threshold: int) -> int | None:
        """Return the value of the closest match within ``threshold``, or None."""
        if self._root is None:
            return None
        best: tuple[int, int] | None = None
        stack = [self._root]
        while stack:
            node = stack.pop()
            d = hamming(key, node[0])
            if d <= threshold and (best is None or d < best[0]):
                best = (d, node[1])
                if d == 0:
                    break
            lo, hi = d - threshold, d + threshold
            stack.extend(c for dist, c in node[2].items() if lo <= dist <= hi)  # type: ignore[misc]
        return best[1] if best else None


class Deduper:
    """Gate every candidate image through, in order, before touching the GPU."""

    def __init__(self, near_dup_threshold: int = 31) -> None:
        self._threshold = near_dup_threshold
        self._sha1: dict[bytes, int] = {}
        self._barren: set[bytes] = set()
        self._tree = BKTree()

    def load(self, rows: list[tuple[int, bytes, bytes | None, int]]) -> None:
        """Seed from Postgres: (image_id, sha1, pdq_bytes, face_count)."""
        for image_id, sha1, pdq, face_count in rows:
            self._sha1[sha1] = image_id
            if face_count == 0:
                self._barren.add(sha1)
            if pdq is not None:
                self._tree.add(np.frombuffer(pdq, dtype=np.uint8), image_id)
        log.info("dedup.loaded", images=len(self._sha1), pdq=len(self._tree))

    def check_bytes(self, raw: bytes) -> DedupResult | None:
        """Stage 2. Call before decoding. Returns None if the image is new."""
        digest = sha1_bytes(raw)
        if digest in self._barren:
            return DedupResult(Verdict.BARREN, digest, matched_image_id=self._sha1.get(digest))
        if (image_id := self._sha1.get(digest)) is not None:
            return DedupResult(Verdict.EXACT_DUP, digest, matched_image_id=image_id)
        return None

    def check_pixels(self, raw: bytes, img_bgr: np.ndarray) -> DedupResult:
        """Stage 3. Call after decoding, before detection."""
        digest = sha1_bytes(raw)
        pdq, quality = perceptual_hash(img_bgr)
        # Low-quality PDQ (flat or noisy images) produces unreliable neighbours;
        # index it but do not trust it for matching.
        if quality >= 50 and (match := self._tree.find(pdq, self._threshold)) is not None:
            return DedupResult(Verdict.NEAR_DUP, digest, pdq, match)
        return DedupResult(Verdict.NEW, digest, pdq)

    def mark_barren(self, sha1: bytes) -> None:
        """Record that this image was examined and held no qualifying face.

        Keyed by sha1, like the rest of the barren set -- the index tier learns
        an image is barren by image_id, so the caller has to resolve it back to
        the digest. Without this, a long-lived process that both crawls and
        indexes keeps re-examining images it already rejected, because the
        in-memory dedup state only learns about barrenness at startup.
        """
        self._barren.add(sha1)

    def register(self, sha1: bytes, pdq: np.ndarray | None, image_id: int, face_count: int) -> None:
        self._sha1[sha1] = image_id
        if face_count == 0:
            self._barren.add(sha1)
        if pdq is not None:
            self._tree.add(pdq, image_id)
