"""Dedup: BK-tree correctness and gate ordering.

The ordering test is the important one. eye_of_web ran its SHA1 check *after* the
blob insert, so a known duplicate was fully downloaded, detected, re-encoded, and
pushed through an unindexed full-blob equality scan before the cheap hash lookup
could short-circuit anything.
"""

from __future__ import annotations

import numpy as np
import pytest

from arc_search.index.dedup import BKTree, Deduper, Verdict, hamming, sha1_bytes


def bits(*byte_values: int) -> np.ndarray:
    return np.array(byte_values, dtype=np.uint8)


class TestHamming:
    def test_identical_is_zero(self):
        assert hamming(bits(0xFF, 0x00), bits(0xFF, 0x00)) == 0

    def test_single_bit(self):
        assert hamming(bits(0x00), bits(0x01)) == 1

    def test_full_byte(self):
        assert hamming(bits(0x00), bits(0xFF)) == 8

    def test_symmetric(self):
        a, b = bits(0xA5, 0x3C), bits(0x5A, 0xC3)
        assert hamming(a, b) == hamming(b, a)


class TestBKTree:
    def test_empty_returns_none(self):
        assert BKTree().find(bits(0x00), 10) is None

    def test_exact_match(self):
        t = BKTree()
        t.add(bits(0xF0, 0x0F), 42)
        assert t.find(bits(0xF0, 0x0F), 0) == 42

    def test_within_threshold(self):
        t = BKTree()
        t.add(bits(0x00, 0x00), 7)
        assert t.find(bits(0x00, 0x03), threshold=2) == 7  # distance 2

    def test_outside_threshold(self):
        t = BKTree()
        t.add(bits(0x00, 0x00), 7)
        assert t.find(bits(0x00, 0xFF), threshold=2) is None  # distance 8

    def test_returns_closest_of_several(self):
        t = BKTree()
        t.add(bits(0x00, 0x00), 1)  # distance 8 from query
        t.add(bits(0x00, 0x01), 2)  # distance 7
        t.add(bits(0x00, 0x7F), 3)  # distance 1
        assert t.find(bits(0x00, 0xFF), threshold=10) == 3

    def test_duplicate_key_is_ignored(self):
        t = BKTree()
        t.add(bits(0x01), 1)
        t.add(bits(0x01), 2)
        assert len(t) == 1
        assert t.find(bits(0x01), 0) == 1

    def test_size_tracks_inserts(self):
        t = BKTree()
        for i in range(1, 9):
            t.add(bits(i), i)
        assert len(t) == 8

    @pytest.mark.parametrize("threshold", [0, 1, 4, 16, 64])
    def test_matches_bruteforce(self, threshold):
        """BK-tree pruning must never change the answer versus linear scan."""
        rng = np.random.default_rng(1234)
        keys = [rng.integers(0, 256, size=32, dtype=np.uint8) for _ in range(200)]
        tree = BKTree()
        for i, k in enumerate(keys):
            tree.add(k, i)

        for _ in range(50):
            q = rng.integers(0, 256, size=32, dtype=np.uint8)
            best_d = min(hamming(q, k) for k in keys)
            got = tree.find(q, threshold)
            if best_d <= threshold:
                assert got is not None
                assert hamming(q, keys[got]) <= threshold
            else:
                assert got is None


class TestDeduperOrdering:
    def test_exact_dup_detected_before_decode(self):
        """check_bytes must resolve a known image without ever touching pixels."""
        d = Deduper()
        raw = b"\x89PNG fake image bytes"
        d.register(sha1_bytes(raw), pdq=None, image_id=99, face_count=3)

        result = d.check_bytes(raw)
        assert result is not None
        assert result.verdict is Verdict.EXACT_DUP
        assert result.matched_image_id == 99

    def test_new_bytes_return_none_so_pipeline_continues(self):
        assert Deduper().check_bytes(b"never seen") is None

    def test_barren_image_short_circuits(self):
        """An image already known to contain no qualifying face must never reach
        the GPU again on a recrawl."""
        d = Deduper()
        raw = b"a landscape with no people"
        d.register(sha1_bytes(raw), pdq=None, image_id=5, face_count=0)

        result = d.check_bytes(raw)
        assert result is not None
        assert result.verdict is Verdict.BARREN

    def test_load_seeds_all_three_structures(self):
        d = Deduper()
        pdq = bits(*([0x0F] * 32))
        d.load(
            [
                (1, sha1_bytes(b"one"), pdq.tobytes(), 2),
                (2, sha1_bytes(b"two"), None, 0),
            ]
        )
        assert d.check_bytes(b"one").verdict is Verdict.EXACT_DUP
        assert d.check_bytes(b"two").verdict is Verdict.BARREN
        assert d.check_bytes(b"three") is None
