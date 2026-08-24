"""FaceExtractor: the quality gate, the crop, and the crop writer.

None of this needs the model. ``_qualifies``, ``_crop`` and ``write_crop`` are
ordinary functions over a detection object and an array, so they are tested
against synthetic detections -- which is the point, because the gate is the part
that decides what enters the index and it should not require a GPU to verify.
Anything that genuinely needs InsightFace belongs behind the ``gpu`` marker.

cv2 is imported at module scope by index/faces.py, and the fast CI job
deliberately installs no opencv (it would blow the one-minute budget). So this
module skips itself there rather than erroring the whole collection. The `gpu`
job installs the full stack and runs it.
"""

from __future__ import annotations

import os
import uuid

import pytest

pytest.importorskip("cv2", reason="fast CI installs no opencv; the gpu job does")

import numpy as np

from arc_search.config import FaceSettings
from arc_search.index.faces import Face, FaceExtractor, register_cuda_runtime


class FakeDetection:
    """Stands in for an insightface Face: bbox, det_score, pose, age, kps."""

    def __init__(self, bbox, det_score=0.9, yaw=0.0, age=30):
        self.bbox = np.asarray(bbox, dtype=np.float32)
        self.det_score = det_score
        self.pose = np.asarray([0.0, yaw, 0.0], dtype=np.float32)
        self.age = age


def _noise(w: int, h: int) -> np.ndarray:
    """A textured image. Flat colour has ~zero Laplacian variance and would trip
    the blur gate for reasons that have nothing to do with the test."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)


@pytest.fixture
def extractor() -> FaceExtractor:
    return FaceExtractor(FaceSettings())


# --- CUDA, and knowing whether you actually have it ------------------------
#
# The failure these guard against: onnxruntime does NOT raise when the CUDA
# provider cannot load. It writes one line to stderr and every session falls
# back to CPU. Measured on an RTX 5070, that is 49.0 img/s -> 4.1 img/s, and the
# only symptom is that the run is slow. arc_search shipped this exact defect
# class once before, in the crawler, where the startup log printed the REQUESTED
# per-host rate while the crawl ran at half of it.


def test_effective_providers_is_empty_before_the_model_loads(extractor):
    """It reports what is loaded, so with nothing loaded it reports nothing --
    rather than optimistically echoing the config back."""
    assert extractor.effective_providers() == {}


def test_effective_providers_reports_sessions_not_config():
    """The whole point: never echo the requested provider list.

    A config asking for CUDA against sessions that fell back to CPU must report
    CPU. This fakes the loaded app so the assertion holds with no GPU present.
    """

    class FakeSession:
        def get_providers(self):
            return ["CPUExecutionProvider"]

    class FakeModel:
        session = FakeSession()

    class FakeApp:
        def __init__(self):
            self.models = {"detection": FakeModel(), "recognition": FakeModel()}

    ex = FaceExtractor(FaceSettings(providers=("CUDAExecutionProvider", "CPUExecutionProvider")))
    ex._app = FakeApp()

    assert ex.effective_providers() == {
        "detection": ["CPUExecutionProvider"],
        "recognition": ["CPUExecutionProvider"],
    }
    # The config still says CUDA. The report must disagree with it.
    assert "CUDAExecutionProvider" in ex._cfg.providers


def test_register_cuda_runtime_is_idempotent():
    """Called on every lazy model load. Repeated calls must not grow PATH without
    bound -- this runs in a long-lived indexing process."""
    import os

    first = register_cuda_runtime()
    path_after_first = os.environ.get("PATH", "")
    second = register_cuda_runtime()

    assert first == second
    assert os.environ.get("PATH", "") == path_after_first


def test_register_cuda_runtime_returns_real_directories():
    """Either the NVIDIA wheels are installed and every returned path exists, or
    they are not and the list is empty. It must never return a path that is not
    there -- a bogus PATH entry would mask the real problem."""
    for d in register_cuda_runtime():
        assert os.path.isdir(d), d


# --- the quality gate ------------------------------------------------------


def test_a_tiny_face_is_rejected(extractor):
    """The eye_of_web failure this gate exists for: it indexed every detection
    over a single det_thresh, so a 12x12 background face in a crowd shot got a
    512-d embedding weighted the same as a clean portrait. At 10M faces those
    are the dominant source of false positives."""
    stats = dict.fromkeys(("too_small", "low_score", "too_blurry", "bad_pose", "underage"), 0)
    det = FakeDetection((0, 0, 12, 12))
    passed, _, _ = extractor._qualifies(det, _noise(12, 12), stats)
    assert passed is False
    assert stats["too_small"] == 1


def test_the_size_gate_is_min_face_px_not_area(extractor):
    """A 200x20 sliver has plenty of area and is not a usable face. The gate is
    on the SHORTER side for that reason."""
    stats = dict.fromkeys(("too_small", "low_score", "too_blurry", "bad_pose", "underage"), 0)
    det = FakeDetection((0, 0, 200, 20))
    passed, _, _ = extractor._qualifies(det, _noise(200, 20), stats)
    assert passed is False
    assert stats["too_small"] == 1


def test_a_blurry_face_is_rejected(extractor):
    """Flat grey has a Laplacian variance of ~0."""
    stats = dict.fromkeys(("too_small", "low_score", "too_blurry", "bad_pose", "underage"), 0)
    flat = np.full((128, 128, 3), 128, dtype=np.uint8)
    passed, blur, _ = extractor._qualifies(FakeDetection((0, 0, 128, 128)), flat, stats)
    assert passed is False
    assert stats["too_blurry"] == 1
    assert blur < FaceSettings().min_blur_var


def test_an_extreme_profile_is_rejected(extractor):
    stats = dict.fromkeys(("too_small", "low_score", "too_blurry", "bad_pose", "underage"), 0)
    det = FakeDetection((0, 0, 128, 128), yaw=75.0)
    passed, _, _ = extractor._qualifies(det, _noise(128, 128), stats)
    assert passed is False
    assert stats["bad_pose"] == 1


def test_an_apparent_minor_is_rejected(extractor):
    """Imperfect -- it is an estimate -- but it is one line and it is the
    highest-value filter in the system. See README 'Legal posture'."""
    stats = dict.fromkeys(("too_small", "low_score", "too_blurry", "bad_pose", "underage"), 0)
    det = FakeDetection((0, 0, 128, 128), age=12)
    passed, _, _ = extractor._qualifies(det, _noise(128, 128), stats)
    assert passed is False
    assert stats["underage"] == 1


def test_a_clean_portrait_passes(extractor):
    stats = dict.fromkeys(("too_small", "low_score", "too_blurry", "bad_pose", "underage"), 0)
    det = FakeDetection((0, 0, 200, 200), det_score=0.95)
    passed, blur, quality = extractor._qualifies(det, _noise(200, 200), stats)
    assert passed is True, stats
    assert sum(stats.values()) == 0
    assert blur > 0
    assert 0.0 < quality <= 1.0


def test_quality_rises_with_size_and_sharpness(extractor):
    """Quality weights re-ranking and decides which crop survives
    canonicalization, so its ordering has to be monotonic in the obvious way."""
    stats = dict.fromkeys(("too_small", "low_score", "too_blurry", "bad_pose", "underage"), 0)
    _, _, small = extractor._qualifies(FakeDetection((0, 0, 70, 70)), _noise(70, 70), stats)
    _, _, big = extractor._qualifies(FakeDetection((0, 0, 300, 300)), _noise(300, 300), stats)
    assert big > small


# --- cropping --------------------------------------------------------------


def test_the_crop_is_square_and_the_configured_size(extractor):
    crop = extractor._crop(_noise(800, 600), (100.0, 100.0, 200.0, 220.0))
    assert crop.shape == (128, 128, 3)


def test_the_crop_margin_is_applied(extractor):
    """15% margin keeps the jaw and hairline in frame, which is what makes the
    crop re-alignable later from the stored landmarks."""
    img = _noise(800, 600)
    box = (100.0, 100.0, 200.0, 200.0)
    cfg = FaceSettings()
    mw = (box[2] - box[0]) * cfg.crop_margin
    assert mw == pytest.approx(15.0)
    # Nothing to assert on the resized output directly, so check the extractor
    # is reading the config rather than a literal.
    assert extractor._crop(img, box).shape[0] == cfg.crop_px


def test_a_crop_at_the_image_edge_is_clamped_not_wrapped(extractor):
    """A face at x=0 has a negative margin. Numpy would happily interpret a
    negative slice bound as counting from the far edge and hand back a crop of
    the opposite corner."""
    crop = extractor._crop(_noise(200, 200), (0.0, 0.0, 60.0, 60.0))
    assert crop.shape == (128, 128, 3)


def test_a_degenerate_box_returns_empty_rather_than_raising(extractor):
    """Detectors do occasionally emit inverted or zero-area boxes. One bad box
    must not take down the pass over an image."""
    assert extractor._crop(_noise(200, 200), (100.0, 100.0, 100.0, 100.0)).size == 0
    assert extractor._crop(_noise(200, 200), (150.0, 150.0, 50.0, 50.0)).size == 0


# --- crop storage ----------------------------------------------------------


def _face(crop: np.ndarray | None = None) -> Face:
    return Face(
        qdrant_id=uuid.UUID("0123456789abcdef0123456789abcdef"),
        embedding=np.zeros(512, dtype=np.float32),
        bbox=(0.0, 0.0, 10.0, 10.0),
        landmarks=np.zeros((5, 2), dtype=np.float32),
        src_width=800,
        src_height=600,
        det_score=0.9,
        blur_var=100.0,
        yaw=0.0,
        quality=0.8,
        age_est=30,
        crop=_noise(128, 128) if crop is None else crop,
    )


def test_crops_are_sharded_two_levels_deep(extractor, tmp_path):
    """A flat directory with 10M entries is a filesystem problem on every OS --
    ext4 slows to a crawl on lookup and NTFS is worse."""
    rel = extractor.write_crop(_face(), tmp_path)
    assert rel == "01/23/0123456789abcdef0123456789abcdef.webp"
    assert (tmp_path / rel).is_file()


def test_the_returned_path_uses_forward_slashes(extractor, tmp_path):
    """It goes into face.crop_path in Postgres and then into a URL. A Windows
    backslash would round-trip into the served path and 404."""
    rel = extractor.write_crop(_face(), tmp_path)
    assert "\\" not in rel


def test_the_crop_is_webp_and_within_the_storage_budget(extractor, tmp_path):
    """ADR-001 budgets ~4 KB/face; the 49.6 GB scale target is built on it.
    Random noise is close to the worst case a real crop can be -- a photo of a
    face compresses better than this."""
    rel = extractor.write_crop(_face(), tmp_path)
    written = (tmp_path / rel).read_bytes()
    assert written[:4] == b"RIFF" and written[8:12] == b"WEBP"
    assert len(written) < 12_000, f"{len(written)} B for pure noise is suspiciously large"


def test_write_crop_creates_missing_shard_directories(extractor, tmp_path):
    root = tmp_path / "does" / "not" / "exist"
    rel = extractor.write_crop(_face(), root)
    assert (root / rel).is_file()


def test_two_faces_never_collide_on_disk(extractor, tmp_path):
    """Crop identity is the qdrant_id, which is also the join key to the vector.
    A collision would silently overwrite one face's crop with another's."""
    import dataclasses

    a = _face()
    b = dataclasses.replace(a, qdrant_id=uuid.uuid4())
    assert extractor.write_crop(a, tmp_path) != extractor.write_crop(b, tmp_path)
    assert len(list(tmp_path.rglob("*.webp"))) == 2
