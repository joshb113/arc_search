"""Face detection, quality gating, cropping, and embedding.

The quality gate is the part that matters. eye_of_web indexed every detection
that cleared a single global ``det_thresh=0.75`` -- no minimum face size, no blur
check, no pose check. A 12x12 background face in a crowd shot got a full 512-d
embedding row weighted identically to a clean portrait.

At 10M+ faces that is not a cosmetic problem. Small, blurred, and extreme-profile
faces are the dominant source of false positives, and a one-in-a-million false
positive rate still yields ~10 bogus matches per query at this scale.

Crops are stored at 128px with a 15% margin and the landmarks are recorded at
ORIGINAL image resolution, so the crop stays re-alignable and a better model in
two years does not require re-crawling. eye_of_web pre-multiplied its landmarks
by 0.5 to line them up with a lossy half-scale thumbnail, using a constant
duplicated across two files with a TODO admitting it should have been shared.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import structlog

from arc_search.config import FaceSettings

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class Face:
    qdrant_id: uuid.UUID
    embedding: np.ndarray  # 512-d, L2-normalized
    bbox: tuple[float, float, float, float]
    landmarks: np.ndarray  # (5, 2) at ORIGINAL resolution
    src_width: int
    src_height: int
    det_score: float
    blur_var: float
    yaw: float | None
    quality: float
    age_est: int | None
    crop: np.ndarray  # BGR, crop_px square


@dataclass(frozen=True)
class RejectStats:
    too_small: int = 0
    low_score: int = 0
    too_blurry: int = 0
    bad_pose: int = 0
    underage: int = 0

    def total(self) -> int:
        return self.too_small + self.low_score + self.too_blurry + self.bad_pose + self.underage


class FaceExtractor:
    def __init__(self, cfg: FaceSettings) -> None:
        self._cfg = cfg
        self._app = None  # lazy: importing insightface costs ~2s

    def _ensure_app(self):
        if self._app is None:
            from insightface.app import FaceAnalysis

            app = FaceAnalysis(name=self._cfg.model_pack, providers=list(self._cfg.providers))
            app.prepare(ctx_id=0, det_size=self._cfg.det_size, det_thresh=self._cfg.det_thresh)
            self._app = app
            log.info(
                "faces.model_ready",
                pack=self._cfg.model_pack,
                providers=self._cfg.providers,
            )
        return self._app

    # ------------------------------------------------------------------ gate

    def _blur_variance(self, crop: np.ndarray) -> float:
        if crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _qualifies(self, det, crop: np.ndarray, stats: dict[str, int]) -> tuple[bool, float, float]:
        """Returns (passed, blur_var, quality)."""
        cfg = self._cfg
        x1, y1, x2, y2 = det.bbox
        w, h = float(x2 - x1), float(y2 - y1)

        if min(w, h) < cfg.min_face_px:
            stats["too_small"] += 1
            return False, 0.0, 0.0

        if float(det.det_score) < cfg.min_det_score:
            stats["low_score"] += 1
            return False, 0.0, 0.0

        blur = self._blur_variance(crop)
        if blur < cfg.min_blur_var:
            stats["too_blurry"] += 1
            return False, blur, 0.0

        yaw = float(det.pose[1]) if getattr(det, "pose", None) is not None else 0.0
        if abs(yaw) > cfg.max_abs_yaw:
            stats["bad_pose"] += 1
            return False, blur, 0.0

        age = int(det.age) if getattr(det, "age", None) is not None else None
        if age is not None and age < cfg.min_age_est:
            stats["underage"] += 1
            return False, blur, 0.0

        # Composite quality, used to weight re-ranking and to decide which crop
        # survives canonicalization. Deliberately simple and monotonic.
        size_term = min(1.0, min(w, h) / 224.0)
        blur_term = min(1.0, blur / 300.0)
        quality = float(det.det_score) * (0.5 + 0.3 * size_term + 0.2 * blur_term)
        return True, blur, quality

    # ------------------------------------------------------------------ crop

    def _crop(self, img: np.ndarray, bbox) -> np.ndarray:
        cfg = self._cfg
        h_img, w_img = img.shape[:2]
        x1, y1, x2, y2 = bbox
        mw, mh = (x2 - x1) * cfg.crop_margin, (y2 - y1) * cfg.crop_margin
        x1 = max(0, int(x1 - mw))
        y1 = max(0, int(y1 - mh))
        x2 = min(w_img, int(x2 + mw))
        y2 = min(h_img, int(y2 + mh))
        if x2 <= x1 or y2 <= y1:
            return np.empty((0, 0, 3), dtype=np.uint8)
        patch = img[y1:y2, x1:x2]
        return cv2.resize(patch, (cfg.crop_px, cfg.crop_px), interpolation=cv2.INTER_AREA)

    # ------------------------------------------------------------------ main

    def extract(self, img_bgr: np.ndarray) -> tuple[list[Face], RejectStats]:
        """Detect, gate, crop, and embed. Returns survivors plus reject counts.

        The reject counts are not decoration -- watch them. If ``too_small``
        dominates you are crawling thumbnail galleries; if ``too_blurry``
        dominates, image quality on the vertical is too low for face search to
        work at all. Both are findings you want in week 1, not month 4.
        """
        app = self._ensure_app()
        h_img, w_img = img_bgr.shape[:2]
        stats = {"too_small": 0, "low_score": 0, "too_blurry": 0, "bad_pose": 0, "underage": 0}
        out: list[Face] = []

        for det in app.get(img_bgr):
            crop = self._crop(img_bgr, det.bbox)
            if crop.size == 0:
                stats["too_small"] += 1
                continue

            passed, blur, quality = self._qualifies(det, crop, stats)
            if not passed:
                continue

            emb = np.asarray(det.normed_embedding, dtype=np.float32)
            if emb.shape != (512,):
                log.warning("faces.bad_embedding_shape", shape=emb.shape)
                continue

            out.append(
                Face(
                    qdrant_id=uuid.uuid4(),
                    embedding=emb,
                    bbox=tuple(float(v) for v in det.bbox),  # type: ignore[arg-type]
                    landmarks=np.asarray(det.kps, dtype=np.float32),  # ORIGINAL res
                    src_width=w_img,
                    src_height=h_img,
                    det_score=float(det.det_score),
                    blur_var=blur,
                    yaw=float(det.pose[1]) if getattr(det, "pose", None) is not None else None,
                    quality=quality,
                    age_est=int(det.age) if getattr(det, "age", None) is not None else None,
                    crop=crop,
                )
            )

        return out, RejectStats(**stats)

    def write_crop(self, face: Face, root: Path) -> str:
        """Persist the crop as WebP. Returns the path relative to ``root``.

        Sharded two levels deep by the uuid hex -- a flat directory with 10M
        entries is a filesystem problem on every OS.
        """
        hexid = face.qdrant_id.hex
        rel = Path(hexid[:2]) / hexid[2:4] / f"{hexid}.webp"
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        ok, buf = cv2.imencode(
            ".webp", face.crop, [int(cv2.IMWRITE_WEBP_QUALITY), self._cfg.crop_quality]
        )
        if not ok:
            raise OSError(f"failed to encode crop {hexid}")
        dest.write_bytes(buf.tobytes())
        return str(rel).replace("\\", "/")
