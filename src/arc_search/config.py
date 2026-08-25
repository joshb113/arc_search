"""Central configuration.

Every tunable lives here. Two rules:

1. No magic numbers scattered through the codebase. eye_of_web had the literal
   ``0.6`` repeated in ~10 call sites and ``0.45`` in five more, with no single
   source of truth and no way to change either coherently.

2. Similarity thresholds are DERIVED, not chosen. The defaults below are marked
   UNCALIBRATED and are placeholders to make the pipeline runnable. Replace them
   with the output of ``python -m arc_search.eval.calibrate`` before trusting any
   result. eye_of_web's only justification for its thresholds was an inline
   comment reading "for balanced precision/recall" -- an assertion, not a
   measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# All four settings groups read the SAME .env, each filtering by its own prefix.
# pydantic-settings defaults a dotenv source to extra="forbid", which means a
# single ARC_CRAWL_* line makes FaceSettings raise ValidationError -- so a .env
# covering more than one group could never load at all. extra="ignore" is what
# makes one shared .env work.
#
# The cost is that a typo (ARC_CRAWL_PER_HOST_RPZ) is now silently ignored
# rather than rejected. That is exactly the failure mode this project refuses
# elsewhere, so `unknown_settings()` below pays it back: run.py calls it at
# startup and names anything that matched no field.
def _cfg(prefix: str) -> SettingsConfigDict:
    return SettingsConfigDict(env_prefix=prefix, env_file=".env", extra="ignore")


class CrawlSettings(BaseSettings):
    model_config = _cfg("ARC_CRAWL_")

    # Identity. One User-Agent, honest, with a contact route.
    # eye_of_web rotated 2010-era Opera strings, forged a Google Referer, masked
    # navigator.webdriver, and kept a Googlebot-Image constant on hand.
    user_agent: str = (
        "arc_search/0.1 (+https://github.com/YOURNAME/arc_search; contact: YOUR_EMAIL_HERE)"
    )

    respect_robots: bool = True
    """Turning this off is not supported. It exists so tests can run offline."""

    # Politeness. Per-host, not global.
    per_host_rps: float = 0.5  # 1 request per 2s per host
    per_host_burst: int = 2
    default_crawl_delay: float = 2.0  # fallback when robots.txt omits one
    max_crawl_delay: float = 30.0  # cap; above this we drop the host

    concurrency: int = 16  # total in-flight fetches across all hosts
    max_depth: int = 6
    timeout_s: float = 20.0

    # Retry. A transient 503 must not permanently drop a page.
    max_retries: int = 4
    backoff_base_s: float = 1.0
    backoff_max_s: float = 60.0

    # ---- Pre-index image filtering -----------------------------------------
    # MEASURED, not chosen. Both of these were magic numbers, and both of them
    # threw away the entire corpus. Sampled 9 FOSDEM 2025 speaker photos on
    # 2026-08-24 (vault/research/seed-vertical-conference-speakers.md):
    #
    #   dimensions  165x180 .. 180x180   -> min_image_dim=200 rejected 9 of 9
    #   byte size   5,399 .. 66,114 B    -> min_image_bytes=8000 rejected 4 of 9
    #
    # A 180x180 portrait is a perfectly good face image. The old 200 came from
    # nowhere and would have produced an empty index with no error anywhere.

    # DERIVED from FaceSettings.min_face_px, not picked. An image whose shorter
    # side is smaller than the smallest face we would accept cannot contain a
    # qualifying face -- that is the entire justification, and it is why these
    # two must move together. test_config.py pins the relationship.
    #
    # Moved 64 -> 48 on 2026-08-24 because min_face_px did; see the derivation
    # there. The invariant, not this literal, is the thing to preserve.
    min_image_dim: int = 48

    # A BANDWIDTH heuristic, not a quality gate. Its only job is to avoid
    # paying for tracking pixels and spacer GIFs, which are well under 1 KB.
    # Quality is decided by min_image_dim above and by the FaceSettings gate
    # after detection. Do not raise this to filter for "good" images; that is
    # what put 5 KB portraits in the bin.
    min_image_bytes: int = 2_000

    max_image_bytes: int = 20_000_000

    frontier_backend: str = "sqlite"  # "sqlite" | "redis"
    frontier_path: Path = Path("data/frontier.sqlite")

    seeds_file: Path = Path("seeds.yaml")


class FaceSettings(BaseSettings):
    model_config = _cfg("ARC_FACE_")

    # antelopev2 is R100/glint360k -- materially stronger than buffalo_l's R50.
    # eye_of_web fell back to buffalo_l because antelopev2 unzips to a nested
    # ~/.insightface/models/antelopev2/antelopev2 path that breaks Docker
    # plug-and-play. Dockerfile flattens it; see docker/flatten_models.sh
    model_pack: str = "antelopev2"
    providers: tuple[str, ...] = ("CUDAExecutionProvider", "CPUExecutionProvider")
    det_size: tuple[int, int] = (640, 640)
    det_thresh: float = 0.5  # detector floor; real gating is below

    # Quality gate. Applied to EVERY detection before it reaches the index.
    # These are the dominant source of false positives at 10M+ scale: a 12x12
    # background face in a crowd shot must not carry the same index weight as a
    # clean portrait.
    # DERIVED, 2026-08-24. See vault/research/face-model-bringup.md.
    #
    # Measured the embedding's cosine similarity to its OWN full-resolution self
    # as the face is downscaled, over real FOSDEM speaker photos:
    #
    #     96px 0.994 | 72px 0.989 | 64px 0.987 | 56px 0.981
    #     48px 0.973 (min 0.951)  | 40px 0.953 (min 0.932)
    #     32px 0.910 (min 0.885)  | 24px 0.815
    #
    # The knee is between 40 and 32, where the vector drops below
    # IndexSettings.canonical_threshold (0.92) -- i.e. where a face stops
    # matching itself. 48 keeps a real margin above that floor on the WORST
    # case, not just the mean.
    #
    # The old 64 was not derived from anything. It sat on the median face size
    # of this corpus (68px) and discarded 40% of all detections, which is the
    # same failure as min_image_dim=200 and min_image_bytes=8000 before it.
    #
    # ⚠️ This measures self-consistency under downscaling, NOT discriminability
    # between different people. The false-positive argument in faces.py is about
    # discrimination and needs labeled pairs -- so this number is still
    # provisional and belongs to arc_search.eval.calibrate in the end. What the
    # measurement establishes is that the MODEL does not require 64.
    min_face_px: int = 48
    min_det_score: float = 0.72
    min_blur_var: float = 45.0  # cv2.Laplacian variance
    max_abs_yaw: float = 50.0  # degrees

    # Age filter. Imperfect (it is an estimate), but it is one line and it is the
    # highest-value filter in the system. See README "Legal posture".
    min_age_est: int = 18

    # Crop storage. 128px + 15% margin + stored landmarks => re-alignable, so a
    # better model in 2027 does not require re-crawling.
    crop_px: int = 128
    crop_margin: float = 0.15
    crop_quality: int = 75  # WebP
    crop_dir: Path = Path("data/crops")

    batch_size: int = 32


@dataclass(frozen=True)
class CollectionSpec:
    """One Qdrant collection. arc_search has three; see ADR-005.

    Split out because ``IndexSettings`` used to describe exactly one collection
    (``collection`` + ``vector_dim``) and now has to describe three with
    different dimensions and different point identities.

    ``point_id`` is the structural difference and it is not cosmetic:

      ``uuid``      one point per FACE. An image with three faces makes three
                    points, so the id cannot be the image id and ``face.qdrant_id``
                    exists to map them back.
      ``image_id``  one point per IMAGE. The Postgres ``image.id`` IS the point
                    id -- verified against a live Qdrant, including 2**63-1 --
                    so no mapping table is needed, deletion is by id rather than
                    by payload filter, and an orphaned vector is structurally
                    impossible because the id and the row are the same thing.
    """

    name: str
    dim: int
    hnsw_m: int = 16
    hnsw_ef_construct: int = 200
    search_ef: int = 256
    point_id: str = "uuid"  # "uuid" | "image_id"


class IndexSettings(BaseSettings):
    model_config = _cfg("ARC_INDEX_")

    qdrant_url: str = "http://127.0.0.1:6333"

    # The FACE collection. Named without a prefix for historical reasons -- these
    # are bound to ARC_INDEX_COLLECTION / ARC_INDEX_VECTOR_DIM and predate there
    # being more than one collection. Use `spec("faces")` rather than reading
    # them directly in new code.
    collection: str = "faces"
    vector_dim: int = 512

    # The whole-image collection: scene and text vectors as NAMED VECTORS on one
    # point. They share a collection because both are per-image, so they share a
    # point identity. Faces cannot join them -- different granularity, see
    # CollectionSpec.point_id and plan-005.
    image_collection: str = "images"
    # 768 is MEASURED, not assumed: facebook/dinov2-base and
    # google/siglip2-base-patch16-384 are both 768-d.
    # See vault/research/image-model-bringup.md.
    scene_dim: int = 768
    text_dim: int = 768

    # int8 scalar, NOT binary. Face embeddings tolerate binarization far worse
    # than CLIP does, and the recall it costs lands precisely on the hard,
    # low-quality matches this engine exists to find.
    quantization: str = "int8"
    hnsw_m: int = 16
    hnsw_ef_construct: int = 200
    search_ef: int = 256

    # Face-level canonicalization: collapse near-identical vectors to one row.
    canonical_threshold: float = 0.92

    pg_dsn: str = Field(
        default="postgresql://arc@127.0.0.1:5432/arc_search",
        description="Password comes from PGPASSWORD or .pgpass, never the DSN.",
    )

    def face_spec(self) -> CollectionSpec:
        return CollectionSpec(
            name=self.collection,
            dim=self.vector_dim,
            hnsw_m=self.hnsw_m,
            hnsw_ef_construct=self.hnsw_ef_construct,
            search_ef=self.search_ef,
            point_id="uuid",
        )

    def image_spec(self) -> CollectionSpec:
        """Scene + text share this collection as named vectors.

        ``dim`` is the SCENE dimension; the text vector's size is carried
        separately by ``text_dim`` and applied by the vector store when it
        creates the named-vector config. They are both 768 today, and writing
        that as one number would hide the day they diverge.
        """
        return CollectionSpec(
            name=self.image_collection,
            dim=self.scene_dim,
            hnsw_m=self.hnsw_m,
            hnsw_ef_construct=self.hnsw_ef_construct,
            search_ef=self.search_ef,
            point_id="image_id",
        )


class SearchSettings(BaseSettings):
    model_config = _cfg("ARC_SEARCH_")

    retrieve_k: int = 500  # candidates pulled before re-rank
    return_k: int = 100  # after re-rank and clustering

    # ---- UNCALIBRATED PLACEHOLDERS -- replace via arc_search.eval.calibrate ----
    # Do not treat these as tuned. They exist so the pipeline runs end to end.
    t_plausible: float = 0.28
    t_strong: float = 0.40
    t_near_certain: float = 0.55
    ensemble_agreement: int = 2  # of 2 models; both must clear t_plausible
    calibrated: bool = False  # flipped to True by the calibrate run
    # --------------------------------------------------------------------------

    bind_host: str = "127.0.0.1"  # see README "Legal posture"
    bind_port: int = 8000


class Settings(BaseSettings):
    crawl: CrawlSettings = Field(default_factory=CrawlSettings)
    face: FaceSettings = Field(default_factory=FaceSettings)
    index: IndexSettings = Field(default_factory=IndexSettings)
    search: SearchSettings = Field(default_factory=SearchSettings)


_GROUPS: tuple[tuple[str, type[BaseSettings]], ...] = (
    ("ARC_CRAWL_", CrawlSettings),
    ("ARC_FACE_", FaceSettings),
    ("ARC_INDEX_", IndexSettings),
    ("ARC_SEARCH_", SearchSettings),
)


# ARC_-prefixed variables that are deliberately NOT pydantic fields. Without
# this, the typo check flags the project's own required variables.
_EXTERNAL_VARS = frozenset(
    {
        "ARC_PG_PASSWORD",  # consumed by docker-compose.yml, never by the app
        "ARC_TEST_PG_DSN",  # opt-in switch for the integration tests
    }
)


def unknown_settings(source: dict[str, str] | None = None) -> list[str]:
    """Names of ``ARC_*`` variables that match no field on any settings group.

    ``extra="ignore"`` is required for one shared .env to load at all, but it
    turns a typo into silence: set ``ARC_CRAWL_PER_HOST_RPZ=5`` and you get the
    default 0.5 rps with no complaint, then wonder why the crawl is slow. This
    is the compensating check -- run.py calls it at startup and names anything
    that landed nowhere.

    Deliberately a warning and not a hard failure: a shared .env legitimately
    picks up unrelated ``ARC_``-prefixed variables in some setups, and refusing
    to start over one is worse than saying so.
    """
    import os

    env = dict(os.environ) if source is None else dict(source)
    known: set[str] = set(_EXTERNAL_VARS)
    for prefix, cls in _GROUPS:
        known |= {prefix + name.upper() for name in cls.model_fields}

    return sorted(key for key in env if key.upper().startswith("ARC_") and key.upper() not in known)


settings = Settings()
