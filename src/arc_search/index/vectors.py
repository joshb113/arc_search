"""Qdrant-backed store for the 512-d face embeddings.

WHAT IS INDEXED
---------------
The face embedding, and nothing else. Non-negotiable #4. Landmarks and bbox are
*payload* in Postgres, not vectors -- eye_of_web built a second 4-dimensional
HNSW index over bounding boxes, which is meaningless: HNSW exists to dodge the
curse of dimensionality, and in 4 dimensions there is no curse to dodge.

The payload carried here is deliberately minimal -- ``image_id``, ``face_id``,
``quality``. Enough to filter and to join back to Postgres, and no more.
Postgres is the source of truth for everything else; duplicating it into the
payload means two things to keep in step and one of them will drift.

CLIENT/SERVER VERSION COUPLING
------------------------------
The pinned image in docker-compose.yml and ``qdrant-client`` in pyproject.toml
must stay on the same minor. They drifted once -- server 1.9.2 against client
1.19 -- and left NO working search path: ``query_points`` 404s against a server
older than 1.10, and ``search`` was removed from the client after 1.13. Writes
kept succeeding, so the collection would have filled normally and the failure
would have surfaced at the first query, not at the first insert.

CONSISTENCY WITH POSTGRES
-------------------------
Two stores, no distributed transaction, so the write order is what buys
correctness. ``image.face_count`` is the commit marker for BOTH stores:

    1. Postgres: DELETE old face rows, INSERT new ones   (face_count still -1)
    2. Qdrant:   delete by image_id filter, then upsert
    3. Postgres: UPDATE image SET face_count = N         <- commits the pair

A crash anywhere before step 3 leaves ``face_count = -1``, so the image is
still in the work queue and gets reprocessed; steps 1 and 2 both clear their
store first, so reprocessing converges rather than duplicating. The invariant
worth remembering:

    face_count > -1  =>  both stores are complete and agree for that image.
    face_count == -1 =>  either store may hold partial state; it is disposable.

This is why the filtered delete in ``delete_image`` matters, and why
``image_id`` carries a payload index -- without one that delete is a full scan.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import structlog
from qdrant_client import QdrantClient, models

from arc_search.config import IndexSettings

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class VectorRecord:
    """One face, ready to index.

    Deliberately not ``faces.Face``: that type carries a decoded crop and drags
    in cv2/insightface. The vector store has no business importing a detector.
    """

    qdrant_id: uuid.UUID
    embedding: np.ndarray  # 512-d float32, L2-normalized
    image_id: int
    face_id: int
    quality: float


@dataclass(frozen=True)
class Hit:
    qdrant_id: uuid.UUID
    score: float
    image_id: int
    face_id: int
    quality: float


class VectorStore:
    """The faces collection. Create it, write to it, query it, count it."""

    def __init__(self, cfg: IndexSettings, client: QdrantClient | None = None) -> None:
        self._cfg = cfg
        self._client = client if client is not None else QdrantClient(url=cfg.qdrant_url)

    @property
    def name(self) -> str:
        return self._cfg.collection

    @property
    def client(self) -> QdrantClient:
        return self._client

    # -- schema ------------------------------------------------------------

    def ensure_collection(self) -> bool:
        """Create the collection if absent. Returns True if it was created.

        Idempotent, and deliberately does NOT reconcile an existing collection
        against the current settings. Changing ``vector_dim`` or the distance
        metric under a populated collection is not a config edit, it is a
        migration, and silently doing half of one is worse than refusing.
        ``verify()`` reports the mismatch instead.
        """
        cfg = self._cfg
        if self._client.collection_exists(self.name):
            return False

        self._client.create_collection(
            self.name,
            vectors_config=models.VectorParams(
                size=cfg.vector_dim,
                distance=models.Distance.COSINE,
                # Originals stay on disk; the int8 copy in RAM is what gets
                # searched. At 10M faces the originals are ~20 GB and the
                # quantized copy ~5 GB -- see the disk-budget note in
                # vault/plans/plan-002-index-and-query.md.
                on_disk=True,
            ),
            # int8 scalar, NOT binary. Face embeddings tolerate binarization far
            # worse than CLIP does, and the recall it costs lands precisely on
            # the hard, low-quality matches this engine exists to find.
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True,
                )
            ),
            hnsw_config=models.HnswConfigDiff(
                m=cfg.hnsw_m,
                ef_construct=cfg.hnsw_ef_construct,
            ),
        )
        # Required for delete_image to be a filtered delete rather than a scan.
        self._client.create_payload_index(
            self.name,
            field_name="image_id",
            field_schema=models.PayloadSchemaType.INTEGER,
        )
        log.info(
            "vectors.collection_created",
            collection=self.name,
            dim=cfg.vector_dim,
            m=cfg.hnsw_m,
        )
        return True

    def verify(self) -> list[str]:
        """Names of settings that disagree with the live collection.

        Empty list means the collection matches this config. A non-empty one
        means someone changed a setting that only takes effect on a rebuild --
        report it loudly rather than writing vectors into a collection shaped
        differently from what the caller believes.
        """
        if not self._client.collection_exists(self.name):
            return ["collection_missing"]

        info = self._client.get_collection(self.name)
        params = info.config.params.vectors
        problems: list[str] = []
        if getattr(params, "size", None) != self._cfg.vector_dim:
            problems.append(f"vector_dim: config={self._cfg.vector_dim} live={params.size}")
        if getattr(params, "distance", None) != models.Distance.COSINE:
            problems.append(f"distance: expected COSINE live={params.distance}")
        return problems

    # -- write -------------------------------------------------------------

    def _point(self, rec: VectorRecord) -> models.PointStruct:
        emb = np.asarray(rec.embedding, dtype=np.float32).reshape(-1)
        if emb.shape != (self._cfg.vector_dim,):
            # Loud, not skipped. A wrong-shaped embedding means the model pack
            # is not what config says it is, and every vector already written
            # is suspect.
            raise ValueError(
                f"embedding for face {rec.face_id} has shape {emb.shape}, "
                f"expected ({self._cfg.vector_dim},)"
            )
        return models.PointStruct(
            id=str(rec.qdrant_id),
            vector=emb.tolist(),
            payload={
                "image_id": rec.image_id,
                "face_id": rec.face_id,
                "quality": float(rec.quality),
            },
        )

    def upsert(self, records: Sequence[VectorRecord]) -> int:
        """Write vectors. Returns how many. Upsert, so replay is harmless."""
        if not records:
            return 0
        self._client.upsert(self.name, points=[self._point(r) for r in records])
        return len(records)

    def delete_image(self, image_id: int) -> None:
        """Drop every vector belonging to one image.

        Called before re-indexing that image, so a reprocessed image cannot
        leave its previous detections behind as orphans. Orphans are not
        cosmetic here: they would score in queries with no Postgres row to
        join, i.e. results the UI cannot explain or attribute.
        """
        self._client.delete(
            self.name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="image_id", match=models.MatchValue(value=image_id)
                        )
                    ]
                )
            ),
        )

    # -- read --------------------------------------------------------------

    def search(
        self,
        embedding: np.ndarray,
        limit: int | None = None,
        exclude: Iterable[uuid.UUID] = (),
    ) -> list[Hit]:
        """Nearest neighbours by cosine similarity.

        ``exclude`` is the opt-out list from the ``exclusion`` table, applied
        as a server-side ``must_not`` so a suppressed face never reaches the
        ranking stage -- filtering after retrieval would let an excluded vector
        consume one of the k slots and silently shrink the result set.

        Scores are raw cosine similarity. They are NOT confidence, and nothing
        here maps them onto t_plausible/t_strong -- those are UNCALIBRATED
        placeholders until arc_search.eval.calibrate has run. Non-negotiable #5.
        """
        emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if emb.shape != (self._cfg.vector_dim,):
            raise ValueError(
                f"query embedding has shape {emb.shape}, expected ({self._cfg.vector_dim},)"
            )

        excluded = [str(u) for u in exclude]
        flt = models.Filter(must_not=[models.HasIdCondition(has_id=excluded)]) if excluded else None

        resp = self._client.query_points(
            self.name,
            query=emb.tolist(),
            limit=limit if limit is not None else 100,
            query_filter=flt,
            search_params=models.SearchParams(
                hnsw_ef=self._cfg.search_ef,
                # Rescore against the on-disk originals. Without this the
                # ranking is decided by the int8 approximation, which is a
                # meaningful accuracy loss exactly at the low-similarity end.
                quantization=models.QuantizationSearchParams(rescore=True),
            ),
            with_payload=True,
        )

        out: list[Hit] = []
        for p in resp.points:
            payload = p.payload or {}
            out.append(
                Hit(
                    qdrant_id=uuid.UUID(str(p.id)),
                    score=float(p.score),
                    image_id=int(payload.get("image_id", -1)),
                    face_id=int(payload.get("face_id", -1)),
                    quality=float(payload.get("quality", 0.0)),
                )
            )
        return out

    def count(self) -> int:
        if not self._client.collection_exists(self.name):
            return 0
        return int(self._client.count(self.name, exact=True).count)

    def close(self) -> None:
        self._client.close()
