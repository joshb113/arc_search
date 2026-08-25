"""Read-only Postgres access for the query tier.

Separate from ``index.store.PostgresWriter`` on purpose. That class is a crawl
sink: it holds dedup state, interning caches and a writable connection, and it
seeds itself by reading the entire ``image`` table at construction. A web
process that answers queries needs none of that and must not pay for it.

This is the layer that turns a Qdrant hit into something a person can read:
which face, from which image, on which page, under what weak label. A result
that cannot answer "where did you see this" is not a search result, it is a
number -- and unauditable results are the thing the eye_of_web audit objected to
most.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace

import psycopg
import structlog

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class FaceHit:
    """One search result, hydrated with everything the UI needs to explain it."""

    face_id: int
    qdrant_id: uuid.UUID
    score: float  # RAW COSINE. Not confidence. See SearchSettings.calibrated.
    quality: float
    crop_path: str | None
    label: str | None  # weak label from alt text, e.g. "Ada Lovelace"
    image_url: str
    src_width: int
    src_height: int
    det_score: float
    age_est: int | None
    pages: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ImageHit:
    """One whole-image result: scene or text mode.

    ``score`` is RAW COSINE. For the text vector that is not even the model's
    own scale -- SigLIP is sigmoid-loss. Nothing here converts it to a verdict.
    """

    image_id: int
    score: float
    url: str
    width: int
    height: int
    face_count: int
    alt: str | None
    pages: list[str] = field(default_factory=list)
    # How many near-identical copies were folded into this result. Surfaced in
    # the UI rather than silently dropped -- "8 results" that were really 40 is
    # the kind of quiet lie that makes an index untrustworthy.
    duplicates: int = 0


class SearchRepo:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._conn = psycopg.connect(dsn, autocommit=True)

    def _exec(self, sql: str, params: tuple | None = None):
        try:
            return self._conn.execute(sql, params)
        except psycopg.Error:
            if self._conn.closed:
                self._conn = psycopg.connect(self._dsn, autocommit=True)
                return self._conn.execute(sql, params)
            raise

    # -- opt-out -----------------------------------------------------------

    def exclusions(self) -> list[uuid.UUID]:
        """Vectors to suppress, consulted on EVERY search.

        A legal obligation, not a ranking preference, which is why it exists
        from day one and why the filter is applied server-side in Qdrant rather
        than by dropping rows after retrieval -- post-filtering would let a
        suppressed face consume one of the k slots and silently shrink the
        result set.

        ⚠️ Scale limit: this materialises the whole list and passes it as a
        ``must_not``. Fine for the hundreds an opt-out list realistically holds.
        Past roughly a thousand, flip the design -- mark the point's payload on
        insert and filter on the flag instead of enumerating ids.
        """
        return [r[0] for r in self._exec("SELECT qdrant_id FROM exclusion").fetchall()]

    # -- hydration ---------------------------------------------------------

    def hydrate(self, hits) -> list[FaceHit]:
        """Turn Qdrant hits into displayable results, preserving rank order.

        One query for the whole page rather than one per hit: at return_k=100
        the per-hit version is 100 round trips to answer a single search.
        """
        if not hits:
            return []

        by_id = {h.face_id: h for h in hits}
        rows = self._exec(
            """
            SELECT f.id,
                   f.qdrant_id,
                   f.quality,
                   f.crop_path,
                   f.src_width,
                   f.src_height,
                   f.det_score,
                   f.age_est,
                   'https://' || d.host || i.url_path            AS image_url,
                   -- The weak label. max() because an image can appear on
                   -- several pages and only some of them carry alt text; a
                   -- LIMIT 1 here silently returns NULL when it picks the
                   -- wrong edge, which reads as "unlabelled" and is a lie.
                   max(t.body)                                   AS label,
                   array_remove(
                       array_agg(DISTINCT 'https://' || pd.host || u.path), NULL
                   )                                             AS pages
            FROM face f
            JOIN image i         ON i.id = f.image_id
            JOIN domain d        ON d.id = i.domain_id
            LEFT JOIN image_source s ON s.image_id = i.id
            LEFT JOIN page pg    ON pg.id = s.page_id
            LEFT JOIN domain pd  ON pd.id = pg.domain_id
            LEFT JOIN url_path u ON u.id = pg.url_path_id
            LEFT JOIN text_blob t ON t.id = s.alt_text_id
            WHERE f.id = ANY(%s)
            GROUP BY f.id, f.qdrant_id, f.quality, f.crop_path, f.src_width,
                     f.src_height, f.det_score, f.age_est, d.host, i.url_path
            """,
            (list(by_id),),
        ).fetchall()

        found = {}
        for r in rows:
            label = r[9]
            if label and label.startswith("Photo of "):
                label = label[len("Photo of ") :]
            found[r[0]] = FaceHit(
                face_id=r[0],
                qdrant_id=r[1],
                score=by_id[r[0]].score,
                quality=r[2],
                crop_path=r[3],
                src_width=r[4],
                src_height=r[5],
                det_score=r[6],
                age_est=r[7],
                image_url=r[8],
                label=label,
                pages=list(r[10] or []),
            )

        # Rank order is Qdrant's, not the database's. A vector with no row is
        # DROPPED rather than shown: it is an orphan the write order is designed
        # to prevent, and a result the UI could not attribute to any page.
        out = [found[h.face_id] for h in hits if h.face_id in found]
        if len(out) != len(hits):
            log.warning(
                "serve.orphan_vectors",
                dropped=len(hits) - len(out),
                detail="Qdrant returned points with no Postgres row; they were "
                "suppressed. Expected only mid-backfill, never at rest.",
            )
        return out

    # -- whole-image results (ADR-005) --------------------------------------

    def hydrate_images(self, hits: list[tuple[int, float]]) -> list[ImageHit]:
        """Turn (image_id, score) pairs into displayable results, in rank order.

        Simpler than ``hydrate`` because the whole-image collection keys points
        by ``image.id``, so there is no uuid to resolve and no orphan class to
        guard against -- the point id and the row id are the same thing.
        """
        if not hits:
            return []
        order = {image_id: i for i, (image_id, _) in enumerate(hits)}
        scores = dict(hits)

        rows = self._exec(
            """
            SELECT i.id,
                   'https://' || d.host || i.url_path,
                   i.width, i.height,
                   i.face_count,
                   max(t.body),
                   array_remove(
                       array_agg(DISTINCT 'https://' || pd.host || u.path), NULL
                   )
            FROM image i
            JOIN domain d ON d.id = i.domain_id
            LEFT JOIN image_source s ON s.image_id = i.id
            LEFT JOIN page pg   ON pg.id = s.page_id
            LEFT JOIN domain pd ON pd.id = pg.domain_id
            LEFT JOIN url_path u ON u.id = pg.url_path_id
            LEFT JOIN text_blob t ON t.id = s.alt_text_id
            WHERE i.id = ANY(%s)
            GROUP BY i.id, d.host, i.url_path, i.width, i.height, i.face_count
            """,
            (list(order),),
        ).fetchall()

        out = []
        for r in rows:
            alt = r[5]
            if alt and alt.startswith("Photo of "):
                alt = alt[len("Photo of ") :]
            out.append(
                ImageHit(
                    image_id=r[0],
                    score=scores[r[0]],
                    url=r[1],
                    width=r[2],
                    height=r[3],
                    face_count=r[4],
                    alt=alt,
                    pages=list(r[6] or []),
                )
            )
        out.sort(key=lambda h: order[h.image_id])
        return out

    def collapse_near_duplicates(self, hits: list[ImageHit], threshold: int = 31) -> list[ImageHit]:
        """Fold republished copies of the same image into one result.

        Measured need: of 225 labelled genuine pairs in this corpus, **149 (66%)
        are the same photograph re-published in a later year**, and 223 of 225
        have a DIFFERENT sha1 because they were re-encoded. Exact-hash dedup
        cannot see them at all, which is what PDQ is for.

        Without this, the first page of a grid fills with the same sponsor logo.

        Greedy and rank-preserving: walk in score order, keep a hit unless it is
        within ``threshold`` of one already kept. The kept copy is therefore
        always the highest-scoring one, and ``duplicates`` records how many were
        folded in so the UI can say so rather than silently dropping results.

        ``threshold`` 31 is PDQ's conventional "same image" boundary for 256
        bits. ⚠️ It is NOT calibrated against this corpus -- measured here, a
        JPEG re-encode of the same image scores 10, so 31 has real headroom, but
        that is one data point and not a derivation.
        """
        if not hits:
            return []

        from arc_search.index.dedup import hamming, pdq_to_bits

        rows = self._exec(
            "SELECT id, pdq FROM image WHERE id = ANY(%s) AND pdq IS NOT NULL",
            ([h.image_id for h in hits],),
        ).fetchall()
        bits = {r[0]: pdq_to_bits(r[1]) for r in rows}

        kept: list[ImageHit] = []
        kept_bits: list = []
        for h in hits:
            mine = bits.get(h.image_id)
            if mine is None:
                # No hash -- undecodable, or too low-quality for PDQ to be
                # meaningful. Show it rather than guess; a missing hash is not
                # evidence of duplication.
                kept.append(h)
                continue
            dup_of = next(
                (i for i, k in enumerate(kept_bits) if hamming(mine, k) <= threshold), None
            )
            if dup_of is None:
                kept.append(h)
                kept_bits.append(mine)
            else:
                kept[dup_of] = replace(kept[dup_of], duplicates=kept[dup_of].duplicates + 1)
        return kept

    def image_url(self, image_id: int) -> str | None:
        row = self._exec(
            """
            SELECT 'https://' || d.host || i.url_path
            FROM image i JOIN domain d ON d.id = i.domain_id WHERE i.id = %s
            """,
            (image_id,),
        ).fetchone()
        return row[0] if row else None

    # -- reporting ---------------------------------------------------------

    def stats(self) -> dict[str, int]:
        row = self._exec(
            """
            SELECT (SELECT count(*) FROM face),
                   (SELECT count(*) FROM image WHERE face_count > 0),
                   (SELECT count(*) FROM image),
                   (SELECT count(*) FROM page),
                   (SELECT count(*) FROM exclusion)
            """
        ).fetchone()
        assert row is not None
        return {
            "faces": row[0],
            "images_with_faces": row[1],
            "images": row[2],
            "pages": row[3],
            "exclusions": row[4],
        }

    def close(self) -> None:
        self._conn.close()
