"""Drain the indexing work queue: re-fetch, detect, embed, store.

This is the tier that finally puts a model in the loop. It reads
``image.face_count = -1``, re-fetches the bytes, runs the detector, and writes
faces to Postgres, vectors to Qdrant, and crops to disk.

WHY IT RE-FETCHES
-----------------
Because the bytes are gone. Non-negotiable #1 and ADR-001: nothing stores full
scene images, so the crawl tier hashed each image, measured it, and dropped it.
ADR-003 added the URL back to ``image`` for exactly this moment. That makes the
backfill a **crawler**, not a batch job -- it goes through ``Fetcher``, so
robots.txt and per-host rate limiting apply identically. There is no back door.

⚠️ **Two processes, one host.** Politeness state is per-process and in-memory.
Running this while ``crawler.run`` is crawling the same host means the host sees
BOTH, i.e. up to twice the configured rate. Either stop the crawl, or halve
``ARC_CRAWL_PER_HOST_RPS`` for both, or pass ``--rps`` here. Nothing can detect
this for you; the check at startup can only warn.

THROUGHPUT
----------
Deliberately sequential. The GPU does ~49 img/s and politeness allows 1 req/s,
so this is fetch-bound by a factor of ~50 and concurrency would buy nothing but
complexity. If the corpus ever spans many hosts, per-host parallelism is the
thing to add -- not a bigger batch.

THE WRITE ORDER IS NOT NEGOTIABLE
---------------------------------
``record_faces()`` -> Qdrant ``delete_image()``/``upsert()`` -> ``mark_examined()``.
``face_count`` is the commit marker for both stores; see ``vectors.py``. A crash
anywhere before the last step leaves the image on the queue, and both stores
clear before rewriting, so a re-run converges instead of duplicating.

AND NOTHING IS RETIRED PERMANENTLY
----------------------------------
``mark_examined(..., calibrated=False)`` records an empty result as ``-2``
(re-examinable), not ``0`` (tombstone). Every gate that can produce "no
qualifying face" is still an uncalibrated placeholder, so a 0 written now would
make today's unjustified numbers irreversible. See ADR-004.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import signal
from dataclasses import dataclass, field
from pathlib import Path

import structlog

from arc_search.config import CrawlSettings, FaceSettings, IndexSettings, SearchSettings
from arc_search.crawler.fetch import Fetcher, Skipped
from arc_search.index.faces import FaceExtractor
from arc_search.index.store import PostgresWriter
from arc_search.index.vectors import VectorRecord, VectorStore

log = structlog.get_logger(__name__)


@dataclass
class BackfillStats:
    """Counts for one run. Every image lands in exactly one outcome bucket."""

    examined: int = 0
    with_faces: int = 0
    provisional_empty: int = 0  # gated out, re-examinable (uncalibrated)
    barren: int = 0  # tombstoned (calibrated only)
    fetch_failed: int = 0
    skipped: int = 0  # robots, screening, non-200
    decode_failed: int = 0
    faces_indexed: int = 0
    embedded: int = 0
    embed_failed: int = 0
    rejects: dict[str, int] = field(default_factory=dict)

    def add_rejects(self, r) -> None:
        for k, v in r.__dict__.items():
            if v:
                self.rejects[k] = self.rejects.get(k, 0) + v

    def report(self) -> str:
        seen = (
            self.with_faces
            + self.provisional_empty
            + self.barren
            + self.fetch_failed
            + self.skipped
            + self.decode_failed
        )
        out = [
            "",
            "  backfill",
            f"    images processed                {seen:>8}",
            f"      with faces                    {self.with_faces:>8}",
            f"      empty, re-examinable (-2)     {self.provisional_empty:>8}",
            f"      barren, tombstoned (0)        {self.barren:>8}",
            f"      fetch failed (left at -1)     {self.fetch_failed:>8}",
            f"      skipped                       {self.skipped:>8}",
            f"      undecodable                   {self.decode_failed:>8}",
            f"    faces indexed                   {self.faces_indexed:>8}",
            f"    images embedded (scene+text)    {self.embedded:>8}",
            f"      embed failed                  {self.embed_failed:>8}",
        ]
        if self.rejects:
            out.append("")
            out.append("  detections rejected by the quality gate")
            # Not decoration. If too_small dominates you are crawling thumbnail
            # galleries; if too_blurry dominates, the vertical cannot support
            # face search at all. Both are findings you want now, not in month 4.
            for k, v in sorted(self.rejects.items(), key=lambda kv: -kv[1]):
                out.append(f"    {k:<32}{v:>8}")
        out.append("")
        return "\n".join(out)


class Backfill:
    def __init__(
        self,
        writer: PostgresWriter,
        vectors: VectorStore,
        extractor: FaceExtractor,
        fetcher: Fetcher,
        *,
        crop_root: Path,
        calibrated: bool = False,
        batch_size: int = 200,
        embedder=None,
        image_vectors=None,
    ) -> None:
        self._w = writer
        self._v = vectors
        self._ex = extractor
        self._f = fetcher
        self._crop_root = crop_root
        self._calibrated = calibrated
        self._batch = batch_size
        self._stop = False
        self.stats = BackfillStats()

        # Whole-image embedding, ADR-005. Optional so a faces-only run still
        # works and so the 13 tests written before ADR-005 stay meaningful.
        self._embedder = embedder
        self._iv = image_vectors
        self._do_embed = embedder is not None and image_vectors is not None

        # Faces are ALWAYS a job this runner can do; embedding only if wired.
        # This matters for the queue: including an image whose sole outstanding
        # job is one we cannot perform means fetching it, doing nothing, and
        # fetching it again next run, forever.
        self._q = {"faces": True, "embed": self._do_embed}

    def stop(self) -> None:
        """Ctrl-C. Finishes the image in flight, then exits cleanly."""
        log.info("backfill.stopping")
        self._stop = True

    # ------------------------------------------------------------------ one

    async def _process(self, item) -> None:
        """Fetch, detect, store one image. Never raises for expected failures."""
        import cv2
        import numpy as np

        try:
            fetched = await self._f.get_image(item.url)
        except Skipped as s:
            self.stats.skipped += 1
            log.debug("backfill.skipped", image_id=item.image_id, reason=str(s))
            return
        except Exception as exc:
            # Broad on purpose. FetchError is the expected case, but a 20,000-image
            # drain will find malformed URLs, TLS oddities and hosts that answer
            # in ways httpx did not anticipate, and none of those are worth
            # ending the run over. The image is counted and named, not swallowed.
            #
            # Left at -1 on purpose: a transient 503 should be retried on the
            # next run. The cost is that a permanently dead link is retried
            # forever, which is the TTL reaper's job in plan-004, not this one.
            self.stats.fetch_failed += 1
            log.warning(
                "backfill.fetch_failed",
                image_id=item.image_id,
                url=item.url,
                error=f"{type(exc).__name__}: {exc}",
            )
            return

        img = cv2.imdecode(np.frombuffer(fetched.body, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            self.stats.decode_failed += 1
            log.warning("backfill.undecodable", image_id=item.image_id, url=item.url)
            return

        # Whole-image embedding first, from the same bytes. Isolated exactly as
        # the crawl-loop sink is: an embedding failure must not cost the face
        # work that this fetch also paid for.
        if self._do_embed and getattr(item, "needs_embed", False):
            self._embed_one(item, fetched.body)

        if not getattr(item, "needs_faces", True):
            return

        faces, rejects = self._ex.extract(img)
        self.stats.add_rejects(rejects)

        if not faces:
            # NOT a tombstone while uncalibrated -- mark_examined decides, and
            # its default is the recoverable one. See ADR-004.
            self._w.mark_examined(item.image_id, 0, calibrated=self._calibrated)
            if self._calibrated:
                self.stats.barren += 1
            else:
                self.stats.provisional_empty += 1
            return

        # 1. Crops to disk first. They are the only artifact not covered by the
        #    two-store commit, and an orphaned crop file is harmless -- an
        #    orphaned face ROW pointing at a missing crop is not.
        records = []
        for face in faces:
            crop_path = self._ex.write_crop(face, self._crop_root)
            records.append(_to_record(face, crop_path))

        # 2. Postgres face rows. face_count still -1.
        face_ids = self._w.record_faces(item.image_id, records)

        # 3. Qdrant, cleared first so a reprocess cannot orphan vectors.
        self._v.delete_image(item.image_id)
        self._v.upsert(
            [
                VectorRecord(
                    qdrant_id=rec.qdrant_id,
                    embedding=face.embedding,
                    image_id=item.image_id,
                    face_id=fid,
                    quality=face.quality,
                )
                for face, rec, fid in zip(faces, records, face_ids, strict=True)
            ]
        )

        # 4. Commit the pair.
        self._w.mark_examined(item.image_id, len(faces), calibrated=self._calibrated)
        self.stats.with_faces += 1
        self.stats.faces_indexed += len(faces)

    def _embed_one(self, item, raw: bytes) -> None:
        """Scene + text vectors for one image. Never raises.

        Batching would be faster on paper, but this path is fetch-bound at 1 rps
        and the GPU does 179 img/s -- buffering would add a partial-batch-on-
        crash failure mode to save nothing. The crawl-loop sink batches because
        it is already holding images anyway; here there is nothing to gain.
        """
        import io

        from PIL import Image

        try:
            pil = Image.open(io.BytesIO(raw)).convert("RGB")
            (vec,) = self._embedder.embed_images([pil])
            # Vectors, then the commit marker. The point id is the image id, so
            # a crash between them leaves the image queued and a re-run
            # overwrites rather than duplicating.
            self._iv.upsert_image(item.image_id, {"scene": vec.scene, "text": vec.text})
            self._w.mark_embedded(item.image_id)
            self.stats.embedded += 1
        except Exception as exc:
            self.stats.embed_failed += 1
            log.warning(
                "backfill.embed_failed",
                image_id=item.image_id,
                error=f"{type(exc).__name__}: {exc}",
            )

    # ------------------------------------------------------------------ run

    async def run(self, limit: int | None = None) -> BackfillStats:
        self._v.ensure_collection()
        if problems := self._v.verify():
            raise RuntimeError(f"qdrant collection disagrees with config: {problems}")

        pending = self._w.pending_count(**self._q)
        log.info(
            "backfill.start",
            pending=pending,
            embedding=self._do_embed,
            limit=limit,
            calibrated=self._calibrated,
            crop_root=str(self._crop_root),
        )
        if not self._calibrated:
            log.info(
                "backfill.provisional",
                detail="thresholds are UNCALIBRATED; empty results record as -2 "
                "(re-examinable), never 0 (tombstone). See ADR-004.",
            )

        # Keyset cursor. The cursor advances even when an image FAILS -- leaving
        # it parked would mean one unreachable URL blocks the entire queue
        # forever, since a failed image keeps face_count = -1 and would be
        # handed back by the very next query.
        after_id = 0
        while not self._stop:
            batch = self._w.pending_images(limit=self._batch, after_id=after_id, **self._q)
            if not batch:
                break

            for item in batch:
                if self._stop:
                    break
                if limit is not None and self.stats.examined >= limit:
                    self._stop = True
                    break

                after_id = item.image_id
                await self._process(item)
                self.stats.examined += 1

                if self.stats.examined % 25 == 0:
                    log.info(
                        "backfill.progress",
                        examined=self.stats.examined,
                        faces=self.stats.faces_indexed,
                        embedded=self.stats.embedded,
                        with_faces=self.stats.with_faces,
                        empty=self.stats.provisional_empty + self.stats.barren,
                        failed=self.stats.fetch_failed,
                        requests=self._f.requests_made,
                    )

        log.info(
            "backfill.done", **{k: v for k, v in self.stats.__dict__.items() if k != "rejects"}
        )
        return self.stats


def _to_record(face, crop_path: str):
    """Face (with pixels) -> FaceRecord (with a path).

    Landmarks are flattened [x,y]*5 and stay at ORIGINAL resolution. Do not
    scale them to the 128px crop -- non-negotiable #3, and the scale factor is
    not stored anywhere, so it is unrecoverable after the fact.
    """
    from arc_search.index.store import FaceRecord

    return FaceRecord(
        qdrant_id=face.qdrant_id,
        bbox=face.bbox,
        landmarks=[float(v) for v in face.landmarks.reshape(-1)],
        src_width=face.src_width,
        src_height=face.src_height,
        det_score=face.det_score,
        blur_var=face.blur_var,
        yaw=face.yaw,
        quality=face.quality,
        age_est=face.age_est,
        crop_path=crop_path,
    )


async def main(argv: list[str] | None = None) -> int:
    import httpx

    from arc_search.crawler.politeness import Politeness

    ap = argparse.ArgumentParser(prog="arc_search.index.backfill")
    ap.add_argument("--limit", type=int, default=None, help="stop after N images")
    ap.add_argument(
        "--rps",
        type=float,
        default=None,
        help="LOWER the per-host rate. Use this if a crawl is running against "
        "the same host -- politeness is per-process, so both add up.",
    )
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--crop-dir", type=Path, default=None, help="default: face.crop_dir")
    ap.add_argument("--dry-run", action="store_true", help="report the queue and exit")
    ap.add_argument(
        "--no-embed",
        action="store_true",
        help="faces only; skip whole-image scene+text embedding. Images then "
        "stay at embed_state=-1 and are NOT put on this run's queue, so they "
        "are not re-fetched for a job this run cannot do.",
    )
    args = ap.parse_args(argv)

    structlog.configure(processors=[structlog.dev.ConsoleRenderer()])

    crawl_cfg = CrawlSettings()
    if args.rps is not None:
        if args.rps > crawl_cfg.per_host_rps:
            log.warning(
                "backfill.rps_ignored",
                requested=args.rps,
                global_rps=crawl_cfg.per_host_rps,
                hint="--rps may only LOWER the global rate, same rule as seeds.yaml",
            )
        else:
            crawl_cfg = crawl_cfg.model_copy(update={"per_host_rps": args.rps})

    face_cfg = FaceSettings()
    index_cfg = IndexSettings()
    search_cfg = SearchSettings()
    crop_root = args.crop_dir or face_cfg.crop_dir

    writer = PostgresWriter(index_cfg.pg_dsn)
    try:
        counts = writer.face_counts()
        embed_counts = writer.embed_counts()
        log.info("backfill.queue", **counts, **{f"embed_{k}": v for k, v in embed_counts.items()})
        if args.dry_run:
            print(f"  faces pending: {counts['unexamined']}  provisional: {counts['provisional']}")
            print(
                f"  embed pending: {embed_counts['unembedded']}  "
                f"embedded: {embed_counts['embedded']}"
            )
            return 0

        log.warning(
            "backfill.politeness",
            effective_rps=crawl_cfg.per_host_rps,
            detail="politeness is per-process. If crawler.run is fetching the "
            "same host right now, that host sees BOTH. Stop it or use --rps.",
        )

        vectors = VectorStore(index_cfg)
        extractor = FaceExtractor(face_cfg)

        # Whole-image embedding shares the fetch with the face pass -- the
        # expensive resource is the politeness budget, not the GPU.
        embedder = image_vectors = None
        if not args.no_embed:
            try:
                from arc_search.index.embed import ImageEmbedder
                from arc_search.index.vectors import VectorStore as VS

                image_vectors = VS(index_cfg, spec=index_cfg.image_spec())
                image_vectors.ensure_collection()
                embedder = ImageEmbedder()
                scene_dim, text_dim = embedder.dims()
                want = dict(index_cfg.image_spec().named)
                if (scene_dim, text_dim) != (want["scene"], want["text"]):
                    raise RuntimeError(
                        f"model dims (scene={scene_dim}, text={text_dim}) do not match "
                        f"collection {image_vectors.name} ({want})"
                    )
                log.info("backfill.embedding_ready", device=embedder.effective_device())
            except Exception as exc:
                # Degrade to faces-only rather than refusing to run. The face
                # queue is still drainable and the images stay queued.
                log.warning(
                    "backfill.embedding_disabled",
                    error=f"{type(exc).__name__}: {exc}",
                    detail="running faces-only; images stay at embed_state=-1",
                )
                embedder = image_vectors = None

        limits = httpx.Limits(max_connections=8, max_keepalive_connections=4)
        async with httpx.AsyncClient(
            http2=True, follow_redirects=True, limits=limits, timeout=crawl_cfg.timeout_s
        ) as client:
            fetcher = Fetcher(crawl_cfg, client, Politeness(crawl_cfg, client))
            job = Backfill(
                writer,
                vectors,
                extractor,
                fetcher,
                crop_root=crop_root,
                calibrated=search_cfg.calibrated,
                batch_size=args.batch_size,
                embedder=embedder,
                image_vectors=image_vectors,
            )

            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                with contextlib.suppress(NotImplementedError):
                    loop.add_signal_handler(sig, job.stop)

            stats = await job.run(limit=args.limit)

        print(stats.report())
        print("  corpus")
        for k, v in {
            **writer.face_counts(),
            **{f"embed_{k}": v for k, v in writer.embed_counts().items()},
        }.items():
            print(f"    {k:<32}{v:>8}")
        print(f"    {'vectors in qdrant':<32}{vectors.count():>8}")
        print()
    finally:
        writer.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(asyncio.run(main()))
