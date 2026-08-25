"""Embed whole images as the crawler passes them through. plan-005 Phase 2b.

WHY IN THE CRAWL LOOP
---------------------
Because the bytes are only here once. Non-negotiable #1 means nothing persists a
scene image, so anything that wants to embed one either does it while the
crawler is holding it or re-fetches it later -- and re-fetching every image at
politeness rates is **347 days** at the 30M scale target. plan-005 works that
arithmetic through.

The alternative considered and rejected was a separate embed worker fed by a
spool directory. It would decouple crawl uptime from GPU health, which is a real
benefit, but it requires writing full-scene images to disk. Transiently, and
deleted afterwards -- which is exactly the reasoning by which non-negotiable #1
would stop meaning anything. The coupling is the cheaper price.

WHY IT IS A DECORATOR
---------------------
``Crawler`` calls ``sink.handle(fetched, ctx)`` and drops the bytes on the next
line. Wrapping the sink rather than editing that loop means the crawl tier does
not learn about models at all, and means this whole tier can be removed by not
wrapping. If the coupling turns out to be the wrong call, the undo is one line
in ``run.py``.

🔴 FAILURE IS ISOLATED, DELIBERATELY
------------------------------------
A crawl is a five-hour job against someone else's server, and the frontier is
the expensive thing. An OOM, a CUDA fault, or one undecodable image must not
cost that. So:

  * every embedding failure is caught, counted, and logged
  * a failed image keeps ``embed_state = -1`` and is left for the backfill
  * if the models will not load at all, the crawl runs exactly as it did before

The crawl is never worse off for having tried. That is what makes putting a GPU
in the loop defensible rather than reckless.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import structlog

from arc_search.config import EmbedSettings, IndexSettings
from arc_search.index.dedup import sha1_bytes

if TYPE_CHECKING:  # pragma: no cover
    from arc_search.crawler.fetch import Fetched
    from arc_search.crawler.run import ImageContext

log = structlog.get_logger(__name__)


class EmbeddingSink:
    """Wraps a ``CrawlSink``, embedding new images on their way past.

    Batched because the GPU wants it -- 70 img/s at batch 1 against 179 at
    batch 7 -- and the crawl yields one image at a time. The buffer holds
    decoded images, not raw bytes, and nothing it holds reaches disk.
    """

    def __init__(
        self,
        inner,
        writer,
        vectors=None,
        embedder=None,
        cfg: EmbedSettings | None = None,
        index_cfg: IndexSettings | None = None,
    ) -> None:
        self._inner = inner
        self._writer = writer
        self._cfg = cfg or EmbedSettings()

        if vectors is None:
            from arc_search.index.vectors import VectorStore

            icfg = index_cfg or IndexSettings()
            vectors = VectorStore(icfg, spec=icfg.image_spec())
        self._vectors = vectors

        if embedder is None:
            from arc_search.index.embed import ImageEmbedder

            embedder = ImageEmbedder(self._cfg)
        self._embedder = embedder

        self._buf: list[tuple[int, object]] = []
        self.embedded = 0
        self.failed = 0
        self.batches = 0
        self._degraded = False

    # -- CrawlSink protocol ------------------------------------------------

    def record_page(self, url: str, title: str | None, status: int) -> int:
        return self._inner.record_page(url, title, status)

    def handle(self, fetched: Fetched, context: ImageContext) -> str:
        """Record via the wrapped sink, then queue the image for embedding.

        Only ``new`` images are queued. A duplicate's bytes are already
        represented by an existing row, and anything crawled before this tier
        existed is the backfill's job -- checking ``embed_state`` per image
        would add a query to a path that does not need one.
        """
        verdict = self._inner.handle(fetched, context)

        if verdict != "new" or self._degraded:
            return verdict

        try:
            image_id = self._writer.image_id_for_sha1(sha1_bytes(fetched.body))
            if image_id is not None:
                from PIL import Image

                # Decode now and keep the decoded image, not the bytes. Nothing
                # here is written anywhere; the buffer is memory only.
                img = Image.open(io.BytesIO(fetched.body)).convert("RGB")
                self._buf.append((image_id, img))
                if len(self._buf) >= self._cfg.batch_size:
                    self.flush()
        except Exception as exc:
            self.failed += 1
            log.warning(
                "embed.queue_failed",
                url=fetched.final_url,
                error=f"{type(exc).__name__}: {exc}",
            )

        return verdict

    # -- embedding ---------------------------------------------------------

    def flush(self) -> None:
        """Embed and store whatever is buffered. Never raises.

        Public because the tail matters and callers other than ``close()`` want
        it -- a long-running crawl that pauses, and tests that need the buffer
        drained without tearing down the database connection underneath them.
        """
        if not self._buf:
            return
        batch, self._buf = self._buf, []

        try:
            vectors = self._embedder.embed_images([img for _, img in batch])
            for (image_id, _), vec in zip(batch, vectors, strict=True):
                # Vectors first, then the commit marker -- the same ordering the
                # face tier uses. The point id is the image id, so a re-run
                # overwrites rather than duplicating and a crash here just
                # leaves the image on the queue.
                self._vectors.upsert_image(image_id, {"scene": vec.scene, "text": vec.text})
                self._writer.mark_embedded(image_id)
            self.embedded += len(batch)
            self.batches += 1
        except Exception as exc:
            self.failed += len(batch)
            log.warning(
                "embed.batch_failed",
                images=len(batch),
                error=f"{type(exc).__name__}: {exc}",
                detail="images stay at embed_state=-1 for the backfill; the crawl continues.",
            )

    def prepare(self) -> bool:
        """Create the collection and load the models. Returns False if degraded.

        Called once at startup so a broken model stack fails *here*, loudly,
        rather than on the first image an hour into a crawl. Returning False
        rather than raising is the point: the crawl should still run.
        """
        try:
            self._vectors.ensure_collection()
            if problems := self._vectors.verify():
                raise RuntimeError(f"images collection disagrees with config: {problems}")
            scene_dim, text_dim = self._embedder.dims()
            want = dict(self._vectors.spec.named)
            if (scene_dim, text_dim) != (want["scene"], want["text"]):
                # A model swap that silently disagrees with the collection would
                # write vectors nobody can search. DINOv3 ViT-L is 1024-d.
                raise RuntimeError(
                    f"model dims (scene={scene_dim}, text={text_dim}) do not match "
                    f"collection {self._vectors.name} ({want}). Rebuild the "
                    f"collection or set ARC_INDEX_SCENE_DIM/TEXT_DIM."
                )
            log.info(
                "embed.sink_ready",
                collection=self._vectors.name,
                device=self._embedder.effective_device(),
                batch_size=self._cfg.batch_size,
            )
            return True
        except Exception as exc:
            self._degraded = True
            log.warning(
                "embed.disabled",
                error=f"{type(exc).__name__}: {exc}",
                detail="whole-image embedding is OFF for this run; the crawl "
                "continues and images stay at embed_state=-1 for the backfill.",
            )
            return False

    # -- lifecycle ---------------------------------------------------------

    def counts(self):
        return self._inner.counts() if hasattr(self._inner, "counts") else None

    def close(self) -> None:
        """Flush the tail before shutting down.

        Without this, up to batch_size-1 images per run are fetched, recorded,
        and never embedded -- a slow leak that only shows up as a work queue
        that never quite empties.
        """
        if not self._degraded:
            self.flush()
        log.info(
            "embed.sink_closed",
            embedded=self.embedded,
            failed=self.failed,
            batches=self.batches,
        )
        self._inner.close()
