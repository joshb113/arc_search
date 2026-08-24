"""The crawl loop: frontier <-> politeness <-> fetch <-> extract.

Week 1 runs this with no model in the path. That is the point -- prove
discovery, scope, politeness, dedup, and crash recovery while every failure is
still cheap and legible. A GPU in the loop makes every bug look like a model
bug.

TWO FRONTIERS
-------------
Pages and images are queued separately, in two ``Frontier`` instances over two
SQLite files. It would have been easy to add a ``kind`` column instead, but two
instances reuse the already-tested lease/complete/fail/recover logic verbatim
and keep the depth semantics honest: an image sits at its page's depth + 1 and
never spawns children. It also means you can throw away the image queue and
recrawl images alone without losing page-level progress.

BYTES ARE NEVER WRITTEN TO DISK
-------------------------------
Non-negotiable #1: no full-scene images, ever. So the loop hands each image's
bytes to an ``ImageSink`` and then drops them. In week 1 the sink is
``MetadataSink``, which hashes and records and keeps nothing. In week 2 the same
call site gets ``FaceIndexSink``, which detects, crops to 128px, and keeps only
the crop. The bytes live in memory for the length of one function call in both
cases. There is deliberately no "spool to disk and process later" mode; that is
how you wake up with 4 TB of other people's photographs.

TERMINATION
-----------
The loop is done when both frontiers are empty *and* no worker is mid-request.
Checking only the queues races: a worker holding the last page is about to
enqueue thirty more. Hence the in-flight counter.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import re
import signal
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit

import httpx
import structlog

from arc_search.config import CrawlSettings, unknown_settings
from arc_search.crawler.extract import FoundImage, extract_images, extract_links, page_title
from arc_search.crawler.fetch import Fetched, Fetcher, FetchError, Skipped
from arc_search.crawler.frontier import Frontier
from arc_search.crawler.politeness import Politeness
from arc_search.crawler.seeds import SeedConfig, Vertical, load_seeds
from arc_search.index.dedup import Deduper, sha1_bytes

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Sinks
# ---------------------------------------------------------------------------


class CrawlSink(Protocol):
    """Where crawl results go. The image bytes are not valid after return."""

    def record_page(self, url: str, title: str | None, status: int) -> object:
        """Called for EVERY fetched page, including ones with no images.

        Recording only pages that yielded images would under-report the crawl
        and leave the recrawl reaper with nothing to sweep.
        """

    def handle(self, fetched: Fetched, context: ImageContext) -> str:
        """Return a short verdict string for the stats counter."""

    def close(self) -> None: ...


# Retained name: the old protocol was image-only.
ImageSink = CrawlSink


@dataclass(frozen=True)
class ImageContext:
    """Provenance for a downloaded image. Everything here is cheap metadata."""

    page_url: str
    page_title: str | None
    vertical: str
    depth: int
    alt: str | None
    extractor: str
    width_hint: int | None


class MetadataSink:
    """Week 1 sink: SHA1, record a JSONL row, discard the bytes.

    The alt text is carried through deliberately. On FOSDEM speaker pages it
    reads ``Photo of <Name>``, which is a weak label we get for free and which
    ``eval/calibrate`` will need a supply of. Throwing it away at crawl time and
    trying to recover it later means a full recrawl.
    """

    def __init__(self, path: Path, deduper: Deduper) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("a", encoding="utf-8")
        self._dedup = deduper
        self._next_id = 0

    def record_page(self, url: str, title: str | None, status: int) -> None:
        """No-op. JSONL keeps page context on the image rows instead.

        This sink does not attempt page-level bookkeeping; use PostgresWriter
        for a run whose page table you intend to keep.
        """

    def handle(self, fetched: Fetched, context: ImageContext) -> str:
        if (hit := self._dedup.check_bytes(fetched.body)) is not None:
            return str(hit.verdict)

        self._next_id += 1
        digest = sha1_bytes(fetched.body)
        # face_count=0 would mark it barren; -1 means "not yet examined".
        self._dedup.register(digest, None, self._next_id, face_count=-1)

        self._fh.write(
            json.dumps(
                {
                    "id": self._next_id,
                    "url": fetched.final_url,
                    "sha1": digest.hex(),
                    "bytes": len(fetched.body),
                    "content_type": fetched.content_type,
                    "width": fetched.width,
                    "height": fetched.height,
                    "page_url": context.page_url,
                    "page_title": context.page_title,
                    "vertical": context.vertical,
                    "depth": context.depth,
                    "alt": context.alt,
                    "extractor": context.extractor,
                    "width_hint": context.width_hint,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        return "new"

    def close(self) -> None:
        self._fh.close()


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class CrawlStats:
    started: float = field(default_factory=time.monotonic)
    pages_fetched: int = 0
    pages_failed: int = 0
    images_seen: int = 0
    images_fetched: int = 0
    robots_blocked: int = 0
    by_host: Counter[str] = field(default_factory=Counter)
    by_extractor: Counter[str] = field(default_factory=Counter)
    skips: Counter[str] = field(default_factory=Counter)
    verdicts: Counter[str] = field(default_factory=Counter)

    def note_skip(self, reason: str) -> None:
        # Reasons carry a payload after ':' (sizes, content types). Bucket on
        # the prefix so the report stays readable at 100k URLs.
        self.skips[reason.split(":", 1)[0]] += 1
        if reason.startswith("robots"):
            self.robots_blocked += 1

    def report(self) -> str:
        elapsed = max(1e-6, time.monotonic() - self.started)
        considered = self.images_fetched + self.robots_blocked
        rate = self.robots_blocked / considered if considered else 0.0
        lines = [
            "",
            "=" * 62,
            f"  elapsed            {elapsed / 60:.1f} min",
            f"  pages fetched      {self.pages_fetched}  ({self.pages_fetched / elapsed:.1f}/s)",
            f"  pages failed       {self.pages_failed}",
            f"  images discovered  {self.images_seen}",
            f"  images fetched     {self.images_fetched}",
            f"  robots exclusion   {rate:.2%}",
            "",
            "  images per host",
        ]
        lines += [f"    {h:<38} {n:>8}" for h, n in self.by_host.most_common(20)]
        lines += ["", "  extraction source"]
        lines += [f"    {s:<38} {n:>8}" for s, n in self.by_extractor.most_common()]
        lines += ["", "  dedup verdicts"]
        lines += [f"    {v:<38} {n:>8}" for v, n in self.verdicts.most_common()]
        lines += ["", "  skips"]
        lines += [f"    {r:<38} {n:>8}" for r, n in self.skips.most_common(20)]
        lines += ["=" * 62, ""]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Crawler
# ---------------------------------------------------------------------------


class Crawler:
    def __init__(
        self,
        cfg: CrawlSettings,
        seeds: SeedConfig,
        fetcher: Fetcher,
        pages: Frontier,
        images: Frontier,
        sink: ImageSink,
    ) -> None:
        self._cfg = cfg
        self._seeds = seeds
        self._fetch = fetcher
        self._pages = pages
        self._images = images
        self._sink = sink
        self.stats = CrawlStats()

        self._inflight = 0
        self._stopping = False
        self._page_budget: Counter[str] = Counter()
        # Verticals that have spent their per-run page budget. Page workers
        # stop leasing once every active vertical is in here.
        self._exhausted: set[str] = set()
        # Image URL -> the page context that found it. Populated at discovery,
        # consumed when the image is dequeued. Bounded by the image queue depth.
        self._ctx: dict[str, ImageContext] = {}

    # -- terminal states ---------------------------------------------------

    def _settle(self, frontier: Frontier, url: str, err: FetchError) -> None:
        """Record the terminal state of a failed URL.

        The subtlety that bit me writing this: ``Frontier.fail`` ALREADY sets
        the row to FAILED once retries are exhausted. Calling ``complete()``
        afterwards silently overwrites that with DONE, and the failed count
        reads zero forever -- a crawl losing a tenth of its images to a dead
        host reports a clean run. ``fail()`` owns the row; do not second-guess
        it.

        A non-retryable error (404, 410, 403) is different: there is nothing to
        retry and nothing wrong with the crawl, so the URL is simply done.
        """
        if err.retryable:
            requeued = frontier.fail(url, self._cfg.max_retries)
            log.debug("url.failed", url=url, requeued=requeued, reason=err.reason)
        else:
            frontier.complete(url)

    # -- scope -------------------------------------------------------------

    def vertical_for(self, url: str) -> Vertical | None:
        host = (urlsplit(url).hostname or "").lower()
        if not host:
            return None
        for v in self._seeds.active:
            if v.host_allowed(host):
                return v
        return None

    def seed(self) -> int:
        added = 0
        for v in self._seeds.active:
            for url in v.seeds:
                if self._pages.add(url, depth=0, max_depth=v.max_depth):
                    added += 1
        log.info("crawl.seeded", urls=added, verticals=[v.name for v in self._seeds.active])
        return added

    # -- page worker -------------------------------------------------------

    async def _do_page(self, url: str, depth: int) -> None:
        vertical = self.vertical_for(url)
        if vertical is None:
            self.stats.note_skip("off_vertical")
            self._pages.complete(url)
            return

        if self._page_budget[vertical.name] >= vertical.max_pages:
            # max_pages is a per-RUN cap, not a verdict on this URL. Release it
            # so the next run can claim it; completing it here would burn the
            # frontier permanently. Mark the vertical exhausted so the page
            # workers wind down instead of spinning on a queue they may not
            # touch.
            self.stats.note_skip("vertical_budget")
            self._exhausted.add(vertical.name)
            self._pages.release(url)
            return

        try:
            page = await self._fetch.get_page(url)
        except Skipped as s:
            self.stats.note_skip(s.reason)
            self._pages.complete(url)
            return
        except FetchError as e:
            self.stats.pages_failed += 1
            self.stats.note_skip(e.reason)
            self._settle(self._pages, url, e)
            return

        self._page_budget[vertical.name] += 1
        self.stats.pages_fetched += 1

        html = page.text
        base = page.final_url
        title = page_title(html)
        self._sink.record_page(base, title, page.status)

        # Links, at depth + 1. Frontier.add records over-depth URLs as DONE
        # rather than dropping them, so rediscovery is free.
        if depth < vertical.max_depth:
            for link in extract_links(html, base):
                ok, reason = self._seeds.in_scope(link, vertical)
                if not ok:
                    self.stats.note_skip(reason)
                    continue
                self._pages.add(link, depth + 1, vertical.max_depth)

        # Images, also at depth + 1, into the other frontier.
        for img in extract_images(html, base):
            self._queue_image(img, vertical, base, title, depth + 1)

        self._pages.complete(url)

    def _queue_image(
        self,
        img: FoundImage,
        vertical: Vertical,
        page_url: str,
        title: str | None,
        depth: int,
    ) -> None:
        ok, reason = self._seeds.in_scope(img.url, vertical)
        if not ok:
            self.stats.note_skip(reason)
            return
        # max_depth here is intentionally huge: an image is a leaf, so the depth
        # ceiling that governs link-following must not also silently drop the
        # images found on the deepest legitimate page.
        if self._images.add(img.url, depth, max_depth=10**6):
            self.stats.images_seen += 1
            self.stats.by_extractor[img.source] += 1
            self._ctx[img.url] = ImageContext(
                page_url=page_url,
                page_title=title,
                vertical=vertical.name,
                depth=depth,
                alt=img.alt,
                extractor=img.source,
                width_hint=img.width_hint,
            )

    # -- image worker ------------------------------------------------------

    async def _do_image(self, url: str, depth: int) -> None:
        ctx = self._ctx.pop(
            url,
            # A restart loses the in-memory map; the queue survives. Record what
            # we still know rather than dropping a legitimately queued image.
            ImageContext(
                page_url="",
                page_title=None,
                vertical=(self.vertical_for(url) or Vertical("?", [])).name,
                depth=depth,
                alt=None,
                extractor="resumed",
                width_hint=None,
            ),
        )
        try:
            got = await self._fetch.get_image(url)
        except Skipped as s:
            self.stats.note_skip(s.reason)
            self._images.complete(url)
            return
        except FetchError as e:
            self.stats.note_skip(e.reason)
            self._settle(self._images, url, e)
            return

        self.stats.images_fetched += 1
        self.stats.by_host[urlsplit(got.final_url).hostname or "?"] += 1
        verdict = self._sink.handle(got, ctx)
        self.stats.verdicts[verdict] += 1
        self._images.complete(url)
        # Bytes go out of scope here and are never written anywhere. See the
        # module docstring.

    # -- driver ------------------------------------------------------------

    def stop(self) -> None:
        self._stopping = True

    def pages_exhausted(self) -> bool:
        """True once every active vertical has spent its per-run page budget.

        Budget-skipped URLs are released back to PENDING rather than completed,
        so the page queue never drains on a budgeted run. Without this the
        workers would spin on it forever and the loop would never terminate.
        """
        active = self._seeds.active
        return bool(active) and all(v.name in self._exhausted for v in active)

    def _idle(self) -> bool:
        # Pages left behind by an exhausted budget do not count as work
        # remaining -- they are deliberately deferred to the next run.
        pages_left = 0 if self.pages_exhausted() else self._pages.stats()["pending"]
        return pages_left == 0 and self._images.stats()["pending"] == 0 and self._inflight == 0

    async def _worker(self, frontier: Frontier, handler, tag: str, stop_when=None) -> None:
        while not self._stopping:
            if stop_when is not None and stop_when():
                return
            leased = frontier.lease(1)
            if not leased:
                if self._idle():
                    return
                await asyncio.sleep(0.25)
                continue
            task = leased[0]
            self._inflight += 1
            try:
                await handler(task.url, task.depth)
            except Exception:
                # Deliberately broad: one malformed page must not take a worker
                # down and strand the rest of the queue behind it.
                log.exception("worker.crashed", tag=tag, url=task.url)
                frontier.fail(task.url, self._cfg.max_retries)
            finally:
                self._inflight -= 1

    async def run(self) -> CrawlStats:
        recovered = self._pages.recover_inflight() + self._images.recover_inflight()
        if recovered:
            log.info("crawl.recovered_inflight", urls=recovered)
        self.seed()

        # Split concurrency between the two queues. Images outnumber pages by
        # roughly an order of magnitude, so weight accordingly.
        n_page = max(1, self._cfg.concurrency // 4)
        n_img = max(1, self._cfg.concurrency - n_page)

        workers = [
            asyncio.create_task(
                self._worker(self._pages, self._do_page, "page", stop_when=self.pages_exhausted)
            )
            for _ in range(n_page)
        ] + [
            asyncio.create_task(self._worker(self._images, self._do_image, "image"))
            for _ in range(n_img)
        ]
        heartbeat = asyncio.create_task(self._heartbeat())
        try:
            await asyncio.gather(*workers)
        finally:
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat
        return self.stats

    async def _heartbeat(self, every: float = 30.0) -> None:
        while True:
            await asyncio.sleep(every)
            log.info(
                "crawl.progress",
                pages=self.stats.pages_fetched,
                images=self.stats.images_fetched,
                discovered=self.stats.images_seen,
                pending_pages=self._pages.stats()["pending"],
                pending_images=self._images.stats()["pending"],
            )


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

# Tokens that mean somebody copied the template and stopped. Checked
# case-insensitively.
_PLACEHOLDERS = ("yourname", "your_email_here", "your-email", "example.com", "changeme", "todo")

_CONTACT = re.compile(r"https?://\S+|[^\s@()]+@[^\s@().]+\.[a-z]{2,}", re.I)

LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0"})


def all_loopback(seeds: SeedConfig) -> bool:
    """True if every enabled seed points at this machine.

    This is the one honest bypass. Crawling your own fixture server is not an
    act that anyone needs to be able to identify or complain about, and forcing
    a contact URL to smoke-test the loop just teaches you to set a fake one --
    which is strictly worse than the check not existing.
    """
    hosts = {(urlsplit(s).hostname or "").lower() for v in seeds.active for s in v.seeds} | {
        h.lower() for v in seeds.active for h in v.allow_hosts
    }
    return bool(hosts) and hosts <= LOOPBACK_HOSTS


def check_user_agent(ua: str, loopback_only: bool = False) -> str | None:
    """Return an error message if this UA must not go on the open internet.

    Non-negotiable #6: one User-Agent, it names the project, and it carries a
    contact route. eye_of_web rotated 2010-era Opera strings and kept a
    Googlebot-Image constant on hand.

    A URL is a complete contact route by itself -- an email is not required, and
    demanding one just pushes people toward a fake one. What is required is that
    a sysadmin looking at their access log at 3am can find out what hit them and
    tell someone to stop.
    """
    if loopback_only:
        return None

    lowered = ua.lower()
    if found := [p for p in _PLACEHOLDERS if p in lowered]:
        return (
            f"refusing to crawl: User-Agent still contains the placeholder(s) {found}.\n"
            f"  current: {ua}\n"
            "Set ARC_CRAWL_USER_AGENT in .env. A repo URL alone is enough -- no\n"
            "email required. For example:\n"
            "  ARC_CRAWL_USER_AGENT='arc_search/0.1 (+https://github.com/you/arc_search)'\n"
            "Non-negotiable #6. To exercise the loop without publishing an identity,\n"
            "point --seeds at a loopback fixture instead; that path skips this check."
        )
    if "arc_search" not in lowered:
        return f"refusing to crawl: User-Agent must name the project.\n  current: {ua}"
    if not _CONTACT.search(ua):
        return (
            "refusing to crawl: User-Agent carries no contact route.\n"
            f"  current: {ua}\n"
            "Include a URL (preferred) or an email so an operator can reach you."
        )
    return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="arc_search.crawler.run")
    ap.add_argument("--seeds", type=Path, default=None, help="default: crawl.seeds_file")
    ap.add_argument("--only", action="append", default=[], help="restrict to named vertical(s)")
    ap.add_argument("--frontier", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("data/images.jsonl"))
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument("--max-pages", type=int, default=None, help="override every vertical")
    ap.add_argument(
        "--sink",
        choices=("postgres", "jsonl"),
        default="postgres",
        help=(
            "postgres (default): resumable, page table, dedup seeded from the "
            "image table. jsonl: no dependencies, but restarts re-record "
            "everything -- smoke tests only."
        ),
    )
    args = ap.parse_args(argv)

    structlog.configure(processors=[structlog.dev.ConsoleRenderer()])

    cfg = CrawlSettings()
    if args.concurrency:
        cfg.concurrency = args.concurrency

    # extra="ignore" is what lets one .env serve four prefixed settings groups,
    # but it also swallows typos. Say so rather than running on a silent default.
    if stray := unknown_settings():
        log.warning("config.unknown_vars", vars=stray, hint="typo? these matched no field")

    seed_path = args.seeds or cfg.seeds_file
    seeds = load_seeds(seed_path)
    if args.only:
        wanted = set(args.only)
        for v in seeds.verticals:
            v.enabled = v.enabled and v.name in wanted
    if args.max_pages:
        for v in seeds.verticals:
            v.max_pages = min(v.max_pages, args.max_pages)
    if not seeds.active:
        print(f"no enabled verticals in {seed_path}", file=sys.stderr)
        return 2

    # The identity check runs AFTER seeds load, so it can tell a real crawl from
    # a loopback smoke test. See check_user_agent.
    if problem := check_user_agent(cfg.user_agent, all_loopback(seeds)):
        print(problem, file=sys.stderr)
        return 2

    frontier_path = args.frontier or cfg.frontier_path
    pages = Frontier(frontier_path)
    images = Frontier(frontier_path.with_name(frontier_path.stem + "-images.sqlite"))

    sink: CrawlSink
    if args.sink == "postgres":
        # Imported here, not at module scope: psycopg is not needed for a jsonl
        # run and the crawl tier should not require a database driver to start.
        import psycopg

        from arc_search.config import IndexSettings
        from arc_search.index.store import PostgresWriter

        try:
            sink = PostgresWriter(IndexSettings().pg_dsn)
        except psycopg.OperationalError as exc:
            print(
                f"cannot reach Postgres: {exc}\n"
                "Start it with `docker compose up -d postgres` (needs ARC_PG_PASSWORD\n"
                "in .env), or run with --sink jsonl for a throwaway crawl.",
                file=sys.stderr,
            )
            return 2
        log.info("sink.postgres", resumed_images=sink.loaded)
    else:
        sink = MetadataSink(args.out, Deduper())
        log.warning(
            "sink.jsonl",
            hint="restarts re-record everything; use --sink postgres for a real run",
        )

    limits = httpx.Limits(max_connections=cfg.concurrency * 2, max_keepalive_connections=32)
    async with httpx.AsyncClient(
        http2=True, follow_redirects=True, limits=limits, timeout=cfg.timeout_s
    ) as client:
        fetcher = Fetcher(cfg, client, Politeness(cfg, client))
        crawler = Crawler(cfg, seeds, fetcher, pages, images, sink)

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError):  # Windows lacks SIGTERM here
                loop.add_signal_handler(sig, crawler.stop)

        try:
            stats = await crawler.run()
        finally:
            table_counts = sink.counts() if hasattr(sink, "counts") else None
            sink.close()
            pages.close()
            images.close()

    print(stats.report())
    if table_counts:
        print("  postgres row counts")
        for table, n in table_counts.items():
            print(f"    {table:<38} {n:>8}")
        print()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(asyncio.run(main()))
