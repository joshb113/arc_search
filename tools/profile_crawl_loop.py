"""Throughput profiler for the crawl loop.

    python tools/profile_crawl_loop.py

Set PROFILE_PG_DSN (a _test database) to include the Postgres sink case.

Serves a synthetic FOSDEM-shaped site on loopback so network latency is ~0,
then runs the REAL Crawler against it under different configurations. If the
loop cannot saturate its own rate limit with a zero-latency server, the
bottleneck is internal and this will show which knob moves it.

Kept because it earned it. On first run it found that robots.txt was being
fetched from the wrong port -- the URL was built from the bare hostname, so
a host on :8080 had its robots.txt requested from :80, the fetch failed,
Politeness failed closed, and the entire host was skipped as robots_disallow.
It then established that the loop saturates its rate limit exactly (1.02/s
at a 1.0 limit, 4.06/s at 4.0 with a single page worker) and that the
internal ceiling is ~114 req/s with the JSONL sink and ~56 req/s with
Postgres -- which ruled out the loop as the cause of an apparent shortfall
that turned out to be a clamped config value.
"""

from __future__ import annotations

import asyncio
import http.server
import io
import random
import socketserver
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, "src")

from PIL import Image

from arc_search.config import CrawlSettings
from arc_search.crawler.fetch import Fetcher
from arc_search.crawler.frontier import Frontier
from arc_search.crawler.politeness import Politeness
from arc_search.crawler.run import Crawler, MetadataSink
from arc_search.crawler.seeds import SeedConfig, Vertical
from arc_search.index.dedup import Deduper

N_SPEAKERS = 400


def _img(seed: int) -> bytes:
    rnd = random.Random(seed)
    im = Image.new("RGB", (240, 240))
    im.putdata([(rnd.randrange(256),) * 3 for _ in range(240 * 240)])
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return buf.getvalue()


IMAGES = {i: _img(i) for i in range(40)}
INDEX = (
    "<html><head><title>Speakers</title></head><body>"
    + "".join(f'<a href="/speaker/{i}/">s{i}</a>' for i in range(N_SPEAKERS))
    + "</body></html>"
).encode()


class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # silence
        pass

    def do_GET(self):
        p = self.path
        if p == "/robots.txt":
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if p.startswith("/speakers"):
            body, ctype = INDEX, "text/html"
        elif p.startswith("/speaker/"):
            n = int(p.strip("/").split("/")[1])
            body = (
                f"<html><head><title>s{n}</title></head><body>"
                f'<img src="/i/{n}.jpg" alt="Photo of Speaker {n}"></body></html>'
            ).encode()
            ctype = "text/html"
        elif p.startswith("/i/"):
            n = int(p.split("/")[2].split(".")[0])
            body, ctype = IMAGES[n % 40], "image/jpeg"
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def start_server() -> int:
    srv = Server(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv.server_address[1]


async def run_case(port, *, rate, concurrency, max_pages, sink_kind, label):
    import httpx

    cfg = CrawlSettings(
        respect_robots=True,
        per_host_rps=rate,
        per_host_burst=2,
        concurrency=concurrency,
        backoff_base_s=0.0,
        min_image_dim=64,
        min_image_bytes=2000,
    )
    seeds = SeedConfig(
        verticals=[
            Vertical(
                name="fx",
                enabled=True,
                seeds=[f"http://127.0.0.1:{port}/speakers/"],
                allow_hosts=["127.0.0.1"],
                max_depth=3,
                max_pages=max_pages,
            )
        ]
    )
    tmp = Path(tempfile.mkdtemp())
    pages, images = Frontier(tmp / "p.sqlite"), Frontier(tmp / "i.sqlite")

    if sink_kind == "postgres":
        import os

        from arc_search.index.store import PostgresWriter

        sink = PostgresWriter(os.environ["PROFILE_PG_DSN"])
    else:
        sink = MetadataSink(tmp / "out.jsonl", Deduper())

    limits = httpx.Limits(max_connections=concurrency * 2, max_keepalive_connections=32)
    async with httpx.AsyncClient(follow_redirects=True, limits=limits, timeout=20) as client:
        f = Fetcher(cfg, client, Politeness(cfg, client))
        c = Crawler(cfg, seeds, f, pages, images, sink)
        t0 = time.monotonic()
        st = await c.run()
        el = time.monotonic() - t0

    sink.close()
    pages.close()
    images.close()
    req = st.pages_fetched + st.images_fetched
    print(
        f"  {label:<44} {el:6.1f}s  pages={st.pages_fetched:<4} imgs={st.images_fetched:<4} "
        f"{req / el:6.2f} req/s   (limit {rate:.1f})"
    )
    return req / el


async def main():
    port = start_server()
    print(f"fixture server on 127.0.0.1:{port}, {N_SPEAKERS} speakers\n")

    print("A. does the loop saturate its rate limit at all? (jsonl sink, no DB)")
    await run_case(
        port, rate=1.0, concurrency=16, max_pages=40, sink_kind="jsonl", label="rate=1.0 conc=16"
    )
    await run_case(
        port, rate=4.0, concurrency=16, max_pages=80, sink_kind="jsonl", label="rate=4.0 conc=16"
    )

    print("\nB. worker split: concurrency//4 page workers vs more")
    await run_case(
        port,
        rate=4.0,
        concurrency=4,
        max_pages=60,
        sink_kind="jsonl",
        label="rate=4.0 conc=4  (1 page worker)",
    )
    await run_case(
        port,
        rate=4.0,
        concurrency=64,
        max_pages=80,
        sink_kind="jsonl",
        label="rate=4.0 conc=64 (16 page workers)",
    )

    print("\nC. unlimited rate -- pure internal ceiling")
    await run_case(
        port,
        rate=1000.0,
        concurrency=16,
        max_pages=120,
        sink_kind="jsonl",
        label="rate=1000 conc=16",
    )

    import os

    if os.environ.get("PROFILE_PG_DSN"):
        print("\nD. Postgres sink vs jsonl (same shape)")
        await run_case(
            port,
            rate=1000.0,
            concurrency=16,
            max_pages=120,
            sink_kind="postgres",
            label="rate=1000 conc=16 POSTGRES",
        )


asyncio.run(main())
