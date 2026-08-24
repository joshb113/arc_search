# arc_search

A self-hosted face search engine over a targeted vertical crawl.

Yandex Images is the best free face search available, and using it means sending
your query image to Yandex. `arc_search` is the same capability with the corpus
and the query staying on your hardware.

## What it is

- **Crawls** a bounded set of domains you choose (not the whole web, not scraped
  from Google or social platforms).
- **Detects and embeds** every qualifying face with InsightFace `antelopev2` (R100).
- **Indexes** 512-d ArcFace embeddings in Qdrant with int8 quantization.
- **Searches** by uploaded photo, re-ranks with a second model, and clusters
  results by identity.

## What it is not

- Not web-scale. You choose the vertical; density beats breadth.
- Not a scraper of other people's search engines.
- Not a public service. It binds to localhost by design.

## Storage budget

The defining constraint. At 30M crawled images / ~10M canonical faces:

| Component | Size |
|---|---|
| Face crops, 128px WebP q75, deduped | 40 GB |
| Vectors, 512-d int8 | 5 GB |
| Qdrant HNSW graph (m=16) | 0.7 GB |
| Metadata, Postgres (string-interned) | 1.5 GB |
| PDQ + SHA1 hashes | 1 GB |
| **Total** | **≈ 48 GB** |

Storing full-scene thumbnails instead would cost ~1.2 TB. That difference is the
core architectural decision — see `vault/decisions/ADR-001-crop-only-storage.md`.

## Quickstart

```bash
uv sync                          # or: pip install -e ".[dev]"
docker compose up -d             # qdrant + postgres + redis
psql -f sql/schema.sql           # create tables
cp seeds.example.yaml seeds.yaml # then edit: add your target domains
python -m arc_search.crawler     # start crawling
python -m arc_search.serve       # http://127.0.0.1:8000
```

## Status

Scaffold. Nothing is load-bearing yet. The week-1 milestone is a crawl of ~10
seed domains to 100k images with no ML at all — prove extraction and storage
before adding a GPU to the loop.

## Legal posture

Face recognition over images of people carries real obligations under GDPR
Art. 9 and Illinois BIPA. This project's defaults reflect that:

- Binds to `127.0.0.1`. A local index and a reachable service are different
  legal objects.
- Ships an age filter that drops faces estimated under 18.
- Ships a hash-based exclusion list, queried on every search.
- Respects `robots.txt` and identifies itself honestly on every request.

Removing any of those is a decision you are making deliberately. Don't do it by
accident.
