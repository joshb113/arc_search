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
uv sync                              # or: pip install -e ".[dev]"
cp .env.example .env                 # then edit it -- see below, two values are required
docker compose up -d postgres        # schema.sql is applied automatically on first start
cp seeds.example.yaml seeds.yaml     # then edit: add your target domains
export PGPASSWORD="$ARC_PG_PASSWORD" # the DSN deliberately carries no password
python -m arc_search.crawler.run --only <vertical> --max-pages 50
```

Two values in `.env` are not optional:

- **`ARC_CRAWL_USER_AGENT`** — the crawler refuses to start on the placeholder.
  It must name the project and carry a contact route; a repo URL alone is
  enough, no email required. See "Legal posture" below.
- **`ARC_PG_PASSWORD`** — `docker compose` refuses to start without it.

Start small. `--max-pages` is a per-run cap and unfetched URLs stay queued, so
you can crawl in sessions and resume. Drop `--max-pages` when you're satisfied
the scope config is right.

## Status

**Week 1: the crawl tier works and is tested.** Frontier, extraction, scope,
politeness, dedup, and the Postgres writer are all exercised by 181 tests in
CI, including a crawl loop run against mocked HTTP and a writer run against a
real database.

Verified against a live corpus: two budgeted runs over one frontier produced 65
pages and 40 images with zero duplicates, surviving restart.

Not written yet: `index/store.py`'s face half, `serve/`, `eval/calibrate.py`.
`python -m arc_search.serve` does not exist — week 2. Every similarity
threshold in `config.py` is marked UNCALIBRATED and is a placeholder until
`eval.calibrate` has been run against a labeled set.

## Development

```bash
pytest                               # 171 tests; database tests skip
```

The database tests **truncate every table**, so they refuse to run against
anything whose database name does not end in `_test`:

```bash
docker exec arc_search-postgres-1 createdb -U arc arc_search_test
docker exec -i arc_search-postgres-1 psql -U arc -d arc_search_test < sql/schema.sql
ARC_TEST_PG_DSN=postgresql://arc@127.0.0.1:5432/arc_search_test pytest
```

Run bare `pytest`, not `python -m pytest` — the latter prepends the working
directory to `sys.path` and will hide import errors that CI then catches.

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
