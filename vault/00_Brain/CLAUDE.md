# arc_search – Brain Stem (CLAUDE.md)

## Project
**arc_search** is a self-hosted face search engine that:
1. Crawls a **bounded** set of domains you choose — not the open web, not scraped
   from anyone else's search engine
2. Detects and embeds every qualifying face with InsightFace `antelopev2` (R100)
3. Indexes 512-d ArcFace embeddings in Qdrant with int8 quantization
4. Serves search-by-uploaded-photo, re-ranked and clustered by identity

**Goal:** Yandex-class face search where the corpus and the query both stay on
your hardware.

**GitHub:** https://github.com/joshb113/arc_search

---

## ⚠️ Root CLAUDE.md vs this file

| File | Role |
|---|---|
| **`CLAUDE.md`** (repo root) | The **contract**. Non-negotiables, layout, scale target. Loaded automatically every session. Change it only with an ADR. |
| **`vault/00_Brain/CLAUDE.md`** (this file) | The **working state**. Current numbers, live gotchas, things learned by running it. Loaded by `/start`. Change it freely as facts change. |

Keep them consistent: if a measured number here contradicts the root file, one
of them is stale and it is usually this one.

---

## Where things are

```
src/arc_search/crawler/   frontier, politeness, extraction, fetch, run, seeds
src/arc_search/index/     dedup, faces, store
src/arc_search/serve/     NOT WRITTEN — week 2
src/arc_search/eval/      NOT WRITTEN — weeks 3-6
sql/schema.sql            applied automatically by docker compose initdb
tools/                    profile_crawl_loop.py
vault/                    this brain, plans, decisions, research, tasks
```

---

## Live operational facts

**Verticals.** `fosdem` and `ccc-media` enabled; `researchr`, `usenix`,
`smaller-confs` parked. Always pass `--only <name>` — a bare run crawls every
enabled vertical at once.

**Rates.** `ARC_CRAWL_PER_HOST_RPS=1.0` globally. A vertical's `per_host_rps`
can only **lower** that; a higher value is clamped and logged as
`politeness.override_ignored`. Effective rate is the slowest of
(global, vertical override, robots `Crawl-delay`).

**Postgres.** `docker compose up -d postgres`. The DSN carries no password —
export `PGPASSWORD` from `ARC_PG_PASSWORD` in `.env`, which is also what Compose
itself needs. Docker requires WSL2 on this machine.

**Tests.** Run **bare `pytest`**, never `python -m pytest` — the latter prepends
the CWD to `sys.path` and hides import errors that CI then catches.
Database tests need `ARC_TEST_PG_DSN` pointing at a database whose name ends in
`_test`; they **TRUNCATE** it, and a guard refuses anything else.

---

## 🔴 Traps, each of which cost real time

- **`face_count` is tri-state**: `-1` never examined, `0` examined-and-empty
  (barren, skipped forever), `>0` faces found. The crawl tier writes `-1`
  explicitly. It defaulted to `0` once and would have made week 2 skip the
  entire corpus, silently.
- **`allow_hosts` matches SUBTREES.** `fosdem.org` admits `lists.fosdem.org` —
  a mailing-list archive with no faces and tens of thousands of pages. Carve
  those out with `deny_hosts`.
- **Thresholds are measured, never chosen.** `min_image_dim` shipped at 200 and
  rejected 9 of 9 real speaker photos (165–180 px). It is now 64, *derived* from
  `min_face_px`, with a test pinning the relationship.
- **`--max-pages` is a per-run cap, not a verdict.** Budget-skipped URLs are
  released back to PENDING; they must never be `complete()`d.
- **Scope is re-checked on dequeue**, not just enqueue — the frontier outlives
  the config, so tightening `seeds.yaml` has to affect already-queued URLs.

---

## Current phase

Week 1. Crawl tier written, tested, and running against the FOSDEM archive.
No model in the loop yet — that is deliberate: prove discovery, scope,
politeness, dedup and resume while every failure is still cheap to read.

See [[plans/INDEX]] for status.
