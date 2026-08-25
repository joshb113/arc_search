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
src/arc_search/index/     dedup, faces, store, vectors, backfill
src/arc_search/serve/     app (FastAPI), repo (read-only Postgres)
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
`_test`; they **TRUNCATE** it, and a guard refuses anything else. Vector tests
delete any collection ending in `_test`, guarded the same way.

**279 tests.** 279 with Postgres + Qdrant up; 246 passed / 33 skipped without.
Full local run:

```
ARC_TEST_PG_DSN=postgresql://arc@127.0.0.1:5432/arc_search_test \
PGPASSWORD=... .venv/Scripts/pytest.exe
```

**The model stack works.** Verified 2026-08-24 on Python 3.14 / RTX 5070 —
see [[research/face-model-bringup]]. `insightface` **1.0.1** (not 0.7.3, which is
sdist-only and will not build here), antelopev2 loaded, all five models including
`genderage`, running on CUDA at **49 img/s**. The venv still has no `fastapi`.

⚠️ **antelopev2 unzips one level too deep** —
`~/.insightface/models/antelopev2/antelopev2/*.onnx` — and `FaceAnalysis` dies on
`assert 'detection' in self.models`. Flatten it up one level. `config.py` cites
`docker/flatten_models.sh` for this; **that file does not exist.**

⚠️ **insightface depends on `opencv-python`, we pin `opencv-python-headless`.**
Both provide `cv2`. Install insightface with `--no-deps` plus explicit deps, or
the full build shadows the headless one.

**Qdrant.** `docker compose up -d qdrant`, HTTP on 6333. Collection `faces`,
created by `VectorStore.ensure_collection()`.

**Running the backfill, and the UI.**

```
PGPASSWORD=... PYTHONPATH=src python -m arc_search.index.backfill [--limit N] [--rps R] [--dry-run]
PGPASSWORD=... PYTHONPATH=src python -m arc_search.serve.app      # http://127.0.0.1:8000
```

⚠️ **Do not inspect JSON with `curl ... | python -m json.tool`.** On Windows that
decodes the UTF-8 stream as cp1252 and re-escapes it, so a clean `è` comes back
as `Ã¨` and looks exactly like a database encoding bug. It cost a false alarm
here. Write the body to a file and open it with `encoding='utf-8'`. The corpus
is clean: **0 mojibake across 2,726 labels**, 248 of them non-ASCII.

🔴 **Stop the crawler first, or pass `--rps`.** Politeness state is per-process
and in-memory, so a backfill and a crawl against the same host each spend their
own token budget and the host sees the sum. Nothing detects this; the startup
warning is all you get.

⚠️ **Finding a running crawler by command line reports TWO processes.** One is
the Windows Python launcher stub — 0 CPU seconds, ~3 MB. The real one has
minutes of CPU and ~80 MB. Distinguish by CPU time, not by count.

---

## 🔴 Traps, each of which cost real time

- **`face_count` has FOUR states**: `-2` examined under an uncalibrated gate
  (re-examinable), `-1` never examined, `0` examined-and-empty (barren, skipped
  **forever**), `>0` faces found. The crawl tier writes `-1` explicitly. It
  defaulted to `0` once and would have made week 2 skip the entire corpus,
  silently. `-2` exists because `0` is a tombstone and every gate that produces
  an empty result is still uncalibrated — see
  [[decisions/ADR-004-uncalibrated-gates-are-not-tombstones]].
  🔴 The backfill queue is `face_count = -1`, **not** `< 0`. Folding `-2` in
  makes every run re-fetch the whole provisional set forever.
  🔴 `mark_examined(..., calibrated=False)` is the default **on purpose**:
  forgetting the flag must cost a re-fetch, never a permanent loss.
- **`allow_hosts` matches SUBTREES.** `fosdem.org` admits `lists.fosdem.org` —
  a mailing-list archive with no faces and tens of thousands of pages. Carve
  those out with `deny_hosts`.
- **Thresholds are measured, never chosen.** `min_image_dim` shipped at 200 and
  rejected 9 of 9 real speaker photos (165–180 px). It is now 64, *derived* from
  `min_face_px`, with a test pinning the relationship.
- **`--max-pages` is a per-run cap, not a verdict.** Budget-skipped URLs are
  released back to PENDING; they must never be `complete()`d.
- **Anything needed on dequeue must be IN the queue.** Image provenance lived
  in an in-memory dict beside the durable frontier; a restart kept the queue
  and lost the context, so resumed images were written with no page link and
  no alt text. It is `Frontier.meta` now. 511 images were recorded that way
  before it was caught — by checking the corpus, not by any test.
- ~~**FOSDEM alt text is year-bounded**: `Photo of NAME` exists 2015–2025, not
  2013–2014.~~ **Measured wrong.** 2013 pages carry the label — every labeled URL
  in the corpus sample is under `archive.fosdem.org/2013/`. The original finding
  sampled one page per year, which was too small.
  → [[research/face-model-bringup]]
- 🔴 **CUDA fails SILENTLY to CPU.** onnxruntime does not raise when the CUDA
  provider cannot load — one stderr line, then every session runs on CPU at
  **1/12th the speed** (measured: 49.0 vs 4.1 img/s). `faces.register_cuda_runtime()`
  puts the NVIDIA wheel DLLs on PATH (`os.add_dll_directory` does **not** work
  for this), and `effective_providers()` reports what actually loaded. Never
  trust a log that echoes the requested provider list — that is the same bug the
  crawler shipped with `politeness.override_ignored`.
- **`min_face_px` is 48, DERIVED** from where a face's embedding stops matching
  its own full-resolution self (0.973 at 48 px; below 0.92 — the canonicalization
  floor — by 32 px). It was 64, which sat on the median face size and discarded
  40% of all detections. `min_image_dim` moved to 48 with it; the invariant
  `min_image_dim <= min_face_px` is what matters, not either literal.
  → [[decisions/ADR-004-uncalibrated-gates-are-not-tombstones]]
- ⚠️ **`min_det_score = 0.72` is the next uncalibrated gate to look at.** Over the
  full index it rejected **284** detections against `too_small`'s 290 — the two
  are 574 of 595 total rejections. It has had no scrutiny at all. Measure first.
- 🔴 **PDQ is a PREREQUISITE for calibration.** 66% of labeled "genuine pairs"
  are the same photo re-published in a later year, and **223 of 225 have
  different sha1** (re-encoded), so exact-hash dedup cannot see them. They
  inflate recall by 2–5 points and `eval.calibrate` would inherit the bias.
  Still never computed (`dedup.loaded pdq=0`).
- ⚠️ **Generic labels are not identities.** `Photo of FOSDEM Staff` and
  `Photo of BSDCG Team` are shared by different people across years — one such
  "genuine" pair scores **-0.060**. Calibration must exclude collective labels.
  → [[research/first-index-and-calibration-preview]]
- **Scope is re-checked on dequeue**, not just enqueue — the frontier outlives
  the config, so tightening `seeds.yaml` has to affect already-queued URLs.
- **`face_count` is the commit marker for BOTH stores.** Write order is
  `record_faces()` → Qdrant `delete_image()`/`upsert()` → `mark_examined()`.
  If `record_faces` ever sets the count itself, a crash between the two stores
  leaves faces that no query can return and nothing reports. `face_count > -1`
  means both stores agree; `-1` means either may be partial and is disposable.
- **Qdrant client and server versions are coupled.** Both pinned to **1.19**.
  A `>=1.9` floor let client 1.19 run against a pinned 1.9.2 server, leaving
  **no working search path at all** — `query_points` 404s below server 1.10,
  and `search` was removed from the client after 1.13. Writes worked fine, so
  it would have surfaced at the first query, not the first insert.
- **The image bytes are gone.** Nothing stores pixels (non-negotiable #1), so
  the indexing tier must **re-fetch** from `domain.host` + `image.url_path` and
  therefore has to go through politeness and robots like any other crawl.
  `store.REFETCH_SCHEME` assumes **https** — the scheme is not a stored column.
- **The work queue is keyset-paginated, not OFFSET.** Rows leave the queue as
  they are processed, so an OFFSET walk over a shrinking result set skips work
  silently — the worst available failure mode for a backfill.

---

## Current phase

Week 2, [[plans/plan-002-index-and-query]] **Phase 1 complete**. The storage
layer for faces exists on both sides — Qdrant collection and Postgres `face`
writer — with the two-store write order pinned by tests. Still **no model in the
loop**: nothing has been detected, embedded, or indexed, and the `faces`
collection is empty.

The crawl continues in parallel (`archive-run9`), so the work queue grows while
the index tier is built.

**The engine answers a query.** The whole corpus is indexed: **1,300 faces** in
Postgres, Qdrant and on disk, all three agreeing exactly, 0 unexamined and
0 tombstoned. A photo goes in and the right person comes back — including the
same person in a *different year*, across 143 multi-year speakers.

**plan-002 is complete.** `serve/` is written and verified against the live
index — upload a speaker photo at `http://127.0.0.1:8000`, get that speaker back
at cosine 1.0000 with their crop, weak label and source page.

🔴 **The UI renders no verdict, and that is load-bearing.** `calibrated` is False
and `t_plausible`/`t_strong`/`t_near_certain` are placeholders — one impostor
pair already scores **0.651**, above `t_near_certain`. The page shows raw cosine
behind an UNCALIBRATED banner, and a test asserts no verdict language appears.
Do not add one before [[plan-003-precision]] runs.

**Next:** plan-003 — but PDQ first (see the trap above); it is a prerequisite,
not a parallel task.

See [[research/first-index-and-calibration-preview]] and
[[research/face-model-bringup]].

See [[plans/INDEX]] for status.
