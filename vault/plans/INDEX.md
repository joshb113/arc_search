# Plans INDEX

**Status legend:** ✅ Complete · 🟡 Active · 📋 Planned · ⬜ Step not started ·
🔴 Trap, do not undo · ⚠️ Handle with care

| Plan | Status | Description |
|------|--------|-------------|
| [[plan-001-crawl-tier]] | 🟡 Active | Crawl tier, no model in the loop. Frontier, scope, politeness, extraction, dedup, Postgres writer — all written and tested (**233 tests**, CI green). Running against the FOSDEM archive at a measured `req_per_s=1.0`. 🔴 **Anything needed on dequeue must be IN the queue.** Image provenance lived in an in-memory dict beside the durable frontier, so a restart kept the queue and lost the context: resumed images were written with **no page link and no alt text**, silently discarding the weak labels [[plan-003-precision]] is built on. 511 images went in that way and the crawl reported perfect health throughout — caught by querying the corpus during a wrap-up, not by any test. Now `Frontier.meta`, with a restart test that goes red on the old design. 🔴 **`face_count` is tri-state** (`-1` unexamined / `0` barren / `>0` found) — it defaulted to `0`, which `Deduper` reads as *never look again*, and would have made week 2 skip the entire corpus. 🔴 **`min_image_dim` shipped at 200 and rejected 9 of 9 real speaker photos** (165–180 px); now 64, *derived* from `min_face_px`, with a test pinning the relationship. ⚠️ `--max-pages` is a per-run cap — budget-skipped URLs are `release()`d, never `complete()`d, or the frontier is consumed. Remaining: the full 13-year run and the 100k exit criterion. |
| [[plan-002-index-and-query]] | ✅ Complete | Week 2 — **the engine answers a query, with a UI.** Corpus fully indexed: **1,300 faces**, Postgres + Qdrant + crops agreeing exactly, **0 unexamined, 0 tombstoned**, 0 failures across 2,219 polite re-fetches. Identity matching works across years (143 multi-year speakers; Adam Samalik 2019→2017 at 0.667 vs best impostor 0.301). `serve/` verified end to end at 127.0.0.1:8000: upload a speaker photo, get that speaker back at cosine 1.0000 with crop, weak label and source page (next-best 0.2625). 🔴 **The UI renders no verdict** — UNCALIBRATED banner, raw cosine, and a test asserting *near certain* / *strong match* / *confident* appear nowhere, because one measured impostor pair scores **0.651**. The uploaded photo is never written to disk. 🔴 **`image.face_count` is the commit marker for BOTH stores**: `record_faces()` → Qdrant `delete_image()`/`upsert()` → `mark_examined()`. If `record_faces` ever sets the count it just wrote, a crash between the stores leaves faces no query can return and nothing reports it. 🔴 **Qdrant client/server versions are coupled**: a `>=1.9` floor let the client reach 1.19 against a pinned 1.9.2 server, leaving **no working search path in either direction** — and writes succeeded throughout, so it would only have failed at the first query. Both pinned to 1.19. ⚠️ **The bytes are gone**, so the backfill re-fetches through `Fetcher` — politeness and robots apply, and `REFETCH_SCHEME` assumes https because the scheme is not a stored column. Measured on real data: crops **1,921 B** against ADR-001's 4 KB, putting 10M faces at **44.8 GB** vs the 49.6 GB target. 🔴 PDQ is still never computed (`dedup.loaded pdq=0`) and is now a **prerequisite for calibration**, not a parallel task. |
| [[plan-003-precision]] | 📋 Planned | Weeks 3–6 — derive every threshold from a measurement. The labeled set comes free from FOSDEM's `alt="Photo of NAME"`, which the crawler already records (**40 of 40** in the first Postgres run). Then AdaFace ensemble re-rank, canonicalization at cosine > 0.92, identity clustering, scale to 1–10M. ⚠️ **The label is year-bounded: 2015–2025 carry it, 2013–2014 do not** — 11 of 13 crawled years, measured one page per year. A fresh crawl showing zero labels is probably not broken; the frontier is breadth-first and the seeds start at 2013. 🔴 `config.py`'s `t_plausible`/`t_strong`/`t_near_certain` are marked UNCALIBRATED and are placeholders — nothing should trust a result until `calibrated` flips to True from a real run. |
| [[plan-004-scale-and-hygiene]] | 📋 Planned | Months 2–6 — TTL reaper for dead links (`page.last_checked` and its `NULLS FIRST` index exist for it), incremental recrawl, frontier → Redis, and dedup as a query rather than a startup preload (`_load_dedup` reads the whole `image` table — fine to low millions, ~1 GB of sha1 dict at 10M). |

---

## Cross-cutting open questions

- **Will the vertical reach 100k?** FOSDEM + ccc is likely 30–50k. The make-up is
  `conf.researchr.org`, parked as tier 2. Decide once the archive run reports a real
  images/host number. → [[plan-001-crawl-tier]]
- **`url_path` interns something that never repeats.** 1:1 with `page`, so it pays a
  row and a UNIQUE index per value for zero deduplication — the same reasoning that
  sent image paths inline in [[ADR-003-store-image-urls]]. → [[plan-002-index-and-query]]
- ~~**What should `min_face_px` be?**~~ → [[ADR-004-uncalibrated-gates-are-not-tombstones]].
  Derived at **48** from an embedding-decay measurement, and more importantly the
  mechanism changed: no uncalibrated threshold may write an irreversible verdict.
- **Qdrant keeps original vectors as well as the int8 copy** — 20.5 GB against 5.1 GB
  at 10M faces, and the largest single line in the disk budget. Total lands at
  **44.8 GB** (measured, real crops) inside the 49.6 GB target — a watch item, not a problem.
  → [[plan-002-index-and-query]]

## Resolved, kept because the reasoning is reusable

- ~~**The crawl loop uses ~46% of its permitted request rate.**~~ **Not a bug.**
  `configured_rate()` returns `min(global, override)`, so `seeds.yaml` asking 1.0 rps
  against the 0.5 global was clamped. The crawler was doing exactly what it was told.
  The real defect was that the startup log printed the *requested* override, so it
  read `1.0` while the crawl ran at `0.5`. It now logs `effective_rps` and warns
  `politeness.override_ignored`. Full profiling writeup in
  [[seed-vertical-conference-speakers]] — including the three hypotheses that were
  wrong, and a benchmark harness that reported a false 3.8/s until it stopped
  counting grants landing after its own measurement window.
- ~~**The image URL is never stored.**~~ → [[ADR-003-store-image-urls]]. Measured
  rather than assumed: interning the path costs 295 B/image against 157 B inline,
  88% more for zero dedup, because image paths are unique per image.

## Decisions

- [[ADR-001-crop-only-storage]]
- [[ADR-002-greenfield-not-fork]]
- [[ADR-003-store-image-urls]]
- [[ADR-004-uncalibrated-gates-are-not-tombstones]] — `min_face_px` 64 → 48, derived
  from where the embedding stops matching itself. Adds `face_count = -2`: an
  uncalibrated gate may not write a permanent verdict.

## Research

- [[eye_of_web-audit]]
- [[seed-vertical-conference-speakers]]
- [[face-model-bringup]] — antelopev2 runs (49 img/s on CUDA). Three defects found,
  two of them silent: CUDA was falling back to CPU at 1/12th speed while the log
  said CUDA, and `min_face_px=64` discards half of all real faces.
- [[first-index-and-calibration-preview]] — the first full index: 1,300 faces, 0
  failures across 2,219 re-fetches, identity matching working across years. Plus a
  calibration preview and the reason it is **not** a calibration: 66% of labeled
  genuine pairs are the same photo re-published, which makes PDQ a prerequisite.

## Other indexes

- [[00_Brain/ISSUES_INDEX]] — issues whose root cause is **outside** arc_search
- [[00_Brain/CLAUDE]] — working state, live numbers, traps
