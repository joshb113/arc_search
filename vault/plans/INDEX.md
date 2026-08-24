# Plans INDEX

**Status legend:** ✅ Complete · 🟡 Active · 📋 Planned · ⬜ Step not started ·
🔴 Trap, do not undo · ⚠️ Handle with care

| Plan | Status | Description |
|------|--------|-------------|
| [[plan-001-crawl-tier]] | 🟡 Active | Crawl tier, no model in the loop. Frontier, scope, politeness, extraction, dedup, Postgres writer — all written and tested (**233 tests**, CI green). Running against the FOSDEM archive at a measured `req_per_s=1.0`. 🔴 **Anything needed on dequeue must be IN the queue.** Image provenance lived in an in-memory dict beside the durable frontier, so a restart kept the queue and lost the context: resumed images were written with **no page link and no alt text**, silently discarding the weak labels [[plan-003-precision]] is built on. 511 images went in that way and the crawl reported perfect health throughout — caught by querying the corpus during a wrap-up, not by any test. Now `Frontier.meta`, with a restart test that goes red on the old design. 🔴 **`face_count` is tri-state** (`-1` unexamined / `0` barren / `>0` found) — it defaulted to `0`, which `Deduper` reads as *never look again*, and would have made week 2 skip the entire corpus. 🔴 **`min_image_dim` shipped at 200 and rejected 9 of 9 real speaker photos** (165–180 px); now 64, *derived* from `min_face_px`, with a test pinning the relationship. ⚠️ `--max-pages` is a per-run cap — budget-skipped URLs are `release()`d, never `complete()`d, or the frontier is consumed. Remaining: the full 13-year run and the 100k exit criterion. |
| [[plan-002-index-and-query]] | 📋 Planned | Week 2 — Qdrant collection, Postgres `face` writer, crops, and the upload-a-photo endpoint. `FaceIndexSink` slots into the **same call site** in `run.py`, so the crawl loop does not change. `image_unexamined_idx` (partial, `face_count < 0`) is the work queue, built for this. 🔴 Landmarks and bbox are stored at **original** resolution with `src_width`/`src_height` alongside — non-negotiable #3, and eye_of_web's exact mistake. Note PDQ is still never computed: the column, the BK-tree and the tests all exist, but near-dup dedup is SHA1-only until this plan runs. |
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

## Research

- [[eye_of_web-audit]]
- [[seed-vertical-conference-speakers]]

## Other indexes

- [[00_Brain/ISSUES_INDEX]] — issues whose root cause is **outside** arc_search
- [[00_Brain/CLAUDE]] — working state, live numbers, traps
