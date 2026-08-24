# Plans — INDEX

## Current phase: week 1, crawl loop written and tested — not yet run for real

The seed vertical is chosen and the loop exists. Nothing has crawled the open
internet yet; the next action is a bounded live run against FOSDEM.

## Milestones

### Week 1 — crawl only, no GPU
Prove extraction and storage before adding a model to the loop.

- [x] Choose a vertical — **conference speakers**
      ([[seed-vertical-conference-speakers]])
- [x] Fill in `seeds.yaml` — 2 verticals enabled, 3 parked, all robots-surveyed
- [x] Wire `crawler/fetch.py` — streaming GET, header screen, magic-byte sniff,
      split retry policy
- [x] Wire `crawler/run.py` — two frontiers, scope enforcement, in-flight
      recovery, JSONL sink
- [x] **Set a real User-Agent** — `arc_search/0.1 (+https://github.com/joshb113/arc_search)`
      in `.env`. Repo URL only, no email published. `check_user_agent()` enforces
      it; loopback-only crawls are the one bypass.
- [x] First bounded live run — 40 pages of FOSDEM `/2025/`, 0 failures, 0 robots
      exclusions, 8/8 speaker photos with weak labels after one tuning pass.
      Numbers in [[seed-vertical-conference-speakers]].
- [x] Postgres writer for `domain`/`url_path`/`page`/`image`/`image_source`
      — `index/store.py`. Dedup is seeded from the `image` table at startup, so
      resume is correct by construction. `--sink postgres` is now the default;
      `--sink jsonl` remains for throwaway runs.
- [x] **Verify the writer against a live Postgres** — WSL2 installed, Docker
      up, schema auto-applied by the initdb mount. 10 integration tests pass
      and are repeatable. Two budgeted runs over one frontier produced 65
      pages / 40 images with zero duplicates, 40 of 40 carrying weak labels.
      Found three real bugs in the process; see
      [[seed-vertical-conference-speakers]].
- [ ] Full FOSDEM archive run (~5 h at 1 rps across 13 years)
- [ ] Crawl to 100k images
- [ ] Report: images/host, extraction source breakdown, robots exclusion rate
      — `CrawlStats.report()` already emits all three

**Exit criterion:** 100k images discovered, deduped, and recorded, with the
crawl surviving a deliberate `kill -9` and resuming.
*Resume is covered by a test; the 100k is not yet attempted.*

### Week 2 — index and query
- [ ] `index/store.py` — Qdrant collection + Postgres face writer
- [ ] Swap `MetadataSink` for `FaceIndexSink` — same call site in `run.py`, so
      the crawl loop does not change
- [ ] Upload-a-photo query endpoint
- [ ] Report reject-stat breakdown (`too_small` / `too_blurry` / `bad_pose`)

**Exit criterion:** ~70k faces indexed, query by upload returns plausible hits.
This is already a working product.

### Weeks 3–6 — precision
- [ ] Labeled eval set + `eval/calibrate.py` → **derive thresholds**
      — seeded from FOSDEM's `alt="Photo of NAME"`, which the crawler already
      records
- [ ] AdaFace second model, ensemble agreement re-rank
- [ ] Face-level canonicalization at cosine > 0.92
- [ ] Identity clustering of results
- [ ] Scale to 1–10M

### Months 2–6 — scale and hygiene
- [ ] TTL reaper for dead source links
- [ ] Incremental recrawl
- [ ] Frontier → Redis if one process stops being enough

## Open questions

- **The crawl loop uses ~46% of its permitted request rate.** Configured at
  1.0 rps against `archive.fosdem.org`, sustained throughput is 0.46–0.50
  requests/sec measured over 180s and 240s windows. Ruled out, by measurement:
  the `TokenBucket` itself is exact (1.00/s at 1/4/16 concurrent workers, and
  correct at 0.5 and 2.0 rps once the harness stopped counting grants that
  landed after the window); duplicate images (`image_source` growth accounts
  for almost nothing); and `_idle()` polling (~5% of one core).

  Not yet isolated. The suspects are all "synchronous work on the event loop":
  `Frontier` is blocking SQLite on one shared connection, `PostgresWriter` is
  blocking psycopg, and both are called from async workers. A page with 700
  links does 700 blocking `add()` calls. Also worth checking the fixed
  `concurrency // 4` page/image worker split — with `pending_images` sitting at
  0, twelve image workers idle while four page workers do everything.

  Consequence is wall-clock only: ~9.3 h for the archive run instead of ~5 h.
  Not a politeness or correctness problem — we are under the limit, not over.
  Profile before scaling to `conf.researchr.org`, where it would matter more.

- **Will the vertical reach 100k?** FOSDEM + ccc is likely 30–50k. The make-up
  is `conf.researchr.org`, parked as tier 2 in `seeds.yaml`. Decide after the
  first real run gives an images/host number.
- PDQ near-dup threshold (default 31) is conventional, not tuned for our corpus.
- Whether to keep crops for *all* faces or only above quality p60
  ([[ADR-001-crop-only-storage]] rejected the hybrid for now; revisit past 50M).
- `MetadataSink` writes JSONL and assigns its own sequential ids. Fine as a
  week-1 stand-in; those ids are not the Postgres ids and nothing should start
  depending on them.

## Deviations from plan, recorded

- **No HEAD pre-filter.** The plan called for one. A streaming GET that abandons
  the body after reading headers saves the same bandwidth in one round trip
  instead of two, and works on servers that 405 a HEAD or answer it
  inconsistently. Rationale is in the `fetch.get_image` docstring.
- **Two frontiers, not one with a `kind` column.** Pages and images each get
  their own SQLite file, reusing the tested lease/fail/recover logic unchanged.
  Rationale in the `run.py` module docstring.

## Decisions

- [[ADR-001-crop-only-storage]]
- [[ADR-002-greenfield-not-fork]]

## Research

- [[eye_of_web-audit]]
- [[seed-vertical-conference-speakers]]
