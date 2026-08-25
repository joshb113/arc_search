# plan-001 – Crawl tier (week 1)

**Status:** 🟡 Active — loop complete and running; the 100k target is not yet met
**Goal:** Prove extraction, scope, politeness, dedup and storage with **no model in
the loop**. A GPU in the pipeline makes every bug look like a model bug.

**Exit criterion:** 100k images discovered, deduped and recorded, with the crawl
surviving a deliberate `kill -9` and resuming.
*Resume is proven — on a real hard kill, `store.resumed images=2238` +
`crawl.recovered_inflight urls=5`. The 100k is not yet reached.*

---

## Phase 1 – Scope and identity ✅

- [x] Choose a vertical — **conference speakers** ([[seed-vertical-conference-speakers]])
- [x] Fill in `seeds.yaml` — 2 verticals enabled, 3 parked, all robots-surveyed by hand
- [x] **Set a real User-Agent** — `arc_search/0.1 (+https://github.com/joshb113/arc_search)`.
      Repo URL only, no email published. `check_user_agent()` enforces it; a
      loopback-only crawl is the single bypass.
- [x] `deny_hosts` — `allow_hosts` matches subtrees, so `fosdem.org` was admitting
      `lists.fosdem.org` (a faceless mailing-list archive)

## Phase 2 – The loop ✅

- [x] `crawler/fetch.py` — streaming GET, header screen, magic-byte sniff, split retry
- [x] `crawler/run.py` — two frontiers, scope enforced at enqueue **and** dequeue,
      in-flight recovery, quiescence detection
- [x] Header-only dimension probing — `image.width/height` are NOT NULL and the crawl
      tier had no way to fill them
- [x] Request counting at the rate limiter — row counts are not a proxy, since a
      skipped or duplicate image spends a token and writes nothing

## Phase 3 – Storage ✅

- [x] Postgres writer for `domain`/`url_path`/`page`/`image`/`image_source` —
      `index/store.py`. Dedup seeded from the `image` table at startup, so resume is
      correct by construction rather than by remembering to do something.
- [x] Verify against a live Postgres — 10 integration tests, repeatable
- [x] Connection resilience — a transient blip must not end a five-hour run
      ([[00_Brain/ISSUES_INDEX#ISS-003]])
- [x] Store the image URL — [[ADR-003-store-image-urls]]
- [x] **Image provenance survives a restart** — the context rode in an
      in-memory dict beside a durable queue. On resume every already-queued
      image was written with no page link and no alt text. `Frontier.meta`
      now carries it with the URL.

## Phase 4 – The archive run 🟡

- [x] First bounded live run — 40 pages of FOSDEM `/2025/`, 0 failures, 0 robots
      exclusions, 8/8 speaker photos with weak labels after one tuning pass
- [x] Validate `ccc-media` — works; logo denies took signal density 30% → 82%
- [ ] Full FOSDEM archive run (~5 h at 1 rps across 13 years)
- [ ] Crawl to 100k images
- [ ] Report: images/host, extraction source breakdown, robots exclusion rate —
      `CrawlStats.report()` already emits all three

---

## 🔴 Open defect — the politeness rate is advisory, not accurate

**Found 2026-08-25.** Token buckets are keyed on **hostname**, but the thing
politeness protects is a **server**. Measured:

```
archive.fosdem.org -> 2600:1702:8247:e10::1
fosdem.org         -> 2600:1702:8247:e10::1
```

Two hostnames, one box, two independent budgets. At the configured
`per_host_rps = 1.0` that server has been receiving up to **2 rps** whenever a
run touches both hosts — which is most of the time, since `seeds.yaml` admits
both.

This is the **same shape** as the `politeness.override_ignored` bug this plan
already fixed once: *the number we log is not the number the other end
experiences.* That one was caught because a throughput profile disagreed with
the config. This one is invisible from inside the process entirely.

⚠️ **It scales badly.** Two aliases of one host is a doubling. A vertical with a
CDN, or several conference sites behind one provider, is an arbitrary multiplier
— and the failure mode is being rate-limited or blocked by someone whose
robots.txt we were scrupulously honouring.

Not urgent at present volumes (10 rps on a static nginx archive is modest), and
deliberately **not** fixed under time pressure while a drain was running. Two
candidate fixes:

- **Key the bucket on the resolved IP.** Correct, and it handles aliases nobody
  declared. Costs a DNS lookup per new host, cached — and needs care with
  round-robin DNS and CDNs, where one hostname legitimately spans many IPs.
- **Alias known-together hosts in `seeds.yaml`.** Cheap and explicit, but only
  catches what someone remembered to declare, which is the weaker guarantee.

Non-negotiable #6 is about robots.txt and honest identification, and both still
hold — so this is a defect against our own stated rate, not a breach of the
contract. Worth noting that `archive.fosdem.org` serves **no robots.txt at all**
(404), so nothing external was constraining the rate either way.

## Open questions

- **Will the vertical reach 100k?** FOSDEM + ccc is likely 30–50k. The make-up is
  `conf.researchr.org`, parked as tier 2 in `seeds.yaml`. Decide once the archive
  run gives a real images/host number.
- **`MetadataSink` assigns its own sequential ids.** Fine as a week-1 stand-in for
  `--sink jsonl`, but those ids are not the Postgres ids and nothing should start
  depending on them.

## Deviations from plan, recorded

- **No HEAD pre-filter.** The plan called for one. A streaming GET that abandons the
  body after reading headers saves the same bandwidth in one round trip instead of
  two, and works on servers that 405 a HEAD or answer it inconsistently. Rationale
  in the `fetch.get_image` docstring.
- **Two frontiers, not one with a `kind` column.** Pages and images each get their
  own SQLite file, reusing the tested lease/fail/recover logic unchanged. Rationale
  in the `run.py` module docstring.
