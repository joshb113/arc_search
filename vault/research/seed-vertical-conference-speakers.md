# Seed vertical: conference speakers

Chosen 2026-08-23. Unblocks week 1. Config lives in `seeds.yaml` (gitignored);
`seeds.example.yaml` remains the generic template.

## Why this vertical

**The easy end of the detection problem, first.** Speaker photos are near-frontal,
lit, and usually one face per image. While the pipeline itself is unproven, every
failure should be legible as a *pipeline* failure. Starting on crowd shots means
spending week 2 unable to tell a detector problem from a plumbing problem.

**Natural positives for calibration.** The same speaker recurs across years and
across conferences. Those repeats are the labeled pairs `eval/calibrate` needs,
and they arrive as a property of the corpus rather than as an annotation project.
Non-negotiable #5 says thresholds are derived, never literal — that requires a
labeled set to derive them *from*, and this vertical produces one for free.

**Defensible posture.** Conference speaker pages are material the subject
published, deliberately, under their own name, to be found. That is the
best-shaped version of this corpus.

## The FOSDEM find

`archive.fosdem.org` renders each speaker page with:

```html
<img src="/2025/schedule/speaker/a_salt/6e7bd54f....jpeg"
     class="speaker-photo" alt="Photo of A. Salt" />
```

The alt text is a **weak label, supplied by the publisher**. Thirteen years
(2013–2025) of static HTML, no JavaScript, no robots.txt (404 → allow-all under
RFC 9309).

This is why `MetadataSink` carries `alt` through to the record even though week 1
has no model in the loop: discarding it at crawl time and trying to recover it in
week 3 means a full recrawl.

Coverage confirmed by hand: `/2013/` through `/2025/schedule/speakers/` all 200.

## Robots survey, 2026-08-23

| Host | Verdict | Notes |
|---|---|---|
| `archive.fosdem.org` | **allow** | no robots.txt (404). Static nginx. Tier 1. |
| `fosdem.org` | **allow** | redirects to archive after the event. |
| `media.ccc.de` | **allow** | robots.txt is a single comment, no rules. Tier 1. |
| `conf.researchr.org` | allow | `Crawl-delay: 2`, `Disallow: /*?`. Tier 2, largest corpus. |
| `www.gophercon.com` | allow | `Disallow:` (empty = allow all). |
| `rustconf.com` | allow | `Disallow:` + `Crawl-delay: 10`. |
| `www.usenix.org` | allow, slow | `Crawl-delay: 10`; a session page timed out at 15s. |
| `neurips.cc` | **skip** | photos render client-side; a plain GET returns navbar logos. |
| `pretalx.com` | **skip** | see below. |
| `thestrangeloop.com` | **skip** | bare S3, robots.txt returns `AccessDenied`. Ambiguous → treat as restricted. |

### pretalx is out, and it matters

`pretalx.com/robots.txt` contains `Disallow: /media/`. pretalx serves speaker
avatars from exactly that prefix. **The images are the disallowed part.**

This rules out every event hosted on pretalx.com — DjangoCon, several EuroPython
years, many regional PyCons. A large and otherwise ideal slice of this vertical
is simply not available to a crawler that respects robots.txt, and we do
(non-negotiable #6). Recorded here so nobody re-derives it in three months and
assumes it was an oversight.

## First live run, 2026-08-23

Two bounded runs against `archive.fosdem.org`. UA
`arc_search/0.1 (+https://github.com/joshb113/arc_search)`, 1 rps, 40 pages of
`/2025/`.

**Run 1 — before tuning.** 40 pages, 0 failures, 0 robots exclusions. 67 images
discovered, 33 fetched. Of those 33, only **8 were speaker photos**; the other
25 were sponsor logos and venue maps under `/2025/sponsored-by/` and
`/2025/assets/` — 1.44 MB of 1.81 MB total, for zero faces.

**Run 2 — after adding those two prefixes to `deny_patterns`.** Identical page
coverage. 8 images fetched, **8 of 8 speaker photos**, 370 KB. 4.9× less
bandwidth and 3.5 min → 2.6 min on the same 40 pages.

The weak labels arrived exactly as predicted: `Photo of A. Salt`,
`Photo of Aapo Alasuutari`, `Photo of Adrian Reber`, … straight into the JSONL
`alt` field with no post-processing.

Two things the run taught that reading the HTML did not:

- **A small `--max-pages` never reaches the payload.** The frontier leases
  `ORDER BY depth, added_at` — breadth-first, correctly. With 14 seed indexes ×
  ~700 speaker links, a 12-page budget is consumed entirely at depth 0 and
  fetches nothing but the site-wide `og:image` logo 12 times. (Which the SHA1
  gate did collapse: 8 `exact_dup` of 9. Non-negotiable #2, visible on real
  data.) Smoke-test one year, not fourteen.
- **Volume estimate, firmed up.** `/2025/` alone showed ~1,330 pages pending
  (speakers + events). At 1 rps that is ~22 min/year, so ~5 hours for the full
  13-year archive. Comfortable.

## Threshold calibration, 2026-08-24

Sampled 9 FOSDEM 2025 speaker photos directly. The result killed two shipped
defaults:

| | range | old default | outcome |
|---|---|---|---|
| short side | 165–180 px | `min_image_dim = 200` | **rejected 9 of 9** |
| byte size | 5,399–66,114 B | `min_image_bytes = 8000` | **rejected 4 of 9** |

Both were magic numbers, which non-negotiable #5 forbids, and between them they
discarded the entire corpus with no error and no log line. The crawl would have
completed, the index would have been empty, and it would have looked like a
model problem.

New values, and their derivations:

- `min_image_dim = 64`, **derived** from `FaceSettings.min_face_px`. An image
  whose shorter side is under the smallest face we would accept cannot contain
  a qualifying face. `test_min_image_dim_is_not_above_the_minimum_face_size`
  pins the two together so they cannot drift apart again.
- `min_image_bytes = 2000`, and explicitly a *bandwidth* filter, not a quality
  gate. Its only job is to avoid paying for tracking pixels and spacers, which
  are under 1 KB. Quality is decided by the dimension floor and by the face
  quality gate after detection.

Confirmed against the live corpus afterwards: the smallest accepted image is
3,040 B and the smallest short side is 116 px (`Photo of Aditi`, 116x180 at
4,522 B) — a perfectly usable portrait that both old thresholds rejected.

## First Postgres run, 2026-08-24

Two budgeted runs over one frontier and one database, `/2025/` only.

| | run A | run B | total |
|---|---|---|---|
| pages fetched | 20 | 45 | 65 |
| images fetched | 7 | 33 | 40 |
| `image` rows | 7 | +33 | 40 |

Verified in the database afterwards: **40 of 40 images carry a `Photo of NAME`
weak label**, all 40 sit at `face_count = -1` (never examined), none at 0, and
UTF-8 survives the round trip (`Photo of Adolfo García Veytia`).

Two bugs surfaced here that unit tests could not have caught:

1. **`face_count` defaulted to 0**, which `Deduper` reads as *barren — never
   look again*. The crawl tier has no detector, so every row it wrote claimed
   to have been examined and found empty. On the next startup the whole corpus
   loaded as barren and week 2 would have skipped all of it. Now `-1`, with a
   CHECK constraint and a partial index serving as the indexer's work queue.
2. **The page budget called `complete()`**, permanently marking budget-skipped
   URLs DONE. Run A marked 1,347 URLs done having fetched 20. `--max-pages`
   silently consumed the frontier, so "crawl some tonight, more tomorrow" did
   nothing on the second night. Budget-skipped URLs are now `release()`d back
   to PENDING.

## Profiling the throughput "gap", 2026-08-24

The live crawl sustained 0.46–0.50 req/s against a configured 1.0. Chased it
properly rather than guessing, and the chase is more instructive than the
answer.

**What was ruled out, by measurement:**

| suspect | verdict |
|---|---|
| `TokenBucket` inaccuracy | exact: 1.00/s at 1, 4 and 16 workers; also exact at 0.5 and 2.0 |
| duplicate images absorbing requests | `image_source` growth accounts for almost none |
| `_idle()` polling the frontier | 0.52 ms/call, ~5% of one core |
| `Frontier` / parsing cost | 1 ms and 1.3 ms per speaker page |
| the `concurrency // 4` worker split | 4.06 req/s at a 4.0 limit with **one** page worker |
| the sink | internal ceiling 114 req/s (JSONL), 56 req/s (Postgres) |

**The answer:** `configured_rate()` returns `min(global, override)`. The global
`per_host_rps` default is 0.5 and `seeds.yaml` asked for 1.0, so it was clamped.
The crawler was doing exactly what it was configured to do. An override may only
ever *lower* the rate — which is the correct contract for a politeness control,
and is what `seeds.example.yaml` documents.

The real defect was observability: the startup log printed the *requested*
override, so it read `1.0` while the crawl ran at `0.5`. It now logs
`effective_rps` and warns `politeness.override_ignored`. Raising
`ARC_CRAWL_PER_HOST_RPS` to 1.0 is the correct lever; measured `req_per_s=1.0`
sustained afterwards, halving the archive run to ~5 h.

**Two things the profiler found on the way**, neither of which was the target:

- **robots.txt was fetched from the wrong port.** The URL was built from the
  bare hostname, so a host on `:8080` had its robots.txt requested from `:80`.
  That connection fails, `Politeness` fails closed, and the entire host is
  silently skipped — reported only as `robots_disallow`, which reads like the
  site said no. Found within seconds of pointing the profiler at a loopback
  server on an ephemeral port. State is now keyed per `scheme://authority` per
  RFC 9309, while the rate budget stays per hostname, because one machine
  answers all of its ports.
- **Request counts were never recorded.** Throughput had to be inferred from
  table growth, which undercounts — a skipped or duplicate image spends a token
  and writes nothing. `Politeness.requests_made` now counts at the limiter,
  where every request passes exactly once, and the heartbeat reports
  `req_per_s`. That turned a two-hour question into a one-line answer.

The harness is kept at `tools/profile_crawl_loop.py`.

## ccc-media validation, 2026-08-24

15-page bounded run against `media.ccc.de`. **Verdict: keep it, with denies.**

Mechanically sound — 18 pages, 130 images, 0 failed, 0 robots exclusions, and
`effective_rps={'media.ccc.de': 0.5}` confirms the per-vertical override
lowering below the 1.0 global for real.

Content, though, was 70% waste:

| | count | share |
|---|---|---|
| conference logos | 86 | 70% |
| video thumbnails (400x225) | 37 | 30% |

The seeds `/b/congress` and `/b/conferences` are *browse* pages — they list
conferences by logo. The thumbnails are deeper. Logos are all named `logo.*`,
`media-logo.*`, `unknown.*` or live under `/logos/`, so three deny patterns
remove them.

Confirmed as predicted: **no weak labels.** The thumbnail alt text is the talk
title (`"Von Ubuntu zu Debian: Ein neuer Upstream für TUXEDO OS"`), not a
speaker name. Detection corpus only, not eval material — which is what this
vertical is here for: on-stage faces, off-axis and variably lit, as a
counterweight to FOSDEM's clean portraits.

Two bugs the run exposed, neither related to ccc:

1. **Body-read timeouts escaped the fetch error model.** `_request` retries the
   request and headers; `resp.aread()` was outside any handler. A slow host
   timing out mid-transfer produced neither `Skipped` nor `FetchError`, so it
   fell through to the worker catch-all, logged `worker.crashed` with a full
   traceback, and **was not counted as a page failure** — the report said
   "pages failed 0" while three pages had timed out. FOSDEM is fast enough that
   this never appeared; media.ccc.de is not.
2. **The page budget overshoots by `n_page - 1`.** `--max-pages 15` fetched 18.
   Check-then-increment with an `await` between lets every page worker pass the
   check before any increments. The slot is now reserved before the await and
   handed back if the fetch yields no page.

## Open: the image URL is never stored

`image` holds sha1, dimensions and byte size — no URL. So you cannot re-fetch an
image, cannot write URL-based deny rules from corpus analysis (I had to re-fetch
a page by hand to find the logo naming above), and provenance only reaches the
*page*, not the file.

Adding it is not free: image paths are near-unique, so interning saves little,
and `url_path` would grow to roughly one row per image — order 1 GB at 10M
against a 48 GB total budget. That is an ADR-sized decision, not a casual
schema edit. Flagging, not deciding.

## Caveats to carry into week 2

- **FOSDEM photos are portraits.** A threshold calibrated on nothing but clean
  headshots will collapse on the first real query. `media.ccc.de` is in tier 1
  specifically as a counterweight: its thumbnails are video keyframes, so the
  faces are on-stage, off-axis, and variably lit. Harder, and necessary.
- **ccc alt text is the talk title, not the speaker name.** Detection corpus
  only; not eval material.
- **Volume is uncertain.** FOSDEM is roughly 700 speakers/year × 13 years ≈ 9k
  photos, plus page furniture. That plus ccc is probably 30–50k images, short of
  the 100k week-1 exit criterion. `conf.researchr.org` (tier 2) is the intended
  make-up and is by far the largest single opportunity here — ICSE, SPLASH,
  ICFP, POPL, OOPSLA and ~100 more, each with author and committee pages. Flip
  it on once the loop has proven itself on a fast host.

## Rejected outright

TED, Web Summit, and the commercial conference circuit. Large headshot corpora,
but ToS-hostile and CDN-defended. The entire point of a bounded vertical crawl
is not to be in that fight.
