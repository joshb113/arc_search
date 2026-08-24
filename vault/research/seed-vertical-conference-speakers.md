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
