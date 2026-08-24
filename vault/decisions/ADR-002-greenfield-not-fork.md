# ADR-002 — Build greenfield, borrow components, do not fork eye_of_web

**Status:** Accepted · **Date:** 2026-08-23

## Context

A GitHub survey across nine query angles found exactly one project with the same
architecture as arc_search: `MehmetYukselSekeroglu/eye_of_web` (316★, MIT,
InsightFace + Milvus + Postgres + crawler). Everything else fell into four
buckets — dead (`dsys/match`, `EagleEye`, `CompreFace`), wrappers around a
third-party index (`Pimeyes-scraper`, `SmartImage`), local library managers with
no crawler (`immich`, `home-gallery`, `hydrus`), or toys (<25★).

Forking eye_of_web would plausibly have saved 3–4 weeks. We audited it across
three parallel deep reads before deciding. Full findings: [[eye_of_web-audit]].

## Decision

**Greenfield, borrowing components.**

The disqualifier is not any single bug. It is that **34k LOC with zero tests,
zero CI, and unpinned dependencies cannot be safely refactored.**

Forking would mean replacing:
- the **storage layer** — load-bearing through the schema, the 0.5× landmark
  scaling, and the dedup path
- the **ranking layer** — which does not exist; raw cosine sort only
- the **crawler** — no robots.txt, no rate limit, no retry, no persistence

That is the whole system, rewritten with no ability to detect what broke, in a
codebase already shipping an inverted similarity comparison and two silently
dead search paths that its author does not appear to know about.

The parts that *are* sound — the InsightFace init, the Milvus collection shape —
are ~200 lines written from scratch in an hour.

## Components we reuse instead

| Repo | ★ | Role |
|---|---|---|
| `deepinsight/insightface` | 29.5k | Detection + embedding |
| `rom1504/img2dataset` | 4.4k | Large-scale URL→image download tier |
| `qdrant/qdrant` | 34k | Vector store |
| `facebook/ThreatExchange` | 1.4k | PDQ reference implementation |
| `cvg/LightGlue` | 4.7k | Geometric verification, if whole-image search is added |

`img2dataset` is the significant one — it is what LAION was built with, and it
moves "80% of the effort is crawler ops" closer to 40%.

## Consequences

- ~1 extra week versus forking, against a codebase we can actually change.
- Tests and CI from commit 1. This is the direct lesson: the absence of a test
  suite is *why* eye_of_web's inverted comparison survived, and *why* it was
  unforkable.
- We keep eye_of_web's string-interning schema design, which was genuinely good.
- The clone lives at `F:\Josh Brannon\scratch\eye_of_web` for reference. It is
  not a submodule and no code is copied from it.

## Note on the survey result

The gap arc_search fills is real. Nothing found combines all four of:

1. Targeted vertical crawl with a **real frontier** (not search-engine scraping)
2. **Crop-only storage** — 48 GB vs 1.3 TB
3. **Two-model ensemble re-rank** — what makes 10M-scale results usable
4. **Face-level canonicalization** — dedup and identity clustering in one pass

eye_of_web has none of 2–4 and gets 1 wrong.
