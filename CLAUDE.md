# arc_search

Self-hosted face search engine over a targeted vertical crawl.
Python. InsightFace (antelopev2/R100) + Qdrant + Postgres.

**Goal:** Yandex-class face search over a corpus you own, running entirely on
local hardware. No third-party image search. No data leaves the machine.

**Start every session by reading:**

> ### 📖 `vault/plans/INDEX.md` — current phase, milestones, open questions
> ### 📖 `vault/decisions/` — ADRs. Do not relitigate these without a new ADR.

This vault root is also the project root: engine code goes in `src/`, planning
and research live in `vault/`. There is exactly one vault, and it is here.

---

## Non-negotiables

These come from the eye_of_web audit (`vault/research/eye_of_web-audit.md`).
Each one is a failure we watched a real 34k-LOC system commit. Do not repeat them.

1. **Never store full-scene images.** Face crops only, on the filesystem, never
   as Postgres blobs. Budget is ~4 KB/face. See `ADR-001`.
2. **Dedup before the GPU, in the right order.** SHA1 exact-match first, then
   PDQ perceptual, *then* detection. Never insert before checking.
3. **Landmarks are stored at original resolution.** Never pre-scale them to
   match a derived artifact. Record source dimensions alongside.
4. **Index only the face embedding.** Landmarks and bbox are payload, not
   indexed vectors. A 4-dim HNSW index is nonsense.
5. **Thresholds are derived, never literal.** Every threshold traces to a run of
   `arc_search.eval.calibrate` against a labeled set. No magic numbers.
6. **Every crawler respects robots.txt and identifies itself honestly.** There is
   one User-Agent, it names the project, and it carries a contact URL.
7. **Tests and CI from commit 1.** The reason eye_of_web was unforkable is that
   34k LOC with zero tests cannot be safely changed.

## Layout

```
src/arc_search/
  crawler/   frontier, politeness, extraction, fetch
  index/     dedup, face detection + embedding, storage
  serve/     FastAPI query API + UI
  eval/      threshold calibration, ROC
sql/         Postgres schema (string-interned metadata)
tests/       pytest, runs in CI
vault/       research, ADRs, plans
```

## Scale target

30M crawled images → ~10M canonical faces → **~49.6 GB total on disk.**
Single machine. One GPU. If a design choice pushes past that, it needs an ADR.
