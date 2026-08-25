# arc_search

Self-hosted **image search engine** over a targeted vertical crawl.
Python. Qdrant + Postgres, with three embedding models:
text→image and scene→scene over every image, and InsightFace
(antelopev2/R100) over faces.

**Goal:** Yandex-class image search over a corpus you own, running entirely on
local hardware. No third-party image search. No data leaves the machine.

**Face similarity from an uploaded photo is one mode, not the product.** It was
the product until 2026-08-25; see `ADR-005`, which restored the whole-image and
text indexes that `ADR-001` had dropped.

**Start every session with `/start`**, or read these by hand:

> ### 📖 `vault/00_Brain/CLAUDE.md` — working state, live numbers, traps
> ### 📖 `vault/plans/INDEX.md` — plan table, status, open questions
> ### 📖 `vault/00_Brain/handoffs/handoff-latest.md` — where the last session stopped
> ### 📖 `vault/decisions/` — ADRs. Do not relitigate these without a new ADR.

This file is the **contract** — non-negotiables, layout, scale target — and
changes only with an ADR. `vault/00_Brain/CLAUDE.md` is the **working state** and
changes freely as facts change.

This vault root is also the project root: engine code goes in `src/`, planning
and research live in `vault/`. There is exactly one vault, and it is here.

## Vault layout

```
vault/00_Brain/     working state, handoffs, ISSUES_INDEX (external-cause issues)
vault/plans/        INDEX.md (table) + plan-NNN-*.md
vault/decisions/    ADR-NNN-*.md
vault/research/     audits and measurements
vault/tasks/        active / backlog / completed
.claude/commands/   /start /status /plan /adr /audit /focus /handoff /wrap-up
```

---

## Non-negotiables

These come from the eye_of_web audit (`vault/research/eye_of_web-audit.md`).
Each one is a failure we watched a real 34k-LOC system commit. Do not repeat them.

1. **Never store full-scene images.** Face crops only, on the filesystem, never
   as Postgres blobs. Budget is ~4 KB/face (measured: 1,921 B). See `ADR-001`.
   Still holds under `ADR-005`: whole-image embeddings are computed at index time
   from bytes that are then discarded, exactly as crops are. Nothing persists a
   scene. Displaying a whole-image result is a **re-fetch from the source URL**.
2. **Dedup before the GPU, in the right order.** SHA1 exact-match first, then
   PDQ perceptual, *then* detection. Never insert before checking.
3. **Landmarks are stored at original resolution.** Never pre-scale them to
   match a derived artifact. Record source dimensions alongside.
4. **Index only embeddings a query actually searches.** Amended by `ADR-005`
   from "only the face embedding" — there are now three: text, scene, face.
   Landmarks and bbox remain **payload, not indexed vectors**, which was this
   rule's real target: eye_of_web built a 4-dimensional HNSW index over bounding
   boxes, and a 4-dim HNSW index is nonsense.
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

30M crawled images → ~10M canonical faces → **~280 GB total on disk.**
Single machine. One GPU. If a design choice pushes past that, it needs an ADR.

Raised from 49.6 GB by `ADR-005` when image search became primary. Three indexes
now, not one, and the two whole-image collections cover **30M images** rather
than 10M faces. Computed from measured dimensions (both models 768-d) and
measured crop size (1,921 B) — see `vault/research/image-model-bringup.md`.

| | |
|---|---|
| face crops + vectors + metadata | ~50 GB |
| scene vectors, 768-d, 30M | ~115 GB |
| text vectors, 768-d, 30M | ~115 GB |

The cost is dominated by keeping float32 originals **alongside** the int8 copies,
so int8 candidates can be rescored against them. Dropping the originals would
land at ~96 GB, and `ADR-005` explains why that trade was refused: it buys 184 GB
with an unmeasured assumption about how well whole-image retrieval survives
quantization.

⚠️ **Docker's disk image must live on a volume with room for this.** Qdrant and
Postgres are Docker *named volumes*, so they sit inside the WSL2 VM disk — which
defaults to `C:`. Relocated to `F:\DockerDesktopWSL` on 2026-08-25; verified
954 GB free at `/qdrant/storage`.
