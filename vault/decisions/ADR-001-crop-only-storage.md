# ADR-001 — Store face crops only, never full scenes

**Status:** Accepted · **Date:** 2026-08-23

## Context

A face search engine needs exactly three things per indexed face: a vector, a
source URL, and enough pixels to let a human visually confirm a match.

An earlier draft of this design stored 512px WebP thumbnails of every crawled
image to support DINOv3 whole-image search and CLIP text search. At 30M images
that is **~1.2 TB** — and 94% of total storage.

[[eye_of_web-audit]] confirmed this is not a hypothetical trap: eye_of_web stores
a half-scale WebP of every scene as a Postgres `BYTEA` (~40 KB/image), with
`save_image = True` hardcoded over the parameter that would have disabled it.

## Decision

Store **only a 128px face crop** (WebP q75, 15% margin) plus landmarks at
original resolution. No whole-image pixels are persisted anywhere.

Images with zero qualifying faces have their pixels discarded the moment
detection returns empty; only a URL and a `face_count = 0` marker survive, so
recrawls skip them.

Crops live on the filesystem under `ARC_FACE_CROP_DIR`, sharded two levels by
uuid hex. Postgres stores the path, never the bytes.

Drop the DINOv3 and CLIP indexes entirely. They served whole-image and text
search, which is not this project's goal.

## Consequences

**Storage at 30M crawled / ~10M canonical faces:**

| Component | Size |
|---|---|
| Face crops, 128px WebP q75, deduped | 40 GB |
| Vectors, 512-d int8 | 5 GB |
| Qdrant HNSW graph (m=16) | 0.7 GB |
| Metadata, Postgres (string-interned) | 1.5 GB |
| PDQ + SHA1 hashes | 1 GB |
| **Total** | **≈ 48 GB** |

**~27× reduction.** Fits on a laptop SSD, and the whole index is trivially
backup-able — which matters more than it sounds four months into a crawl.

**Accepted costs:**
- No whole-image or text search. Adding it later means a recrawl.
- Crops are a lossy derivative. Mitigated by storing landmarks at **original**
  resolution with `src_width`/`src_height`, so a better model can re-align from
  the crop without re-crawling.

**Rejected alternatives:**
- *Vectors only, re-fetch source URL to display* (~8 GB). Dies to link rot at
  ~15%/yr; results silently rot into broken images.
- *Hybrid: crops only above quality p60* (~25 GB). Reasonable, but adds a code
  path for 23 GB. Revisit if the corpus passes 50M faces.

## Enforcement

- `sql/schema.sql` has no blob column on any crawl table.
- Non-negotiable #1 in the root `CLAUDE.md`.
