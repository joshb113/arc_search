# plan-005 – Whole-image index (scene + text)

**Status:** 📋 Planned — this is the critical path
**Follows:** [[ADR-005-image-search-is-primary]] (accepted 2026-08-25)
**Goal:** Make arc_search an image search engine. Every crawled image becomes
searchable by text and by visual similarity, whether or not it contains a face.

**Exit criterion:** text query and image query both return plausible, attributable
results over the full corpus, with **zero images excluded for lacking a face**.

---

## 🔴 The one architectural decision: embed at crawl time, not by re-fetch

[[plan-002-index-and-query]]'s backfill re-fetches every image because the bytes
were discarded at crawl time. That worked: 2,468 images, 41 minutes, zero
failures. **It does not scale, and the arithmetic is not close:**

| | images | at 1 rps |
|---|---|---|
| face backfill (done) | 2,468 | 41 min |
| current corpus | 4,712 | 1.3 h |
| **scale target** | **30,000,000** | **347 days** |

Faces were a *subset* — 2,828 faces from 4,712 images, and only images that
passed a face gate ever needed a second fetch. Whole-image embedding applies to
**every image**, so re-fetch stops being a backfill and becomes a second full
crawl at politeness rates.

**Therefore: the embedding step moves into the crawl loop**, at the
`FaceIndexSink` call site `run.py` already has. The crawler holds the decoded
bytes for exactly as long as it takes to hash and measure them; that is the
moment to embed, before they are dropped. Non-negotiable #1 is untouched —
nothing is persisted, the bytes still die in the same function.

The re-fetch backfill survives for exactly one job: **the ~4,712 images already
crawled**, which is 1.3 h and a one-off. After that it is a repair tool, not the
pipeline.

⚠️ **This inverts a plan-002 assumption.** That plan deliberately kept the model
out of the crawl loop so failures stayed cheap to read. That was right for
bring-up and is wrong at 30M. The GPU does 179 img/s and politeness allows 1
req/s, so the model cannot be the bottleneck — but it *can* stall the loop if it
throws, and the crawl must survive a model failure without losing the frontier.

---

## Phase 1 – Collections and schema ✅

**Three collections, not one with named vectors.** ADR-005 left this open;
measurement closes it. Qdrant's named vectors put several vectors on **one
point**, which requires one identity. The granularities differ:

```
faces   1 point per FACE    2,828 points   (a crowd photo makes several)
scene   1 point per IMAGE   4,712 points
text    1 point per IMAGE   4,712 points
```

Faces cannot share points with scene/text. Scene and text *could* share a
collection via named vectors, since both are per-image — worth doing, since it
halves point overhead and keeps their payloads in step.

**Done 2026-08-25.** 342 tests (was 327), lint clean, no model involved.

- [x] `CollectionSpec` in `config.py`; `IndexSettings.face_spec()` /
      `.image_spec()`. `collection`/`vector_dim` stay bound to their existing
      env vars so every pre-ADR-005 call site is untouched.
- [x] `VectorStore(cfg, client, spec=None)` — defaults to the face collection.
      Dimension validation now follows the **spec**, not the global config; a
      test pins that a 768-d scene vector is not accepted against the face 512.
- [x] Both keep float32 originals + int8 (`rescore=True`), per ADR-005.
- [x] `image.embed_state` + two partial indexes, migrated on both databases.
      `-1` unembedded / `1` embedded / `-2` reserved. **No 0** — `face_count`
      uses 0 for *examined, found nothing*, so a reader would read 0 here as
      failure rather than success.
- [x] `unembedded_images()` / `unembedded_count()` / `mark_embedded()` /
      `embed_counts()`, keyset-paginated like the face queue.

**Measured while building it:** `image_id` works directly as a Qdrant point id,
verified to 2**63-1 against the live server. So the whole-image collection needs
**no mapping table** (unlike `face.qdrant_id`), deletion is by id rather than a
payload filter, and an orphaned vector is *structurally impossible* — the point
id and the row id are the same thing.

## Phase 2 – Embedding in the crawl loop ⬜

- [ ] Scene + text embedding at the existing sink call site, from bytes already
      in hand
- [ ] 🔴 **A model failure must not kill the crawl.** The frontier is durable and
      the crawl is a five-hour job; an OOM or a corrupt-image exception has to
      leave the image re-embeddable and let the loop continue. Same lesson as
      `backfill._process`, which catches broadly on purpose.
- [ ] Batch across the loop — the GPU wants batches (179 img/s at batch 7 vs 70
      at batch 1) and the crawl produces images one at a time
- [ ] One-off re-fetch backfill for the 4,712 already-crawled images
- [ ] ⚠️ **Whole-image embedding has no quality gate**, unlike faces. There is no
      `too_small`/`bad_pose` equivalent — every image gets a vector. The only
      exclusion is `min_image_dim`, which is uncalibrated (below).

## Phase 3 – Face-less images become first-class ⬜

- [ ] 🔴 **Remove the `face_count == 0` tombstone.** Four sites in
      `index/dedup.py`: `:156` (load), `:165` (check_bytes), `:181` (mark_barren),
      `:194` (register). Under a face goal, "no face, never look again" is a good
      optimization; under an image goal it deletes most of the corpus.
- [ ] ⚠️ **`min_image_dim = 48` is now underived.** It was derived from
      `min_face_px` — *"an image whose shorter side is smaller than the smallest
      face we would accept cannot contain a qualifying face"* — and that argument
      is void when face-less images are wanted. `test_config.py` pins the
      relationship and **will need to change**; do not just re-type the number.
      Re-measure under non-negotiable #5: what is the smallest image worth
      indexing for *visual* search?
- [ ] Revisit `deny_patterns`. They were tuned to discard face-less images —
      run 2 cut 33 images to 8 and called it a 4.9× bandwidth win. Those 25
      sponsor logos and venue maps are **corpus** now.
      ⚠️ But 86 identical ccc logos are still junk. The criterion should become
      **repetition, not facelessness**, with PDQ doing the work.

## Phase 4 – Query ⬜

- [ ] Text → image search
- [ ] Image → image search (upload, or "more like this" from a result)
- [ ] Face mode stays, unchanged, as one tab of the same UI
- [ ] 🔴 **PDQ near-duplicate collapse before display.** Already a prerequisite
      for calibration ([[research/first-index-and-calibration-preview]]: 223 of
      225 duplicate pairs have *different* sha1). For an image grid it is more
      visible — without it the first page fills with the same sponsor logo.
- [ ] ⚠️ **Result display is re-fetch, and this is the weakest point in the
      design.** A face result shows a crop we own; an image result is remote
      pixels, and per the brain stem re-fetch must pass politeness and robots.
      A 20-result grid concentrated on a few hosts renders over ~20 s at 1 rps.
      ADR-005 notes a size-capped evictable LRU thumbnail cache as the middle
      path — bounded, so non-negotiable #1's intent survives. **Undecided.**

---

## Open questions

- **Which models, finally.** [[research/image-model-bringup]] measured
  `dinov2-base` (768-d, 179 img/s) and `siglip2-base-patch16-384` (768-d,
  93 img/s), both discriminating correctly on real corpus images.
  ⚠️ **`facebook/dinov3-*` is `gated=manual`** — DINOv2 was a substitute. If
  DINOv3 is the intended model, access must be granted; dims and throughput will
  differ and the 280 GB target moves with them.
- **Truncation is uncosted.** ADR-005 withdrew the 512-d option because the spike
  could not test it (n=7, one negative, and a structurally invalid PCA arm).
  Deciding it needs hundreds of images across known groups. Worth doing — it is
  the difference between 280 GB and ~170 GB.
- **Does whole-image retrieval need `rescore=True`?** ADR-005 kept originals
  because nobody has measured whether quantization hurts scene search the way it
  hurts face search. Measuring it is worth 184 GB.
- **HNSW has never been exercised here.** `indexed_vectors_count: 0` — the
  collection is under Qdrant's 10,000 `indexing_threshold`, so every query so far
  has been a full scan. **No latency number in this project means anything for
  scale yet.**
- **Where does the second venv go?** `.venv-spike` proved the models; torch
  alongside the fragile `--no-deps` InsightFace install in `.venv` is a real
  packaging question, not a formality.

## What this plan does *not* change

The crawler, frontier, politeness, robots handling, SHA1/PDQ dedup, the Postgres
schema's interning, resume-after-kill, the two-store write order, and the
`serve/` skeleton are all source-agnostic and stay. ADR-003 (store the image URL)
and ADR-004 (no uncalibrated tombstones) both turn out to have paid for a goal
neither was written for: the URLs make re-embedding possible, and the absence of
tombstones means **all 1,886 face-less images are still present and re-examinable**
rather than needing a recrawl.
