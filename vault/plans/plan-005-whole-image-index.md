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

## Phase 2 – Embedding in the crawl loop ✅

**Embedder + write path done 2026-08-25.** 359 tests (+4 gpu-marked, excluded by
default). Proven end to end on real corpus images: *"a photo of a fish"* returns
the fish at p=0.335 with every person at 0.000; *"a bearded man"* returns the
bearded man; scene more-like-this from the fish scores 1.000 against ~0.01.

- [x] `index/embed.py` — `ImageEmbedder`, models swappable by `ARC_EMBED_*`.
      Reports its **effective** device, never the requested one.
- [x] Named-vector collection support in `VectorStore` — `upsert_image()` /
      `search_named()`. Refuses a half-written pair, refuses a wrong dim,
      refuses an unknown vector name.
- [x] **Scene + text embedding in the crawl loop** — `index/embed_sink.py`.
      A **decorator** around the sink, not a change to the loop: `Crawler` never
      learns a model exists, and the undo is one line in `run.py`. `--no-embed`
      opts out.
      ✅ Verified live: a 60-page crawl embedded **41 images, 0 failures**, at
      `req_per_s=1.0` — **unchanged**, because at 1 rps the GPU is ~99% idle.
      Then text-searched those 41 vectors successfully.
- [x] 🔴 **A model failure does not kill the crawl.** Eight tests cover it: OOM,
      Qdrant down, undecodable image, dimension mismatch, and models that will
      not load at all. In every case the image is recorded, stays at
      `embed_state = -1`, and the crawl continues. `prepare()` returns False
      rather than raising, so a broken GPU degrades to the pre-ADR-005 crawl.
- [x] Batched (default 16), with the **tail flushed on close** — without that,
      up to `batch_size-1` images per run are fetched, recorded and never
      embedded: a leak visible only as a queue that never quite empties.
- [x] **One-off re-fetch backfill — done 2026-08-25.** The corpus is fully
      embedded: **4,753 images / 4,753 vectors**, 0 unembedded, 0 failures.
      One queue served both tiers, and the two were disjoint: 41 images needed
      faces (crawled after the sink existed, so already embedded) and 4,712
      needed embedding (crawled before it, already face-examined).
      ⚠️ Run at `ARC_CRAWL_PER_HOST_RPS=5.0` for the tail rather than 1.0.
      FOSDEM serves **no robots.txt** (404), so the 1.0 was purely our own
      choice, not a host requirement — and `--rps` can only *lower* the global,
      so going faster means the env var. See the open politeness defect in
      [[plan-001-crawl-tier]]: buckets are keyed per hostname and both FOSDEM
      hostnames share one IP, so the server saw ~2x whatever was configured.
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

## Phase 4 – Query ✅

**Done 2026-08-25.** 392 tests. Verified against the live 4,753-image index.

- [x] **Text → image.** `"a conference logo"` returns the actual FOSDEM 2016
      flyer — a face-less image the old engine discarded entirely. That single
      result is ADR-005's whole argument working.
- [x] **Image → image**, by upload (`POST /similar`) and as *more like this*
      from any result (`GET /similar/{id}`). The latter reuses the **stored**
      vector rather than re-fetching and re-embedding — the point id is the
      image id, so it is a retrieve, not a round trip to the source host.
- [x] Face mode kept, unchanged, as one tab of three.
- [ ] 🔴 **PDQ near-duplicate collapse before display — STILL OPEN.** The grid
      currently shows the same image republished across years more than once,
      and the UI says so rather than pretending otherwise. Already a prerequisite
      for calibration ([[research/first-index-and-calibration-preview]]: 223 of
      225 duplicate pairs have *different* sha1). For an image grid it is more
      visible — without it the first page fills with the same sponsor logo.
- [x] ⚠️ ~~**Result display is re-fetch**~~ — **decided: the SERVER fetches, and
      caches.** The obvious alternative is `<img src="https://theirsite/…">`,
      letting the browser do it: free, instant, no politeness budget. It is the
      one thing this project cannot do — the source host would then get a
      request per result *from the user's own address*, revealing exactly which
      images they looked at. For a face search engine that inverts the premise.
      "The corpus and the query stay on your hardware" has to include **which
      results you looked at**.
      So: `/thumb/{id}` proxies through the same politeness layer, behind a
      **bounded, memory-only LRU** (64 MB default). Bounded because ADR-001
      forbids persisting scene images — an unbounded or on-disk cache becomes
      exactly the store non-negotiable #1 exists to prevent. Worst case is a
      slow page, never a full disk.

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
