# plan-002 – Index and query (week 2)

**Status:** 🟡 Active — Phase 1 complete
**Goal:** Put a model in the loop and answer a query. At the end of this plan the
project is already a working product.

**Exit criterion:** ~70k faces indexed, query by upload returns plausible hits.

---

## Phase 1 – Face writer ✅

Storage layer only — no model in the loop yet. **279 tests** (was 233), lint and
format clean, full suite green against live Postgres + Qdrant.

- [x] Qdrant collection — 512-d, COSINE, int8 scalar quantization (`quantile=0.99`,
      `always_ram`), HNSW `m=16` / `ef_construct=200`, originals `on_disk`.
      `src/arc_search/index/vectors.py`: `VectorStore` with `ensure_collection`,
      `verify`, `upsert`, `delete_image`, `search`, `count`.
- [x] Postgres `face` writer — `record_faces` / `mark_examined` / `face_counts` /
      `unexamined_images`. bbox and landmarks at **original** resolution with
      `src_width`/`src_height` alongside; non-negotiable #3, pinned by
      `test_record_faces_stores_geometry_at_original_resolution`.
- [x] Crop storage — 128px + 15% margin, WebP q75, sharded two levels by uuid hex.
      Already existed in `faces.py` but had **never been imported by a test** —
      cv2 was not in the venv at all. Now covered, gate included.

### 🔴 The write order is the correctness argument

Two stores, no distributed transaction. `image.face_count` is the commit marker
for **both**:

```
1. Postgres: DELETE old face rows, INSERT new     (face_count still -1)
2. Qdrant:   delete by image_id filter, then upsert
3. Postgres: UPDATE image SET face_count = N      <- commits the pair
```

```
face_count > -1  =>  both stores are complete and agree for that image
face_count == -1 =>  either store may hold partial state; it is disposable
```

A crash before step 3 leaves the image on the work queue, and both stores clear
before rewriting, so a re-run converges instead of duplicating. If `record_faces`
is ever "tidied up" to set the count it just wrote, a crash between the stores
leaves faces **no query can return and nothing reports**. Three tests exist
solely to stop that.

### 🔴 Client and server versions are coupled

`qdrant/qdrant` in docker-compose.yml and `qdrant-client` in pyproject.toml must
stay on the same minor — both are now **1.19**. A `>=1.9` floor had let the
client reach 1.19 against a pinned 1.9.2 server, which left **no working search
path in either direction**: `query_points` 404s on a server older than 1.10, and
`search` was deleted from the client after 1.13. Writes succeeded throughout, so
the collection would have filled normally and failed at the first *query*. Found
by probing the live server before writing against it, not by reading changelogs.

### Measured: the disk budget holds

128px WebP q75 crops came in at **1,738 B** mean (n=20, photo-like input),
against ADR-001's 4 KB/face budget.

Re-measured on **1,300 real crops** after the backfill: mean **1,921 B**
(median 1,884, p90 2,460, max 4,524) against ADR-001's 4,096 B budget.

| At 10M faces | |
|---|---|
| crops | 19.2 GB |
| Qdrant int8 (RAM) | 5.1 GB |
| Qdrant originals (disk) | 20.5 GB |
| **total** | **44.8 GB** vs the 49.6 GB target |

~4.8 GB left for Postgres metadata. The budget holds on measured data.

## Phase 2 – Put the model in the loop 🟡

**The engine answers a query.** A photo goes in, the right person comes back at
cosine **1.0000** with the next-best at **0.085**, carrying its weak label and
its source page. That is the product working end to end for the first time.

- [x] **Backfill runner** — `src/arc_search/index/backfill.py`, 13 tests.
      Re-fetches through `Fetcher`, so robots.txt and rate limiting apply
      identically; it is a crawler, not a batch job, and there is no back door.
      ⚠️ **The bytes are gone** (non-negotiable #1), which is exactly why ADR-003
      put the URL back on `image`.
      🔴 **The cursor advances even on failure.** A failed image keeps
      `face_count = -1`, so it is still first in the queue — parking the cursor
      would spin forever on one dead URL while looking busy.
      ⚠️ **Politeness is per-process.** Running this alongside `crawler.run`
      against the same host doubles the rate. Stop the crawl or pass `--rps`;
      nothing can detect it for you.
- [x] ⚠️ **Scheme is not a stored column.** `store.REFETCH_SCHEME` assumes https.
      Held for the whole FOSDEM archive — 0 fetch failures across the drain.
- [ ] **Dead links are retried forever.** A 404 stays at `-1` and is re-attempted
      every run. Correct for a transient 503, wrong for a permanently gone file.
      That is the TTL reaper in [[plan-004-scale-and-hygiene]], not this plan.
- [ ] Swap `MetadataSink`'s replacement for `FaceIndexSink` at the **same call site**
      in `run.py`, so the crawl loop does not change
- [ ] 🔴 **Compute and store PDQ — now a PREREQUISITE for calibration, not a
      nice-to-have.** Measured: of 225 labeled genuine pairs, **149 (66%) are the
      same photo re-published in a later year**, and **223 of 225 have different
      sha1** because they were re-encoded. Exact-hash dedup cannot see them, so
      they inflate every recall number by 2–5 points and `eval.calibrate` would
      inherit that bias wholesale. The column, the BK-tree and the tests all
      exist; nothing writes it (`dedup.loaded pdq=0`).
      → [[research/first-index-and-calibration-preview]]
- [x] ~~**Does the model tier run at all?**~~ **Yes** — verified 2026-08-24,
      antelopev2 on CUDA at **49 img/s**, real faces on real FOSDEM photos.
      Needs `insightface>=1.0.1` (0.7.3 will not build on 3.14) and a manual
      flatten of the antelopev2 unzip. → [[research/face-model-bringup]]
- [x] ~~**Decide `min_face_px` before running the backfill.**~~ →
      [[ADR-004-uncalibrated-gates-are-not-tombstones]]. Now **48**, derived from
      an embedding-decay curve, and `face_count = -2` means a provisional pass
      can no longer tombstone anything. The backfill is re-runnable after
      calibration at the cost of one re-fetch per gated image.

## Phase 3 – Query ⬜

- [ ] Upload-a-photo endpoint, bound to `127.0.0.1`
- [ ] Report reject-stat breakdown (`too_small` / `too_blurry` / `bad_pose`)
- [ ] 🔴 **Do not present a score as confidence.** `t_plausible`/`t_strong`/
      `t_near_certain` are UNCALIBRATED placeholders and `calibrated` is False.
      Return raw cosine plus the flag; verdicts wait for
      [[plan-003-precision]]. Non-negotiable #5.

---

## Open questions

- **`url_path` interns something that never repeats.** It is 1:1 with `page`, so it
  pays a row and a UNIQUE index per page for no deduplication — the same reasoning
  that sent image paths inline in [[ADR-003-store-image-urls]]. Run the same
  measurement against it before the corpus is large enough to make the migration
  painful.
- **`serve/` binds to localhost by design.** A local index and a reachable service
  are different legal objects; see README "Legal posture" before changing it.
