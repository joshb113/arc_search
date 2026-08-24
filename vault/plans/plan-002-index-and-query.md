# plan-002 – Index and query (week 2)

**Status:** 📋 Planned — blocked on [[plan-001-crawl-tier]] reaching a usable corpus
**Goal:** Put a model in the loop and answer a query. At the end of this plan the
project is already a working product.

**Exit criterion:** ~70k faces indexed, query by upload returns plausible hits.

---

## Phase 1 – Face writer ⬜

- [ ] Qdrant collection — 512-d, int8 scalar quantization, HNSW m=16
- [ ] Postgres `face` writer — bbox and landmarks at **original** resolution, with
      `src_width`/`src_height` alongside. Never pre-scaled to match a derived
      artifact; that is non-negotiable #3 and eye_of_web's exact mistake.
- [ ] Crop storage — 128px + 15% margin, WebP q75, on the filesystem

## Phase 2 – Wire it into the crawl ⬜

- [ ] Swap `MetadataSink`'s replacement for `FaceIndexSink` at the **same call site**
      in `run.py`, so the crawl loop does not change
- [ ] Work queue — `image_unexamined_idx` is a partial index on `face_count < 0`,
      built for exactly this
- [ ] Compute and store PDQ — the column exists, the BK-tree exists, nothing writes
      it yet, so near-dup dedup is currently SHA1-only

## Phase 3 – Query ⬜

- [ ] Upload-a-photo endpoint, bound to `127.0.0.1`
- [ ] Report reject-stat breakdown (`too_small` / `too_blurry` / `bad_pose`)

---

## Open questions

- **`url_path` interns something that never repeats.** It is 1:1 with `page`, so it
  pays a row and a UNIQUE index per page for no deduplication — the same reasoning
  that sent image paths inline in [[ADR-003-store-image-urls]]. Run the same
  measurement against it before the corpus is large enough to make the migration
  painful.
- **`serve/` binds to localhost by design.** A local index and a reachable service
  are different legal objects; see README "Legal posture" before changing it.
