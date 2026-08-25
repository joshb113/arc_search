# ADR-005 — Image search is primary; face search is one mode of it

**Status:** ✅ Accepted, 2026-08-25 · **Date:** 2026-08-25
**Amends:** [[ADR-001-crop-only-storage]] (partially reverses)
**Amends:** root `CLAUDE.md` — non-negotiables #1 and #4, and the scale target
**Prerequisite:** Docker's disk image must move to F: before indexing at scale
(see *Storage policy* below)

## Context

The project has been built for two days as a **face** search engine: a photo
goes in, the same person comes back. [[ADR-001-crop-only-storage]] followed from
that goal and dropped whole-image and text search explicitly —

> Drop the DINOv3 and CLIP indexes entirely. They served whole-image and text
> search, which is not this project's goal.

The goal has been restated: **arc_search is an image search engine.** Face
similarity from a user-uploaded photo is a secondary mode, not the product.

ADR-001 is therefore being relitigated deliberately, which is what an ADR is for.
What follows is the *narrow* reversal — it restores the two dropped indexes and
leaves crop-only storage standing.

## Decision

**1. Two whole-image indexes are restored.**

| index | serves | over |
|---|---|---|
| text embedding (CLIP/SigLIP class) | text → image | all images |
| scene embedding (DINOv3 class) | image → image, whole scene | all images |
| face embedding (ArcFace, exists) | photo → same person | faces only |

Non-negotiable #4 ("index only the face embedding") is amended to: **index only
embeddings that a query actually searches.** Its original target — eye_of_web's
4-dimensional HNSW index over bbox coordinates — remains forbidden. Landmarks and
bbox stay payload.

**2. Storage stays crop-only. Non-negotiable #1 survives intact.**

No whole-scene pixels are persisted. Embeddings are computed at index time from
bytes that are then discarded, exactly as face crops are today. Display of
whole-image results is by **re-fetch from the source URL** — ADR-001's rejected
alternative, adopted knowingly (see Risks).

**3. `face_count = 0` stops being a corpus tombstone.**

Today `index/dedup.py:156` and `:194` treat `face_count == 0` as *barren, never
look again*. Under a face-search goal that is a correct optimization. Under an
image-search goal it would discard 40% of the corpus. Face-less images are now
first-class members: they carry scene and text embeddings and are returned by
queries. `face_count` reverts to meaning only what its name says — how many faces
this image has — and stops gating whether the image exists in the index.

⚠️ **This is a prospective risk, not a historical loss.** Measured against the
live database 2026-08-25:

| `face_count` | images | |
|---|---|---|
| `> 0` | 2,826 | have faces |
| `-2` | **1,886** | examined, empty, **re-examinable** |
| `0` | **0** | permanently tombstoned |
| `-1` | 0 | unexamined |
| **total** | **4,712** | |

**[[ADR-004-uncalibrated-gates-are-not-tombstones]] pre-solved this migration.**
Because every gate that has run so far is uncalibrated, empty results were written
as `-2` rather than `0`, so not one face-less image has been permanently discarded.
The 1,886 are 40% of the corpus, they are precisely what an image search needs,
and they still carry their URLs. **No recrawl is required for them** — an extended
backfill can embed them in place.

`dedup.py` must still be fixed before plan-003 calibrates a gate, because a
*calibrated* empty result legitimately writes `0` and would start destroying
corpus from that moment on.

**4. Crawl gates lose their face derivation.**

`min_image_dim = 48` is currently *derived* from `FaceSettings.min_face_px`, with
`test_min_image_dim_is_not_above_the_minimum_face_size` pinning them together.
That derivation is void: the smallest useful *image* has nothing to do with the
smallest detectable *face*. Under non-negotiable #5 the new floor must be
measured, not chosen, and the pinning test must be replaced rather than deleted.

The FOSDEM and ccc `deny_patterns` need the same review. Run 2's celebrated
4.9× bandwidth saving — 33 images down to 8 — discarded 25 sponsor logos and
venue maps that are now corpus. Logos likely still deserve denying, but on
*repetition* grounds (PDQ collapses them), not on *facelessness*.

## Consequences

🔴 **A first draft of this ADR put the total at ~105 GB. That was wrong**, and the
correction is the most important number in this document. It assumed the new
collections would cost int8 only. **Qdrant stores the float32 originals as well**
— confirmed against the live `faces` collection, which is created with
`on_disk=True` originals plus an `always_ram` int8 copy (`index/vectors.py:114`),
and already recorded in [[plans/INDEX]]: *"20.5 GB against 5.1 GB at 10M faces,
and the largest single line in the disk budget."*

Originals are **4 bytes per dimension**, so they dominate everything. Whole-image
indexes run over **30M images**, not 10M faces, which multiplies the mistake by
three. Projections below use the project's own model (`dim × 4` originals +
`dim × 1` int8), which reproduces the measured 20.5/5.1 GB figures exactly.

**Updated 2026-08-25 with measured dimensions** from
[[research/image-model-bringup]]. The earlier table guessed 1024-d for scene and
768-d for text. Both models are **768-d, measured**:
`facebook/dinov2-base` 768, `google/siglip2-base-patch16-384` 768.

| Component | ADR-001 | A: as-is | C: no originals |
|---|---|---|---|
| Face crops (measured 1,921 B) | 40 GB | 19 GB | 19 GB |
| Face vectors, 512-d, 10M | 5 GB | 26 GB | 26 GB |
| Scene vectors, 768-d, 30M | — | 115 GB | 23 GB |
| Text vectors, 768-d, 30M | — | 115 GB | 23 GB |
| Metadata + PDQ + SHA1 | 2.5 GB | 5 GB | 5 GB |
| **Total** | **≈ 48 GB** | **≈ 280 GB** | **≈ 96 GB** |

- **A** — native 768-d, Qdrant configured exactly as `faces` is today
  (`on_disk=True` float32 originals **plus** the int8 copy).
- **C** — A, minus the float32 originals for the two whole-image collections
  (quantized-only, no rescore). Face search keeps its originals, because ADR-001's
  reasoning still applies: the recall lost to quantization lands on exactly the
  hard matches the face engine exists to find. Whether whole-image retrieval is
  more tolerant is **not established** — a real open question, not a formality.

⚠️ **Option B (truncate to 512-d) has been withdrawn from this table.** It
assumed Matryoshka-style truncation is free. Neither model advertises MRL
training, and the spike could not settle it: n=7 images with one negative, and
the PCA arm was structurally invalid (7 samples yield ≤6 components, which is why
512/384/256 all returned an identical figure). Naive truncation *appearing* to
improve separation was noise. **It may still be the right answer — it is simply
uncosted, and quoting a number for it would be inventing one.**

**So the scale target is a decision, not a derivation: 96 GB or 280 GB, on one
config flag.**

### Storage policy: A — keep the originals. New target 280 GB.

**Decided 2026-08-25.** The two whole-image collections are configured exactly as
`faces` is today: `on_disk=True` float32 originals **plus** the int8 copy, with
`rescore=True` on search.

The reasoning is the one ADR-001 used for faces, applied consistently: int8
quantization loses recall, and rescoring against the originals recovers it. The
alternative saves 184 GB by betting that whole-image retrieval tolerates
quantization better than face retrieval does — and **nobody has measured that**.
Buying 184 GB with an unmeasured assumption is the trade this project refuses
everywhere else (non-negotiable #5). Disk is the cheap resource here; a silently
degraded index is not.

This is reversible in the cheap direction. Turning originals **off** later is a
config change plus a re-index. Discovering you need them after building on
quantized-only would mean re-embedding 30M images.

### ✅ Prerequisite satisfied — Docker's disk moved to F:

`qdrant_data` and `pg_data` are Docker *named volumes*, so they live inside the
WSL2 VM disk, which defaults to `C:`. Measured 2026-08-25 **before** the move:

```
C:\Users\<user>\AppData\Local\Docker\wsl\disk\docker_data.vhdx   3.7 GB
C: free  9.8 GB          F: free  504 GB
```

280 GB of vectors cannot land on a disk with 9.8 GB free, and Postgres was in the
same volume. **Relocated to `F:\DockerDesktopWSL`** via Docker Desktop → Settings
→ Resources → Advanced → *Disk image location*, which keeps native ext4
performance inside the VM. A Windows→WSL2 bind mount was considered and rejected:
the 9p/virtiofs layer is a poor fit for a vector store's random I/O.

Verified after the move — nothing was lost, and all three stores still agree:

| | |
|---|---|
| Qdrant `points_count` | 2,828 |
| Postgres `face` rows | 2,828 |
| crop files on disk | 2,828 |
| images / pages | 4,712 / 9,363 |
| **free at `/qdrant/storage`** | **954 GB** |

954 GB against a 280 GB target is 3.4× headroom.

What has *not* changed is the property the target was protecting: no option here
stores whole-scene pixels, and option C still fits a laptop SSD.

**The recrawl is much smaller than ADR-001 implied.** ADR-001 warned that adding
these indexes later "means a recrawl." Measured, that is mostly not true here.
Every image the crawl *kept* — all 4,712, including the 1,886 with no faces — can
be re-fetched from its stored URL and embedded by an extended backfill. What
genuinely needs recrawling is only what the crawl *refused*: URLs excluded by
face-motivated `deny_patterns`. That is a re-run of known seeds with a relaxed
filter, not a rebuild.

This is the second time ADR-003 (store the image URL) has paid for itself, and the
first time ADR-004 has. Neither was written with this goal in mind.

**plan-003 is no longer the critical path.** It is entirely face-threshold
calibration. It stays valid for the secondary face mode and should be resequenced,
not cancelled. The PDQ prerequisite it carries becomes *more* important, not less:
PDQ is now the primary defence against a result grid full of the same logo.

## Risks

🔴 **Re-fetch-to-display is a harder problem for a grid than for a face result,
and ADR-001 did not price this.** A face result page shows crops we own; only the
"view source" link was remote. An image result page *is* remote images, every one
of them. Three consequences:

- **Latency.** The brain stem records that re-fetching must pass politeness and
  robots like any other crawl. At 1 rps per host, a 20-result grid concentrated
  on a few hosts renders over seconds, not milliseconds.
- **Rot.** ADR-001's original objection stands — ~15%/yr link rot, and results
  "silently rot into broken images." Worse here, because the images are the
  product rather than a confirmation aid.
- **Posture.** Hotlinking at query time is a different relationship with a host
  than crawling it politely once. Non-negotiable #6 is about honest crawling; this
  deserves its own thought.

**Mitigation to consider (not decided here):** a bounded LRU thumbnail cache —
capped at a fixed budget and evictable, so it is a *cache*, not the 1.2 TB store
ADR-001 rejected. This preserves non-negotiable #1's intent (no unbounded pixel
store) while making the common result path local.

⚠️ **`store.REFETCH_SCHEME` hardcodes https** and scheme is not a stored column.
Tolerable when re-fetch was an indexing-time detail; it is user-visible now.

⚠️ **Two new models enter the venv,** which currently has a fragile InsightFace
install (`--no-deps`, the headless-opencv conflict, the antelopev2 nesting bug).
Adding torch-based models beside onnxruntime is its own bring-up, and should get
a `research/` note like [[face-model-bringup]] rather than being assumed to work.

## Enforcement

- `sql/schema.sql` still has no blob column on any crawl table.
- Any thumbnail cache must be size-capped and evictable, and must live outside
  Postgres.
- The replacement for `test_min_image_dim_is_not_above_the_minimum_face_size`
  must pin the new floor to whatever measurement produces it.

## Open — blocking

- ~~🔴 **Qdrant vector-storage policy for the two new collections**~~
  **DECIDED 2026-08-25: option A, keep the originals, target 280 GB.** See
  *Storage policy* above. Root `CLAUDE.md` updated in the same commit.
  ✅ Its prerequisite is met: Docker's disk moved to `F:\DockerDesktopWSL`,
  954 GB free at `/qdrant/storage`, all 2,828 vectors intact across the move.
- ~~🔴 **Do the models run here at all?**~~ **ANSWERED 2026-08-25** →
  [[research/image-model-bringup]]. Yes, in a separate venv: torch
  **2.11.0+cu128** on Python 3.14, **sm_120** confirmed by a real GPU matmul.
  `facebook/dinov2-base` **768-d, 179 img/s**;
  `google/siglip2-base-patch16-384` **768-d, 93 img/s**. Both discriminate on
  real corpus images — DINOv2 separates a photo of a fish from six photos of
  people by **+0.306**; SigLIP2 scores *"a photo of a fish"* at **0.335** on the
  fish and **0.000** on every person.
  🔴 The silent-CUDA trap is live here too: the **default PyPI torch wheel is
  CPU-only on Windows**. `--index-url .../whl/cu128` is mandatory, and Blackwell
  needs ≥12.8 regardless.
  ⚠️ `facebook/dinov3-*` is **`gated=manual`**; DINOv2 was substituted. If DINOv3
  is the intended model, access must be granted before these numbers are final.
- 🔴 **Is dimension truncation viable?** Withdrawn from the storage table as
  uncosted. Neither model advertises Matryoshka training, and the spike's n=7
  sample could not answer it. Needs a proper measurement — hundreds of images
  across known groups — before 512-d can be priced.

## Open — non-blocking

- Which text/scene models specifically — SigLIP2 vs CLIP, DINOv3 size class.
- Whether one Qdrant collection with named vectors beats three collections.
  `VectorStore` currently assumes one collection at one `cfg.vector_dim`, and
  `ensure_collection()` deliberately refuses to reconcile a changed dim — correct,
  and it means this is a migration to design, not a config edit.
- Whether face search stays on the same upload path or becomes a mode switch.
- Whether `deny_patterns` should switch from facelessness to *repetition* as the
  criterion, with PDQ doing the work. 86 identical ccc logos are still junk.
