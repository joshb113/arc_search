# Re-deriving `min_image_dim` for visual search

**Date:** 2026-08-25
**Why:** [[ADR-005-image-search-is-primary]] voided the argument this number was
derived from, and `test_config.py` was still enforcing it.

---

## The dead derivation

`min_image_dim` was justified by one sentence:

> An image whose shorter side is smaller than the smallest face we would accept
> cannot contain a qualifying face.

True, and irrelevant once image search is primary. **A 60px logo has no face in
it and is corpus.** The test pinning `min_image_dim <= min_face_px` was worse
than no test: a green assertion enforcing reasoning nobody believes any more,
which is how a dead argument survives a change of goal.

## The measurement that replaces it

Same method as [[research/face-model-bringup]] used for `min_face_px`: take real
corpus images, downscale, and ask how far the embedding drifts from its own
full-resolution self. 14 images with a short side ≥ 180 px.

| short side | scene mean | scene min | text mean |
|---|---|---|---|
| 160 | 0.996 | 0.993 | 0.994 |
| 128 | 0.988 | 0.974 | 0.988 |
| 96 | 0.965 | 0.860 | 0.976 |
| 80 | 0.953 | 0.867 | 0.965 |
| 64 | 0.938 | 0.856 | 0.952 |
| **48** *(current)* | **0.906** | **0.831** | **0.936** |
| 32 | 0.840 | 0.754 | 0.909 |
| 24 | 0.774 | 0.618 | 0.881 |
| 16 | 0.634 | 0.230 | 0.813 |

### 🔴 Text degrades far more slowly than scene

At 32 px the scene vector has fallen to 0.840 while text is still 0.909. Text is
the **primary** mode under ADR-005, so a gate tuned to protect scene similarity
would discard images that text search can still answer with.

They are different jobs and they do not want the same threshold. That is an
argument for **not** having one gate pretend to serve both.

## The decision: 48 stays, for a completely different reason

Not because of `min_face_px`. Because of the asymmetry ADR-004 established:

- **Excluding an image at crawl time is irreversible.** Nothing stores scene
  pixels, so a rejected image needs a whole recrawl to recover.
- **Admitting a marginal image is reversible.** It costs a filter at query time,
  changeable without touching the corpus.

So the gate belongs low, and its only job is to skip tracking pixels and spacer
GIFs — the job `min_image_bytes` already does from the other direction. It is
not a quality judgement, and quality judgements do not belong at admission.

**Measured on the live corpus, 48 excludes 0 of 4,753 images.** It is not
currently binding at all, which is exactly where an uncalibrated gate should sit.

| gate | would exclude |
|---|---|
| 48 | 0 (0.0%) |
| 64 | 1 (0.0%) |
| 96 | 29 (0.6%) |
| 128 | 176 (3.7%) |

## ⚠️ What this does *not* establish

- **n=14.** Small, and the variance is wide — scene mean 0.965 at 96 px against a
  min of 0.860 means one image degrades far worse than the rest. The mean is not
  the number to design against.
- It measures **self-similarity under downscaling**, not whether a small image is
  *useful* to a searcher. Those are different questions and only the first one is
  cheap to measure.
- The right home for this curve is probably **query time, not crawl time**:
  down-weighting a 32px result in scene mode is a ranking decision, and ranking
  can change without a recrawl. Recorded here rather than acted on.

## Also settled while here: the `face_count = 0` tombstone stays

plan-005 Phase 3 called for removing it, on the grounds that under an image goal
it "would discard 40% of the corpus". **That premise was wrong** — traced through
the code rather than assumed:

- `BARREN` and `EXACT_DUP` are handled **identically** in `PostgresWriter.handle`:
  both link provenance and return. An image in `_barren` is already in the
  `image` table.
- Nothing outside `dedup.py` reads `BARREN` at all.
- The embed path never consults `face_count`. Face-less images were already
  first-class: crawled, recorded, embedded, searchable.

What `face_count = 0` actually means is *"the detector has looked, do not run it
again"*, which stays correct under an image goal. It gates the **face** queue,
not corpus membership. Left in place.
