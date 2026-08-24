# ADR-004 — `min_face_px = 48`, and uncalibrated gates must not tombstone

**Status:** accepted, 2026-08-24
**Amends** the `face_count` tri-state in `sql/schema.sql` to four states.
**Supersedes nothing.** Provisional with respect to `arc_search.eval.calibrate`,
which is expected to revisit the number but not the mechanism.

## Context

Bringing the model up for the first time ([[research/face-model-bringup]])
surfaced that `FaceSettings.min_face_px = 64` was rejecting most real faces in
the corpus. Measured over a hash-random sample of 60 crawled images:

```
face short side px:  min=12  p10=43  p25=55  MEDIAN=68  p75=100  p90=117
min_face_px=64  keeps 24/40 detections (60%)
min_face_px=48  keeps 32/40 detections (80%)
min_face_px=40  keeps 37/40 detections (92%)
```

The threshold sat essentially on the median face size and discarded 40% of all
detections. 64 was never derived from anything — it was a placeholder, like the
`min_image_dim = 200` that rejected 9 of 9 real photos and the
`min_image_bytes = 8000` that rejected 4 of 9 before it. This is the third
instance of the same failure in this project.

The obvious fix — lower the number — is not obviously safe, because `faces.py`
argues, correctly, that small faces are the dominant source of false positives
at 10M+ scale. So the question is not "which number keeps more faces" but
**"below what size does the model stop being able to represent a face at all?"**

### The measurement that settles it

Cosine similarity of a face's embedding to its **own full-resolution
embedding**, as the source image is progressively downscaled (10 baseline faces
≥100 px, real FOSDEM speaker photos):

| face px | mean cos | min cos |
|---|---|---|
| 96 | 0.994 | 0.991 |
| 72 | 0.989 | 0.984 |
| 64 | 0.987 | 0.978 |
| 56 | 0.981 | 0.959 |
| **48** | **0.973** | **0.951** |
| 40 | 0.953 | 0.932 |
| 32 | 0.910 | 0.885 |
| 24 | 0.815 | 0.749 |

The knee is between **40 and 32 px**, where the vector falls below
`IndexSettings.canonical_threshold = 0.92` — the point at which a face stops
matching *itself*. Not at 64.

⚠️ **What this does and does not establish.** It measures self-consistency under
downscaling, not discriminability between different people. The false-positive
argument is about discrimination and requires labeled pairs, which is
`eval.calibrate`'s job. What it does establish is that the *model* does not
require 64, so the burden shifts back to the number that was never justified.

## Decision

### 1. `min_face_px: 64 → 48`

Chosen as the smallest size retaining margin above 0.92 on the **worst** case
(min 0.951), not merely the mean. 40 was rejected: its worst case (0.932) leaves
almost no headroom above the canonicalization floor.

`CrawlSettings.min_image_dim` follows to 48, preserving the derivation that an
image shorter than the smallest acceptable face cannot contain one.
`test_min_image_dim_is_not_above_the_minimum_face_size` pins the relationship
and caught this automatically.

Measured effect on the original failing sample: **5 of 6 real speakers indexed,
up from 2 of 6.**

### 2. `face_count` gains a fourth state: `-2`, `PROVISIONAL_EMPTY`

```
-2  examined under an UNCALIBRATED gate, nothing qualified  => re-examine
-1  never examined by a detector       (the crawl tier writes this)
 0  examined, no qualifying face       => barren, skip forever
>0  this many faces indexed
```

`mark_examined()` takes `calibrated: bool = False`. While it is False, an empty
result is stored as `-2` rather than `0`.

`unexamined_images()` selects `= -1`, **not** `< 0` — folding `-2` in would make
every backfill re-fetch the entire provisional set forever. A separate
`provisional_images()` drains the recheck queue after calibration.

## Why the second half matters more than the first

`face_count = 0` is a **tombstone**. `Deduper` reads it as never-look-again,
permanently, and the image's bytes are not stored — so recovering one means
re-fetching it. Every gate that can produce an empty result (`min_face_px`,
`min_det_score`, `min_blur_var`, `max_abs_yaw`) is an uncalibrated placeholder.

Writing `0` before calibration therefore takes numbers nobody has justified and
makes them **irreversible**. Had the backfill run at `min_face_px = 64`, 40% of
the corpus would have been permanently retired on the strength of a placeholder,
and the only symptom would have been a corpus that seemed smaller than expected.

That is the exact shape of every failure this project was founded to avoid: not
a crash, but a silent, plausible-looking, unrecoverable loss.

The asymmetry decides it. Indexing a marginal face is **reversible** — `quality`
is stored per face and it can be filtered at query time. *Not* indexing it is
not. So the two decisions get separated: index generously, filter at query time,
and let calibration set the query-time threshold.

`calibrated` defaults to `False` so that forgetting it costs a re-fetch rather
than a permanent loss — the cheap mistake rather than the expensive one, pinned
by `test_the_safe_default_is_the_cheap_mistake`.

## Consequences

- A provisional backfill is **fully re-runnable** after calibration. Cost is one
  re-fetch per gated image, at crawl politeness rates.
- `face_counts()` reports `provisional` alongside `barren`; the two must not be
  conflated in any progress display, because only one is final.
- Lowering `min_image_dim` to 48 means the crawler now accepts smaller images,
  slightly increasing crawl volume and storage. Within the ADR-001 budget, which
  [[research/face-model-bringup]] measured at 43.0 GB against a 49.6 GB target.
- Existing databases need the constraint widened and the partial index rebuilt:

```sql
ALTER TABLE image DROP CONSTRAINT face_count_valid;
ALTER TABLE image ADD CONSTRAINT face_count_valid CHECK (face_count >= -2);
DROP INDEX image_unexamined_idx;
CREATE INDEX image_unexamined_idx  ON image (id) WHERE face_count = -1;
CREATE INDEX image_provisional_idx ON image (id) WHERE face_count = -2;
```

Applied to `arc_search` and `arc_search_test` on 2026-08-24.

## What would reopen this

A calibration run against labeled pairs showing that faces in the 48–64 px band
produce materially more false positives than they do true positives. That is a
measurement, and it is exactly what `eval.calibrate` exists to produce — at which
point `calibrated=True` starts writing real tombstones and the `-2` backlog gets
its final pass.

The **mechanism** should outlive any particular number: no uncalibrated
threshold may write an irreversible verdict.
