# First full index — and a calibration preview

**Date:** 2026-08-24
**Run:** `python -m arc_search.index.backfill` over the whole FOSDEM corpus.

The engine answers a query. This is the first time that has been true.

---

## The drain

```
images processed                    2219
  with faces                        1283
  empty, re-examinable (-2)          936
  barren, tombstoned (0)               0     <- ADR-004 holding
  fetch failed / skipped / undecodable 0
faces indexed                       1285
```

**Zero fetch failures across 2,219 re-fetches.** Two `LocalProtocolError`s
(HTTP/2 connection reuse) appeared mid-run and the existing retry absorbed both —
the crawl tier's resilience work paying off in a tier it was not written for.

Final corpus state, all three stores agreeing exactly:

| | |
|---|---|
| `face` rows in Postgres | **1300** |
| points in Qdrant | **1300** |
| crop files on disk | **1300** |
| images unexamined | 0 |
| images tombstoned | **0** |

## Crop size, measured on real data

Previously estimated at 1,738 B from synthetic noise. Real crops:

```
mean 1921 B   median 1884   p90 2460   max 4524     (ADR-001 budget: 4096)
```

Re-projected to the 10M scale target: 19.2 GB crops + 5.1 GB int8 + 20.5 GB
originals = **44.8 GB** against the 49.6 GB target. Still fits, with ~4.8 GB for
Postgres metadata.

## The quality gate, in aggregate

```
too_small     290
low_score     284
bad_pose       14
too_blurry      7
```

🔴 **`min_det_score = 0.72` is now the co-dominant gate.** With `min_face_px`
fixed at 48 (ADR-004), `low_score` rejects almost as many detections as
`too_small`, and together they are 574 of 595 rejections. It is the next
uncalibrated number to measure, and it has had no scrutiny at all.

`bad_pose` and `too_blurry` being near-zero says the corpus is clean, posed
headshots — which is what a conference speaker archive should be. If either ever
dominates on a new vertical, that vertical cannot support face search.

---

## Identity matching works across years

143 speakers carry a `Photo of NAME` label in more than one FOSDEM year. The
engine links them:

```
QUERY: Adam Samalik (2019)
  1.0000  2019  Photo of Adam Samalik
  0.6671  2017  Photo of Adam Samalik      <-- different photo, correctly matched
  0.3010  2016  Photo of Maciej Borzecki   <-- best impostor
```

A 0.37 margin between the true 2017 match and the nearest wrong answer.

## ⚠️ A calibration preview, and why it is NOT a calibration

Over 1,156 labeled faces / 975 identities: 225 genuine pairs, 20,000 sampled
impostor pairs.

```
genuine : p01=0.317  p05=0.581  median=0.997
impostor: median=0.011  p95=0.124  p99=0.176  p99.9=0.244  MAX=0.651
```

| threshold | recall | false-match |
|---|---|---|
| `t_plausible` 0.28 | 99.1% | 0.030% (1 in 3,333) |
| `t_strong` 0.40 | 98.7% | 0.005% |
| `t_near_certain` 0.55 | 97.3% | 0.005% |

**Do not act on this table.** The genuine median of **0.997** gives it away:

```
genuine pairs total       225
  near-identical (>=0.95) 149  (66%)   <- the SAME PHOTO reused, not evidence
  genuinely different      76  (34%)   <- the only pairs that test anything
```

Two thirds of the "genuine pairs" are the same headshot re-published in a later
year. They measure JPEG re-encoding, not face recognition. On the honest subset:

```
HARD genuine pairs: min=-0.060  p10=0.578  median=0.738  p90=0.836
  recall at 0.28: 97.4%  (inflated figure said 99.1%)
  recall at 0.40: 96.1%  (98.7%)
  recall at 0.55: 92.1%  (97.3%)
```

The placeholders are not catastrophic, but they are flattered by 2–5 points, and
`t_near_certain` loses ~8 points of recall once the duplicates are removed. The
impostor **max of 0.651** also sits above `t_near_certain` — at least one wrong
pair would be reported as near-certain today.

### 🔴 This is the argument for PDQ

**223 of those 225 pairs have DIFFERENT sha1.** They are re-encoded copies, so
exact-hash dedup cannot see them — which is precisely what the PDQ perceptual
hash exists for. The column, the BK-tree and the tests all exist; nothing writes
it (`dedup.loaded pdq=0`).

Until PDQ runs, near-duplicates inflate every recall number the project produces,
and `eval.calibrate` would inherit exactly this bias. **PDQ is a prerequisite for
calibration, not a parallel nice-to-have.**

### ⚠️ Generic labels are not identities

The hard subset's `min = -0.060` — a "genuine" pair with negative similarity —
is not a model failure. `Photo of FOSDEM Staff` and `Photo of BSDCG Team` are
shared by different people across years. The weak label is an identity only when
it names an individual.

`eval.calibrate` must exclude collective labels, or it will train against pairs
that are correctly non-matching.

---

## What this changes for plan-003

The labeled set is real and larger than assumed — 975 identities from the
crawler's alt text, for free. But three things must happen before a number
derived from it means anything:

1. **Compute PDQ** and collapse near-duplicates, or recall is inflated ~2–5 pts.
2. **Exclude collective labels** (`FOSDEM Staff`, `BSDCG Team`, …).
3. **Measure `min_det_score`**, now the co-dominant gate at 284 rejections.
