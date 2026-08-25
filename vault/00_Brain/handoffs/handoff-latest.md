# Handoff — 2026-08-24 (week 2)

## Where the project is

**The engine answers a query.** A photo goes in, the right person comes back with
their weak label and source page — including the same person in a *different
FOSDEM year*. That is the product working end to end for the first time.

Week 2, [[plans/plan-002-index-and-query]] **Phases 1 and 2 complete**.
**303 tests**, lint and format clean, CI green.

| | |
|---|---|
| `face` rows / Qdrant points / crop files | **1300 / 1300 / 1300** (exact agreement) |
| images with faces | 1298 |
| images provisional (`-2`) | 946 |
| images tombstoned (`0`) | **0** |
| images unexamined (`-1`) | 69 *(new, from the restarted crawl)* |
| pages | 5381 |

Branch **`week2-index-and-query`**, pushed, 4 commits, tracking origin.
PR link: https://github.com/joshb113/arc_search/pull/new/week2-index-and-query

## Completed this session

- **Phase 1 — storage.** `index/vectors.py` (Qdrant 512-d COSINE, int8, HNSW
  m=16), Postgres `face` writer, crop storage tested for the first time.
- **Model bring-up.** antelopev2 runs on CUDA at 49 img/s.
  → [[research/face-model-bringup]]
- **[[decisions/ADR-004-uncalibrated-gates-are-not-tombstones]]** — `min_face_px`
  64 → 48, derived; `face_count` gains `-2`.
- **Phase 2 — backfill.** `index/backfill.py`, full corpus drained.
  → [[research/first-index-and-calibration-preview]]

## ⚠️ The restart command in the last handoff DOES NOT WORK

Two corrections, both cost time this session:

```
# needs PYTHONPATH -- arc_search is NOT installed into .venv
PGPASSWORD=... PYTHONPATH=src .venv/Scripts/python.exe -u \
  -m arc_search.crawler.run --only fosdem --frontier data/archive.sqlite
```

1. Without `PYTHONPATH=src`: `ModuleNotFoundError: No module named 'arc_search'`.
   There is no editable install. The old command only worked because the
   launching shell already had it exported.
2. **`nohup ... &` does not survive here.** The backgrounded crawler died
   silently — zero bytes on *both* streams — once its parent shell was reaped.
   `nohup` ignores SIGHUP, not process-group teardown. Launch it under whatever
   supervises long jobs in your environment, not from a shell that exits.

Also: **do not check a short-lived process with `| tr | head`.** Those
block-buffer when stdout is not a tty, so a few hundred bytes never flush and it
looks like the process produced nothing. Redirect straight to a file.

## Long-running work

**The FOSDEM crawl is running.** `data/archive-run10.log`, empty `.err`,
`req_per_s=1.0`, ~12,130 pages pending.

⚠️ Matching on the command line reports **TWO** processes. One is the Windows
Python launcher stub — **0 CPU seconds, ~3 MB**. The real one has real CPU time
and ~76 MB. **Distinguish by CPU time, not by count.** (At time of writing: stub
21388, real **19820**.) The `[[00_Brain/ISSUES_INDEX#ISS-006]]` PID-file BOM
problem still applies — do not trust the PID file.

🔴 **Stop the crawler before running the backfill again.** Politeness state is
per-process and in-memory, so a crawl and a backfill against the same host each
spend their own token budget and the host sees the sum. Nothing detects this;
`--rps` and a startup warning are all there is.

## Exact next step

**[[plans/plan-002-index-and-query]] Phase 3 — `src/arc_search/serve/`.** The
query path is already proven; what is missing is the HTTP wrapper. Bind to
`127.0.0.1` (see README "Legal posture").

🔴 **Do not render a verdict.** `search.calibrated` is False and
`t_plausible`/`t_strong`/`t_near_certain` are placeholders. One impostor pair
already scores **0.651**, above `t_near_certain` (0.55). Return raw cosine plus
the `calibrated` flag. Non-negotiable #5.

The result page join is already written and tested — see
`tests/test_index_roundtrip.py::test_a_hit_traces_all_the_way_back_to_the_page_it_came_from`.

## Blockers / open questions

- 🔴 **PDQ is a prerequisite for calibration, not a parallel task.** 66% of
  labeled genuine pairs are the same photo re-published in a later year, and
  **223 of 225 have different sha1** (re-encoded), so exact-hash dedup cannot see
  them. They inflate recall 2–5 points and `eval.calibrate` would inherit the
  bias. Still never computed (`dedup.loaded pdq=0`).
- ⚠️ **`min_det_score = 0.72` is the next uncalibrated gate.** 284 rejections
  against `too_small`'s 290 — together 574 of 595. No scrutiny at all so far.
- ⚠️ **Generic labels are not identities.** `Photo of FOSDEM Staff` is shared by
  different people; one such "genuine" pair scores **-0.060**. Calibration must
  exclude collective labels.
- ⚠️ **Dead links are retried forever.** A 404 stays at `-1` and is re-attempted
  every backfill run. Right for a transient 503, wrong for a permanently gone
  file → TTL reaper, [[plan-004-scale-and-hygiene]].
- **No `.gitattributes`.** Line endings drift on this machine:
  `plan-001` and `plan-003` showed 125 changed lines that were **pure CRLF churn**
  and were restored, not committed. A `text=auto` rule would fix it but triggers
  a repo-wide renormalization — its own commit, its own decision.
- **946 images sit at `-2`** awaiting calibration. After `eval.calibrate` runs,
  drain them with `PostgresWriter.provisional_images()` and `calibrated=True`.
- Unchanged from last session: `git config --global user.email` still carries the
  personal gmail; only this repo uses the noreply alias.

## Running things

```
docker compose up -d postgres qdrant

# tests (bare pytest, never python -m pytest)
ARC_TEST_PG_DSN=postgresql://arc@127.0.0.1:5432/arc_search_test \
PGPASSWORD=... .venv/Scripts/pytest.exe

# backfill -- stop the crawler first
PGPASSWORD=... PYTHONPATH=src .venv/Scripts/python.exe \
  -m arc_search.index.backfill [--limit N] [--rps R] [--dry-run]
```

⚠️ antelopev2 unzips one level too deep
(`~/.insightface/models/antelopev2/antelopev2/*.onnx`) and `FaceAnalysis` dies on
`assert 'detection' in self.models`. Flatten it up one level. `config.py` cites
`docker/flatten_models.sh`; **that file does not exist.**
