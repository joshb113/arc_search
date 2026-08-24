# Handoff — 2026-08-24

## Where the project is

Week 1. The crawl tier is complete, tested (**233 tests**, CI green on `main`), and
running against the FOSDEM archive at a measured `req_per_s=1.0`. No model in the
loop yet, by design.

Corpus at time of writing: ~4,600 pages, ~540 images, provenance rebuilding after
the fix below. Frontier holds ~16,400 pages pending.

## Completed this session

- Crawl loop finished: two frontiers, scope enforced at enqueue **and** dequeue,
  in-flight recovery, quiescence detection, request counting at the rate limiter
- Postgres writer verified against a live database; connection resilience added
- [[ADR-003-store-image-urls]] — image URLs stored, **inline not interned**,
  measured (157 B/image vs 295 B; interning costs 88% more for zero dedup)
- `ccc-media` validated — kept, with logo denies taking signal 30% → 82%
- Vault reorganised: `.claude/commands/` (9), `00_Brain/`, plan files, indexes

## Exact next step

**[[plan-001-crawl-tier]] Phase 4** — let the archive run finish, then read
`CrawlStats.report()`. Its images/host and extraction-source breakdown decides
whether `conf.researchr.org` gets enabled to reach the 100k exit criterion.

**Before trusting the run, check provenance is landing:**

```sql
SELECT count(*) FROM image_source;
SELECT count(DISTINCT s.image_id) FROM image_source s
  JOIN text_blob t ON t.id = s.alt_text_id WHERE t.body LIKE 'Photo of %';
```

Both were zero for 511 images this session and nothing complained. If labels read
zero, check **which years** the crawl has reached first — 2013–2014 legitimately
carry no alt text.

## The bug worth knowing about

Image provenance rode in an in-memory dict beside the durable frontier. A restart
kept the queue and lost the context, so every already-queued image was recorded
with no page link and no alt text — silently discarding the weak labels the whole
calibration plan depends on, while the crawl reported perfect health.

Fixed: `Frontier.meta` carries the payload with the URL, and `_ctx` is gone.
`test_image_provenance_survives_a_restart` goes red if the old design returns
(verified by re-introducing the bug).

The general lesson is in [[00_Brain/CLAUDE]]: **anything needed on dequeue must be
in the queue.**

## Long-running work

A FOSDEM crawl is probably still running. Safe to kill — resume is proven.

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -like "*arc_search.crawler.run*" }
```

⚠️ Do **not** track it by the PID file — PowerShell 5.1 writes a BOM and the PID
never matches (`[[00_Brain/ISSUES_INDEX#ISS-006]]`). Match on the command line, or
check whether `data/archive-run*.log` is still growing.

Restart (needs `PGPASSWORD` exported from `ARC_PG_PASSWORD` in `.env`):

```
python -m arc_search.crawler.run --only fosdem --frontier data/archive.sqlite
```

Healthy heartbeat: `crawl.progress ... req_per_s=1.0` and an empty `.err` file.

## Blockers / open questions

- None blocking. Cross-cutting questions are in [[plans/INDEX]].
- ⚠️ `PGPASSWORD` must be exported for any unattended run — the DSN deliberately
  carries no password. A `.pgpass` would beat the current wrapper.
- GitHub repo description still needs setting by hand (no token available here).
  Suggested text is in the session history.
- `git config --global user.email` still carries the personal gmail; only this
  repo is set to the noreply alias.
- `.claude/commands/` was created this session, so those commands load from the
  **next** session onward.
