# Handoff — 2026-08-24

## Completed this session

- Crawl tier finished and proven against the live FOSDEM archive
- Postgres writer, verified against a real database (10 integration tests)
- [[ADR-003-store-image-urls]] — image URLs now stored, inline not interned
- Vault reorganised: `.claude/commands/`, `00_Brain/`, plan files, this handoff

## Exact next step

**[[plan-001-crawl-tier]] Phase 4** — let the archive run finish, then read
`CrawlStats.report()`. Its images/host and extraction-source breakdown decide
whether `conf.researchr.org` gets enabled to reach the 100k exit criterion.

## Long-running work

A FOSDEM crawl may still be running. It is safe to kill — resume is proven.

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -like "*arc_search.crawler.run*" }
```

Restart (needs `PGPASSWORD` exported from `ARC_PG_PASSWORD` in `.env`):

```
python -m arc_search.crawler.run --only fosdem --frontier data/archive.sqlite
```

Logs are `data/archive-run*.log`. A healthy line looks like:
`crawl.progress ... req_per_s=1.0`

## Blockers / open questions

- None blocking. Cross-cutting questions are listed in [[plans/INDEX]].
- ⚠️ `PGPASSWORD` must be exported for any unattended run — the DSN deliberately
  carries no password. A `.pgpass` would be cleaner than the current wrapper.
- Repo description on GitHub still needs setting by hand (no token available).
- `git config --global user.email` still carries the personal gmail; only this
  repo is set to the noreply alias.
