# plan-004 – Scale and hygiene (months 2–6)

**Status:** 📋 Planned
**Goal:** Keep a 10M-face index correct and current on one machine.

---

## Phase 1 – Freshness ⬜

- [ ] TTL reaper for dead source links — `page.last_checked` and its
      `NULLS FIRST` index exist for this
- [ ] Incremental recrawl

## Phase 2 – Throughput ⬜

- [ ] Frontier → Redis if one process stops being enough. The `Frontier` interface
      is the contract; `frontier_backend` already switches on it.
- [ ] Dedup as a query rather than a preload. `PostgresWriter._load_dedup` reads the
      whole `image` table at startup — fine to low millions, roughly a gigabyte of
      sha1 dict at 10M, with a slower BK-tree rebuild behind it.

## Phase 3 – Budget ⬜

- [ ] Verify the ~49.6 GB projection against real data once the corpus is large
      enough for the extrapolation to mean anything
