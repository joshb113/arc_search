# /status
Dashboard — current project state.

1. Read vault/plans/INDEX.md
2. Count ✅/🟡/⬜ steps across all active plans — display progress bar
3. List unblocked next steps
4. List any blocked steps and their blockers
5. Report vault health: missing outputs, stale indexes
6. Report crawl health if a crawler is running: last `crawl.progress` line from
   data/*.log (pages, images, req_per_s) and whether stderr is empty
