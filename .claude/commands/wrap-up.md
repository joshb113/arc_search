# /wrap-up
End-of-session wrap-up for arc_search.

1. Mark completed plan steps as ✅ in the active plan file
2. Move any completed task files from vault/tasks/active/ to vault/tasks/completed/
3. Update vault/plans/INDEX.md status
4. Write vault/00_Brain/handoffs/handoff-latest.md summarising: what was done, what's next, any blockers
5. Update vault/00_Brain/CLAUDE.md if any key parameters or decisions changed
6. Verify green before committing: bare `pytest` (not `python -m pytest`),
   `ruff check src tests tools`, `ruff format --check src tests tools`,
   `uv lock --check`
7. Git commit with message: `vault: session wrap-up — [brief summary]`
