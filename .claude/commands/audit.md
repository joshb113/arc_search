# /audit
Vault integrity check — run before starting new plans.

1. Check all active plans for steps referencing outputs that don't exist yet
2. Check vault/plans/INDEX.md is in sync with actual plan files
3. Check vault/00_Brain/handoffs/handoff-latest.md exists and is not stale
4. Report any tasks in vault/tasks/active/ with no corresponding plan step
5. Report any src/ files not referenced in a plan
6. Report any threshold in src/arc_search/config.py still marked UNCALIBRATED
   that a plan claims is derived
7. Output a pass/fail summary with actionable fixes
