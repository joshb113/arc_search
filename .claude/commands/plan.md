# /plan [feature]
Create a new execution plan for arc_search.

1. Create vault/plans/plan-NNN-[feature-slug].md with:
   - Goal statement
   - Status: 🟡 Active
   - Steps broken into phases with checkboxes
   - Dependencies and outputs
2. Add entry to vault/plans/INDEX.md
3. Break steps into task files in vault/tasks/backlog/ if needed

If the plan changes anything in CLAUDE.md's non-negotiables or pushes past the
~49.6 GB storage budget, it needs an ADR first — use /adr.
