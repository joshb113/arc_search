# /adr [decision]
Log an architecture decision record mid-session.

1. Create vault/decisions/ADR-NNN-[decision-slug].md with:
   - Title
   - Date
   - Status: Accepted
   - Context (why this decision was needed)
   - Decision (what was chosen)
   - Consequences (tradeoffs)
2. Link ADR from the relevant plan step
3. Add to the Decisions list in vault/plans/INDEX.md

Numbers continue from the highest existing ADR in vault/decisions/.

Two arc_search rules the ADR must respect:
- Any storage claim is MEASURED, not estimated. ADR-003 exists because a
  hand-waved "~1 GB" was wrong in both magnitude and design.
- If it changes the ~49.6 GB budget, amend the table in README.md and the
  scale target in CLAUDE.md in the same commit.
