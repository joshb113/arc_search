# /handoff [agent]
Write a targeted briefing for the next agent or human picking up the session.

1. Read current active plan and completed steps this session
2. Read vault/tasks/active/ for in-progress work
3. Write vault/00_Brain/handoffs/handoff-latest.md containing:
   - What was completed this session
   - Exact next step (plan step ID + description)
   - Any blockers or open questions
   - Files modified this session
   - Any long-running crawl: its PID, log path, and how to resume it
   - If [agent] is specified, tailor the briefing tone and focus to that agent's role
4. Confirm file written
