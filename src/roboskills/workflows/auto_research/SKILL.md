---
name: auto-research
description: Research a task in live development episodes with resets, then submit `solve(bot)` programs run on fresh episodes of every development case; read after agent-as-policy, whose rules it amends.
---

You are researching this task, not playing one episode: the rules below replace "One episode, no reset", "Once status
shows ended ... call finish()" and "When done ... call finish() once" above. Your deliverable is a program.

- There are several development cases (the notes say how many); you start in case 0. `reset(case=None)` ends this
  episode and starts a fresh one of `case` (default: this case again); it returns the ended episode's native outcome
  {{success, score, reason, steps}} and your remaining resets and submissions. Your variables stay; the robot and the
  objects start over, so observe again: earlier objects, holds and plans are stale.
- Use resets for controlled experiments: change one measured cause at a time, keep the stages that worked, and try
  every case before trusting a procedure.
- `submit(source)`: `source` is Python defining `solve(bot)`. It runs on a fresh episode of every development case and
  returns each case's outcome and what it printed; then you continue in a fresh episode of this case. The solution
  gets only `bot` (its tools and `bot.model`), `np` and `math`: no `show`, `finish`, `reset` or variables of yours, and
  no imports but math and numpy. It must observe and find everything itself. Invalid code costs no submission.
- Print what the solution's checks decide, so a failed case shows where it went wrong. Submit once a procedure works
  on the cases you tried, then repair what the outcomes show.
- `finish()` ends the research; the submission that succeeded on the most development cases is your solution. Call it
  when your submissions are spent or a submission succeeded on every case.
