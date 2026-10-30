# Auto Research

agent_as_policy's turn loop, `bot` and `SKILL.md`, plus this folder's `SKILL.md` for research: the agent plays live development episodes, resets them
to rerun a case or switch to another, and writes the result down as a program. In its namespace, besides `bot`:

- `reset(case=None)`: end this episode and start a fresh one of development case `case` (default: this one). Its
  variables stay. Returns the ended episode's native outcome (success, score, reason, steps) and the budgets left.
- `submit(source)`: `source` defines `solve(bot)`; it runs on a fresh episode of every development case and returns
  each one's outcome and printed output. [`policy.validate`](policy.py) checks it first (public verbs and
  `bot.model`, math and numpy; invalid code spends no submission). The live episode ends first; the agent continues in a fresh one.
- `finish()`: end the research. The submission with the most development successes (the latest of equals) is frozen
  as `solution.py` and run on the validation and test cases without a model.

```sh
# The agent in this repo (the auto-research skill, .claude/skills/auto-research): interactive, or one instruction
claude  # then: /auto-research Research stack_blocks in RoboDojo on development layouts 10 and 11, test on layouts 0-4.
claude -p "/auto-research Research stack_blocks in RoboDojo on development layouts 10 and 11, test on layouts 0-4."
codex exec '$auto-research Research stack_blocks in RoboDojo on development layouts 10 and 11, test on layouts 0-4.'
# The agent anywhere
claude -p "Follow the auto-research skill in github.com/openrobokit/roboskills to research stack_blocks in RoboDojo on development layouts 10 and 11, test on layouts 0-4."
```

```bash
# RoboDojo stack_blocks: research on development layouts 10 and 11, the frozen solution on eval layouts 0-4
python -m roboskills.workflows.auto_research.run --adapter roboskills.eval.robodojo.launch --out runs/auto_research/stack_blocks \
  --dev "layouts=10 mode=dev" --dev "layouts=11 mode=dev" --test "layouts=0,1,2,3,4 mode=eval" \
  --max-resets 5 --max-submits 3 -- task=stack_blocks gpu=0 suite=full5
python -m pytest tests/workflows/auto_research
```

A case is the adapter's keywords after "--", updated by the case's "k=v ..." pairs. A development case gives one env;
a validation or test case may give a batch (RoboDojo `make_vector`: `layouts=0,1,2,3,4`), run at once. One simulator
runs at a time; every episode shares one SAM 3. `--max-resets` counts across all cases. The agent never sees
validation or test cases, their outcomes, or native fields beyond the four above (no dev `truth`).

Results: `dev<i>/attempt<k>/env0/` (each development episode: `result.json`), `log.jsonl` (the research's turns),
`submit<k>/` (`solution.py` and each case's run), `solution.py`, `val/`, `test/` and `summary.json` (turns, cost,
resets, attempts per case, each submission's development outcomes, the frozen solution's dev, val and test successes).
An LLM failure marks the research `void`; the best submission so far is still checked.
