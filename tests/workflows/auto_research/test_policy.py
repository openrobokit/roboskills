"""A submitted solve(bot) program may use only the public bot verbs (and bot.model), math and NumPy."""

import pytest

from roboskills.workflows.auto_research.policy import VERBS, validate


PUBLIC = VERBS | {"model"}  # what auto_research allows


@pytest.mark.parametrize(
    "source",
    [
        "def solve(bot):\n return bot.robot.env",
        "import os\ndef solve(bot):\n return os.getcwd()",
        'def solve(bot):\n return getattr(bot,"robot")',
        "def solve(bot):\n bot.finish()",
        'import numpy as n\ndef solve(bot):\n return n.load("hidden.npy")',
        'from numpy import load as read\ndef solve(bot):\n return read("hidden.npy")',
        "def step(bot):\n return bot.status()",
    ],
)
def test_submissions_cannot_reach_native_state_files_or_scoring(source):
    with pytest.raises(ValueError):
        validate(source, PUBLIC)


def test_submissions_may_use_the_bot_verbs_and_local_state():
    validate(
        'def solve(bot):\n    state={}\n    def update():\n        state["x"]=1\n    update()\n    return state\n',
        PUBLIC,
    )
    validate('def solve(bot):\n    return bot.execute(bot.plan(bot.grasp_poses("right", [[0, 0, 1]], 0.75)))\n', PUBLIC)
    with pytest.raises(ValueError, match="grasp"):
        validate('def solve(bot):\n    return bot.grasp("o1")\n', PUBLIC)
