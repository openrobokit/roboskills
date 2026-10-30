import json

import pytest

from roboskills.workflows import real2sim
from roboskills.workflows.real2sim.agent import agent, kept


def test_a_later_round_stays_only_with_no_more_noes_and_no_worse_residual():
    assert kept([["size"], []], [0.05, 0.05]) == 2
    assert kept([["size"], []], [0.05, 0.2]) == 1
    assert kept([[], ["size"]], [0.05, 0.05]) == 1


def test_passing_track_spends_no_tokens(tmp_path):
    (tmp_path / "track").mkdir()
    (tmp_path / "track/status.json").write_text(json.dumps(dict(ok=True, failed=[])))
    record = agent(tmp_path, tmp_path / "agent", dict(shape="align", support="pedestal"))
    assert record == dict(failed=[], rounds=0, tokens={})
    assert (tmp_path / "agent/agent.json").exists()


def test_it_edits_align_only(tmp_path):
    with pytest.raises(ValueError, match="fit"):
        agent(tmp_path, tmp_path / "agent", dict(shape="fit", support="pedestal"))


def test_physics_waits_for_it():
    assert "agent" in real2sim.NEEDS["retarget"].split()
    assert list(real2sim.TOOLS).index("mujoco") < list(real2sim.TOOLS).index("agent")
