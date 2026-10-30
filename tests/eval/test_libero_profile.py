"""Matched native settings preserve history and constrain reset/control semantics."""

import random
import time
from types import SimpleNamespace

import numpy as np
import pytest

from roboskills.eval.libero.robot import RUNTIME, Libero, profile_settings


def test_profiles_keep_historical_identity_and_distinct_native_horizons():
    old, matched = profile_settings(), profile_settings("matched-v1")
    assert old["runtime"] == RUNTIME == "native-v10"
    assert old["policy_ticks"] == 1000 and old["joint_kp"] == 150 and old["joint_delta"] == 0.2
    assert matched["runtime"] != RUNTIME and matched["policy_ticks"] == 990
    assert matched["native_horizon"] == 1000 and matched["settle_ticks"] == 10
    assert matched["joint_kp"] == 50 and matched["joint_delta"] == 0.05
    assert old["joint_tolerance"] == 0.002 and matched["joint_tolerance"] == 0.01
    old["joint_kp"] = 999
    assert profile_settings()["joint_kp"] == 150
    with pytest.raises(ValueError, match="Unknown native"):
        profile_settings("unknown")


def test_matched_reset_uses_measured_gripper_and_reseeds_before_native_reset():
    robot = object.__new__(Libero)
    robot.profile, robot.n, robot.arms = "matched-v1", 0, "previous robot"
    actions = []

    def step(action):
        actions.append(action.copy())
        return {"robot0_gripper_qpos": [0.02 if len(actions) == 1 else 0.0]}, 0, False, {}

    robot.wrapper = SimpleNamespace(
        seed=lambda seed: None,
        reset=lambda: {"robot0_gripper_qpos": [0.04]},
        step=step,
        check_success=lambda: True,
    )
    robot.env = SimpleNamespace(
        sim=SimpleNamespace(
            get_state=lambda: np.array([1.0]),
            model=SimpleNamespace(nbody=1, ngeom=0),
            data=SimpleNamespace(xpos=np.zeros((1, 3)), xmat=np.eye(3).reshape(1, 9)),
        )
    )
    robot._robot_roots = lambda: []
    robot._robot_facts = lambda: None
    robot._capture = lambda: None
    robot.reset(501)
    np.testing.assert_allclose(np.asarray(actions)[:3, -1], [-1.0, 0.0, 1.0])
    assert len(actions) == 10 and robot.steps == 0 and not robot.actions
    assert robot.initial_success is True and robot.arms is None and robot.command == 0.0
    assert random.random() == random.Random(501).random()
    assert np.random.random() == np.random.RandomState(501).random_sample()


def test_matched_policy_tick_cap_prevents_extra_native_actions():
    robot = object.__new__(Libero)
    robot.deadline, robot.steps, robot.step_limit = time.monotonic() + 30, 990, 990
    robot.wrapper = SimpleNamespace(step=lambda action: pytest.fail("Uncharged native step"))
    robot._tick()
    assert robot.ended


@pytest.mark.parametrize("profile", ["matched-v2", "matched-v3"])
def test_native_done_is_honored_in_new_phase_without_changing_historical_profiles(profile):
    robot = object.__new__(Libero)
    robot.deadline, robot.steps, robot.step_limit = time.monotonic() + 30, 0, 990
    robot.ended, robot.success, robot.command, robot.actions = False, False, 1.0, []
    robot.respect_native_done = profile_settings(profile).get("respect_native_done", False)
    calls = []

    def step(action):
        calls.append(action)
        return {}, 1.0, True, {}

    robot.wrapper = SimpleNamespace(step=step, check_success=lambda: True)
    robot._capture = lambda: None
    robot._tick()
    assert robot.success and robot.ended is (profile == "matched-v3")
    robot._tick()
    assert len(calls) == (1 if profile == "matched-v3" else 2)


def test_terminal_correction_keeps_matched_reset_controller_and_camera_settings():
    old, corrected = profile_settings("matched-v2"), profile_settings("matched-v3")
    assert corrected.pop("respect_native_done") is True
    assert corrected.pop("runtime") == "native-matched-v3"
    old.pop("runtime")
    assert corrected == old


def test_matched_mujoco_mismatch_rejected_before_gpu_or_sdk_access(monkeypatch, tmp_path):
    monkeypatch.setattr("roboskills.eval.libero.robot.importlib.metadata.version", lambda name: "3.3.2")
    with pytest.raises(ValueError, match="requires MuJoCo 3.3.0"):
        Libero("libero_object_task", tmp_path, profile="matched-v1")
