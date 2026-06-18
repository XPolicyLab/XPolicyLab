from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

repo_root = Path(__file__).resolve().parents[2]
import sys

if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

from task_env.real_env_client import RealEnv, validate_deploy_cfg


def test_validate_deploy_cfg_normalizes_case_and_whitespace():
    assert validate_deploy_cfg({"action_type": " EE "}) == "ee"
    assert validate_deploy_cfg({"action_type": "Joint"}) == "joint"


@pytest.mark.parametrize(
    "deploy_cfg",
    [
        {},
        {"action_type": None},
        {"action_type": ""},
        {"action_type": "   "},
    ],
)
def test_validate_deploy_cfg_rejects_empty(deploy_cfg: dict[str, Any]):
    with pytest.raises(ValueError, match="empty or missing"):
        validate_deploy_cfg(deploy_cfg)


def test_validate_deploy_cfg_rejects_invalid_value():
    with pytest.raises(ValueError, match="invalid action_type='absolute'"):
        validate_deploy_cfg(
            {
                "action_type": "absolute",
                "trial_id": "case-1-r01",
                "policy_name": "demo_policy",
            }
        )


def _make_env_stub(*, action_type: str = "ee") -> RealEnv:
    env = RealEnv.__new__(RealEnv)
    env.action_type = action_type
    env.deploy_cfg = {"trial_id": "case-1-r01", "policy_name": "demo_policy"}
    env.robot_action_dim_info = {"arm_dim": [7, 7], "ee_dim": [1, 1]}
    env.episode_step = 0
    env._stop_check = None
    env.is_replay_robot = False
    return env


def test_validate_action_payload_rejects_missing_ee_pose():
    env = _make_env_stub(action_type="ee")
    with pytest.raises(ValueError, match="missing required key 'left_ee_pose'"):
        env._validate_action_payload(
            {"left_ee_joint_state": [0.0]},
            "ee",
            "ee_pose",
            "ee_joint_state",
        )


def test_validate_action_payload_rejects_empty_joint_vector():
    env = _make_env_stub(action_type="joint")
    with pytest.raises(ValueError, match="empty array for 'left_arm_joint_state'"):
        env._validate_action_payload(
            {
                "left_arm_joint_state": [],
                "left_ee_joint_state": [0.0],
            },
            "joint",
            "arm_joint_state",
            "ee_joint_state",
        )


def test_validate_action_payload_accepts_valid_ee_pose():
    env = _make_env_stub(action_type="ee")
    env._validate_action_payload(
        {
            "left_ee_pose": np.ones(7),
            "left_ee_joint_state": [0.0],
            "right_ee_pose": np.ones(7),
            "right_ee_joint_state": [0.0],
        },
        "ee",
        "ee_pose",
        "ee_joint_state",
    )


def test_assert_episode_executed_raises_on_zero_steps():
    env = _make_env_stub()
    with pytest.raises(RuntimeError, match="completed with 0 action steps"):
        env._assert_episode_executed()


def test_assert_episode_executed_allows_stop_before_first_action():
    env = _make_env_stub()
    env._stop_check = lambda: True
    env._assert_episode_executed()
