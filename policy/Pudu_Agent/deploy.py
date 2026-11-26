"""Run the RGB-only Inspect EEF policy inside RoboDojo."""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any

from XPolicyLab.policy.Pudu_Agent.inspect import deploy as joint_deploy
from XPolicyLab.policy.Pudu_Agent.inspect.policy import (
    InfrastructureFailure,
)

from .docs import eef_docs_from_joint_docs
from .policy import EefAgentPolicy


def _planner(task_env: Any):
    manager = getattr(task_env, "robot_manager", None)
    if manager is None:
        raise InfrastructureFailure(
            "Inspect EEF requires RoboDojo robot_manager and its Cartesian planner"
        )

    def plan(*, arm: str, target_pose):
        robot = manager.get_robot_by_arm_name(f"{arm}_arm")
        planner = manager.planner.get(robot.robot_name)
        if planner is None:
            return {"status": "Unavailable"}
        env_idx = int(getattr(task_env, "env_idx", 0))
        current = manager.get_joint(robot, env_idx_list=[env_idx])[env_idx]
        return planner.plan_path(
            current,
            target_pose,
            real_robot_pose=copy.deepcopy(robot.entity_origin_pose),
        )

    return plan


def _policy_factory(*, action_spec, env, task_env):
    return EefAgentPolicy(
        action_spec=replace(
            action_spec, docs=eef_docs_from_joint_docs(action_spec.docs)
        ),
        env=env,
        planner=_planner(task_env),
    )


def eval_one_episode(TASK_ENV: Any, model_client: Any) -> None:
    joint_deploy.eval_one_episode(
        TASK_ENV,
        model_client,
        policy_factory=_policy_factory,
    )


def eval_one_episode_batch(TASK_ENV: Any, model_client: Any) -> None:
    num_envs = int(getattr(TASK_ENV, "num_envs", 1) or 1)
    if num_envs > 1:
        raise InfrastructureFailure(
            f"RoboDojo started {num_envs} environments; Inspect EEF supports one "
            "conversation per environment. Set eval_batch=false."
        )
    eval_one_episode(TASK_ENV, model_client)
