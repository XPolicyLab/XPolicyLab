"""EchoPolicy: Pi05 with optional VLM subgoal orchestration."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.policy.Pi_05.model import Model as Pi05Model
from XPolicyLab.policy.EchoPolicy.vlm_orchestrator.strategies.base import (
    SessionState,
    StrategyContext,
)
from XPolicyLab.policy.EchoPolicy.vlm_orchestrator.strategies.subgoal import (
    SubgoalConfig,
    SubgoalStrategy,
)
from XPolicyLab.policy.EchoPolicy.vlm_orchestrator.vlm import GoogleVLM, PassthroughVLM

logger = logging.getLogger(__name__)


class Model(ModelTemplate):
    """Wrap the official Pi05 adapter with EchoPolicy's subgoal planner.

    The XPolicyLab server owns the WebSocket transport. This adapter only
    translates standard XPolicyLab observations into the planner's canonical
    view, rewrites the current instruction, and delegates action inference to
    the upstream Pi05 model adapter.
    """

    def __init__(self, model_cfg: dict[str, Any]):
        self._model_cfg = dict(model_cfg)
        self._states: dict[int, SessionState] = {}
        self._latest_env_ids: list[int] = [0]
        self._latest_observations: list[dict[str, Any]] = []
        self._build_planner(model_cfg)

        pi_cfg = dict(model_cfg)
        # Keep downloaded weights under this policy directory. The download
        # script described in README.md populates this path.
        pi_cfg["model_path"] = str(
            Path(__file__).resolve().parent / "checkpoints" / "pi05_robodojo_59999"
        )
        self._pi05 = Pi05Model(pi_cfg)
        self.model = self._pi05.model

    def _build_planner(self, cfg: dict[str, Any]) -> None:
        api_key = os.environ.get("VLM_API_KEY")
        if api_key:
            vlm = GoogleVLM(
                model=os.environ.get("VLM_MODEL", "gemini-3.8-flash"),
                base_url=os.environ.get(
                    "VLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"
                ),
                api_key=api_key,
                thinking_level=os.environ.get("VLM_THINKING_LEVEL", "low"),
                proxy_url=os.environ.get("VLM_PROXY_URL"),
            )
            logger.info("EchoPolicy VLM enabled: model=%s", vlm.model)
        else:
            vlm = PassthroughVLM()
            logger.warning("VLM_API_KEY is unset; using passthrough planning")
        self._strategy = SubgoalStrategy(
            StrategyContext(
                vlm=vlm,
                image_key="observation/exterior_image_1_left",
                prompt_key="prompt",
                extra_image_keys=[
                    "observation/wrist_image_left",
                    "observation/wrist_image_right",
                ],
            ),
            SubgoalConfig(
                progress_interval=int(os.environ.get("ECHO_PROGRESS_INTERVAL", "20"))
            ),
        )

    @staticmethod
    def _state_array(state: dict[str, Any], *names: str) -> np.ndarray | None:
        values = []
        for name in names:
            if name in state:
                values.append(np.asarray(state[name]).reshape(-1))
        if not values:
            return None
        return np.concatenate(values)

    @classmethod
    def _canonical_observation(cls, obs: dict[str, Any]) -> dict[str, Any]:
        """Expose standard XPolicyLab observations to the planner."""
        result = dict(obs)
        vision = obs.get("vision") or {}
        cameras = {
            "cam_head": "observation/exterior_image_1_left",
            "cam_left_wrist": "observation/wrist_image_left",
            "cam_right_wrist": "observation/wrist_image_right",
        }
        for camera, key in cameras.items():
            payload = vision.get(camera)
            if isinstance(payload, dict) and payload.get("color") is not None:
                result[key] = payload["color"]
        result["prompt"] = str(
            obs.get("instruction", obs.get("prompt", obs.get("instructions", "")))
        )
        state = obs.get("state")
        if isinstance(state, dict):
            joints = cls._state_array(
                state, "left_arm_joint_state", "left_ee_joint_state",
                "right_arm_joint_state", "right_ee_joint_state",
                "arm_joint_state", "ee_joint_state",
            )
            if joints is not None:
                result["observation/joint_position"] = joints
            gripper = cls._state_array(
                state, "left_ee_joint_state", "right_ee_joint_state", "ee_joint_state"
            )
            if gripper is not None:
                result["observation/gripper_position"] = gripper
        return result

    @staticmethod
    def _planner_output(original: dict[str, Any], planned: dict[str, Any]) -> dict[str, Any]:
        result = dict(original)
        instruction = planned.get("prompt", planned.get("instruction", ""))
        result["instruction"] = str(instruction)
        return result

    def update_obs(self, obs: dict[str, Any]) -> None:
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list: list[dict[str, Any]]) -> None:
        planned: list[dict[str, Any]] = []
        self._latest_env_ids = []
        for index, obs in enumerate(obs_list):
            env_id = int(obs.get("env_idx", index))
            self._latest_env_ids.append(env_id)
            state = self._states.setdefault(env_id, SessionState())
            canonical = self._canonical_observation(obs)
            transformed, _ = self._strategy.process(canonical, state)
            planned.append(self._planner_output(obs, transformed))
        self._latest_observations = planned
        self._pi05.update_obs_batch(planned)

    def get_action(self, **kwargs):
        actions = self.get_action_batch(env_idx_list=[self._latest_env_ids[0]], **kwargs)
        return actions[0]

    def get_action_batch(self, env_idx_list=None, **kwargs):
        ids = list(env_idx_list or self._latest_env_ids)
        actions = self._pi05.get_action_batch(ids, **kwargs)
        for env_id, action in zip(ids, actions):
            state = self._states.setdefault(int(env_id), SessionState())
            steps = len(action) if hasattr(action, "__len__") else 1
            state.action_step_count += int(steps)
            state.episode_step += int(steps)
        return actions

    def reset(self):
        self._states.clear()
        self._latest_env_ids = [0]
        self._latest_observations = []
        self._pi05.reset()
