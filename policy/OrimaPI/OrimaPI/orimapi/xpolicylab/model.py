# coding=utf-8
"""Translate the public XPolicyLab callbacks into serial Runtime calls."""

from __future__ import annotations

import io
from copy import deepcopy
from numbers import Integral

from ..state import Observation, RGBImage, json_value
from ..supervisor import SupervisorEpisodeError
from .evaluation import fast_trajectory_chunk_size

try:
    from XPolicyLab.model_template import ModelTemplate
except ModuleNotFoundError as error:
    if error.name not in {"XPolicyLab", "XPolicyLab.model_template"}:
        raise
    # Core/codec tests do not require an installed XPolicyLab checkout.
    ModelTemplate = object


CAMERAS = {
    "cam_high": ("cam_high", "cam_head", "head_camera", "top_camera"),
    "cam_left_wrist": ("cam_left_wrist", "left_camera", "left_wrist", "wrist_left"),
    "cam_right_wrist": (
        "cam_right_wrist",
        "right_camera",
        "right_wrist",
        "wrist_right",
    ),
}
ROBOT_FIELDS = {
    "joint_state",
    "arm_joint_state",
    "ee_joint_state",
    "ee_pose",
    "tcp_pose",
    "delta_ee_pose",
    "left_arm_joint_state",
    "right_arm_joint_state",
    "left_ee_joint_state",
    "right_ee_joint_state",
    "left_ee_pose",
    "right_ee_pose",
    "left_tcp_pose",
    "right_tcp_pose",
    "left_delta_ee_pose",
    "right_delta_ee_pose",
}


def env_index(value) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError("Environment IDs must be nonnegative integers")
    return int(value)


def convert_observation(
    obs: dict, *, env_id: int, step: int, camera_map=None, raw_images=False
) -> Observation:
    """XPolicyLab decodes its own image bitstreams; this boundary accepts arrays."""
    import numpy as np
    from PIL import Image

    if not isinstance(obs, dict):
        raise ValueError("An XPolicyLab observation must be an object")
    vision = obs.get("vision", obs.get("images"))
    if not isinstance(vision, dict):
        raise ValueError("Observation requires public vision/images")
    images = {}
    for target, sources in (camera_map or CAMERAS).items():
        sources = [sources] if isinstance(sources, str) else sources
        source = next((name for name in sources if name in vision), None)
        if source is None:
            raise ValueError(f"Missing observed camera for {target}: {sources}")
        value = vision[source]
        value = value.get("color") if isinstance(value, dict) else value
        pixels = np.asarray(value)
        if (
            pixels.ndim == 3
            and pixels.shape[-1] not in (3, 4)
            and pixels.shape[0] in (3, 4)
        ):
            pixels = pixels.transpose(1, 2, 0)
        if (
            pixels.dtype != np.uint8
            or pixels.ndim != 3
            or pixels.shape[-1] not in (3, 4)
            or min(pixels.shape[:2]) < 1
        ):
            raise ValueError(
                "Camera color must be a decoded uint8 RGB/RGBA array; let XPolicyLab decode image bytes"
            )
        if raw_images:
            images[target] = RGBImage(
                pixels[:, :, :3].tobytes(), pixels.shape[1], pixels.shape[0]
            )
        else:
            stream = io.BytesIO()
            Image.fromarray(pixels).convert("RGB").save(stream, format="PNG")
            images[target] = stream.getvalue()
    state = obs.get("state")
    if isinstance(state, dict):
        state = {key: value for key, value in state.items() if key in ROBOT_FIELDS}
        if not state:
            raise ValueError("Observation requires public robot-state fields")
    elif state is None:
        raise ValueError("Observation requires robot state")
    return Observation(env_id, step, obs.get("instruction"), images, state)


class Model(ModelTemplate):
    """Adapt one active episode to Runtime; count actions through observations."""

    def __init__(self, model_cfg, *, runtime=None):
        super().__init__()
        self.action_type = model_cfg.get("action_type")
        if self.action_type is not None and self.action_type not in {"joint", "ee"}:
            raise ValueError("action_type must be joint or ee")
        self._owned_trace = None
        if runtime is None:
            import os
            from pathlib import Path
            from ..__main__ import build_runtime, load_config
            from ..trace import TraceWriter

            path = model_cfg.get("orimapi_config") or os.environ.get("ORIMAPI_CONFIG")
            output = model_cfg.get("result_dir") or os.environ.get(
                "ORIMAPI_POLICY_OUTPUT"
            )
            if not path or not output:
                raise ValueError(
                    "Provide orimapi_config and a new result_dir, or inject a runtime"
                )
            trace = TraceWriter(Path(output))
            try:
                config = load_config(path)
                trace.write_json("config.json", config.public())
                runtime = build_runtime(config, trace)
            except BaseException:
                trace.close()
                raise
            self._owned_trace = trace
        self.runtime = runtime
        self.raw_images = model_cfg.get("raw_images", False)
        self.camera_map = deepcopy(model_cfg.get("camera_map") or CAMERAS)
        self._env_id = None
        self._step = -1
        self._remaining_actions = 0
        self._fresh = False
        self._case = {}
        self._closed = False
        self._failed = False
        self._supervisor_error = None
        self.job_supervisor_error = None
        self.job_policy_error = None

    def _open(self):
        if self._closed or self._failed:
            raise RuntimeError("XPolicyLab adapter is closed or failed; reset required")
        if self._supervisor_error:
            raise SupervisorEpisodeError(self._supervisor_error)

    def begin_job(self):
        self.job_supervisor_error = None
        self.job_policy_error = None
        self.reset()

    def prepare_job(self, seed):
        try:
            self.begin_job()
            return self.runtime.prepare_job(seed)
        except SupervisorEpisodeError as error:
            self._supervision_failed(error)
            raise
        except Exception as error:
            self._policy_failed(error)
            raise

    def _supervision_failed(self, error):
        self._supervisor_error = str(error)
        self.job_supervisor_error = str(error)

    def _policy_failed(self, error):
        self._failed = True
        self.job_policy_error = str(error)

    def prepare_case(self, case_meta=None):
        self._open()
        self._case = json_value(case_meta or {})
        # Evaluator identities are evidence only, never supervisor/policy inputs.
        self.runtime.trace.write("evaluator_case", metadata=self._case)

    # Generic CALL does not require the optional transport action_case_id.
    record_case = prepare_case

    def update_obs(self, obs):
        self._open()
        if not isinstance(obs, dict):
            raise ValueError("An observation object is required")
        index = env_index(obs.get("env_idx", 0))
        if self._env_id is not None and index != self._env_id:
            raise ValueError(
                "Only one environment is supported; reset before switching episodes"
            )
        try:
            if self._step >= 0 and not self._remaining_actions:
                raise RuntimeError(
                    "Received another observation without an executed action"
                )
            step_delta = 1
            if self._step >= 0 and fast_trajectory_chunk_size() > 1:
                chunk_steps = obs.get("executed_chunk_steps")
                if chunk_steps is not None:
                    if (
                        type(chunk_steps) is not int
                        or not 1 <= chunk_steps <= self._remaining_actions
                    ):
                        raise ValueError("Invalid executed_chunk_steps in observation")
                    step_delta = chunk_steps
            observation = convert_observation(
                obs,
                env_id=index,
                step=self._step + step_delta,
                camera_map=self.camera_map,
                raw_images=self.raw_images,
            )
            self.runtime.update_observation(observation)
            self._env_id, self._step = index, observation.step
            self._remaining_actions = max(0, self._remaining_actions - step_delta)
            self._fresh = True
            if self._step == 0:
                self.runtime.trace.write(
                    "evaluator_episode",
                    env_id=index,
                    episode_id=self.runtime.state(index).episode_id,
                    case=self._case,
                )
        except SupervisorEpisodeError as error:
            self._supervision_failed(error)
            raise
        except Exception as error:
            self._policy_failed(error)
            raise

    def get_action(self):
        self._open()
        if not self._fresh:
            raise ValueError(
                "Fresh observations are required before requesting actions"
            )
        if self._remaining_actions:
            raise RuntimeError("The previous delivered action chunk has not finished")
        try:
            # A previous episode's GPT request may still be finishing after reset.
            actions = self.runtime.get_actions(self._env_id, wait_for_capacity=True)
            if self.action_type is not None:
                grippers = {
                    "ee_joint_state",
                    "left_ee_joint_state",
                    "right_ee_joint_state",
                }
                arms = (
                    {
                        "joint_state",
                        "arm_joint_state",
                        "left_arm_joint_state",
                        "right_arm_joint_state",
                    }
                    if self.action_type == "joint"
                    else {"ee_pose", "left_ee_pose", "right_ee_pose"}
                )
                if any(set(action) - grippers - arms for action in actions):
                    raise ValueError(
                        f"Policy actions do not match the declared {self.action_type} action contract"
                    )
            self._remaining_actions = len(actions)
            self._fresh = False
            self.runtime.trace.write(
                "adapter_actions_delivered",
                env_id=self._env_id,
                steps=len(actions),
                observation_step=self._step,
            )
            return actions
        except SupervisorEpisodeError as error:
            self._supervision_failed(error)
            raise
        except Exception as error:
            self._policy_failed(error)
            raise

    def on_trial_end(self, result=None):
        if self._closed:
            return
        if result is None:
            return self.reset()
        result = json_value(result)
        if not isinstance(result, dict):
            raise ValueError("An evaluator result object is required")
        if self._env_id is None or (
            "env_idx" in result and env_index(result["env_idx"]) != self._env_id
        ):
            raise ValueError("Terminal result references an inactive environment")
        self.runtime.end_episode(self._env_id, result)
        self._env_id, self._step = None, -1
        self._remaining_actions = 0
        self._fresh = False

    def reset(self):
        if self._closed:
            raise RuntimeError("XPolicyLab adapter is closed")
        try:
            self.runtime.reset()
        except Exception as error:
            self._policy_failed(error)
            raise
        self._env_id, self._step = None, -1
        self._remaining_actions = 0
        self._fresh = False
        self._case.clear()
        self._failed = False
        self._supervisor_error = None

    def close(self):
        if self._closed:
            return
        try:
            self.runtime.close()
        finally:
            self._closed = True
            if self._owned_trace is not None:
                self._owned_trace.close()
