"""Batched RoboDojo execution and action conversion."""
from __future__ import annotations

from collections.abc import Mapping
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.process_data import get_robot_action_dim_info

POLICY_DIR = Path(__file__).resolve().parents[1]
SUPPORTED_ENV_CFG = "arx_x5"
SUPPORTED_ACTION_TYPE = "joint"
SUPPORTED_INPUT_COLOR_ORDERS = {"rgb", "bgr"}


def _boolean_config(value: Any, *, name: str, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def _integer_config(value: Any, *, name: str, default: int, minimum: int) -> int:
    resolved = default if value is None else value
    if isinstance(resolved, bool):
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    try:
        integer = int(resolved)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}") from exc
    if integer != resolved or integer < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return integer


def _cat_optional_tensors(values: list[torch.Tensor | None], *, name: str) -> torch.Tensor | None:
    present = [value is not None for value in values]
    if not any(present):
        return None
    if not all(present):
        raise ValueError(f"Cannot batch mixed presence for {name}")
    return torch.cat(values, dim=0)  # type: ignore[arg-type]


def _merge_optional_lists(values: list[list[Any] | None], *, name: str) -> list[Any] | None:
    present = [value is not None for value in values]
    if not any(present):
        return None
    if not all(present):
        raise ValueError(f"Cannot batch mixed presence for {name}")
    return [item for value in values for item in value]  # type: ignore[union-attr]


def _stack_observation_batches(observations: list[Any]) -> Any:
    if not observations:
        raise ValueError("Cannot stack an empty observation batch")
    cls = type(observations[0])
    if any(type(observation) is not cls for observation in observations):
        raise TypeError("All policy observations must have the same type")
    history_options = {}
    histories = [getattr(value, "visual_history", None) for value in observations]
    if any(history is not None for history in histories):
        from worldscape_policy.memory.visual.history import collate_histories
        # CPU BF16 autocast does not support packing FP16 archive tensors.
        # Preserve their training storage dtype; VAE encoding owns its autocast.
        with torch.autocast(device_type="cpu", enabled=False):
            history_options["visual_history"] = collate_histories(histories)
    return cls(
        images=torch.cat([value.images for value in observations], dim=0),
        head_view=torch.cat([value.head_view for value in observations], dim=0),
        proprioception=torch.cat([value.proprioception for value in observations], dim=0),
        embodiment_id=torch.cat([value.embodiment_id for value in observations], dim=0),
        vlm_history_images=_cat_optional_tensors(
            [value.vlm_history_images for value in observations],
            name="vlm_history_images",
        ),
        vlm_history_mask=_cat_optional_tensors(
            [value.vlm_history_mask for value in observations],
            name="vlm_history_mask",
        ),
        **history_options,
    )


def _stack_prompt_batches(prompts: list[Any]) -> Any:
    if not prompts:
        raise ValueError("Cannot stack an empty prompt batch")
    cls = type(prompts[0])
    if any(type(prompt) is not cls for prompt in prompts):
        raise TypeError("All policy prompts must have the same type")
    return cls(
        vlm_planning_text=_merge_optional_lists(
            [value.vlm_planning_text for value in prompts], name="vlm_planning_text"
        ),
        language_instruction=_merge_optional_lists(
            [value.language_instruction for value in prompts], name="language_instruction"
        ),
        negative_vlm_text=_merge_optional_lists(
            [value.negative_vlm_text for value in prompts], name="negative_vlm_text"
        ),
        negative_language_instruction=_merge_optional_lists(
            [value.negative_language_instruction for value in prompts],
            name="negative_language_instruction",
        ),
        planning_labels_text=_merge_optional_lists(
            [value.planning_labels_text for value in prompts], name="planning_labels_text"
        ),
        visual_prompt="none",
    )


def _instruction(obs: Mapping[str, Any], fallback: str) -> str:
    value = obs.get("instruction", obs.get("instructions", fallback))
    if isinstance(value, (list, tuple)):
        value = value[0] if value else fallback
    if hasattr(value, "item"):
        value = value.item()
    text = str(value).strip() if value is not None else ""
    return text or fallback


def _input_color_order_config(model_cfg: Mapping[str, Any]) -> str:
    resolved = str(model_cfg.get("input_color_order") or "rgb").strip().lower()
    if resolved not in SUPPORTED_INPUT_COLOR_ORDERS:
        raise ValueError(
            f"input_color_order must be one of {sorted(SUPPORTED_INPUT_COLOR_ORDERS)}, "
            f"got {resolved!r}"
        )
    return resolved


def _rgb_camera(obs: Mapping[str, Any], camera_name: str) -> np.ndarray:
    vision = obs.get("vision")
    if not isinstance(vision, Mapping) or camera_name not in vision:
        available = sorted(vision) if isinstance(vision, Mapping) else []
        raise KeyError(f"Missing camera {camera_name!r}; available cameras: {available}")
    camera = vision[camera_name]
    if not isinstance(camera, Mapping) or "color" not in camera:
        raise KeyError(f"Camera {camera_name!r} is missing decoded 'color'")
    image = np.asarray(camera["color"])
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Camera {camera_name!r} must be HWC RGB, got {image.shape}")
    if image.dtype != np.uint8:
        raise TypeError(f"Camera {camera_name!r} must be uint8 RGB, got {image.dtype}")
    image = np.ascontiguousarray(image)
    # Transport decoders can return read-only views backed by message bytes.
    # torch.from_numpy requires writable storage even when inference only reads.
    return image if image.flags.writeable else image.copy()


def _state_fields(
    obs: Mapping[str, Any],
    *,
    robot_action_dim_info: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    state = obs.get("state")
    if not isinstance(state, Mapping):
        raise KeyError("Observation is missing state")
    arm_dims = list(robot_action_dim_info["arm_dim"])
    gripper_dims = list(robot_action_dim_info["ee_dim"])
    if len(arm_dims) != 2 or len(gripper_dims) != 2:
        raise ValueError("WorldScape_Policy_2.5-Preview joint mode requires two arms and two grippers")
    result: dict[str, np.ndarray] = {}
    for side, arm_dim, gripper_dim in zip(
        ("left", "right"), arm_dims, gripper_dims, strict=True
    ):
        joint_key = f"{side}_arm_joint_state"
        gripper_key = f"{side}_ee_joint_state"
        if joint_key not in state or gripper_key not in state:
            raise KeyError(
                f"Joint state must contain {joint_key!r} and {gripper_key!r}"
            )
        joint = np.asarray(state[joint_key], dtype=np.float32).reshape(-1)
        gripper = np.asarray(state[gripper_key], dtype=np.float32).reshape(-1)
        if joint.shape != (int(arm_dim),) or not np.isfinite(joint).all():
            raise ValueError(
                f"{joint_key} must contain {arm_dim} finite values, got {joint}"
            )
        if gripper.shape != (int(gripper_dim),) or not np.isfinite(gripper).all():
            raise ValueError(
                f"{gripper_key} must contain {gripper_dim} finite values, got {gripper}"
            )
        result[f"state.{side}_joint"] = joint.reshape(1, int(arm_dim))
        result[f"state.{side}_gripper"] = gripper.reshape(1, int(gripper_dim))
    return result


def _relative_action_keys(transform: Any) -> set[str]:
    keys = {str(key) for key in getattr(transform, "relative_action_keys", ())}
    metadata = getattr(transform, "modality_metadata", None)
    if isinstance(metadata, Mapping):
        keys.update(
            str(key)
            for key, value in metadata.items()
            if str(key).startswith("action.") and getattr(value, "absolute", True) is False
        )
    for child in getattr(transform, "transforms", ()):
        keys.update(_relative_action_keys(child))
    return keys


def _fields_to_robodojo_actions(
    fields: Mapping[str, np.ndarray],
    *,
    action_steps: int,
    robot_action_dim_info: Mapping[str, Any],
) -> list[dict[str, np.ndarray]]:
    arrays = {key: np.asarray(value, dtype=np.float32) for key, value in fields.items()}
    arm_dims = list(robot_action_dim_info["arm_dim"])
    gripper_dims = list(robot_action_dim_info["ee_dim"])
    if len(arm_dims) != 2 or len(gripper_dims) != 2:
        raise ValueError("WorldScape_Policy_2.5-Preview joint mode requires two arms and two grippers")
    keys = (
        "action.left_joint",
        "action.left_gripper",
        "action.right_joint",
        "action.right_gripper",
    )
    missing = sorted(set(keys).difference(arrays))
    if missing:
        raise KeyError(f"Checkpoint action transform is missing fields: {missing}")
    horizon = min(action_steps, *(arrays[key].shape[0] for key in keys))
    if horizon < 1:
        raise ValueError("WorldScape_Policy_2.5-Preview returned an empty action chunk")
    for key, width in zip(
        keys,
        (arm_dims[0], gripper_dims[0], arm_dims[1], gripper_dims[1]),
        strict=True,
    ):
        if arrays[key].ndim != 2 or arrays[key].shape[1] != int(width):
            raise ValueError(
                f"{key} must have shape [H,{width}], got {arrays[key].shape}"
            )
    return [
        {
            "left_arm_joint_state": arrays["action.left_joint"][step].copy(),
            "left_ee_joint_state": np.asarray(
                arrays["action.left_gripper"][step], dtype=np.float32
            ).copy(),
            "right_arm_joint_state": arrays["action.right_joint"][step].copy(),
            "right_ee_joint_state": np.asarray(
                arrays["action.right_gripper"][step], dtype=np.float32
            ).copy(),
        }
        for step in range(horizon)
    ]


@dataclass
class _PendingPrediction:
    env_idx_list: list[int]
    horizons: dict[int, int]
    history_start: dict[int, int]


class RoboDojoModel(ModelTemplate, ABC):
    """Shared batched execution for the Preview LeRobot adapter."""

    supports_batch = True
    requires_explicit_commit = True

    @abstractmethod
    def _validate_checkpoint_contract(self, checkpoint_config):
        raise NotImplementedError

    @abstractmethod
    def _checkpoint_builder_options(self):
        raise NotImplementedError

    @abstractmethod
    def _decode_actions(self, output, current_state):
        raise NotImplementedError

    def __init__(self, model_cfg: Mapping[str, Any]):
        from .runtime_config import validate_config
        self.model_cfg = validate_config({"runtime_profile": "robodojo", **model_cfg})
        self.constrain_action_padding = self.model_cfg["constrain_action_padding"]
        self.wam_history_mode = self.model_cfg["wam_history_mode"]
        self.env_cfg_type = str(self.model_cfg.get("env_cfg_type"))
        self.action_type = str(self.model_cfg.get("action_type") or SUPPORTED_ACTION_TYPE)
        if self.env_cfg_type != SUPPORTED_ENV_CFG:
            raise ValueError(
                f"WorldScape_Policy_2.5-Preview supports env_cfg_type={SUPPORTED_ENV_CFG!r}, got {self.env_cfg_type!r}"
            )
        if self.action_type != SUPPORTED_ACTION_TYPE:
            raise ValueError(
                f"WorldScape_Policy_2.5-Preview supports only action_type={SUPPORTED_ACTION_TYPE!r}, "
                f"got {self.action_type!r}"
            )
        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        if len(self.robot_action_dim_info["arm_dim"]) != 2:
            raise ValueError("WorldScape_Policy_2.5-Preview RoboDojo checkpoints require a dual-arm robot")

        from .runtime_config import configure_policy_inputs, prepare_source
        self.upstream_root = prepare_source(self.model_cfg)

        self.checkpoint = resolve_checkpoint_root(
            self.model_cfg,
            POLICY_DIR / "checkpoints",
            policy_dir=POLICY_DIR,
        )
        for artifact in ("transform_bundle.json", "model_config.yaml"):
            if not (self.checkpoint / artifact).is_file():
                raise FileNotFoundError(f"Checkpoint is missing {artifact}: {self.checkpoint}")

        from evals.agilex.observation_adapter import AgileXObservationAdapter
        from evals.common.checkpoint_runtime import load_checkpoint_transform_bundle
        from worldscape_policy.native_builder import (
            _load_native_model_config,
            build_wan22_policy_from_checkpoint,
        )

        self.device = torch.device(str(self.model_cfg.get("device") or "cuda"))
        self.seed = int(self.model_cfg.get("seed") or 0)
        self.input_color_order = _input_color_order_config(self.model_cfg)
        action_steps = self.model_cfg.get("replan_steps")
        self.observation_history_segments = self.model_cfg["observation_history_segments"]
        self.use_observation_history = self.model_cfg["use_observation_history"]
        self.use_causal_memory = self.model_cfg["use_causal_memory"]
        self.memory_reset_chunks = self.model_cfg["memory_reset_chunks"]
        self.reset_independent_blocks = self.model_cfg["reset_independent_blocks"]
        self.default_instruction = str(self.model_cfg.get("default_instruction") or "follow the instruction")
        from worldscape_policy.types import InteractionMode
        checkpoint_config = _load_native_model_config(self.checkpoint).model
        self.mode = InteractionMode.parse(checkpoint_config.mode)
        requested_mode = self.model_cfg.get("mode")
        if requested_mode is not None:
            from worldscape_policy.native_builder import checkpoint_supports_mode
            requested_mode = InteractionMode.parse(requested_mode)
            if not checkpoint_supports_mode(self.mode, requested_mode):
                raise ValueError("Requested mode is incompatible with the checkpoint")
            self.mode = requested_mode
        from .runtime_config import resolve_rollout_config
        self.model_cfg = resolve_rollout_config(self.model_cfg, checkpoint_config)
        # Validate deployment geometry from metadata before allocating model weights.
        self.action_horizon = self.model_cfg["action_horizon"]
        self.action_steps = self.action_horizon if action_steps is None else int(action_steps)
        if self.action_steps > self.action_horizon:
            raise ValueError("replan_steps exceeds checkpoint action_horizon")
        interval = self.model_cfg.get("observation_interval")
        if interval is None:
            if self.observation_history_segments < 1 or self.action_steps % self.observation_history_segments:
                raise ValueError(
                    "action_steps must be divisible by observation_history_segments "
                    "when observation_interval is unset"
                )
            interval = self.action_steps // self.observation_history_segments
        interval = int(interval)
        if interval < 1 or self.action_steps % interval or self.action_steps // interval != self.observation_history_segments:
            raise ValueError("observation_interval must agree with replan_steps and observation_history_segments")
        self.observation_interval = interval

        self.bundle = load_checkpoint_transform_bundle(self.checkpoint, self._checkpoint_embodiment)
        self._validate_checkpoint_contract(checkpoint_config)
        self.joint_action_representation = (
            "relative" if _relative_action_keys(self.bundle.transform) else "absolute"
        )
        self.visual_input_range = self.bundle.transform.image_input_range
        self.policy = build_wan22_policy_from_checkpoint(
            self.checkpoint,
            visual_input_range=self.visual_input_range,
            device=self.device,
            expected_mode=self.mode,
            expected_action_horizon=self.action_horizon,
            training=False,
            diffusion_view_layout=self.model_cfg.get("diffusion_view_layout"),
            **self._checkpoint_builder_options(),
        )
        configure_policy_inputs(self.policy, self.model_cfg)
        self._observation_adapter_class = AgileXObservationAdapter
        self._generator: torch.Generator | None = None
        self._prediction_index = 0
        self._latest_obs_by_env: dict[int, Mapping[str, Any]] = {}
        self._obs_history_by_env: dict[int, list[Mapping[str, Any]]] = {}
        self._observation_index_by_env: dict[int, int] = {}
        self._latest_env_idx_list: list[int] = []
        self.reset()

    def update_obs(self, obs: Mapping[str, Any]):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list: list[Mapping[str, Any]]):
        if not obs_list:
            raise ValueError("update_obs_batch requires at least one observation")
        env_idx_list: list[int] = []
        for fallback_idx, obs in enumerate(obs_list):
            if not isinstance(obs, Mapping):
                raise TypeError(f"update_obs_batch expects mappings, got {type(obs).__name__} at index {fallback_idx}")
            env_idx_list.append(int(obs.get("env_idx", fallback_idx)))
        if len(set(env_idx_list)) != len(env_idx_list):
            raise ValueError(f"update_obs_batch received duplicate env indices: {env_idx_list}")

        for env_idx, obs in zip(env_idx_list, obs_list):
            self._ensure_env(env_idx)
            self._latest_obs_by_env[env_idx] = obs
            self._obs_history_by_env[env_idx].append(obs)
            self._observation_index_by_env[env_idx] += 1
        self._latest_env_idx_list = env_idx_list

    def _ensure_env(self, env_idx: int) -> None:
        if env_idx in self._observation_index_by_env:
            return
        self._observation_index_by_env[env_idx] = 0
        self._obs_history_by_env[env_idx] = []

    @abstractmethod
    def _prepare_observation_cameras(self, env_idx: int) -> list[np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def _adapt_observation(self, env_idx: int, *, cameras: list[np.ndarray] | None = None) -> Any:
        raise NotImplementedError

    def get_action(self) -> list[dict[str, np.ndarray]]:
        env_idx = self._latest_env_idx_list[0] if self._latest_env_idx_list else 0
        return self.get_action_batch([env_idx])[0]

    def get_action_batch(self, env_idx_list=None):
        if self._pending_prediction is not None:
            raise RuntimeError("Commit or discard the pending prediction before predicting again")
        if env_idx_list is None:
            env_idx_list = self._latest_env_idx_list
        env_idx_list = [int(env_idx) for env_idx in env_idx_list]
        if not env_idx_list:
            raise ValueError("get_action_batch requires at least one environment")
        if len(set(env_idx_list)) != len(env_idx_list):
            raise ValueError(f"get_action_batch received duplicate env indices: {env_idx_list}")

        missing_envs = [env_idx for env_idx in env_idx_list if env_idx not in self._latest_obs_by_env]
        if missing_envs:
            raise KeyError(f"Missing observations for env indices: {missing_envs}")

        if self._generator is None:
            raise RuntimeError("WorldScape batch generator is not initialized")
        generator_state = self._generator.get_state()
        try:
            self.policy.reset_episode()
            adapted_batch = [self._adapt_observation(env_idx) for env_idx in env_idx_list]
            observation = _stack_observation_batches([adapted.observation for adapted in adapted_batch])
            prompts = _stack_prompt_batches([adapted.prompts for adapted in adapted_batch])
            output = self.policy.sample(
                mode=self.mode,
                observation=observation,
                prompts=prompts,
                generator=self._generator,
                event_memory=None,
                visual_memory=None,
            )
            action = output.require_action()
            if action.ndim != 3 or action.shape[0] != len(env_idx_list):
                raise ValueError(
                    "WorldScape_Policy_2.5-Preview must return action [B,H,D] matching the request; "
                    f"got {tuple(action.shape)} for B={len(env_idx_list)}"
                )

            action_batch = []
            for batch_index, (adapted, env_idx) in enumerate(zip(adapted_batch, env_idx_list)):
                env_output = replace(output, action=action[batch_index : batch_index + 1])
                env_actions = self._decode_actions(env_output, adapted.raw_state)
                action_batch.append(env_actions)

        except Exception:
            self._generator.set_state(generator_state)
            raise

        self._pending_prediction = _PendingPrediction(
            env_idx_list=list(env_idx_list),
            horizons={env: len(actions) for env, actions in zip(env_idx_list, action_batch, strict=True)},
            history_start={env: len(self._obs_history_by_env[env]) - 1 for env in env_idx_list},
        )
        return action_batch

    def commit_action(self, execution: Mapping[str, Any] | None = None) -> None:
        """Commit a completed chunk; retire environments whose episodes ended.

        The deployment loop reports executed_steps and finished_env_idx_list.
        Direct Python callers may omit the report to confirm full execution of
        every returned chunk, matching native PolicyRuntime.commit().
        """

        pending = self._pending_prediction
        if pending is None:
            raise RuntimeError("There is no pending prediction to commit")
        if execution is None:
            executed = pending.horizons
            finished = set()
        else:
            if not isinstance(execution, Mapping) or not isinstance(execution.get("executed_steps"), Mapping):
                raise ValueError("commit_action requires an executed_steps mapping")
            executed = {
                int(env): _integer_config(count, name="executed_steps", default=0, minimum=0)
                for env, count in execution["executed_steps"].items()
            }
            if len(executed) != len(execution["executed_steps"]):
                raise ValueError("Duplicate environment indices in executed_steps")
            finished = {int(env) for env in execution.get("finished_env_idx_list", [])}
        expected = set(pending.env_idx_list)
        if set(executed) != expected or not finished.issubset(expected):
            raise ValueError("Execution report must match the pending environment indices")
        for env, horizon in pending.horizons.items():
            count = executed[env]
            if count > horizon or (env not in finished and count != horizon):
                raise ValueError(
                    f"Environment {env} executed {count}/{horizon} actions; "
                    "discard a partial chunk before continuing inference"
                )
        active = [env for env in pending.env_idx_list if env not in finished]
        for env in pending.env_idx_list:
            # The stored list is the stride window. Keep the frame that started
            # this prediction plus frames captured during execution; commit drops
            # anything older than that boundary.
            self._obs_history_by_env[env] = (
                self._obs_history_by_env[env][pending.history_start[env]:] if env in active else []
            )
        self._prediction_index += 1
        self._pending_prediction = None

    def discard_action(self, execution: Mapping[str, Any] | None = None) -> None:
        """Discard an unexecuted candidate, or rebase after partial/uncertain execution."""

        execution_started = False
        if execution is not None:
            if not isinstance(execution, Mapping):
                raise TypeError("discard_action expects an execution mapping")
            execution_started = _boolean_config(
                execution.get("execution_started"), name="execution_started", default=False
            )
        pending = self._pending_prediction
        if pending is None:
            return
        # Like native PolicyRuntime.discard(), a successful but discarded
        # prediction has already consumed its random samples.
        self._pending_prediction = None
        if execution_started:
            # Execution cannot be rolled back; restart context from a live observation.
            self._latest_obs_by_env = {}
            self._latest_env_idx_list = []
            self._obs_history_by_env = {env: [] for env in self._observation_index_by_env}
            self.policy.reset_episode()

    def reset(self):
        self.policy.reset_episode()
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(self.seed)
        self._prediction_index = 0
        self._pending_prediction: _PendingPrediction | None = None
        self._latest_obs_by_env = {}
        self._obs_history_by_env = {}
        self._observation_index_by_env = {}
        self._latest_env_idx_list = []
