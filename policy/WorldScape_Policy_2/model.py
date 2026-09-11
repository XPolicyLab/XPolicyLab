from __future__ import annotations

import json
import os
import sys
import time
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import candidate_checkpoint_roots
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)


POLICY_DIR = Path(__file__).resolve().parent
# WorldScape Policy source checkout in worldscape-policy/ (src/worldscape_policy, evals/,
# configs/, recipes/). WORLDSCAPE_POLICY_ROOT / worldscape_root override it.
DEFAULT_WORLDSCAPE_ROOT = POLICY_DIR / "worldscape-policy"
CHECKPOINTS_DIR = POLICY_DIR / "checkpoints"

DEFAULT_ACTION_HORIZON = 24
OBSERVATION_HISTORY_FRAMES = 9
DEFAULT_VLM_COT_PROMPT = (
    "You are a robot planner. Instructions: {task}. Given the current high-level "
    "task instruction and current head-view observation, predict the next atomic "
    "action subtask for the next second."
)
DEFAULT_T5_PROMPT_TEMPLATE = (
    "A video shows that a robot {instruction} The robot {instruction}"
)
CAMERA_KEYS = (
    ("cam_head", "head_camera"),
    ("cam_left_wrist", "left_camera"),
    ("cam_right_wrist", "right_camera"),
)


def _parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _none_like(value: Any) -> bool:
    return value is None or (
        isinstance(value, str) and value.strip().lower() in {"", "none", "null"}
    )


def _instruction(obs: dict[str, Any], fallback: str) -> str:
    value = obs.get("instruction", obs.get("instructions"))
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if value is None:
        return fallback
    text = str(value).strip()
    return text or fallback


def _rgb_image(value: Any, *, camera_name: str) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(
            f"{camera_name} must be an HWC RGB image, got shape {image.shape}"
        )
    if image.dtype != np.uint8:
        raise ValueError(
            f"{camera_name} must be uint8 RGB, got dtype {image.dtype}"
        )
    return np.ascontiguousarray(image)


def resolve_worldscape_root(model_cfg: dict[str, Any]) -> Path:
    """Return the WorldScape checkout: config/env override, else worldscape-policy/."""
    root_value = model_cfg.get("worldscape_root")
    if _none_like(root_value):
        root_value = os.environ.get("WORLDSCAPE_POLICY_ROOT")
    if _none_like(root_value):
        root = DEFAULT_WORLDSCAPE_ROOT
    else:
        root = Path(str(root_value)).expanduser()
    root = root.resolve()
    if not (root / "src/worldscape_policy").is_dir():
        raise FileNotFoundError(
            f"Invalid WorldScape source checkout: {root} "
            "(expected src/worldscape_policy). Set worldscape_root or "
            "WORLDSCAPE_POLICY_ROOT, or restore the worldscape-policy/ tree."
        )
    return root


def resolve_checkpoint(model_cfg: dict[str, Any], worldscape_root: Path) -> Path:
    """Resolve the checkpoint directory.

    Precedence: explicit ``checkpoint_path`` / ``ckpt_setting``; ``ckpt_name`` as a
    path (relative to the policy directory); the shared XPolicyLab layout
    ``checkpoints/<bench>-<ckpt>-<env_cfg>-<action>-<seed>`` then
    ``checkpoints/<ckpt_name>``; finally ``<worldscape_root>/<ckpt_name>``.
    """
    cfg = dict(model_cfg)
    ckpt_setting = cfg.get("ckpt_setting")
    if _none_like(cfg.get("checkpoint_path")) and not _none_like(ckpt_setting):
        cfg["checkpoint_path"] = ckpt_setting
    if _none_like(cfg.get("checkpoint_path")) and _none_like(cfg.get("ckpt_name")):
        raise ValueError("checkpoint_path or ckpt_name is required")

    candidates = candidate_checkpoint_roots(
        cfg,
        CHECKPOINTS_DIR,
        policy_dir=POLICY_DIR,
        explicit_keys=("checkpoint_path",),
    )
    ckpt_name = cfg.get("ckpt_name")
    if not _none_like(ckpt_name):
        candidates.append((worldscape_root / str(ckpt_name)).resolve())
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    checked = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"WorldScape checkpoint not found. Checked:\n  {checked}"
    )


class Model(ModelTemplate):
    """One-environment XPolicyLab adapter for the WorldScape Policy 2 runtime.

    XPolicyLab owns transport and simulator execution. This adapter owns the
    stateful rollout: raw three-view observations, rolling nine-frame WAM
    history, action prediction, and transactional visual/event memory commits.
    """

    def __init__(self, model_cfg: dict[str, Any]):
        self.model_cfg = dict(model_cfg)
        self.action_type = str(self.model_cfg.get("action_type") or "joint")
        if self.action_type != "joint":
            raise ValueError(
                "WorldScape_Policy_2 supports XPolicyLab action_type='joint' only"
            )

        self.env_cfg_type = str(self.model_cfg.get("env_cfg_type") or "")
        if not self.env_cfg_type:
            raise ValueError("env_cfg_type is required")
        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        arm_dims = list(self.robot_action_dim_info["arm_dim"])
        ee_dims = list(self.robot_action_dim_info["ee_dim"])
        if arm_dims != [6, 6] or ee_dims != [1, 1]:
            raise ValueError(
                "WorldScape_Policy_2 requires a dual-arm 6+1+6+1 action profile, "
                f"got arm_dim={arm_dims}, ee_dim={ee_dims}"
            )
        self.action_dim = sum(arm_dims) + sum(ee_dims)

        self.action_horizon = int(
            self.model_cfg.get("action_horizon") or DEFAULT_ACTION_HORIZON
        )
        if self.action_horizon not in {24, 48}:
            raise ValueError(
                "WorldScape_Policy_2 action_horizon must be 24 or 48"
            )
        self.replan_steps = int(
            self.model_cfg.get("replan_steps") or self.action_horizon
        )
        self.observation_interval = int(
            self.model_cfg.get("observation_interval")
            or self.action_horizon // (OBSERVATION_HISTORY_FRAMES - 1)
        )
        if (
            self.observation_interval <= 0
            or self.action_horizon % self.observation_interval
            or self.action_horizon // self.observation_interval
            != OBSERVATION_HISTORY_FRAMES - 1
        ):
            raise ValueError(
                "observation_interval must produce nine frames over one "
                f"action block; expected {self.action_horizon // 8} for "
                f"action_horizon={self.action_horizon}"
            )
        if not 1 <= self.replan_steps <= self.action_horizon:
            raise ValueError(
                "replan_steps must be between 1 and action_horizon"
            )
        if self.replan_steps % self.observation_interval:
            raise ValueError(
                "replan_steps must be divisible by observation_interval"
            )

        self.frames_per_replan = OBSERVATION_HISTORY_FRAMES
        self.memory_reset_chunks = int(
            self.model_cfg.get("memory_reset_chunks") or 0
        )
        if self.memory_reset_chunks < 0:
            raise ValueError("memory_reset_chunks cannot be negative")
        self.vlm_history_num_frames = int(
            self.model_cfg.get("vlm_history_num_frames") or 4
        )
        if self.vlm_history_num_frames < 1:
            raise ValueError("vlm_history_num_frames must be positive")

        self.default_instruction = str(
            self.model_cfg.get("default_instruction")
            or "follow the instruction"
        )
        self.vlm_cot_prompt = str(
            self.model_cfg.get("vlm_cot_prompt") or DEFAULT_VLM_COT_PROMPT
        )
        self.t5_prompt_template = str(
            self.model_cfg.get("t5_prompt_template")
            or DEFAULT_T5_PROMPT_TEMPLATE
        )
        try:
            self.vlm_cot_prompt.format(task="test")
            self.t5_prompt_template.format(instruction="test")
        except (KeyError, ValueError) as exc:
            raise ValueError(
                "Prompt templates must contain valid {task} and {instruction} fields"
            ) from exc

        self.device = str(self.model_cfg.get("device") or "cuda")
        self.allow_dummy_policy = _parse_bool(
            self.model_cfg.get("allow_dummy_policy"), False
        )
        self.log_inference = _parse_bool(
            self.model_cfg.get("log_inference"), True
        )
        self.runtime = None
        self.adapter = None
        self.generator = None
        self.mode = None
        self.diffusion_view_layout = None
        self.worldscape_root = None
        self.checkpoint_path = None

        if not self.allow_dummy_policy:
            self._load_worldscape_runtime()

        self.reset()
        print(
            "[WorldScape_Policy_2-config] "
            + json.dumps(
                {
                    "action_horizon": self.action_horizon,
                    "action_type": self.action_type,
                    "diffusion_view_layout": self.diffusion_view_layout,
                    "env_cfg_type": self.env_cfg_type,
                    "memory_reset_chunks": self.memory_reset_chunks,
                    "observation_interval": self.observation_interval,
                    "replan_steps": self.replan_steps,
                },
                sort_keys=True,
            )
        )

    def _load_worldscape_runtime(self) -> None:
        worldscape_root = resolve_worldscape_root(self.model_cfg)
        self.worldscape_root = worldscape_root
        for path in (worldscape_root, worldscape_root / "src"):
            text = str(path)
            if text not in sys.path:
                sys.path.insert(0, text)

        checkpoint = resolve_checkpoint(self.model_cfg, worldscape_root)
        self.checkpoint_path = checkpoint
        print(
            f"[WorldScape_Policy_2] worldscape_root={worldscape_root} "
            f"checkpoint={checkpoint}"
        )

        import torch

        from evals.robotwin2.adapter import RoboTwin2Adapter
        from evals.robotwin2.checkpoint import (
            load_robotwin2_checkpoint_transform,
        )
        from worldscape_policy.native_builder import (
            build_wan22_policy_from_checkpoint,
            checkpoint_mode,
            checkpoint_supports_mode,
        )
        from worldscape_policy.rollout.session import PolicyRuntime
        from worldscape_policy.types import InteractionMode

        primary_mode = checkpoint_mode(checkpoint, validate_artifacts=False)
        self.mode = InteractionMode.parse(
            self.model_cfg.get("mode") or primary_mode.value
        )
        if not checkpoint_supports_mode(primary_mode, self.mode):
            raise ValueError(
                f"checkpoint mode is {primary_mode.value!r}, "
                f"not requested {self.mode.value!r}"
            )

        requested_layout = self.model_cfg.get("diffusion_view_layout")
        if _none_like(requested_layout):
            requested_layout = None
        else:
            requested_layout = str(requested_layout)
            if requested_layout not in {"mosaic_2x2", "robotwin_concat"}:
                raise ValueError(
                    "diffusion_view_layout must be mosaic_2x2 or robotwin_concat"
                )

        transform = load_robotwin2_checkpoint_transform(checkpoint)
        policy = build_wan22_policy_from_checkpoint(
            checkpoint,
            visual_input_range="zero_one",
            diffusion_view_layout=requested_layout,
            device=self.device,
            expected_mode=self.mode,
            expected_action_horizon=self.action_horizon,
            vlm_cot_prompt=self.vlm_cot_prompt,
            validate_checkpoint_artifacts=_parse_bool(
                self.model_cfg.get("validate_checkpoint_artifacts"), False
            ),
        )
        self.diffusion_view_layout = str(
            policy.visual_memory.codec.diffusion_view_layout
        )
        if self.mode is InteractionMode.AUTO:
            kernel = getattr(getattr(policy, "wam", None), "_numerical_kernel", None)
            kernel_config = getattr(kernel, "config", None)
            if kernel_config is None:
                raise RuntimeError("Auto mode could not locate WAM kernel config")
            kernel.config = replace(kernel_config, cfg_scale=1.0)

        self.runtime = PolicyRuntime(policy)
        self.adapter = RoboTwin2Adapter(
            checkpoint_transform=transform,
            vlm_history_num_frames=self.vlm_history_num_frames,
            preserve_camera_resolution=(
                self.diffusion_view_layout == "robotwin_concat"
            ),
            action_horizon=self.action_horizon,
        )
        self.generator = torch.Generator(
            device=torch.device(self.device)
        ).manual_seed(int(self.model_cfg.get("seed") or 0))

    def _native_observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        vision = obs.get("vision")
        if not isinstance(vision, dict):
            raise KeyError("obs['vision'] is required")
        cameras: dict[str, dict[str, np.ndarray]] = {}
        for xpl_name, native_name in CAMERA_KEYS:
            camera = vision.get(xpl_name)
            if not isinstance(camera, dict) or "color" not in camera:
                raise KeyError(f"obs['vision']['{xpl_name}']['color'] is required")
            cameras[native_name] = {
                "rgb": _rgb_image(camera["color"], camera_name=xpl_name)
            }
        state = pack_robot_state(
            obs,
            self.action_type,
            self.robot_action_dim_info,
            source_type="obs",
            state_type="state",
        )
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.shape != (self.action_dim,):
            raise ValueError(
                f"Packed joint state must have shape ({self.action_dim},), "
                f"got {state.shape}"
            )
        return {
            "observation": cameras,
            "joint_action": {"vector": state.copy()},
        }

    @staticmethod
    def _camera_frames(
        observation: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        data = observation["observation"]
        frames = tuple(
            np.asarray(data[name]["rgb"], dtype=np.uint8)
            for name in ("head_camera", "left_camera", "right_camera")
        )
        if any(frame.ndim != 3 or frame.shape[-1] != 3 for frame in frames):
            raise ValueError("Camera observations must be individual HWC RGB frames")
        return frames  # type: ignore[return-value]

    def _append_history_observation(
        self, observation: dict[str, Any]
    ) -> None:
        if self.chunk_frames is None:
            raise RuntimeError("Missing rolling observation history")
        for history, frame in zip(
            self.chunk_frames, self._camera_frames(observation)
        ):
            history.append(frame.copy())

    def _policy_observation(
        self,
        current: dict[str, Any],
        *,
        append_current: bool,
    ) -> dict[str, Any]:
        import copy

        if self.chunk_frames is None:
            result = copy.deepcopy(current)
            for name, frame in zip(
                ("head_camera", "left_camera", "right_camera"),
                self._camera_frames(current),
            ):
                result["observation"][name]["rgb"] = np.repeat(
                    frame[None, ...], self.frames_per_replan, axis=0
                )
            return result

        if append_current:
            self._append_history_observation(current)
        counts = tuple(len(history) for history in self.chunk_frames)
        if len(set(counts)) != 1 or counts[0] > self.frames_per_replan:
            raise RuntimeError(f"Invalid rolling history counts: {counts}")

        result = copy.deepcopy(current)
        for name, history in zip(
            ("head_camera", "left_camera", "right_camera"),
            self.chunk_frames,
        ):
            values = list(history)
            padded = [values[0]] * (
                self.frames_per_replan - len(values)
            ) + values
            result["observation"][name]["rgb"] = np.stack(padded, axis=0)
        return result

    def update_obs(self, obs: dict[str, Any]) -> None:
        native = self._native_observation(obs)
        self.latest_observation = native
        self.latest_instruction = _instruction(obs, self.default_instruction)

        if not self.chunk_active:
            return

        self.post_action_updates += 1
        if self.post_action_updates > self.active_chunk_length:
            raise RuntimeError(
                "Received more post-action observations than returned actions"
            )
        sampled = self.post_action_updates % self.observation_interval == 0
        if sampled:
            self._append_history_observation(native)
        self.latest_observation_buffered = sampled

        if self.post_action_updates == self.active_chunk_length:
            if not sampled:
                raise RuntimeError(
                    "Chunk boundary is not aligned to observation_interval"
                )
            self._commit_completed_chunk()

    def update_obs_batch(self, obs_list) -> None:
        del obs_list
        raise NotImplementedError(
            "WorldScape_Policy_2 is stateful; use eval_batch=false and one "
            "policy-server instance per environment"
        )

    def _predict_flat_actions(self) -> np.ndarray:
        if self.latest_observation is None:
            raise ValueError("Call update_obs() before get_action()")
        if self.allow_dummy_policy:
            if self.chunk_frames is None:
                self.chunk_frames = tuple(
                    deque([frame.copy()], maxlen=self.frames_per_replan)
                    for frame in self._camera_frames(self.latest_observation)
                )
            return np.zeros(
                (self.replan_steps, self.action_dim), dtype=np.float32
            )
        if self.runtime is None or self.adapter is None or self.generator is None:
            raise RuntimeError("WorldScape runtime was not initialized")

        native_observation = self._policy_observation(
            self.latest_observation,
            append_current=not self.latest_observation_buffered,
        )
        policy_observation = self.adapter.observation(
            native_observation, device=self.device
        )
        instruction = self.latest_instruction
        if self.mode.value == "interactive":
            instruction = self.t5_prompt_template.format(
                instruction=instruction.lower()
            )
        prompts = self.adapter.prompt(instruction, mode=self.mode)

        started = time.perf_counter()
        output = self.runtime.predict(
            observation=policy_observation,
            prompts=prompts,
            generator=self.generator,
        )
        actions = self.adapter.action(output)
        elapsed = time.perf_counter() - started
        if actions.shape != (self.action_horizon, self.action_dim):
            self.runtime.discard()
            raise ValueError(
                f"Policy produced {actions.shape}, expected "
                f"({self.action_horizon}, {self.action_dim})"
            )
        self.pending_output = output
        self.inference_seconds.append(elapsed)

        if self.chunk_frames is None:
            self.chunk_frames = tuple(
                deque([frame.copy()], maxlen=self.frames_per_replan)
                for frame in self._camera_frames(self.latest_observation)
            )
        if self.log_inference:
            print(
                f"[WorldScape_Policy_2-infer] count={len(self.inference_seconds)} "
                f"infer_s={elapsed:.3f}"
            )
        return np.asarray(actions[: self.replan_steps], dtype=np.float32)

    def get_action(self) -> list[dict[str, np.ndarray]]:
        if self.chunk_active:
            raise RuntimeError(
                "Cannot predict before the previously returned chunk is complete"
            )
        actions = self._predict_flat_actions()
        self.chunk_active = True
        self.active_chunk_length = int(actions.shape[0])
        self.post_action_updates = 0
        self.latest_observation_buffered = False
        return unpack_robot_state(
            actions,
            self.action_type,
            self.robot_action_dim_info,
            source_type="obs",
        )

    def get_action_batch(self, env_idx_list=None):
        del env_idx_list
        raise NotImplementedError(
            "WorldScape_Policy_2 is stateful; batched multi-environment "
            "evaluation is not supported"
        )

    def _commit_completed_chunk(self) -> None:
        if not self.chunk_active:
            raise RuntimeError("No active action chunk to commit")
        if not self.allow_dummy_policy:
            if self.runtime is None or self.pending_output is None:
                raise RuntimeError("Missing pending WorldScape prediction")
            self.runtime.commit(self.pending_output)
        self.pending_output = None
        self.chunk_active = False
        self.active_chunk_length = 0
        self.post_action_updates = 0
        self.memory_chunks_since_reset += 1
        if (
            self.memory_reset_chunks
            and self.memory_chunks_since_reset >= self.memory_reset_chunks
        ):
            if self.runtime is not None:
                self.runtime.reset(self.mode.value)
            if self.adapter is not None:
                self.adapter.reset()
            self.chunk_frames = None
            self.latest_observation_buffered = False
            self.memory_chunks_since_reset = 0

    def _discard_pending_chunk(self) -> None:
        if (
            not self.allow_dummy_policy
            and self.runtime is not None
            and self.runtime.has_pending_prediction
        ):
            self.runtime.discard()
        self.pending_output = None
        self.chunk_active = False
        self.active_chunk_length = 0
        self.post_action_updates = 0

    def on_trial_end(self, result=None) -> dict[str, Any]:
        self._discard_pending_chunk()
        count = len(self.inference_seconds)
        timing = {
            "episode_s": time.perf_counter() - self.episode_started,
            "infer_count": count,
            "infer_s": sum(self.inference_seconds),
            "infer_s_mean": (
                sum(self.inference_seconds) / count if count else 0.0
            ),
        }
        print(
            "[WorldScape_Policy_2-trial] "
            + json.dumps(
                {
                    "result": dict(result or {}),
                    "timing": timing,
                },
                sort_keys=True,
            )
        )
        return timing

    def reset(self) -> None:
        if hasattr(self, "runtime") and self.runtime is not None:
            if self.runtime.has_pending_prediction:
                self.runtime.discard()
            self.runtime.reset(self.mode.value)
        if hasattr(self, "adapter") and self.adapter is not None:
            self.adapter.reset()
        self.latest_observation: dict[str, Any] | None = None
        self.latest_instruction = self.default_instruction
        self.latest_observation_buffered = False
        self.chunk_frames: tuple[deque[np.ndarray], ...] | None = None
        self.pending_output = None
        self.chunk_active = False
        self.active_chunk_length = 0
        self.post_action_updates = 0
        self.memory_chunks_since_reset = 0
        self.inference_seconds: list[float] = []
        self.episode_started = time.perf_counter()
