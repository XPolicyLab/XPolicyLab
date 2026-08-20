"""Inference-only XPolicyLab adapter for the RoboDojo DM0.5-Mem checkpoint."""

from __future__ import annotations

from collections import deque
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import candidate_checkpoint_roots
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)

_POLICY_DIR = Path(__file__).resolve().parent
_UPSTREAM_ROOT = _POLICY_DIR / "opendm"
_CHECKPOINTS_DIR = _POLICY_DIR / "checkpoints"
_MODEL_MARKERS = ("config.json", "adapter_config.json")
_DEFAULT_ROBOT_TYPE = "Dual ARX5"

_CAMERA_CANDIDATES = {
    "images_1": ("cam_head", "cam_high", "cam_third_view"),
    "images_2": ("cam_left_wrist", "left_wrist"),
    "images_3": ("cam_right_wrist", "right_wrist"),
}


def _as_optional_path(value: Any) -> Path | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    path = Path(os.path.expanduser(str(value)))
    if not path.is_absolute():
        path = _POLICY_DIR / path
    return path.resolve()


def _as_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"expected a boolean value, got {value!r}")


def _find_loadable_checkpoint(root: Path) -> Path | None:
    if root.is_dir() and any((root / marker).is_file() for marker in _MODEL_MARKERS):
        return root
    if not root.is_dir():
        return None

    def checkpoint_step(path: Path) -> int:
        try:
            return int(path.name.rsplit("-", 1)[-1])
        except ValueError:
            return -1

    for checkpoint in sorted(root.glob("checkpoint-*"), key=checkpoint_step, reverse=True):
        if any((checkpoint / marker).is_file() for marker in _MODEL_MARKERS):
            return checkpoint.resolve()
    return None


def _resolve_model_assets(model_cfg: dict) -> tuple[Path, Path]:
    checked: list[Path] = []
    model_path: Path | None = None
    for candidate in candidate_checkpoint_roots(
        model_cfg,
        _CHECKPOINTS_DIR,
        policy_dir=_POLICY_DIR,
        explicit_keys=("model_path", "checkpoint_path", "ckpt_path"),
    ):
        candidate = Path(candidate)
        checked.append(candidate)
        model_path = _find_loadable_checkpoint(candidate)
        if model_path is not None:
            break

    if model_path is None:
        rendered = "\n  ".join(str(path) for path in checked) or "<no candidates>"
        raise FileNotFoundError(
            "No loadable OpenDM checkpoint found. Pass ckpt_name as a path or "
            "place it under policy/OpenDM/checkpoints. Checked:\n  " + rendered
        )

    norm_stats_path = _as_optional_path(model_cfg.get("norm_stats_path"))
    if norm_stats_path is None:
        norm_stats_path = model_path / "norm_stats.json"
    if not norm_stats_path.is_file():
        raise FileNotFoundError(f"OpenDM normalization statistics are required: {norm_stats_path}")
    return model_path, norm_stats_path


def _load_experiment(model_cfg: dict):
    experiment_path = _as_optional_path(model_cfg.get("experiment_path") or "scripts/robodojo_dm05_history.py")
    if experiment_path is None or not experiment_path.is_file():
        raise FileNotFoundError(f"OpenDM experiment file not found: {experiment_path}")

    module_name = "_xpolicylab_opendm_exp_" + hashlib.sha256(str(experiment_path).encode()).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(module_name, experiment_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load OpenDM experiment from {experiment_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    experiment_class = getattr(module, "DM05Exp", None)
    if experiment_class is None:
        raise AttributeError(f"{experiment_path} does not define DM05Exp")
    return experiment_class(), experiment_path


def _extract_rgb_image(observation: dict, image_key: str) -> np.ndarray:
    """Normalize an already-decoded XPolicyLab RGB image to HWC uint8."""
    vision = observation.get("vision") or {}
    for camera_name in _CAMERA_CANDIDATES[image_key]:
        if camera_name not in vision:
            continue
        camera_data = vision[camera_name]
        image = camera_data.get("color", camera_data.get("colors")) if isinstance(camera_data, dict) else camera_data
        if image is None:
            continue

        array = np.asarray(image)
        if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
            array = np.transpose(array, (1, 2, 0))
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=-1)
        if array.ndim != 3 or array.shape[-1] not in (3, 4):
            raise ValueError(f"{camera_name} must be an HWC RGB image, got {array.shape}")
        return np.ascontiguousarray(array[..., :3], dtype=np.uint8)

    candidates = ", ".join(_CAMERA_CANDIDATES[image_key])
    raise KeyError(f"Missing OpenDM camera {image_key}; tried observation['vision'][{candidates}]")


def _normalize_prompt(observation: dict, default_prompt: str) -> str:
    instruction = observation.get("instruction", observation.get("instructions"))
    if isinstance(instruction, (list, tuple)):
        instruction = instruction[0] if instruction else None
    return default_prompt if not instruction or not str(instruction).strip() else str(instruction)


class Model(ModelTemplate):
    """Run DM0.5-Mem inference in the XPolicyLab policy-server process."""

    def __init__(self, model_cfg: dict):
        self.model_cfg = model_cfg
        self.action_type = str(model_cfg.get("action_type") or "joint")
        if self.action_type != "joint":
            raise ValueError("OpenDM RoboDojo sim supports action_type=joint only")

        self.env_cfg_type = str(model_cfg.get("env_cfg_type") or "arx_x5")
        if self.env_cfg_type != "arx_x5":
            raise ValueError("This checkpoint supports RoboDojo sim env_cfg_type=arx_x5 only")
        shared_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        self.robot_action_dim_info = {
            "arm_dim": list(shared_dim_info["arm_dim"]),
            "ee_dim": list(shared_dim_info["ee_dim"]),
        }
        if len(self.robot_action_dim_info["arm_dim"]) != len(self.robot_action_dim_info["ee_dim"]):
            raise ValueError("arm_dim and ee_dim must describe the same number of arms")
        self.action_dim = sum(self.robot_action_dim_info["arm_dim"]) + sum(self.robot_action_dim_info["ee_dim"])
        if self.action_dim != 14:
            raise ValueError(
                "DM0.5-Mem expects left 6+1 and right 6+1 joint actions; "
                f"the shared robot configuration reports {self.action_dim} dimensions"
            )
        launcher_action_dim = model_cfg.get("action_dim")
        if launcher_action_dim is not None and int(launcher_action_dim) != self.action_dim:
            raise ValueError(
                f"RoboDojo launcher reports action_dim={launcher_action_dim}, "
                f"but the shared robot configuration requires {self.action_dim}"
            )

        self.action_chunk_size = int(model_cfg.get("action_chunk_size") or 50)
        self.action_steps = int(model_cfg.get("action_steps") or 25)
        if self.action_chunk_size <= 0 or not 0 < self.action_steps <= self.action_chunk_size:
            raise ValueError("action_steps must be in [1, action_chunk_size]")
        self.default_prompt = str(
            model_cfg.get("prompt") or model_cfg.get("task_name") or "Perform the instructed manipulation task."
        )
        self.model_action_mode = str(model_cfg.get("model_action_mode") or "absolute").lower()
        if self.model_action_mode != "absolute":
            raise ValueError("The RoboDojo sim checkpoint requires absolute actions")
        self.add_state = _as_bool(model_cfg.get("add_state"), default=True)
        self._configure_history(model_cfg)

        model_path, norm_stats_path = _resolve_model_assets(model_cfg)
        self._initialize_runtime(model_path, norm_stats_path)
        self._observations: dict[Any, dict] = {}
        self._history_by_env: dict[Any, deque[tuple[int, int, np.ndarray]]] = {}
        self._history_sequence_by_env: dict[Any, int] = {}
        self._latest_env_idx_list: list[Any] = [0]
        self._latest_env_idx_by_evaluation: dict[str, list[Any]] = {}

        print(
            f"[OpenDM] checkpoint={model_path} action_dim={self.action_dim} "
            f"chunk={self.action_chunk_size} execute={self.action_steps} "
            f"history={self.history_slots}@{self.history_fps:g}Hz"
        )

    def _configure_history(self, model_cfg: dict[str, Any]) -> None:
        if not _as_bool(model_cfg.get("history_enabled"), default=True):
            raise ValueError("The DM0.5-Mem checkpoint requires history_enabled=true")
        self.history_enabled = True
        self.history_image_key = str(model_cfg.get("history_image_key") or "images_1")
        if self.history_image_key not in _CAMERA_CANDIDATES:
            raise ValueError(f"unknown history_image_key: {self.history_image_key}")
        self.history_slots = int(model_cfg.get("history_slots") or 20)
        self.runtime_fps = float(model_cfg.get("runtime_fps") or 25.0)
        configured_history_fps = float(model_cfg.get("history_fps") or 1.0)
        self.history_tokens_per_slot = int(model_cfg.get("history_tokens_per_slot") or 16)
        if self.history_slots <= 0:
            raise ValueError("history_slots must be positive")
        if not math.isfinite(self.runtime_fps) or self.runtime_fps <= 0:
            raise ValueError("runtime_fps must be positive")
        if not math.isfinite(configured_history_fps) or configured_history_fps <= 0:
            raise ValueError("history_fps must be positive")

        default_interval = int(round(self.runtime_fps / configured_history_fps))
        self.history_action_interval = int(model_cfg.get("history_action_interval") or default_interval)
        if self.history_action_interval <= 0:
            raise ValueError("history_action_interval must be positive")
        self.history_fps = self.runtime_fps / self.history_action_interval
        self._history_buffer_maxlen = self.history_slots + 1

    @staticmethod
    def _evaluation_id(
        payload: dict[str, Any] | None,
        *,
        required: bool = False,
    ) -> str | None:
        if payload is None:
            if required:
                raise ValueError("evaluation scope is required")
            return None
        if not isinstance(payload, dict):
            raise TypeError("evaluation scope must be a dict")
        evaluation_id = payload.get("evaluation_id")
        if evaluation_id is None:
            if required:
                raise ValueError("evaluation scope is missing evaluation_id")
            return None
        if not isinstance(evaluation_id, str) or not evaluation_id.strip():
            raise ValueError("evaluation_id must be a non-empty string")
        return evaluation_id

    @staticmethod
    def _state_key(env_idx: Any, evaluation_id: str | None = None) -> Any:
        return env_idx if evaluation_id is None else (evaluation_id, env_idx)

    def _set_latest_env_indices(
        self,
        env_idx_list: list[Any],
        evaluation_id: str | None = None,
    ) -> None:
        indices = list(env_idx_list)
        if evaluation_id is None:
            self._latest_env_idx_list = indices
        else:
            self._latest_env_idx_by_evaluation[evaluation_id] = indices

    def _latest_env_indices(
        self,
        evaluation_id: str | None = None,
    ) -> list[Any]:
        if evaluation_id is None:
            return list(self._latest_env_idx_list)
        return list(self._latest_env_idx_by_evaluation.get(evaluation_id, [0]))

    def _initialize_runtime(self, model_path: Path, norm_stats_path: Path) -> None:
        if str(_UPSTREAM_ROOT) not in sys.path:
            sys.path.insert(0, str(_UPSTREAM_ROOT))

        from opendm.constants.robot import (
            HISTORY_TOKENS_PER_IMAGE,
            ActionMode,
            RobotStateDesc,
        )

        experiment, experiment_path = _load_experiment(self.model_cfg)
        if not bool(getattr(experiment.data_config, "is_history", False)):
            raise ValueError(f"history experiment required: {experiment_path}")
        if self.history_tokens_per_slot != HISTORY_TOKENS_PER_IMAGE:
            raise ValueError(
                f"history token mismatch: configured={self.history_tokens_per_slot}, model={HISTORY_TOKENS_PER_IMAGE}"
            )

        runtime_model_cfg = experiment.model_config
        runtime_model_cfg.model_name_or_path = str(model_path)
        runtime_model_cfg.chunk_size = self.action_chunk_size
        runtime_model_cfg.bf16 = _as_bool(self.model_cfg.get("bf16"), default=False)
        runtime_model_cfg.force_fp32_action_path = _as_bool(self.model_cfg.get("force_fp32_action_path"), default=True)
        runtime_model_cfg.llm_attn_implementation = str(self.model_cfg.get("llm_attn_implementation") or "sdpa")
        runtime_model_cfg.vision_attn_implementation = str(self.model_cfg.get("vision_attn_implementation") or "sdpa")
        runtime_model_cfg.action_attn_implementation = str(self.model_cfg.get("action_attn_implementation") or "sdpa")
        runtime_model_cfg.liger_kernel = _as_bool(self.model_cfg.get("liger_kernel"), default=True)

        model = runtime_model_cfg.build_model(use_lora=False)
        inference = experiment.inference_config
        inference.enable_bf16_compute = _as_bool(self.model_cfg.get("enable_bf16_compute"), default=True)
        inference.diffusion_steps = int(self.model_cfg.get("diffusion_steps") or 10)
        inference.diffusion_integration_dtype = str(
            self.model_cfg.get("diffusion_integration_dtype") or "model"
        ).lower()
        if inference.diffusion_integration_dtype not in {"model", "float32"}:
            raise ValueError("diffusion_integration_dtype must be model or float32")
        noise_seed = self.model_cfg.get("diffusion_noise_seed")
        inference.diffusion_noise_seed = (
            None if noise_seed is None or str(noise_seed).strip() == "" else int(noise_seed)
        )
        inference.output_action_dim = self.action_dim
        inference.image_keys = list(_CAMERA_CANDIDATES)
        experiment.data_config.action_mode = ActionMode(self.model_action_mode)
        inference._initialize(
            model=model,
            model_name_or_path=str(model_path),
            norm_stats_path=str(norm_stats_path),
            n_bins=int(self.model_cfg.get("n_bins") or 256),
            model_max_length=int(self.model_cfg.get("model_max_length") or 1536),
            use_absolute_action=False,
            add_state=self.add_state,
            is_history=True,
        )

        state_desc = []
        for arm_dim, ee_dim in zip(
            self.robot_action_dim_info["arm_dim"],
            self.robot_action_dim_info["ee_dim"],
        ):
            state_desc.extend([RobotStateDesc.JOINT] * arm_dim)
            state_desc.extend([RobotStateDesc.GRIPPER] * ee_dim)

        self._inference = inference
        self._robot_type = str(self.model_cfg.get("robot_type") or _DEFAULT_ROBOT_TYPE)
        self._state_desc = state_desc

    def update_obs(self, obs: dict) -> None:
        evaluation_id = self._evaluation_id(obs)
        env_idx = obs.get("env_idx", 0)
        if "env_idx" not in obs:
            obs = {**obs, "env_idx": env_idx}
        self._set_latest_env_indices([env_idx], evaluation_id)
        self._observations[self._state_key(env_idx, evaluation_id)] = obs
        self._append_history_observation(env_idx, obs, evaluation_id)

    def update_obs_batch(self, obs_list: list[dict]) -> None:
        evaluation_id = self._evaluation_id(obs_list[0]) if obs_list else None
        env_idx_list = []
        for fallback_idx, obs in enumerate(obs_list):
            if self._evaluation_id(obs) != evaluation_id:
                raise ValueError("all observations in a batch must share evaluation_id")
            env_idx = obs.get("env_idx", fallback_idx)
            if "env_idx" not in obs:
                obs = {**obs, "env_idx": env_idx}
            env_idx_list.append(env_idx)
            self._observations[self._state_key(env_idx, evaluation_id)] = obs
            self._append_history_observation(env_idx, obs, evaluation_id)
        self._set_latest_env_indices(env_idx_list, evaluation_id)

    def _append_history_observation(
        self,
        env_idx: Any,
        obs: dict,
        evaluation_id: str | None = None,
    ) -> None:
        state_key = self._state_key(env_idx, evaluation_id)
        sequence = self._history_sequence_by_env.get(state_key, 0) + 1
        self._history_sequence_by_env[state_key] = sequence
        completed_actions = sequence - 1
        if completed_actions % self.history_action_interval != 0:
            return
        history = self._history_by_env.setdefault(
            state_key,
            deque(maxlen=self._history_buffer_maxlen),
        )
        history.append(
            (
                sequence,
                completed_actions,
                _extract_rgb_image(obs, self.history_image_key).copy(),
            )
        )

    def _history_images_for(
        self,
        env_idx: Any,
        evaluation_id: str | None = None,
    ) -> list[Any | None]:
        """Return 20 oldest-to-newest past slots, left-padded with None."""
        from PIL import Image

        state_key = self._state_key(env_idx, evaluation_id)
        retained = list(self._history_by_env.get(state_key, ()))
        completed_actions = max(
            self._history_sequence_by_env.get(state_key, 0) - 1,
            0,
        )
        frames = [frame for frame in retained if completed_actions - frame[1] >= self.history_action_interval][
            -self.history_slots :
        ]
        return [None] * (self.history_slots - len(frames)) + [
            Image.fromarray(image, mode="RGB") for _, _, image in frames
        ]

    def get_action(
        self,
        scope: dict[str, Any] | None = None,
    ) -> list[dict[str, np.ndarray]]:
        evaluation_id = self._evaluation_id(scope, required=scope is not None)
        env_idx = self._latest_env_indices(evaluation_id)[0]
        return self._predict(self._observation_for(env_idx, evaluation_id))

    def get_action_batch(
        self,
        env_idx_list: dict[str, Any] | list[Any] | None = None,
    ) -> list[list[dict[str, np.ndarray]]]:
        request = env_idx_list
        if isinstance(request, dict):
            evaluation_id = self._evaluation_id(request, required=True)
            indices = request.get("env_idx_list")
            if indices is None:
                indices = self._latest_env_indices(evaluation_id)
            elif not isinstance(indices, (list, tuple)):
                raise TypeError("env_idx_list must be a list or tuple")
        else:
            evaluation_id = None
            indices = self._latest_env_indices() if request is None else request
        return [
            self._predict(self._observation_for(env_idx, evaluation_id))
            for env_idx in indices
        ]

    def _observation_for(
        self,
        env_idx: Any,
        evaluation_id: str | None = None,
    ) -> dict:
        state_key = self._state_key(env_idx, evaluation_id)
        if state_key not in self._observations:
            raise RuntimeError(f"No observation buffered for env_idx={env_idx!r}; call update_obs first")
        return self._observations[state_key]

    def _predict(self, observation: dict) -> list[dict[str, np.ndarray]]:
        from PIL import Image

        evaluation_id = self._evaluation_id(observation)
        env_idx = observation.get("env_idx", 0)
        state = pack_robot_state(
            observation,
            self.action_type,
            self.robot_action_dim_info,
            source_type="obs",
            state_type="state",
        ).astype(np.float32)
        payload = {
            image_key: Image.fromarray(_extract_rgb_image(observation, image_key)) for image_key in _CAMERA_CANDIDATES
        }
        payload.update(
            {
                "history_images": self._history_images_for(env_idx, evaluation_id),
                "prompt": _normalize_prompt(observation, self.default_prompt),
                "state": state,
                "meta_data": {
                    "robot_type": self._robot_type,
                    "control_mode": self.model_cfg.get("control_mode"),
                    "speed": str(self.model_cfg.get("speed", "0.5")),
                    "state_desc": self._state_desc,
                    "valid_dim_mask": np.ones(self.action_dim, dtype=bool),
                },
            }
        )

        action_chunk = np.asarray(self._inference._predict(payload), dtype=np.float32)
        if action_chunk.ndim == 1:
            action_chunk = action_chunk[None, :]
        if action_chunk.ndim != 2 or action_chunk.shape[1] != self.action_dim:
            raise ValueError(f"OpenDM returned {action_chunk.shape}; expected (horizon, {self.action_dim})")
        return [
            unpack_robot_state(
                action,
                self.action_type,
                self.robot_action_dim_info,
                source_type="obs",
            )
            for action in action_chunk[: self.action_steps]
        ]

    def reset(self) -> None:
        self._observations.clear()
        self._history_by_env.clear()
        self._history_sequence_by_env.clear()
        self._latest_env_idx_by_evaluation.clear()
        self._latest_env_idx_list = [0]

    def reset_evaluation(self, scope: dict[str, Any]) -> None:
        """Clear one OpenDM client without touching concurrent evaluations."""

        evaluation_id = self._evaluation_id(scope, required=True)
        assert evaluation_id is not None
        self._clear_evaluation_state(evaluation_id)

    def _clear_evaluation_state(self, evaluation_id: str) -> None:
        for mapping in (
            self._observations,
            self._history_by_env,
            self._history_sequence_by_env,
        ):
            for state_key in list(mapping):
                if (
                    isinstance(state_key, tuple)
                    and len(state_key) == 2
                    and state_key[0] == evaluation_id
                ):
                    del mapping[state_key]
        self._latest_env_idx_by_evaluation.pop(evaluation_id, None)
