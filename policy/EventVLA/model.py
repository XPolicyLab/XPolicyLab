from __future__ import annotations

import ast
import logging
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import Any, Sequence

import cv2
import numpy as np

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)


_CUR_DIR = Path(__file__).resolve().parent
_EVENTVLA_RAW_TO_XPOLICY = np.asarray(
    [0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13],
    dtype=np.int64,
)
_XPOLICY_TO_EVENTVLA_RAW = np.asarray(
    [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13],
    dtype=np.int64,
)


@dataclass
class RuntimeState:
    image_history: Any = None
    initial_images: list[np.ndarray] | None = None
    initial_step: int | None = None
    raw_actions: np.ndarray | None = None
    plan_start_step: int | None = None
    pending_commit_plan_start_step: int | None = None
    pending_commit_offset: int | None = None
    pending_commit_confidence: float = 0.0
    pending_commit_done: bool = True
    last_committed_keyframe_step: int | None = None
    committed_keyframe_count: int = 0
    keyframe_image_memory: list[dict[str, Any]] = field(default_factory=list)
    first_chunk_random_replan_step: int | None = None
    first_chunk_random_replan_done: bool = False
    first_chunk_random_replan_rng: Random | None = None
    task_description: str | None = None
    initial_state: np.ndarray | None = None
    prev_action: np.ndarray | None = None
    step: int = 0


def _optional_path(value: str | None, *base_dirs: Path) -> Path | None:
    if value in (None, "", "null", "None"):
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    for base_dir in base_dirs:
        candidate = base_dir / path
        if candidate.exists():
            return candidate
    return base_dirs[0] / path


def _decode_image(image: Any) -> np.ndarray:
    if isinstance(image, (bytes, bytearray, memoryview)):
        image = np.frombuffer(bytes(image), dtype=np.uint8)

    image = np.asarray(image)
    if image.ndim == 1 and image.dtype == np.uint8:
        decoded = cv2.imdecode(image, cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError("Failed to decode compressed image bytes.")
        image = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

    if image.ndim != 3:
        raise ValueError(f"Expected HWC/CHW image, got shape {image.shape}.")
    if image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.shape[-1] != 3:
        raise ValueError(f"Expected 3 image channels, got shape {image.shape}.")

    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255.0).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)
    return image


def _extract_camera(observation: dict[str, Any], camera_names: list[str]) -> np.ndarray:
    vision = observation.get("vision", {})
    for camera_name in camera_names:
        if camera_name not in vision:
            continue
        camera_obs = vision[camera_name]
        if isinstance(camera_obs, dict):
            for image_key in ("color", "rgb", "colors"):
                if image_key in camera_obs:
                    return _decode_image(camera_obs[image_key])
        else:
            return _decode_image(camera_obs)
    raise KeyError(f"Missing camera from candidates: {camera_names}")


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off", "none", "null"}
    return bool(value)


def _maybe_literal(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except Exception:
        return value


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = dict(model_cfg)
        self.action_type = self.model_cfg.get("action_type", "joint")
        if self.action_type != "joint":
            raise ValueError("EventVLA currently supports action_type='joint' first.")

        self.env_cfg_type = self.model_cfg.get("env_cfg_type")
        if self.env_cfg_type is None:
            raise ValueError("EventVLA requires env_cfg_type.")
        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        self.action_dim = sum(self.robot_action_dim_info["arm_dim"]) + sum(
            self.robot_action_dim_info["ee_dim"]
        )
        if self.action_dim != 14:
            raise NotImplementedError("EventVLA adapter currently expects arx_x5 joint action_dim=14.")

        eventvla_root = _optional_path(
            self.model_cfg.get("eventvla_root"),
            _CUR_DIR,
        ) or (_CUR_DIR / "source_eventvla")
        eventvla_root = eventvla_root.resolve()
        if str(eventvla_root) not in sys.path:
            sys.path.insert(0, str(eventvla_root))
        self.eventvla_root = eventvla_root

        from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
        from eventvla.model.legacy_compat import normalize_legacy_checkpoint_config
        from eventvla.model.memory_ablation import KEYFRAME_IMAGE_MEMORY_MODES
        from eventvla.model.tools import read_mode_config

        self._client_cls = WebsocketClientPolicy
        self.checkpoint_path = _optional_path(
            self.model_cfg.get("checkpoint_path"),
            _CUR_DIR,
        )
        if self.checkpoint_path is None or not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing EventVLA checkpoint: {self.checkpoint_path}")
        self.checkpoint_path = self.checkpoint_path.resolve()

        model_config, norm_stats = read_mode_config(self.checkpoint_path)
        self.model_config, compat_info = normalize_legacy_checkpoint_config(model_config)
        if compat_info.get("enabled", False):
            logging.info(
                "Using legacy %s checkpoint as %s: %s",
                compat_info.get("source_framework_name"),
                compat_info.get("target_framework_name"),
                compat_info.get("reason"),
            )
        self.norm_stats = norm_stats
        self.unnorm_key = self._check_unnorm_key(
            self.norm_stats,
            self.model_cfg.get("unnorm_key", None),
        )
        self.vla_data_config = self._nested_get(self.model_config, ("datasets", "vla_data"), {}) or {}
        self.framework_config = self._nested_get(self.model_config, ("framework",), {}) or {}
        self.memory_config = self._nested_get(self.model_config, ("framework", "memory_buffer"), {}) or {}
        self.qwen_memory_injection_config = self.memory_config.get("qwen_memory_injection", {}) or {}
        self.memory_injection_mode = str(
            self.qwen_memory_injection_config.get(
                "mode",
                self.framework_config.get("memory_ablation_mode", ""),
            )
        ).strip().lower()
        self.uses_keyframe_image_inputs = self.memory_injection_mode in set(KEYFRAME_IMAGE_MEMORY_MODES)
        self.max_keyframe_images = int(
            self.qwen_memory_injection_config.get(
                "max_keyframe_images",
                self.memory_config.get("max_keyframe_images", 4),
            )
        )

        self.action_mode = str(self.model_cfg.get("action_mode", self.vla_data_config.get("action_mode", "abs"))).lower()
        self.action_norm_stats = self._get_action_stats(self.unnorm_key, self.action_mode)
        self.state_norm_stats = self._get_state_stats(self.unnorm_key)
        self.action_chunk_size = self._get_action_chunk_size()

        trained_sampling_interval = self.vla_data_config.get("sampling_interval", None)
        sampling_interval = self.model_cfg.get("sampling_interval", None)
        if sampling_interval in (None, "", "null", "None"):
            sampling_interval = trained_sampling_interval
        self.sampling_interval = (
            max(1, int(sampling_interval)) if sampling_interval is not None else int(self.action_chunk_size)
        )
        self.use_ddim = _as_bool(self.model_cfg.get("use_ddim", True), True)
        self.num_ddim_steps = int(self.model_cfg.get("num_ddim_steps", 10))
        self.image_size = self._resolve_image_size(
            override=self.model_cfg.get("image_size", None),
            trained=self.vla_data_config.get("image_size", None),
        )
        temporal_image_config = self._nested_get(self.vla_data_config, ("temporal", "image"), {}) or {}
        self.temporal_absolute_indices = self._resolve_int_sequence(
            override=self.model_cfg.get("temporal_absolute_indices", None),
            trained=temporal_image_config.get("absolute_indices", None),
            fallback=(0,),
        )
        self.temporal_anchor_deltas = self._resolve_int_sequence(
            override=self.model_cfg.get("temporal_anchor_deltas", None),
            trained=temporal_image_config.get("delta_indices", None),
            fallback=(-30, -15, 0),
        )
        temporal_view_names = self.model_cfg.get(
            "temporal_view_names",
            temporal_image_config.get("view_names", None),
        )
        temporal_view_names = _maybe_literal(temporal_view_names)
        if temporal_view_names in (None, "", "null", "None"):
            temporal_view_names = ("cam_high", "cam_left_wrist", "cam_right_wrist")
        self.temporal_view_names = tuple(
            str(name) for name in temporal_view_names
        )
        negative_delta_history = abs(min([0, *self.temporal_anchor_deltas])) + 1
        absolute_history = max([0, *[int(idx) for idx in self.temporal_absolute_indices]]) + 1
        self.temporal_history_capacity = max(negative_delta_history, absolute_history)

        self.first_chunk_random_replan = _as_bool(
            self.model_cfg.get("first_chunk_random_replan", True),
            True,
        )
        self.first_chunk_random_replan_seed = int(self.model_cfg.get("first_chunk_random_replan_seed", 42))
        fixed_replan = self.model_cfg.get("first_chunk_fixed_replan_step", None)
        self.first_chunk_fixed_replan_step = (
            None if fixed_replan in (None, "", "null", "None") else max(1, int(fixed_replan))
        )
        self.replan_after_keyframe_commit = (
            _as_bool(self.model_cfg.get("replan_after_keyframe_commit", True), True)
            and self.uses_keyframe_image_inputs
        )
        keyframe_cluster_timestep_window = self.model_cfg.get("keyframe_cluster_timestep_window", None)
        if keyframe_cluster_timestep_window in (None, "", "null", "None"):
            keyframe_cluster_timestep_window = self.memory_config.get("keyframe_cluster_timestep_window", 20)
        self.keyframe_cluster_timestep_window = int(keyframe_cluster_timestep_window or 20)

        self.client = self._connect_client()
        self.obs_by_env: dict[int, dict[str, Any]] = {}
        self.runtime_by_env: dict[int, RuntimeState] = {}
        self._latest_env_idx_list = [0]

        print(
            f"[EventVLA] connected to EventVLA server, action_dim={self.action_dim}, "
            f"chunk={self.action_chunk_size}, sampling_interval={self.sampling_interval}, "
            f"unnorm_key={self.unnorm_key}, action_mode={self.action_mode}, "
            f"temporal_abs={self.temporal_absolute_indices}, temporal_delta={self.temporal_anchor_deltas}, "
            f"uses_keyframe_images={self.uses_keyframe_image_inputs}"
        )

    def _connect_client(self):
        return self._client_cls(
            self.model_cfg.get("eventvla_server_host", "127.0.0.1"),
            int(self.model_cfg.get("eventvla_server_port", 5694)),
        )

    def _call_server_with_reconnect(self, method_name: str, *args):
        try:
            return getattr(self.client, method_name)(*args)
        except Exception as first_error:
            logging.warning("EventVLA websocket %s failed; reconnecting once: %s", method_name, first_error)
            try:
                self.client.close()
            except Exception:
                pass
            self.client = self._connect_client()
            return getattr(self.client, method_name)(*args)

    def _new_runtime(self) -> RuntimeState:
        runtime = RuntimeState()
        runtime.image_history = deque(maxlen=self.temporal_history_capacity)
        runtime.first_chunk_random_replan_rng = Random(self.first_chunk_random_replan_seed)
        return runtime

    def _get_runtime(self, env_idx: int) -> RuntimeState:
        if env_idx not in self.runtime_by_env:
            self.runtime_by_env[env_idx] = self._new_runtime()
        return self.runtime_by_env[env_idx]

    def _convert_obs(self, observation: dict[str, Any]) -> dict[str, Any]:
        images = [
            _extract_camera(observation, ["cam_head", "head_camera"]),
            _extract_camera(observation, ["cam_left_wrist", "left_camera"]),
            _extract_camera(observation, ["cam_right_wrist", "right_camera"]),
        ]

        instruction = observation.get("instruction") or observation.get("instructions")
        if isinstance(instruction, (list, tuple)):
            instruction = instruction[0] if instruction else ""
        if instruction in (None, ""):
            instruction = self.model_cfg.get("task_name", "")

        converted_obs = {
            "lang": str(instruction),
            "image": images,
        }
        try:
            converted_obs["state"] = pack_robot_state(
                observation,
                self.action_type,
                self.robot_action_dim_info,
                source_type="obs",
            ).astype(np.float32)
        except KeyError:
            if self.action_mode in {"delta", "rel"}:
                raise
        return converted_obs

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self._latest_env_idx_list = []
        for obs in obs_list:
            env_idx = int(obs.get("env_idx", 0))
            self._latest_env_idx_list.append(env_idx)
            self.obs_by_env[env_idx] = self._convert_obs(obs)

    def _step_runtime(self, runtime: RuntimeState, obs: dict[str, Any], step: int) -> np.ndarray:
        state = obs.get("state", None)
        state_for_mode = None
        if state is not None:
            state_for_mode = self._xpolicy_to_eventvla_raw(np.asarray(state, dtype=np.float32))

        if self.action_mode in {"delta", "rel"} and runtime.initial_state is None:
            if state_for_mode is None:
                raise ValueError(f"action_mode='{self.action_mode}' requires state in observation.")
            runtime.initial_state = state_for_mode.copy()

        task_description = obs.get("lang", "")
        if task_description != runtime.task_description:
            self._reset_runtime(runtime, keep_step=False)
            runtime.task_description = task_description
            if self.action_mode in {"delta", "rel"} and state_for_mode is not None:
                runtime.initial_state = state_for_mode.copy()

        raw_images = self._clone_images(obs["image"])
        images = [self._resize_image(image) for image in raw_images]
        self._record_temporal_images(runtime, images=images, step=int(step))
        temporal_images, image_metas = self._build_temporal_anchor_images(runtime, step=int(step))

        example_copy = {
            "lang": str(task_description),
            "image": temporal_images,
            "image_metas": image_metas,
            "timestep": int(step),
        }

        replan_after_keyframe_commit = self._maybe_commit_pending_event(
            runtime=runtime,
            step=int(step),
            raw_images=raw_images,
        )
        self._inject_keyframe_memory_fields(runtime, example_copy, step=int(step))

        action_idx = None if runtime.plan_start_step is None else int(step) - int(runtime.plan_start_step)
        replan_at_random_second_anchor = self._should_trigger_first_chunk_random_replan(runtime, step=int(step))
        needs_replan = (
            runtime.raw_actions is None
            or runtime.plan_start_step is None
            or action_idx is None
            or action_idx < 0
            or action_idx >= len(runtime.raw_actions)
            or replan_at_random_second_anchor
            or replan_after_keyframe_commit
        )

        if needs_replan:
            is_first_plan = runtime.raw_actions is None or runtime.plan_start_step is None
            if not is_first_plan:
                self._consume_first_chunk_random_replan(runtime)
            runtime.plan_start_step = int(step)
            vla_input = {
                "examples": [example_copy],
                "do_sample": False,
                "use_ddim": self.use_ddim,
                "num_ddim_steps": self.num_ddim_steps,
            }
            response = self._call_server_with_reconnect("predict_action", vla_input)
            self._raise_for_server_error(response)
            response_data = response.get("data", {}) if isinstance(response, dict) else {}
            if "normalized_actions" not in response_data:
                raise KeyError(f"EventVLA response missing normalized_actions: {response_data.keys()}")
            normalized_actions = np.asarray(response_data["normalized_actions"][0], dtype=np.float32)
            self._update_pending_commit_from_response(runtime, response_data=response_data, step=int(step))
            raw_actions = self._unnormalize_actions(normalized_actions, self.action_norm_stats)

            if self.action_mode == "delta":
                runtime.raw_actions = self._delta_to_absolute(runtime, raw_actions, state_for_mode)
            elif self.action_mode == "rel":
                runtime.raw_actions = self._rel_to_absolute(runtime, raw_actions)
            else:
                runtime.raw_actions = raw_actions

            if is_first_plan:
                self._arm_first_chunk_random_replan(runtime, plan_start_step=int(step))

        if runtime.plan_start_step is None or runtime.raw_actions is None:
            raise RuntimeError("EventVLA failed to create an action plan.")
        action_idx = int(step) - int(runtime.plan_start_step)
        if action_idx >= len(runtime.raw_actions):
            raise IndexError(
                f"Action index {action_idx} out of range for cached chunk of length {len(runtime.raw_actions)}"
            )
        current_action = np.asarray(runtime.raw_actions[action_idx], dtype=np.float32)
        if self.action_mode == "delta":
            runtime.prev_action = current_action.copy()
        return current_action

    def _next_action_vector(self, env_idx: int) -> np.ndarray:
        if env_idx not in self.obs_by_env:
            raise AssertionError("update_obs must be called before get_action.")

        runtime = self._get_runtime(env_idx)
        raw_action = self._step_runtime(runtime, self.obs_by_env[env_idx], runtime.step)
        runtime.step += 1
        action = self._eventvla_raw_to_xpolicy(raw_action)
        if action.shape[-1] != self.action_dim:
            raise ValueError(f"Expected action dim {self.action_dim}, got {action.shape[-1]}.")
        return action.astype(np.float32)

    def get_action(self):
        return self.get_action_batch(env_idx_list=[self._latest_env_idx_list[0]])[0]

    def get_action_batch(self, env_idx_list=None):
        if env_idx_list is None:
            env_idx_list = self._latest_env_idx_list
        return [
            [
                unpack_robot_state(
                    self._next_action_vector(int(env_idx)),
                    self.action_type,
                    self.robot_action_dim_info,
                    source_type="obs",
                )
            ]
            for env_idx in env_idx_list
        ]

    def reset(self):
        self.obs_by_env.clear()
        self.runtime_by_env.clear()
        self._latest_env_idx_list = [0]
        try:
            self._call_server_with_reconnect("reset_memory")
        except Exception as exc:
            logging.warning("Failed to reset EventVLA server memory: %s", exc)

    def _reset_runtime(self, runtime: RuntimeState, keep_step: bool) -> None:
        step = runtime.step if keep_step else 0
        rng = runtime.first_chunk_random_replan_rng or Random(self.first_chunk_random_replan_seed)
        runtime.image_history.clear()
        runtime.initial_images = None
        runtime.initial_step = None
        runtime.raw_actions = None
        runtime.plan_start_step = None
        runtime.pending_commit_plan_start_step = None
        runtime.pending_commit_offset = None
        runtime.pending_commit_confidence = 0.0
        runtime.pending_commit_done = True
        runtime.last_committed_keyframe_step = None
        runtime.committed_keyframe_count = 0
        runtime.keyframe_image_memory.clear()
        runtime.first_chunk_random_replan_step = None
        runtime.first_chunk_random_replan_done = False
        runtime.first_chunk_random_replan_rng = rng
        runtime.initial_state = None
        runtime.prev_action = None
        runtime.step = step

    @staticmethod
    def _clone_images(images: Sequence[np.ndarray]) -> list[np.ndarray]:
        return [np.asarray(image).copy() for image in images]

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image_array = np.asarray(image)
        if image_array.dtype != np.uint8:
            image_array = np.clip(image_array, 0, 255).astype(np.uint8)
        return cv2.resize(image_array, tuple(self.image_size), interpolation=cv2.INTER_AREA)

    def _record_temporal_images(self, runtime: RuntimeState, images: Sequence[np.ndarray], step: int) -> None:
        cloned = self._clone_images(images)
        if runtime.initial_images is None:
            runtime.initial_images = self._clone_images(cloned)
            runtime.initial_step = int(step)

        record = {"step": int(step), "images": cloned}
        if runtime.image_history and int(runtime.image_history[-1]["step"]) == int(step):
            runtime.image_history[-1] = record
        else:
            runtime.image_history.append(record)

    def _images_at_or_before(self, runtime: RuntimeState, target_step: int) -> list[np.ndarray]:
        if not runtime.image_history:
            if runtime.initial_images is None:
                raise RuntimeError("temporal image history is empty")
            return self._clone_images(runtime.initial_images)

        selected = None
        for record in runtime.image_history:
            if int(record["step"]) <= int(target_step):
                selected = record
            else:
                break

        if selected is None:
            if runtime.initial_images is not None:
                return self._clone_images(runtime.initial_images)
            selected = runtime.image_history[0]
        return self._clone_images(selected["images"])

    def _build_temporal_anchor_images(self, runtime: RuntimeState, step: int) -> tuple[list[np.ndarray], list[dict]]:
        if runtime.initial_images is None:
            raise RuntimeError("initial images are not recorded")

        frame_specs = []
        initial_step = int(runtime.initial_step or 0)
        for absolute_index in self.temporal_absolute_indices:
            absolute_index = int(absolute_index)
            if absolute_index == 0:
                images = self._clone_images(runtime.initial_images)
                time_role = "first"
            else:
                images = self._images_at_or_before(runtime, initial_step + absolute_index)
                time_role = "absolute"
            frame_specs.append(
                {
                    "images": images,
                    "time_role": time_role,
                    "delta": None,
                    "absolute_index": absolute_index,
                }
            )
        for delta in self.temporal_anchor_deltas:
            role = "current" if int(delta) == 0 else "history"
            frame_specs.append(
                {
                    "images": self._images_at_or_before(runtime, int(step) + int(delta)),
                    "time_role": role,
                    "delta": int(delta),
                    "absolute_index": None,
                }
            )

        anchor_images: list[np.ndarray] = []
        image_metas: list[dict] = []
        for frame_idx, frame_spec in enumerate(frame_specs):
            for view_idx, image in enumerate(frame_spec["images"]):
                view_name = (
                    self.temporal_view_names[view_idx]
                    if view_idx < len(self.temporal_view_names)
                    else f"view_{view_idx}"
                )
                anchor_images.append(image)
                image_metas.append(
                    {
                        "role": "anchor",
                        "time_index": int(frame_idx),
                        "time_role": frame_spec["time_role"],
                        "delta": frame_spec["delta"],
                        "absolute_index": frame_spec["absolute_index"],
                        "view": view_name,
                        "view_index": int(view_idx),
                    }
                )
        return anchor_images, image_metas

    def _select_main_view_for_keyframe_memory(self, images: Sequence[np.ndarray]) -> int:
        if len(images) <= 0:
            raise RuntimeError("keyframe image memory requires at least one raw image")
        memory_views = self.memory_config.get("memory_views", {}) or {}
        include_names = tuple(
            str(name).lower()
            for name in self.qwen_memory_injection_config.get(
                "include_names",
                memory_views.get("include_names", ["cam_high", "head", "main"]),
            )
        )
        exclude_patterns = tuple(
            str(name).lower()
            for name in self.qwen_memory_injection_config.get(
                "exclude_name_patterns",
                memory_views.get("exclude_name_patterns", ["wrist"]),
            )
        )
        for idx, view_name in enumerate(self.temporal_view_names):
            view_text = str(view_name).lower()
            if any(pattern and pattern in view_text for pattern in exclude_patterns):
                continue
            if any(name and name in view_text for name in include_names):
                return int(idx)
        for idx, view_name in enumerate(self.temporal_view_names):
            if "wrist" not in str(view_name).lower():
                return int(idx)
        return 0

    def _commit_keyframe_image_memory(
        self,
        runtime: RuntimeState,
        raw_images: Sequence[np.ndarray] | None,
        step: int,
        confidence: float,
    ) -> None:
        if not raw_images:
            return
        main_idx = self._select_main_view_for_keyframe_memory(raw_images)
        main_idx = min(max(main_idx, 0), len(raw_images) - 1)
        image = self._resize_image(np.asarray(raw_images[main_idx]))
        view_name = (
            self.temporal_view_names[main_idx]
            if main_idx < len(self.temporal_view_names)
            else f"view_{main_idx}"
        )
        entry = {
            "step": int(step),
            "images": [image],
            "metas": [
                {
                    "role": "memory_keyframe",
                    "time_role": "memory",
                    "source_timestep": int(step),
                    "view": str(view_name),
                    "view_index": int(main_idx),
                    "confidence": float(confidence),
                }
            ],
            "confidence": float(confidence),
        }
        if runtime.keyframe_image_memory and int(runtime.keyframe_image_memory[-1]["step"]) == int(step):
            runtime.keyframe_image_memory[-1] = entry
        else:
            runtime.keyframe_image_memory.append(entry)
        if self.max_keyframe_images > 0 and len(runtime.keyframe_image_memory) > self.max_keyframe_images:
            runtime.keyframe_image_memory = runtime.keyframe_image_memory[-self.max_keyframe_images :]

    def _build_keyframe_memory_images(
        self,
        runtime: RuntimeState,
        step: int,
    ) -> tuple[list[np.ndarray], list[dict], list[int]]:
        if not self.uses_keyframe_image_inputs:
            return [], [], []

        visible_entries = [
            entry
            for entry in runtime.keyframe_image_memory
            if int(entry.get("step", -1)) <= int(step)
        ]
        if self.max_keyframe_images > 0:
            visible_entries = visible_entries[-self.max_keyframe_images :]
        else:
            visible_entries = []

        images: list[np.ndarray] = []
        metas: list[dict] = []
        steps: list[int] = []
        for entry in visible_entries:
            entry_step = int(entry.get("step", -1))
            for image, meta in zip(entry.get("images", []), entry.get("metas", [])):
                images.append(np.asarray(image).copy())
                metas.append(dict(meta))
                steps.append(entry_step)
        return images, metas, steps

    def _inject_keyframe_memory_fields(self, runtime: RuntimeState, example: dict, step: int) -> None:
        memory_images, memory_metas, memory_steps = self._build_keyframe_memory_images(runtime, int(step))
        example["memory_keyframe_images"] = memory_images
        example["memory_keyframe_image_metas"] = memory_metas
        example["memory_keyframe_steps"] = memory_steps
        example["memory_keyframe_count"] = int(len(memory_images))

    def _clear_pending_commit(self, runtime: RuntimeState) -> None:
        runtime.pending_commit_plan_start_step = None
        runtime.pending_commit_offset = None
        runtime.pending_commit_confidence = 0.0
        runtime.pending_commit_done = True

    @staticmethod
    def _extract_scalar_response_value(response_data: dict, key: str, default=None, cast=None):
        value = response_data.get(key, default)
        if value is None:
            return default
        value_array = np.asarray(value)
        if value_array.size == 0:
            return default
        scalar = value_array.reshape(-1)[0].item()
        if cast is None:
            return scalar
        return cast(scalar)

    def _update_pending_commit_from_response(self, runtime: RuntimeState, response_data: dict, step: int) -> None:
        if not self.uses_keyframe_image_inputs:
            self._clear_pending_commit(runtime)
            return

        should_trigger = bool(self._extract_scalar_response_value(response_data, "should_trigger_event", False, bool))
        if not should_trigger:
            return

        event_offset = self._extract_scalar_response_value(response_data, "pred_event_offset", -1, int)
        if event_offset is None or int(event_offset) < 0:
            return

        runtime.pending_commit_plan_start_step = int(step)
        runtime.pending_commit_offset = int(event_offset)
        runtime.pending_commit_confidence = float(
            self._extract_scalar_response_value(response_data, "pred_event_confidence", 0.0, float)
        )
        runtime.pending_commit_done = False

    def _maybe_commit_pending_event(
        self,
        runtime: RuntimeState,
        step: int,
        raw_images: Sequence[np.ndarray] | None = None,
    ) -> bool:
        if not self.uses_keyframe_image_inputs:
            self._clear_pending_commit(runtime)
            return False
        if runtime.pending_commit_done:
            return False
        if runtime.pending_commit_plan_start_step is None or runtime.pending_commit_offset is None:
            return False

        local_step = int(step) - int(runtime.pending_commit_plan_start_step)
        if local_step != int(runtime.pending_commit_offset):
            return False

        self._commit_keyframe_image_memory(
            runtime=runtime,
            raw_images=raw_images,
            step=int(step),
            confidence=runtime.pending_commit_confidence,
        )
        runtime.pending_commit_done = True
        runtime.committed_keyframe_count += 1
        runtime.last_committed_keyframe_step = int(step)
        return self.replan_after_keyframe_commit and runtime.committed_keyframe_count < 4

    def _arm_first_chunk_random_replan(self, runtime: RuntimeState, plan_start_step: int) -> None:
        if (not self.first_chunk_random_replan) or runtime.first_chunk_random_replan_done:
            return
        if self.first_chunk_fixed_replan_step is not None:
            runtime.first_chunk_random_replan_step = int(plan_start_step) + int(self.first_chunk_fixed_replan_step)
            return
        upper = max(1, min(int(self.sampling_interval), int(self.action_chunk_size)))
        rng = runtime.first_chunk_random_replan_rng or Random(self.first_chunk_random_replan_seed)
        random_offset = int(rng.randint(1, upper))
        runtime.first_chunk_random_replan_rng = rng
        runtime.first_chunk_random_replan_step = int(plan_start_step) + random_offset

    @staticmethod
    def _consume_first_chunk_random_replan(runtime: RuntimeState) -> None:
        runtime.first_chunk_random_replan_step = None
        runtime.first_chunk_random_replan_done = True

    def _should_trigger_first_chunk_random_replan(self, runtime: RuntimeState, step: int) -> bool:
        if (not self.first_chunk_random_replan) or runtime.first_chunk_random_replan_done:
            return False
        if runtime.first_chunk_random_replan_step is None:
            return False
        return int(step) >= int(runtime.first_chunk_random_replan_step)

    @staticmethod
    def _raise_for_server_error(response: Any) -> None:
        if not isinstance(response, dict) or response.get("ok", True):
            return
        error = response.get("error", {})
        message = error.get("message", response) if isinstance(error, dict) else error
        raise RuntimeError(f"EventVLA server returned error: {message}")

    @staticmethod
    def _unnormalize_actions(normalized_actions: np.ndarray, action_norm_stats: dict) -> np.ndarray:
        mask = np.asarray(action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool)), dtype=bool)
        action_high = np.asarray(action_norm_stats["max"], dtype=np.float32)
        action_low = np.asarray(action_norm_stats["min"], dtype=np.float32)
        normalized_actions = np.clip(np.asarray(normalized_actions, dtype=np.float32), -1, 1)
        return np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        ).astype(np.float32)

    def _delta_to_absolute(
        self,
        runtime: RuntimeState,
        delta_actions: np.ndarray,
        current_state: np.ndarray | None,
    ) -> np.ndarray:
        if runtime.initial_state is None and current_state is None:
            raise ValueError("delta action mode requires an initial state.")
        abs_actions = np.zeros_like(delta_actions)
        mask = np.asarray(self.action_norm_stats.get("mask", np.ones(delta_actions.shape[-1], dtype=bool)), dtype=bool)
        base = runtime.prev_action if runtime.prev_action is not None else runtime.initial_state
        for idx in range(len(delta_actions)):
            abs_actions[idx] = np.where(mask, delta_actions[idx] + base, delta_actions[idx])
            base = abs_actions[idx]
        return abs_actions

    def _rel_to_absolute(self, runtime: RuntimeState, rel_actions: np.ndarray) -> np.ndarray:
        if runtime.initial_state is None:
            raise ValueError("rel action mode requires an initial state.")
        abs_actions = np.zeros_like(rel_actions)
        mask = np.asarray(self.action_norm_stats.get("mask", np.ones(rel_actions.shape[-1], dtype=bool)), dtype=bool)
        for idx in range(len(rel_actions)):
            abs_actions[idx] = np.where(mask, rel_actions[idx] + runtime.initial_state, rel_actions[idx])
        return abs_actions

    @staticmethod
    def _nested_get(mapping: dict, keys: Sequence[str], default=None):
        current = mapping
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                return default
            current = current[key]
        return current

    @staticmethod
    def _resolve_image_size(override: Any, trained: Any) -> list[int]:
        value = _maybe_literal(override) if override is not None else trained
        if value in (None, "", "null", "None"):
            value = (224, 224)
        value = list(value)
        if len(value) < 2:
            raise ValueError(f"image_size must contain width and height, got {value}")
        return [int(value[0]), int(value[1])]

    @staticmethod
    def _resolve_int_sequence(override: Any, trained: Any, fallback: Sequence[int]) -> tuple[int, ...]:
        value = _maybe_literal(override) if override is not None else trained
        if value in (None, "", "null", "None"):
            value = fallback
        return tuple(int(item) for item in value)

    @staticmethod
    def _check_unnorm_key(norm_stats: dict, unnorm_key: str | None):
        if unnorm_key in (None, "", "null", "None"):
            return next(iter(norm_stats.keys()))
        if unnorm_key not in norm_stats:
            fallback_key = next(iter(norm_stats.keys()))
            logging.warning(
                "unnorm_key=%s is not present in dataset_statistics.json; using %s instead.",
                unnorm_key,
                fallback_key,
            )
            return fallback_key
        return unnorm_key

    def _get_action_stats(self, unnorm_key: str, action_mode: str) -> dict:
        stats = self.norm_stats[unnorm_key]
        if action_mode in stats:
            mode_stats = stats[action_mode]
            return mode_stats.get("action", mode_stats)
        if "action" in stats:
            if action_mode != "abs":
                logging.warning(
                    "Statistics file only has abs action stats, but action_mode=%s was requested.",
                    action_mode,
                )
            return stats["action"]
        raise ValueError(f"Invalid statistics format for unnorm_key={unnorm_key}")

    def _get_state_stats(self, unnorm_key: str) -> dict | None:
        stats = self.norm_stats[unnorm_key]
        if "state" in stats:
            return stats["state"]
        if self.action_mode in stats and "state" in stats[self.action_mode]:
            return stats[self.action_mode]["state"]
        return None

    def _get_action_chunk_size(self) -> int:
        action_model_config = self._nested_get(self.model_config, ("framework", "action_model"), {}) or {}
        future_window = int(action_model_config.get("future_action_window_size", 0))
        past_window = int(action_model_config.get("past_action_window_size", 0))
        return past_window + 1 + future_window

    @staticmethod
    def _eventvla_raw_to_xpolicy(action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        if action.shape[-1] != len(_EVENTVLA_RAW_TO_XPOLICY):
            raise ValueError(f"Expected EventVLA raw action dim 14, got {action.shape[-1]}.")
        return action[..., _EVENTVLA_RAW_TO_XPOLICY]

    @staticmethod
    def _xpolicy_to_eventvla_raw(state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32)
        if state.shape[-1] != len(_XPOLICY_TO_EVENTVLA_RAW):
            raise ValueError(f"Expected XPolicy state dim 14, got {state.shape[-1]}.")
        return state[..., _XPOLICY_TO_EVENTVLA_RAW]
