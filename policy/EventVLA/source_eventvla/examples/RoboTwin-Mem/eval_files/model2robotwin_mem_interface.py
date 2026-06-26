import atexit
import collections
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from random import Random
from typing import Any, Dict, Optional, Sequence
from collections import deque

import numpy as np
import cv2 as cv
from PIL import Image
from websockets.exceptions import ConnectionClosed


from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from eventvla.model.memory_ablation import (
    CANONICAL_MEMORY_ABLATION_MODES,
    KEYFRAME_IMAGE_MEMORY_MODES,
    validate_temporal_image_profile,
)
from eventvla.model.legacy_compat import normalize_legacy_checkpoint_config
from eventvla.model.tools import read_mode_config

try:
    from examples.SimplerEnv.eval_files.adaptive_ensemble import AdaptiveEnsembler
except ImportError:
    AdaptiveEnsembler = None


class ModelClient:
    _KEYFRAME_IMAGE_INPUT_MODES = set(KEYFRAME_IMAGE_MEMORY_MODES)

    def __init__(
        self,
        policy_ckpt_path,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "robotwin_mem",
        horizon: int = 0,
        action_ensemble=False,
        action_ensemble_horizon: Optional[int] = 3,
        image_size: Optional[Sequence[int]] = None,
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        adaptive_ensemble_alpha=0.1,
        host="127.0.0.1",
        port=5694,
        action_mode: str = "abs",
        replan_stride: Optional[int] = None,
        first_chunk_random_replan: bool = True,
        first_chunk_random_replan_seed: int = 42,
        first_chunk_fixed_replan_step: Optional[int] = None,
        sampling_interval: Optional[int] = None,
        replan_after_keyframe_commit: bool = True,
        temporal_anchor_deltas: Optional[Sequence[int]] = None,
        temporal_absolute_indices: Optional[Sequence[int]] = None,
        temporal_view_names: Optional[Sequence[str]] = None,
        keyframe_commit_confidence_threshold: Optional[float] = None,
        keyframe_cluster_timestep_window: Optional[int] = None,
    ) -> None:

        self.host = host
        self.port = port
        self.client: Optional[WebsocketClientPolicy] = None
        self.policy_setup = policy_setup
        self.model_config, self.norm_stats = read_mode_config(policy_ckpt_path)
        self.model_config, compat_info = normalize_legacy_checkpoint_config(self.model_config)
        if compat_info.get("enabled", False):
            logging.info(
                "Using legacy %s checkpoint config as %s in RoboTwin-Mem interface: %s",
                compat_info.get("source_framework_name"),
                compat_info.get("target_framework_name"),
                compat_info.get("reason"),
            )
        self.unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        self.vla_data_config = self._nested_get(self.model_config, ("datasets", "vla_data"), {}) or {}
        self.memory_config = self._nested_get(self.model_config, ("framework", "memory_buffer"), {}) or {}
        self.memory_ablation_mode = str(
            self._nested_get(self.model_config, ("framework", "memory_ablation_mode"), "")
        ).strip().lower()
        if self.memory_ablation_mode not in set(CANONICAL_MEMORY_ABLATION_MODES):
            raise ValueError(
                "Checkpoint config is missing a canonical framework.memory_ablation_mode. "
                "Old checkpoint configs are not supported in this branch."
            )
        self.qwen_memory_injection_config = self.memory_config.get("qwen_memory_injection", {}) or {}
        self.memory_injection_mode = str(
            self.qwen_memory_injection_config.get("mode", "")
        ).lower()
        supported_memory_injection_modes = set(CANONICAL_MEMORY_ABLATION_MODES)
        if self.memory_injection_mode not in supported_memory_injection_modes:
            raise ValueError(
                f"Unsupported qwen_memory_injection.mode `{self.memory_injection_mode}`. "
                f"Use one of: {sorted(supported_memory_injection_modes)}."
            )
        if self.memory_injection_mode != self.memory_ablation_mode:
            raise ValueError(
                "Resolved checkpoint config mismatch: "
                f"framework.memory_ablation_mode={self.memory_ablation_mode} but "
                f"framework.memory_buffer.qwen_memory_injection.mode={self.memory_injection_mode}."
            )
        self.uses_keyframe_image_inputs = self._use_keyframe_image_inputs()
        self.supports_runtime_memory_commits = self.uses_keyframe_image_inputs
        self.max_keyframe_images = int(self.qwen_memory_injection_config.get("max_keyframe_images", 4))
        self.keyframe_image_position = str(
            self.qwen_memory_injection_config.get(
                "keyframe_image_position",
                "after_anchor_images_before_action",
            )
        ).lower()

        trained_action_mode = self.vla_data_config.get("action_mode", None)
        if trained_action_mode is not None and str(trained_action_mode).lower() != str(action_mode).lower():
            logging.warning(
                "Deployment action_mode=%s differs from checkpoint training action_mode=%s.",
                action_mode,
                trained_action_mode,
            )

        self.action_norm_stats = self.get_action_stats(
            self.unnorm_key, policy_ckpt_path=policy_ckpt_path, action_mode=action_mode
        )
        self.action_chunk_size = self.get_action_chunk_size(policy_ckpt_path=policy_ckpt_path)
        trained_sampling_interval = self.vla_data_config.get("sampling_interval", None)
        if sampling_interval is None:
            sampling_interval = trained_sampling_interval
        self.sampling_interval = (
            max(1, int(sampling_interval)) if sampling_interval is not None else int(self.action_chunk_size)
        )
        if replan_stride is not None and int(replan_stride) > 0 and int(replan_stride) != self.action_chunk_size:
            logging.warning(
                "Ignoring replan_stride=%s in noreplan evaluation. The policy executes the full action chunk before replanning.",
                replan_stride,
            )
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.image_size = self._resolve_image_size(
            override=image_size,
            trained=self.vla_data_config.get("image_size", None),
        )
        self.horizon = horizon
        self.action_ensemble = action_ensemble and (AdaptiveEnsembler is not None)
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon

        # Action mode: "abs", "delta", or "rel"
        self.action_mode = action_mode
        self.first_chunk_random_replan = bool(first_chunk_random_replan)
        self.first_chunk_random_replan_rng = Random(int(first_chunk_random_replan_seed))
        self.first_chunk_fixed_replan_step = (
            None if first_chunk_fixed_replan_step is None else max(1, int(first_chunk_fixed_replan_step))
        )
        self.request_replan_after_keyframe_commit = bool(replan_after_keyframe_commit)
        self.replan_after_keyframe_commit = (
            self.request_replan_after_keyframe_commit and self.supports_runtime_memory_commits
        )
        if self.request_replan_after_keyframe_commit and not self.replan_after_keyframe_commit:
            logging.info(
                "Disable replan_after_keyframe_commit for mode=%s because it does not update runtime memory during eval.",
                self.memory_injection_mode,
            )
        # State tracking for delta/rel modes
        self.initial_state = None  # s_0 for rel mode
        self.prev_action = None  # last absolute action for delta mode

        self.task_description = None
        temporal_image_config = self._nested_get(self.vla_data_config, ("temporal", "image"), {}) or {}
        self.temporal_absolute_indices = self._resolve_int_sequence(
            override=temporal_absolute_indices,
            trained=temporal_image_config.get("absolute_indices", None),
            fallback=(0,),
        )
        self.temporal_anchor_deltas = self._resolve_int_sequence(
            override=temporal_anchor_deltas,
            trained=temporal_image_config.get("delta_indices", None),
            fallback=(-30, -15, 0),
        )
        validate_temporal_image_profile(
            mode=self.memory_injection_mode,
            absolute_indices=self.temporal_absolute_indices,
            delta_indices=self.temporal_anchor_deltas,
            source="RoboTwin-Mem eval temporal image config",
        )
        if temporal_view_names is None:
            temporal_view_names = temporal_image_config.get("view_names", None)
        self.temporal_view_names = tuple(str(name) for name in (temporal_view_names or ("head", "left_wrist", "right_wrist")))
        negative_delta_history = abs(min([0, *self.temporal_anchor_deltas])) + 1
        absolute_history = max([0, *[int(idx) for idx in self.temporal_absolute_indices]]) + 1
        self.temporal_history_capacity = max(int(self.horizon or 0), negative_delta_history, absolute_history)
        self.image_history = deque(maxlen=self.temporal_history_capacity)
        self.initial_images = None
        self.initial_step = None
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None
        self.num_image_history = 0

        self.state_norm_stats = self.get_state_stats(self.unnorm_key, policy_ckpt_path=policy_ckpt_path)
        self.raw_actions = None
        self.plan_start_step = None
        self.pending_commit_plan_start_step = None
        self.pending_commit_offset = None
        self.pending_commit_confidence = 0.0
        if keyframe_commit_confidence_threshold is None:
            keyframe_commit_confidence_threshold = self.memory_config.get("event_commit_threshold", 0.55)
        if keyframe_cluster_timestep_window is None:
            keyframe_cluster_timestep_window = self.memory_config.get("keyframe_cluster_timestep_window", 20)
        self.pending_commit_confidence_threshold = float(keyframe_commit_confidence_threshold)
        self.keyframe_cluster_timestep_window = int(keyframe_cluster_timestep_window)
        self.model_side_event_filter_enabled = True
        self.last_committed_keyframe_step = None
        self.pending_commit_done = True
        self.committed_keyframe_count = 0
        self.keyframe_image_memory = []
        self.first_chunk_random_replan_step = None
        self.first_chunk_random_replan_done = False
        print(
            f"*** policy_setup: {policy_setup}, unnorm_key: {self.unnorm_key}, "
            f"action_mode: {action_mode}, action_chunk_size: {self.action_chunk_size}, "
            f"sampling_interval: {self.sampling_interval}, image_size: {self.image_size}, "
            f"temporal_abs: {self.temporal_absolute_indices}, temporal_delta: {self.temporal_anchor_deltas}, "
            f"memory_ablation_mode: {self.memory_ablation_mode}, "
            f"keyframe_image_mode: {self.memory_injection_mode}, "
            f"uses_keyframe_images: {self.uses_keyframe_image_inputs}, "
            f"supports_runtime_memory_commits: {self.supports_runtime_memory_commits}, "
            f"model_side_event_filter: {self.model_side_event_filter_enabled}, "
            f"no_replan_eval: True ***"
        )
        ckpt_name = Path(policy_ckpt_path).stem
        run_tag = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.progress_output_dir = Path(__file__).resolve().parent / "progress_curves" / f"{ckpt_name}_{run_tag}"
        self.progress_output_dir.mkdir(parents=True, exist_ok=True)
        self.keyframe_output_dir = Path(__file__).resolve().parent / "keyframe" / f"{ckpt_name}_{run_tag}"
        self.keyframe_output_dir.mkdir(parents=True, exist_ok=True)
        self.current_episode_keyframe_dir: Optional[Path] = None
        self.keyframe_episode_index = 0
        self.progress_episode_index = 0
        self.progress_records = []
        atexit.register(self._flush_progress_episode)

    def _connect_client(self) -> WebsocketClientPolicy:
        if self.client is not None:
            self.client.close()
        self.client = WebsocketClientPolicy(self.host, self.port)
        return self.client

    def _ensure_client(self) -> WebsocketClientPolicy:
        if self.client is None:
            return self._connect_client()
        return self.client

    def _call_server_with_reconnect(self, method_name: str, *args):
        client = self._ensure_client()
        try:
            return getattr(client, method_name)(*args)
        except (ConnectionClosed, OSError) as e:
            logging.warning("Websocket %s failed; reconnecting once: %s", method_name, e)
            client = self._connect_client()
            return getattr(client, method_name)(*args)

    def reset(self, task_description: str) -> None:
        self._flush_progress_episode()
        self.task_description = task_description
        self.image_history.clear()
        self.initial_images = None
        self.initial_step = None
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0
        self.raw_actions = None
        self.plan_start_step = None
        self.pending_commit_plan_start_step = None
        self.pending_commit_offset = None
        self.pending_commit_confidence = 0.0
        self.last_committed_keyframe_step = None
        self.pending_commit_done = True
        self.committed_keyframe_count = 0
        self.keyframe_image_memory.clear()
        self.first_chunk_random_replan_step = None
        self.first_chunk_random_replan_done = False
        self.current_episode_keyframe_dir = None
        # Reset state tracking for delta/rel modes
        self.initial_state = None
        self.prev_action = None
        # Reset server-side memory buffer
        try:
            self._call_server_with_reconnect("reset_memory")
        except Exception as e:
            logging.warning(f"Failed to reset server memory: {e}")

    def step(
        self,
        example: dict,
        step: int = 0,
    ) -> np.ndarray:
        state = example.get("state", None)
        # if state is not None:
        #     state = self.normalize_state(state, self.state_norm_stats)
        #     state = state[[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13]]
        #     example["state"] = state.reshape(1, -1)

        # Store initial state for delta/rel modes
        if self.action_mode in ["delta", "rel"] and self.initial_state is None:
            if state is None:
                raise ValueError(f"action_mode='{self.action_mode}' requires state to be provided in example")
            self.initial_state = np.array(state).copy()

        task_description = example.get("lang", None)
        images = example["image"]

        if example is not None:
            if task_description != self.task_description:
                self.reset(task_description)
                # Re-store initial state after reset if in delta/rel mode
                if self.action_mode in ["delta", "rel"] and state is not None:
                    self.initial_state = np.array(state).copy()

        raw_images = [np.asarray(image).copy() for image in images]
        images = [self._resize_image(image) for image in images]
        self._record_temporal_images(images=images, step=int(step))
        temporal_images, image_metas = self._build_temporal_anchor_images(step=int(step))
        example["image"] = temporal_images
        example["image_metas"] = image_metas
        example_copy = example.copy()
        example_copy.pop("state", None)
        example_copy["timestep"] = int(step)

        # Commit a matured delayed keyframe write before any fresh planning so the
        # current observation can influence the next raw-keyframe image plan.
        replan_after_keyframe_commit = self._maybe_commit_pending_event(
            example_copy=example_copy,
            step=int(step),
            raw_images=raw_images,
        )
        self._inject_keyframe_memory_fields(example_copy, step=int(step))
        vla_input = {
            "examples": [example_copy],
            "do_sample": False,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
        }

        action_chunk_size = self.action_chunk_size
        action_idx = None if self.plan_start_step is None else step - self.plan_start_step
        replan_at_random_second_anchor = self._should_trigger_first_chunk_random_replan(step=int(step))
        needs_replan = (
            self.raw_actions is None
            or self.plan_start_step is None
            or action_idx is None
            or action_idx < 0
            or action_idx >= len(self.raw_actions)
            or replan_at_random_second_anchor
            or replan_after_keyframe_commit
        )

        if needs_replan:
            is_first_plan = self.raw_actions is None or self.plan_start_step is None
            if not is_first_plan:
                self._consume_first_chunk_random_replan()
            self.plan_start_step = int(step)
            response = self._call_server_with_reconnect("predict_action", vla_input)
            self._raise_for_server_error(response)
            try:
                normalized_actions = response["data"]["normalized_actions"]  # B, chunk, D
            except KeyError:
                print(f"Response data: {response}")
                response_data = response.get("data", {}) if isinstance(response, dict) else {}
                raise KeyError(f"Key 'normalized_actions' not found in response data: {response_data.keys()}")

            normalized_actions = normalized_actions[0]
            response_data = response.get("data", {})
            progress_pred = self._extract_progress_prediction(response_data)
            if progress_pred is not None:
                self.progress_records.append(
                    {
                        "env_step": int(step),
                        "progress_pred": float(progress_pred),
                    }
                )
            self._update_pending_commit_from_response(response_data=response_data, step=int(step))
            # Unnormalize to get delta/rel values
            raw_actions = self.unnormalize_actions(
                normalized_actions=normalized_actions, action_norm_stats=self.action_norm_stats
            )

            # Convert delta/rel to absolute actions
            if self.action_mode == "delta":
                self.raw_actions = self._delta_to_absolute(raw_actions, state)
            elif self.action_mode == "rel":
                self.raw_actions = self._rel_to_absolute(raw_actions)
            else:
                self.raw_actions = raw_actions

            if is_first_plan:
                self._arm_first_chunk_random_replan(plan_start_step=int(step))

        action_idx = step - self.plan_start_step
        if action_idx >= len(self.raw_actions):
            raise IndexError(
                f"Action index {action_idx} out of range for cached chunk of length {len(self.raw_actions)}"
            )
        current_action = self.raw_actions[action_idx]

        # Update prev_action for delta mode (for cross-chunk continuity)
        if self.action_mode == "delta":
            self.prev_action = current_action.copy()

        current_action = current_action[[0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13]]
        return current_action

    @staticmethod
    def _clone_images(images: Sequence[np.ndarray]) -> list[np.ndarray]:
        return [np.asarray(image).copy() for image in images]

    def _record_temporal_images(self, images: Sequence[np.ndarray], step: int) -> None:
        cloned = self._clone_images(images)
        if self.initial_images is None:
            self.initial_images = self._clone_images(cloned)
            self.initial_step = int(step)

        record = {"step": int(step), "images": cloned}
        if self.image_history and int(self.image_history[-1]["step"]) == int(step):
            self.image_history[-1] = record
        else:
            self.image_history.append(record)
        self.num_image_history = len(self.image_history)

    def _images_at_or_before(self, target_step: int) -> list[np.ndarray]:
        if not self.image_history:
            if self.initial_images is None:
                raise RuntimeError("temporal image history is empty")
            return self._clone_images(self.initial_images)

        selected = None
        for record in self.image_history:
            if int(record["step"]) <= int(target_step):
                selected = record
            else:
                break

        if selected is None:
            if self.initial_images is not None:
                return self._clone_images(self.initial_images)
            selected = self.image_history[0]
        return self._clone_images(selected["images"])

    def _build_temporal_anchor_images(self, step: int) -> tuple[list[np.ndarray], list[dict]]:
        if self.initial_images is None:
            raise RuntimeError("initial images are not recorded")

        frame_specs = []
        initial_step = int(self.initial_step or 0)
        for absolute_index in self.temporal_absolute_indices:
            absolute_index = int(absolute_index)
            if absolute_index == 0:
                images = self._clone_images(self.initial_images)
                time_role = "first"
            else:
                images = self._images_at_or_before(initial_step + absolute_index)
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
                    "images": self._images_at_or_before(int(step) + int(delta)),
                    "time_role": role,
                    "delta": int(delta),
                    "absolute_index": None,
                }
            )

        anchor_images: list[np.ndarray] = []
        image_metas: list[dict] = []
        for frame_idx, frame_spec in enumerate(frame_specs):
            frame_images = frame_spec["images"]
            for view_idx, image in enumerate(frame_images):
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

    def _use_keyframe_image_inputs(self) -> bool:
        return self.memory_injection_mode in self._KEYFRAME_IMAGE_INPUT_MODES

    def _select_main_view_for_keyframe_memory(self, images: Sequence[np.ndarray]) -> int:
        if len(images) <= 0:
            raise RuntimeError("keyframe image memory requires at least one raw image")
        include_names = tuple(
            str(name).lower()
            for name in self.qwen_memory_injection_config.get(
                "include_names",
                self.memory_config.get("memory_views", {}).get("include_names", ["cam_high", "head", "main"]),
            )
        )
        exclude_patterns = tuple(
            str(name).lower()
            for name in self.qwen_memory_injection_config.get(
                "exclude_name_patterns",
                self.memory_config.get("memory_views", {}).get("exclude_name_patterns", ["wrist"]),
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
        raw_images: Optional[Sequence[np.ndarray]],
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
        if self.keyframe_image_memory and int(self.keyframe_image_memory[-1]["step"]) == int(step):
            self.keyframe_image_memory[-1] = entry
        else:
            self.keyframe_image_memory.append(entry)
        if self.max_keyframe_images > 0 and len(self.keyframe_image_memory) > int(self.max_keyframe_images):
            self.keyframe_image_memory = self.keyframe_image_memory[-int(self.max_keyframe_images):]

    def _build_keyframe_memory_images(self, step: int) -> tuple[list[np.ndarray], list[dict], list[int]]:
        if not self._use_keyframe_image_inputs():
            return [], [], []

        visible_entries = [
            entry
            for entry in self.keyframe_image_memory
            if int(entry.get("step", -1)) <= int(step)
        ]
        max_keyframes = max(0, int(self.max_keyframe_images))
        if max_keyframes > 0:
            visible_entries = visible_entries[-max_keyframes:]
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

    def _inject_keyframe_memory_fields(self, example: dict, step: int) -> None:
        memory_images, memory_metas, memory_steps = self._build_keyframe_memory_images(step=int(step))
        example["memory_keyframe_images"] = memory_images
        example["memory_keyframe_image_metas"] = memory_metas
        example["memory_keyframe_steps"] = memory_steps
        example["memory_keyframe_count"] = int(len(memory_images))

    def _clear_pending_commit(self) -> None:
        self.pending_commit_plan_start_step = None
        self.pending_commit_offset = None
        self.pending_commit_confidence = 0.0
        self.pending_commit_done = True

    def _arm_first_chunk_random_replan(self, plan_start_step: int) -> None:
        if (not self.first_chunk_random_replan) or self.first_chunk_random_replan_done:
            return
        if self.first_chunk_fixed_replan_step is not None:
            fixed_offset = int(self.first_chunk_fixed_replan_step)
            self.first_chunk_random_replan_step = int(plan_start_step) + fixed_offset
            logging.info(
                "Armed first-chunk fixed replan: plan_start=%d offset=%d target_step=%d",
                int(plan_start_step),
                int(fixed_offset),
                int(self.first_chunk_random_replan_step),
            )
            return
        upper = max(1, min(int(self.sampling_interval), int(self.action_chunk_size)))
        random_offset = int(self.first_chunk_random_replan_rng.randint(1, upper))
        self.first_chunk_random_replan_step = int(plan_start_step) + random_offset
        logging.info(
            "Armed first-chunk random replan: plan_start=%d offset=%d target_step=%d",
            int(plan_start_step),
            int(random_offset),
            int(self.first_chunk_random_replan_step),
        )

    def _consume_first_chunk_random_replan(self) -> None:
        self.first_chunk_random_replan_step = None
        self.first_chunk_random_replan_done = True

    def _should_trigger_first_chunk_random_replan(self, step: int) -> bool:
        if (not self.first_chunk_random_replan) or self.first_chunk_random_replan_done:
            return False
        if self.first_chunk_random_replan_step is None:
            return False
        return int(step) >= int(self.first_chunk_random_replan_step)

    def _pending_commit_step(self) -> Optional[int]:
        if self.pending_commit_plan_start_step is None or self.pending_commit_offset is None:
            return None
        return int(self.pending_commit_plan_start_step) + int(self.pending_commit_offset)

    def _is_same_keyframe_cluster(self, step_a: Optional[int], step_b: Optional[int]) -> bool:
        if step_a is None or step_b is None:
            return False
        return abs(int(step_a) - int(step_b)) <= int(self.keyframe_cluster_timestep_window)

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

    def _update_pending_commit_from_response(self, response_data: dict, step: int) -> None:
        if not self.supports_runtime_memory_commits:
            self._clear_pending_commit()
            return

        should_trigger = bool(self._extract_scalar_response_value(response_data, "should_trigger_event", False, bool))
        if not should_trigger:
            return

        event_offset = self._extract_scalar_response_value(response_data, "pred_event_offset", -1, int)
        if event_offset is None or int(event_offset) < 0:
            return

        candidate_plan_start_step = int(step)
        candidate_offset = int(event_offset)
        candidate_confidence = float(
            self._extract_scalar_response_value(response_data, "pred_event_confidence", 0.0, float)
        )

        self.pending_commit_plan_start_step = candidate_plan_start_step
        self.pending_commit_offset = candidate_offset
        self.pending_commit_confidence = candidate_confidence
        self.pending_commit_done = False

    def _maybe_commit_pending_event(
        self,
        example_copy: dict,
        step: int,
        raw_images: Optional[Sequence[np.ndarray]] = None,
    ) -> bool:
        if not self.supports_runtime_memory_commits:
            self._clear_pending_commit()
            return False

        if self.pending_commit_done:
            return False
        if self.pending_commit_plan_start_step is None or self.pending_commit_offset is None:
            return False

        local_step = int(step) - int(self.pending_commit_plan_start_step)
        if local_step != int(self.pending_commit_offset):
            return False

        commit_step = int(step)
        committed = False
        try:
            self._commit_keyframe_image_memory(
                raw_images=raw_images,
                step=commit_step,
                confidence=self.pending_commit_confidence,
            )
            committed = True
        finally:
            self.pending_commit_done = True

        if not committed:
            return False

        self.committed_keyframe_count += 1
        self.last_committed_keyframe_step = commit_step
        self._save_keyframe_images(
            images=raw_images,
            step=commit_step,
            confidence=self.pending_commit_confidence,
            commit_index=self.committed_keyframe_count,
        )
        # Replan immediately after the first three committed keyframes so the
        # new memory can affect the next action chunk. The fourth keyframe is the
        # last inspect anchor, so let the current chunk finish and replan at the
        # next natural chunk boundary instead of interrupting mid-retreat.
        return self.replan_after_keyframe_commit and self.committed_keyframe_count < 4

    @staticmethod
    def normalize_state(state: dict[str, np.ndarray], state_norm_stats: Dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Normalize the state
        """
        mask = [True, True, True, True, True, True, True, True, True, True, True, True, False, False]
        mask = np.array(mask, dtype=bool)
        state_high, state_low = np.array(state_norm_stats["max"]), np.array(state_norm_stats["min"])
        normalized_state = np.where(
            mask,
            (state - state_low) / (state_high - state_low) * 2 - 1,
            state,
        )
        normalized_state = np.where(~mask, (normalized_state > 0.5).astype(normalized_state.dtype), normalized_state)
        return normalized_state

    @staticmethod
    def unnormalize_actions(normalized_actions: np.ndarray, action_norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])
        normalized_actions = np.clip(normalized_actions, -1, 1)

        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        return actions

    def _delta_to_absolute(self, delta_actions: np.ndarray, current_state: np.ndarray) -> np.ndarray:
        """
        Convert delta actions to absolute actions.

        Training: delta[0] = a[0] - s[0], delta[t] = a[t] - a[t-1]
        Deployment: a[0] = delta[0] + base, a[t] = delta[t] + a[t-1]

        Where base is:
        - First chunk: initial_state (s_0)
        - Subsequent chunks: prev_action (last action from previous chunk)
        """
        abs_actions = np.zeros_like(delta_actions)
        mask = self.action_norm_stats.get("mask", np.ones(delta_actions.shape[-1], dtype=bool))

        # Determine base action
        base = self.prev_action if self.prev_action is not None else self.initial_state

        for i in range(len(delta_actions)):
            abs_actions[i] = np.where(mask, delta_actions[i] + base, delta_actions[i])
            base = abs_actions[i]

        return abs_actions

    def _rel_to_absolute(self, rel_actions: np.ndarray) -> np.ndarray:
        """
        Convert relative actions to absolute actions.

        Training: rel[t] = a[t] - s[0]
        Deployment: a[t] = rel[t] + s[0]
        """
        abs_actions = np.zeros_like(rel_actions)
        mask = self.action_norm_stats.get("mask", np.ones(rel_actions.shape[-1], dtype=bool))

        for i in range(len(rel_actions)):
            abs_actions[i] = np.where(mask, rel_actions[i] + self.initial_state, rel_actions[i])

        return abs_actions

    @staticmethod
    def get_action_stats(unnorm_key: str, policy_ckpt_path, action_mode: str = "abs") -> dict:
        policy_ckpt_path = Path(policy_ckpt_path)
        model_config, norm_stats = read_mode_config(policy_ckpt_path)
        unnorm_key = ModelClient._check_unnorm_key(norm_stats, unnorm_key)

        stats = norm_stats[unnorm_key]

        # Support two formats:
        # New format: {"robotwin": {"abs": {...}, "delta": {...}, "rel": {...}}}
        # Old format: {"robotwin": {"action": {...}, "state": {...}}}

        if action_mode in stats:
            # New format: directly use the corresponding mode stats
            mode_stats = stats[action_mode]
            return mode_stats.get("action", mode_stats)
        elif "action" in stats:
            # Old format: only supports abs mode
            if action_mode != "abs":
                print(f"[WARNING] Statistics file only has abs mode, but {action_mode} was requested. Using abs stats.")
            return stats["action"]
        else:
            raise ValueError(f"Invalid statistics file format for key: {unnorm_key}")

    @staticmethod
    def get_state_stats(unnorm_key: str, policy_ckpt_path) -> dict:
        policy_ckpt_path = Path(policy_ckpt_path)
        model_config, norm_stats = read_mode_config(policy_ckpt_path)
        unnorm_key = ModelClient._check_unnorm_key(norm_stats, unnorm_key)
        return norm_stats[unnorm_key]["state"]

    @staticmethod
    def get_action_chunk_size(policy_ckpt_path):
        model_config, _ = read_mode_config(policy_ckpt_path)
        action_model_config = model_config["framework"]["action_model"]
        future_window = int(action_model_config.get("future_action_window_size", 0))
        past_window = int(action_model_config.get("past_action_window_size", 0))
        return past_window + 1 + future_window

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image_array = np.asarray(image)
        if image_array.dtype != np.uint8:
            image_array = np.clip(image_array, 0, 255).astype(np.uint8)
        pil_image = Image.fromarray(image_array)
        if pil_image.mode not in ("RGB", "L"):
            pil_image = pil_image.convert("RGB")
        return np.asarray(pil_image.resize(tuple(self.image_size)))

    @staticmethod
    def _raise_for_server_error(response: Any) -> None:
        if not isinstance(response, dict) or response.get("ok", True):
            return
        error = response.get("error", {})
        message = error.get("message", response) if isinstance(error, dict) else error
        raise RuntimeError(f"Policy server returned error: {message}")

    @staticmethod
    def _nested_get(mapping: dict, keys: Sequence[str], default=None):
        current = mapping
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                return default
            current = current[key]
        return current

    @staticmethod
    def _resolve_image_size(override: Optional[Sequence[int]], trained: Optional[Sequence[int]]) -> list[int]:
        value = override if override is not None else trained
        if value is None:
            value = (224, 224)
        value = list(value)
        if len(value) < 2:
            raise ValueError(f"image_size must contain width and height, got {value}")
        return [int(value[0]), int(value[1])]

    @staticmethod
    def _resolve_int_sequence(
        override: Optional[Sequence[int]],
        trained: Optional[Sequence[int]],
        fallback: Sequence[int],
    ) -> tuple[int, ...]:
        value = override if override is not None else trained
        if value is None:
            value = fallback
        return tuple(int(item) for item in value)

    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        if unnorm_key is None:
            if len(norm_stats) == 1:
                unnorm_key = next(iter(norm_stats.keys()))
            else:
                unnorm_key = next(iter(norm_stats.keys()))

        if unnorm_key not in norm_stats:
            fallback_key = next(iter(norm_stats.keys()))
            logging.warning(
                "unnorm_key=%s is not present in dataset_statistics.json; using %s instead.",
                unnorm_key,
                fallback_key,
            )
            unnorm_key = fallback_key

        return unnorm_key

    @staticmethod
    def _sanitize_name(name: Optional[str]) -> str:
        if not name:
            return "unknown_task"
        sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name).strip())
        return sanitized.strip("._-") or "unknown_task"

    @staticmethod
    def _extract_progress_prediction(response_data: Dict[str, Any]) -> Optional[float]:
        progress = response_data.get("progress", None)
        if progress is None:
            return None
        progress_array = np.asarray(progress, dtype=np.float32).reshape(-1)
        if progress_array.size == 0:
            return None
        return float(progress_array[0])

    def _ensure_episode_keyframe_dir(self) -> Path:
        if self.current_episode_keyframe_dir is None:
            # 1. 正常清理文件名中的特殊字符
            task_name = self._sanitize_name(self.task_description)
            
            # 2. 强制截断长度，例如限制在前 64 个字符
            task_name_short = task_name[:10]
            
            # 3. 拼接短命名
            episode_dir_name = f"episode_{self.keyframe_episode_index:04d}_{task_name_short}"
            
            self.current_episode_keyframe_dir = self.keyframe_output_dir / episode_dir_name
            self.current_episode_keyframe_dir.mkdir(parents=True, exist_ok=True)
            self.keyframe_episode_index += 1
            
        return self.current_episode_keyframe_dir

    @staticmethod
    def _prepare_image_for_save(image: np.ndarray) -> np.ndarray:
        image_array = np.asarray(image)
        if image_array.ndim == 2:
            image_array = np.repeat(image_array[..., None], 3, axis=2)
        elif image_array.ndim == 3 and image_array.shape[2] == 1:
            image_array = np.repeat(image_array, 3, axis=2)
        if image_array.dtype != np.uint8:
            image_array = np.clip(image_array, 0, 255).astype(np.uint8)
        if image_array.ndim == 3 and image_array.shape[2] == 3:
            # RoboTwin-Mem observations are RGB; OpenCV writes BGR images.
            image_array = cv.cvtColor(image_array, cv.COLOR_RGB2BGR)
        return image_array

    def _save_keyframe_images(
        self,
        images: Optional[Sequence[np.ndarray]],
        step: int,
        confidence: float,
        commit_index: int,
    ) -> None:
        if not images:
            return

        prepared_images = [self._prepare_image_for_save(image) for image in images]
        target_height = max(image.shape[0] for image in prepared_images)
        labeled_tiles = []
        view_names = ("head", "left", "right")
        for idx, image in enumerate(prepared_images):
            if image.shape[0] != target_height:
                target_width = max(1, int(round(image.shape[1] * (target_height / max(1, image.shape[0])))))
                image = cv.resize(image, (target_width, target_height), interpolation=cv.INTER_AREA)
            canvas = image.copy()
            view_name = view_names[idx] if idx < len(view_names) else f"view_{idx}"
            cv.putText(
                canvas,
                view_name,
                (12, 28),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv.LINE_AA,
            )
            labeled_tiles.append(canvas)

        keyframe_canvas = np.concatenate(labeled_tiles, axis=1)
        episode_dir = self._ensure_episode_keyframe_dir()
        filename = (
            f"keyframe_{int(commit_index):02d}_step_{int(step):04d}_"
            f"conf_{float(confidence):.4f}.png"
        )
        output_path = episode_dir / filename
        cv.imwrite(str(output_path), keyframe_canvas)
        logging.info("Saved keyframe image to %s", output_path)

    def _flush_progress_episode(self) -> None:
        if len(self.progress_records) == 0:
            return

        task_name = self._sanitize_name(self.task_description)
        episode_prefix = f"episode_{self.progress_episode_index:04d}_{task_name}"
        json_path = self.progress_output_dir / f"{episode_prefix}.json"
        png_path = self.progress_output_dir / f"{episode_prefix}.png"

        payload = {
            "episode_index": int(self.progress_episode_index),
            "task_description": self.task_description,
            "records": self.progress_records,
        }
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

        self._save_progress_plot(png_path=png_path, records=self.progress_records, task_name=task_name)
        logging.info("Saved progress curve to %s", png_path)

        self.progress_records = []
        self.progress_episode_index += 1

    def _save_progress_plot(self, png_path: Path, records: Sequence[Dict[str, Any]], task_name: str) -> None:
        width, height = 960, 540
        margin_left, margin_right = 80, 40
        margin_top, margin_bottom = 60, 70
        canvas = np.full((height, width, 3), 255, dtype=np.uint8)
        plot_x0, plot_y0 = margin_left, margin_top
        plot_x1, plot_y1 = width - margin_right, height - margin_bottom
        plot_w = max(1, plot_x1 - plot_x0)
        plot_h = max(1, plot_y1 - plot_y0)

        xs = [int(item["env_step"]) for item in records]
        ys = [float(item["progress_pred"]) for item in records]
        x_min = min(xs) if xs else 0
        x_max = max(xs) if xs else 1
        if x_max <= x_min:
            x_max = x_min + 1
        y_min = min(0.0, min(ys) if ys else 0.0)
        y_max = max(1.0, max(ys) if ys else 1.0)
        if y_max <= y_min:
            y_max = y_min + 1.0

        cv.rectangle(canvas, (plot_x0, plot_y0), (plot_x1, plot_y1), (220, 220, 220), 1)
        cv.line(canvas, (plot_x0, plot_y1), (plot_x1, plot_y1), (0, 0, 0), 2)
        cv.line(canvas, (plot_x0, plot_y0), (plot_x0, plot_y1), (0, 0, 0), 2)

        for y_tick in np.linspace(y_min, y_max, num=5):
            y_px = plot_y1 - int(round((y_tick - y_min) / (y_max - y_min) * plot_h))
            cv.line(canvas, (plot_x0, y_px), (plot_x1, y_px), (235, 235, 235), 1)
            cv.putText(
                canvas,
                f"{y_tick:.2f}",
                (8, y_px + 5),
                cv.FONT_HERSHEY_SIMPLEX,
                0.45,
                (80, 80, 80),
                1,
                cv.LINE_AA,
            )

        for x_tick in np.linspace(x_min, x_max, num=5):
            x_px = plot_x0 + int(round((x_tick - x_min) / (x_max - x_min) * plot_w))
            cv.line(canvas, (x_px, plot_y0), (x_px, plot_y1), (240, 240, 240), 1)
            cv.putText(
                canvas,
                f"{int(round(x_tick))}",
                (x_px - 15, plot_y1 + 25),
                cv.FONT_HERSHEY_SIMPLEX,
                0.45,
                (80, 80, 80),
                1,
                cv.LINE_AA,
            )

        points = []
        for env_step, progress_pred in zip(xs, ys):
            x_px = plot_x0 + int(round((env_step - x_min) / (x_max - x_min) * plot_w))
            y_px = plot_y1 - int(round((progress_pred - y_min) / (y_max - y_min) * plot_h))
            points.append((x_px, y_px))

        if len(points) >= 2:
            cv.polylines(canvas, [np.array(points, dtype=np.int32)], False, (40, 110, 220), 2, cv.LINE_AA)
        for point in points:
            cv.circle(canvas, point, 4, (20, 70, 180), -1, cv.LINE_AA)

        title = f"Progress Curve | {task_name} | episode {self.progress_episode_index:04d}"
        summary = (
            f"points={len(records)}  step_range=[{xs[0]}, {xs[-1]}]  "
            f"progress_range=[{min(ys):.3f}, {max(ys):.3f}]"
        )
        cv.putText(canvas, title, (plot_x0, 28), cv.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv.LINE_AA)
        cv.putText(canvas, summary, (plot_x0, 50), cv.FONT_HERSHEY_SIMPLEX, 0.48, (70, 70, 70), 1, cv.LINE_AA)
        cv.putText(
            canvas,
            "env_step",
            (plot_x0 + plot_w // 2 - 30, height - 20),
            cv.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            1,
            cv.LINE_AA,
        )
        cv.putText(canvas, "progress_pred", (10, plot_y0 - 20), cv.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv.LINE_AA)
        cv.imwrite(str(png_path), canvas)


def get_model(usr_args):
    policy_ckpt_path = usr_args.get("policy_ckpt_path")
    host = usr_args.get("host", "127.0.0.1")
    port = usr_args.get("port", 5694)
    unnorm_key = usr_args.get("unnorm_key", None)
    action_mode = usr_args.get("action_mode", "abs")

    if policy_ckpt_path is None:
        raise ValueError("policy_ckpt_path must be provided in config")

    return ModelClient(
        policy_ckpt_path=policy_ckpt_path,
        host=host,
        port=port,
        unnorm_key=unnorm_key,
        action_mode=action_mode,
        first_chunk_random_replan=usr_args.get("first_chunk_random_replan", True),
        first_chunk_random_replan_seed=usr_args.get("first_chunk_random_replan_seed", 42),
        first_chunk_fixed_replan_step=usr_args.get("first_chunk_fixed_replan_step", None),
        sampling_interval=usr_args.get("sampling_interval", None),
        replan_after_keyframe_commit=usr_args.get(
            "replan_after_keyframe_commit",
            usr_args.get("replan_after_fourth_keyframe_commit", True),
        ),
        image_size=usr_args.get("image_size", None),
        temporal_absolute_indices=usr_args.get("temporal_absolute_indices", None),
        temporal_anchor_deltas=usr_args.get("temporal_anchor_deltas", None),
        temporal_view_names=usr_args.get("temporal_view_names", None),
        keyframe_commit_confidence_threshold=usr_args.get("keyframe_commit_confidence_threshold", None),
        keyframe_cluster_timestep_window=usr_args.get("keyframe_cluster_timestep_window", None),
    )


def reset_model(model):
    model.reset(task_description="")


def eval(TASK_ENV, model, observation):
    # Environment reset implies a new episode -> reset memory.
    if getattr(TASK_ENV, "take_action_cnt", 0) == 0:
        model.reset(task_description="")

    # Get instruction
    instruction = TASK_ENV.get_instruction()

    # Prepare images
    head_img = observation["observation"]["head_camera"]["rgb"]
    left_img = observation["observation"]["left_camera"]["rgb"]
    right_img = observation["observation"]["right_camera"]["rgb"]

    # Order: [head, left, right] to match training order
    images = [head_img, left_img, right_img]

    state = observation["joint_action"]["vector"]
    example = {
        "lang": str(instruction),
        "image": images,
        "state": state,  # Required for delta/rel action modes
    }

    action = model.step(example, step=TASK_ENV.take_action_cnt)

    # Execute action
    TASK_ENV.take_action(action)
