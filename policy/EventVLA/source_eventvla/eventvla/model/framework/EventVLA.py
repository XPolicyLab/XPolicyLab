# Copyright 2025 eventvla community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
EventVLA Framework

A lightweight implementation that uses an action special token to parallelly predict continuous actions
conditioned on temporal anchor images, raw keyframe image memory, and a language instruction.
"""

from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from deployment.model_server.tools.image_tools import to_pil_preserve
from eventvla.model.framework.base_framework import baseframework
from eventvla.model.memory_ablation import (
    CANONICAL_MEMORY_ABLATION_MODES,
    KEYFRAME_IMAGE_MEMORY_MODES,
    MEMORY_WRITE_POLICY_DISABLED,
    validate_temporal_image_profile,
)
from eventvla.model.modules.action_model.MLP_ActionHeader import get_action_model
from eventvla.model.modules.vlm import get_vlm_model
from eventvla.model.tools import FRAMEWORK_REGISTRY
from eventvla.training.trainer_utils import initialize_overwatch
from eventvla.training.trainer_utils.trainer_tools import resize_images

logger = initialize_overwatch(__name__)

IGNORE_INDEX = -100


@FRAMEWORK_REGISTRY.register("EventVLA")
class EventVLA(baseframework):
    """Multimodal vision-language-action model."""

    _PURE_KEYFRAME_IMAGE_MODES = set(KEYFRAME_IMAGE_MEMORY_MODES)
    _SUPPORTED_MEMORY_INJECTION_MODES = set(CANONICAL_MEMORY_ABLATION_MODES)

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = config
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        config.framework.action_model.action_hidden_dim = self.qwen_vl_interface.model.config.hidden_size
        self.action_model = get_action_model(config=self.config)

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size

        self.action_token = "🔍"
        self.action_token_id = self.qwen_vl_interface.processor.tokenizer(
            self.action_token,
            add_special_tokens=False,
        )["input_ids"][0]

        framework_cfg = getattr(config, "framework", None)
        datasets_cfg = getattr(config, "datasets", None)
        vla_data_cfg = getattr(datasets_cfg, "vla_data", None)
        memory_cfg = getattr(config.framework, "memory_buffer", None)

        def _memory_cfg_value(name: str, default):
            if memory_cfg is None:
                return default
            if hasattr(memory_cfg, "get"):
                return memory_cfg.get(name, default)
            return getattr(memory_cfg, name, default)

        def _cfg_value(container, name: str, default):
            if container is None:
                return default
            if hasattr(container, "get"):
                return container.get(name, default)
            return getattr(container, name, default)

        def _cfg_bool(value, default: bool = False) -> bool:
            if value is None:
                return bool(default)
            if isinstance(value, str):
                return value.strip().lower() not in {"", "0", "false", "no", "off", "none", "null"}
            return bool(value)

        injection_cfg = _memory_cfg_value("qwen_memory_injection", {})
        self.memory_ablation_mode = str(
            _cfg_value(framework_cfg, "memory_ablation_mode", "")
        ).strip().lower()
        if self.memory_ablation_mode not in self._SUPPORTED_MEMORY_INJECTION_MODES:
            supported_modes = ", ".join(sorted(self._SUPPORTED_MEMORY_INJECTION_MODES))
            raise ValueError(
                f"Unsupported framework.memory_ablation_mode `{self.memory_ablation_mode}`. "
                f"EventVLA only supports: {supported_modes}."
            )
        self.memory_injection_enabled = _cfg_bool(
            _cfg_value(injection_cfg, "enabled", _memory_cfg_value("qwen_memory_injection_enabled", True))
        )
        if not self.memory_injection_enabled:
            raise ValueError(
                "EventVLA requires raw image injection to be enabled."
            )
        self.memory_injection_mode = str(
            _cfg_value(
                injection_cfg,
                "mode",
                "",
            )
        ).lower()
        if self.memory_injection_mode not in self._SUPPORTED_MEMORY_INJECTION_MODES:
            supported_modes = ", ".join(sorted(self._SUPPORTED_MEMORY_INJECTION_MODES))
            raise ValueError(
                f"Unsupported qwen_memory_injection.mode `{self.memory_injection_mode}`. "
                f"EventVLA only supports: {supported_modes}."
            )
        if self.memory_injection_mode != self.memory_ablation_mode:
            raise ValueError(
                "Resolved config mismatch: "
                f"framework.memory_ablation_mode={self.memory_ablation_mode} but "
                f"configured keyframe image mode={self.memory_injection_mode}."
            )
        self.max_keyframe_images = int(
            _cfg_value(injection_cfg, "max_keyframe_images", _memory_cfg_value("max_keyframe_images", 4))
        )
        self.keyframe_image_position = str(
            _cfg_value(
                injection_cfg,
                "keyframe_image_position",
                _memory_cfg_value("keyframe_image_position", "after_anchor_images_before_action"),
            )
        ).lower()
        self.use_image_role_text = _cfg_bool(
            _cfg_value(injection_cfg, "use_image_role_text", _memory_cfg_value("use_image_role_text", True))
        )
        self.memory_write_policy = str(
            _memory_cfg_value("memory_write_policy", MEMORY_WRITE_POLICY_DISABLED)
        ).strip().lower()
        if self.memory_write_policy != MEMORY_WRITE_POLICY_DISABLED:
            raise ValueError(
                "EventVLA stores raw keyframe images outside the model; "
                "model-side memory writes must be disabled."
            )
        memory_enable = False
        if memory_cfg is not None:
            if hasattr(memory_cfg, "get"):
                memory_enable = _cfg_bool(memory_cfg.get("enable", False), False)
            else:
                memory_enable = _cfg_bool(getattr(memory_cfg, "enable", False), False)

        if memory_enable:
            raise ValueError("EventVLA supported raw-image modes require framework.memory_buffer.enable=false.")

        keyframe_image_cfg = _cfg_value(vla_data_cfg, "keyframe_image_memory", {})
        keyframe_image_memory_enabled = _cfg_bool(_cfg_value(keyframe_image_cfg, "enabled", False), False)
        if keyframe_image_memory_enabled != self._requires_keyframe_image_memory():
            raise ValueError(
                "Resolved config mismatch: "
                f"datasets.vla_data.keyframe_image_memory.enabled={keyframe_image_memory_enabled} but "
                f"mode={self.memory_injection_mode} requires_keyframe_image_memory={self._requires_keyframe_image_memory()}."
            )
        provide_teacher_commit_images = _cfg_bool(
            _cfg_value(vla_data_cfg, "provide_teacher_commit_images", False),
            False,
        )
        if provide_teacher_commit_images:
            raise ValueError(
                "EventVLA supported raw-image modes require "
                "datasets.vla_data.provide_teacher_commit_images=false."
            )
        temporal_cfg = _cfg_value(vla_data_cfg, "temporal", {}) or {}
        temporal_image_cfg = _cfg_value(temporal_cfg, "image", {}) or {}
        validate_temporal_image_profile(
            mode=self.memory_injection_mode,
            absolute_indices=_cfg_value(temporal_image_cfg, "absolute_indices", []),
            delta_indices=_cfg_value(temporal_image_cfg, "delta_indices", [0]),
            source="datasets.vla_data.temporal.image",
        )
        logger.info(
            "EventVLA mode=%s enabled: temporal anchors absolute_indices=%s, "
            "delta_indices=%s, max_keyframe_images=%s.",
            self.memory_injection_mode,
            _cfg_value(temporal_image_cfg, "absolute_indices", []),
            _cfg_value(temporal_image_cfg, "delta_indices", [0]),
            self.max_keyframe_images,
        )

        self.l1_loss = nn.L1Loss()
        hidden_dim = self.qwen_vl_interface.model.config.hidden_size

        self.keyframe_loss_weight = float(_memory_cfg_value("keyframe_loss_weight", 1.0))
        self.keyframe_positive_weight = float(_memory_cfg_value("keyframe_positive_weight", 7.0))
        self.keyframe_threshold = float(_memory_cfg_value("keyframe_threshold", 0.5))
        self.keyframe_predict_mode = str(_memory_cfg_value("keyframe_predict_mode", "chunk_future")).lower()
        raw_use_keyframe_head = _memory_cfg_value("use_keyframe_predict_head", "enabled")
        if isinstance(raw_use_keyframe_head, bool):
            self.keyframe_predict_head_mode = "enabled" if raw_use_keyframe_head else "disabled"
        else:
            self.keyframe_predict_head_mode = str(raw_use_keyframe_head).lower()
        if self.keyframe_predict_head_mode in {"true", "yes", "on", "1"}:
            self.keyframe_predict_head_mode = "enabled"
        elif self.keyframe_predict_head_mode in {"false", "no", "off", "0", "none"}:
            self.keyframe_predict_head_mode = "disabled"
        elif self.keyframe_predict_head_mode not in {"enabled", "disabled", "auto"}:
            logger.warning(
                "Unsupported use_keyframe_predict_head=%s; fallback to enabled.",
                raw_use_keyframe_head,
            )
            self.keyframe_predict_head_mode = "enabled"
        self.event_future_min_offset = max(0, int(_memory_cfg_value("event_future_min_offset", 1)))
        self.event_commit_threshold = float(_memory_cfg_value("event_commit_threshold", 0.55))
        cluster_window_default = _memory_cfg_value("keyframe_cluster_timestep_window", 20)
        self.enable_keyframe_inference_event_filter = _cfg_bool(
            _memory_cfg_value("enable_keyframe_inference_event_filter", True),
            True,
        )
        self.keyframe_nms_window = max(
            0,
            int(
                _memory_cfg_value(
                    "keyframe_inference_nms_window",
                    _memory_cfg_value("keyframe_nms_window", cluster_window_default),
                )
            ),
        )
        self.keyframe_cooldown_steps = max(
            0,
            int(
                _memory_cfg_value(
                    "keyframe_inference_cooldown_steps",
                    _memory_cfg_value("keyframe_cooldown_steps", cluster_window_default),
                )
            ),
        )
        self.enable_delayed_chunk_event_commit = _cfg_bool(
            _memory_cfg_value("enable_delayed_chunk_event_commit", True),
            True,
        )
        configured_force_current_write = _cfg_bool(
            _memory_cfg_value("force_memory_write_current_for_event_commit", False),
            False,
        )
        if configured_force_current_write:
            raise ValueError("EventVLA pure_image_keyframe_memory requires force_memory_write_current_for_event_commit=false.")
        self.force_memory_write_current_for_event_commit = False
        self.disable_current_frame_keyframe_write_in_eval = _cfg_bool(
            _memory_cfg_value("disable_current_frame_keyframe_write_in_eval", True),
            True,
        )
        configured_teacher_future_write = _cfg_bool(
            _memory_cfg_value("use_teacher_future_frame_write_in_train", False),
            False,
        )
        if configured_teacher_future_write:
            raise ValueError("EventVLA pure_image_keyframe_memory requires use_teacher_future_frame_write_in_train=false.")
        self.use_teacher_future_frame_write_in_train = False
        self.keyframe_train_memory_source = str(_memory_cfg_value("keyframe_train_memory_source", "teacher_to_predict")).lower()
        self.keyframe_eval_memory_source = str(_memory_cfg_value("keyframe_eval_memory_source", "predict")).lower()
        self.keyframe_train_memory_schedule = str(
            _memory_cfg_value("keyframe_train_memory_schedule", self.keyframe_train_memory_source)
        ).lower()
        if (
            self.keyframe_train_memory_source in {"predict", "student"}
            or self.keyframe_train_memory_schedule in {"predict", "student"}
        ):
            self.use_teacher_future_frame_write_in_train = False
        logger.info(
            "Resolved keyframe memory policy: train_source=%s train_schedule=%s "
            "use_teacher_future_frame_write_in_train=%s",
            self.keyframe_train_memory_source,
            self.keyframe_train_memory_schedule,
            self.use_teacher_future_frame_write_in_train,
        )
        self.keyframe_schedule_warmup_steps = int(_memory_cfg_value("keyframe_schedule_warmup_steps", -1))
        self.keyframe_schedule_transition_steps = int(_memory_cfg_value("keyframe_schedule_transition_steps", -1))
        self.keyframe_schedule_teacher_prob_start = float(_memory_cfg_value("keyframe_schedule_teacher_prob_start", 1.0))
        self.keyframe_schedule_teacher_prob_end = float(_memory_cfg_value("keyframe_schedule_teacher_prob_end", 0.0))
        self.keyframe_schedule_mix_granularity = str(
            _memory_cfg_value("keyframe_schedule_mix_granularity", "sample")
        ).lower()
        if self.keyframe_schedule_mix_granularity not in {"sample", "batch"}:
            logger.warning(
                "Unsupported keyframe_schedule_mix_granularity=%s; fallback to sample.",
                self.keyframe_schedule_mix_granularity,
            )
            self.keyframe_schedule_mix_granularity = "sample"
        self.keyframe_schedule_completed_steps = 0
        self.keyframe_schedule_max_train_steps = 0
        self.keyframe_schedule_progress = 0.0
        self.keyframe_memory_teacher_prob = float(self.keyframe_schedule_teacher_prob_start)
        self._last_memory_keyframe_teacher_selector: Optional[torch.Tensor] = None
        self._last_keyframe_input_teacher_selector: Optional[torch.Tensor] = None
        self._last_keyframe_input_metrics: dict = {}
        self._runtime_keyframe_image_bank: List[List[dict]] = []
        self._runtime_pending_keyframe_writes: List[List[dict]] = []
        self._runtime_slot_episode_ids: List[object] = []
        self._inference_event_state: List[dict] = []

        self.keyframe_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.register_buffer(
            "_keyframe_annotations_observed",
            torch.tensor(False, dtype=torch.bool),
            persistent=True,
        )

    def reset_memory(self, batch_size: int = 1) -> None:
        batch_size = max(int(batch_size), 0)
        self._runtime_keyframe_image_bank = [[] for _ in range(batch_size)]
        self._runtime_pending_keyframe_writes = [[] for _ in range(batch_size)]
        self._runtime_slot_episode_ids = [None for _ in range(batch_size)]
        self._inference_event_state = [self._new_inference_event_state() for _ in range(batch_size)]

    def reset_memory_by_mask(self, reset_mask: torch.Tensor, episode_ids: Optional[object] = None) -> None:
        if reset_mask is None:
            return
        mask_values = reset_mask.detach().to(dtype=torch.bool).view(-1).cpu().tolist()
        episode_list = self._normalize_episode_id_list(episode_ids, len(mask_values))
        self._ensure_runtime_slots(len(mask_values), episode_list)
        for slot_idx, should_reset in enumerate(mask_values):
            if not should_reset:
                continue
            self._runtime_keyframe_image_bank[slot_idx] = []
            self._runtime_pending_keyframe_writes[slot_idx] = []
            self._runtime_slot_episode_ids[slot_idx] = episode_list[slot_idx]
            self._inference_event_state[slot_idx] = self._new_inference_event_state(
                episode_id=episode_list[slot_idx]
            )

    @staticmethod
    def _scalar_to_python(value: object) -> object:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "item"):
            try:
                return value.item()
            except ValueError:
                pass
        return value

    @classmethod
    def _normalize_episode_identity(cls, value: object) -> Optional[str]:
        value = cls._scalar_to_python(value)
        if value is None:
            return None
        return str(value)

    @classmethod
    def _normalize_episode_id_list(cls, episode_ids: Optional[object], batch_size: int) -> List[object]:
        if episode_ids is None:
            return [None for _ in range(batch_size)]
        if hasattr(episode_ids, "detach"):
            values = episode_ids.detach().cpu().view(-1).tolist()
        elif isinstance(episode_ids, (list, tuple)):
            values = list(episode_ids)
        else:
            values = [episode_ids]
        if len(values) < batch_size:
            values = values + [None for _ in range(batch_size - len(values))]
        return [cls._scalar_to_python(value) for value in values[:batch_size]]

    @classmethod
    def _example_scalar(cls, example: dict, key: str, default: object = None) -> object:
        if not isinstance(example, dict):
            return default
        return cls._scalar_to_python(example.get(key, default))

    @classmethod
    def _example_int(cls, example: dict, key: str, default: Optional[int] = None) -> Optional[int]:
        value = cls._example_scalar(example, key, default)
        if value is None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @classmethod
    def _example_episode_id(cls, example: dict) -> object:
        return cls._example_scalar(example, "episode_id", None)

    def _ensure_runtime_slots(self, batch_size: int, episode_ids: Optional[List[object]] = None) -> None:
        batch_size = max(int(batch_size), 0)
        if not hasattr(self, "_inference_event_state"):
            self._inference_event_state = []
        while len(self._runtime_keyframe_image_bank) < batch_size:
            self._runtime_keyframe_image_bank.append([])
        while len(self._runtime_pending_keyframe_writes) < batch_size:
            self._runtime_pending_keyframe_writes.append([])
        while len(self._runtime_slot_episode_ids) < batch_size:
            self._runtime_slot_episode_ids.append(None)
        while len(self._inference_event_state) < batch_size:
            self._inference_event_state.append(self._new_inference_event_state())

        if episode_ids is None:
            return
        for slot_idx in range(batch_size):
            episode_id = episode_ids[slot_idx] if slot_idx < len(episode_ids) else None
            if episode_id is None:
                continue
            previous_episode = self._runtime_slot_episode_ids[slot_idx]
            if (
                previous_episode is not None
                and self._normalize_episode_identity(previous_episode)
                != self._normalize_episode_identity(episode_id)
            ):
                self._runtime_keyframe_image_bank[slot_idx] = []
                self._runtime_pending_keyframe_writes[slot_idx] = []
                self._inference_event_state[slot_idx] = self._new_inference_event_state(
                    episode_id=episode_id
                )
            self._runtime_slot_episode_ids[slot_idx] = episode_id
            self._inference_event_state[slot_idx]["episode_id"] = episode_id

    def _ensure_runtime_slots_for_examples(self, examples: Optional[List[dict]]) -> List[object]:
        if examples is None:
            return []
        episode_ids = [self._example_episode_id(example) for example in examples]
        self._ensure_runtime_slots(len(examples), episode_ids)
        return episode_ids

    def _same_episode(self, left: object, right: object) -> bool:
        if left is None or right is None:
            return left is right
        return self._normalize_episode_identity(left) == self._normalize_episode_identity(right)

    @staticmethod
    def _module_compute_dtype(module: Optional[nn.Module], fallback: Optional[torch.Tensor] = None) -> torch.dtype:
        if module is not None:
            for param in module.parameters():
                if param.is_floating_point():
                    return param.dtype
            for buffer in module.buffers():
                if buffer.is_floating_point():
                    return buffer.dtype
        if isinstance(fallback, torch.Tensor) and fallback.dtype.is_floating_point:
            return fallback.dtype
        return torch.float32

    @staticmethod
    def _as_python_bool(value: object) -> bool:
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, str):
            return value.strip().lower() not in {"", "0", "false", "no", "off", "none", "null"}
        return bool(value)

    @staticmethod
    def _example_keyframe_steps(example: dict) -> List[int]:
        raw_steps = example.get("keyframe_steps", example.get("inspect_keyframe_steps", []))
        if raw_steps is None:
            return []
        return [int(step.item() if hasattr(step, "item") else step) for step in raw_steps]

    def _extract_keyframe_annotation_mask(
        self,
        examples: Optional[List[dict]],
        device: Optional[torch.device] = None,
    ) -> Optional[torch.Tensor]:
        if examples is None or len(examples) == 0:
            return None

        values: List[bool] = []
        saw_explicit_flag = False
        for example in examples:
            if "use_keyframe_supervision" in example:
                saw_explicit_flag = True
                values.append(self._as_python_bool(example.get("use_keyframe_supervision", False)))
            elif "has_keyframe_annotations" in example:
                saw_explicit_flag = True
                values.append(self._as_python_bool(example.get("has_keyframe_annotations", False)))
            elif "has_inspect_keyframe_annotations" in example:
                saw_explicit_flag = True
                values.append(self._as_python_bool(example.get("has_inspect_keyframe_annotations", False)))
            else:
                values.append(
                    example.get("chunk_keyframe_target", None) is not None
                    or example.get("teacher_should_commit", None) is not None
                    or example.get("is_keyframe", None) is not None
                    or example.get("is_keyframe_proxy", None) is not None
                )

        if not saw_explicit_flag and not any(values):
            return None

        return torch.tensor(values, device=device, dtype=torch.bool)

    def _mark_keyframe_annotations_observed(self, annotation_mask: Optional[torch.Tensor]) -> None:
        if annotation_mask is None:
            return
        if bool(annotation_mask.detach().to(dtype=torch.bool).any().item()):
            self._keyframe_annotations_observed.fill_(True)

    def _should_use_keyframe_predict_head(
        self,
        annotation_mask: Optional[torch.Tensor],
    ) -> bool:
        if self.keyframe_predict_head_mode == "enabled":
            return True
        if self.keyframe_predict_head_mode == "disabled":
            return False

        if self.training:
            return annotation_mask is None or bool(annotation_mask.detach().to(dtype=torch.bool).any().item())

        if annotation_mask is not None:
            return bool(annotation_mask.detach().to(dtype=torch.bool).any().item())
        return bool(self._keyframe_annotations_observed.detach().cpu().item())

    def _zero_keyframe_head_loss(self, reference: torch.Tensor) -> torch.Tensor:
        zero = reference.new_zeros(())
        for param in self.keyframe_head.parameters():
            if param.requires_grad:
                zero = zero + param.sum().to(dtype=reference.dtype) * 0.0
        return zero

    def _empty_keyframe_predictions(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        probs = torch.zeros((batch_size, self.chunk_len), device=device, dtype=dtype)
        pred_mask = torch.zeros((batch_size, self.chunk_len), device=device, dtype=torch.bool)
        event_offset = torch.full((batch_size,), -1, device=device, dtype=torch.long)
        event_confidence = torch.zeros((batch_size,), device=device, dtype=dtype)
        should_commit = torch.zeros((batch_size,), device=device, dtype=torch.bool)
        return probs, pred_mask, event_offset, event_confidence, should_commit

    @staticmethod
    def _new_inference_event_state(episode_id: Optional[object] = None) -> dict:
        return {
            "episode_id": episode_id,
            "pending_step": None,
            "pending_confidence": 0.0,
            "pending_plan_start_step": None,
            "pending_offset": None,
            "last_committed_step": None,
        }

    @staticmethod
    def _tensor_to_numpy(tensor: torch.Tensor):
        tensor = tensor.detach()
        if tensor.is_floating_point() and tensor.dtype not in (torch.float32, torch.float64):
            tensor = tensor.to(dtype=torch.float32)
        return tensor.cpu().numpy()

    def set_keyframe_schedule_state(self, completed_steps: int, max_train_steps: Optional[int] = None) -> None:
        self.keyframe_schedule_completed_steps = max(int(completed_steps), 0)
        if max_train_steps is not None:
            self.keyframe_schedule_max_train_steps = max(int(max_train_steps), 0)
        teacher_prob, schedule_progress = self._compute_current_keyframe_teacher_prob()
        self.keyframe_memory_teacher_prob = teacher_prob
        self.keyframe_schedule_progress = schedule_progress

    def _forward_qwen_without_memory(self, qwen_inputs) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            return qwenvl_outputs.hidden_states[-1]

    def _requires_keyframe_image_memory(self) -> bool:
        return self.memory_injection_mode in self._PURE_KEYFRAME_IMAGE_MODES

    def _use_keyframe_image_inputs(self) -> bool:
        return (
            self.memory_injection_enabled
            and self.memory_injection_mode in self._PURE_KEYFRAME_IMAGE_MODES
        )

    @staticmethod
    def _cfg_lookup(container: object, name: str, default: object = None) -> object:
        if container is None:
            return default
        if hasattr(container, "get"):
            return container.get(name, default)
        return getattr(container, name, default)

    def _runtime_memory_view_matches(self, meta: dict) -> bool:
        cfg = getattr(self.config.datasets.vla_data, "keyframe_image_memory", {}) or {}
        view_text = " ".join(
            str(meta.get(key, "")).lower()
            for key in ("view", "video_key", "modality_key", "camera", "name")
        )
        exclude_patterns = tuple(
            str(name).lower()
            for name in self._cfg_lookup(cfg, "exclude_name_patterns", ["wrist"])
        )
        if any(pattern and pattern in view_text for pattern in exclude_patterns):
            return False
        include_names = tuple(
            str(name).lower()
            for name in self._cfg_lookup(cfg, "include_names", ["cam_high", "head", "main"])
        )
        if any(name and name in view_text for name in include_names):
            return True
        return "wrist" not in view_text

    def _select_runtime_memory_view(self, images: List[Image.Image], metas: List[dict]) -> int:
        if len(images) == 0:
            raise ValueError("runtime keyframe exact fetch returned no images")
        if len(metas) != len(images):
            metas = [{} for _ in images]
        current_indices = [
            idx
            for idx, meta in enumerate(metas)
            if str(meta.get("time_role", "")).lower() == "current" or meta.get("delta", None) == 0
        ]
        search_indices = current_indices if current_indices else list(range(len(images)))
        candidates = [idx for idx in search_indices if self._runtime_memory_view_matches(metas[idx])]
        return int(candidates[0] if candidates else search_indices[0])

    def _append_runtime_keyframe_entry(self, slot_idx: int, entry: dict) -> None:
        self._ensure_runtime_slots(slot_idx + 1)
        episode_id = entry.get("episode_id", None)
        step = int(entry.get("step", entry.get("target_step", -1)))
        bank = [
            item
            for item in self._runtime_keyframe_image_bank[slot_idx]
            if not (
                self._same_episode(item.get("episode_id", None), episode_id)
                and int(item.get("step", -1)) == step
            )
        ]
        bank.append(entry)
        bank.sort(key=lambda item: int(item.get("step", -1)))
        if self.max_keyframe_images > 0:
            bank = bank[-int(self.max_keyframe_images) :]
        self._runtime_keyframe_image_bank[slot_idx] = bank

    def _runtime_entries_for_example(self, slot_idx: int, example: dict) -> List[dict]:
        self._ensure_runtime_slots(slot_idx + 1)
        episode_id = self._example_episode_id(example)
        current_step = self._example_int(example, "timestep", None)
        entries = []
        for entry in self._runtime_keyframe_image_bank[slot_idx]:
            if episode_id is not None and not self._same_episode(entry.get("episode_id", None), episode_id):
                continue
            step = int(entry.get("step", entry.get("target_step", -1)))
            if current_step is not None and step > int(current_step):
                continue
            entries.append(entry)
        entries.sort(key=lambda item: int(item.get("step", -1)))
        if self.max_keyframe_images > 0:
            entries = entries[-int(self.max_keyframe_images) :]
        return entries

    def _runtime_memory_fields_for_example(self, slot_idx: int, example: dict) -> Tuple[List[Image.Image], List[dict], List[int]]:
        images: List[Image.Image] = []
        metas: List[dict] = []
        steps: List[int] = []
        for entry in self._runtime_entries_for_example(slot_idx, example):
            entry_images = entry.get("images", []) or []
            entry_metas = entry.get("metas", []) or []
            images.extend(entry_images)
            metas.extend(entry_metas)
            steps.extend([int(entry.get("step", entry.get("target_step", -1)))] * len(entry_images))
        return images, metas, steps

    def _consume_runtime_memory_exact_fetches(self, examples: Optional[List[dict]]) -> dict:
        metrics = {"runtime_memory_exact_fetch_consumed": 0.0, "runtime_memory_exact_fetch_dropped": 0.0}
        if not examples:
            return metrics
        self._ensure_runtime_slots_for_examples(examples)
        fetched_items = examples[0].get("runtime_memory_exact_fetches", []) or []
        if not isinstance(fetched_items, (list, tuple)):
            fetched_items = [fetched_items]

        for fetched in fetched_items:
            if not isinstance(fetched, dict):
                metrics["runtime_memory_exact_fetch_dropped"] += 1.0
                continue
            slot_idx = self._example_int(fetched, "slot_idx", None)
            if slot_idx is None or slot_idx < 0 or slot_idx >= len(examples):
                metrics["runtime_memory_exact_fetch_dropped"] += 1.0
                continue
            slot_episode = self._runtime_slot_episode_ids[slot_idx]
            fetched_episode = fetched.get("episode_id", None)
            if slot_episode is not None and fetched_episode is not None and not self._same_episode(slot_episode, fetched_episode):
                metrics["runtime_memory_exact_fetch_dropped"] += 1.0
                continue

            raw_images = fetched.get("images", []) or []
            images = self._ensure_image_list(to_pil_preserve(raw_images)) if raw_images else []
            if len(images) == 0:
                metrics["runtime_memory_exact_fetch_dropped"] += 1.0
                continue
            raw_metas = fetched.get("image_metas", None) or [{} for _ in images]
            metas = [dict(meta) if isinstance(meta, dict) else {"value": meta} for meta in raw_metas]
            selected_idx = self._select_runtime_memory_view(images, metas)
            selected_meta = dict(metas[selected_idx]) if selected_idx < len(metas) else {}
            target_step = int(fetched.get("target_step", selected_meta.get("source_timestep", -1)))
            memory_meta = {
                **selected_meta,
                "role": "memory_keyframe",
                "time_role": "memory",
                "source_timestep": target_step,
                "view": selected_meta.get("view", "main"),
                "view_index": int(selected_meta.get("view_index", 0) or 0),
                "trajectory_id": fetched.get("trajectory_id", selected_meta.get("trajectory_id", None)),
                "dataset_index": fetched.get("dataset_index", None),
                "source": fetched.get("source", "predict_exact"),
                "confidence": float(fetched.get("confidence", 0.0) or 0.0),
            }
            self._append_runtime_keyframe_entry(
                slot_idx=slot_idx,
                entry={
                    "step": target_step,
                    "images": [images[selected_idx]],
                    "metas": [memory_meta],
                    "confidence": float(fetched.get("confidence", 0.0) or 0.0),
                    "source": fetched.get("source", "predict_exact"),
                    "episode_id": fetched_episode,
                    "dataset_index": fetched.get("dataset_index", None),
                    "trajectory_id": fetched.get("trajectory_id", None),
                    "sample_step": fetched.get("sample_step", None),
                },
            )
            metrics["runtime_memory_exact_fetch_consumed"] += 1.0

        examples[0]["runtime_memory_exact_fetch_consumed"] = int(metrics["runtime_memory_exact_fetch_consumed"])
        examples[0]["runtime_memory_exact_fetch_dropped"] = int(metrics["runtime_memory_exact_fetch_dropped"])
        return metrics

    def _resolve_keyframe_input_sources(self, batch_size: int) -> Tuple[List[str], torch.Tensor, dict]:
        source = self.keyframe_train_memory_source if self.training else self.keyframe_eval_memory_source
        teacher_prob = 0.0
        schedule_progress = 1.0
        if source in {"teacher", "gt"}:
            selector = torch.ones((batch_size,), dtype=torch.bool)
            sources = ["teacher" for _ in range(batch_size)]
            teacher_prob = 1.0
            schedule_progress = 0.0
        elif source in {"predict", "student"}:
            selector = torch.zeros((batch_size,), dtype=torch.bool)
            sources = ["predict" for _ in range(batch_size)]
        elif source in {"none", "disabled"}:
            selector = torch.zeros((batch_size,), dtype=torch.bool)
            sources = ["none" for _ in range(batch_size)]
        elif source == "union":
            selector = torch.zeros((batch_size,), dtype=torch.bool)
            sources = ["union" for _ in range(batch_size)]
        elif source in {"teacher_to_predict", "teacher_then_predict", "schedule", "scheduled", "mixed"}:
            teacher_prob, schedule_progress = self._compute_current_keyframe_teacher_prob()
            if self.keyframe_schedule_mix_granularity == "batch":
                use_teacher = bool(torch.rand(1).item() < teacher_prob)
                selector = torch.full((batch_size,), use_teacher, dtype=torch.bool)
            else:
                selector = torch.rand((batch_size,), dtype=torch.float32) < float(teacher_prob)
            sources = ["teacher" if bool(value) else "predict" for value in selector.tolist()]
        else:
            raise ValueError(f"Unsupported keyframe memory source: {source}")

        self.keyframe_memory_teacher_prob = float(teacher_prob)
        self.keyframe_schedule_progress = float(schedule_progress)
        self._last_keyframe_input_teacher_selector = selector.detach()
        teacher_usage = float(selector.float().mean().item()) if batch_size > 0 else 0.0
        predict_usage = float(sum(1 for value in sources if value == "predict") / max(batch_size, 1))
        metrics = {
            "keyframe_input_teacher_prob": float(teacher_prob),
            "keyframe_input_teacher_usage": teacher_usage,
            "keyframe_input_predict_usage": predict_usage,
            "keyframe_input_schedule_progress": float(schedule_progress),
        }
        return sources, selector, metrics

    @staticmethod
    def _normalize_memory_field_list(value: object) -> list:
        if value is None:
            return []
        if isinstance(value, list):
            return list(value)
        if isinstance(value, tuple):
            return list(value)
        return [value]

    def _combine_memory_fields(
        self,
        first: Tuple[list, list, list],
        second: Tuple[list, list, list],
    ) -> Tuple[list, list, list]:
        combined = []
        for images, metas, steps in (first, second):
            images = self._normalize_memory_field_list(images)
            metas = self._normalize_memory_field_list(metas)
            steps = self._normalize_memory_field_list(steps)
            for idx, image in enumerate(images):
                step = int(steps[idx]) if idx < len(steps) else -1
                meta = dict(metas[idx]) if idx < len(metas) and isinstance(metas[idx], dict) else {}
                combined.append((step, image, meta))
        dedup = {}
        for step, image, meta in combined:
            dedup[step] = (image, meta)
        items = sorted(dedup.items(), key=lambda item: int(item[0]))
        if self.max_keyframe_images > 0:
            items = items[-int(self.max_keyframe_images) :]
        return (
            [image for _, (image, _) in items],
            [meta for _, (_, meta) in items],
            [int(step) for step, _ in items],
        )

    def _resolve_training_keyframe_inputs(self, examples: Optional[List[dict]]) -> List[dict]:
        if not examples:
            self._last_keyframe_input_metrics = {}
            self._last_keyframe_input_teacher_selector = None
            return examples

        fetch_metrics = self._consume_runtime_memory_exact_fetches(examples)
        self._ensure_runtime_slots_for_examples(examples)
        sources, _, source_metrics = self._resolve_keyframe_input_sources(len(examples))
        selected_counts = []
        runtime_counts = []
        pending_counts = []

        for slot_idx, example in enumerate(examples):
            dataloader_images = self._normalize_memory_field_list(
                example.get("_keyframe_dataloader_memory_images", example.get("memory_keyframe_images", []))
            )
            dataloader_metas = self._normalize_memory_field_list(
                example.get("_keyframe_dataloader_memory_metas", example.get("memory_keyframe_image_metas", []))
            )
            dataloader_steps = self._normalize_memory_field_list(
                example.get("_keyframe_dataloader_memory_steps", example.get("memory_keyframe_steps", []))
            )
            if "_keyframe_dataloader_memory_images" not in example:
                example["_keyframe_dataloader_memory_images"] = list(dataloader_images)
                example["_keyframe_dataloader_memory_metas"] = list(dataloader_metas)
                example["_keyframe_dataloader_memory_steps"] = list(dataloader_steps)

            runtime_images, runtime_metas, runtime_steps = self._runtime_memory_fields_for_example(slot_idx, example)
            source = sources[slot_idx]
            if source == "teacher":
                selected_images, selected_metas, selected_steps = dataloader_images, dataloader_metas, dataloader_steps
            elif source == "predict":
                selected_images, selected_metas, selected_steps = runtime_images, runtime_metas, runtime_steps
            elif source == "union":
                selected_images, selected_metas, selected_steps = self._combine_memory_fields(
                    (dataloader_images, dataloader_metas, dataloader_steps),
                    (runtime_images, runtime_metas, runtime_steps),
                )
            else:
                selected_images, selected_metas, selected_steps = [], [], []

            example["memory_keyframe_images"] = list(selected_images)
            example["memory_keyframe_image_metas"] = list(selected_metas)
            example["memory_keyframe_steps"] = [int(step) for step in selected_steps]
            example["memory_keyframe_count"] = int(len(selected_images))
            example["keyframe_input_memory_source"] = source
            example["keyframe_input_dataloader_steps"] = [int(step) for step in dataloader_steps]
            example["keyframe_input_runtime_steps"] = [int(step) for step in runtime_steps]
            example["keyframe_input_steps"] = [int(step) for step in selected_steps]
            selected_counts.append(len(selected_images))
            runtime_counts.append(len(runtime_images))
            pending_counts.append(len(self._runtime_pending_keyframe_writes[slot_idx]))

        self._last_keyframe_input_metrics = {
            **source_metrics,
            **fetch_metrics,
            "keyframe_input_memory_count": float(np.mean(selected_counts)) if selected_counts else 0.0,
            "runtime_keyframe_bank_count": float(np.mean(runtime_counts)) if runtime_counts else 0.0,
            "runtime_pending_keyframe_count": float(np.mean(pending_counts)) if pending_counts else 0.0,
        }
        return examples

    def collect_due_predict_exact_fetch_requests(self, examples: Optional[List[dict]]) -> List[dict]:
        if not examples:
            return []
        self._ensure_runtime_slots_for_examples(examples)
        requests: List[dict] = []
        for slot_idx, example in enumerate(examples):
            current_step = self._example_int(example, "timestep", None)
            episode_id = self._example_episode_id(example)
            if current_step is None:
                continue
            still_pending = []
            for pending in self._runtime_pending_keyframe_writes[slot_idx]:
                pending_episode = pending.get("episode_id", None)
                if episode_id is not None and pending_episode is not None and not self._same_episode(episode_id, pending_episode):
                    continue
                target_step = int(pending.get("target_step", -1))
                if target_step <= int(current_step):
                    if pending.get("trajectory_id", None) is None:
                        continue
                    request = dict(pending)
                    request["slot_idx"] = int(slot_idx)
                    request["request_id"] = pending.get(
                        "request_id",
                        f"predict_exact:{slot_idx}:{self._normalize_episode_identity(pending_episode)}:{pending.get('sample_step', -1)}:{target_step}",
                    )
                    requests.append(request)
                else:
                    still_pending.append(pending)
            self._runtime_pending_keyframe_writes[slot_idx] = still_pending
        return requests

    def _register_predict_keyframe_writes(
        self,
        examples: Optional[List[dict]],
        pred_event_offset: torch.Tensor,
        pred_event_confidence: torch.Tensor,
        predicted_should_commit: torch.Tensor,
        keyframe_annotation_mask: Optional[torch.Tensor] = None,
    ) -> int:
        if not self.training or not examples:
            return 0
        self._ensure_runtime_slots_for_examples(examples)
        offsets = pred_event_offset.detach().to(dtype=torch.long).view(-1).cpu().tolist()
        confidences = pred_event_confidence.detach().to(dtype=torch.float32).view(-1).cpu().tolist()
        should_commit = predicted_should_commit.detach().to(dtype=torch.bool).view(-1).cpu().tolist()
        if keyframe_annotation_mask is not None:
            annotation_values = keyframe_annotation_mask.detach().to(dtype=torch.bool).view(-1).cpu().tolist()
        else:
            annotation_values = [True for _ in should_commit]

        registered = 0
        for slot_idx, example in enumerate(examples):
            if slot_idx >= len(should_commit) or not bool(should_commit[slot_idx]):
                continue
            if slot_idx < len(annotation_values) and not bool(annotation_values[slot_idx]):
                continue
            sample_step = self._example_int(example, "timestep", None)
            offset = int(offsets[slot_idx]) if slot_idx < len(offsets) else -1
            if sample_step is None or offset < 0:
                continue
            target_step = int(sample_step) + int(offset)
            episode_id = self._example_episode_id(example)
            pending = {
                "slot_idx": int(slot_idx),
                "dataset_index": self._example_int(example, "dataset_index", 0),
                "trajectory_id": self._example_scalar(example, "trajectory_id", None),
                "episode_id": episode_id,
                "sample_step": int(sample_step),
                "target_step": int(target_step),
                "confidence": float(confidences[slot_idx]) if slot_idx < len(confidences) else 0.0,
                "source": "predict_exact",
                "instruction": example.get("lang", ""),
            }
            pending["request_id"] = (
                f"predict_exact:{slot_idx}:{self._normalize_episode_identity(episode_id)}:"
                f"{int(sample_step)}:{int(target_step)}"
            )
            existing = self._runtime_pending_keyframe_writes[slot_idx]
            duplicate = any(
                self._same_episode(item.get("episode_id", None), episode_id)
                and int(item.get("target_step", -1)) == int(target_step)
                for item in existing
            )
            if not duplicate:
                existing.append(pending)
                registered += 1

        if examples:
            examples[0]["predict_exact_pending_registered"] = int(registered)
        return registered

    def _keyframe_input_metric_tensors(self, reference: torch.Tensor) -> dict:
        metrics = {}
        for name, value in self._last_keyframe_input_metrics.items():
            metrics[name] = torch.tensor(float(value), device=reference.device, dtype=torch.float32).detach()
        return metrics

    @staticmethod
    def _ensure_image_list(images) -> List[Image.Image]:
        if isinstance(images, (list, tuple)):
            return list(images)
        return [images]

    def _fallback_image_metas_for_sample(self, n_images: int) -> List[dict]:
        if n_images <= 0:
            return []
        if n_images >= 4 and n_images % 4 == 0:
            frame_roles = [
                ("first", None, 0),
                ("history", -30, None),
                ("history", -15, None),
                ("current", 0, None),
            ]
        elif n_images >= 3 and n_images % 3 == 0:
            frame_roles = [
                ("history", -30, None),
                ("history", -15, None),
                ("current", 0, None),
            ]
        else:
            frame_roles = [("current", 0, None)]

        frame_count = len(frame_roles)
        views_per_frame = max(1, n_images // frame_count)
        metas: List[dict] = []
        for image_idx in range(n_images):
            frame_idx = min(image_idx // views_per_frame, frame_count - 1)
            view_idx = image_idx % views_per_frame
            time_role, delta, absolute_index = frame_roles[frame_idx]
            metas.append(
                {
                    "role": "anchor",
                    "time_role": time_role,
                    "delta": delta,
                    "absolute_index": absolute_index,
                    "view": "main" if view_idx == 0 else f"wrist_{view_idx}",
                    "view_index": view_idx,
                }
            )
        return metas

    def _infer_batch_image_metas(self, examples: List[dict], batch_images: List[List[Image.Image]]) -> List[List[dict]]:
        batch_metas: List[List[dict]] = []
        for example, images in zip(examples, batch_images):
            raw_metas = example.get("image_metas", None)
            if raw_metas is not None and len(raw_metas) == len(images):
                metas = [dict(meta) if isinstance(meta, dict) else {"value": meta} for meta in raw_metas]
            else:
                metas = self._fallback_image_metas_for_sample(len(images))
            batch_metas.append(metas)
        return batch_metas

    def _augment_batch_images_with_keyframe_images(
        self,
        batch_images: List[List[Image.Image]],
        batch_image_metas: List[List[dict]],
        examples: List[dict],
    ) -> Tuple[List[List[Image.Image]], List[List[dict]]]:
        if not self._use_keyframe_image_inputs():
            return batch_images, batch_image_metas

        if self.keyframe_image_position not in {
            "after_anchor_images_before_action",
            "before_anchor_images",
        }:
            raise ValueError(f"Unsupported keyframe_image_position={self.keyframe_image_position}")

        max_keyframes = max(0, int(self.max_keyframe_images))
        augmented_images: List[List[Image.Image]] = []
        augmented_metas: List[List[dict]] = []
        for sample_idx, (anchor_images, anchor_metas, example) in enumerate(
            zip(batch_images, batch_image_metas, examples)
        ):
            raw_mem_images = example.get("memory_keyframe_images", []) or []
            if len(raw_mem_images) > 0:
                mem_images = self._ensure_image_list(to_pil_preserve(raw_mem_images))
            else:
                mem_images = []
            raw_mem_metas = example.get("memory_keyframe_image_metas", []) or []
            mem_metas = [
                dict(meta) if isinstance(meta, dict) else {"value": meta}
                for meta in raw_mem_metas
            ]
            mem_steps = example.get("memory_keyframe_steps", []) or []

            if max_keyframes == 0:
                mem_images = []
                mem_metas = []
            else:
                mem_images = list(mem_images)[-max_keyframes:]
                mem_metas = list(mem_metas)[-max_keyframes:]

            if len(mem_metas) < len(mem_images):
                offset = len(mem_images) - len(mem_metas)
                padded_metas = []
                for idx in range(offset):
                    step_idx = idx - offset
                    source_step = None
                    if len(mem_steps) >= len(mem_images):
                        source_step = int(mem_steps[step_idx])
                    padded_metas.append(
                        {
                            "role": "memory_keyframe",
                            "time_role": "memory",
                            "source_timestep": source_step,
                            "view": "main",
                            "view_index": 0,
                            "sample_index": int(sample_idx),
                        }
                    )
                mem_metas = padded_metas + mem_metas
            elif len(mem_metas) > len(mem_images):
                mem_metas = mem_metas[-len(mem_images):] if mem_images else []

            anchor_images_i = list(anchor_images)
            anchor_metas_i = [dict(meta) for meta in anchor_metas]
            mem_metas_i = [
                {
                    **dict(meta),
                    "role": "memory_keyframe",
                    "time_role": "memory",
                }
                for meta in mem_metas
            ]
            if self.keyframe_image_position == "before_anchor_images":
                images_i = list(mem_images) + anchor_images_i
                metas_i = mem_metas_i + anchor_metas_i
            else:
                images_i = anchor_images_i + list(mem_images)
                metas_i = anchor_metas_i + mem_metas_i
            augmented_images.append(images_i)
            augmented_metas.append(metas_i)

        return augmented_images, augmented_metas

    def forward(self, examples: List[dict] = None, **kwargs) -> Tuple:
        examples = self._resolve_training_keyframe_inputs(examples)
        batch_images = [self._ensure_image_list(example["image"]) for example in examples]
        batch_image_metas = self._infer_batch_image_metas(examples, batch_images)
        instructions = self._build_vla_prompts([example["lang"] for example in examples])
        raw_actions = [example["action"] for example in examples]

        qwen_images, qwen_image_metas = self._augment_batch_images_with_keyframe_images(
            batch_images=batch_images,
            batch_image_metas=batch_image_metas,
            examples=examples,
        )
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=qwen_images,
            instructions=instructions,
            image_metas=qwen_image_metas,
            use_image_role_text=self.use_image_role_text,
        )
        input_ids = qwen_inputs.get("input_ids", None)
        batch_timesteps = self._extract_batch_timesteps(examples=examples, device=input_ids.device)
        last_hidden = self._forward_qwen_without_memory(qwen_inputs)

        teacher_commit_timesteps = None
        predicted_peak_timesteps = None
        predict_exact_pending_registered = 0

        with torch.autocast("cuda", dtype=torch.float32):
            action_queries = self._gather_action_token_embeddings(
                last_hidden,
                input_ids,
                action_token_id=self.action_token_id,
            )
            keyframe_annotation_mask = self._extract_keyframe_annotation_mask(
                examples=examples,
                device=action_queries.device,
            )
            self._mark_keyframe_annotations_observed(keyframe_annotation_mask)
            use_keyframe_predict_head = self._should_use_keyframe_predict_head(keyframe_annotation_mask)
            keyframe_device = action_queries.device
            keyframe_dtype = action_queries.dtype
            keyframe_loss = None
            keyframe_head_zero_loss = action_queries.new_zeros(())

            if use_keyframe_predict_head:
                chunk_keyframe_logits = self.keyframe_head(action_queries).squeeze(-1)
                keyframe_device = chunk_keyframe_logits.device
                keyframe_dtype = chunk_keyframe_logits.dtype
                chunk_keyframe_probs = torch.sigmoid(chunk_keyframe_logits)
                predicted_chunk_keyframe_mask = chunk_keyframe_probs >= self.keyframe_threshold
                pred_event_offset, pred_event_confidence, predicted_should_commit = self._select_chunk_event(
                    chunk_keyframe_probs
                )
                keyframe_targets = self._extract_batch_keyframe_targets(
                    examples=examples,
                    device=keyframe_device,
                    dtype=keyframe_dtype,
                )
                teacher_event_offset, teacher_event_confidence, teacher_should_commit = (
                    self._extract_batch_teacher_event_targets(
                        examples=examples,
                        device=keyframe_device,
                        dtype=keyframe_dtype,
                    )
                )
                keyframe_loss = self._compute_keyframe_loss(
                    keyframe_logits=chunk_keyframe_logits,
                    keyframe_targets=keyframe_targets,
                    supervision_mask=keyframe_annotation_mask,
                )
                if keyframe_loss is None:
                    keyframe_head_zero_loss = self._zero_keyframe_head_loss(action_queries)
                if teacher_should_commit is not None and keyframe_annotation_mask is not None:
                    supervised_mask = keyframe_annotation_mask.to(
                        device=teacher_should_commit.device,
                        dtype=torch.bool,
                    ).view(-1)
                    teacher_should_commit = teacher_should_commit & supervised_mask
            else:
                (
                    chunk_keyframe_probs,
                    predicted_chunk_keyframe_mask,
                    pred_event_offset,
                    pred_event_confidence,
                    predicted_should_commit,
                ) = self._empty_keyframe_predictions(
                    batch_size=action_queries.shape[0],
                    device=keyframe_device,
                    dtype=keyframe_dtype,
                )
                keyframe_targets = None
                teacher_event_offset, teacher_event_confidence, teacher_should_commit = None, None, None
                keyframe_head_zero_loss = self._zero_keyframe_head_loss(action_queries)

            teacher_commit_timesteps = self._resolve_event_timesteps(
                base_timesteps=batch_timesteps,
                event_offsets=teacher_event_offset,
                event_mask=teacher_should_commit,
            )

            disable_regular_current_frame_write = (
                self.enable_delayed_chunk_event_commit
                and (not self.training)
                and self.disable_current_frame_keyframe_write_in_eval
            )
            if use_keyframe_predict_head:
                memory_keyframe_mask, memory_control_metrics = self._resolve_memory_keyframe_mask(
                    predicted_mask=predicted_should_commit,
                    keyframe_targets=teacher_should_commit,
                    force_memory_write_current=False,
                    disable_regular_current_frame_write=disable_regular_current_frame_write,
                )
                if keyframe_annotation_mask is not None:
                    memory_keyframe_mask = memory_keyframe_mask & keyframe_annotation_mask.to(
                        device=memory_keyframe_mask.device,
                        dtype=torch.bool,
                    ).view(-1)
                predict_exact_pending_registered = self._register_predict_keyframe_writes(
                    examples=examples,
                    pred_event_offset=pred_event_offset,
                    pred_event_confidence=pred_event_confidence,
                    predicted_should_commit=predicted_should_commit,
                    keyframe_annotation_mask=keyframe_annotation_mask,
                )
            else:
                memory_keyframe_mask = torch.zeros_like(predicted_should_commit, dtype=torch.bool)
                zero_metric = torch.tensor(0.0, device=keyframe_device, dtype=torch.float32)
                memory_control_metrics = {
                    "keyframe_memory_teacher_prob": zero_metric.detach(),
                    "keyframe_memory_teacher_usage": zero_metric.detach(),
                    "keyframe_memory_predict_usage": zero_metric.detach(),
                    "keyframe_memory_schedule_progress": zero_metric.detach(),
                }

            pred_actions = self.action_model.predict_action(action_queries)
            actions_tensor = torch.tensor(np.array(raw_actions), device=pred_actions.device, dtype=pred_actions.dtype)
            actions_target = actions_tensor[:, -self.chunk_len :, :]
            action_loss = self.l1_loss(pred_actions, actions_target)
            total_loss = action_loss + keyframe_head_zero_loss
            if keyframe_loss is not None:
                total_loss = total_loss + self.keyframe_loss_weight * keyframe_loss

        output = {
            "action_loss": action_loss,
            "total_loss": total_loss,
            "chunk_keyframe_prob": chunk_keyframe_probs.detach(),
            "chunk_keyframe_pred_mask": predicted_chunk_keyframe_mask.detach(),
            "pred_event_offset": pred_event_offset.detach(),
            "pred_event_confidence": pred_event_confidence.detach(),
            "should_trigger_event": predicted_should_commit.detach(),
            "keyframe_prob": pred_event_confidence.detach(),
            "predicted_is_keyframe": predicted_should_commit.detach(),
            "memory_is_keyframe": memory_keyframe_mask.detach(),
            "keyframe_memory_rate": memory_keyframe_mask.float().mean().detach(),
            "keyframe_head_enabled": torch.tensor(
                float(use_keyframe_predict_head),
                device=keyframe_device,
                dtype=torch.float32,
            ).detach(),
            "keyframe_annotation_rate": (
                torch.tensor(1.0, device=keyframe_device, dtype=torch.float32)
                if keyframe_annotation_mask is None
                else keyframe_annotation_mask.float().mean()
            ).detach(),
            "predict_exact_pending_registered": torch.tensor(
                float(predict_exact_pending_registered),
                device=keyframe_device,
                dtype=torch.float32,
            ).detach(),
        }
        output.update(memory_control_metrics)
        output.update(self._keyframe_input_metric_tensors(action_queries))
        if keyframe_loss is not None:
            output["keyframe_loss"] = keyframe_loss
        if teacher_event_offset is not None:
            output["teacher_event_offset"] = teacher_event_offset.detach()
        if teacher_event_confidence is not None:
            output["teacher_event_confidence"] = teacher_event_confidence.detach()
        if teacher_should_commit is not None:
            output["teacher_should_commit"] = teacher_should_commit.detach()
        if teacher_commit_timesteps is not None:
            output["teacher_commit_timestep"] = teacher_commit_timesteps.detach()
        metric_sample_mask = None
        if keyframe_annotation_mask is not None:
            metric_sample_mask = keyframe_annotation_mask.to(
                device=predicted_chunk_keyframe_mask.device,
                dtype=torch.bool,
            ).view(-1)
        if keyframe_targets is not None and (metric_sample_mask is None or bool(metric_sample_mask.any().item())):
            metric_targets = keyframe_targets
            metric_predictions = predicted_chunk_keyframe_mask
            if metric_sample_mask is not None:
                metric_targets = metric_targets[metric_sample_mask]
                metric_predictions = metric_predictions[metric_sample_mask]
            target_mask = metric_targets >= 0.5
            chunk_accuracy = (metric_predictions == target_mask).float().mean().detach()
            chunk_pred_rate = metric_predictions.float().mean().detach()
            chunk_target_rate = target_mask.float().mean().detach()
            output["chunk_keyframe_accuracy"] = chunk_accuracy
            output["chunk_keyframe_pred_rate"] = chunk_pred_rate
            output["chunk_keyframe_target_rate"] = chunk_target_rate
            output["keyframe_accuracy"] = chunk_accuracy
            output["keyframe_pred_rate"] = chunk_pred_rate
            output["keyframe_target_rate"] = chunk_target_rate
            if bool(target_mask.any().item()):
                chunk_recall = metric_predictions[target_mask].float().mean().detach()
                output["chunk_keyframe_recall"] = chunk_recall
                output["keyframe_recall"] = chunk_recall
            if bool(metric_predictions.any().item()):
                chunk_precision = target_mask[metric_predictions].float().mean().detach()
                output["chunk_keyframe_precision"] = chunk_precision
                output["keyframe_precision"] = chunk_precision
        if teacher_should_commit is not None and (metric_sample_mask is None or bool(metric_sample_mask.any().item())):
            metric_predicted_should_commit = predicted_should_commit
            metric_teacher_should_commit = teacher_should_commit
            metric_teacher_event_offset = teacher_event_offset
            metric_pred_event_offset = pred_event_offset
            if metric_sample_mask is not None:
                metric_predicted_should_commit = metric_predicted_should_commit[metric_sample_mask]
                metric_teacher_should_commit = metric_teacher_should_commit[metric_sample_mask]
                metric_teacher_event_offset = metric_teacher_event_offset[metric_sample_mask]
                metric_pred_event_offset = metric_pred_event_offset[metric_sample_mask]
            event_accuracy = (metric_predicted_should_commit == metric_teacher_should_commit).float().mean().detach()
            event_pred_rate = metric_predicted_should_commit.float().mean().detach()
            event_target_rate = metric_teacher_should_commit.float().mean().detach()
            output["event_commit_accuracy"] = event_accuracy
            output["event_commit_pred_rate"] = event_pred_rate
            output["event_commit_target_rate"] = event_target_rate
            if bool(metric_teacher_should_commit.any().item()):
                output["event_commit_recall"] = (
                    metric_predicted_should_commit[metric_teacher_should_commit].float().mean().detach()
                )
                valid_offsets = metric_teacher_should_commit & (metric_teacher_event_offset >= 0)
                if bool(valid_offsets.any().item()):
                    offset_error = (
                        metric_pred_event_offset[valid_offsets].float()
                        - metric_teacher_event_offset[valid_offsets].float()
                    ).abs()
                    output["event_offset_mae"] = offset_error.mean().detach()
            if bool(metric_predicted_should_commit.any().item()):
                output["event_commit_precision"] = (
                    metric_teacher_should_commit[metric_predicted_should_commit].float().mean().detach()
                )
        return output

    @torch.inference_mode()
    def predict_action(self, examples: List[dict] = None, **kwargs: str) -> np.ndarray:
        kwargs.pop("isolated_memory_bank", None)
        memory_update_only = bool(kwargs.pop("memory_update_only", False))
        if memory_update_only:
            return {
                "memory_updated": False,
                "message": "runtime image memory is handled by the eval interface or disabled for this mode",
            }
        kwargs.pop("force_memory_write_current", None)
        disable_regular_current_frame_keyframe_write = bool(
            kwargs.pop(
                "disable_regular_current_frame_keyframe_write",
                self.disable_current_frame_keyframe_write_in_eval,
            )
        )
        batch_images = [self._ensure_image_list(to_pil_preserve(example["image"])) for example in examples]
        batch_image_metas = self._infer_batch_image_metas(examples, batch_images)
        instructions = self._build_vla_prompts([example["lang"] for example in examples])

        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        qwen_images, qwen_image_metas = self._augment_batch_images_with_keyframe_images(
            batch_images=batch_images,
            batch_image_metas=batch_image_metas,
            examples=examples,
        )
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=qwen_images,
            instructions=instructions,
            image_metas=qwen_image_metas,
            use_image_role_text=self.use_image_role_text,
        )
        input_ids = qwen_inputs.get("input_ids", None)
        last_hidden = self._forward_qwen_without_memory(qwen_inputs)

        action_queries = self._gather_action_token_embeddings(
            last_hidden,
            input_ids,
            action_token_id=self.action_token_id,
        )
        keyframe_dtype = self._module_compute_dtype(self.keyframe_head, action_queries)
        action_dtype = self._module_compute_dtype(self.action_model, action_queries)
        keyframe_annotation_mask = self._extract_keyframe_annotation_mask(
            examples=examples,
            device=action_queries.device,
        )
        use_keyframe_predict_head = self._should_use_keyframe_predict_head(keyframe_annotation_mask)
        keyframe_device = action_queries.device
        if use_keyframe_predict_head:
            keyframe_queries = action_queries.to(dtype=keyframe_dtype)
            chunk_keyframe_logits = self.keyframe_head(keyframe_queries).squeeze(-1)
            keyframe_device = chunk_keyframe_logits.device
            keyframe_dtype = chunk_keyframe_logits.dtype
            chunk_keyframe_probs = torch.sigmoid(chunk_keyframe_logits)
            (
                pred_event_offset,
                pred_event_confidence,
                predicted_should_commit,
                raw_pred_event_offset,
                raw_pred_event_confidence,
                raw_predicted_should_commit,
                keyframe_event_suppressed_by_filter,
            ) = self._select_inference_chunk_event(
                chunk_keyframe_probs,
                examples=examples,
            )
            teacher_event_offset, teacher_event_confidence, teacher_should_commit = (
                self._extract_batch_teacher_event_targets(
                    examples=examples,
                    device=keyframe_device,
                    dtype=keyframe_dtype,
                )
            )
            if teacher_should_commit is not None and keyframe_annotation_mask is not None:
                teacher_should_commit = teacher_should_commit & keyframe_annotation_mask.to(
                    device=teacher_should_commit.device,
                    dtype=torch.bool,
                ).view(-1)
            disable_regular_current_frame_write = (
                self.enable_delayed_chunk_event_commit
                and disable_regular_current_frame_keyframe_write
            )
            memory_keyframe_mask, _ = self._resolve_memory_keyframe_mask(
                predicted_mask=predicted_should_commit,
                keyframe_targets=teacher_should_commit,
                force_memory_write_current=False,
                disable_regular_current_frame_write=disable_regular_current_frame_write,
            )
            if keyframe_annotation_mask is not None:
                memory_keyframe_mask = memory_keyframe_mask & keyframe_annotation_mask.to(
                    device=memory_keyframe_mask.device,
                    dtype=torch.bool,
                ).view(-1)
        else:
            (
                chunk_keyframe_probs,
                _,
                pred_event_offset,
                pred_event_confidence,
                predicted_should_commit,
            ) = self._empty_keyframe_predictions(
                batch_size=action_queries.shape[0],
                device=keyframe_device,
                dtype=action_queries.dtype,
            )
            raw_pred_event_offset = pred_event_offset
            raw_pred_event_confidence = pred_event_confidence
            raw_predicted_should_commit = predicted_should_commit
            keyframe_event_suppressed_by_filter = torch.zeros_like(predicted_should_commit, dtype=torch.bool)
            teacher_event_offset, teacher_event_confidence, teacher_should_commit = None, None, None
            memory_keyframe_mask = torch.zeros_like(predicted_should_commit, dtype=torch.bool)

        action_queries_for_head = action_queries.to(dtype=action_dtype)
        action_autocast_enabled = action_queries_for_head.is_cuda and action_dtype in (torch.float16, torch.bfloat16)
        if action_autocast_enabled:
            with torch.autocast("cuda", dtype=action_dtype):
                pred_actions = self.action_model.predict_action(action_queries_for_head)
        else:
            pred_actions = self.action_model.predict_action(action_queries_for_head)

        output_dict = {
            "normalized_actions": self._tensor_to_numpy(pred_actions),
            "chunk_keyframe_prob": self._tensor_to_numpy(chunk_keyframe_probs),
            "pred_event_offset": self._tensor_to_numpy(pred_event_offset),
            "pred_event_confidence": self._tensor_to_numpy(pred_event_confidence),
            "should_trigger_event": self._tensor_to_numpy(predicted_should_commit),
            "raw_pred_event_offset": self._tensor_to_numpy(raw_pred_event_offset),
            "raw_pred_event_confidence": self._tensor_to_numpy(raw_pred_event_confidence),
            "raw_should_trigger_event": self._tensor_to_numpy(raw_predicted_should_commit),
            "keyframe_event_suppressed_by_filter": self._tensor_to_numpy(keyframe_event_suppressed_by_filter),
            "keyframe_prob": self._tensor_to_numpy(pred_event_confidence),
            "predicted_is_keyframe": self._tensor_to_numpy(predicted_should_commit),
            "memory_is_keyframe": self._tensor_to_numpy(memory_keyframe_mask),
            "keyframe_head_enabled": float(use_keyframe_predict_head),
        }
        if teacher_event_offset is not None:
            output_dict["teacher_event_offset"] = self._tensor_to_numpy(teacher_event_offset)
        if teacher_event_confidence is not None:
            output_dict["teacher_event_confidence"] = self._tensor_to_numpy(teacher_event_confidence)
        if teacher_should_commit is not None:
            output_dict["teacher_should_commit"] = self._tensor_to_numpy(teacher_should_commit)
        return output_dict

    def _build_vla_prompts(self, instructions: List[str]) -> List[str]:
        action_tokens = self.action_token * self.chunk_len
        prompt_suffix = (
            f" Please predict the next {self.chunk_len} robot actions and estimate which future observation positions inside this chunk are key events: <action>{action_tokens}<action>."
        )
        return [instruction + prompt_suffix for instruction in instructions]

    def _gather_single_token_embedding(
        self,
        last_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        token_id: int,
        token_name: str,
    ) -> torch.Tensor:
        mask = input_ids == int(token_id)
        counts = mask.sum(dim=1)
        if (counts < 1).any():
            missing = (counts < 1).nonzero(as_tuple=False).flatten().tolist()
            raise RuntimeError(f"samples missing {token_name} token: {missing} | counts={counts.tolist()}")
        first_pos = mask.to(dtype=torch.long).argmax(dim=1)
        hidden_dim = last_hidden.shape[-1]
        gather_index = first_pos.view(-1, 1, 1).expand(-1, 1, hidden_dim)
        return last_hidden.gather(dim=1, index=gather_index).squeeze(1)

    def _compute_keyframe_loss(
        self,
        keyframe_logits: torch.Tensor,
        keyframe_targets: Optional[torch.Tensor],
        supervision_mask: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if keyframe_targets is None:
            return None
        if supervision_mask is not None:
            resolved_mask = supervision_mask.to(device=keyframe_logits.device, dtype=torch.bool).view(-1)
            if resolved_mask.shape[0] != keyframe_logits.shape[0]:
                raise ValueError(
                    f"keyframe supervision mask batch mismatch: mask={resolved_mask.shape[0]} logits={keyframe_logits.shape[0]}"
                )
            if not bool(resolved_mask.any().item()):
                return None
            keyframe_logits = keyframe_logits[resolved_mask]
            keyframe_targets = keyframe_targets[resolved_mask]
        pos_weight = torch.tensor(
            [self.keyframe_positive_weight],
            device=keyframe_logits.device,
            dtype=keyframe_logits.dtype,
        )
        return F.binary_cross_entropy_with_logits(
            keyframe_logits,
            keyframe_targets,
            pos_weight=pos_weight,
        )

    @staticmethod
    def _resolve_event_timesteps(
        base_timesteps: Optional[torch.Tensor],
        event_offsets: Optional[torch.Tensor],
        event_mask: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if base_timesteps is None or event_offsets is None:
            return None
        resolved = torch.full_like(base_timesteps, -1)
        offsets = event_offsets.to(device=base_timesteps.device, dtype=torch.long).view(-1)
        valid_mask = offsets >= 0
        if event_mask is not None:
            valid_mask = valid_mask & event_mask.to(device=base_timesteps.device, dtype=torch.bool).view(-1)
        if bool(valid_mask.any().item()):
            resolved[valid_mask] = base_timesteps[valid_mask] + offsets[valid_mask]
        return resolved

    def _select_chunk_event(
        self,
        chunk_keyframe_probs: torch.Tensor,
        threshold: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if chunk_keyframe_probs.ndim != 2:
            raise ValueError(
                f"chunk_keyframe_probs must be [B, T], got {tuple(chunk_keyframe_probs.shape)}"
            )
        batch_size, chunk_len = chunk_keyframe_probs.shape
        if chunk_len <= 0:
            return (
                torch.full((batch_size,), -1, device=chunk_keyframe_probs.device, dtype=torch.long),
                torch.zeros((batch_size,), device=chunk_keyframe_probs.device, dtype=chunk_keyframe_probs.dtype),
                torch.zeros((batch_size,), device=chunk_keyframe_probs.device, dtype=torch.bool),
            )

        event_threshold = self.event_commit_threshold if threshold is None else float(threshold)
        start_offset = min(max(int(self.event_future_min_offset), 0), chunk_len)
        if start_offset >= chunk_len:
            return (
                torch.full((batch_size,), -1, device=chunk_keyframe_probs.device, dtype=torch.long),
                torch.zeros((batch_size,), device=chunk_keyframe_probs.device, dtype=chunk_keyframe_probs.dtype),
                torch.zeros((batch_size,), device=chunk_keyframe_probs.device, dtype=torch.bool),
            )

        future_probs = chunk_keyframe_probs[:, start_offset:]
        confidence, rel_offset = future_probs.max(dim=1)
        event_offset = rel_offset + start_offset
        should_commit = confidence >= float(event_threshold)
        return event_offset.long(), confidence, should_commit

    def _select_chunk_event_candidates(
        self,
        chunk_keyframe_probs: torch.Tensor,
        threshold: Optional[float] = None,
        nms_window: Optional[int] = None,
    ) -> List[List[Tuple[int, float]]]:
        if chunk_keyframe_probs.ndim != 2:
            raise ValueError(
                f"chunk_keyframe_probs must be [B, T], got {tuple(chunk_keyframe_probs.shape)}"
            )

        batch_size, chunk_len = chunk_keyframe_probs.shape
        if chunk_len <= 0:
            return [[] for _ in range(batch_size)]

        event_threshold = self.event_commit_threshold if threshold is None else float(threshold)
        start_offset = min(max(int(self.event_future_min_offset), 0), chunk_len)
        if start_offset >= chunk_len:
            return [[] for _ in range(batch_size)]

        window = max(0, int(self.keyframe_nms_window if nms_window is None else nms_window))
        probs_cpu = chunk_keyframe_probs.detach().to(dtype=torch.float32).cpu()
        batch_candidates: List[List[Tuple[int, float]]] = []
        for batch_idx in range(batch_size):
            row = probs_cpu[batch_idx]
            raw_candidates = [
                (int(offset), float(row[offset].item()))
                for offset in range(start_offset, chunk_len)
                if float(row[offset].item()) >= float(event_threshold)
            ]
            raw_candidates.sort(key=lambda item: (-item[1], item[0]))
            if window <= 0:
                batch_candidates.append(raw_candidates)
                continue

            selected: List[Tuple[int, float]] = []
            suppressed = [False for _ in range(chunk_len)]
            for offset, confidence in raw_candidates:
                if suppressed[offset]:
                    continue
                selected.append((offset, confidence))
                begin = max(start_offset, int(offset) - window)
                end = min(chunk_len, int(offset) + window + 1)
                for suppressed_offset in range(begin, end):
                    suppressed[suppressed_offset] = True
            batch_candidates.append(selected)
        return batch_candidates

    @staticmethod
    def _is_within_timestep_window(
        step_a: Optional[int],
        step_b: Optional[int],
        window: int,
    ) -> bool:
        if step_a is None or step_b is None:
            return False
        return abs(int(step_a) - int(step_b)) <= max(0, int(window))

    def _refresh_inference_event_state(self, slot_idx: int, current_step: Optional[int]) -> dict:
        self._ensure_runtime_slots(slot_idx + 1)
        state = self._inference_event_state[slot_idx]
        pending_step = state.get("pending_step", None)
        if current_step is not None and pending_step is not None and int(current_step) >= int(pending_step):
            state["last_committed_step"] = int(pending_step)
            state["pending_step"] = None
            state["pending_confidence"] = 0.0
            state["pending_plan_start_step"] = None
            state["pending_offset"] = None
        return state

    def _select_inference_chunk_event(
        self,
        chunk_keyframe_probs: torch.Tensor,
        examples: Optional[List[dict]] = None,
        threshold: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_event_offset, raw_event_confidence, raw_should_commit = self._select_chunk_event(
            chunk_keyframe_probs,
            threshold=threshold,
        )
        suppressed_by_filter = torch.zeros_like(raw_should_commit, dtype=torch.bool)
        if not bool(getattr(self, "enable_keyframe_inference_event_filter", True)):
            return (
                raw_event_offset,
                raw_event_confidence,
                raw_should_commit,
                raw_event_offset,
                raw_event_confidence,
                raw_should_commit,
                suppressed_by_filter,
            )

        batch_size = int(chunk_keyframe_probs.shape[0])
        if examples:
            episode_ids = self._ensure_runtime_slots_for_examples(examples)
        else:
            episode_ids = [None for _ in range(batch_size)]
            self._ensure_runtime_slots(batch_size)

        candidates = self._select_chunk_event_candidates(
            chunk_keyframe_probs,
            threshold=threshold,
            nms_window=self.keyframe_nms_window,
        )
        event_offset = torch.full(
            (batch_size,),
            -1,
            device=chunk_keyframe_probs.device,
            dtype=torch.long,
        )
        event_confidence = torch.zeros(
            (batch_size,),
            device=chunk_keyframe_probs.device,
            dtype=chunk_keyframe_probs.dtype,
        )
        should_commit = torch.zeros(
            (batch_size,),
            device=chunk_keyframe_probs.device,
            dtype=torch.bool,
        )

        for slot_idx in range(batch_size):
            example = examples[slot_idx] if examples and slot_idx < len(examples) else {}
            current_step = self._example_int(example, "timestep", None) if isinstance(example, dict) else None
            state = self._refresh_inference_event_state(slot_idx, current_step)
            if slot_idx < len(episode_ids):
                state["episode_id"] = episode_ids[slot_idx]

            accepted = False
            for candidate_offset, candidate_confidence in candidates[slot_idx]:
                candidate_step = (
                    None
                    if current_step is None
                    else int(current_step) + int(candidate_offset)
                )
                if self._is_within_timestep_window(
                    candidate_step,
                    state.get("last_committed_step", None),
                    self.keyframe_cooldown_steps,
                ):
                    continue

                pending_step = state.get("pending_step", None)
                if self._is_within_timestep_window(
                    candidate_step,
                    pending_step,
                    self.keyframe_cooldown_steps,
                ):
                    pending_confidence = float(state.get("pending_confidence", 0.0))
                    keep_candidate = (
                        float(candidate_confidence) > pending_confidence
                        or (
                            abs(float(candidate_confidence) - pending_confidence) <= 1e-8
                            and candidate_step is not None
                            and pending_step is not None
                            and int(candidate_step) >= int(pending_step)
                        )
                    )
                    if not keep_candidate:
                        continue

                event_offset[slot_idx] = int(candidate_offset)
                event_confidence[slot_idx] = float(candidate_confidence)
                should_commit[slot_idx] = True
                accepted = True
                if candidate_step is not None:
                    state["pending_step"] = int(candidate_step)
                    state["pending_confidence"] = float(candidate_confidence)
                    state["pending_plan_start_step"] = int(current_step)
                    state["pending_offset"] = int(candidate_offset)
                break

            if slot_idx < raw_should_commit.numel() and bool(raw_should_commit[slot_idx].item()):
                filtered_offset = int(event_offset[slot_idx].item()) if accepted else -1
                raw_offset = int(raw_event_offset[slot_idx].item())
                if filtered_offset != raw_offset:
                    suppressed_by_filter[slot_idx] = True

        return (
            event_offset,
            event_confidence,
            should_commit,
            raw_event_offset,
            raw_event_confidence,
            raw_should_commit,
            suppressed_by_filter,
        )

    def _resolve_keyframe_schedule_steps(self) -> Tuple[int, int]:
        max_steps = max(int(self.keyframe_schedule_max_train_steps), 0)
        warmup_steps = int(self.keyframe_schedule_warmup_steps)
        transition_steps = int(self.keyframe_schedule_transition_steps)
        if warmup_steps < 0:
            warmup_steps = int(round(max_steps * 0.125)) if max_steps > 0 else 0
        if transition_steps < 0:
            transition_steps = int(round(max_steps * 0.375)) if max_steps > 0 else 0
        return max(warmup_steps, 0), max(transition_steps, 0)

    def _compute_current_keyframe_teacher_prob(self) -> Tuple[float, float]:
        schedule_type = self.keyframe_train_memory_schedule
        if schedule_type in {"predict", "student"}:
            return 0.0, 1.0
        if schedule_type in {"teacher", "gt"}:
            return 1.0, 0.0
        if schedule_type in {"teacher_to_predict", "teacher_then_predict", "schedule", "scheduled", "mixed"}:
            warmup_steps, transition_steps = self._resolve_keyframe_schedule_steps()
            completed_steps = max(int(self.keyframe_schedule_completed_steps), 0)
            if transition_steps <= 0:
                progress = 1.0 if completed_steps >= warmup_steps else 0.0
            else:
                progress = (completed_steps - warmup_steps) / float(transition_steps)
                progress = min(max(progress, 0.0), 1.0)
            teacher_prob = self.keyframe_schedule_teacher_prob_start + (
                self.keyframe_schedule_teacher_prob_end - self.keyframe_schedule_teacher_prob_start
            ) * progress
            teacher_prob = min(max(float(teacher_prob), 0.0), 1.0)
            return teacher_prob, progress
        return 1.0, 0.0

    def _resolve_memory_keyframe_mask(
        self,
        predicted_mask: torch.Tensor,
        keyframe_targets: Optional[torch.Tensor] = None,
        force_memory_write_current: bool = False,
        disable_regular_current_frame_write: bool = False,
    ) -> Tuple[torch.Tensor, dict]:
        teacher_mask = None
        if keyframe_targets is not None:
            teacher_mask = keyframe_targets.to(device=predicted_mask.device, dtype=torch.bool).view(-1)
        predicted_mask = predicted_mask.to(device=predicted_mask.device, dtype=torch.bool).view(-1)
        self._last_memory_keyframe_teacher_selector = None
        source = self.keyframe_train_memory_source if self.training else self.keyframe_eval_memory_source

        def _metric_tensor(value: float) -> torch.Tensor:
            return torch.tensor(float(value), device=predicted_mask.device, dtype=torch.float32)

        stats = {
            "keyframe_memory_teacher_prob": _metric_tensor(0.0).detach(),
            "keyframe_memory_teacher_usage": _metric_tensor(0.0).detach(),
            "keyframe_memory_predict_usage": _metric_tensor(1.0).detach(),
            "keyframe_memory_schedule_progress": _metric_tensor(self.keyframe_schedule_progress).detach(),
        }
        input_teacher_selector = self._last_keyframe_input_teacher_selector
        if input_teacher_selector is not None and input_teacher_selector.numel() == predicted_mask.numel():
            input_teacher_selector = input_teacher_selector.to(device=predicted_mask.device, dtype=torch.bool).view(-1)

        if force_memory_write_current:
            self.keyframe_memory_teacher_prob = 0.0
            self._last_memory_keyframe_teacher_selector = torch.zeros_like(predicted_mask, dtype=torch.bool)
            return torch.ones_like(predicted_mask, dtype=torch.bool), stats

        if disable_regular_current_frame_write:
            stats["keyframe_memory_predict_usage"] = _metric_tensor(0.0).detach()
            self._last_memory_keyframe_teacher_selector = torch.zeros_like(predicted_mask, dtype=torch.bool)
            return torch.zeros_like(predicted_mask, dtype=torch.bool), stats

        if source in {"teacher", "gt"}:
            self.keyframe_memory_teacher_prob = 1.0
            self.keyframe_schedule_progress = 0.0
            resolved_mask = teacher_mask if teacher_mask is not None else predicted_mask
            stats["keyframe_memory_teacher_prob"] = _metric_tensor(1.0).detach()
            stats["keyframe_memory_teacher_usage"] = _metric_tensor(1.0 if teacher_mask is not None else 0.0).detach()
            stats["keyframe_memory_predict_usage"] = _metric_tensor(0.0 if teacher_mask is not None else 1.0).detach()
            stats["keyframe_memory_schedule_progress"] = _metric_tensor(0.0).detach()
            if teacher_mask is None:
                self._last_memory_keyframe_teacher_selector = torch.zeros_like(predicted_mask, dtype=torch.bool)
            elif input_teacher_selector is not None:
                self._last_memory_keyframe_teacher_selector = input_teacher_selector
            else:
                self._last_memory_keyframe_teacher_selector = torch.ones_like(predicted_mask, dtype=torch.bool)
            return resolved_mask, stats

        if source in {"predict", "student"}:
            self.keyframe_memory_teacher_prob = 0.0
            self.keyframe_schedule_progress = 1.0
            stats["keyframe_memory_schedule_progress"] = _metric_tensor(1.0).detach()
            self._last_memory_keyframe_teacher_selector = torch.zeros_like(predicted_mask, dtype=torch.bool)
            return predicted_mask, stats

        if source in {"teacher_to_predict", "teacher_then_predict", "schedule", "scheduled", "mixed"}:
            teacher_prob, schedule_progress = self._compute_current_keyframe_teacher_prob()
            self.keyframe_memory_teacher_prob = teacher_prob
            self.keyframe_schedule_progress = schedule_progress
            stats["keyframe_memory_teacher_prob"] = _metric_tensor(teacher_prob).detach()
            stats["keyframe_memory_schedule_progress"] = _metric_tensor(schedule_progress).detach()
            if teacher_mask is None:
                self._last_memory_keyframe_teacher_selector = torch.zeros_like(predicted_mask, dtype=torch.bool)
                return predicted_mask, stats
            if input_teacher_selector is not None:
                teacher_selector = input_teacher_selector
            elif self.keyframe_schedule_mix_granularity == "batch":
                use_teacher = bool(torch.rand(1, device=predicted_mask.device).item() < teacher_prob)
                teacher_selector = torch.full_like(predicted_mask, use_teacher, dtype=torch.bool)
            else:
                teacher_selector = torch.rand(predicted_mask.shape, device=predicted_mask.device) < teacher_prob
            stats["keyframe_memory_teacher_usage"] = teacher_selector.float().mean().detach()
            stats["keyframe_memory_predict_usage"] = (1.0 - teacher_selector.float().mean()).detach()
            self._last_memory_keyframe_teacher_selector = teacher_selector.detach()
            return torch.where(teacher_selector, teacher_mask, predicted_mask), stats

        if source == "union":
            self.keyframe_memory_teacher_prob = 1.0 if teacher_mask is not None else 0.0
            self.keyframe_schedule_progress = 1.0
            stats["keyframe_memory_teacher_prob"] = _metric_tensor(self.keyframe_memory_teacher_prob).detach()
            stats["keyframe_memory_teacher_usage"] = _metric_tensor(1.0 if teacher_mask is not None else 0.0).detach()
            stats["keyframe_memory_predict_usage"] = _metric_tensor(1.0).detach()
            stats["keyframe_memory_schedule_progress"] = _metric_tensor(1.0).detach()
            if teacher_mask is None:
                self._last_memory_keyframe_teacher_selector = torch.zeros_like(predicted_mask, dtype=torch.bool)
                return predicted_mask, stats
            self._last_memory_keyframe_teacher_selector = torch.zeros_like(predicted_mask, dtype=torch.bool)
            return teacher_mask | predicted_mask, stats

        if source == "none":
            self.keyframe_memory_teacher_prob = 0.0
            self.keyframe_schedule_progress = 1.0
            stats["keyframe_memory_predict_usage"] = _metric_tensor(0.0).detach()
            stats["keyframe_memory_schedule_progress"] = _metric_tensor(1.0).detach()
            self._last_memory_keyframe_teacher_selector = torch.zeros_like(predicted_mask, dtype=torch.bool)
            return torch.zeros_like(predicted_mask, dtype=torch.bool), stats

        raise ValueError(f"Unsupported keyframe memory source: {source}")

    def _gather_action_token_embeddings(
        self,
        last_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        action_token_id=None,
    ) -> torch.Tensor:
        if action_token_id is None:
            raise ValueError("action_token_id cannot be None")

        device = input_ids.device
        batch_size, seq_len, hidden_dim = last_hidden.shape

        if isinstance(action_token_id, (list, tuple, set)):
            id_list = torch.tensor(list(action_token_id), device=device, dtype=input_ids.dtype)
            mask = torch.isin(input_ids, id_list)
        else:
            mask = input_ids == action_token_id

        counts = mask.sum(dim=1)
        if (counts < self.chunk_len).any():
            insufficient = (counts < self.chunk_len).nonzero(as_tuple=False).flatten().tolist()
            raise RuntimeError(
                f"samples with insufficient action tokens (<{self.chunk_len}): "
                f"{insufficient} | counts={counts.tolist()}"
            )

        idx = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        masked_pos = torch.where(mask, idx, torch.full_like(idx, -1))
        topk_pos = masked_pos.topk(k=self.chunk_len, dim=-1).values
        selected_pos = topk_pos.sort(dim=-1).values

        expanded_index = selected_pos.unsqueeze(-1).expand(-1, -1, hidden_dim)
        action_queries = last_hidden.gather(dim=1, index=expanded_index)
        return action_queries

    def _extract_batch_keyframe_targets(
        self,
        examples: List[dict],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if examples is None or len(examples) == 0:
            return None

        targets = []
        for example in examples:
            chunk_target = example.get("chunk_keyframe_target", None)
            if chunk_target is None:
                annotated_is_keyframe = example.get("is_keyframe_proxy", example.get("is_keyframe", None))
                if annotated_is_keyframe is None:
                    return None
                target_tensor = torch.zeros((self.chunk_len,), device=device, dtype=dtype)
                target_tensor[0] = float(bool(annotated_is_keyframe))
            else:
                target_tensor = torch.as_tensor(chunk_target, device=device, dtype=dtype).flatten()
                if target_tensor.numel() < self.chunk_len:
                    padded = torch.zeros((self.chunk_len,), device=device, dtype=dtype)
                    padded[: target_tensor.numel()] = target_tensor
                    target_tensor = padded
                elif target_tensor.numel() > self.chunk_len:
                    target_tensor = target_tensor[: self.chunk_len]
            targets.append(target_tensor)
        return torch.stack(targets, dim=0)

    def _extract_batch_teacher_event_targets(
        self,
        examples: Optional[List[dict]],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if examples is None or len(examples) == 0:
            return None, None, None

        offsets = []
        confidences = []
        should_commit = []
        for example in examples:
            offset = example.get("teacher_event_offset", None)
            confidence = example.get("teacher_event_confidence", None)
            should = example.get("teacher_should_commit", None)
            if offset is None or confidence is None or should is None:
                chunk_target = example.get("chunk_keyframe_target", None)
                if chunk_target is None:
                    return None, None, None
                target_tensor = torch.as_tensor(chunk_target, device=device, dtype=dtype).flatten()
                if target_tensor.numel() < self.chunk_len:
                    padded = torch.zeros((self.chunk_len,), device=device, dtype=dtype)
                    padded[: target_tensor.numel()] = target_tensor
                    target_tensor = padded
                elif target_tensor.numel() > self.chunk_len:
                    target_tensor = target_tensor[: self.chunk_len]
                pred_offset, pred_confidence, pred_should = self._select_chunk_event(
                    target_tensor.unsqueeze(0),
                    threshold=self.event_commit_threshold,
                )
                offsets.append(int(pred_offset[0].item()))
                confidences.append(float(pred_confidence[0].item()))
                should_commit.append(bool(pred_should[0].item()))
                continue

            if hasattr(offset, "item"):
                offset = offset.item()
            if hasattr(confidence, "item"):
                confidence = confidence.item()
            if hasattr(should, "item"):
                should = should.item()
            offsets.append(int(offset))
            confidences.append(float(confidence))
            should_commit.append(bool(should))

        return (
            torch.tensor(offsets, device=device, dtype=torch.long),
            torch.tensor(confidences, device=device, dtype=dtype),
            torch.tensor(should_commit, device=device, dtype=torch.bool),
        )

    @staticmethod
    def _extract_batch_timesteps(examples: List[dict], device: torch.device) -> Optional[torch.Tensor]:
        if examples is None or len(examples) == 0:
            return None
        timesteps: List[int] = []
        for example in examples:
            timestep = example.get("timestep", None)
            if timestep is None:
                return None
            if hasattr(timestep, "item"):
                timestep = timestep.item()
            timesteps.append(int(timestep))
        return torch.tensor(timesteps, device=device, dtype=torch.long)


if __name__ == "__main__":
    from omegaconf import OmegaConf
    import argparse
    import debugpy

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="./examples/RoboTwin-Mem/train_files/eventvla_robotwin_mem.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    debugpy.listen(("0.0.0.0", 10092))
    print("Rank 0 waiting for debugger attach on port 10092...")
    debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)
    cfg.framework.action_model.action_hidden_dim = 2048
    cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Florence-2-large"

    model = EventVLA(cfg)
    print(model)

    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image": [image],
        "lang": "This is a fake instruction for testing.",
    }

    sample2 = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image": [image],
        "lang": "A second fake instruction for batch testing.",
    }

    batch = [sample, sample2]

    model.train()
    output = model(batch)
    print(output)

    model.eval()
    action = model.predict_action(batch)
    print(action)
