"""CogWAM: a VLM planner driving a dual-stream world/action mixture-of-transformers.

The vision-language backbone consumes the current observation plus an
event-driven semantic memory, and emits two banks of learned queries --
``<WORLD_PLAN>`` and ``<ACTION_PLAN>``.  Those plans condition a 30-layer MoT
that jointly denoises a future DINO latent and a 25-step action chunk under one
flow-matching objective.  The semantic state itself is supervised as
next-token prediction over an explicit ``<KEEP>`` / ``<UPDATE>`` decision, so
the planner only rewrites its memory at task-phase boundaries.

See the Model section of README.md for the architecture walk-through.
"""

from __future__ import annotations

import json
import re
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from cogwam.data.composite import to_pil_preserve
from cogwam.data.event_memory import (
    KEEP_DECISION,
    UPDATE_DECISION,
)
from cogwam.models.base import CogWAMBase
from cogwam.models.dino_v3 import (
    DINOv3Backbone,
    dino_patch_grid,
    normalize_dino_image_size,
    resolve_dino_spec,
)
from cogwam.models.vlm_interface import build_vlm
from cogwam.models.world_action_mot import CausalDINOActionMoT
from cogwam.recipe import RECIPE_PROFILE, validate_recipe
from cogwam.training.config import merge_framework_config


def _qwen_enable_thinking(config) -> bool:
    framework = getattr(config, "framework", None) if config is not None else None
    qwenvl = framework.get("qwenvl", {}) if framework is not None else {}
    return bool(qwenvl.get("enable_thinking", False))


def _get_nested_attribute(root, path: str):
    value = root
    for name in path.split("."):
        value = getattr(value, name, None)
        if value is None:
            return None
    return value


def _truncate_text_backbone(full_model: nn.Module, keep_layers: int) -> int:
    """Keep the first N decoder layers and align every exposed HF config.

    Qwen3-VL and RynnBrain wrappers expose the text decoder through slightly
    different nesting.  Resolve the concrete ModuleList instead of relying on
    a model-name check so new checkpoints fail early rather than silently
    running the full VLM.
    """

    if int(keep_layers) <= 0:
        raise ValueError("truncate_vlm_layers must be a positive integer")
    candidate_paths = (
        "model.language_model.layers",
        "language_model.layers",
        "model.layers",
        "layers",
    )
    layers = None
    resolved_path = None
    for path in candidate_paths:
        candidate = _get_nested_attribute(full_model, path)
        if isinstance(candidate, nn.ModuleList):
            layers = candidate
            resolved_path = path
            break
    if layers is None:
        raise RuntimeError(
            "Could not locate the VLM text decoder layers for first-N truncation; "
            f"checked {candidate_paths}"
        )
    original_layers = len(layers)
    if int(keep_layers) > original_layers:
        raise ValueError(
            f"truncate_vlm_layers={keep_layers} exceeds decoder depth={original_layers}"
        )
    while len(layers) > int(keep_layers):
        del layers[-1]

    config_candidates = [
        getattr(full_model, "config", None),
        getattr(getattr(full_model, "config", None), "text_config", None),
        getattr(_get_nested_attribute(full_model, "model"), "config", None),
        getattr(_get_nested_attribute(full_model, "model.language_model"), "config", None),
        getattr(_get_nested_attribute(full_model, "language_model"), "config", None),
    ]
    for config in config_candidates:
        if config is not None and hasattr(config, "num_hidden_layers"):
            config.num_hidden_layers = int(keep_layers)
    if len(layers) != int(keep_layers):
        raise RuntimeError(
            f"Failed to truncate VLM via {resolved_path}: got={len(layers)}, "
            f"expected={keep_layers}"
        )
    return original_layers


@dataclass
class CogWAMDefaultConfig:
    name: str = "CogWAM"
    enable_world_action_mot: bool = False
    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": "./playground/Pretrained_models/Qwen3-VL-2B-Instruct",
            "attn_implementation": "flash_attention_2",
            "require_attn_implementation": True,
            "enable_gradient_checkpointing": True,
            "vl_hidden_dim": 2048,
        }
    )
    dino: dict = field(
        default_factory=lambda: {
            "name": "dinov3_vitb16",
            "model_size": "base",
            "hf_model_id": "facebook/dinov3-vitb16-pretrain-lvd1689m",
            "repo_or_dir": "facebookresearch/dinov3",
            "weights": None,
            "loader": "auto",
            "image_size": [384, 320],
            # Optional future-target resolution. None reuses image_size.
            "future_image_size": None,
            "patch_size": 16,
            "embed_dim": 768,
            "stats_path": None,
            "load_live_backbone": False,
            "force_online": True,
            "dino_pool": 2,
            # Optional full-resolution clean-current prefix. Future/world
            # targets continue to use dino_pool. None preserves historical
            # checkpoints, where current and future both use dino_pool.
            "current_dino_pool": None,
            # Optional multi-layer DINO fusion. Applies to both the clean
            # current prefix and the future denoising target so they share one
            # feature space. None keeps the historical single last-layer path.
            "dino_layers": None,
        }
    )
    action_model: dict = field(
        default_factory=lambda: {
            "action_dim": 14,
            "state_dim": 0,
            "action_horizon": 16,
        }
    )
    planner: dict = field(
        default_factory=lambda: {
            "num_world_queries": 16,
            "num_action_queries": 16,
            "world_placeholder_token": "<WORLD_PLAN>",
            "action_placeholder_token": "<ACTION_PLAN>",
            "text_supervision": {
                "enabled": False,
                "mode": "legacy_full_text",
                "subtask_field": "subtask_text",
                "completed_subtask_field": "completed_subtask_text",
                "prompt_template": (
                    "{instruction}\nReport the current subtask and the completed subtask."
                ),
                "response_template": (
                    "Current subtask: {subtask_text}\n"
                    "Completed subtask: {completed_subtask_text}"
                ),
                "max_new_tokens": 64,
                "do_sample": False,
                "history": {
                    "enabled": False,
                    "image_field": "planner_history_images",
                    "finished_task_list_field": "finished_task_list",
                    "finished_task_list_prefix": "Finished Task List:",
                    "empty_finished_task_list": "None",
                    "history_image_size": [160, 192],
                },
            },
        }
    )
    world_action_mot: dict = field(
        default_factory=lambda: {
            "hidden_size": 1024,
            "num_layers": 12,
            "num_attention_heads": 16,
            "attention_pattern": "alternating_condition_joint",
            "action_mlp_ratio": 4.0,
            "world_mlp_ratio": 4.0,
            "dropout": 0.1,
            "time_frequency_dim": 256,
            "max_world_tokens": 512,
            # num_world_views is the denoising-target view count. Current
            # observations may contain additional policy views (for example,
            # official LIBERO uses primary + wrist but predicts primary only).
            "num_world_views": 1,
            "num_current_world_views": None,
            "enable_gradient_checkpointing": True,
            "num_inference_timesteps": 10,
            "repeated_diffusion_steps": 1,
            "action_prediction_type": "velocity",
            "action_velocity_target": "noise_minus_clean",
            "jit_t_eps": 0.05,
            "flow_time_sampling": "uniform",
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            "action_loss_weight": 1.0,
            "world_loss_weight": 0.1,
            "text_loss_weight": 0.0,
        }
    )


class LearnedQueryBank(nn.Module):
    def __init__(self, count: int, hidden_size: int) -> None:
        super().__init__()
        if count <= 0:
            raise ValueError("learned-query count must be positive")
        self.count = int(count)
        self.embedding = nn.Parameter(torch.empty(1, self.count, int(hidden_size)))
        nn.init.normal_(self.embedding, std=0.02)

    def forward(self, batch_size: int, *, device) -> torch.Tensor:
        return self.embedding.to(device=device).expand(int(batch_size), -1, -1)


class CogWAM(CogWAMBase):
    """VLM learned-query encoder and short-range dual-expert physical model.

    Text/subtask planning is an optional, separate supervision path.  Base
    checkpoints keep it disabled while still using WORLD/ACTION learned query
    suffixes as physical conditioning.
    """

    planner_query_mask_contract = "positional_suffix_v2"

    def __init__(self, config: Optional[dict] = None, **_kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(CogWAMDefaultConfig, config)
        framework = self.config.framework
        reproduction_profile = str(
            framework.get("reproduction_profile", "") or ""
        )
        if reproduction_profile != RECIPE_PROFILE:
            raise ValueError(
                "framework.reproduction_profile must be "
                f"{RECIPE_PROFILE!r}, got {reproduction_profile!r}"
            )
        # Validate the frozen recipe before allocating multi-billion-parameter
        # backbones, so a config drift costs seconds instead of a GPU-hour.
        validate_recipe(self.config)
        if not bool(framework.get("enable_world_action_mot", False)):
            raise ValueError(
                "CogWAM requires framework.enable_world_action_mot=true"
            )
        incompatible = []
        if bool(framework.get("wam", {}).get("enabled", False)):
            incompatible.append("wam")
        if bool(framework.get("jointflow", {}).get("enabled", False)):
            incompatible.append("jointflow")
        if bool(framework.get("enable_action_world_coflow", False)):
            incompatible.append("action_world_coflow")
        if incompatible:
            raise ValueError(
                "World--Action MoT is a standalone framework; disable " + ", ".join(incompatible)
            )

        data_cfg = self.config.datasets.vla_data
        if str(data_cfg.get("dataset_py", "")) != "jointflow":
            raise ValueError("CogWAM currently requires dataset_py=jointflow")
        if not bool(data_cfg.get("online_dino", False)):
            raise ValueError("CogWAM requires online_dino=true")
        self.include_state = bool(data_cfg.get("include_state", False))
        configured_state_dim = int(framework.action_model.get("state_dim", 0) or 0)
        if self.include_state != (configured_state_dim > 0):
            raise ValueError(
                "CogWAM state contract requires include_state=true exactly when "
                f"action_model.state_dim > 0, got include_state={self.include_state} "
                f"and state_dim={configured_state_dim}"
            )
        if not bool(data_cfg.get("decode_future_video", True)):
            raise ValueError("CogWAM requires decode_future_video=true")

        self.qwen_vl_interface = build_vlm(config=self.config)
        full_model = self.qwen_vl_interface.model
        mot_cfg = framework.world_action_mot
        self.layerwise_planner_coupling = bool(
            mot_cfg.get("layerwise_planner_coupling", False)
        )
        truncate_vlm_layers = int(
            framework.qwenvl.get("truncate_vlm_layers", 0) or 0
        )
        physical_layers = int(mot_cfg.get("num_layers", 0) or 0)
        if self.layerwise_planner_coupling:
            if truncate_vlm_layers != physical_layers:
                raise ValueError(
                    "Layer-wise planner coupling requires one truncated VLM layer "
                    "per physical layer: "
                    f"truncate_vlm_layers={truncate_vlm_layers}, "
                    f"physical_layers={physical_layers}"
                )
            self.original_vlm_layers = _truncate_text_backbone(
                full_model,
                truncate_vlm_layers,
            )
        elif truncate_vlm_layers:
            raise ValueError(
                "truncate_vlm_layers is only supported with "
                "world_action_mot.layerwise_planner_coupling=true"
            )
        else:
            self.original_vlm_layers = None
        text_config = getattr(full_model.config, "text_config", full_model.config)
        planner_dim = int(text_config.hidden_size)
        planner_cfg = framework.planner
        self.text_supervision = dict(planner_cfg.get("text_supervision", {}))
        self.event_memory_enabled = bool(
            self.text_supervision.get("enabled", False)
            and str(self.text_supervision.get("mode", "legacy_full_text")).lower()
            == "event_driven_memory_ntp"
        )
        self.num_world_queries = int(planner_cfg.get("num_world_queries", 16))
        self.num_action_queries = int(planner_cfg.get("num_action_queries", 16))
        self.action_horizon = int(framework.action_model.get("action_horizon", 16))
        self.action_dim = int(framework.action_model.get("action_dim", 14))
        if self.num_action_queries != self.action_horizon:
            raise ValueError(
                "ACTION-PLAN queries must align one-to-one with the action chunk: "
                f"queries={self.num_action_queries}, horizon={self.action_horizon}"
            )

        tokenizer = self.qwen_vl_interface.processor.tokenizer
        self.world_placeholder = str(
            planner_cfg.get("world_placeholder_token", "<WORLD_PLAN>")
        )
        self.action_placeholder = str(
            planner_cfg.get("action_placeholder_token", "<ACTION_PLAN>")
        )
        additional_special_tokens = [
            self.world_placeholder,
            self.action_placeholder,
        ]
        if getattr(self, "event_memory_enabled", False):
            self.keep_token = str(self.text_supervision.get("keep_token", "<KEEP>"))
            self.update_token = str(
                self.text_supervision.get("update_token", "<UPDATE>")
            )
            additional_special_tokens.extend([self.keep_token, self.update_token])
        tokenizer.add_special_tokens(
            {
                "additional_special_tokens": additional_special_tokens
            }
        )
        self.world_placeholder_id = int(
            tokenizer.convert_tokens_to_ids(self.world_placeholder)
        )
        self.action_placeholder_id = int(
            tokenizer.convert_tokens_to_ids(self.action_placeholder)
        )
        if self.event_memory_enabled:
            self.keep_token_id = int(tokenizer.convert_tokens_to_ids(self.keep_token))
            self.update_token_id = int(
                tokenizer.convert_tokens_to_ids(self.update_token)
            )
            event_ids = {
                self.world_placeholder_id,
                self.action_placeholder_id,
                self.keep_token_id,
                self.update_token_id,
            }
            if len(event_ids) != 4:
                raise ValueError(
                    "Event-memory WORLD/ACTION/KEEP/UPDATE tokens must be distinct"
                )
            for token, token_id in (
                (self.keep_token, self.keep_token_id),
                (self.update_token, self.update_token_id),
            ):
                encoded = tokenizer.encode(token, add_special_tokens=False)
                if encoded != [token_id]:
                    raise ValueError(
                        f"Event decision token must encode as one token: {token!r} -> {encoded}"
                    )
        full_model.resize_token_embeddings(len(tokenizer))
        self.world_plan_queries = LearnedQueryBank(self.num_world_queries, planner_dim)
        self.action_plan_queries = LearnedQueryBank(self.num_action_queries, planner_dim)

        self._dino_spec = resolve_dino_spec(framework.dino)
        self.dino_dim = int(self._dino_spec["embed_dim"])
        self.dino_pool = int(framework.dino.get("dino_pool", 1))
        configured_future_image_size = framework.dino.get(
            "future_image_size",
            None,
        )
        self.future_dino_image_size = normalize_dino_image_size(
            self._dino_spec["image_size"]
            if configured_future_image_size is None
            else configured_future_image_size
        )
        configured_current_dino_pool = framework.dino.get(
            "current_dino_pool",
            None,
        )
        self.current_dino_pool = (
            None
            if configured_current_dino_pool is None
            else int(configured_current_dino_pool)
        )
        configured_dino_layers = framework.dino.get("dino_layers", None)
        self.dino_layers = (
            None
            if configured_dino_layers is None
            else tuple(int(index) for index in configured_dino_layers)
        )
        if self.dino_layers is not None and (
            not self.dino_layers or any(index < 0 for index in self.dino_layers)
        ):
            raise ValueError(
                "framework.dino.dino_layers must be a non-empty list of "
                f"non-negative layer indices, got {configured_dino_layers!r}"
            )
        current_rows, current_columns = dino_patch_grid(
            self._dino_spec["image_size"], self._dino_spec["patch_size"]
        )
        future_rows, future_columns = dino_patch_grid(
            self.future_dino_image_size,
            self._dino_spec["patch_size"],
        )
        if (
            self.dino_pool <= 0
            or future_rows % self.dino_pool
            or future_columns % self.dino_pool
        ):
            raise ValueError(
                "dino_pool="
                f"{self.dino_pool} must divide future DINO grid "
                f"{(future_rows, future_columns)}"
            )
        current_pool = (
            self.dino_pool
            if self.current_dino_pool is None
            else self.current_dino_pool
        )
        if (
            current_pool <= 0
            or current_rows % current_pool
            or current_columns % current_pool
        ):
            raise ValueError(
                "current DINO pool="
                f"{current_pool} must divide current DINO grid "
                f"{(current_rows, current_columns)}"
            )
        if self.current_dino_pool is not None:
            if self.current_dino_pool >= self.dino_pool:
                raise ValueError(
                    "current_dino_pool must be smaller than dino_pool so the "
                    "clean current prefix is genuinely higher resolution than "
                    "the future denoising target; got "
                    f"current={self.current_dino_pool}, future={self.dino_pool}"
                )
        self.num_world_views = int(
            framework.world_action_mot.get("num_world_views", 1)
        )
        if self.num_world_views <= 0:
            raise ValueError(
                f"world_action_mot.num_world_views must be positive, got {self.num_world_views}"
            )
        configured_current_views = framework.world_action_mot.get(
            "num_current_world_views",
            None,
        )
        self.num_current_world_views = (
            self.num_world_views
            if configured_current_views is None
            else int(configured_current_views)
        )
        if self.num_current_world_views <= 0:
            raise ValueError(
                "world_action_mot.num_current_world_views must be positive, "
                f"got {self.num_current_world_views}"
            )
        pooled_rows = future_rows // self.dino_pool
        pooled_columns = future_columns // self.dino_pool
        pooled_tokens = self.num_world_views * pooled_rows * pooled_columns
        resolved_current_grid = (
            self.num_current_world_views * (current_rows // current_pool),
            current_columns // current_pool,
        )
        resolved_future_grid = (
            self.num_world_views * pooled_rows,
            pooled_columns,
        )
        current_grid = (
            None
            if resolved_current_grid == resolved_future_grid
            else resolved_current_grid
        )
        max_world_tokens = int(framework.world_action_mot.get("max_world_tokens", 512))
        if pooled_tokens > max_world_tokens:
            raise ValueError(
                f"pooled DINO tokens={pooled_tokens} exceeds max_world_tokens={max_world_tokens}"
            )
        self.register_buffer("_dino_mean", torch.zeros(self.dino_dim), persistent=False)
        self.register_buffer("_dino_std", torch.ones(self.dino_dim), persistent=False)
        self._load_dino_stats(framework.dino.get("stats_path", None))
        object.__setattr__(self, "_dino_teacher", None)
        if bool(framework.dino.get("load_live_backbone", False)):
            self._set_dino_teacher(DINOv3Backbone(**self._dino_spec))

        # The attribute name `action_model` is load-bearing: it becomes the
        # top-level checkpoint key prefix for the whole physical model, and the
        # optimizer's per-group learning rates are keyed off it.
        physical_architecture = str(
            framework.world_action_mot.get("architecture", "")
        ).lower()
        if physical_architecture != "causal_dino_mot":
            raise ValueError(
                "framework.world_action_mot.architecture must be "
                f"'causal_dino_mot', got {physical_architecture!r}"
            )
        configured_world_tokens = int(
            framework.world_action_mot.get("world_grid_height", 12)
        ) * int(framework.world_action_mot.get("world_grid_width", 10))
        if pooled_tokens != configured_world_tokens:
            raise ValueError(
                "pooled online-DINO token count must equal the configured "
                "physical world grid: "
                f"pooled={pooled_tokens}, configured={configured_world_tokens}"
            )
        self.action_model = CausalDINOActionMoT(
            planner_dim=planner_dim,
            world_dim=self.dino_dim,
            action_config=framework.action_model,
            mot_config=framework.world_action_mot,
            current_world_grid=current_grid,
        )
        planner_dtype = next(full_model.parameters()).dtype
        self.action_model.to(dtype=planner_dtype)
        self.text_loss_weight = float(
            framework.world_action_mot.get("text_loss_weight", 0.0)
        )
        self.text_planning_enabled = bool(
            self.text_supervision.get("enabled", False)
        )
        self.text_history = dict(self.text_supervision.get("history", {}))
        history_requested = bool(self.text_history.get("enabled", False))
        # Image history is also a valid standalone physical-policy input for
        # the no-text MEM ablation.  The legacy configuration location remains
        # nested under text_supervision.history so the text+MEM recipe and its
        # checkpoints keep the same schema.
        self.text_history_enabled = history_requested
        if self.text_planning_enabled != (self.text_loss_weight > 0):
            raise ValueError(
                "Unified text planning requires planner.text_supervision.enabled and "
                "world_action_mot.text_loss_weight > 0 to be enabled or disabled together. "
                "Teacher-forcing text into WORLD/ACTION queries without NTP supervision would "
                "break train/inference consistency."
            )
        annotations_enabled = bool(
            data_cfg.get("text_annotations", {}).get("enabled", False)
        )
        if self.text_planning_enabled and not annotations_enabled:
            raise ValueError(
                "Unified text planning requires "
                "datasets.vla_data.text_annotations.enabled=true"
            )
        data_history = dict(
            data_cfg.get("text_annotations", {}).get("history", {})
        )
        self.event_data_config = dict(
            data_cfg.get("text_annotations", {}).get("event_memory", {})
        )
        data_event_enabled = bool(self.event_data_config.get("enabled", False))
        if self.event_memory_enabled != data_event_enabled:
            raise ValueError(
                "Planner and dataset event-memory modes must be enabled or disabled "
                "together"
            )
        if self.event_memory_enabled:
            if history_requested or bool(data_history.get("enabled", False)):
                raise ValueError(
                    "Event-driven semantic memory is the no-visual-history path; "
                    "planner/dataset history must be disabled"
                )
            required_templates = (
                "prompt_template",
                "update_response_template",
            )
            missing_templates = [
                key for key in required_templates if not self.text_supervision.get(key)
            ]
            if missing_templates:
                raise ValueError(
                    "Event-memory text supervision is missing templates: "
                    f"{missing_templates}"
                )
            schedule = dict(self.text_supervision.get("scheduled_sampling", {}))
            if not bool(schedule.get("enabled", False)):
                raise ValueError(
                    "Event-memory training requires explicit plan-level "
                    "scheduled_sampling.enabled=true"
                )
            points = schedule.get(
                "points", [[0, 0.0], [30000, 0.0], [40000, 0.2], [50000, 0.5]]
            )
            self.event_schedule_points = sorted(
                (int(step), float(probability)) for step, probability in points
            )
            self.event_max_new_tokens = int(
                self.text_supervision.get("max_new_tokens", 96)
            )
            if self.event_max_new_tokens <= 0:
                raise ValueError("Event-memory max_new_tokens must be positive")
            if bool(self.text_supervision.get("do_sample", False)):
                raise ValueError(
                    "Event-memory condition generation is constrained greedy; "
                    "text_supervision.do_sample must be false"
                )
            if (
                not self.event_schedule_points
                or self.event_schedule_points[0][0] != 0
                or any(step < 0 or not 0.0 <= probability <= 1.0 for step, probability in self.event_schedule_points)
                or any(
                    right[0] <= left[0]
                    for left, right in zip(
                        self.event_schedule_points,
                        self.event_schedule_points[1:],
                    )
                )
            ):
                raise ValueError(
                    "scheduled_sampling.points must be strictly increasing "
                    "[step, probability] pairs beginning at step 0"
                )
            self.event_semantic_fields = {
                "memory": str(
                    self.event_data_config.get(
                        "semantic_memory_field", "semantic_memory"
                    )
                ),
                "cached_subtask": str(
                    self.event_data_config.get(
                        "cached_subtask_field", "cached_current_subtask"
                    )
                ),
                "decision": str(
                    self.event_data_config.get(
                        "decision_field", "semantic_decision"
                    )
                ),
                "memory_add": str(
                    self.event_data_config.get("memory_add_field", "memory_add")
                ),
                "cache_valid": str(
                    self.event_data_config.get(
                        "cache_valid_field", "semantic_cache_valid"
                    )
                ),
            }
        if self.text_history_enabled != bool(data_history.get("enabled", False)):
            raise ValueError(
                "Planner and dataset history must be enabled or disabled together: "
                "framework.planner.text_supervision.history.enabled must match "
                "datasets.vla_data.text_annotations.history.enabled"
            )
        self.text_history_frame_offsets = [
            int(offset) for offset in data_history.get("frame_offsets", [])
        ]
        self.text_history_memory_offset = int(
            data_history.get("memory_offset", 0)
        )
        self.mem_vision_encoder = dict(
            framework.qwenvl.get("mem_vision_encoder", {}) or {}
        )
        self.mem_vision_encoder_enabled = bool(
            self.mem_vision_encoder.get("enabled", False)
        )
        if self.event_memory_enabled and self.mem_vision_encoder_enabled:
            raise ValueError(
                "Event-driven semantic memory is the no-RGB-history path and "
                "cannot enable qwenvl.mem_vision_encoder"
            )
        # Where the planner/VLM takes its current pixels from.  "composite"
        # (default) keeps the historical behaviour of reusing the same stitched
        # image the DINO branch consumes.  "separate_views" routes the VLM to a
        # parallel per-camera stream so the stitched layout -- which the VLM was
        # never pretrained on -- stops being the only thing it sees.  The DINO
        # branch is unaffected either way.
        self.vlm_view_source = str(
            framework.qwenvl.get("vlm_view_source", "composite")
        ).lower()
        if self.vlm_view_source not in ("composite", "separate_views"):
            raise ValueError(
                "framework.qwenvl.vlm_view_source must be 'composite' or "
                f"'separate_views', got {self.vlm_view_source!r}"
            )
        configured_vlm_view_size = self.config.datasets.vla_data.get(
            "vlm_view_size",
            None,
        )
        self.vlm_view_size = (
            tuple(int(value) for value in configured_vlm_view_size)
            if configured_vlm_view_size
            else None
        )
        if self.vlm_view_source == "separate_views":
            if self.vlm_view_size is None:
                raise ValueError(
                    "vlm_view_source='separate_views' requires "
                    "datasets.vla_data.vlm_view_size (width, height)"
                )
            if self.mem_vision_encoder_enabled:
                raise ValueError(
                    "Rynn MEM consumes one composite current image and cannot "
                    "be combined with vlm_view_source='separate_views'"
                )
        if self.text_history_enabled:
            if not self.text_history_frame_offsets:
                raise ValueError(
                    "Text-history planning requires dataset history.frame_offsets"
                )
            if any(offset >= 0 for offset in self.text_history_frame_offsets):
                raise ValueError(
                    "Text-history frame offsets must be negative, got "
                    f"{self.text_history_frame_offsets}"
                )
            if self.text_history_memory_offset >= 0:
                raise ValueError(
                    "Text-history memory_offset must be negative, got "
                    f"{self.text_history_memory_offset}"
                )
            interface_pairs = (
                (
                    "image_field",
                    "planner_history_images",
                ),
                (
                    "finished_task_list_field",
                    "finished_task_list",
                ),
                (
                    "empty_finished_task_list",
                    "None",
                ),
            )
            mismatches = {}
            for key, default in interface_pairs:
                planner_value = str(self.text_history.get(key, default))
                dataset_value = str(data_history.get(key, default))
                if planner_value != dataset_value:
                    mismatches[key] = {
                        "planner": planner_value,
                        "dataset": dataset_value,
                    }
            if mismatches:
                raise ValueError(
                    "Planner/dataset text-history interface mismatch: "
                    f"{mismatches}"
                )
        if self.mem_vision_encoder_enabled:
            if not self.text_history_enabled:
                raise ValueError(
                    "Rynn MEM vision encoding requires planner image history"
                )
            configured_frames = int(
                self.mem_vision_encoder.get("num_frames", 6)
            )
            actual_frames = len(self.text_history_frame_offsets) + 1
            if configured_frames != actual_frames:
                raise ValueError(
                    "Rynn MEM num_frames must equal history frames plus the "
                    f"current frame: configured={configured_frames}, "
                    f"history+current={actual_frames}"
                )
            if not bool(
                self.mem_vision_encoder.get(
                    "match_current_to_history_resolution",
                    False,
                )
            ):
                raise ValueError(
                    "Rynn MEM requires "
                    "match_current_to_history_resolution=true"
                )
            if str(
                self.mem_vision_encoder.get("output_token_policy", "")
            ).strip().lower() != "current_only":
                raise ValueError(
                    "Rynn MEM requires output_token_policy=current_only so "
                    "history tokens never enter the language decoder"
                )
            history_size = self.text_history.get("history_image_size", None)
            if not history_size or len(history_size) != 2:
                raise ValueError(
                    "Rynn MEM requires planner history_image_size=[width,height]"
                )
            runtime_mem = getattr(
                self.qwen_vl_interface,
                "mem_vision_encoder",
                {},
            )
            if not bool(runtime_mem.get("enabled", False)):
                raise RuntimeError(
                    "Rynn MEM was configured but not installed on the Qwen3.5 "
                    "vision tower"
                )

        # The vocabulary head is needed only by unified AR text planning.
        # Otherwise exclude it from optimizer/ZeRO state.
        if not self.text_planning_enabled:
            language_head = getattr(full_model, "lm_head", None)
            input_embeddings = full_model.get_input_embeddings()
            if language_head is not None:
                for parameter in language_head.parameters():
                    if parameter is not input_embeddings.weight:
                        parameter.requires_grad_(False)

    @property
    def device(self):
        return next(self.parameters()).device

    @staticmethod
    def _cast_plans(
        plans: torch.Tensor | list[torch.Tensor],
        dtype: torch.dtype,
    ) -> torch.Tensor | list[torch.Tensor]:
        if isinstance(plans, list):
            return [plan.to(dtype=dtype) for plan in plans]
        return plans.to(dtype=dtype)

    def _uses_action_state(self) -> bool:
        return self.include_state

    def set_dino_stats(self, stats_path: str) -> None:
        self._load_dino_stats(stats_path)

    def _qwen_backbone(self):
        full_model = self.qwen_vl_interface.model
        backbone = getattr(full_model, "model", None)
        return backbone if backbone is not None and backbone is not full_model else full_model

    @staticmethod
    def _view_list(value) -> list:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return list(value)
        array = np.asarray(value)
        return [array[index] for index in range(array.shape[0])] if array.ndim == 4 else [array]

    def _current_views(self, example: dict) -> list[Image.Image]:
        views = self._view_list(example.get("image_0", example.get("image")))
        if not views:
            raise KeyError("World--Action MoT requires current image_0/image")
        images = [to_pil_preserve(view) for view in views]
        target_size = self.config.datasets.vla_data.get("obs_image_size", None)
        if target_size:
            size = tuple(int(value) for value in target_size)
            images = [image if image.size == size else image.resize(size) for image in images]
        return images

    def _future_views(self, example: dict) -> list[Image.Image]:
        views = self._view_list(example.get("image_1"))
        if not views:
            raise KeyError("World--Action MoT training requires decoded future image_1")
        return [to_pil_preserve(view) for view in views]

    def _planner_history_views(self, example: dict) -> list[Image.Image]:
        if not getattr(self, "text_history_enabled", False):
            return []
        history_cfg = getattr(self, "text_history", {})
        image_field = str(
            history_cfg.get("image_field", "planner_history_images")
        )
        views = self._view_list(example.get(image_field))
        expected = len(getattr(self, "text_history_frame_offsets", []))
        if len(views) != expected:
            raise ValueError(
                "Text planner history must align with configured frame offsets: "
                f"field={image_field!r}, images={len(views)}, expected={expected}"
            )
        images = [to_pil_preserve(view) for view in views]
        configured_size = history_cfg.get("history_image_size", None)
        if configured_size:
            images = self._resize_planner_views(images, configured_size)
        return images

    @staticmethod
    def _resize_planner_views(
        images: list[Image.Image],
        configured_size,
    ) -> list[Image.Image]:
        size = tuple(int(value) for value in configured_size)
        if len(size) != 2 or min(size) <= 0:
            raise ValueError(
                f"history_image_size must be [width,height], got {configured_size}"
            )
        # Keep OpenCV optional for every non-history / non-MEM model path.
        import cv2

        return [
            image
            if image.size == size
            else Image.fromarray(
                cv2.resize(
                    np.asarray(image),
                    size,
                    interpolation=cv2.INTER_AREA,
                )
            )
            for image in images
        ]

    def _vlm_views(self, example: dict) -> list[Image.Image]:
        """Per-camera current views for the VLM only.

        Deliberately does not reuse ``_current_views``: that method force-resizes
        to ``obs_image_size`` (the composite geometry), which would squash these
        back onto the stitched layout.  Training samples carry ``image_0_vlm``
        and eval payloads carry ``image_vlm``, mirroring the existing
        ``image_0``/``image`` asymmetry.
        """

        views = self._view_list(
            example.get("image_0_vlm", example.get("image_vlm"))
        )
        if not views:
            raise KeyError(
                "vlm_view_source='separate_views' requires per-camera pixels in "
                "image_0_vlm (training) or image_vlm (eval); image_0 holds the "
                "composite reserved for the DINO/world branch"
            )
        images = [to_pil_preserve(view) for view in views]
        if self.vlm_view_size is not None:
            size = self.vlm_view_size
            images = [
                image if image.size == size else image.resize(size)
                for image in images
            ]
        return images

    def _planner_current_views(self, example: dict) -> list[Image.Image]:
        if self.vlm_view_source == "separate_views":
            return self._vlm_views(example)
        images = self._current_views(example)
        if getattr(self, "mem_vision_encoder_enabled", False):
            if len(images) != 1:
                raise ValueError(
                    "Rynn MEM expects one composite current image per sample; "
                    f"got {len(images)} current views"
                )
            images = self._resize_planner_views(
                images,
                self.text_history["history_image_size"],
            )
        return images

    def _prepare_mem_vision_inputs(self, inputs, examples: List[dict]):
        """Replace current pixels with K-frame pixels while keeping one image token span.

        The processor-built ``image_grid_thw`` and input IDs continue to
        describe only the current image.  The full K-frame grid is carried as
        a private tensor until the vision call, whose MEM wrapper collapses its
        output back to current-frame tokens before Qwen's language decoder.
        """

        if not getattr(self, "mem_vision_encoder_enabled", False):
            return inputs
        current_grid = inputs.get("image_grid_thw")
        if not torch.is_tensor(current_grid) or tuple(current_grid.shape) != (
            len(examples),
            3,
        ):
            raise ValueError(
                "Rynn MEM processor must publish one current image grid per "
                f"example, got {getattr(current_grid, 'shape', type(current_grid))}"
            )

        frames = []
        expected_frames = int(self.mem_vision_encoder.get("num_frames", 6))
        for example in examples:
            group = self._planner_history_views(example)
            group.extend(self._planner_current_views(example))
            if len(group) != expected_frames:
                raise ValueError(
                    "Rynn MEM frame group changed after planner preprocessing: "
                    f"got={len(group)}, expected={expected_frames}"
                )
            frames.extend(group)

        image_processor = getattr(
            self.qwen_vl_interface.processor,
            "image_processor",
            None,
        )
        if image_processor is None:
            raise RuntimeError("Rynn MEM requires a Qwen image_processor")
        packed = image_processor(images=frames, return_tensors="pt")
        pixel_values = packed.get("pixel_values")
        mem_grid = packed.get("image_grid_thw")
        if not torch.is_tensor(pixel_values) or not torch.is_tensor(mem_grid):
            raise RuntimeError(
                "Rynn MEM image_processor must return pixel_values and image_grid_thw"
            )
        expected_segments = len(examples) * expected_frames
        if tuple(mem_grid.shape) != (expected_segments, 3):
            raise ValueError(
                "Rynn MEM packed grid must contain B*K still images, got "
                f"{tuple(mem_grid.shape)}, expected={(expected_segments, 3)}"
            )
        if not torch.equal(
            current_grid[:, 1:].cpu(),
            mem_grid[expected_frames - 1 :: expected_frames, 1:].cpu(),
        ):
            raise ValueError(
                "Rynn MEM current placeholder resolution differs from its "
                "processed temporal frame group"
            )
        inputs["pixel_values"] = pixel_values
        inputs["_cogwam_mem_image_grid_thw"] = mem_grid
        return inputs

    @contextmanager
    def _activate_mem_vision_inputs(self, inputs):
        """Expose the private K-frame grid only during the Qwen vision call."""

        if not getattr(self, "mem_vision_encoder_enabled", False):
            yield
            return
        mem_grid = inputs.pop("_cogwam_mem_image_grid_thw", None)
        if not torch.is_tensor(mem_grid):
            raise RuntimeError(
                "Rynn MEM inputs lost their private full-frame image grid"
            )
        visual = getattr(self._qwen_backbone(), "visual", None)
        if visual is None or not bool(
            getattr(visual, "_cogwam_mem_enabled", False)
        ):
            raise RuntimeError("Rynn MEM visual wrapper is not active")
        previous = getattr(visual, "_cogwam_mem_active_grid_thw", None)
        visual._cogwam_mem_active_grid_thw = mem_grid
        try:
            yield
        finally:
            visual._cogwam_mem_active_grid_thw = previous

    def _planner_user_message(self, example: dict, prompt: str) -> dict:
        content = []
        if getattr(self, "text_history_enabled", False):
            if not getattr(self, "mem_vision_encoder_enabled", False):
                history = self._planner_history_views(example)
                for index, image in enumerate(history, start=1):
                    content.extend(
                        [
                            {
                                "type": "text",
                                "text": f"History observation {index} (oldest to newest):",
                            },
                            {"type": "image", "image": image},
                        ]
                    )
            content.append({"type": "text", "text": "Current observation:"})
        content.extend(
            {"type": "image", "image": image}
            for image in self._planner_current_views(example)
        )
        content.append({"type": "text", "text": str(prompt)})
        return {"role": "user", "content": content}

    def _physical_prompt(self, example: dict) -> str:
        prompt_template = self.config.datasets.vla_data.get(
            "CoT_prompt", "{instruction}"
        )
        return str(prompt_template).replace(
            "{instruction}", str(example.get("lang", ""))
        )

    def _text_prompt(self, example: dict) -> str:
        if getattr(self, "event_memory_enabled", False):
            fields = self.event_semantic_fields
            semantic_memory = str(example.get(fields["memory"], "") or "").strip()
            cached_subtask = str(
                example.get(fields["cached_subtask"], "") or ""
            ).strip()
            if not semantic_memory or not cached_subtask:
                raise ValueError(
                    "Event-memory prompt requires non-empty persistent state: "
                    f"{fields['memory']}={semantic_memory!r}, "
                    f"{fields['cached_subtask']}={cached_subtask!r}"
                )
            return str(self.text_supervision["prompt_template"]).format(
                instruction=str(example.get("lang", "")),
                semantic_memory=semantic_memory,
                cached_current_subtask=cached_subtask,
                # Label-bearing values are never exposed to the user prompt.
                memory_add="",
                subtask_text="",
            )
        history_cfg = getattr(self, "text_history", {})
        finished_field = str(
            history_cfg.get("finished_task_list_field", "finished_task_list")
        )
        finished = ""
        if getattr(self, "text_history_enabled", False):
            finished = str(example.get(finished_field, "") or "").strip()
            if not finished:
                raise ValueError(
                    "Text-history planner requires a non-empty previous Finished "
                    f"Task List in example[{finished_field!r}]"
                )
        values = {
            "instruction": str(example.get("lang", "")),
            "finished_task_list": finished,
            # Targets are deliberately unavailable to the prompt.  Keeping the
            # keys permits older templates to format, but prevents label leak.
            "subtask_text": "",
            "completed_subtask_text": "",
        }
        return str(self.text_supervision["prompt_template"]).format(**values)

    @staticmethod
    def _append_query_suffix(inputs, world_token_id: int, action_token_id: int, n_world: int, n_action: int):
        input_ids = inputs.get("input_ids")
        if not torch.is_tensor(input_ids) or input_ids.ndim != 2:
            raise ValueError("VLM learned queries require processor input_ids [B,T]")
        if int(world_token_id) == int(action_token_id):
            raise ValueError(
                "WORLD/ACTION learned-query placeholder token ids must be distinct"
            )
        if int(n_world) <= 0 or int(n_action) <= 0:
            raise ValueError(
                "WORLD/ACTION learned-query counts must both be positive"
            )
        if bool(
            ((input_ids == int(world_token_id)) | (input_ids == int(action_token_id))).any()
        ):
            raise ValueError(
                "learned-query placeholder token leaked into the user prompt"
            )
        batch, context_length = input_ids.shape
        world_suffix = input_ids.new_full((batch, n_world), int(world_token_id))
        action_suffix = input_ids.new_full((batch, n_action), int(action_token_id))
        inputs["input_ids"] = torch.cat([input_ids, world_suffix, action_suffix], dim=1)
        attention = inputs.get("attention_mask", torch.ones_like(input_ids))
        if (
            not torch.is_tensor(attention)
            or attention.ndim != 2
            or tuple(attention.shape) != (batch, context_length)
        ):
            raise ValueError(
                "learned-query attention_mask must align with pre-suffix input_ids "
                f"{(batch, context_length)}, got "
                f"{getattr(attention, 'shape', type(attention))}"
            )
        suffix_attention = torch.ones(
            (batch, n_world + n_action),
            device=attention.device,
            dtype=attention.dtype,
        )
        inputs["attention_mask"] = torch.cat([attention, suffix_attention], dim=1)
        # Transformers >=5.14 publishes Qwen3.5 modality ids explicitly.
        # WORLD/ACTION placeholders are ordinary text tokens (type 0), so every
        # sequence-aligned modality field must grow with input_ids.  Leaving
        # mm_token_type_ids at the pre-suffix length is the source of the
        # historical 172-vs-140 inference mask failure.
        for key in ("mm_token_type_ids", "token_type_ids"):
            token_types = inputs.get(key)
            if token_types is None:
                continue
            if (
                not torch.is_tensor(token_types)
                or token_types.ndim != 2
                or tuple(token_types.shape) != (batch, context_length)
            ):
                raise ValueError(
                    f"learned-query {key} must align with pre-suffix input_ids "
                    f"{(batch, context_length)}, got "
                    f"{getattr(token_types, 'shape', type(token_types))}"
                )
            text_suffix = token_types.new_zeros(
                (batch, n_world + n_action)
            )
            inputs[key] = torch.cat([token_types, text_suffix], dim=1)
        inputs.pop("position_ids", None)
        inputs.pop("cache_position", None)
        world_mask = torch.zeros_like(inputs["input_ids"], dtype=torch.bool)
        action_mask = torch.zeros_like(world_mask)
        world_mask[:, context_length : context_length + n_world] = True
        action_mask[:, context_length + n_world :] = True
        if (
            not bool((world_mask.sum(dim=1) == int(n_world)).all())
            or not bool((action_mask.sum(dim=1) == int(n_action)).all())
            or bool((world_mask & action_mask).any())
        ):
            raise RuntimeError(
                "failed to construct disjoint positional WORLD/ACTION "
                "learned-query masks"
            )
        return world_mask, action_mask

    @contextmanager
    def _replace_query_embeddings(
        self, world_mask: torch.Tensor, action_mask: torch.Tensor
    ):
        embedding = self.qwen_vl_interface.model.get_input_embeddings()
        world_queries = self.world_plan_queries(
            world_mask.shape[0], device=world_mask.device
        )
        action_queries = self.action_plan_queries(
            action_mask.shape[0], device=action_mask.device
        )
        matched = 0

        def hook(_module, _args, output):
            nonlocal matched
            if not torch.is_tensor(output) or output.ndim != 3:
                return output
            if tuple(output.shape[:2]) != tuple(world_mask.shape):
                return output
            replaced = output.clone()
            replaced[world_mask.to(output.device)] = world_queries.to(
                device=output.device, dtype=output.dtype
            ).reshape(-1, output.shape[-1])
            replaced[action_mask.to(output.device)] = action_queries.to(
                device=output.device, dtype=output.dtype
            ).reshape(-1, output.shape[-1])
            matched += 1
            return replaced

        handle = embedding.register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()
        if matched != 1:
            raise RuntimeError(
                "learned-query embedding replacement matched "
                f"{matched} calls; expected exactly one"
            )

    def _run_planner_backbone(
        self,
        inputs,
        world_mask: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor | list[torch.Tensor]:
        with self._activate_mem_vision_inputs(inputs):
            with self._replace_query_embeddings(world_mask, action_mask):
                device_type = self.device.type
                autocast = (
                    torch.autocast(device_type, dtype=torch.bfloat16)
                    if device_type == "cuda"
                    else nullcontext()
                )
                with autocast:
                    outputs = self._qwen_backbone()(
                        **inputs,
                        output_hidden_states=self.layerwise_planner_coupling,
                        return_dict=True,
                        use_cache=False,
                    )
        if self.layerwise_planner_coupling:
            hidden_states = getattr(outputs, "hidden_states", None)
            if hidden_states is None:
                raise RuntimeError(
                    "Layer-wise planner coupling requires VLM hidden_states"
                )
            expected = int(self.action_model.num_layers)
            # HF returns embedding output followed by one tensor per decoder
            # layer.  Taking the last N therefore maps block outputs 1..N to
            # physical layers 1..N after first-N truncation.
            if len(hidden_states) != expected + 1:
                raise RuntimeError(
                    "Truncated VLM hidden-state count mismatch: "
                    f"got={len(hidden_states)}, expected={expected + 1}"
                )
            return list(hidden_states[-expected:])
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            hidden = outputs[0]
        return hidden

    def _extract_plans(
        self,
        hidden: torch.Tensor | list[torch.Tensor],
        world_mask: torch.Tensor,
        action_mask: torch.Tensor,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[list[torch.Tensor], list[torch.Tensor]]:
        hidden_layers = hidden if isinstance(hidden, list) else [hidden]
        action_plans, world_plans = [], []
        for layer_hidden in hidden_layers:
            if not torch.is_tensor(layer_hidden) or layer_hidden.ndim != 3:
                raise ValueError(
                    "VLM learned-query hidden state must be [B,T,D], got "
                    f"{getattr(layer_hidden, 'shape', type(layer_hidden))}"
                )
            expected_prefix = tuple(layer_hidden.shape[:2])
            if (
                tuple(world_mask.shape) != expected_prefix
                or tuple(action_mask.shape) != expected_prefix
            ):
                raise ValueError(
                    "learned-query masks do not align with the VLM hidden "
                    f"sequence: hidden={expected_prefix}, "
                    f"world_mask={tuple(world_mask.shape)}, "
                    f"action_mask={tuple(action_mask.shape)}. This usually "
                    "means a Qwen3.5 processor sequence field was not extended."
                )
            world_counts = world_mask.sum(dim=1)
            action_counts = action_mask.sum(dim=1)
            if (
                not bool((world_counts == self.num_world_queries).all())
                or not bool((action_counts == self.num_action_queries).all())
            ):
                raise ValueError(
                    "VLM learned-query mask changed during the backbone forward: "
                    f"world={world_counts.tolist()} expected={self.num_world_queries}, "
                    f"action={action_counts.tolist()} expected={self.num_action_queries}. "
                    "These positional masks were validated before the forward; "
                    "this is not text planning and usually indicates a CUDA "
                    "extension/kernel ABI or memory-corruption failure."
                )
            hidden_size = layer_hidden.shape[-1]
            world_plans.append(
                layer_hidden[world_mask].reshape(
                    batch_size, self.num_world_queries, hidden_size
                )
            )
            action_plans.append(
                layer_hidden[action_mask].reshape(
                    batch_size, self.num_action_queries, hidden_size
                )
            )
        if isinstance(hidden, list):
            return action_plans, world_plans
        return action_plans[0], world_plans[0]

    def _planner_hidden(self, examples: List[dict]):
        """Encode physical conditioning with learned queries; no text generation."""

        processor = self.qwen_vl_interface.processor
        messages = [
            [self._planner_user_message(example, self._physical_prompt(example))]
            for example in examples
        ]
        old_padding_side = processor.tokenizer.padding_side
        processor.tokenizer.padding_side = "left"
        try:
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                padding=True,
                add_generation_prompt=True,
                enable_thinking=_qwen_enable_thinking(self.config),
                return_dict=True,
                return_tensors="pt",
            )
        finally:
            processor.tokenizer.padding_side = old_padding_side
        inputs = self._prepare_mem_vision_inputs(inputs, examples)
        inputs = inputs.to(self.device)
        world_mask, action_mask = self._append_query_suffix(
            inputs,
            self.world_placeholder_id,
            self.action_placeholder_id,
            self.num_world_queries,
            self.num_action_queries,
        )
        hidden = self._run_planner_backbone(inputs, world_mask, action_mask)
        action_plan, world_plan = self._extract_plans(
            hidden, world_mask, action_mask, batch_size=len(examples)
        )
        return action_plan, world_plan

    def _set_dino_teacher(self, teacher: DINOv3Backbone | None) -> None:
        if teacher is not None:
            teacher.requires_grad_(False)
            teacher.eval()
        object.__setattr__(self, "_dino_teacher", teacher)

    def _ensure_dino(self) -> DINOv3Backbone:
        teacher = getattr(self, "_dino_teacher", None)
        if teacher is None:
            teacher = DINOv3Backbone(**self._dino_spec)
        teacher = teacher.to(self.device).eval()
        self._set_dino_teacher(teacher)
        return teacher

    def _load_dino_stats(self, stats_path: str | None) -> None:
        if not stats_path:
            return
        path = Path(stats_path)
        if not path.exists():
            raise FileNotFoundError(f"Configured DINO stats do not exist: {path}")
        with path.open("r", encoding="utf-8") as handle:
            stats = json.load(handle)
        mean = torch.as_tensor(stats["mean"], dtype=torch.float32)
        std = torch.as_tensor(stats["std"], dtype=torch.float32).clamp_min(1.0e-6)
        if mean.numel() != self.dino_dim or std.numel() != self.dino_dim:
            raise ValueError(
                f"DINO stats dim={mean.numel()} does not match encoder dim={self.dino_dim}"
            )
        self._dino_mean.copy_(mean)
        self._dino_std.copy_(std)

    def _pool_dino(
        self,
        features: torch.Tensor,
        pool: int | None = None,
        image_size=None,
    ) -> torch.Tensor:
        pool = self.dino_pool if pool is None else int(pool)
        if pool == 1:
            return features
        rows, columns = dino_patch_grid(
            self._dino_spec["image_size"] if image_size is None else image_size,
            self._dino_spec["patch_size"],
        )
        batch, _, dim = features.shape
        if features.shape[1] != rows * columns:
            raise ValueError(
                "DINO feature/grid mismatch: "
                f"features={features.shape[1]}, grid={(rows, columns)}"
            )
        features = features.reshape(batch, rows, columns, dim)
        features = features.reshape(
            batch, rows // pool, pool, columns // pool, pool, dim
        ).mean(dim=(2, 4))
        return features.reshape(batch, -1, dim)

    @torch.no_grad()
    def _encode_dino_raw(
        self,
        batch_views: list[list[Image.Image]],
        *,
        image_size=None,
        expected_views: int | None = None,
        layer_indices: Sequence[int] | None = None,
    ) -> tuple[torch.Tensor, int]:
        teacher = self._ensure_dino()
        view_counts = {len(views) for views in batch_views}
        if len(view_counts) != 1:
            raise ValueError(f"all examples must have the same view count, got {view_counts}")
        views_per_example = next(iter(view_counts))
        expected_views = (
            self.num_world_views
            if expected_views is None
            else int(expected_views)
        )
        if views_per_example != expected_views:
            raise ValueError(
                "current/future image view count does not match the checkpoint contract: "
                f"got={views_per_example}, configured={expected_views}"
            )
        flat = [image for views in batch_views for image in views]
        tensor = teacher.preprocess_batch(flat, image_size=image_size)
        features = (
            teacher(tensor)
            if layer_indices is None
            else teacher.forward_multi_layer(tensor, layer_indices)
        ).float()
        return features, views_per_example

    def _finalize_dino(
        self,
        features: torch.Tensor,
        *,
        batch_size: int,
        views_per_example: int,
        pool: int,
        image_size=None,
    ) -> torch.Tensor:
        features = self._pool_dino(
            features,
            pool=pool,
            image_size=image_size,
        )
        features = features.reshape(
            int(batch_size),
            int(views_per_example) * features.shape[1],
            features.shape[2],
        )
        return (features - self._dino_mean) / self._dino_std

    @torch.no_grad()
    def _encode_dino(self, batch_views: list[list[Image.Image]]) -> torch.Tensor:
        """Encode the pooled world-model DINO grid (legacy public helper)."""

        features, views_per_example = self._encode_dino_raw(
            batch_views,
            image_size=self.future_dino_image_size,
            expected_views=self.num_world_views,
            layer_indices=self.dino_layers,
        )
        return self._finalize_dino(
            features,
            batch_size=len(batch_views),
            views_per_example=views_per_example,
            pool=self.dino_pool,
            image_size=self.future_dino_image_size,
        )

    @torch.no_grad()
    def _encode_current_dino(
        self,
        batch_views: list[list[Image.Image]],
    ) -> torch.Tensor:
        """Encode the clean current-observation grid at its configured resolution."""

        features, views_per_example = self._encode_dino_raw(
            batch_views,
            expected_views=self.num_current_world_views,
            layer_indices=self.dino_layers,
        )
        pool = (
            self.dino_pool
            if self.current_dino_pool is None
            else self.current_dino_pool
        )
        return self._finalize_dino(
            features,
            batch_size=len(batch_views),
            views_per_example=views_per_example,
            pool=pool,
            image_size=self._dino_spec["image_size"],
        )

    def _stack_actions(self, examples: List[dict], dtype: torch.dtype) -> torch.Tensor:
        values = np.stack([np.asarray(example["action"]) for example in examples])
        actions = torch.as_tensor(values, device=self.device, dtype=dtype)
        if actions.shape[1] < self.action_horizon or actions.shape[2] < self.action_dim:
            raise ValueError(
                f"action batch {tuple(actions.shape)} cannot satisfy "
                f"H={self.action_horizon}, D={self.action_dim}"
            )
        return actions[:, : self.action_horizon, : self.action_dim]

    def _stack_state(
        self, examples: List[dict], dtype: torch.dtype
    ) -> torch.Tensor | None:
        if not self.include_state:
            return None
        if not all("state" in example for example in examples):
            raise KeyError(
                "CogWAM requires normalized current state for every example"
            )
        values = np.stack([np.asarray(example["state"]) for example in examples])
        state = torch.as_tensor(values, device=self.device, dtype=dtype)
        if state.ndim == 2:
            state = state[:, None, :]
        state_dim = int(self.config.framework.action_model.state_dim)
        if state.ndim != 3 or state.shape[-1] != state_dim:
            raise ValueError(
                f"state batch must have shape [B,N,{state_dim}], got {tuple(state.shape)}"
            )
        # The joint dataset is configured at delta_indices=[0], so this is the
        # normalized current proprioception rather than a state trajectory.
        return state[:, :1]

    def _action_pad_mask(self, examples: List[dict]) -> torch.Tensor | None:
        if not all("action_is_pad" in example for example in examples):
            return None
        values = np.stack([np.asarray(example["action_is_pad"]) for example in examples])
        return torch.as_tensor(values, device=self.device, dtype=torch.bool)[
            :, : self.action_horizon
        ]

    def _event_decision(self, example: dict) -> str:
        field = self.event_semantic_fields["decision"]
        decision = str(example.get(field, "") or "").strip().upper()
        if decision not in {KEEP_DECISION, UPDATE_DECISION}:
            raise ValueError(
                f"Event-memory example requires {field}=KEEP|UPDATE, got {decision!r}"
            )
        return decision

    def _event_ground_truth_response(self, example: dict) -> str:
        decision = self._event_decision(example)
        if decision == KEEP_DECISION:
            return self.keep_token
        memory_add_field = self.event_semantic_fields["memory_add"]
        subtask_field = str(self.text_supervision.get("subtask_field", "subtask_text"))
        memory_add = str(example.get(memory_add_field, "") or "").strip()
        subtask = str(example.get(subtask_field, "") or "").strip()
        if not memory_add or not subtask:
            raise ValueError(
                "UPDATE supervision requires non-empty memory delta and subtask: "
                f"{memory_add_field}={memory_add!r}, {subtask_field}={subtask!r}"
            )
        response = str(self.text_supervision["update_response_template"]).format(
            update_token=self.update_token,
            memory_add=memory_add,
            subtask_text=subtask,
            instruction=str(example.get("lang", "")),
        ).strip()
        if not response.startswith(self.update_token):
            raise ValueError(
                "update_response_template must begin with the configured UPDATE token"
            )
        return response

    def _event_generated_response(self, decision: str, body: str) -> str:
        if decision == KEEP_DECISION:
            return self.keep_token
        if decision != UPDATE_DECISION:
            raise ValueError(f"Unknown generated semantic decision: {decision!r}")
        normalized = str(body or "").strip()
        return self.update_token + (f"\n{normalized}" if normalized else "")

    def _event_chat_inputs(
        self,
        examples: List[dict],
        responses: list[str],
    ):
        if len(examples) != len(responses):
            raise ValueError("Event responses must align with examples")
        users, conversations = [], []
        for example, response in zip(examples, responses):
            user = self._planner_user_message(example, self._text_prompt(example))
            users.append([user])
            conversations.append(
                [
                    user,
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": str(response)}],
                    },
                ]
            )
        processor = self.qwen_vl_interface.processor
        old_padding_side = processor.tokenizer.padding_side
        processor.tokenizer.padding_side = "left"
        try:
            full = processor.apply_chat_template(
                conversations,
                tokenize=True,
                padding=True,
                add_generation_prompt=False,
                enable_thinking=_qwen_enable_thinking(self.config),
                return_dict=True,
                return_tensors="pt",
            )
            prompts = processor.apply_chat_template(
                users,
                tokenize=True,
                padding=True,
                add_generation_prompt=True,
                enable_thinking=_qwen_enable_thinking(self.config),
                return_dict=True,
                return_tensors="pt",
            )
        finally:
            processor.tokenizer.padding_side = old_padding_side
        full = self._prepare_mem_vision_inputs(full, examples)
        self._validate_prompt_prefix(full, prompts)
        prompt_lengths = prompts["attention_mask"].sum(dim=1)
        full = self._truncate_assistant_at_eos(full, prompt_lengths)
        return full, prompt_lengths

    def _event_decision_positions(
        self,
        inputs,
        prompt_lengths: torch.Tensor,
        decisions: list[str],
    ) -> list[int]:
        ids = inputs["input_ids"]
        attention = inputs["attention_mask"].to(dtype=torch.bool)
        full_lengths = attention.sum(dim=1)
        positions: list[int] = []
        for row, decision in enumerate(decisions):
            start = ids.shape[1] - int(full_lengths[row])
            position = start + int(prompt_lengths[row])
            expected = (
                self.keep_token_id
                if decision == KEEP_DECISION
                else self.update_token_id
            )
            if position >= ids.shape[1] or int(ids[row, position]) != expected:
                actual = None if position >= ids.shape[1] else int(ids[row, position])
                raise ValueError(
                    "The decision must be the first assistant token and exactly one "
                    f"special token: row={row}, expected={expected}, actual={actual}"
                )
            positions.append(position)
        return positions

    def _event_remove_keep_eos(
        self,
        inputs,
        prompt_lengths: torch.Tensor,
        decisions: list[str],
    ):
        """Make KEEP exactly one assistant token before WORLD/ACTION queries."""

        decision_positions = self._event_decision_positions(
            inputs, prompt_lengths, decisions
        )
        input_ids = inputs["input_ids"]
        attention = inputs["attention_mask"].to(dtype=torch.bool)
        aligned_fields = {
            key: inputs[key]
            for key in ("mm_token_type_ids", "token_type_ids")
            if key in inputs
        }
        rows: list[torch.Tensor] = []
        aligned_rows = {key: [] for key in aligned_fields}
        for row, decision in enumerate(decisions):
            valid_ids = input_ids[row][attention[row]]
            start = input_ids.shape[1] - int(attention[row].sum())
            if decision == KEEP_DECISION:
                kept = decision_positions[row] - start + 1
                valid_ids = valid_ids[:kept]
            rows.append(valid_ids)
            for key, values in aligned_fields.items():
                valid_values = values[row][attention[row]]
                if decision == KEEP_DECISION:
                    valid_values = valid_values[:kept]
                aligned_rows[key].append(valid_values)

        tokenizer = self.qwen_vl_interface.processor.tokenizer
        eos_ids = self._token_id_set(getattr(tokenizer, "eos_token_id", None))
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is None:
            if not eos_ids:
                raise ValueError("Event planner tokenizer has neither pad nor EOS")
            pad_id = min(eos_ids)
        width = max(row.numel() for row in rows)
        rebuilt_ids = input_ids.new_full((len(rows), width), int(pad_id))
        rebuilt_attention = inputs["attention_mask"].new_zeros((len(rows), width))
        rebuilt_fields = {
            key: values.new_zeros((len(rows), width))
            for key, values in aligned_fields.items()
        }
        for row, values in enumerate(rows):
            rebuilt_ids[row, -values.numel() :] = values
            rebuilt_attention[row, -values.numel() :] = 1
            for key in rebuilt_fields:
                aligned = aligned_rows[key][row]
                rebuilt_fields[key][row, -aligned.numel() :] = aligned
        inputs["input_ids"] = rebuilt_ids
        inputs["attention_mask"] = rebuilt_attention
        inputs.update(rebuilt_fields)
        inputs.pop("position_ids", None)
        inputs.pop("cache_position", None)
        return inputs

    def _run_text_backbone(self, inputs) -> torch.Tensor:
        with self._activate_mem_vision_inputs(inputs):
            device_type = self.device.type
            autocast = (
                torch.autocast(device_type, dtype=torch.bfloat16)
                if device_type == "cuda"
                else nullcontext()
            )
            with autocast:
                outputs = self._qwen_backbone()(
                    **inputs,
                    output_hidden_states=False,
                    return_dict=True,
                    use_cache=False,
                )
        hidden = getattr(outputs, "last_hidden_state", None)
        return outputs[0] if hidden is None else hidden

    def _event_physical_hidden_from_responses(
        self,
        examples: List[dict],
        decisions: list[str],
        responses: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs, prompt_lengths = self._event_chat_inputs(examples, responses)
        inputs = self._event_remove_keep_eos(inputs, prompt_lengths, decisions)
        inputs = inputs.to(self.device)
        world_mask, action_mask = self._append_query_suffix(
            inputs,
            self.world_placeholder_id,
            self.action_placeholder_id,
            self.num_world_queries,
            self.num_action_queries,
        )
        hidden = self._run_planner_backbone(inputs, world_mask, action_mask)
        return self._extract_plans(
            hidden, world_mask, action_mask, batch_size=len(examples)
        )

    def _event_semantic_objective(self, examples: List[dict]) -> dict[str, torch.Tensor]:
        if not isinstance(examples, list) or not examples:
            raise ValueError(
                "Event-memory training requires a non-empty semantic_examples batch"
            )
        decisions = [self._event_decision(example) for example in examples]
        sampler_cfg = dict(self.event_data_config.get("sampler", {}))
        expected_batch = int(sampler_cfg.get("per_device_batch_size", 6))
        expected_updates = int(sampler_cfg.get("update_per_batch", 2))
        actual_updates = sum(
            decision == UPDATE_DECISION for decision in decisions
        )
        if len(examples) != expected_batch or actual_updates != expected_updates:
            raise ValueError(
                "Semantic dataloader lost its exact per-device boundary mix: "
                f"batch={len(examples)} (expected {expected_batch}), "
                f"updates={actual_updates} (expected {expected_updates})"
            )
        responses = [self._event_ground_truth_response(example) for example in examples]
        inputs, prompt_lengths = self._event_chat_inputs(examples, responses)
        decision_positions = self._event_decision_positions(
            inputs, prompt_lengths, decisions
        )
        ids = inputs["input_ids"]
        attention = inputs["attention_mask"].to(dtype=torch.bool)
        decision_labels = ids.new_full(ids.shape, -100)
        body_labels = ids.new_full(ids.shape, -100)
        for row, (decision, position) in enumerate(zip(decisions, decision_positions)):
            decision_labels[row, position] = ids[row, position]
            if decision == UPDATE_DECISION:
                valid_end = ids.shape[1] - 1
                # Left padding means the final valid token is at the right edge.
                if not bool(attention[row, valid_end]):
                    raise RuntimeError("Event chat rows must be left padded")
                body_labels[row, position + 1 : valid_end + 1] = ids[
                    row, position + 1 : valid_end + 1
                ]
        inputs = inputs.to(self.device)
        decision_labels = decision_labels.to(self.device)
        body_labels = body_labels.to(self.device)
        hidden = self._run_text_backbone(inputs)
        decision_loss = self._next_token_loss(hidden, decision_labels)
        body_loss = self._next_token_loss(hidden, body_labels)

        prediction_positions = torch.as_tensor(
            [position - 1 for position in decision_positions],
            device=self.device,
            dtype=torch.long,
        )
        rows = torch.arange(len(examples), device=self.device)
        prediction_hidden = hidden[rows, prediction_positions]
        output_head = self.qwen_vl_interface.model.get_output_embeddings()
        decision_logits = output_head(prediction_hidden)[:,
            [self.keep_token_id, self.update_token_id]
        ]
        targets = torch.as_tensor(
            [0 if decision == KEEP_DECISION else 1 for decision in decisions],
            device=self.device,
        )
        accuracy = decision_logits.float().argmax(dim=-1).eq(targets).float().mean()
        update_count = actual_updates
        result = {
            "loss": decision_loss + body_loss,
            "decision_loss": decision_loss.detach(),
            "body_loss": body_loss.detach(),
            "decision_accuracy": accuracy.detach(),
            "sample_count": hidden.new_tensor(float(len(examples))),
            "update_count": hidden.new_tensor(float(update_count)),
        }
        return result

    def _event_prompt_inputs(self, examples: List[dict]):
        processor = self.qwen_vl_interface.processor
        messages = [
            [self._planner_user_message(example, self._text_prompt(example))]
            for example in examples
        ]
        old_padding_side = processor.tokenizer.padding_side
        processor.tokenizer.padding_side = "left"
        try:
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                padding=True,
                add_generation_prompt=True,
                enable_thinking=_qwen_enable_thinking(self.config),
                return_dict=True,
                return_tensors="pt",
            )
        finally:
            processor.tokenizer.padding_side = old_padding_side
        return self._prepare_mem_vision_inputs(inputs, examples).to(self.device)

    @staticmethod
    def _append_text_token(inputs, token_id: int):
        ids = inputs["input_ids"]
        inputs["input_ids"] = torch.cat(
            [ids, ids.new_full((ids.shape[0], 1), int(token_id))], dim=1
        )
        attention = inputs["attention_mask"]
        inputs["attention_mask"] = torch.cat(
            [attention, attention.new_ones((attention.shape[0], 1))], dim=1
        )
        for key in ("mm_token_type_ids", "token_type_ids"):
            values = inputs.get(key)
            if values is not None:
                inputs[key] = torch.cat(
                    [values, values.new_zeros((values.shape[0], 1))], dim=1
                )
        inputs.pop("position_ids", None)
        inputs.pop("cache_position", None)
        return inputs

    @torch.no_grad()
    def _event_generate_update_bodies(self, examples: List[dict]) -> list[str]:
        if not examples:
            return []
        inputs = self._event_prompt_inputs(examples)
        inputs = self._append_text_token(inputs, self.update_token_id)
        max_new_tokens = int(self.event_max_new_tokens)
        if max_new_tokens <= 0:
            raise ValueError("Event max_new_tokens must be positive")
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "use_cache": True,
        }
        device_type = self.device.type
        autocast = (
            torch.autocast(device_type, dtype=torch.bfloat16)
            if device_type == "cuda"
            else nullcontext()
        )
        with self._activate_mem_vision_inputs(inputs):
            with autocast:
                generated = self.qwen_vl_interface.model.generate(
                    **inputs, **generation_kwargs
                )
        generated_ids = getattr(generated, "sequences", generated)
        _canonical, bodies = self._generated_sequence_inputs(inputs, generated_ids)
        return [str(body).strip() for body in bodies]

    @torch.no_grad()
    def _event_generate_states(
        self, examples: List[dict]
    ) -> tuple[list[str], list[str]]:
        inputs = self._event_prompt_inputs(examples)
        hidden = self._run_text_backbone(inputs)
        last_hidden = hidden[:, -1]
        output_head = self.qwen_vl_interface.model.get_output_embeddings()
        logits = output_head(last_hidden)[:, [self.keep_token_id, self.update_token_id]]
        update_mask = logits.float().argmax(dim=-1).eq(1)
        cache_valid_field = self.event_semantic_fields["cache_valid"]
        for row, example in enumerate(examples):
            if not bool(example.get(cache_valid_field, True)):
                update_mask[row] = True
        decisions = [
            UPDATE_DECISION if bool(value) else KEEP_DECISION
            for value in update_mask.tolist()
        ]
        update_rows = [
            row for row, decision in enumerate(decisions) if decision == UPDATE_DECISION
        ]
        update_bodies = self._event_generate_update_bodies(
            [examples[row] for row in update_rows]
        )
        bodies = [""] * len(examples)
        for row, body in zip(update_rows, update_bodies):
            bodies[row] = body
        return decisions, bodies

    def _event_sampling_probability(self, global_step: int | None) -> float:
        step = max(0, int(global_step or 0))
        points = self.event_schedule_points
        if step >= points[-1][0]:
            return float(points[-1][1])
        for (left_step, left_p), (right_step, right_p) in pairwise(points):
            if left_step <= step < right_step:
                ratio = (step - left_step) / float(right_step - left_step)
                return float(left_p + ratio * (right_p - left_p))
        return float(points[0][1])

    def _event_response_preserves_prompt_prefix(
        self, example: dict, response: str
    ) -> bool:
        """Return whether assistant text keeps the chat-template causal prefix."""

        processor = self.qwen_vl_interface.processor
        user = self._planner_user_message(example, self._text_prompt(example))
        users = [[user]]
        conversations = [
            [
                user,
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": str(response)}],
                },
            ]
        ]
        old_padding_side = processor.tokenizer.padding_side
        processor.tokenizer.padding_side = "left"
        try:
            full = processor.apply_chat_template(
                conversations,
                tokenize=True,
                padding=True,
                add_generation_prompt=False,
                enable_thinking=_qwen_enable_thinking(self.config),
                return_dict=True,
                return_tensors="pt",
            )
            prompts = processor.apply_chat_template(
                users,
                tokenize=True,
                padding=True,
                add_generation_prompt=True,
                enable_thinking=_qwen_enable_thinking(self.config),
                return_dict=True,
                return_tensors="pt",
            )
            self._validate_prompt_prefix(full, prompts)
            return True
        except ValueError:
            return False
        finally:
            processor.tokenizer.padding_side = old_padding_side

    def _event_try_scheduled_response(
        self,
        example: dict,
        decision: str,
        body: str,
    ) -> tuple[str, str] | None:
        """Accept a model condition only when it is safe for physical decode.

        Malformed UPDATE bodies or chat-template-breaking generations fall back
        to GT conditioning for that row; they must never abort the training step.
        """

        if decision == UPDATE_DECISION:
            try:
                self._parse_event_update_body(body)
            except ValueError:
                return None
        elif decision != KEEP_DECISION:
            return None
        response = self._event_generated_response(decision, body)
        if not self._event_response_preserves_prompt_prefix(example, response):
            return None
        return decision, response

    def _event_training_physical_hidden(
        self,
        examples: List[dict],
        global_step: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        gt_decisions = [self._event_decision(example) for example in examples]
        gt_responses = [
            self._event_ground_truth_response(example) for example in examples
        ]
        decisions = list(gt_decisions)
        responses = list(gt_responses)
        probability = self._event_sampling_probability(global_step)
        selected_rows: list[int] = []
        accepted_rows: list[int] = []
        predicted_update_count = 0
        fallback_count = 0
        if probability > 0.0:
            selected = torch.rand(len(examples), device=self.device).lt(probability)
            selected_rows = selected.nonzero(as_tuple=False).flatten().tolist()
            if selected_rows:
                selected_examples = [examples[row] for row in selected_rows]
                predicted_decisions, predicted_bodies = self._event_generate_states(
                    selected_examples
                )
                for row, decision, body in zip(
                    selected_rows, predicted_decisions, predicted_bodies
                ):
                    accepted = self._event_try_scheduled_response(
                        examples[row], decision, body
                    )
                    if accepted is None:
                        fallback_count += 1
                        continue
                    decisions[row], responses[row] = accepted
                    accepted_rows.append(row)
                    predicted_update_count += int(
                        accepted[0] == UPDATE_DECISION
                    )
        try:
            action_plan, world_plan = self._event_physical_hidden_from_responses(
                examples, decisions, responses
            )
        except ValueError as exc:
            # Last resort: a batched template edge case must not kill the job.
            if "exact causal prefix" not in str(exc):
                raise
            fallback_count = len(selected_rows)
            accepted_rows = []
            predicted_update_count = 0
            action_plan, world_plan = self._event_physical_hidden_from_responses(
                examples, gt_decisions, gt_responses
            )
        reference = action_plan[-1] if isinstance(action_plan, list) else action_plan
        metrics = {
            "scheduled_probability": reference.new_tensor(probability),
            "scheduled_count": reference.new_tensor(float(len(accepted_rows))),
            "scheduled_update_count": reference.new_tensor(
                float(predicted_update_count)
            ),
            "scheduled_fallback_count": reference.new_tensor(float(fallback_count)),
        }
        return action_plan, world_plan, metrics

    @staticmethod
    def _parse_event_update_body(body: str) -> tuple[str, str]:
        match = re.search(
            r"(?:^|\n)\s*Memory Add:\s*(.*?)\s*\n\s*Current Subtask:\s*(.*?)\s*$",
            str(body).strip(),
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match is None:
            raise ValueError(
                "UPDATE generation must contain `Memory Add:` followed by "
                f"`Current Subtask:`; got {body!r}"
            )
        memory_add, subtask = (match.group(1).strip(), match.group(2).strip())
        if not memory_add or not subtask:
            raise ValueError(f"UPDATE generation contains an empty field: {body!r}")
        return memory_add, subtask

    def _text_target(self, example: dict) -> str:
        """Build the required supervised response; prompts never contain targets."""

        cfg = self.text_supervision
        if not self.text_planning_enabled:
            raise RuntimeError("_text_target must only be called when text planning is enabled")
        subtask_field = str(cfg.get("subtask_field", "subtask_text"))
        completed_field = str(
            cfg.get("completed_subtask_field", "completed_subtask_text")
        )
        subtask = str(example.get(subtask_field, "") or "").strip()
        completed = str(
            example.get(completed_field, "") or ""
        ).strip()
        missing = [
            field
            for field, value in (
                (subtask_field, subtask),
                (completed_field, completed),
            )
            if not value
        ]
        if missing:
            location = ", ".join(
                f"{name}={example.get(name)!r}"
                for name in ("dataset_name", "trajectory_id", "base_index")
                if name in example
            )
            raise ValueError(
                "Unified text planning requires non-empty text annotations for "
                f"every row; missing={missing}"
                + (f", {location}" if location else "")
            )
        if getattr(self, "text_history_enabled", False):
            history_cfg = getattr(self, "text_history", {})
            finished_field = str(
                history_cfg.get(
                    "finished_task_list_field",
                    "finished_task_list",
                )
            )
            previous = str(example.get(finished_field, "") or "").strip()
            if not previous:
                raise ValueError(
                    "History-aware text supervision requires a non-empty previous "
                    f"Finished Task List in example[{finished_field!r}]"
                )
            completed = self._merge_finished_task_lists(previous, completed)
        values = {
            "instruction": str(example.get("lang", "")),
            "subtask_text": subtask,
            "completed_subtask_text": completed,
            "finished_task_list": completed,
        }
        return str(cfg["response_template"]).format(**values)

    def _finished_task_list_from_planner_text(self, planner_text: str) -> str:
        """Extract the explicit state field emitted by a history-aware planner."""

        history_cfg = getattr(self, "text_history", {})
        prefix = str(
            history_cfg.get(
                "finished_task_list_prefix",
                "Finished Task List:",
            )
        ).strip()
        if not prefix:
            raise ValueError("finished_task_list_prefix must be non-empty")
        prefix_lower = prefix.lower()
        for line in str(planner_text).splitlines():
            stripped = line.strip()
            if stripped.lower().startswith(prefix_lower):
                value = stripped[len(prefix) :].strip()
                if value:
                    return value
                break
        raise ValueError(
            "History-aware planner response omitted a non-empty "
            f"{prefix!r} field: {planner_text!r}"
        )

    def _finished_task_items(self, value: str) -> list[str]:
        """Split a task-list string conservatively for stable monotonic union."""

        history_cfg = getattr(self, "text_history", {})
        empty_value = str(
            history_cfg.get("empty_finished_task_list", "None")
        ).strip()
        normalized_value = re.sub(r"\s+", " ", str(value)).strip(" \t.;")
        empty_markers = {
            re.sub(r"\s+", " ", empty_value).strip(" \t.;").lower(),
            "none",
            "nothing",
            "n/a",
        }
        if normalized_value.lower() in empty_markers:
            return []
        pieces = re.split(
            r"(?:\r?\n|\s*[;|]\s*|(?<=[.!?])\s+)",
            str(value).strip(),
        )
        return [
            re.sub(r"\s+", " ", piece).strip(" \t.;")
            for piece in pieces
            if re.sub(r"\s+", " ", piece).strip(" \t.;")
        ]

    def _merge_finished_task_lists(self, previous: str, predicted: str) -> str:
        """Keep completed tasks monotonic while accepting newly observed tasks."""

        history_cfg = getattr(self, "text_history", {})
        empty_value = str(
            history_cfg.get("empty_finished_task_list", "None")
        ).strip()
        merged: list[str] = []
        seen = set()
        for item in [
            *self._finished_task_items(previous),
            *self._finished_task_items(predicted),
        ]:
            key = re.sub(r"[^a-z0-9]+", " ", item.lower()).strip()
            if key and key not in seen:
                seen.add(key)
                merged.append(item)
        if not merged:
            return empty_value
        return ". ".join(merged) + "."

    def _reconcile_generated_planner_text(
        self,
        example: dict,
        planner_text: str,
    ) -> str:
        """Rewrite generated state so Finished Task List cannot regress."""

        normalized = str(planner_text).strip()
        if not getattr(self, "text_history_enabled", False):
            return normalized
        history_cfg = getattr(self, "text_history", {})
        field = str(
            history_cfg.get("finished_task_list_field", "finished_task_list")
        )
        previous = str(example.get(field, "") or "").strip()
        if not previous:
            raise ValueError(
                f"History-aware inference requires example[{field!r}]"
            )
        predicted = self._finished_task_list_from_planner_text(normalized)
        merged = self._merge_finished_task_lists(previous, predicted)
        prefix = str(
            history_cfg.get(
                "finished_task_list_prefix",
                "Finished Task List:",
            )
        ).strip()
        lines = normalized.splitlines()
        for index, line in enumerate(lines):
            if line.strip().lower().startswith(prefix.lower()):
                lines[index] = f"{prefix} {merged}"
                return "\n".join(lines).strip()
        raise RuntimeError("Finished Task List disappeared during reconciliation")

    @staticmethod
    def _assistant_labels(input_ids, attention_mask, prompt_lengths):
        labels = input_ids.clone()
        valid = attention_mask.to(dtype=torch.bool)
        labels[~valid] = -100
        full_lengths = valid.sum(dim=1)
        width = labels.shape[1]
        for row in range(labels.shape[0]):
            start = width - int(full_lengths[row])
            labels[row, : start + int(prompt_lengths[row])] = -100
        return labels

    @staticmethod
    def _validate_prompt_prefix(full, prompts) -> None:
        for row in range(full["input_ids"].shape[0]):
            full_tokens = full["input_ids"][row][
                full["attention_mask"][row].to(dtype=torch.bool)
            ]
            prompt_tokens = prompts["input_ids"][row][
                prompts["attention_mask"][row].to(dtype=torch.bool)
            ]
            if (
                prompt_tokens.numel() > full_tokens.numel()
                or not torch.equal(
                    full_tokens[: prompt_tokens.numel()], prompt_tokens
                )
            ):
                raise ValueError(
                    "assistant training prompt is not an exact causal prefix of "
                    "the text->EOS sequence"
                )

    def _truncate_assistant_at_eos(self, inputs, prompt_lengths):
        """Remove chat-template tokens after the assistant EOS boundary."""

        tokenizer = self.qwen_vl_interface.processor.tokenizer
        full_model = getattr(self.qwen_vl_interface, "model", None)
        eos_ids = self._token_id_set(getattr(tokenizer, "eos_token_id", None))
        eos_ids.update(
            self._token_id_set(
                getattr(
                    getattr(full_model, "generation_config", None),
                    "eos_token_id",
                    None,
                )
            )
        )
        if not eos_ids:
            raise ValueError("unified text planner tokenizer has no EOS token")
        pad_id = getattr(tokenizer, "pad_token_id", None)
        pad_id = min(eos_ids) if pad_id is None else int(pad_id)

        input_ids = inputs["input_ids"]
        attention = inputs["attention_mask"].to(dtype=torch.bool)
        aligned_fields = {}
        for key in ("mm_token_type_ids", "token_type_ids"):
            values = inputs.get(key)
            if values is None:
                continue
            if (
                not torch.is_tensor(values)
                or values.ndim != 2
                or tuple(values.shape) != tuple(input_ids.shape)
            ):
                raise ValueError(
                    f"assistant {key} must align with input_ids "
                    f"{tuple(input_ids.shape)}, got "
                    f"{getattr(values, 'shape', type(values))}"
                )
            aligned_fields[key] = values
        rows = []
        aligned_rows = {key: [] for key in aligned_fields}
        for row in range(input_ids.shape[0]):
            tokens = input_ids[row][attention[row]]
            prompt_length = int(prompt_lengths[row])
            if prompt_length > tokens.numel():
                raise ValueError(
                    f"assistant prompt length {prompt_length} exceeds sequence "
                    f"length {tokens.numel()}"
                )
            eos_position = None
            for index in range(prompt_length, tokens.numel()):
                if int(tokens[index]) in eos_ids:
                    eos_position = index
                    break
            if eos_position is None:
                raise ValueError(
                    "assistant chat template did not terminate its response with EOS"
                )
            kept_length = eos_position + 1
            rows.append(tokens[:kept_length])
            for key, values in aligned_fields.items():
                aligned_rows[key].append(
                    values[row][attention[row]][:kept_length]
                )

        width = max(tokens.numel() for tokens in rows)
        rebuilt_ids = input_ids.new_full((len(rows), width), pad_id)
        rebuilt_attention = inputs["attention_mask"].new_zeros((len(rows), width))
        rebuilt_fields = {
            key: values.new_zeros((len(rows), width))
            for key, values in aligned_fields.items()
        }
        for row, tokens in enumerate(rows):
            rebuilt_ids[row, width - tokens.numel() :] = tokens
            rebuilt_attention[row, width - tokens.numel() :] = 1
            for key in rebuilt_fields:
                values = aligned_rows[key][row]
                rebuilt_fields[key][row, width - values.numel() :] = values
        inputs["input_ids"] = rebuilt_ids
        inputs["attention_mask"] = rebuilt_attention
        inputs.update(rebuilt_fields)
        for stale_key in ("position_ids", "cache_position"):
            inputs.pop(stale_key, None)
        return inputs

    def _next_token_loss(
        self,
        hidden: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute NTP only at supervised response/EOS positions.

        Selecting hidden states before the vocabulary projection avoids
        materializing [batch, full_sequence, vocab] logits.  The shift exactly
        matches Hugging Face causal-LM loss: hidden[t] predicts labels[t + 1].
        """

        shifted_labels = labels[:, 1:]
        supervised = shifted_labels.ne(-100)
        output_head = self.qwen_vl_interface.model.get_output_embeddings()
        if output_head is None:
            raise RuntimeError(
                "Unified text planning requires a causal-LM output embedding/head"
            )
        if not bool(supervised.any()):
            raise ValueError(
                "Unified text planning produced no supervised response/EOS tokens"
            )
        prediction_hidden = hidden[:, :-1][supervised]
        targets = shifted_labels[supervised]
        logits = output_head(prediction_hidden)
        return F.cross_entropy(logits.float(), targets)

    def _teacher_forced_planner_hidden(
        self,
        examples: List[dict],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """One causal training pass: context -> text -> EOS -> WORLD -> ACTION."""

        if not self.text_planning_enabled:
            action_plan, world_plan = self._planner_hidden(examples)
            reference = (
                action_plan[-1]
                if isinstance(action_plan, list)
                else action_plan
            )
            return action_plan, world_plan, reference.new_zeros(()), 0

        responses = [self._text_target(example) for example in examples]
        users, conversations = [], []
        for example, response in zip(examples, responses):
            user = self._planner_user_message(example, self._text_prompt(example))
            users.append([user])
            conversations.append(
                [
                    user,
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": response,
                            }
                        ],
                    },
                ]
            )
        processor = self.qwen_vl_interface.processor
        old_padding_side = processor.tokenizer.padding_side
        processor.tokenizer.padding_side = "left"
        try:
            full = processor.apply_chat_template(
                conversations,
                tokenize=True,
                padding=True,
                add_generation_prompt=False,
                enable_thinking=_qwen_enable_thinking(self.config),
                return_dict=True,
                return_tensors="pt",
            )
            prompts = processor.apply_chat_template(
                users,
                tokenize=True,
                padding=True,
                add_generation_prompt=True,
                enable_thinking=_qwen_enable_thinking(self.config),
                return_dict=True,
                return_tensors="pt",
            )
        finally:
            processor.tokenizer.padding_side = old_padding_side

        full = self._prepare_mem_vision_inputs(full, examples)

        self._validate_prompt_prefix(full, prompts)
        prompt_lengths = prompts["attention_mask"].sum(dim=1)
        full = self._truncate_assistant_at_eos(full, prompt_lengths)
        labels = self._assistant_labels(
            full["input_ids"], full["attention_mask"], prompt_lengths
        )

        full = full.to(self.device)
        labels = labels.to(self.device)
        world_mask, action_mask = self._append_query_suffix(
            full,
            self.world_placeholder_id,
            self.action_placeholder_id,
            self.num_world_queries,
            self.num_action_queries,
        )
        query_labels = labels.new_full(
            (labels.shape[0], self.num_world_queries + self.num_action_queries),
            -100,
        )
        labels = torch.cat([labels, query_labels], dim=1)
        hidden = self._run_planner_backbone(full, world_mask, action_mask)
        action_plan, world_plan = self._extract_plans(
            hidden, world_mask, action_mask, batch_size=len(examples)
        )
        text_hidden = hidden[-1] if isinstance(hidden, list) else hidden
        return (
            action_plan,
            world_plan,
            self._next_token_loss(text_hidden, labels),
            len(responses),
        )

    @staticmethod
    def _token_id_set(value) -> set[int]:
        if value is None:
            return set()
        if isinstance(value, (list, tuple, set)):
            return {int(token_id) for token_id in value}
        return {int(value)}

    def _generated_sequence_inputs(
        self,
        inputs,
        generated_ids: torch.Tensor,
    ) -> tuple[object, list[str]]:
        """Trim generation padding and left-pad complete context->text->EOS rows."""

        prompt_ids = inputs["input_ids"]
        prompt_attention = inputs["attention_mask"].to(dtype=torch.bool)
        prompt_width = prompt_ids.shape[1]
        aligned_fields = {}
        for key in ("mm_token_type_ids", "token_type_ids"):
            values = inputs.get(key)
            if values is None:
                continue
            if (
                not torch.is_tensor(values)
                or values.ndim != 2
                or tuple(values.shape) != tuple(prompt_ids.shape)
            ):
                raise ValueError(
                    f"planner generation {key} must align with prompt input_ids "
                    f"{tuple(prompt_ids.shape)}, got "
                    f"{getattr(values, 'shape', type(values))}"
                )
            aligned_fields[key] = values
        if generated_ids.ndim != 2 or generated_ids.shape[0] != prompt_ids.shape[0]:
            raise ValueError(
                "planner generation must return [B,T] token ids, got "
                f"{tuple(generated_ids.shape)}"
            )
        if generated_ids.shape[1] < prompt_width:
            raise ValueError("decoder-only planner generation omitted the input prompt")

        tokenizer = self.qwen_vl_interface.processor.tokenizer
        full_model = self.qwen_vl_interface.model
        eos_ids = self._token_id_set(getattr(tokenizer, "eos_token_id", None))
        eos_ids.update(
            self._token_id_set(
                getattr(getattr(full_model, "generation_config", None), "eos_token_id", None)
            )
        )
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is None:
            if not eos_ids:
                raise ValueError("text planner tokenizer defines neither pad nor EOS token")
            pad_id = min(eos_ids)
        pad_id = int(pad_id)

        sequences = []
        aligned_sequences = {key: [] for key in aligned_fields}
        decoded = []
        for row in range(prompt_ids.shape[0]):
            prompt = prompt_ids[row][prompt_attention[row]]
            completion = generated_ids[row, prompt_width:]
            end = completion.numel()
            found_eos = False
            for index, token_id in enumerate(completion.tolist()):
                if int(token_id) in eos_ids:
                    end = index + 1
                    found_eos = True
                    break
            if not found_eos and pad_id not in eos_ids:
                while end > 0 and int(completion[end - 1]) == pad_id:
                    end -= 1
            completion = completion[:end]
            if not found_eos:
                if not eos_ids:
                    raise ValueError(
                        "text planner generation reached its token limit but no EOS id is defined"
                    )
                completion = torch.cat(
                    [completion, completion.new_tensor([min(eos_ids)])], dim=0
                )
            sequences.append(torch.cat([prompt, completion], dim=0))
            for key, values in aligned_fields.items():
                prompt_values = values[row][prompt_attention[row]]
                completion_values = prompt_values.new_zeros(completion.shape)
                aligned_sequences[key].append(
                    torch.cat([prompt_values, completion_values], dim=0)
                )
            decoded.append(
                tokenizer.decode(completion.tolist(), skip_special_tokens=True).strip()
            )

        width = max(sequence.numel() for sequence in sequences)
        rebuilt_ids = prompt_ids.new_full((len(sequences), width), pad_id)
        rebuilt_attention = inputs["attention_mask"].new_zeros(
            (len(sequences), width)
        )
        rebuilt_fields = {
            key: values.new_zeros((len(sequences), width))
            for key, values in aligned_fields.items()
        }
        for row, sequence in enumerate(sequences):
            rebuilt_ids[row, width - sequence.numel() :] = sequence
            rebuilt_attention[row, width - sequence.numel() :] = 1
            for key in rebuilt_fields:
                values = aligned_sequences[key][row]
                rebuilt_fields[key][row, width - values.numel() :] = values
        inputs["input_ids"] = rebuilt_ids
        inputs["attention_mask"] = rebuilt_attention
        inputs.update(rebuilt_fields)
        for stale_key in ("position_ids", "cache_position"):
            inputs.pop(stale_key, None)
        return inputs, decoded

    def _generate_planner_sequence(
        self,
        examples: List[dict],
    ) -> tuple[object, list[str]]:
        """Generate planner text and return canonical context->text->EOS rows."""
        processor = self.qwen_vl_interface.processor
        messages = [
            [self._planner_user_message(example, self._text_prompt(example))]
            for example in examples
        ]
        old_padding_side = processor.tokenizer.padding_side
        processor.tokenizer.padding_side = "left"
        try:
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                padding=True,
                add_generation_prompt=True,
                enable_thinking=_qwen_enable_thinking(self.config),
                return_dict=True,
                return_tensors="pt",
            )
        finally:
            processor.tokenizer.padding_side = old_padding_side
        inputs = self._prepare_mem_vision_inputs(inputs, examples)
        inputs = inputs.to(self.device)

        generation_kwargs = {
            "max_new_tokens": int(self.text_supervision.get("max_new_tokens", 64)),
            "do_sample": bool(self.text_supervision.get("do_sample", False)),
            "use_cache": True,
        }
        if generation_kwargs["max_new_tokens"] <= 0:
            raise ValueError("text_supervision.max_new_tokens must be positive")
        if generation_kwargs["do_sample"]:
            generation_kwargs["temperature"] = float(
                self.text_supervision.get("temperature", 1.0)
            )
        device_type = self.device.type
        autocast = (
            torch.autocast(device_type, dtype=torch.bfloat16)
            if device_type == "cuda"
            else nullcontext()
        )
        with self._activate_mem_vision_inputs(inputs):
            with autocast:
                generated = self.qwen_vl_interface.model.generate(
                    **inputs,
                    **generation_kwargs,
                )
        generated_ids = getattr(generated, "sequences", generated)
        return self._generated_sequence_inputs(inputs, generated_ids)

    def _planner_hidden_from_text(
        self,
        examples: List[dict],
        planner_text: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Recompute physical plans from current images and cached planner text.

        The sequence exactly follows text-supervised training:
        current image + prompt -> assistant text -> EOS -> WORLD -> ACTION.
        Only the autoregressive text generation is skipped.
        """
        if len(planner_text) != len(examples):
            raise ValueError(
                "planner_text must align with examples; "
                f"got {len(planner_text)} texts for {len(examples)} examples"
            )
        normalized = [str(text).strip() for text in planner_text]
        if any(not text for text in normalized):
            raise ValueError("cached planner_text entries must be non-empty strings")

        users, conversations = [], []
        for example, response in zip(examples, normalized):
            user = self._planner_user_message(example, self._text_prompt(example))
            users.append([user])
            conversations.append(
                [
                    user,
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": response}],
                    },
                ]
            )
        processor = self.qwen_vl_interface.processor
        old_padding_side = processor.tokenizer.padding_side
        processor.tokenizer.padding_side = "left"
        try:
            full = processor.apply_chat_template(
                conversations,
                tokenize=True,
                padding=True,
                add_generation_prompt=False,
                enable_thinking=_qwen_enable_thinking(self.config),
                return_dict=True,
                return_tensors="pt",
            )
            prompts = processor.apply_chat_template(
                users,
                tokenize=True,
                padding=True,
                add_generation_prompt=True,
                enable_thinking=_qwen_enable_thinking(self.config),
                return_dict=True,
                return_tensors="pt",
            )
        finally:
            processor.tokenizer.padding_side = old_padding_side

        full = self._prepare_mem_vision_inputs(full, examples)

        self._validate_prompt_prefix(full, prompts)
        prompt_lengths = prompts["attention_mask"].sum(dim=1)
        full = self._truncate_assistant_at_eos(full, prompt_lengths).to(self.device)
        world_mask, action_mask = self._append_query_suffix(
            full,
            self.world_placeholder_id,
            self.action_placeholder_id,
            self.num_world_queries,
            self.num_action_queries,
        )
        hidden = self._run_planner_backbone(full, world_mask, action_mask)
        return self._extract_plans(
            hidden, world_mask, action_mask, batch_size=len(examples)
        )

    def _cached_or_generated_planner_hidden(
        self,
        examples: List[dict],
        cached_planner_texts: list[str | None] | tuple[str | None, ...] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[str], list[bool]]:
        """Generate only cache misses, then plan every row from its current image."""
        if cached_planner_texts is None:
            action_plan, world_plan, text = self._generated_planner_hidden(examples)
            return action_plan, world_plan, text, [True] * len(examples)
        if not isinstance(cached_planner_texts, (list, tuple)):
            raise TypeError("cached_planner_texts must be a list/tuple or None")
        if len(cached_planner_texts) != len(examples):
            raise ValueError(
                "cached_planner_texts must align with examples; "
                f"got {len(cached_planner_texts)} entries for {len(examples)} examples"
            )

        text: list[str | None] = []
        refresh_indices = []
        for index, cached in enumerate(cached_planner_texts):
            normalized = None if cached is None else str(cached).strip()
            if not normalized:
                normalized = None
            text.append(normalized)
            if normalized is None:
                refresh_indices.append(index)

        if len(refresh_indices) == len(examples):
            generated_inputs, generated = self._generate_planner_sequence(
                examples
            )
            del generated_inputs
            if getattr(self, "text_history_enabled", False):
                normalized = [
                    self._reconcile_generated_planner_text(example, value)
                    for example, value in zip(examples, generated)
                ]
            else:
                normalized = [str(value).strip() for value in generated]
            if any(not value for value in normalized):
                raise RuntimeError("text planner generated an empty response")
            action_plan, world_plan = self._planner_hidden_from_text(
                examples,
                normalized,
            )
            return (
                action_plan,
                world_plan,
                normalized,
                [True] * len(examples),
            )

        if refresh_indices:
            refresh_examples = [examples[index] for index in refresh_indices]
            generated_inputs, generated = self._generate_planner_sequence(
                refresh_examples
            )
            del generated_inputs
            for index, refresh_example, generated_text in zip(
                refresh_indices,
                refresh_examples,
                generated,
            ):
                if getattr(self, "text_history_enabled", False):
                    normalized = self._reconcile_generated_planner_text(
                        refresh_example,
                        generated_text,
                    )
                else:
                    normalized = str(generated_text).strip()
                if not normalized:
                    raise RuntimeError(
                        f"text planner generated an empty response for batch row {index}"
                    )
                text[index] = normalized

        final_text = [str(value) for value in text]
        action_plan, world_plan = self._planner_hidden_from_text(
            examples,
            final_text,
        )
        refresh_set = set(refresh_indices)
        refreshed = [index in refresh_set for index in range(len(examples))]
        return action_plan, world_plan, final_text, refreshed

    def _generated_planner_hidden(
        self,
        examples: List[dict],
    ) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        """Inference: context -> AR text through EOS -> WORLD -> ACTION."""

        inputs, planner_text = self._generate_planner_sequence(examples)
        if getattr(self, "text_history_enabled", False):
            del inputs
            planner_text = [
                self._reconcile_generated_planner_text(example, text)
                for example, text in zip(examples, planner_text)
            ]
            action_plan, world_plan = self._planner_hidden_from_text(
                examples,
                planner_text,
            )
            return action_plan, world_plan, planner_text
        world_mask, action_mask = self._append_query_suffix(
            inputs,
            self.world_placeholder_id,
            self.action_placeholder_id,
            self.num_world_queries,
            self.num_action_queries,
        )
        hidden = self._run_planner_backbone(inputs, world_mask, action_mask)
        action_plan, world_plan = self._extract_plans(
            hidden, world_mask, action_mask, batch_size=len(examples)
        )
        return action_plan, world_plan, planner_text

    def forward(
        self,
        examples: List[dict] | None = None,
        semantic_examples: List[dict] | None = None,
        global_step: int | None = None,
        **_kwargs,
    ) -> dict:
        if not isinstance(examples, list) or not examples:
            raise ValueError("CogWAM.forward expects a non-empty example list")
        if self.event_memory_enabled:
            action_plan, world_plan, schedule_metrics = (
                self._event_training_physical_hidden(examples, global_step)
            )
            semantic = self._event_semantic_objective(semantic_examples)
            text_loss = semantic["loss"]
            text_samples = int(semantic["sample_count"].item())
        else:
            if semantic_examples is not None:
                raise ValueError(
                    "semantic_examples is only valid for event-memory checkpoints"
                )
            (
                action_plan,
                world_plan,
                text_loss,
                text_samples,
            ) = self._teacher_forced_planner_hidden(examples)
            reference = action_plan[-1] if isinstance(action_plan, list) else action_plan
            schedule_metrics = {
                "scheduled_probability": reference.new_zeros(()),
                "scheduled_count": reference.new_zeros(()),
                "scheduled_update_count": reference.new_zeros(()),
                "scheduled_fallback_count": reference.new_zeros(()),
            }
            semantic = {
                "decision_loss": reference.new_zeros(()),
                "body_loss": reference.new_zeros(()),
                "decision_accuracy": reference.new_zeros(()),
                "update_count": reference.new_zeros(()),
            }
        current_views = [
            self._current_views(example)
            for example in examples
        ]
        current_world = self._encode_current_dino(current_views)
        target_world = self._encode_dino([self._future_views(example) for example in examples])
        model_dtype = next(self.action_model.parameters()).dtype
        action_dtype_getter = getattr(self.action_model, "action_flow_dtype", None)
        action_dtype = (
            action_dtype_getter()
            if callable(action_dtype_getter)
            else model_dtype
        )
        action_plan = CogWAM._cast_plans(action_plan, model_dtype)
        world_plan = CogWAM._cast_plans(world_plan, model_dtype)
        current_world = current_world.to(device=self.device, dtype=model_dtype)
        target_world = target_world.to(device=self.device, dtype=model_dtype)
        # Preserve normalized action/state values until the opt-in fp32 shell
        # has performed its input projections.  Casting them to the bf16 core
        # dtype here would irreversibly quantize them before that boundary.
        target_action = self._stack_actions(examples, action_dtype)
        current_state = self._stack_state(examples, action_dtype)
        future_valid = torch.as_tensor(
            [example.get("future_valid", True) for example in examples],
            device=self.device,
            dtype=torch.float32,
        )
        physical = self.action_model.forward_train(
            action_plan=action_plan,
            world_plan=world_plan,
            current_world=current_world,
            target_action=target_action,
            target_world=target_world,
            action_is_pad=self._action_pad_mask(examples),
            future_valid=future_valid,
            current_state=current_state,
        )
        text_objective = self.text_loss_weight * text_loss
        total = physical["loss"] + text_objective
        action_plan_layers = (
            action_plan
            if isinstance(action_plan, list)
            else [action_plan]
        )
        world_plan_layers = world_plan if isinstance(world_plan, list) else [world_plan]
        result = {
            "action_loss": total,
            # Weighted objectives and their exact layer-wise VLM->MoT
            # interface tensors are retained only until the trainer finishes
            # the current backward pass.  They let diagnostics measure task
            # competition without asking ZeRO-managed VLM parameters for
            # auxiliary gradients.
            "mot_action_objective": physical["action_objective"],
            "mot_world_objective": physical["world_objective"],
            "mot_shared_vlm_interface": tuple(
                [*world_plan_layers, *action_plan_layers]
            ),
            "mot_action_loss_raw": physical["action_loss_raw"],
            "mot_world_loss_raw": physical["world_loss_raw"],
            "mot_action_loss_weighted": physical["action_objective"].detach(),
            "mot_world_loss_weighted": physical["world_objective"].detach(),
            "mot_world_loss_weight": total.new_tensor(
                float(self.action_model.world_loss_weight)
            ),
            "mot_text_loss_raw": text_loss.detach(),
            "mot_text_loss_weighted": text_objective.detach(),
            "mot_text_loss_weight": total.new_tensor(float(self.text_loss_weight)),
            "mot_text_sample_count": total.new_tensor(float(text_samples)),
        }
        if self.event_memory_enabled:
            result.update(
                mot_text_decision_loss=semantic["decision_loss"],
                mot_text_update_body_loss=semantic["body_loss"],
                mot_text_decision_accuracy=semantic["decision_accuracy"],
                mot_text_update_count=semantic["update_count"],
                mot_text_scheduled_probability=schedule_metrics[
                    "scheduled_probability"
                ].detach(),
                mot_text_scheduled_count=schedule_metrics[
                    "scheduled_count"
                ].detach(),
                mot_text_scheduled_update_count=schedule_metrics[
                    "scheduled_update_count"
                ].detach(),
                mot_text_scheduled_fallback_count=schedule_metrics[
                    "scheduled_fallback_count"
                ].detach(),
            )
        return result

    @torch.no_grad()
    def predict_action(self, examples: List[dict], **kwargs) -> dict:
        if not isinstance(examples, list):
            examples = [examples]
        if not examples:
            raise ValueError("CogWAM.predict_action expects at least one example")
        planner_text = None
        planner_text_refreshed = None
        event_output = None
        if getattr(self, "event_memory_enabled", False):
            if kwargs.get("cached_planner_texts", None) is not None:
                raise ValueError(
                    "Event-memory inference uses semantic_memory and "
                    "cached_current_subtask fields, not cached_planner_texts"
                )
            decisions, bodies = self._event_generate_states(examples)
            responses = [
                self._event_generated_response(decision, body)
                for decision, body in zip(decisions, bodies)
            ]
            action_plan, world_plan = self._event_physical_hidden_from_responses(
                examples, decisions, responses
            )
            memory_adds: list[str | None] = []
            current_subtasks: list[str | None] = []
            for decision, body in zip(decisions, bodies):
                if decision == KEEP_DECISION:
                    memory_adds.append(None)
                    current_subtasks.append(None)
                else:
                    memory_add, current_subtask = self._parse_event_update_body(body)
                    memory_adds.append(memory_add)
                    current_subtasks.append(current_subtask)
            event_output = {
                "semantic_decision": decisions,
                "semantic_memory_add": memory_adds,
                "semantic_current_subtask": current_subtasks,
                "planner_text": responses,
                "planner_text_refreshed": [
                    decision == UPDATE_DECISION for decision in decisions
                ],
            }
        elif self.text_planning_enabled:
            (
                action_plan,
                world_plan,
                planner_text,
                planner_text_refreshed,
            ) = self._cached_or_generated_planner_hidden(
                examples,
                kwargs.get("cached_planner_texts", None),
            )
        else:
            if kwargs.get("cached_planner_texts", None) is not None:
                raise ValueError(
                    "cached_planner_texts was provided to a checkpoint without "
                    "text planning"
                )
            action_plan, world_plan = self._planner_hidden(examples)
        current_views = [
            self._current_views(example)
            for example in examples
        ]
        current_world = self._encode_current_dino(current_views)
        model_dtype = next(self.action_model.parameters()).dtype
        action_dtype_getter = getattr(self.action_model, "action_flow_dtype", None)
        action_dtype = (
            action_dtype_getter()
            if callable(action_dtype_getter)
            else model_dtype
        )
        current_state = self._stack_state(examples, action_dtype)
        num_inference_steps = kwargs.get(
            "num_inference_steps",
            kwargs.get("num_ddim_steps", None),
        )
        previous_actions = kwargs.get("prev_action_chunk_normalized", None)
        if previous_actions is not None and not bool(
            getattr(self.action_model, "rtc_guidance_supported", False)
        ):
            raise RuntimeError(
                "RTC guidance requires world_action_mot.architecture="
                "causal_dino_mot"
            )
        sample_kwargs = dict(
            action_plan=CogWAM._cast_plans(action_plan, model_dtype),
            world_plan=CogWAM._cast_plans(world_plan, model_dtype),
            current_world=current_world.to(device=self.device, dtype=model_dtype),
            current_state=current_state,
            seed=kwargs.get("mot_inference_seed", None),
            num_inference_steps=num_inference_steps,
        )
        if previous_actions is not None:
            sample_kwargs.update(
                prev_action_chunk_normalized=previous_actions,
                rtc_prefix_lengths=kwargs.get("rtc_prefix_lengths", None),
                inference_delay=kwargs.get("inference_delay", 0),
                execution_horizon=kwargs.get("execution_horizon", None),
                prefix_attention_schedule=kwargs.get(
                    "prefix_attention_schedule",
                    "exp",
                ),
                max_guidance_weight=kwargs.get("max_guidance_weight", 10.0),
            )
        action, _future_world = self.action_model.sample(**sample_kwargs)
        output = {"normalized_actions": action.float().cpu().numpy()}
        if event_output is not None:
            output.update(event_output)
        if planner_text is not None:
            output["planner_text"] = planner_text
            output["planner_text_refreshed"] = planner_text_refreshed
            if getattr(self, "text_history_enabled", False):
                output["planner_finished_task_list"] = [
                    self._finished_task_list_from_planner_text(text)
                    for text in planner_text
                ]
        return output
