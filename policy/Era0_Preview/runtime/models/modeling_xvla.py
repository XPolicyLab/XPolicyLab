# ------------------------------------------------------------------------------
# Copyright 2025 2toINF (https://github.com/2toINF)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import threading
import traceback
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Dict, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from transformers import PreTrainedModel
from .action_norm_routing import resolve_action_norm_key
from .action_conditioning import (
    apply_preprocessed_condition_masks,
    normalize_optional_mask,
    should_reapply_preprocessed_condition_masks,
)
from .transformer import SoftPromptedTransformer
from .action_hub import build_action_space
from .configuration_xvla import XVLAConfig
from .vlm_backends import build_vlm_backend

try:
    from transformers.conversion_mapping import (
        get_checkpoint_conversion_mapping,
        register_checkpoint_conversion_mapping,
    )
    from transformers.core_model_loading import (
        ConversionOps,
        WeightConverter,
    )
except ImportError:
    ConversionOps = None


if ConversionOps is not None:
    class _ReshapeToModelParameter(ConversionOps):
        """Adapt legacy scalar parameters before Transformers shape checks."""

        @torch.no_grad()
        def convert(
            self,
            input_dict,
            source_patterns,
            target_patterns,
            **kwargs,
        ):
            tensor = next(iter(input_dict.values()))
            tensor = tensor[0] if isinstance(tensor, list) else tensor
            expected = kwargs["model"].get_parameter(
                kwargs["full_layer_name"]
            )
            return {
                kwargs["full_layer_name"]: (
                    tensor.reshape(expected.shape).contiguous()
                )
            }

        @property
        def reverse_op(self):
            return _ReshapeToModelParameter()


    _xvla_conversions = list(
        get_checkpoint_conversion_mapping("XVLA") or []
    )
    _raw_gate_pattern = (
        r"transformer\.context_cross_attn\.(\d+)\.raw_gate"
    )
    _raw_gate_target = r"transformer.context_cross_attn.\1.raw_gate"
    if not any(
        _raw_gate_pattern in conversion.source_patterns
        for conversion in _xvla_conversions
    ):
        _xvla_conversions.append(
            WeightConverter(
                source_patterns=_raw_gate_pattern,
                target_patterns=_raw_gate_target,
                operations=[_ReshapeToModelParameter()],
            )
        )
        register_checkpoint_conversion_mapping(
            "XVLA",
            _xvla_conversions,
            overwrite=bool(
                get_checkpoint_conversion_mapping("XVLA")
            ),
        )


def _prepare_generate_action_masks(
    *,
    proprio: torch.Tensor,
    proprio_valid_mask: torch.Tensor | None,
    action_valid_mask: torch.Tensor | None,
    num_actions: int,
    dim_action: int,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    proprio_valid_mask = normalize_optional_mask(
        proprio_valid_mask,
        ref=proprio,
        name="proprio_valid_mask",
    )
    if proprio_valid_mask is not None:
        proprio = proprio * proprio_valid_mask

    action_ref = proprio.new_empty((proprio.shape[0], int(num_actions), int(dim_action)))
    action_valid_mask = normalize_optional_mask(
        action_valid_mask,
        ref=action_ref,
        name="action_valid_mask",
    )
    return proprio, proprio_valid_mask, action_valid_mask


def _normalize_dorsal_history_valid_mask(
    value: Any,
    *,
    num_frames: int,
    num_payload_views: int,
    num_model_views: int,
) -> torch.Tensor:
    """Map request-level history validity onto the model's padded view slots."""
    if num_payload_views <= 0 or num_payload_views > num_model_views:
        raise ValueError(
            "Dorsal payload camera count must be within the model view capacity: "
            f"got {num_payload_views}, capacity {num_model_views}."
        )

    mask_np = np.array(value, dtype=np.bool_, copy=True)
    if mask_np.ndim == 1:
        if mask_np.shape != (num_frames,):
            raise ValueError(
                "dorsal_history_valid [T] length must match the checkpoint "
                f"history: expected {(num_frames,)}, got {mask_np.shape}."
            )
        mask_np = np.repeat(mask_np[:, None], num_payload_views, axis=1)
    elif mask_np.ndim != 2:
        raise ValueError(
            "dorsal_history_valid must be [T] or [T,V], got "
            f"{mask_np.shape}."
        )

    payload_shape = (num_frames, num_payload_views)
    model_shape = (num_frames, num_model_views)
    if mask_np.shape == model_shape:
        return torch.from_numpy(mask_np)
    if mask_np.shape != payload_shape:
        raise ValueError(
            "dorsal_history_valid must match the cameras present in the request "
            f"or the padded model slots: expected {payload_shape} or "
            f"{model_shape}, got {mask_np.shape}."
        )

    padded = np.zeros(model_shape, dtype=np.bool_)
    padded[:, :num_payload_views] = mask_np
    return torch.from_numpy(padded)


class XVLA(PreTrainedModel):
    """
    XVLA: HuggingFace-compatible Vision-Language-Action policy.

    Components:
      • Qwen3.5 visual language backbone
      • SoftPromptedTransformer (temporal/action head)
      • Action space (pre/post-processing)
    """
    config_class = XVLAConfig
    base_model_prefix = "xvla"
    supports_gradient_checkpointing = False

    def __init__(self, config: XVLAConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        if config.vlm_backend != "qwen3_5":
            raise ValueError("Era0-Preview evaluation requires its Qwen3.5 checkpoint.")

        # Core settings
        self.num_actions: int = config.num_actions
        self.use_proprio: bool = config.use_proprio
        self.action_mode: str = config.action_mode.lower()
        # Action space (dimensions + hooks)
        if config.action_mode.lower() == "auto":
            self.action_space = build_action_space(
                config.action_mode.lower(),
                real_dim=config.real_action_dim,
                max_dim=config.max_action_dim,
            )
        elif config.action_mode.lower() == "cross_emb":
            self.action_space = build_action_space(
                config.action_mode.lower(),
                base_dim=config.real_action_dim,
                max_dim=config.max_action_dim,
                ee_equivalent=getattr(config, "cross_emb_ee_equivalent", False),
                norm_bank_count=int(getattr(config, "action_norm_bank_count", 1) or 1),
            )
        elif config.action_mode.lower() == "ee6d_mobile15":
            self.action_space = build_action_space(
                config.action_mode.lower(),
                reweight_base_motion=bool(getattr(config, "reweight_base_motion", False)),
                base_scale_ratio=float(getattr(config, "base_scale_ratio", 3.0)),
                mode_scale_ratio=float(getattr(config, "mode_scale_ratio", 3.0)),
            )
        else:
            self.action_space = build_action_space(config.action_mode.lower())
        dim_action = self.action_space.dim_action
        dim_proprio = getattr(self.action_space, "dim_proprio", dim_action)
        self.action_loss_mode = self._normalize_action_loss_mode(
            getattr(config, "action_loss_mode", "regression")
        )
        self.keep_vlm_decoder = bool(getattr(config, "keep_vlm_decoder", False))
        self._reapply_preprocessed_condition_masks = (
            should_reapply_preprocessed_condition_masks(config.vlm_backend)
        )
        if hasattr(self.action_space, "configure_target_mode"):
            target_mode = "velocity" if self.action_loss_mode == "flow_matching" else "action"
            self.action_space.configure_target_mode(target_mode)
        if hasattr(self.action_space, "configure_loss_profile"):
            action_loss_profile = str(
                getattr(config, "action_loss_profile", "hy_normalized_mse")
            )
            # A native CrossEmb model historically used its physical-coordinate
            # weighted loss. The Hy adapter attests its normalized RoboDojo
            # contract explicitly; only that contract opts into uniform MSE.
            if (
                self.action_mode == "cross_emb"
                and not str(getattr(config, "hy_action_contract", "")).strip()
            ):
                action_loss_profile = "cross_emb_native"
            self.action_space.configure_loss_profile(
                action_loss_profile
            )
        if hasattr(self.action_space, "configure_gripper_loss"):
            self.action_space.configure_gripper_loss(
                str(getattr(config, "gripper_loss_type", "bce"))
            )
        if hasattr(self.action_space, "configure_keep_gripper_proprio"):
            self.action_space.configure_keep_gripper_proprio(
                bool(getattr(config, "keep_gripper_proprio", False))
            )
        if hasattr(self.action_space, "configure_regression_normalization"):
            self.action_space.configure_regression_normalization(
                bool(getattr(config, "ee6d_regression_normalization", False))
            )

        # Semantic VLM plugin. The module remains registered as ``vlm`` so
        # Florence checkpoints retain their exact historical state-dict keys.
        self._vlm_backend = build_vlm_backend(config.vlm_backend)
        self._vlm_backend.validate_runtime(config)
        self.vlm = self._vlm_backend.build_model(config)
        self.vlm_output_adapter = self._vlm_backend.build_output_adapter(
            config, self.vlm
        )
        self.vlm_geometry_adapter = self._vlm_backend.build_geometry_adapter(
            config, self.vlm
        )
        requested_context_mode = str(
            getattr(config, "action_context_mode", "auto") or "auto"
        ).strip().lower()
        if requested_context_mode == "auto":
            requested_context_mode = self._vlm_backend.action_context_mode
        config.action_context_mode = requested_context_mode
        requested_policy_contract = str(
            getattr(config, "vlm_policy_contract", "auto") or "auto"
        ).strip().lower()
        if requested_policy_contract == "auto":
            requested_policy_contract = self._vlm_backend.policy_contract
        config.vlm_policy_contract = requested_policy_contract
        projection_dim = self._vlm_backend.output_hidden_size(config, self.vlm)
        self.vlm_policy_modules = self._vlm_backend.build_policy_modules(
            config,
            model=self.vlm,
            output_dim=projection_dim,
        )
        self.far_aux_head = None

        # This release uses the self-contained Qwen policy and action expert.
        if bool(getattr(config, "dorsal_enabled", False)) or bool(getattr(config, "dino_enabled", False)):
            raise ValueError("Era0-Preview evaluation does not include auxiliary predictors.")
        self.dorsal = None
        transformer_dino = False

        # Construct the action expert with its checkpoint-compatible namespace.
        self.transformer = SoftPromptedTransformer(
            hidden_size=config.hidden_size,
            multi_modal_input_size=projection_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            num_domains=config.num_domains,
            dim_action=dim_action,
            dim_propio=dim_proprio,
            len_soft_prompts=config.len_soft_prompts,
            dim_time=config.dim_time,
            time_embedding_max_period=int(
                getattr(config, "time_embedding_max_period", 100)
            ),
            time_embedding_input_scale=float(
                getattr(config, "time_embedding_input_scale", 1.0)
            ),
            max_len_seq=config.max_len_seq,
            use_hetero_proj=config.use_hetero_proj,
            dino_enabled=transformer_dino,
            dino_target_dim=int(getattr(config, "dino_target_dim", 1024)),
            dino_num_latent_tokens=int(getattr(config, "dino_num_latent_tokens", 4)),
            dino_detach_action=bool(getattr(config, "dino_detach_action", False)),
            dino_pos_emb_decoupled=bool(getattr(config, "dino_pos_emb_decoupled", False)),
            dorsal_token_routing=bool(getattr(config, "dorsal_enabled", False)),
            dorsal_router_init_prob=float(
                getattr(config, "dorsal_router_init_prob", 0.1)
            ),
            action_context_mode=config.action_context_mode,
            context_cross_attn_every_n_layers=int(
                getattr(
                    config,
                    "action_context_cross_attn_every_n_layers",
                    4,
                )
            ),
            context_cross_attn_gate_init=float(
                getattr(config, "action_context_cross_attn_gate_init", 0.1)
            ),
            context_semantic_route=getattr(
                config,
                "action_context_semantic_route",
                None,
            ),
            history_memory_enabled=bool(
                getattr(config, "qwen_history_memory_enabled", False)
            ),
            history_memory_gate_init=float(
                getattr(config, "qwen_history_memory_gate_init", 0.0)
            ),
            history_memory_num_views=int(
                getattr(config, "vlm_max_view_embeddings", 8)
            ),
            num_kv_heads=getattr(config, "num_kv_heads", None),
            attn_head_dim=getattr(config, "attn_head_dim", None),
            time_adalayernorm_enabled=bool(
                getattr(config, "qwen_action_adalayernorm_enabled", False)
            ),
            )
        if getattr(config, "vlm_lora", None) is not None:
            from .vlm_lora import install_vlm_lora
            install_vlm_lora(self, config.vlm_lora)

        # Transformers 5 requires the top-level composite model to expose the
        # tied-weight mapping collected from child PreTrainedModel instances.
        # Calling post_init() here would reinitialize existing XVLA modules, so
        # collect only the static mapping needed by from_pretrained().
        if hasattr(self, "get_expanded_tied_weights_keys"):
            self.all_tied_weights_keys = self.get_expanded_tied_weights_keys(
                all_submodels=True
            )

    # ------------------------------------------------------------------
    # Action-norm bank shape contract
    # ------------------------------------------------------------------
    # ``CrossEmbActionSpace`` keeps per-dataset norm banks in buffers shaped
    # (num_banks, dim_action). num_banks comes from --action_norm_stats at
    # runtime, so a checkpoint trained with 33 banks cannot be rebuilt from
    # config alone: from_pretrained would construct 1-row buffers and the
    # strict shape check would reject the checkpoint (this is exactly what
    # broke every resume from a multi-bank PT checkpoint). The count is now
    # serialized in XVLAConfig; for checkpoints written before that field
    # existed we probe the safetensors header, which is O(header) and needs no
    # weight load.
    _NORM_BANK_PROBE_KEY = "action_space._bank_mean"

    @classmethod
    def _probe_checkpoint_norm_bank_count(cls, path) -> int | None:
        """Return the checkpoint's norm-bank row count, or None if unknown."""
        if path is None:
            return None
        base = str(path)
        if not os.path.isdir(base):
            return None
        import json as _json
        import struct as _struct

        shard = os.path.join(base, "model.safetensors")
        index = os.path.join(base, "model.safetensors.index.json")
        if os.path.isfile(index):
            try:
                with open(index, "r", encoding="utf-8") as handle:
                    weight_map = _json.load(handle).get("weight_map", {})
            except Exception:
                return None
            name = weight_map.get(cls._NORM_BANK_PROBE_KEY)
            if not name:
                return None
            shard = os.path.join(base, name)
        if not os.path.isfile(shard):
            return None
        try:
            with open(shard, "rb") as handle:
                header_len = _struct.unpack("<Q", handle.read(8))[0]
                header = _json.loads(handle.read(header_len))
        except Exception:
            return None
        shape = (header.get(cls._NORM_BANK_PROBE_KEY) or {}).get("shape")
        if isinstance(shape, (list, tuple)) and len(shape) == 2:
            return max(1, int(shape[0]))
        return None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path=None, *model_args, **kwargs):
        bank_count = cls._probe_checkpoint_norm_bank_count(pretrained_model_name_or_path)
        if bank_count is not None and bank_count > 1:
            config = kwargs.get("config")
            if config is None:
                # Checkpoints saved by current code already carry the count in
                # config.json, so only legacy ones need us to materialize it.
                try:
                    config = cls.config_class.from_pretrained(
                        pretrained_model_name_or_path,
                        **{
                            key: kwargs[key]
                            for key in ("trust_remote_code", "revision", "cache_dir", "token")
                            if key in kwargs
                        },
                    )
                except Exception:
                    config = None
                if config is not None and int(
                    getattr(config, "action_norm_bank_count", 1) or 1
                ) != bank_count:
                    config.action_norm_bank_count = bank_count
                    kwargs["config"] = config
            elif int(getattr(config, "action_norm_bank_count", 1) or 1) != bank_count:
                config.action_norm_bank_count = bank_count
        loaded = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        model = loaded[0] if isinstance(loaded, tuple) else loaded
        if getattr(model.config, "vlm_lora", None) is not None:
            from .vlm_lora import install_vlm_lora
            install_vlm_lora(model, model.config.vlm_lora)
        return loaded

    def _sync_action_norm_bank_config(self) -> None:
        """Keep config.action_norm_bank_count equal to the live buffer rows."""
        bank_mean = getattr(getattr(self, "action_space", None), "_bank_mean", None)
        if isinstance(bank_mean, torch.Tensor) and bank_mean.ndim == 2:
            self.config.action_norm_bank_count = max(1, int(bank_mean.shape[0]))


    @property
    def dtype(self) -> torch.dtype:
        """Report the outer policy dtype for mixed-precision checkpoint I/O."""
        transformer = getattr(self, "transformer", None)
        if transformer is not None:
            parameter = next(transformer.parameters(), None)
            if parameter is not None:
                return parameter.dtype
        vlm = getattr(self, "vlm", None)
        if vlm is not None:
            parameter = next(vlm.parameters(), None)
            if parameter is not None:
                return parameter.dtype
        return torch.get_default_dtype()

    @staticmethod
    def _normalize_action_loss_mode(mode: str) -> str:
        mode_norm = str(mode or "regression").strip().lower().replace("-", "_")
        if mode_norm in {"flow", "fm"}:
            mode_norm = "flow_matching"
        if mode_norm not in {"regression", "flow_matching"}:
            mode_norm = "regression"
        return mode_norm

    def set_action_loss_mode(self, mode: str) -> None:
        self.action_loss_mode = self._normalize_action_loss_mode(mode)
        if hasattr(self.action_space, "configure_target_mode"):
            target_mode = "velocity" if self.action_loss_mode == "flow_matching" else "action"
            self.action_space.configure_target_mode(target_mode)

    def _has_vlm_decoder(self) -> bool:
        backend = getattr(self, "_vlm_backend", None)
        if backend is not None:
            return backend.has_lm_head(self.vlm)
        language_model = getattr(self.vlm, "language_model", None)
        return bool(
            language_model is not None
            and hasattr(language_model, "model")
            and hasattr(language_model.model, "decoder")
            and hasattr(language_model, "lm_head")
        )

    def get_vlm_backend_state(self) -> Dict[str, Any]:
        """Return the effective plugin contract for logs and deployment checks."""
        return self._vlm_backend.describe(self.config, self.vlm)


    @staticmethod
    def _select_primary_view(
        image_input: torch.Tensor,
        image_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if image_input.dim() == 4:
            return image_input
        if image_input.dim() != 5:
            raise ValueError(
                f"image_input must be [B,C,H,W] or [B,V,C,H,W], got shape={tuple(image_input.shape)}"
            )
        batch_size, num_views = image_input.shape[:2]
        if num_views <= 0:
            raise ValueError("image_input has zero views.")
        if image_mask is None:
            return image_input[:, 0]
        if image_mask.dim() != 2 or image_mask.shape[0] != batch_size:
            raise ValueError(
                f"image_mask must be [B,V], got shape={tuple(image_mask.shape)} for image_input={tuple(image_input.shape)}"
            )
        mask = image_mask.to(device=image_input.device, dtype=torch.bool)
        if mask.shape[1] < num_views:
            raise ValueError(
                f"image_mask view dim ({mask.shape[1]}) must cover image_input views ({num_views})."
            )
        first_valid = torch.zeros(batch_size, dtype=torch.long, device=image_input.device)
        has_valid = mask.any(dim=1)
        if bool(has_valid.any()):
            first_valid[has_valid] = torch.argmax(mask[has_valid].to(torch.int64), dim=1)
        batch_idx = torch.arange(batch_size, device=image_input.device)
        return image_input[batch_idx, first_valid]


    @staticmethod
    def _normalize_view_mask(
        image_mask: torch.Tensor | None,
        *,
        batch_size: int,
        num_views: int,
        device: torch.device,
        name: str,
        require_non_empty: bool,
    ) -> torch.Tensor:
        if image_mask is None:
            mask = torch.ones((batch_size, num_views), dtype=torch.bool, device=device)
        else:
            mask = torch.as_tensor(image_mask, device=device, dtype=torch.bool)
            if tuple(mask.shape) != (batch_size, num_views):
                raise ValueError(f"{name} must be {(batch_size, num_views)}, got {tuple(mask.shape)}")
        if require_non_empty and not bool(mask.any(dim=1).all()):
            raise ValueError(f"{name} requires at least one valid view per sample.")
        return mask

    @staticmethod
    def _normalize_temporal_mask(
        image_mask_seq: torch.Tensor | None,
        *,
        batch_size: int,
        num_frames: int,
        num_views: int,
        device: torch.device,
        require_non_empty: bool,
    ) -> torch.Tensor:
        if image_mask_seq is None:
            mask = torch.ones((batch_size, num_frames, num_views), dtype=torch.bool, device=device)
        else:
            mask = torch.as_tensor(image_mask_seq, device=device, dtype=torch.bool)
            if mask.dim() == 2 and tuple(mask.shape) == (batch_size, num_frames):
                mask = mask.unsqueeze(-1).expand(batch_size, num_frames, num_views)
            elif tuple(mask.shape) != (batch_size, num_frames, num_views):
                raise ValueError(
                    "image_mask_seq must be "
                    f"{(batch_size, num_frames, num_views)} or {(batch_size, num_frames)}, got {tuple(mask.shape)}"
                )
        if require_non_empty and not bool(mask.any(dim=1).any(dim=1).all()):
            raise ValueError("image_mask_seq requires at least one valid temporal view per sample.")
        return mask

    @staticmethod
    def _reorder_visual_features(
        visual_features: torch.Tensor,
        visual_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_views = visual_mask.shape
        primary = torch.argmax(visual_mask.to(torch.int64), dim=1)
        if num_views == 1:
            order_tensor = primary.unsqueeze(1)
        else:
            tail = torch.arange(
                num_views - 1,
                device=visual_mask.device,
                dtype=torch.long,
            ).expand(batch_size, -1)
            tail = tail + (tail >= primary.unsqueeze(1)).to(dtype=torch.long)
            order_tensor = torch.cat((primary.unsqueeze(1), tail), dim=1)
        gather_index = order_tensor.unsqueeze(-1).unsqueeze(-1).expand_as(visual_features)
        ordered_features = torch.gather(visual_features, dim=1, index=gather_index)
        ordered_mask = torch.gather(visual_mask, dim=1, index=order_tensor)
        return ordered_features, ordered_mask

    def _encode_pixel_features(
        self,
        pixel_values: torch.Tensor,
        *,
        frame_mask: torch.Tensor | None = None,
        mem_video: bool = False,
    ) -> torch.Tensor:
        backend = getattr(self, "_vlm_backend", None)
        if (
            mem_video
            and getattr(self.config, "vlm_backend", "florence2") == "florence2"
            and hasattr(self.vlm, "_encode_image")
        ):
            return self.vlm_output_adapter(
                self.vlm._encode_image(pixel_values, mem_video=True)
            )
        if backend is None:
            return self.vlm._encode_image(pixel_values, frame_mask=frame_mask)
        return backend.encode_image(self, pixel_values, frame_mask=frame_mask)

    def _encode_current_visual_features(
        self,
        image_input: torch.Tensor | None,
        image_mask: torch.Tensor | None,
        *,
        require_non_empty: bool,
        image_hist_size: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if image_input is None:
            return None, None
        if image_input.dim() == 4:
            image_input = image_input.unsqueeze(1)
        if image_input.dim() == 6:
            batch_size, num_views, history_len, channels, height, width = image_input.shape
            if image_mask is None:
                frame_mask = torch.ones(
                    (batch_size, num_views, history_len),
                    dtype=torch.bool,
                    device=image_input.device,
                )
            else:
                frame_mask = torch.as_tensor(image_mask, device=image_input.device, dtype=torch.bool)
                if tuple(frame_mask.shape) != (batch_size, num_views, history_len):
                    raise ValueError(
                        "image_mask for 6D image_input must be [B,V,K], got "
                        f"{tuple(frame_mask.shape)} for image_input={tuple(image_input.shape)}"
                    )
            view_mask = frame_mask.any(dim=2)
            if require_non_empty and not bool(view_mask.any(dim=1).all()):
                raise ValueError("image_mask requires at least one valid view per sample.")
            if history_len == 1:
                return self._encode_current_visual_features(
                    image_input[:, :, 0],
                    view_mask,
                    require_non_empty=require_non_empty,
                )

            flat_stacks = image_input.reshape(batch_size * num_views, history_len, channels, height, width)
            flat_mask = frame_mask.reshape(batch_size * num_views, history_len)
            valid_slots = view_mask.reshape(-1)
            if not bool(valid_slots.any()):
                return None, view_mask

            if image_hist_size is None:
                valid_features = self._encode_pixel_features(
                    flat_stacks[valid_slots],
                    frame_mask=flat_mask[valid_slots],
                    mem_video=True,
                )
                num_tokens, hidden_dim = int(valid_features.shape[1]), int(valid_features.shape[2])
                image_features = valid_features.new_zeros(
                    (batch_size * num_views, num_tokens, hidden_dim)
                )
                image_features[valid_slots] = valid_features
                return image_features.view(batch_size, num_views, num_tokens, hidden_dim), view_mask

            hist_size = torch.as_tensor(image_hist_size, device=image_input.device, dtype=torch.long)
            if tuple(hist_size.shape) != (batch_size, num_views):
                raise ValueError(
                    "image_hist_size must be [B,V] to match image_input, got "
                    f"{tuple(hist_size.shape)} for image_input={tuple(image_input.shape[:2])}"
                )
            flat_hist = hist_size.reshape(-1).clamp(min=1, max=history_len)
            image_features: torch.Tensor | None = None
            num_tokens = hidden_dim = 0
            for kv in sorted(int(v) for v in torch.unique(flat_hist[valid_slots]).detach().cpu().tolist()):
                group = valid_slots & (flat_hist == kv)
                if not bool(group.any()):
                    continue
                pixels = flat_stacks[group, -kv:]
                mask = flat_mask[group, -kv:]
                if kv == 1:
                    encoded = self._encode_pixel_features(pixels[:, 0])
                else:
                    encoded = self._encode_pixel_features(
                        pixels,
                        frame_mask=mask,
                        mem_video=True,
                    )
                if image_features is None:
                    num_tokens, hidden_dim = int(encoded.shape[1]), int(encoded.shape[2])
                    image_features = encoded.new_zeros(
                        (batch_size * num_views, num_tokens, hidden_dim)
                    )
                elif encoded.shape[1:] != (num_tokens, hidden_dim):
                    raise RuntimeError(
                        "history visual encoding produced inconsistent token shapes: "
                        f"got {tuple(encoded.shape[1:])}, expected {(num_tokens, hidden_dim)}"
                    )
                image_features[group] = encoded
            if image_features is None:
                return None, view_mask
            return image_features.view(batch_size, num_views, num_tokens, hidden_dim), view_mask
        if image_input.dim() != 5:
            raise ValueError(
                f"image_input must be [B,C,H,W], [B,V,C,H,W] or [B,V,K,C,H,W], got shape={tuple(image_input.shape)}"
            )
        batch_size, num_views, channels, height, width = image_input.shape
        mask = self._normalize_view_mask(
            image_mask,
            batch_size=batch_size,
            num_views=num_views,
            device=image_input.device,
            name="image_mask",
            require_non_empty=require_non_empty,
        )
        flat_pixels = image_input.view(batch_size * num_views, channels, height, width)
        valid_slots = mask.reshape(-1)
        if not bool(valid_slots.any()):
            return None, mask
        valid_features = self._encode_pixel_features(flat_pixels[valid_slots])
        num_tokens, hidden_dim = int(valid_features.shape[1]), int(valid_features.shape[2])
        image_features = valid_features.new_zeros((batch_size * num_views, num_tokens, hidden_dim))
        image_features[valid_slots] = valid_features
        return image_features.view(batch_size, num_views, num_tokens, hidden_dim), mask

    def _encode_temporal_visual_features(
        self,
        image_seq_input: torch.Tensor | None,
        image_mask_seq: torch.Tensor | None,
        *,
        require_non_empty: bool,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if image_seq_input is None:
            return None, None
        if image_seq_input.dim() == 5:
            image_seq_input = image_seq_input.unsqueeze(2)
        if image_seq_input.dim() != 6:
            raise ValueError(
                "image_seq_input must be [B,T,C,H,W] or [B,T,V,C,H,W], "
                f"got shape={tuple(image_seq_input.shape)}"
            )
        batch_size, num_frames, num_views, channels, height, width = image_seq_input.shape
        mask_seq = self._normalize_temporal_mask(
            image_mask_seq,
            batch_size=batch_size,
            num_frames=num_frames,
            num_views=num_views,
            device=image_seq_input.device,
            require_non_empty=require_non_empty,
        )
        flat_seq = image_seq_input.permute(0, 2, 1, 3, 4, 5).reshape(
            batch_size * num_views, num_frames, channels, height, width
        )
        flat_mask = mask_seq.permute(0, 2, 1).reshape(batch_size * num_views, num_frames)
        valid_slots = flat_mask.any(dim=1)
        view_mask = valid_slots.view(batch_size, num_views)
        if not bool(valid_slots.any()):
            return None, view_mask
        valid_features = self._encode_pixel_features(
            flat_seq[valid_slots],
            frame_mask=flat_mask[valid_slots],
        )
        num_tokens, hidden_dim = int(valid_features.shape[1]), int(valid_features.shape[2])
        seq_features = valid_features.new_zeros((batch_size * num_views, num_tokens, hidden_dim))
        seq_features[valid_slots] = valid_features
        return seq_features.view(batch_size, num_views, num_tokens, hidden_dim), view_mask

    def _collect_visual_features(
        self,
        *,
        image_input: torch.Tensor | None,
        image_mask: torch.Tensor | None,
        image_seq_input: torch.Tensor | None,
        image_mask_seq: torch.Tensor | None,
        require_visual: bool,
        image_hist_size: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        current_features, current_mask = self._encode_current_visual_features(
            image_input,
            image_mask,
            require_non_empty=False,
            image_hist_size=image_hist_size,
        )
        temporal_features, temporal_mask = self._encode_temporal_visual_features(
            image_seq_input,
            image_mask_seq,
            require_non_empty=False,
        )
        if temporal_features is not None and temporal_mask is not None:
            if current_features is None or current_mask is None:
                current_features, current_mask = temporal_features, temporal_mask
            else:
                if tuple(current_features.shape) != tuple(temporal_features.shape):
                    raise ValueError(
                        "image_input and image_seq_input must encode to the same shape when both are provided, "
                        f"got current={tuple(current_features.shape)} temporal={tuple(temporal_features.shape)}"
                    )
                if tuple(current_mask.shape) != tuple(temporal_mask.shape):
                    raise ValueError(
                        "image_mask and image_mask_seq must align on encoded view shape when both are provided, "
                        f"got current={tuple(current_mask.shape)} temporal={tuple(temporal_mask.shape)}"
                    )
                merged_features = current_features.clone()
                temporal_slots = temporal_mask.unsqueeze(-1).unsqueeze(-1)
                merged_features = torch.where(temporal_slots, temporal_features, merged_features)
                current_features = merged_features
                current_mask = current_mask | temporal_mask
        if current_features is None or current_mask is None:
            if require_visual:
                raise ValueError("At least one visual input is required.")
            return None, None
        if require_visual and not bool(current_mask.any(dim=1).all()):
            raise ValueError("At least one visual stream must be valid per sample.")
        return current_features, current_mask

    def _build_vlm_encoder_inputs(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None,
        image_input: torch.Tensor | None,
        image_mask: torch.Tensor | None,
        image_seq_input: torch.Tensor | None,
        image_mask_seq: torch.Tensor | None,
        flatten_all_visual_tokens: bool,
        require_visual: bool,
        precomputed_features: torch.Tensor | None = None,
        image_hist_size: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        inputs_embeds = self.vlm.get_input_embeddings()(input_ids)
        text_attention = (
            torch.ones(input_ids.shape, dtype=torch.long, device=inputs_embeds.device)
            if attention_mask is None
            else attention_mask.to(device=inputs_embeds.device, dtype=torch.long)
        )

        # Shared-DaViT fast path: when the caller already encoded the vision
        # tower for this exact `image_input` (same images across the action
        # expert + the VLM CE supervisions), reuse those raw features and skip
        # the redundant `self.vlm._encode_image`. Mask is still derived from
        # this path's own `image_mask`, so per-path masking is preserved and
        # the result is numerically identical to encoding here. Only valid when
        # there is no temporal stream (precompute covers `image_input` only).
        if precomputed_features is not None and image_seq_input is None:
            visual_features = precomputed_features
            batch_size, num_views = int(visual_features.shape[0]), int(visual_features.shape[1])
            if image_mask is not None and image_mask.dim() == 3:
                image_mask = torch.as_tensor(image_mask, device=visual_features.device, dtype=torch.bool).any(dim=2)
            visual_mask = self._normalize_view_mask(
                image_mask,
                batch_size=batch_size,
                num_views=num_views,
                device=visual_features.device,
                name="image_mask",
                require_non_empty=require_visual,
            )
            if require_visual and not bool(visual_mask.any(dim=1).all()):
                raise ValueError("At least one visual stream must be valid per sample.")
        else:
            visual_features, visual_mask = self._collect_visual_features(
                image_input=image_input,
                image_mask=image_mask,
                image_seq_input=image_seq_input,
                image_mask_seq=image_mask_seq,
                require_visual=require_visual,
                image_hist_size=image_hist_size,
            )
        if visual_features is None or visual_mask is None:
            return {
                "inputs_embeds": inputs_embeds,
                "attention_mask": text_attention,
                "vlm_feature_mask": text_attention.to(dtype=torch.bool),
                "aux_visual_inputs": inputs_embeds.new_zeros(
                    (input_ids.shape[0], 0, inputs_embeds.shape[-1])
                ),
                "aux_visual_mask": torch.zeros(
                    (input_ids.shape[0], 0),
                    device=inputs_embeds.device,
                    dtype=torch.bool,
                ),
                "dorsal_visual_inputs": inputs_embeds.new_zeros(
                    (input_ids.shape[0], 0, inputs_embeds.shape[-1])
                ),
            }

        batch_size, num_views, num_tokens, hidden_dim = visual_features.shape
        ordered_visual, ordered_mask = self._reorder_visual_features(
            visual_features, visual_mask
        )
        primary_visual = ordered_visual[:, 0]
        if flatten_all_visual_tokens:
            visual_token_mask = visual_mask.unsqueeze(-1).expand(batch_size, num_views, num_tokens).reshape(
                batch_size, num_views * num_tokens
            )
            flat_visual = visual_features.reshape(batch_size, num_views * num_tokens, hidden_dim)
            flat_visual = flat_visual * visual_token_mask.unsqueeze(-1).to(dtype=flat_visual.dtype)
            return {
                "inputs_embeds": torch.cat([flat_visual, inputs_embeds], dim=1),
                "attention_mask": torch.cat(
                    [visual_token_mask.to(dtype=text_attention.dtype), text_attention],
                    dim=1,
                ),
                "aux_visual_inputs": inputs_embeds.new_zeros((batch_size, 0, hidden_dim)),
                "dorsal_visual_inputs": primary_visual,
            }

        aux_visual = ordered_visual[:, 1:].reshape(batch_size, -1, hidden_dim)
        aux_visual_mask = ordered_mask[:, 1:].unsqueeze(-1).expand(
            batch_size,
            max(0, num_views - 1),
            num_tokens,
        ).reshape(batch_size, -1)
        aux_visual = aux_visual * aux_visual_mask.unsqueeze(-1).to(dtype=aux_visual.dtype)
        primary_attention = torch.ones(
            (batch_size, primary_visual.shape[1]),
            dtype=text_attention.dtype,
            device=text_attention.device,
        )
        return {
            "inputs_embeds": torch.cat([primary_visual, inputs_embeds], dim=1),
            "attention_mask": torch.cat([primary_attention, text_attention], dim=1),
            "vlm_feature_mask": torch.cat([primary_attention, text_attention], dim=1).to(dtype=torch.bool),
            "aux_visual_inputs": aux_visual,
            "aux_visual_mask": aux_visual_mask,
            "dorsal_visual_inputs": primary_visual,
        }

    # ============================= KI gradient isolation =============================


    def _apply_inference_dorsal(self, enc: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Drop unused auxiliary streams before action inference."""
        out = dict(enc)
        for key in ("dorsal_visual_inputs", "dorsal_visual_mask", "dorsal_motion_inputs"):
            out.pop(key, None)
        return out

    def forward_vlm(
        self,
        input_ids: torch.LongTensor,        # [B, L]
        pixel_values: torch.FloatTensor,    # [B, V, C, H, W]
        image_mask: torch.Tensor,           # [B, V] (bool or 0/1)
        attention_mask: torch.Tensor | None = None,  # [B, L]
        token_type_ids: torch.Tensor | None = None,
        precomputed_features: Any = None,
        image_hist_size: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Encode policy image/text inputs through the configured VLM plugin."""
        if (
            image_hist_size is not None
            and getattr(self.config, "vlm_backend", "florence2") == "florence2"
        ):
            return self._forward_vlm_florence(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_mask=image_mask,
                attention_mask=attention_mask,
                precomputed_features=precomputed_features,
                image_hist_size=image_hist_size,
            )
        backend_kwargs = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "image_mask": image_mask,
            "attention_mask": attention_mask,
            "precomputed_features": precomputed_features,
        }
        if self.config.vlm_backend == "dm05":
            backend_kwargs["token_type_ids"] = token_type_ids
        return self._vlm_backend.forward_policy(self, **backend_kwargs)

    def _forward_vlm_florence(
        self,
        *,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        image_mask: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        precomputed_features: torch.Tensor | None = None,
        image_hist_size: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Legacy Florence policy path used by the compatibility plugin."""
        shared = self._build_vlm_encoder_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_input=pixel_values,
            image_mask=image_mask,
            image_seq_input=None,
            image_mask_seq=None,
            flatten_all_visual_tokens=False,
            require_visual=True,
            precomputed_features=precomputed_features,
            image_hist_size=image_hist_size,
        )

        enc_out = self.vlm.language_model.model.encoder(
            attention_mask=shared["attention_mask"],
            inputs_embeds=shared["inputs_embeds"],
        )[0]  # [B, T_enc, D]

        return {
            "vlm_features": enc_out,
            "vlm_feature_mask": shared["vlm_feature_mask"],
            "aux_visual_inputs": shared["aux_visual_inputs"],
            "aux_visual_mask": shared["aux_visual_mask"],
            "dorsal_visual_inputs": shared["dorsal_visual_inputs"],
        }


    # ============================== inference helpers =============================
    def _extract_transformer_output(self, transformer_out):
        """Unpack transformer output which may be a dict (dino mode) or plain tensor."""
        if isinstance(transformer_out, dict):
            return transformer_out["action_pred"], transformer_out.get("dino_pred")
        return transformer_out, None


    # ================================= inference =================================
    @torch.no_grad()
    def generate_actions(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        domain_id: torch.LongTensor,
        proprio: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        proprio_valid_mask: torch.Tensor | None = None,
        action_valid_mask: torch.Tensor | None = None,
        dorsal_history_image_input: torch.Tensor | None = None,
        dorsal_history_mask: torch.Tensor | None = None,
        dorsal_history_time_offsets: torch.Tensor | None = None,
        dorsal_history_cache_keys: torch.Tensor | None = None,
        steps: int = 10,
        action_norm_key=None,
        initial_noise: torch.Tensor | None = None,
        image_hist_size: torch.Tensor | None = None,
        apply_postprocess: bool = True,
    ) -> torch.Tensor:
        """
        Iterative denoising (linear schedule).
        Applies action_space.postprocess at the end (e.g., sigmoid on gripper).
        """
        self.eval()
        reapply_preprocessed_condition_masks = bool(
            getattr(
                self,
                "_reapply_preprocessed_condition_masks",
                should_reapply_preprocessed_condition_masks(
                    getattr(getattr(self, "config", None), "vlm_backend", "florence2")
                ),
            )
        )
        enc = self.forward_vlm(
            input_ids,
            image_input,
            image_mask,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            image_hist_size=image_hist_size,
        )
        enc = self._apply_inference_dorsal(enc)
        if getattr(self.transformer, "prefix_kv", False):
            # Prefill the native VLM prefix K/V once so the Euler loop reuses
            # it across every flow-matching step instead of re-running the VLM.
            if "prefix_kv_by_layer" not in enc:
                raise RuntimeError(
                    "prefix_kv inference requires prefix_kv_by_layer from the "
                    "Qwen policy forward."
                )
            enc["prepared_context"] = (
                self.transformer.prepare_prefix_kv_context(
                    prefix_kv_by_layer=enc.pop("prefix_kv_by_layer"),
                    prefix_kv_mask=enc.pop("prefix_kv_mask"),
                    prefix_kv_position_offset=enc.pop("prefix_kv_position_offset", None),
                )
            )
        elif getattr(self.transformer, "context_readonly", False):
            enc["prepared_context"] = (
                self.transformer.prepare_readonly_context(
                    domain_id=domain_id,
                    vlm_features=enc["vlm_features"],
                    aux_visual_inputs=enc["aux_visual_inputs"],
                    vlm_feature_mask=enc.get("vlm_feature_mask"),
                    aux_visual_mask=enc.get("aux_visual_mask"),
                    memory_features=enc.get("memory_features"),
                    memory_mask=enc.get("memory_mask"),
                    memory_time_offsets=enc.get("memory_time_offsets"),
                    memory_view_ids=enc.get("memory_view_ids"),
                )
            )

        B = input_ids.shape[0]
        D = self.action_space.dim_action
        proprio, proprio_valid_mask, action_valid_mask = _prepare_generate_action_masks(
            proprio=proprio,
            proprio_valid_mask=proprio_valid_mask,
            action_valid_mask=action_valid_mask,
            num_actions=self.num_actions,
            dim_action=D,
        )

        steps = max(1, int(steps))
        expected_noise_shape = (B, self.num_actions, D)
        if initial_noise is not None:
            if tuple(initial_noise.shape) != expected_noise_shape:
                raise ValueError(
                    "initial_noise shape mismatch: "
                    f"got={tuple(initial_noise.shape)} expected={expected_noise_shape}"
                )
            initial_noise = initial_noise.to(
                device=proprio.device,
                dtype=proprio.dtype,
            )
        if self.action_loss_mode == "flow_matching":
            # Euler integration in normalized action space.
            dummy_action = torch.zeros(B, self.num_actions, D, device=proprio.device, dtype=proprio.dtype)
            proprio_m, _ = self.action_space.preprocess(proprio, dummy_action, norm_key=action_norm_key)
            if reapply_preprocessed_condition_masks:
                proprio_m, _, _, _ = apply_preprocessed_condition_masks(
                    proprio=proprio_m,
                    action=dummy_action,
                    proprio_valid_mask=proprio_valid_mask,
                    action_valid_mask=None,
                )
            action = (
                initial_noise.clone()
                if initial_noise is not None
                else torch.randn(
                    B,
                    self.num_actions,
                    D,
                    device=proprio.device,
                    dtype=proprio.dtype,
                )
            )
            if action_valid_mask is not None:
                action = action * action_valid_mask
            dt = 1.0 / steps
            for i in range(steps, 0, -1):
                t = torch.full((B,), i / steps, device=proprio.device, dtype=proprio.dtype)
                raw_out = self.transformer(
                    domain_id=domain_id,
                    action_with_noise=action,
                    proprio=proprio_m,
                    t=t,
                    **enc,
                )
                velocity, _ = self._extract_transformer_output(raw_out)
                if action_valid_mask is not None:
                    velocity = velocity * action_valid_mask
                action = action - velocity * dt
                if action_valid_mask is not None:
                    action = action * action_valid_mask
            out = (
                self.action_space.postprocess(action, norm_key=action_norm_key)
                if apply_postprocess
                else action
            )
            if action_valid_mask is not None:
                out = out * action_valid_mask
            return out

        x1 = (
            initial_noise.clone()
            if initial_noise is not None
            else torch.randn(
                B,
                self.num_actions,
                D,
                device=proprio.device,
                dtype=proprio.dtype,
            )
        )
        if action_valid_mask is not None:
            x1 = x1 * action_valid_mask
        action_state = torch.zeros_like(x1)
        pred_action = action_state

        for i in range(steps, 0, -1):
            t = torch.full((B,), i / steps, device=proprio.device, dtype=proprio.dtype)
            x_t = x1 * t.view(-1, 1, 1) + action_state * (1 - t).view(-1, 1, 1)
            if action_valid_mask is not None:
                x_t = x_t * action_valid_mask
            proprio_m, x_t_m = self.action_space.preprocess(proprio, x_t, norm_key=action_norm_key)
            if reapply_preprocessed_condition_masks:
                proprio_m, x_t_m, _, _ = apply_preprocessed_condition_masks(
                    proprio=proprio_m,
                    action=x_t_m,
                    proprio_valid_mask=proprio_valid_mask,
                    action_valid_mask=action_valid_mask,
                )
            raw_out = self.transformer(
                domain_id=domain_id,
                action_with_noise=x_t_m,
                proprio=proprio_m,
                t=t,
                **enc,
            )
            pred_action, _ = self._extract_transformer_output(raw_out)
            if action_valid_mask is not None:
                pred_action = pred_action * action_valid_mask
            action_state = self.action_space.to_diffusion_state(pred_action, norm_key=action_norm_key)
            if action_valid_mask is not None:
                action_state = action_state * action_valid_mask
        out = (
            self.action_space.postprocess(pred_action, norm_key=action_norm_key)
            if apply_postprocess
            else pred_action
        )
        if action_valid_mask is not None:
            out = out * action_valid_mask
        return out
