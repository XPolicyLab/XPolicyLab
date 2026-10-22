#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

logger = logging.getLogger(__name__)
_DA3_REFERENCE_LOG_PATCHED = False


def patch_da3_reference_view_logger_once() -> None:
    global _DA3_REFERENCE_LOG_PATCHED
    if _DA3_REFERENCE_LOG_PATCHED:
        return

    try:
        vision_transformer = importlib.import_module("depth_anything_3.model.dinov2.vision_transformer")
    except ImportError:
        return

    vt_logger = getattr(vision_transformer, "logger", None)
    if vt_logger is None or not hasattr(vt_logger, "info"):
        return

    original_info = vt_logger.info
    seen_reference_log = False

    def info_once(*args, **kwargs):
        nonlocal seen_reference_log
        if args:
            first_arg = str(args[0])
            if first_arg.startswith("Selecting reference view using strategy:"):
                if seen_reference_log:
                    return
                seen_reference_log = True
        return original_info(*args, **kwargs)

    vt_logger.info = info_once
    _DA3_REFERENCE_LOG_PATCHED = True


DA3_BACKBONE_DEFAULTS = {
    "large": {
        "teacher_layers": (11, 15, 19, 23),
        "query_dim": 2048,
    },
    "giant": {
        "teacher_layers": (19, 27, 33, 39),
        "query_dim": 3072,
    },
}


def infer_da3_backbone_profile(model_path_or_name: str) -> str:
    lowered = str(model_path_or_name).lower()
    if "giant" in lowered or "vitg" in lowered:
        return "giant"
    if "large" in lowered or "vitl" in lowered:
        return "large"
    return "giant"


def resolve_da3_backbone_defaults(model_path_or_name: str, variant: str = "auto") -> dict[str, int | tuple[int, ...]]:
    normalized_variant = variant.lower()
    if normalized_variant == "auto":
        normalized_variant = infer_da3_backbone_profile(model_path_or_name)
    if normalized_variant not in DA3_BACKBONE_DEFAULTS:
        raise ValueError(
            f"Unsupported DA3 variant {variant!r}. Expected one of: auto, large, giant."
        )
    return DA3_BACKBONE_DEFAULTS[normalized_variant]


def resolve_da3_import(code_root: str | None) -> type:
    try:
        from depth_anything_3.api import DepthAnything3
    except ImportError as first_exc:
        if code_root is None:
            raise ImportError(
                "Failed to import DepthAnything3. Install the `depth_anything_3` package or set "
                "`policy.da3_code_root` to a standalone DA3 repository root or its `src` directory."
            ) from first_exc

        candidate = Path(code_root).expanduser().resolve()
        candidate_src = candidate / "src" if (candidate / "src").exists() else candidate
        candidate_src_str = str(candidate_src)
        if not candidate_src.exists():
            raise FileNotFoundError(
                f"policy.da3_code_root={code_root!r} does not exist or does not contain a `src` directory"
            )
        if candidate_src_str not in sys.path:
            sys.path.append(candidate_src_str)
        try:
            from depth_anything_3.api import DepthAnything3
        except ImportError as second_exc:
            raise ImportError(
                "Failed to import DepthAnything3 from the configured `policy.da3_code_root`. "
                "Expected a standalone DA3 checkout or installed `depth_anything_3` package."
            ) from second_exc

    return DepthAnything3


class DA3BackboneTeacher(nn.Module):
    def __init__(
        self,
        model_path_or_name: str,
        process_res: int = 504,
        dtype: torch.dtype = torch.bfloat16,
        teacher_layers: tuple[int, ...] | None = None,
        code_root: str | None = None,
    ):
        super().__init__()
        DepthAnything3 = resolve_da3_import(code_root)
        patch_da3_reference_view_logger_once()

        wrapper = DepthAnything3.from_pretrained(model_path_or_name)
        self.backbone = wrapper.model.backbone
        self.out_layers = tuple(int(layer_idx) for layer_idx in wrapper.config.net.out_layers)
        self.feature_dim = int(wrapper.config.head.dim_in)
        self.variant = infer_da3_backbone_profile(getattr(wrapper.config.net, "name", model_path_or_name))
        self.teacher_layers = self.out_layers if teacher_layers is None else tuple(int(layer_idx) for layer_idx in teacher_layers)
        missing_layers = [layer_idx for layer_idx in self.teacher_layers if layer_idx not in self.out_layers]
        if missing_layers:
            raise ValueError(
                f"Requested DA3 layers {self.teacher_layers} are not available from teacher backbone layers {self.out_layers}. "
                f"Missing: {missing_layers}"
            )
        self.process_res = process_res
        self._dtype = dtype
        self._logged_sdpa_backend = False

        self.backbone.to(dtype=dtype)
        self.backbone.eval()
        self.backbone.requires_grad_(False)
        del wrapper

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=dtype).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225], dtype=dtype).view(1, 1, 3, 1, 1),
            persistent=False,
        )

    def _log_attention_backend_once(self, x: torch.Tensor) -> None:
        if self._logged_sdpa_backend:
            return

        attention_modules = [module for module in self.backbone.modules() if hasattr(module, "fused_attn")]
        fused_attn_count = sum(bool(getattr(module, "fused_attn", False)) for module in attention_modules)
        total_attn_count = len(attention_modules)

        flash_enabled = None
        mem_efficient_enabled = None
        math_enabled = None
        if x.device.type == "cuda" and hasattr(torch.backends, "cuda"):
            flash_enabled = getattr(torch.backends.cuda, "flash_sdp_enabled", lambda: None)()
            mem_efficient_enabled = getattr(torch.backends.cuda, "mem_efficient_sdp_enabled", lambda: None)()
            math_enabled = getattr(torch.backends.cuda, "math_sdp_enabled", lambda: None)()

        flash_eligible = (
            x.device.type == "cuda"
            and x.dtype in {torch.float16, torch.bfloat16}
            and total_attn_count > 0
            and fused_attn_count == total_attn_count
            and flash_enabled is True
        )

        logger.info(
            "DA3 teacher attention backend (best-effort): impl=sdpa, "
            f"device={x.device}, dtype={x.dtype}, "
            f"fused_attn_modules={fused_attn_count}/{total_attn_count}, "
            f"flash_enabled={flash_enabled}, "
            f"mem_efficient_enabled={mem_efficient_enabled}, "
            f"math_enabled={math_enabled}, "
            f"flash_eligible={flash_eligible}"
        )
        self._logged_sdpa_backend = True

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        model_param = next(self.backbone.parameters())
        x = images.to(device=model_param.device, dtype=model_param.dtype)
        if x.max() > 1.0:
            x = x / 255.0
        x = (x - self.mean.to(device=x.device, dtype=x.dtype)) / self.std.to(device=x.device, dtype=x.dtype)

        bsize, num_views, channels, height, width = x.shape
        x = x.view(bsize * num_views, channels, height, width)
        x = F.interpolate(
            x,
            size=(self.process_res, self.process_res),
            mode="bilinear",
            align_corners=False,
        )
        x = x.view(bsize, num_views, channels, self.process_res, self.process_res)

        self._log_attention_backend_once(x)
        features_tuple, _ = self.backbone(x)
        layer_outputs = {layer_idx: layer_out for layer_idx, layer_out in zip(self.out_layers, features_tuple, strict=False)}
        teacher_features = []
        for layer_idx in self.teacher_layers:
            layer_out = layer_outputs[layer_idx]
            patch_tokens = layer_out[0]
            teacher_features.append(patch_tokens.reshape(bsize, -1, patch_tokens.shape[-1]))
        return teacher_features
