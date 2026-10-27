# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

import importlib.util
import logging
import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import nn

logger = logging.getLogger(__name__)
from awomo.awomo.training.shared.utils import dtype_from_str

HAVE_TE = importlib.util.find_spec("transformer_engine") is not None
HAVE_DEEP_EP = importlib.util.find_spec("deep_ep") is not None
HAVE_UCCL_EP = importlib.util.find_spec("uccl") is not None or importlib.util.find_spec("ep") is not None
HAVE_GMM = importlib.util.find_spec("grouped_gemm") is not None


IS_OPTIM_STEP = False
IS_FIRST_MICROBATCH: bool | None = None


def set_is_optim_step(value: bool) -> None:
    global IS_OPTIM_STEP
    IS_OPTIM_STEP = value


def get_is_optim_step() -> bool:
    return IS_OPTIM_STEP


def set_is_first_microbatch(value: bool | None) -> None:
    global IS_FIRST_MICROBATCH
    IS_FIRST_MICROBATCH = value


def get_is_first_microbatch() -> bool | None:
    return IS_FIRST_MICROBATCH


def is_tensor_unallocated(tensor: torch.Tensor) -> bool:
    try:
        return tensor.data_ptr() == 0 or tensor.numel() == 0
    except Exception:
        return True


@dataclass(kw_only=True)
class TEFp8Config:
    """Configuration for Transformer Engine FP8 quantization."""

    recipe: Literal["current", "block"] | Any = "current"

    def build_recipe(self):
        if not HAVE_TE:
            return None

        if not isinstance(self.recipe, str):
            return self.recipe

        from transformer_engine.common.recipe import Float8BlockScaling, Float8CurrentScaling

        if self.recipe == "block":
            return Float8BlockScaling()
        return Float8CurrentScaling()

    def maybe_te_autocast(self):
        if not HAVE_TE:
            return nullcontext()
        from transformer_engine.pytorch.quantization import autocast as te_autocast

        return te_autocast(enabled=True, recipe=self.build_recipe())


@dataclass(kw_only=True)
class BackendConfig:
    """Backend configuration for model components."""

    attn: Literal["te", "sdpa", "flex"] = "te" if HAVE_TE and torch.cuda.is_available() else "sdpa"
    linear: Literal["torch", "te"] = "te" if HAVE_TE and torch.cuda.is_available() else "torch"
    rms_norm: Literal["torch", "torch_fp32", "te"] = "torch_fp32"
    rope_fusion: bool = HAVE_TE and torch.cuda.is_available()
    experts: Literal["torch", "te", "gmm", "torch_mm"] = "torch_mm" if torch.cuda.is_available() else "torch"
    dispatcher: Literal["torch", "deepep", "hybridep", "uccl_ep", "nccl_alltoall"] = (
        "deepep"
        if HAVE_DEEP_EP and torch.cuda.is_available()
        else "uccl_ep" if HAVE_UCCL_EP and torch.cuda.is_available() else "torch"
    )
    dispatcher_num_sms: int = 20
    enable_deepep: bool | None = None
    fake_balanced_gate: bool = False
    fake_gate_noise: float = 0.0
    enable_hf_state_dict_adapter: bool = True
    enable_fsdp_optimizations: bool = False
    te_fp8: TEFp8Config | None = None
    gate_precision: str | torch.dtype | None = None

    def __post_init__(self):
        if isinstance(self.te_fp8, dict):
            self.te_fp8 = TEFp8Config(**self.te_fp8)

        if isinstance(self.gate_precision, str):
            self.gate_precision = dtype_from_str(self.gate_precision, default=None)

        if self.enable_deepep is not None:
            warnings.warn(
                "enable_deepep is deprecated and will be removed in a future release. "
                "Use experts='gmm' and dispatcher='deepep' instead of enable_deepep=True, "
                "or dispatcher='torch' instead of enable_deepep=False.",
                DeprecationWarning,
                stacklevel=2,
            )
            if self.enable_deepep:
                self.experts = "gmm"
                self.dispatcher = "deepep"
            else:
                self.dispatcher = "torch"
            self.enable_deepep = None

        if self.experts in ("te", "gmm") and self.dispatcher not in (
            "deepep",
            "hybridep",
            "uccl_ep",
            "nccl_alltoall",
        ):
            if (
                torch.distributed.is_initialized() and torch.distributed.get_rank() == 0
            ) or not torch.distributed.is_initialized():
                logger.info(
                    f"experts='{self.experts}' requires dispatcher='deepep' or 'uccl_ep', "
                    f"but got dispatcher='{self.dispatcher}'. "
                    "Setting dispatcher to torch and experts to torch_mm."
                )
            self.dispatcher = "torch"
            self.experts = "torch_mm"

        if self.te_fp8 is not None and self.linear != "te" and self.experts != "te":
            raise ValueError(
                "te_fp8 requires at least one TE backend "
                f"(linear='te' or experts='te'), but got linear='{self.linear}', experts='{self.experts}'"
            )


@torch.compile(fullgraph=True, dynamic=True)
def _float32_rms_norm_fwd(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    input_dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (weight * x).to(input_dtype)


class Float32RMSNorm(nn.Module):
    """RMSNorm with explicit fp32 computation for training stability."""

    def __init__(self, dim, eps=1e-5, device=None, dtype=torch.bfloat16):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

    def forward(self, x):
        return _float32_rms_norm_fwd(x, self.weight, self.eps)


def initialize_rms_norm_module(
    rms_norm_impl: str,
    dim: int,
    eps: float = 1e-5,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    if rms_norm_impl == "te":
        from transformer_engine.pytorch.module.rmsnorm import RMSNorm as TransformerEngineRMSNorm

        _patch_te_modules()
        return TransformerEngineRMSNorm(normalized_shape=dim, eps=eps, device=device, params_dtype=dtype)
    elif rms_norm_impl == "torch":
        return nn.RMSNorm(dim, eps=eps, device=device, dtype=dtype)
    elif rms_norm_impl == "torch_fp32":
        return Float32RMSNorm(dim, eps=eps, device=device, dtype=dtype)
    else:
        raise ValueError(f"Unsupported RMSNorm implementation: {rms_norm_impl}")


def initialize_linear_module(
    linear_impl: str,
    in_features: int,
    out_features: int,
    bias: bool = False,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    if linear_impl == "torch":
        return nn.Linear(in_features, out_features, bias=bias, device=device, dtype=dtype)
    elif linear_impl == "te":
        from transformer_engine.pytorch.module.linear import Linear as TransformerEngineLinear

        _patch_te_modules()
        return TransformerEngineLinear(
            in_features=in_features, out_features=out_features, bias=bias, device=device, params_dtype=dtype
        )
    else:
        raise ValueError(f"Unsupported Linear implementation: {linear_impl}")


def _make_lazy_te_patcher():
    patched = False

    def _patch():
        nonlocal patched
        if patched:
            return
        patched = True

        from transformer_engine.pytorch.module.grouped_linear import GroupedLinear as TEGroupedLinear
        from transformer_engine.pytorch.module.linear import Linear as TELinear
        from transformer_engine.pytorch.module.rmsnorm import RMSNorm as TERMSNorm

        _original_rmsnorm_forward = TERMSNorm.forward
        _original_linear_forward = TELinear.forward
        _original_grouped_linear_forward = TEGroupedLinear.forward

        def _patched_rmsnorm_forward(self, x):
            if is_tensor_unallocated(x) and not torch.compiler.is_compiling():
                return torch.empty_like(x)
            return _original_rmsnorm_forward(self, x)

        def _patched_linear_forward(self, x, is_first_microbatch=None, **kwargs):
            if is_tensor_unallocated(x):
                out_shape = x.shape[:-1] + (self.weight.shape[0],)
                return torch.empty(out_shape, dtype=x.dtype, device=x.device)
            if is_first_microbatch is None:
                is_first_microbatch = get_is_first_microbatch()
            return _original_linear_forward(self, x, is_first_microbatch=is_first_microbatch, **kwargs)

        def _patched_grouped_linear_forward(self, inp, m_splits, is_first_microbatch=None):
            if is_first_microbatch is None:
                is_first_microbatch = get_is_first_microbatch()
            return _original_grouped_linear_forward(self, inp, m_splits, is_first_microbatch=is_first_microbatch)

        TERMSNorm.forward = _patched_rmsnorm_forward
        TELinear.forward = _patched_linear_forward
        TEGroupedLinear.forward = _patched_grouped_linear_forward

    return _patch


_patch_te_modules = _make_lazy_te_patcher()


def get_rope_config(config) -> tuple[float, dict, float]:
    rope_parameters = config.rope_parameters
    rope_theta = rope_parameters["rope_theta"]
    partial_rotary_factor = rope_parameters.get("partial_rotary_factor", 1.0)
    return rope_theta, rope_parameters, partial_rotary_factor


def cast_model_to_dtype(model: nn.Module, dtype: torch.dtype = torch.bfloat16) -> None:
    fp32_keywords = _get_fp32_module_keywords(model)

    model.to(dtype)
    if fp32_keywords:
        if _has_dtensor_params(model):
            logger.warning(
                "Model parameters are DTensors (FSDP2) — skipping fp32 parameter "
                "restoration for keywords=%s. Only buffers will be restored to fp32. "
                "FSDP2 requires uniform dtype within each parameter group.",
                fp32_keywords,
            )
            _restore_fp32_buffers(model, fp32_keywords)
        else:
            _restore_fp32_modules(model, fp32_keywords)


def _get_fp32_module_keywords(model: nn.Module) -> list[str]:
    keywords: list[str] = []
    for attr in ("_keep_in_fp32_modules_strict", "_keep_in_fp32_modules"):
        val = getattr(model, attr, None)
        if isinstance(val, list):
            keywords.extend(val)

    return list(dict.fromkeys(keywords))


def _has_dtensor_params(model: nn.Module) -> bool:
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        return False
    return any(isinstance(p, DTensor) for p in model.parameters())


def _restore_fp32_modules(model: nn.Module, fp32_keywords: list[str]) -> None:
    for name, module in model.named_modules():
        if any(kw in name for kw in fp32_keywords):
            module.to(torch.float32)


def _restore_fp32_buffers(model: nn.Module, fp32_keywords: list[str]) -> None:
    for name, module in model.named_modules():
        if any(kw in name for kw in fp32_keywords):
            for buf_name, buf in module.named_buffers(recurse=False):
                module._buffers[buf_name] = buf.to(torch.float32)


__all__ = [
    "BackendConfig",
    "Float32RMSNorm",
    "TEFp8Config",
    "cast_model_to_dtype",
    "get_is_first_microbatch",
    "get_is_optim_step",
    "get_rope_config",
    "initialize_linear_module",
    "initialize_rms_norm_module",
    "set_is_first_microbatch",
    "set_is_optim_step",
]
