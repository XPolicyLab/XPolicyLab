"""Qwen3.5-VL backend for the canonical XVLA image/text contract."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import logging
import os
from typing import Any, Dict, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..architecture_contract import (
    PREFIX_KV_ACTION_CONTEXT_MODE,
    is_vlm_fast_action_bridge_version,
)

from .base import VLMBackend
from .registry import register_vlm_backend

logger = logging.getLogger(__name__)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_QWEN_MEAN = (0.5, 0.5, 0.5)
_QWEN_STD = (0.5, 0.5, 0.5)
QWEN_POLICY_CONTRACT_V4 = (
    "qwen35_224_letterbox_premerge196_semantic_pyramid_v4"
)
QWEN_POLICY_CONTRACT = QWEN_POLICY_CONTRACT_V4
QWEN_POLICY_ACTION_INIT_CONTRACT = (
    "random_from_florence_v113_architecture_v1"
)
QWEN_ACTION_CONTEXT_MODE_V2 = "readonly_cross_attention_v2"
QWEN_POLICY_IMAGE_SIZE = (224, 224)
QWEN_POLICY_PATCH_SIZE = 16
QWEN_POLICY_TEMPORAL_PATCH_SIZE = 2
QWEN_POLICY_SPATIAL_MERGE_SIZE = 2
QWEN_POLICY_GEOMETRY_GRID = (14, 14)
QWEN_POLICY_LANGUAGE_GRID = (7, 7)
QWEN_POLICY_GEOMETRY_TOKENS_PER_VIEW = 196
QWEN_POLICY_LANGUAGE_TOKENS_PER_VIEW = 49
# DM05-compatible history compression: each past frame is pooled
# independently from 7x7 merged Qwen tokens to a 4x4 (16-token) grid.
# The current frame remains at 49 tokens so the pretrained current-view
# contract and current-frame spatial detail are preserved.
QWEN_POLICY_HISTORY_POOL_SIZE = 4
QWEN_POLICY_HISTORY_TOKENS_PER_FRAME = (
    QWEN_POLICY_HISTORY_POOL_SIZE * QWEN_POLICY_HISTORY_POOL_SIZE
)
QWEN_POLICY_HISTORY_GRID = (
    QWEN_POLICY_HISTORY_POOL_SIZE * QWEN_POLICY_SPATIAL_MERGE_SIZE,
    QWEN_POLICY_HISTORY_POOL_SIZE * QWEN_POLICY_SPATIAL_MERGE_SIZE,
)
QWEN_POLICY_PRETRAIN_NUM_VIEWS = 3
QWEN_POLICY_PRETRAIN_GEOMETRY_TOKENS = 588
# Historical hard upper bound for an unpooled multi-frame ``image_grid_thw``
# block. The current P3 packer does not emit such blocks: each history frame
# is a separate T=1 4x4 block after DM05-style pooling. Keep the constant and
# its direct RoPE regression test as a Transformers compatibility reference.
#
# ``Qwen3VLModel.get_rope_index`` advances its running text position by only
# ``max(grid_h, grid_w) // spatial_merge_size`` per vision block, regardless
# of T, while ``get_vision_position_ids`` lays the temporal axis out as
# ``current_pos + 0 .. current_pos + T - 1``. So the block's largest temporal
# position collides with the ``<vision_end>`` token once ``T == 8`` and
# overruns the following text positions entirely at ``T >= 10``, silently
# destroying RoPE monotonicity across the prompt. Verified empirically
# against transformers 5.13.0:
#
#   T= 6 -> vision t 4..9,  vision_end t=11, next text 12  (ok)
#   T= 8 -> vision t 4..11, vision_end t=11, next text 12  (collision)
#   T=10 -> vision t 4..13, vision_end t=11, next text 12  (non-monotonic)
#
# 14 // 2 = 7 is therefore the largest T that keeps the whole packed
# sequence strictly ordered.
QWEN_POLICY_MAX_NATIVE_VIDEO_FRAMES = (
    max(QWEN_POLICY_GEOMETRY_GRID) // QWEN_POLICY_SPATIAL_MERGE_SIZE
)
# Florence's landed mixed11 reference uses a 50-token action-path cap.
QWEN_POLICY_ACTION_LANGUAGE_MAX_LENGTH = 50
QWEN_TEMPORAL_EMBEDDING_CONTRACT = "fixed_sinusoidal_v1"
QWEN_TEMPORAL_EMBEDDING_SPACETIME = "spacetime_mem_v1"
QWEN_TEMPORAL_EMBEDDING_BASE = 10000.0
# Default stride used when `qwen_temporal_embedding=spacetime_mem_v1` is
# selected without an explicit `qwen_spacetime_stride`. Matches the Hy-VLA
# MEM recipe and the Florence DaViT port.
QWEN_SPACETIME_DEFAULT_STRIDE = 4


def _pool_qwen_history_frame_tokens(
    frame_tokens: torch.Tensor,
) -> torch.Tensor:
    """Pool each frame independently using the DM05 2D history contract.

    Args:
        frame_tokens: ``[N, 49, D]`` merged tokens, one row per frame.

    Returns:
        ``[N, 16, D]`` pooled tokens. The batch/frame dimension is never
        reduced, so this operation cannot average different time steps.
    """
    if frame_tokens.ndim != 3:
        raise ValueError(
            "Qwen history pooling expects [N, tokens, hidden], got "
            f"{tuple(frame_tokens.shape)}."
        )
    tokens = int(frame_tokens.shape[1])
    spatial = int(tokens**0.5)
    if spatial * spatial != tokens:
        raise ValueError(
            "Qwen history pooling requires a square per-frame token grid, got "
            f"tokens={tokens}."
        )
    hidden = int(frame_tokens.shape[2])
    grid = frame_tokens.reshape(-1, spatial, spatial, hidden).permute(0, 3, 1, 2)
    pooled = F.adaptive_avg_pool2d(
        grid,
        output_size=(
            QWEN_POLICY_HISTORY_POOL_SIZE,
            QWEN_POLICY_HISTORY_POOL_SIZE,
        ),
    )
    return pooled.permute(0, 2, 3, 1).reshape(
        -1, QWEN_POLICY_HISTORY_TOKENS_PER_FRAME, hidden
    )


@dataclass(frozen=True)
class QwenPolicySemantics:
    """How one policy contract turns a packed LM sequence into expert context.

    ``visual_positions_semantic`` decides whether the language model's output at
    the image-placeholder positions reaches the action expert at all, and
    ``prompt_before_views`` decides whether those positions were even able to
    read the instruction: Qwen3.5 is causal (and partly linear-attention, so a
    Florence-style bidirectional prefix is not available), therefore an image
    placed before the prompt can never attend to it.

    The v5 policy restores that property within a causal decoder: the instruction is
    packed first, then the views, and the image positions join the semantic
    mask. The expert then reads ~147 language-conditioned visual vectors plus
    ~20 text vectors instead of ~20 text vectors alone.
    """

    visual_positions_semantic: bool
    prompt_before_views: bool


_QWEN_POLICY_SEMANTICS = QwenPolicySemantics(
    visual_positions_semantic=True,
    prompt_before_views=True,
)


def qwen_policy_semantics(contract_or_config) -> QwenPolicySemantics:
    """Resolve the one formal Qwen v5 semantic packing contract."""
    if isinstance(contract_or_config, str):
        contract = contract_or_config
    else:
        contract = _config_value(
            contract_or_config, "vlm_policy_contract", "auto"
        )
    contract = str(contract or "auto").strip().lower()
    if contract == "auto":
        contract = QWEN_POLICY_CONTRACT
    if contract != QWEN_POLICY_CONTRACT:
        raise ValueError(
            f"Unknown Qwen policy contract {contract!r}; expected "
            f"{QWEN_POLICY_CONTRACT!r}."
        )
    return _QWEN_POLICY_SEMANTICS


def qwen_policy_vision_contract() -> dict[str, Any]:
    return {
        "version": "qwen35_224_letterbox_premerge196_v3",
        "image_size": list(QWEN_POLICY_IMAGE_SIZE),
        "resize_mode": "letterbox",
        "pad_rgb": [128, 128, 128],
        "canonical_input_normalization": {
            "name": "imagenet_clip",
            "mean": list(_IMAGENET_MEAN),
            "std": list(_IMAGENET_STD),
        },
        "qwen_visual_normalization": {
            "mean": list(_QWEN_MEAN),
            "std": list(_QWEN_STD),
        },
        "normalization_bridge": (
            "denormalize_canonical_to_rgb_then_qwen_v1"
        ),
        "patch_size": QWEN_POLICY_PATCH_SIZE,
        "temporal_patch_size": QWEN_POLICY_TEMPORAL_PATCH_SIZE,
        "spatial_merge_size": QWEN_POLICY_SPATIAL_MERGE_SIZE,
        "geometry_grid": list(QWEN_POLICY_GEOMETRY_GRID),
        "geometry_tokens_per_view": QWEN_POLICY_GEOMETRY_TOKENS_PER_VIEW,
        "language_grid": list(QWEN_POLICY_LANGUAGE_GRID),
        "language_tokens_per_view": QWEN_POLICY_LANGUAGE_TOKENS_PER_VIEW,
        "pretrain_num_views": QWEN_POLICY_PRETRAIN_NUM_VIEWS,
        "pretrain_geometry_tokens": QWEN_POLICY_PRETRAIN_GEOMETRY_TOKENS,
    }


def _config_value(config_or_dict, name: str, default=None):
    if isinstance(config_or_dict, dict):
        return config_or_dict.get(name, default)
    return getattr(config_or_dict, name, default)


def validate_qwen_policy_vision_contract(config_or_dict) -> None:
    serialized = _config_value(
        config_or_dict,
        "vlm_policy_vision_contract",
        None,
    )
    expected = qwen_policy_vision_contract()
    if serialized != expected:
        raise ValueError(
            "Qwen policy vision contract mismatch. Expected the formal "
            f"224/49/196 contract {expected}, got {serialized!r}."
        )

    vlm_config = _config_value(config_or_dict, "vlm_config", None)
    if not isinstance(vlm_config, dict):
        raise ValueError("Qwen policy contract requires serialized vlm_config.")
    vision_config = vlm_config.get("vision_config")
    if not isinstance(vision_config, dict):
        raise ValueError(
            "Qwen policy contract requires vlm_config.vision_config."
        )
    expected_vision = {
        "patch_size": QWEN_POLICY_PATCH_SIZE,
        "temporal_patch_size": QWEN_POLICY_TEMPORAL_PATCH_SIZE,
        "spatial_merge_size": QWEN_POLICY_SPATIAL_MERGE_SIZE,
    }
    mismatches = {}
    for key, expected_value in expected_vision.items():
        actual_value = vision_config.get(key)
        try:
            matches = int(actual_value) == expected_value
        except (TypeError, ValueError):
            matches = False
        if not matches:
            mismatches[key] = (actual_value, expected_value)
    if mismatches:
        raise ValueError(
            "Qwen vision backbone is incompatible with the formal 224/49/196 "
            f"policy contract: {mismatches}."
        )


def validate_qwen_policy_checkpoint(
    config_or_dict,
    *,
    require_trained: bool,
) -> None:
    backend = str(
        _config_value(config_or_dict, "vlm_backend", "florence2")
    ).strip().lower()
    if backend != "qwen3_5":
        return
    action_context_mode = str(
        _config_value(config_or_dict, "action_context_mode", "")
    ).strip().lower()
    if action_context_mode != QWEN_ACTION_CONTEXT_MODE_V2:
        raise ValueError(
            "Qwen policy requires action_context_mode="
            f"{QWEN_ACTION_CONTEXT_MODE_V2!r}."
        )
    semantic_route = _config_value(
        config_or_dict,
        "action_context_semantic_route",
        None,
    )
    if not isinstance(semantic_route, (list, tuple)) or not semantic_route:
        raise ValueError(
            "Qwen v5 policy requires an explicit "
            "action_context_semantic_route."
        )
    depth = int(_config_value(config_or_dict, "depth", 0) or 0)
    every_n = int(
        _config_value(
            config_or_dict,
            "action_context_cross_attn_every_n_layers",
            0,
        )
        or 0
    )
    semantic_layers = list(
        _config_value(
            config_or_dict,
            "vlm_policy_semantic_layers",
            None,
        )
        or []
    )
    if depth <= 0 or every_n <= 0 or not semantic_layers:
        raise ValueError(
            "Qwen v5 policy route validation requires depth, "
            "action_context_cross_attn_every_n_layers, and "
            "vlm_policy_semantic_layers."
        )
    expected_route_length = len(range(0, depth, every_n))
    if len(semantic_route) != expected_route_length:
        raise ValueError(
            "Qwen v5 action_context_semantic_route needs "
            f"{expected_route_length} entries, got {len(semantic_route)}."
        )
    if any(
        int(index) < 0 or int(index) >= len(semantic_layers)
        for index in semantic_route
    ):
        raise ValueError(
            "Qwen v5 action_context_semantic_route contains an invalid "
            f"depth index: route={semantic_route}, "
            f"num_depths={len(semantic_layers)}."
        )
    if (
        str(
            _config_value(config_or_dict, "vlm_policy_contract", "")
        ).strip().lower()
        != QWEN_POLICY_CONTRACT
    ):
        raise ValueError(
            "Qwen policy requires vlm_policy_contract="
            f"{QWEN_POLICY_CONTRACT!r}."
        )
    if (
        str(
            _config_value(config_or_dict, "action_initialization_contract", "")
        ).strip().lower()
        != QWEN_POLICY_ACTION_INIT_CONTRACT
    ):
        raise ValueError(
            "Qwen policy requires action_initialization_contract="
            f"{QWEN_POLICY_ACTION_INIT_CONTRACT}."
        )
    validate_qwen_policy_vision_contract(config_or_dict)
    action_language_max_length = int(
        _config_value(config_or_dict, "action_language_max_length", 0) or 0
    )
    if action_language_max_length != QWEN_POLICY_ACTION_LANGUAGE_MAX_LENGTH:
        raise ValueError(
            "Qwen policy requires action_language_max_length="
            f"{QWEN_POLICY_ACTION_LANGUAGE_MAX_LENGTH}, got "
            f"{action_language_max_length}."
        )
    if require_trained and not bool(
        _config_value(config_or_dict, "action_context_trained", False)
    ):
        raise ValueError(
            "Qwen policy checkpoint contains untrained action-context modules."
        )


def _normalize_processor_hw(value) -> tuple[int, int] | None:
    if isinstance(value, dict):
        height = value.get("height")
        width = value.get("width")
        if height is not None and width is not None:
            return int(height), int(width)
        shortest = value.get("shortest_edge")
        if shortest is not None:
            return int(shortest), int(shortest)
    height = getattr(value, "height", None)
    width = getattr(value, "width", None)
    if height is not None and width is not None:
        return int(height), int(width)
    shortest = getattr(value, "shortest_edge", None)
    if shortest is not None:
        return int(shortest), int(shortest)
    if isinstance(value, int):
        return int(value), int(value)
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return int(value[0]), int(value[1])
    return None


def validate_qwen_processor_contract(processor) -> None:
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None and hasattr(processor, "size"):
        image_processor = processor
    if image_processor is None:
        raise ValueError("Qwen policy requires an image processor.")
    expected = QWEN_POLICY_IMAGE_SIZE
    actual_size = _normalize_processor_hw(
        getattr(image_processor, "size", None)
    )
    if actual_size != expected:
        raise ValueError(
            "Qwen policy processor size must be 224x224 for the 49/196 "
            f"contract, got {actual_size}."
        )
    crop_size = getattr(image_processor, "crop_size", None)
    if crop_size is not None:
        actual_crop = _normalize_processor_hw(crop_size)
        if actual_crop != expected:
            raise ValueError(
                "Qwen policy processor crop_size must be 224x224 for the "
                f"49/196 contract, got {actual_crop}."
            )
    resize_mode = str(
        getattr(processor, "image_resize_mode", "")
    ).strip().lower()
    if resize_mode != "letterbox":
        raise ValueError(
            "Qwen policy processor must preserve aspect ratio with "
            f"image_resize_mode=letterbox, got {resize_mode!r}."
        )
    pad_rgb = list(getattr(processor, "image_pad_rgb", []))
    if pad_rgb != [128, 128, 128]:
        raise ValueError(
            "Qwen policy processor requires neutral letterbox padding "
            f"[128, 128, 128], got {pad_rgb!r}."
        )
    actual_mean = tuple(
        float(value)
        for value in getattr(image_processor, "image_mean", ())
    )
    actual_std = tuple(
        float(value)
        for value in getattr(image_processor, "image_std", ())
    )
    if actual_mean != _IMAGENET_MEAN or actual_std != _IMAGENET_STD:
        raise ValueError(
            "Qwen XVLA uses canonical ImageNet-normalized tensors and "
            "converts them to Qwen normalization inside the backend. "
            f"Expected mean/std={_IMAGENET_MEAN}/{_IMAGENET_STD}, got "
            f"{actual_mean}/{actual_std}."
        )


@dataclass(frozen=True)
class _NativeImageFeatures:
    merged: torch.Tensor
    detail: torch.Tensor


@dataclass(frozen=True)
class _QwenPolicyVisualCache:
    merged: torch.Tensor
    detail: torch.Tensor
    mask: torch.Tensor
    # frames_per_view: [B, V] long tensor. All-ones under the default
    # single-frame contract (keeps every K=1 checkpoint bit-identical).
    # When ``keep_per_frame_views`` is used, view slots carry their valid
    # frame count; downstream packing emits each older frame as an independent
    # DM05-pooled T=1 block followed by a native current-frame block.
    frames_per_view: torch.Tensor | None = None


@dataclass
class _PendingPrefixKV:
    """One captured full-attention input awaiting K/V projection.

    Hooks only record tensors: either native K/V projection outputs (opt-in)
    or attention inputs for the compatibility projection path. Tensor work
    stays in ``_finalize_prefix_kv`` after the language forward, outside hooks.
    """

    module: nn.Module
    hidden_states: torch.Tensor
    kwargs: dict
    key: torch.Tensor | None = None
    value: torch.Tensor | None = None

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)

class _PolicyOutputAdapter(nn.Sequential):
    """Project native Qwen BF16 states into XVLA's FP32 policy contract."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        parameter = next(self.parameters(), None)
        if parameter is not None:
            inputs = inputs.to(dtype=parameter.dtype)
        return super().forward(inputs).float()


class _PerDepthOutputAdapter(nn.Module):
    """Give every sampled Qwen depth its own projection into the policy dim.

    Each sampled depth is read by a distinct action cross-attention layer, so
    the projections are free to learn different policy subspaces.
    """

    def __init__(self, adapters: list[nn.Module]) -> None:
        super().__init__()
        self.adapters = nn.ModuleList(adapters)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.dim() != 4 or inputs.shape[1] != len(self.adapters):
            raise ValueError(
                f"Expected [B, {len(self.adapters)}, L, D] per-depth semantic "
                f"states, got {tuple(inputs.shape)}."
            )
        return torch.stack(
            [
                adapter(inputs[:, depth])
                for depth, adapter in enumerate(self.adapters)
            ],
            dim=1,
        )

def _transformers_version_tuple() -> tuple[int, ...]:
    try:
        import transformers
    except ImportError:
        return ()
    raw = str(getattr(transformers, "__version__", ""))
    parts = []
    for piece in raw.split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _require_qwen35_classes():
    installed = _transformers_version_tuple()
    if installed < (5, 2, 0):
        raise RuntimeError(
            "Qwen3.5 backend requires transformers>=5.2.0; "
            f"found {'.'.join(map(str, installed)) or 'missing'}. "
            "Use the dedicated Qwen XVLA environment; the legacy Florence "
            "environment intentionally remains on transformers 4.x."
        )
    try:
        from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError(
            "The installed transformers package does not expose "
            "Qwen3_5ForConditionalGeneration."
        ) from exc
    return Qwen3_5Config, Qwen3_5ForConditionalGeneration


def _resolve_model_dtype(config) -> torch.dtype:
    requested = str(getattr(config, "vlm_dtype", "auto") or "auto").lower()
    if requested == "auto":
        requested = str(
            (getattr(config, "vlm_config", None) or {})
            .get("text_config", {})
            .get("dtype", "bfloat16")
        ).lower()
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if requested not in aliases:
        raise ValueError(
            "Qwen3.5 vlm_dtype must be auto, bfloat16, float16, or float32; "
            f"got {requested!r}."
        )
    return aliases[requested]


def _cuda_capability_major() -> "int | None":
    """Compute-capability major of the device we will train on, None on CPU."""
    try:
        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_capability(torch.cuda.current_device())[0]
    except Exception:
        return None


def _flash_attn_3_supported() -> bool:
    """Whether FA3 kernels can actually run here.

    Availability only proves the package imports. The installed wheel has
    Hopper cubins and no forward-compatible PTX, so screen the device before
    the first kernel launch.
    """
    try:
        from transformers.utils import is_flash_attn_3_available
    except ImportError:
        return False
    major = _cuda_capability_major()
    return bool(is_flash_attn_3_available()) and major == 9


def _causal_conv1d_cuda_supported() -> bool:
    """Whether the installed causal_conv1d CUDA kernels cover this device.

    The installed wheel has sm75/80/87/90 cubins and no PTX. A failed launch
    may poison the CUDA context, so use a static capability guard.
    """
    try:
        import causal_conv1d  # noqa: F401
    except ImportError:
        return False
    return _cuda_capability_major() in (7, 8, 9)


def _torch_seq_idx_causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: "torch.Tensor | None" = None,
    activation: "str | None" = None,
    seq_idx: "torch.Tensor | None" = None,
    **_unused,
) -> torch.Tensor:
    """seq_idx-aware causal depthwise conv1d in pure torch.

    Transformers' own fallback (F.silu(self.conv1d(...))) silently DROPS
    seq_idx, so on padding-free packed batches -- which is how this repo feeds
    the CE tasks, see _run_lm_losses_padding_free_impl -- the K-1 leading
    positions of every packed sequence would convolve over the tail of the
    previous, unrelated sequence. That is a correctness bug, not a slowdown,
    so replicate the kernel's semantics instead: tap k is masked out wherever
    it would reach across a sequence boundary.

    Tests compare this path with the CUDA implementation and verify packed
    sequence isolation.
    """
    if activation not in (None, "silu", "swish"):
        raise ValueError(
            f"_torch_seq_idx_causal_conv1d supports silu/swish/None, got {activation!r}."
        )
    seq_len = x.shape[-1]
    kernel_size = weight.shape[-1]
    x_f32 = x.float()
    weight_f32 = weight.float()
    acc = None
    for tap in range(kernel_size):
        shift = kernel_size - 1 - tap
        if shift:
            shifted = F.pad(x_f32, (shift, 0))[..., :seq_len]
            if seq_idx is not None:
                # -1 sentinel makes the left pad differ from every real id, so
                # positions reaching before the packed row are dropped too.
                source_ids = F.pad(seq_idx, (shift, 0), value=-1)[..., :seq_len]
                shifted = shifted * (source_ids == seq_idx).unsqueeze(1).to(x_f32.dtype)
        else:
            shifted = x_f32
        term = shifted * weight_f32[:, tap].view(1, -1, 1)
        acc = term if acc is None else acc + term
    if bias is not None:
        acc = acc + bias.float().view(1, -1, 1)
    if activation is not None:
        acc = F.silu(acc)
    return acc.to(x.dtype)


def _install_causal_conv1d_fallback(model: nn.Module) -> int:
    """Point every Gated-DeltaNet layer at the torch conv when CUDA cannot run.

    Returns the number of patched layers so the caller can log it; a return of
    0 on a Blackwell device means the model shape changed upstream and the
    guard silently stopped protecting anything.
    """
    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            torch_causal_conv1d_update,
        )
    except ImportError:  # pragma: no cover - transformers layout change
        torch_causal_conv1d_update = None
    patched = 0
    for module in model.modules():
        if not hasattr(module, "causal_conv1d_fn"):
            continue
        module.causal_conv1d_fn = _torch_seq_idx_causal_conv1d
        if torch_causal_conv1d_update is not None:
            # Decode path (deploy/eval): transformers already ships a torch
            # equivalent, it just prefers the CUDA one when importable.
            module.causal_conv1d_update = torch_causal_conv1d_update
        patched += 1
    return patched


def _resolve_attention_implementation(config=None, *, requested=None) -> str:
    requested = str(
        (os.environ.get("XVLA_QWEN_ATTN_IMPLEMENTATION", "auto")
         if requested is None else requested) or "auto"
    ).strip().lower()
    if config is not None:
        from ..qwen_image_attention import CAUSAL, image_attention_mode

        if image_attention_mode(config) != CAUSAL:
            if requested not in {"auto", "sdpa"}:
                raise ValueError("bidirectional_image_blocks requires XVLA_QWEN_ATTN_IMPLEMENTATION=sdpa or auto.")
            return "sdpa"
    aliases = {
        "fa2": "flash_attention_2",
        "flash_attn_2": "flash_attention_2",
        "flash_attention_2": "flash_attention_2",
        "fa3": "flash_attention_3",
        "flash_attn_3": "flash_attention_3",
        "flash_attention_3": "flash_attention_3",
        "sdpa": "sdpa",
        "eager": "eager",
    }
    if requested == "auto":
        # Pick the best kernel the hardware can actually run: FA3 on Hopper,
        # FA2 elsewhere (it ships sm_80/90/100/120 cubins), sdpa if flash-attn
        # is missing entirely.
        from transformers.utils import is_flash_attn_2_available

        if _flash_attn_3_supported():
            return "flash_attention_3"
        return (
            "flash_attention_2"
            if is_flash_attn_2_available()
            else "sdpa"
        )
    if requested not in aliases:
        raise ValueError(
            "XVLA_QWEN_ATTN_IMPLEMENTATION must be auto, "
            "flash_attention_2, flash_attention_3, sdpa, or eager; "
            f"got {requested!r}."
        )
    resolved = aliases[requested]
    if resolved == "flash_attention_3" and not _flash_attn_3_supported():
        major = _cuda_capability_major()
        raise RuntimeError(
            "XVLA_QWEN_ATTN_IMPLEMENTATION=flash_attention_3 was requested but "
            "FA3 cannot run here: the installed flash_attn_3 wheel only carries "
            "sm_90a cubins (no PTX), so it needs a Hopper device (compute "
            f"capability 9.x); this device reports major={major}. Failing loudly "
            "rather than silently falling back, so a benchmark arm labelled fa3 "
            "never reports FA2 numbers. Use 'auto' to get the best kernel per "
            "machine (FA2 covers sm80/90/100/120)."
        )
    return resolved


@register_vlm_backend("qwen3_5")
class Qwen35Backend(VLMBackend):
    handles_expert_detach = True
    action_context_mode = QWEN_ACTION_CONTEXT_MODE_V2
    policy_contract = QWEN_POLICY_CONTRACT

    @staticmethod
    def _temporal_embedding_mode(owner) -> str:
        """Return the normalized temporal-embedding mode name.

        Includes legacy averaging, ViT fusion, and language-only history fusion.
        Any other value raises so a typo in a Hydra override is caught
        at forward time rather than silently downgrading to ``"off"``.
        """
        mode = str(
            getattr(owner.config, "qwen_temporal_embedding", "off") or "off"
        ).strip().lower()
        if mode not in {
            "off",
            QWEN_TEMPORAL_EMBEDDING_CONTRACT,
            QWEN_TEMPORAL_EMBEDDING_SPACETIME,
            "vlm_history_current_v1",
            "vlm_history_value_v1",
        }:
            raise ValueError(
                "Unsupported Qwen temporal embedding mode "
                f"{mode!r}; expected 'off', "
                f"{QWEN_TEMPORAL_EMBEDDING_CONTRACT!r}, or "
                f"{QWEN_TEMPORAL_EMBEDDING_SPACETIME!r}, 'vlm_history_current_v1', or 'vlm_history_value_v1'."
            )
        return mode

    @classmethod
    def _temporal_embedding_enabled(cls, owner) -> bool:
        """True when fixed sinusoidal frame positions should be added.

        The spacetime MEM mode is intentionally excluded here: it feeds
        its time signal into the ViT attention block. The fixed mode is
        valid for both the legacy averaged path and P3, where each frame
        remains separately addressable.
        """
        return cls._temporal_embedding_mode(owner) == QWEN_TEMPORAL_EMBEDDING_CONTRACT

    @classmethod
    def _spacetime_enabled(cls, owner) -> bool:
        """True when the Qwen ViT should run the MEM space-time patch."""
        return cls._temporal_embedding_mode(owner) == QWEN_TEMPORAL_EMBEDDING_SPACETIME

    @classmethod
    def _vlm_history_current_enabled(cls, owner) -> bool:
        return cls._temporal_embedding_mode(owner) == "vlm_history_current_v1"

    @classmethod
    def _history_value_enabled(cls, owner) -> bool:
        return cls._temporal_embedding_mode(owner) == "vlm_history_value_v1"

    def _policy_keep_views(self, owner, pixel_values) -> tuple[int, ...]:
        if self._vlm_history_current_enabled(owner) or self._history_value_enabled(owner):
            if bool(getattr(owner.config, "qwen_history_memory_enabled", False)) or bool(
                getattr(owner.config, "qwen_native_video_history_enabled", False)
            ):
                raise ValueError(f"{self._temporal_embedding_mode(owner)} cannot be combined with P3/history bank.")
            visual = getattr(getattr(owner.vlm, "model", None), "visual", None)
            if getattr(visual, "_hy_spacetime_stride", None) is not None:
                raise ValueError(f"{self._temporal_embedding_mode(owner)} requires an unpatched vision tower.")
            return tuple(range(pixel_values.shape[1] if pixel_values.dim() >= 5 else 1))
        return self._native_video_history_views(owner)

    @staticmethod
    def _native_video_history_views(owner) -> tuple[int, ...]:
        """Raw camera view indices whose history frames should traverse the
        Qwen native multi-frame ``image_grid_thw`` path (T > 1).

        The returned indices are in the dataset's *pre*-reorder view order
        (head=0, left_wrist=1, right_wrist=2 for Hy-VLA), matching how
        ``_encode_valid_views`` keys ``keep_per_frame_views`` off
        ``flat_slot % num_views``. They are NOT post-
        ``_reorder_visual_features`` indices: that helper rotates the first
        *valid* view into slot 0 purely to normalize packing order.

        Empty tuple means every view uses the historical single-token
        contract (K=1 in the packed sequence, averaged over frames when
        multi-frame batches are provided). Non-empty tuple activates the
        P3 experiment.
        """
        if not bool(getattr(owner.config, "qwen_native_video_history_enabled", False)):
            return ()
        raw = getattr(owner.config, "qwen_native_video_history_views", ()) or ()
        return tuple(int(v) for v in raw)

    @staticmethod
    def _spacetime_stride(owner) -> int:
        """Return the block-wrap stride for the space-time patch."""
        raw = getattr(owner.config, "qwen_spacetime_stride", None)
        if raw is None:
            return QWEN_SPACETIME_DEFAULT_STRIDE
        stride = int(raw)
        if stride < 1:
            raise ValueError(
                f"qwen_spacetime_stride must be >= 1, got {stride}."
            )
        return stride

    @staticmethod
    def _fixed_time_embedding(
        positions: torch.Tensor,
        dim: int,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return Florence-compatible sinusoidal embeddings for frame slots."""
        if dim <= 0 or dim % 2:
            raise ValueError(f"Qwen temporal embedding dim must be positive/even, got {dim}.")
        positions = positions.to(dtype=torch.float32)
        inv_freq = torch.exp(
            torch.arange(
                0, dim, 2, device=positions.device, dtype=torch.float32
            )
            * (-torch.log(positions.new_tensor(QWEN_TEMPORAL_EMBEDDING_BASE)) / dim)
        )
        angles = positions.unsqueeze(-1) * inv_freq
        embedding = torch.zeros(
            (*positions.shape, dim), device=positions.device, dtype=torch.float32
        )
        embedding[..., 0::2] = torch.sin(angles)
        embedding[..., 1::2] = torch.cos(angles)
        return embedding.to(dtype=dtype)

    @classmethod
    def _add_frame_time_embedding(
        cls,
        features: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Add frame positions to ``[N,S,D]`` features without new parameters."""
        if features.ndim != 3 or positions.ndim != 1:
            raise ValueError(
                "Qwen frame temporal embedding expects features [N,S,D] and "
                f"positions [N], got {tuple(features.shape)} / {tuple(positions.shape)}"
            )
        if features.shape[0] != positions.shape[0]:
            raise ValueError("Qwen frame positions must align with encoded frames.")
        return features + cls._fixed_time_embedding(
            positions,
            int(features.shape[-1]),
            dtype=features.dtype,
        ).unsqueeze(1)

    @staticmethod
    def _language_layers(model) -> nn.ModuleList:
        language_model = model.model.language_model
        layers = getattr(language_model, "layers", None)
        if layers is None:
            raise RuntimeError(
                "Qwen3.5 language model does not expose decoder layers."
            )
        return layers

    @staticmethod
    def _full_attention_modules(model) -> list[tuple[int, nn.Module]]:
        """Decoder layers publishing standard K/V.

        Qwen3.5 alternates linear-attention and full-attention layers; only
        the latter own a ``self_attn`` submodule, so this filter yields the
        8 full-attention depths of a 32-layer backbone.
        """
        modules = []
        for index, layer in enumerate(Qwen35Backend._language_layers(model)):
            attn = getattr(layer, "self_attn", None)
            if attn is None:
                continue
            modules.append((index, attn))
        return modules

    @staticmethod
    def _apply_qwen_rope_to_key(
        key: torch.Tensor,
        cos: torch.Tensor | None,
        sin: torch.Tensor | None,
    ) -> torch.Tensor:
        if cos is None or sin is None:
            return key
        orig_dtype = key.dtype
        key_fp = key.float()
        cos_fp = cos.float()
        sin_fp = sin.float()
        while cos_fp.dim() < key_fp.dim():
            cos_fp = cos_fp.unsqueeze(1)
            sin_fp = sin_fp.unsqueeze(1)
        seq_len = key_fp.shape[-2]
        if cos_fp.shape[-2] != seq_len:
            cos_fp = cos_fp[..., :seq_len, :]
            sin_fp = sin_fp[..., :seq_len, :]
        rotary_dim = int(cos_fp.shape[-1])
        rotated, passed = key_fp[..., :rotary_dim], key_fp[..., rotary_dim:]
        rotated = rotated * cos_fp + _rotate_half(rotated) * sin_fp
        return torch.cat((rotated, passed), dim=-1).to(dtype=orig_dtype)

    @staticmethod
    def _qwen_attn_num_kv_heads(attn: nn.Module, head_dim: int) -> int:
        """K/V head count of one Qwen attention module.

        ``Qwen3_5Attention`` keeps only ``config``/``num_key_value_groups``;
        it never sets ``num_key_value_heads`` on the module, so reading the
        attribute directly raises AttributeError.
        """
        for name in ("num_key_value_heads", "num_kv_heads"):
            value = getattr(attn, name, None)
            if value:
                return int(value)
        value = getattr(
            getattr(attn, "config", None), "num_key_value_heads", None
        )
        if value:
            return int(value)
        # k_proj is Linear(hidden, num_kv_heads * head_dim) by construction.
        out_features = int(
            getattr(getattr(attn, "k_proj", None), "out_features", 0) or 0
        )
        if head_dim > 0 and out_features and out_features % head_dim == 0:
            return out_features // head_dim
        raise RuntimeError(
            "Cannot resolve the Qwen K/V head count for the prefix_kv hook: "
            f"{type(attn).__name__} exposes no num_key_value_heads, no "
            "config.num_key_value_heads, and no usable k_proj geometry."
        )

    @staticmethod
    def _project_qwen_full_attn_kv(
        attn: nn.Module,
        hidden_states: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = hidden_states.shape[:2]
        head_dim = int(getattr(attn, "head_dim"))
        num_kv_heads = Qwen35Backend._qwen_attn_num_kv_heads(attn, head_dim)
        key = attn.k_proj(hidden_states).view(
            batch_size, seq_len, num_kv_heads, head_dim
        )
        k_norm = getattr(attn, "k_norm", None)
        if k_norm is not None:
            key = k_norm(key)
        key = key.transpose(1, 2)
        value = attn.v_proj(hidden_states).view(
            batch_size, seq_len, num_kv_heads, head_dim
        ).transpose(1, 2)
        pos = kwargs.get("position_embeddings")
        cos = sin = None
        if isinstance(pos, tuple) and len(pos) == 2:
            cos, sin = pos
        else:
            cos = kwargs.get("cos")
            sin = kwargs.get("sin")
        key = Qwen35Backend._apply_qwen_rope_to_key(key, cos, sin)
        return key, value

    def _install_prefix_kv_hooks(self, owner, store: dict[int, Any], *, skip_layers=()):
        """Record full-attention inputs; call ``_finalize_prefix_kv`` after.

        Hook bodies stay op-free during Qwen forward and checkpoint replay.
        Opted-in callers reuse native projections; other callers retain the
        delayed projection path. RoPE is finalized after forward in both cases.
        """
        handles = []
        reuse = os.environ.get("XVLA_G05_REUSE_PREFIX_PROJECTIONS", "0") == "1"

        def _make_hook(index: int, attn: nn.Module):
            def _hook(_module, args, kwargs):
                hidden = args[0] if args else kwargs.get("hidden_states")
                if hidden is None:
                    raise RuntimeError(
                        "Qwen G0.5 prefix K/V hook did not receive "
                        "hidden_states."
                    )
                store[index] = _PendingPrefixKV(
                    module=attn,
                    hidden_states=hidden,
                    kwargs=dict(kwargs or {}),
                )

            return _hook

        for index, attn in self._full_attention_modules(owner.vlm):
            if index in skip_layers:
                continue
            handles.append(
                attn.register_forward_pre_hook(
                    _make_hook(index, attn), with_kwargs=True
                )
            )
            if reuse and isinstance(getattr(attn, "k_norm", None), nn.Module):
                # Capture existing projection outputs without doing any tensor
                # work inside checkpointed forwards. Recompute hooks are removed
                # before backward, just like the input-capture hook above.
                def capture_key(_module, _args, output, index=index):
                    pending = store[index]
                    pending.key = output

                def capture_value(_module, _args, output, index=index):
                    pending = store[index]
                    pending.value = output

                handles.append(attn.k_norm.register_forward_hook(capture_key))
                handles.append(attn.v_proj.register_forward_hook(capture_value))
        return handles

    @contextmanager
    def _terminal_prefix_only(self, owner, store):
        """Omit unused work after the last prefix, only in this prefix reader.

        Keep every parameter and optimizer partition. Native decoder/LM calls
        are restored immediately; checkpoint replay retains its native path.
        """
        layers = self._language_layers(owner.vlm)
        language = owner.vlm.model.language_model
        layer = layers[-1]
        if (os.environ.get("XVLA_G05_TERMINAL_PREFIX_ONLY", "0") != "1"
                or getattr(layer, "block_type", None) != "full_attention"
                or float(layer.self_attn.attention_dropout) != 0
                or getattr(language, "gradient_checkpointing", False)
                or any(getattr(item, "gradient_checkpointing", False) for item in layers)):
            yield ()
            return
        index = len(layers) - 1
        had_forward = "forward" in layer.__dict__
        original = layer.forward

        def forward(hidden_states, position_embeddings, **kwargs):
            # The final layer's normalized K/V are the last values consumed
            # by G0.5. Its queries, attention result and MLP have no consumer.
            store[index] = self._project_qwen_full_attn_kv(
                layer.self_attn, layer.input_layernorm(hidden_states),
                dict(kwargs, position_embeddings=position_embeddings),
            )
            return hidden_states

        layer.forward = forward
        try:
            yield (index,)
        finally:
            if had_forward:
                layer.forward = original
            else:
                del layer.forward

    def _finalize_prefix_kv(self, store: dict[int, Any]) -> None:
        """Turn captured full-attention inputs into (key, value) in place."""
        for index, pending in list(store.items()):
            if not isinstance(pending, _PendingPrefixKV):
                continue
            if pending.key is not None and pending.value is not None:
                key = pending.key.transpose(1, 2)
                value = pending.value.view(*pending.key.shape).transpose(1, 2)
                pos = pending.kwargs.get("position_embeddings")
                cos, sin = pos if isinstance(pos, tuple) and len(pos) == 2 else (
                    pending.kwargs.get("cos"), pending.kwargs.get("sin"))
                store[index] = (self._apply_qwen_rope_to_key(key, cos, sin), value)
                continue
            store[index] = self._project_qwen_full_attn_kv(
                pending.module,
                pending.hidden_states,
                dict(pending.kwargs),
            )

    @staticmethod
    def _pad_prefix_kv(
        per_sample: list[dict[int, tuple[torch.Tensor, torch.Tensor]]],
        *,
        num_layers: int,
        batch_size: int,
        prefix_lens: list[int],
        device,
        dtype,
        num_kv_heads: int,
        head_dim: int,
    ) -> tuple[list[tuple[torch.Tensor, torch.Tensor] | None], torch.Tensor]:
        """Right-pad per-sample prefix K/V into one [B,Hkv,T,Dh] batch.

        Returns one entry per *expert* layer; depths whose VLM layer published
        no K/V stay ``None`` so the expert runs suffix self-attention there.
        """
        max_prefix = max(1, max(prefix_lens, default=0))
        prefix_mask = torch.zeros(
            (batch_size, max_prefix), device=device, dtype=torch.bool
        )
        for batch_index, length in enumerate(prefix_lens):
            prefix_mask[batch_index, :length] = True
        padded: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * num_layers
        layer_keys = {
            layer_idx
            for sample in per_sample
            if sample
            for layer_idx in sample
        }
        for layer_idx in sorted(layer_keys):
            key = torch.zeros(
                (batch_size, num_kv_heads, max_prefix, head_dim),
                device=device,
                dtype=dtype,
            )
            value = torch.zeros_like(key)
            for batch_index, sample in enumerate(per_sample):
                if not sample or layer_idx not in sample:
                    continue
                sample_k, sample_v = sample[layer_idx]
                length = int(sample_k.shape[-2])
                key[batch_index, :, :length] = sample_k
                value[batch_index, :, :length] = sample_v
            padded[layer_idx] = (key, value)
        return padded, prefix_mask

    def _capture_prefix_kv(
        self,
        owner,
        *,
        packed: dict,
    ) -> tuple[list, torch.Tensor]:
        """Run the language model once and slice per-sample prefix K/V.

        The Hy policy path never teacher-forces FAST targets, so the prefix is
        exactly the valid (non-padding) packed sequence.
        """
        captured_kv: dict[int, Any] = {}
        time_kwargs = {}
        time_spec = getattr(owner.config, "qwen_history_time_rope", None)
        if time_spec is not None:
            from ..qwen_history_time_rope import coefficients
            values = coefficients(packed["history_time_offsets"], time_spec)
            if values is not None:
                time_kwargs["xvla_time_rope"] = (*values, time_spec["dim"])
        history_value = packed.get("history_value")
        if history_value is not None and history_value["history"].shape[1] > 0:
            from XPolicyLab.policy.Era0_Preview.runtime.models.qwen_history_value import forward_history_value

            _, per_layer = forward_history_value(
                owner.vlm.model.language_model,
                inputs_embeds=packed["inputs_embeds"],
                attention_mask=packed["attention_mask"],
                position_ids=packed["position_ids"],
                **history_value,
                selected_layers=(len(self._language_layers(owner.vlm)) - 1,),
                rope_theta=float(owner.config.qwen_history_value_rope_theta),
                return_prefix_kv=True,
            )
            captured_kv = {i: kv for i, kv in enumerate(per_layer) if kv is not None}
        else:
            with self._terminal_prefix_only(owner, captured_kv) as skip_layers:
                handles = self._install_prefix_kv_hooks(
                    owner, captured_kv, skip_layers=skip_layers)
                try:
                    owner.vlm.model(
                        input_ids=None,
                        inputs_embeds=packed["inputs_embeds"],
                        attention_mask=self._policy_attention_mask(owner, packed),
                        position_ids=packed["position_ids"],
                        use_cache=False,
                        return_dict=True,
                        **time_kwargs,
                    )
                finally:
                    for handle in handles:
                        handle.remove()
            self._finalize_prefix_kv(captured_kv)
        if not captured_kv:
            raise RuntimeError(
                "Qwen G0.5 prefix K/V hooks captured no full-attention layer."
            )
        valid = packed["attention_mask"].to(dtype=torch.bool)
        fast_mask = packed.get("policy_fast_mask")
        prefix_sel = (
            valid
            if fast_mask is None
            else valid & ~fast_mask.to(dtype=torch.bool)
        )
        from ..qwen_image_attention import CURRENT_ONLY_IMAGE_BLOCKS, image_attention_mode

        if image_attention_mode(owner.config) == CURRENT_ONLY_IMAGE_BLOCKS:
            # Match the 2B policy readout: current visual tokens plus semantic
            # text, never historical images or visual sentinels. Keep the K/V
            # in their original per-layer native representation.
            prefix_sel &= packed["policy_semantic_mask"]
        return self._pack_prefix_kv(
            captured_kv, prefix_sel,
            num_layers=len(self._language_layers(owner.vlm)),
        )

    @staticmethod
    def _pack_prefix_kv(
        captured_kv: dict[int, tuple[torch.Tensor, torch.Tensor]],
        prefix_sel: torch.Tensor,
        *,
        num_layers: int,
    ) -> tuple[list, torch.Tensor]:
        """Compact native K/V with one batched gather per tensor.

        Keep token order and zero right padding exactly as the former
        per-sample select/copy path. That path built B CopySlices nodes per
        K/V tensor; backward repeatedly copied the whole padded batch and
        CUDA boolean indexing synchronized for every sample and layer.
        Only the small length vector crosses to the host here, once.
        """
        lengths = prefix_sel.sum(dim=1)
        host_lengths = lengths.tolist()
        if not host_lengths or min(host_lengths) == 0:
            raise RuntimeError(
                "G0.5 policy forward requires at least one non-FAST prefix "
                "token per sample."
            )
        max_prefix = max(host_lengths)
        prefix_mask = torch.arange(max_prefix, device=prefix_sel.device)[None] < lengths[:, None]
        # Valid positions have unique ascending keys; all excluded positions
        # sort after them. Their relative order is irrelevant because masked
        # output positions and their gradients are zero.
        positions = torch.arange(prefix_sel.shape[1], device=prefix_sel.device)
        indices = positions.expand_as(prefix_sel).masked_fill(~prefix_sel, prefix_sel.shape[1])
        indices = indices.argsort(dim=1)[:, :max_prefix]
        padded: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * num_layers
        invalid = ~prefix_mask[:, None, :, None]
        for layer_idx, (key, value) in captured_kv.items():
            gather_index = indices[:, None, :, None].expand(-1, key.shape[1], -1, key.shape[-1])
            padded[layer_idx] = (
                key.gather(2, gather_index).masked_fill(invalid, 0),
                value.gather(2, gather_index).masked_fill(invalid, 0),
            )
        return padded, prefix_mask

    def _resolve_semantic_layer_indices(self, config, model) -> tuple[int, ...]:
        num_layers = len(self._language_layers(model))
        configured = getattr(config, "vlm_policy_semantic_layers", None)
        if configured is None:
            # Four evenly spaced decoder depths, including the final layer.
            indices = tuple(
                sorted(
                    {
                        max(0, round(num_layers * fraction) - 1)
                        for fraction in (0.25, 0.5, 0.75, 1.0)
                    }
                )
            )
        else:
            indices = tuple(int(index) for index in configured)
        if not indices:
            raise ValueError("Qwen semantic layer selection cannot be empty.")
        if tuple(sorted(set(indices))) != indices:
            raise ValueError(
                "Qwen semantic layer indices must be unique and increasing, "
                f"got {indices}."
            )
        if indices[-1] != num_layers - 1:
            raise ValueError(
                "Qwen semantic layer selection must include the final decoder "
                f"layer {num_layers - 1}, got {indices}."
            )
        if indices[0] < 0 or indices[-1] >= num_layers:
            raise ValueError(
                f"Qwen semantic layer indices {indices} exceed depth={num_layers}."
            )
        config.vlm_policy_semantic_layers = list(indices)
        return indices

    def build_policy_modules(self, config, *, model, output_dim):
        max_views = int(getattr(config, "vlm_max_view_embeddings", 8))
        view_embeddings = nn.Embedding(max_views, output_dim)
        nn.init.normal_(view_embeddings.weight, std=0.02)
        self._resolve_semantic_layer_indices(config, model)
        return nn.ModuleDict({"view_embeddings": view_embeddings})

    def build_output_adapter(self, config, model):
        num_depths = len(
            self._resolve_semantic_layer_indices(config, model)
        )
        return _PerDepthOutputAdapter(
            [
                self._build_semantic_adapter(config, model)
                for _ in range(num_depths)
            ]
        )

    def _build_semantic_adapter(self, config, model):
        native_dim = self.native_hidden_size(model)
        output_dim = self.output_hidden_size(config, model)
        if native_dim == output_dim:
            return _PolicyOutputAdapter()
        projection = nn.Linear(native_dim, output_dim, bias=False)
        nn.init.xavier_uniform_(projection.weight)
        return _PolicyOutputAdapter(
            nn.LayerNorm(native_dim),
            projection,
        )

    def build_geometry_adapter(self, config, model):
        native_dim = int(model.config.vision_config.hidden_size)
        output_dim = self.output_hidden_size(config, model)
        if native_dim == output_dim:
            return _PolicyOutputAdapter()
        projection = nn.Linear(native_dim, output_dim, bias=False)
        nn.init.xavier_uniform_(projection.weight)
        return _PolicyOutputAdapter(
            nn.LayerNorm(native_dim),
            projection,
        )

    def validate_runtime(self, config) -> None:
        from ..qwen_image_attention import validate_image_attention

        validate_image_attention(config)
        _require_qwen35_classes()
        if not isinstance(getattr(config, "vlm_config", None), dict):
            raise ValueError(
                "Qwen3.5 backend requires a serialized Qwen config in "
                "XVLAConfig.vlm_config."
            )
        requested_context = str(
            getattr(config, "action_context_mode", "auto") or "auto"
        ).strip().lower()
        allowed_context = {"auto", QWEN_ACTION_CONTEXT_MODE_V2}
        if is_vlm_fast_action_bridge_version(
            getattr(config, "architecture_version", "legacy_cross_emb_v5")
        ):
            # The G0.5 expert consumes native per-layer prefix K/V instead of
            # the projected semantic pyramid, so it owns its own context mode.
            allowed_context = {"auto", PREFIX_KV_ACTION_CONTEXT_MODE}
        if requested_context not in allowed_context:
            raise ValueError(
                "Qwen3.5 requires action_context_mode in "
                f"{sorted(allowed_context)}, got {requested_context!r}."
            )
        requested_contract = str(
            getattr(config, "vlm_policy_contract", "auto") or "auto"
        ).strip().lower()
        if requested_contract not in {"auto", QWEN_POLICY_CONTRACT}:
            raise ValueError(
                "Qwen3.5 requires policy contract=auto or "
                f"{QWEN_POLICY_CONTRACT!r}, got {requested_contract!r}."
            )
        validate_qwen_policy_vision_contract(config)

    def build_model(self, config):
        self.validate_runtime(config)
        Qwen3_5Config, Qwen3_5ForConditionalGeneration = (
            _require_qwen35_classes()
        )
        qwen_config = Qwen3_5Config(**config.vlm_config)
        qwen_config.use_cache = False
        model_dtype = _resolve_model_dtype(config)
        qwen_config.dtype = model_dtype
        qwen_config._attn_implementation = (
            _resolve_attention_implementation(config)
        )
        # Image-block visibility belongs to the language decoder. The vision
        # encoder attends independently within each image and can keep its
        # native varlen FlashAttention path instead of a Python loop per image.
        vision_attention = os.environ.get("XVLA_QWEN_VISION_ATTN_IMPLEMENTATION")
        if vision_attention:
            qwen_config.vision_config.dtype = model_dtype
            qwen_config.vision_config._attn_implementation = (
                _resolve_attention_implementation(requested=vision_attention)
            )
        model = Qwen3_5ForConditionalGeneration(qwen_config).to(model_dtype)
        from ..qwen_history_time_rope import install as install_time_rope
        install_time_rope(model, getattr(config, "qwen_history_time_rope", None))
        if os.environ.get("XVLA_QWEN_FIXED_IMAGE_VISION", "0") == "1":
            from ..qwen_fixed_vision import install_fixed_image_vision

            count = install_fixed_image_vision(model.model.visual)
            logger.info("Qwen fixed-image vision: enabled for %d layers", count)
        logger.info("Qwen attention backends: language=%s vision=%s",
                    model.config.text_config._attn_implementation,
                    model.config.vision_config._attn_implementation)
        if not _causal_conv1d_cuda_supported():
            patched = _install_causal_conv1d_fallback(model)
            logger.warning(
                "causal_conv1d CUDA kernels do not cover this device "
                "(compute capability major=%s); patched %d Gated-DeltaNet "
                "layer(s) to the seq_idx-aware torch conv. Slower than the "
                "fused kernel but numerically equivalent (<=2 bf16 ULP) and, "
                "unlike transformers' own fallback, it honours seq_idx so "
                "padding-free packed sequences stay isolated.",
                _cuda_capability_major(),
                patched,
            )
        # Unlike Florence's standalone decoder, Qwen's tied lm_head is part of
        # the causal-LM checkpoint contract used by Transformers during load.
        # Keeping it does not duplicate the tied embedding parameter, while
        # deleting it makes mark_tied_weights_as_initialized() fail.
        return model

    def native_hidden_size(self, model: nn.Module) -> int:
        return int(model.config.text_config.hidden_size)

    def has_lm_head(self, model: nn.Module) -> bool:
        return hasattr(model, "lm_head")

    @staticmethod
    def _canonical_to_qwen_pixels(images: torch.Tensor) -> torch.Tensor:
        """Convert shared XVLA pixels to the Qwen-native visual distribution."""
        if images.dim() != 4 or images.shape[1] != 3:
            raise ValueError(
                "Qwen3.5 expects canonical images [N,3,H,W], got "
                f"{tuple(images.shape)}"
            )
        mean = images.new_tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
        std = images.new_tensor(_IMAGENET_STD).view(1, 3, 1, 1)
        qwen_mean = images.new_tensor(_QWEN_MEAN).view(1, 3, 1, 1)
        qwen_std = images.new_tensor(_QWEN_STD).view(1, 3, 1, 1)
        rgb = (images * std + mean).clamp(0.0, 1.0)
        return (rgb - qwen_mean) / qwen_std

    @staticmethod
    def _patchify(
        images: torch.Tensor,
        *,
        patch_size: int,
        temporal_patch_size: int,
        merge_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        images = Qwen35Backend._canonical_to_qwen_pixels(images)
        count, channels, height, width = images.shape
        factor = patch_size * merge_size
        if height % factor or width % factor:
            raise ValueError(
                "Canonical XVLA image size must be divisible by Qwen "
                f"patch_size*merge_size={factor}, got {height}x{width}."
            )
        grid_h, grid_w = height // patch_size, width // patch_size
        frames = images.unsqueeze(1).expand(
            count, temporal_patch_size, channels, height, width
        )
        patches = frames.reshape(
            count,
            1,
            temporal_patch_size,
            channels,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )
        patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
        patches = patches.reshape(
            count * grid_h * grid_w,
            channels * temporal_patch_size * patch_size * patch_size,
        )
        grids = torch.tensor(
            [[1, grid_h, grid_w]] * count,
            device=images.device,
            dtype=torch.long,
        )
        return patches.contiguous(), grids

    def _encode_image_native(
        self,
        owner,
        pixel_values: torch.Tensor,
    ) -> _NativeImageFeatures:
        actual_image_size = tuple(int(value) for value in pixel_values.shape[-2:])
        if actual_image_size != QWEN_POLICY_IMAGE_SIZE:
            raise ValueError(
                "Qwen policy images must be 224x224 to preserve the formal "
                f"49/196 token contract, got {actual_image_size}."
            )
        vision_cfg = owner.vlm.config.vision_config
        patches, grids = self._patchify(
            pixel_values,
            patch_size=int(vision_cfg.patch_size),
            temporal_patch_size=int(vision_cfg.temporal_patch_size),
            merge_size=int(vision_cfg.spatial_merge_size),
        )
        vision_parameter = next(owner.vlm.model.visual.parameters())
        patches = patches.to(dtype=vision_parameter.dtype)
        vision = owner.vlm.model.visual
        if getattr(vision, "_xvla_fixed_image_geometry", False):
            from ..qwen_fixed_vision import fixed_image_geometry

            grid_h = actual_image_size[0] // int(vision_cfg.patch_size)
            grid_w = actual_image_size[1] // int(vision_cfg.patch_size)
            geometry = fixed_image_geometry(
                vision, int(pixel_values.shape[0]), device=patches.device,
                height=grid_h, width=grid_w,
            )
            outputs = vision(
                patches, grid_thw=grids, return_dict=True,
                xvla_fixed_image_tokens=grid_h * grid_w, **geometry,
            )
        else:
            outputs = owner.vlm.model.get_image_features(
                pixel_values=patches,
                image_grid_thw=grids,
                return_dict=True,
            )
        image_count = int(pixel_values.shape[0])
        detail_counts = [QWEN_POLICY_GEOMETRY_TOKENS_PER_VIEW] * image_count
        merged_counts = [QWEN_POLICY_LANGUAGE_TOKENS_PER_VIEW] * image_count
        pooled_output = outputs.pooler_output
        if isinstance(pooled_output, torch.Tensor):
            merged_rows = list(
                torch.split(pooled_output, merged_counts, dim=0)
            )
        elif isinstance(pooled_output, (tuple, list)):
            merged_rows = list(pooled_output)
            if len(merged_rows) != len(merged_counts):
                raise RuntimeError(
                    "Qwen get_image_features returned an unexpected number "
                    f"of pooled image rows: expected {len(merged_counts)}, "
                    f"got {len(merged_rows)}."
                )
        else:
            raise TypeError(
                "Qwen get_image_features pooler_output must be a Tensor or "
                f"tuple/list of Tensors, got {type(pooled_output).__name__}."
            )
        if len({int(item.shape[0]) for item in merged_rows}) != 1:
            raise RuntimeError(
                "Qwen merged image token counts differ within a batch: "
                f"{[int(item.shape[0]) for item in merged_rows]}"
            )
        detail_rows = list(
            torch.split(outputs.last_hidden_state, detail_counts, dim=0)
        )
        if len({int(item.shape[0]) for item in detail_rows}) != 1:
            raise RuntimeError(
                "Qwen pre-merge image token counts differ within a batch: "
                f"{[int(item.shape[0]) for item in detail_rows]}"
            )
        if any(
            int(item.shape[0]) != QWEN_POLICY_LANGUAGE_TOKENS_PER_VIEW
            for item in merged_rows
        ):
            raise RuntimeError(
                "Qwen language visual token count drifted from 49/view: "
                f"{[int(item.shape[0]) for item in merged_rows]}."
            )
        if any(
            int(item.shape[0]) != QWEN_POLICY_GEOMETRY_TOKENS_PER_VIEW
            for item in detail_rows
        ):
            raise RuntimeError(
                "Qwen Action geometry token count drifted from 196/view: "
                f"{[int(item.shape[0]) for item in detail_rows]}."
            )
        return _NativeImageFeatures(
            merged=torch.stack(merged_rows, dim=0),
            detail=torch.stack(detail_rows, dim=0),
        )

    def _ensure_spacetime_patch_applied(self, owner) -> None:
        """Lazily install the MEM space-time patch on the Qwen vision tower.

        This is called on the first spacetime encode; subsequent calls
        are a cheap `isinstance` scan (idempotent). We deliberately do
        NOT install the patch inside `Qwen35Backend.__init__` because
        that would attach `SpaceTimeVisionBlock` wrappers to every Qwen
        checkpoint save/load in the repository (Florence and legacy
        Qwen off/fixed_sinusoidal_v1 runs), inflating diffs and risking
        state_dict-shape audits.
        """
        from XPolicyLab.policy.Era0_Preview.runtime.models.qwen_spacetime import apply_spacetime_patch

        visual = owner.vlm.model.visual
        stride = self._spacetime_stride(owner)
        if getattr(visual, "_hy_spacetime_stride", None) == stride:
            return
        apply_spacetime_patch(visual, stride=stride)

    def _encode_image_native_spacetime(
        self,
        owner,
        pixel_values: torch.Tensor,
        group_sizes: torch.Tensor,
        group_sizes_list: tuple[int, ...] | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Run the Qwen ViT with MEM time mixing; return current-frame tokens.

        Signature choice: this function returns per-group ``(merged,
        detail)`` lists rather than ``_NativeImageFeatures``, because it
        already collapses the K frames per group down to one visual
        token bundle. The caller stacks the lists to obtain the
        ``[num_valid_groups, 49/196, D]`` tensors the downstream flow
        expects for a K=1 encode.
        """
        actual_image_size = tuple(int(value) for value in pixel_values.shape[-2:])
        if actual_image_size != QWEN_POLICY_IMAGE_SIZE:
            raise ValueError(
                "Qwen policy images must be 224x224 to preserve the formal "
                f"49/196 token contract, got {actual_image_size}."
            )
        if int(group_sizes.sum().item()) != int(pixel_values.shape[0]):
            raise RuntimeError(
                "Qwen spacetime encode: sum(group_sizes) "
                f"({int(group_sizes.sum().item())}) does not match number of "
                f"pixel_value images ({int(pixel_values.shape[0])})."
            )

        self._ensure_spacetime_patch_applied(owner)

        vision_cfg = owner.vlm.config.vision_config
        patches, grids = self._patchify(
            pixel_values,
            patch_size=int(vision_cfg.patch_size),
            temporal_patch_size=int(vision_cfg.temporal_patch_size),
            merge_size=int(vision_cfg.spatial_merge_size),
        )
        vision_parameter = next(owner.vlm.model.visual.parameters())
        patches = patches.to(dtype=vision_parameter.dtype)

        from XPolicyLab.policy.Era0_Preview.runtime.models.qwen_spacetime import encode_with_current_frame_only

        visual = owner.vlm.model.visual
        merged_rows, detail_rows = encode_with_current_frame_only(
            visual=visual,
            get_image_features_fn=owner.vlm.model.get_image_features,
            pixel_values_patches=patches,
            image_grid_thw=grids,
            group_sizes=group_sizes,
            tokens_per_frame=QWEN_POLICY_GEOMETRY_TOKENS_PER_VIEW,
            group_sizes_list=group_sizes_list,
        )

        if len(merged_rows) != int(group_sizes.shape[0]):
            raise RuntimeError(
                "Qwen spacetime encode: encode_with_current_frame_only "
                f"returned {len(merged_rows)} groups; expected "
                f"{int(group_sizes.shape[0])}."
            )
        # Sanity-check token counts. `encode_with_current_frame_only`
        # already guarantees 49/196 per view under the standard Qwen
        # policy contract, but this hard-fails if a future config drift
        # violates it -- matches the contract audit in
        # ``_encode_image_native``.
        for row in merged_rows:
            if int(row.shape[0]) != QWEN_POLICY_LANGUAGE_TOKENS_PER_VIEW:
                raise RuntimeError(
                    "Qwen spacetime merged token count drifted from 49/view: "
                    f"{int(row.shape[0])}."
                )
        for row in detail_rows:
            if int(row.shape[0]) != QWEN_POLICY_GEOMETRY_TOKENS_PER_VIEW:
                raise RuntimeError(
                    "Qwen spacetime detail token count drifted from 196/view: "
                    f"{int(row.shape[0])}."
                )
        return merged_rows, detail_rows

    def encode_image(self, owner, pixel_values, *, frame_mask=None):
        if pixel_values.dim() == 4:
            native = self._encode_image_native(owner, pixel_values).merged
            return owner.vlm_output_adapter(native)
        if pixel_values.dim() != 5:
            raise ValueError(
                "Qwen temporal images must be [N,T,3,H,W], got "
                f"{tuple(pixel_values.shape)}"
            )
        batch_size, num_frames = pixel_values.shape[:2]
        if self._spacetime_enabled(owner) and num_frames > 1:
            # The legacy [B,T,C,H,W] fast path is exclusively used by
            # SoftPromptedTransformer.forward_image_encoder for the older
            # single-view MEM helper; the RoboDojo policy always reaches
            # this backend through `_encode_valid_views`, which owns the
            # supported spacetime path. Fail loudly instead of silently
            # falling back to time averaging so callers surface the
            # mismatch during smoke-testing rather than after a long run.
            raise RuntimeError(
                "qwen_temporal_embedding=spacetime_mem_v1 is supported only "
                "through the multi-view policy path (`_encode_valid_views`); "
                "the legacy per-view temporal average was invoked with "
                f"num_frames={num_frames}. Enable spacetime only on RoboDojo "
                "training runs whose data pipeline emits the three-view "
                "history contract."
            )
        native = self._encode_image_native(
            owner, pixel_values.reshape(-1, *pixel_values.shape[2:])
        ).merged.view(
            batch_size,
            num_frames,
            -1,
            self.native_hidden_size(owner.vlm),
        )
        if frame_mask is None:
            mask = torch.ones(
                (batch_size, num_frames),
                device=pixel_values.device,
                dtype=native.dtype,
            )
        else:
            mask = torch.as_tensor(
                frame_mask, device=pixel_values.device, dtype=native.dtype
            )
        if self._temporal_embedding_enabled(owner) and num_frames > 1:
            # Keep shared temporal slots; right-aligned K=1 views use the current slot.
            positions = torch.arange(num_frames, device=pixel_values.device, dtype=torch.long).expand(batch_size, -1)

            temporal = self._fixed_time_embedding(
                positions,
                int(native.shape[-1]),
                dtype=native.dtype,
            )
            native = native + temporal.unsqueeze(2)
        weights = mask / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        native = (native * weights[:, :, None, None]).sum(dim=1)
        return owner.vlm_output_adapter(native)

    def _encode_valid_views(
        self,
        owner,
        pixel_values: torch.Tensor,
        image_mask: torch.Tensor | None,
        *,
        require_visual: bool,
        keep_per_frame_views: tuple[int, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode per-view visual features.

        Returns ``(merged, detail, view_mask, frames_per_view)`` where
        ``merged`` is ``[B, V, 49 * F_max, D]`` and ``frames_per_view`` is
        ``[B, V]`` long tensor giving the temporal patch count each view
        occupies in the packed sequence. In the default single-frame path
        ``F_max == 1`` so ``merged`` retains its historical ``[B, V, 49, D]``
        shape and ``frames_per_view`` is all ones.

        When ``keep_per_frame_views`` is non-empty, those view indices skip
        the ``index_add_`` averaging and instead concatenate all valid frame
        tokens end-to-end (padded to ``F_max`` frames per view). The frame
        count reported for each such view equals its number of valid frames,
        allowing the packing code to retain their temporal order. P3 pools
        older frames to 4x4; vlm_history_current_v1 retains all 49 tokens.
        Passing ``keep_per_frame_views`` while
        ``spacetime_mem_v1`` is enabled is rejected to avoid double time
        conditioning.
        """
        keep_set: set[int] = set(int(v) for v in (keep_per_frame_views or ()))
        if keep_set:
            # P3 preserves time as an ordered sequence of independently pooled
            # history blocks. ViT-level spacetime mixing would duplicate the
            # temporal signal; fixed sinusoidal positions are intentionally
            # retained and are applied before per-frame pooling.
            temporal_mode = self._temporal_embedding_mode(owner)
            if temporal_mode == QWEN_TEMPORAL_EMBEDDING_SPACETIME:
                raise ValueError(
                    "keep_per_frame_views (qwen_native_video_history) does not "
                    "support qwen_temporal_embedding='spacetime_mem_v1'; got "
                    f"{temporal_mode!r}. Disable one of the two."
                )
        if pixel_values.dim() == 4:
            pixel_values = pixel_values.unsqueeze(1)
        if pixel_values.dim() == 6:
            batch_size, num_views, num_frames, channels, height, width = pixel_values.shape
            if (self._vlm_history_current_enabled(owner) or self._history_value_enabled(owner)) and image_mask is not None:
                check_mask = torch.as_tensor(image_mask, dtype=torch.bool)
                if check_mask.dim() == 3 and bool(
                    (check_mask.any(dim=-1) & ~check_mask[:, :, -1]).any()
                ):
                    raise ValueError(
                        f"{self._temporal_embedding_mode(owner)} requires a valid current slot for every active view."
                    )
            host_frame_counts: tuple[int, ...] | None = None
            host_has_visual: bool | None = None
            host_all_samples_visual: bool | None = None
            host_valid_views = None
            if image_mask is None:
                host_frame_counts = (num_frames,) * (batch_size * num_views)
            elif not image_mask.is_cuda:
                host_frame_mask = torch.as_tensor(image_mask, dtype=torch.bool)
                if host_frame_mask.dim() == 2:
                    host_frame_mask = host_frame_mask.unsqueeze(-1).expand(
                        batch_size, num_views, num_frames
                    )
                elif tuple(host_frame_mask.shape) != (
                    batch_size,
                    num_views,
                    num_frames,
                ):
                    raise ValueError(
                        "Qwen policy image_mask for history images must be [B,V,K] "
                        f"or [B,V], got {tuple(host_frame_mask.shape)} for "
                        f"pixel_values={tuple(pixel_values.shape)}"
                    )
                host_has_visual = bool(host_frame_mask.any())
                host_valid_views_by_sample = host_frame_mask.any(dim=2)
                host_all_samples_visual = bool(host_valid_views_by_sample.any(dim=1).all())
                host_valid_views = host_valid_views_by_sample.reshape(-1)
                host_frame_counts = tuple(
                    int(value)
                    for value in host_frame_mask.reshape(
                        batch_size * num_views, num_frames
                    )[host_valid_views].sum(dim=1).tolist()
                )

            if image_mask is None:
                frame_mask = torch.ones(
                    (batch_size, num_views, num_frames),
                    device=pixel_values.device,
                    dtype=torch.bool,
                )
            else:
                frame_mask = torch.as_tensor(
                    image_mask,
                    device=pixel_values.device,
                    dtype=torch.bool,
                )
                if frame_mask.dim() == 2 and tuple(frame_mask.shape) == (batch_size, num_views):
                    frame_mask = frame_mask.unsqueeze(-1).expand(batch_size, num_views, num_frames)
                elif tuple(frame_mask.shape) != (batch_size, num_views, num_frames):
                    raise ValueError(
                        "Qwen policy image_mask for history images must be [B,V,K] "
                        f"or [B,V], got {tuple(frame_mask.shape)} for "
                        f"pixel_values={tuple(pixel_values.shape)}"
                    )
            mask = frame_mask.any(dim=2)
            if require_visual and (
                not host_all_samples_visual
                if host_all_samples_visual is not None
                else not bool(mask.any(dim=1).all())
            ):
                raise ValueError("image_mask requires at least one valid view per sample.")
            flat_mask = mask.reshape(-1)
            merged_dim = self.native_hidden_size(owner.vlm)
            detail_dim = int(owner.vlm.config.vision_config.hidden_size)
            if host_has_visual is False or (
                host_has_visual is None and not bool(flat_mask.any())
            ):
                return (
                    pixel_values.new_zeros((batch_size, num_views, 0, merged_dim)),
                    pixel_values.new_zeros((batch_size, num_views, 0, detail_dim)),
                    mask,
                    torch.zeros(
                        (batch_size, num_views),
                        device=pixel_values.device,
                        dtype=torch.long,
                    ),
                )
            flat_pixels = pixel_values.reshape(
                batch_size * num_views,
                num_frames,
                channels,
                height,
                width,
            )
            flat_frame_mask = frame_mask.reshape(batch_size * num_views, num_frames)
            valid_pixels = flat_pixels[flat_mask]
            valid_frame_mask = flat_frame_mask[flat_mask]
            valid_pixel_mask = valid_frame_mask.reshape(-1)
            frame_counts = valid_frame_mask.sum(dim=1).to(dtype=torch.long)
            spacetime_active = self._spacetime_enabled(owner) and num_frames > 1
            # frames_per_view: [B, V] long tensor giving the packed T for
            # each view. Default (single-frame) is all ones. Views in
            # keep_set report their actual per-view valid-frame counts.
            frames_per_view_flat = torch.zeros(
                batch_size * num_views,
                device=pixel_values.device,
                dtype=torch.long,
            )
            if spacetime_active:
                merged_rows, detail_rows = self._encode_image_native_spacetime(
                    owner,
                    valid_pixels.reshape(-1, channels, height, width)[valid_pixel_mask],
                    group_sizes=frame_counts,
                    group_sizes_list=host_frame_counts,
                )
                merged = torch.stack(merged_rows, dim=0)
                detail = torch.stack(detail_rows, dim=0)
                # single-frame contract: T=1 downstream
                frames_per_view_flat[flat_mask] = 1
            else:
                valid_features = self._encode_image_native(
                    owner,
                    valid_pixels.reshape(-1, channels, height, width)[valid_pixel_mask],
                )
                if self._temporal_embedding_enabled(owner) and bool(
                    (frame_counts > 1).any()
                ):
                    slot_positions = torch.arange(
                        num_frames, device=pixel_values.device, dtype=torch.long
                    ).expand(valid_frame_mask.shape[0], -1)
                    valid_positions = slot_positions[valid_frame_mask]
                    valid_features = _NativeImageFeatures(
                        merged=self._add_frame_time_embedding(
                            valid_features.merged, valid_positions
                        ),
                        detail=self._add_frame_time_embedding(
                            valid_features.detail, valid_positions
                        ),
                    )
                # valid_features.merged has shape [sum(frame_counts), 49, D].
                # Two code paths merge them back per view:
                #   default: index_add_ average -> [n_valid_views, 49, D]
                #   keep_per_frame_views: pad to [n_valid_views, F_max*49, D]
                #     with per-view frame count preserved for downstream.
                n_valid_views = int(valid_pixels.shape[0])
                if keep_set:
                    # ``flat_mask`` is ``mask.reshape(-1)`` over a [B, V]
                    # tensor, so flat slot == b * num_views + v and the view
                    # column is ``slot % num_views``. NOTE these are the
                    # *pre*-reorder (raw camera) view indices, which is what
                    # ``qwen_native_video_history_views`` names.
                    valid_view_indices = torch.nonzero(
                        flat_mask, as_tuple=False
                    ).flatten()
                    valid_view_col = valid_view_indices % num_views
                    keep_mask_valid = torch.zeros(
                        n_valid_views,
                        device=pixel_values.device,
                        dtype=torch.bool,
                    )
                    for v in keep_set:
                        keep_mask_valid |= valid_view_col == v
                    # Per-view T: valid frame count for expanded views, 1 for
                    # the rest (which keep the historical averaged contract).
                    per_view_T = torch.where(
                        keep_mask_valid, frame_counts, torch.ones_like(frame_counts)
                    )
                    tokens_per_frame = valid_features.merged.shape[1]
                    merged_dim_out = valid_features.merged.shape[2]
                    # Pad to the *static* K from the input tensor rather than
                    # the batch-dependent max, so the intermediate shape does
                    # not vary across microbatches (avoids torch.compile /
                    # DDP-bucket churn). Trailing padding is sliced off by
                    # ``_pack_multimodal`` using ``frames_per_view``.
                    frame_capacity = num_frames
                    # Row -> owning view, and row -> frame slot within its view.
                    # ``valid_features`` rows are grouped per view and ordered
                    # oldest -> newest inside each group.
                    frame_owner = torch.arange(
                        n_valid_views,
                        device=pixel_values.device,
                        dtype=torch.long,
                    ).repeat_interleave(frame_counts)
                    group_start = torch.cumsum(frame_counts, dim=0) - frame_counts
                    row_index = torch.arange(
                        valid_features.merged.shape[0],
                        device=pixel_values.device,
                        dtype=torch.long,
                    )
                    frame_slot = row_index - group_start[frame_owner]
                    # Expanded views scatter each frame to its own slot;
                    # non-expanded views accumulate every frame into slot 0
                    # and are divided by their frame count afterwards, which
                    # reproduces the legacy mean exactly.
                    keep_row = keep_mask_valid[frame_owner]
                    dest_slot = torch.where(
                        keep_row, frame_slot, torch.zeros_like(frame_slot)
                    )
                    merged_slots = valid_features.merged.new_zeros(
                        (n_valid_views * frame_capacity, tokens_per_frame, merged_dim_out)
                    )
                    merged_slots.index_add_(
                        0, frame_owner * frame_capacity + dest_slot, valid_features.merged
                    )
                    denom = torch.ones(
                        (n_valid_views, frame_capacity),
                        device=pixel_values.device,
                        dtype=valid_features.merged.dtype,
                    )
                    denom[:, 0] = torch.where(
                        keep_mask_valid,
                        torch.ones_like(frame_counts),
                        frame_counts,
                    ).to(dtype=valid_features.merged.dtype)
                    merged_slots = merged_slots / denom.reshape(-1, 1, 1)
                    merged = merged_slots.reshape(
                        n_valid_views, frame_capacity * tokens_per_frame, merged_dim_out
                    )
                    # Detail (pre-merge 196-token) stream feeds the
                    # aux_visual / dorsal geometry path, which enforces a
                    # fixed 196-tokens-per-view contract and therefore cannot
                    # carry K frames. For an *expanded* view we hand it the
                    # CURRENT frame rather than the temporal mean: the mean of
                    # 6 frames is a motion-blurred image whose geometry tokens
                    # no longer localize anything precisely, while history is
                    # already delivered losslessly through the merged/semantic
                    # channel above. This also makes P3 differ from a K=1 run
                    # *only* by the added history tokens, which is the
                    # ablation we actually want to read.
                    #
                    # Slot K-1 is the current frame (left-padded history; see
                    # robodojo_dataset.get_history_indices, which forces
                    # indices[-1] = step_id), so within a compacted group the
                    # current frame is the LAST row.
                    detail_sum = valid_features.detail.new_zeros(
                        (
                            n_valid_views,
                            valid_features.detail.shape[1],
                            valid_features.detail.shape[2],
                        )
                    )
                    detail_sum.index_add_(0, frame_owner, valid_features.detail)
                    detail_mean = detail_sum / frame_counts.to(
                        dtype=detail_sum.dtype
                    ).view(-1, 1, 1)
                    current_row = group_start + frame_counts - 1
                    detail_current = valid_features.detail[current_row]
                    detail = torch.where(
                        keep_mask_valid.view(-1, 1, 1), detail_current, detail_mean
                    )
                    frames_per_view_flat[flat_mask] = per_view_T
                else:
                    frame_owner = torch.arange(
                        n_valid_views,
                        device=pixel_values.device,
                        dtype=torch.long,
                    ).repeat_interleave(frame_counts)
                    merged = valid_features.merged.new_zeros(
                        (
                            n_valid_views,
                            valid_features.merged.shape[1],
                            valid_features.merged.shape[2],
                        )
                    )
                    detail = valid_features.detail.new_zeros(
                        (
                            n_valid_views,
                            valid_features.detail.shape[1],
                            valid_features.detail.shape[2],
                        )
                    )
                    merged.index_add_(0, frame_owner, valid_features.merged)
                    detail.index_add_(0, frame_owner, valid_features.detail)
                    merged = merged / frame_counts.to(dtype=merged.dtype).view(-1, 1, 1)
                    detail = detail / frame_counts.to(dtype=detail.dtype).view(-1, 1, 1)
                    frames_per_view_flat[flat_mask] = 1
            merged_visual = merged.new_zeros(
                (batch_size * num_views, merged.shape[1], merged.shape[2])
            )
            detail_visual = detail.new_zeros(
                (batch_size * num_views, detail.shape[1], detail.shape[2])
            )
            merged_visual[flat_mask] = merged
            detail_visual[flat_mask] = detail
            return (
                merged_visual.view(batch_size, num_views, merged.shape[1], merged.shape[2]),
                detail_visual.view(batch_size, num_views, detail.shape[1], detail.shape[2]),
                mask,
                frames_per_view_flat.view(batch_size, num_views),
            )
        if pixel_values.dim() != 5:
            raise ValueError(
                "Qwen policy images must be [B,V,3,H,W] or [B,V,K,3,H,W], got "
                f"{tuple(pixel_values.shape)}"
            )
        batch_size, num_views = pixel_values.shape[:2]
        mask = owner._normalize_view_mask(
            image_mask,
            batch_size=batch_size,
            num_views=num_views,
            device=pixel_values.device,
            name="image_mask",
            require_non_empty=require_visual,
        )
        flat_mask = mask.reshape(-1)
        merged_dim = self.native_hidden_size(owner.vlm)
        detail_dim = int(owner.vlm.config.vision_config.hidden_size)
        if not bool(flat_mask.any()):
            return (
                pixel_values.new_zeros(
                    (batch_size, num_views, 0, merged_dim)
                ),
                pixel_values.new_zeros(
                    (batch_size, num_views, 0, detail_dim)
                ),
                mask,
                torch.zeros(
                    (batch_size, num_views),
                    device=pixel_values.device,
                    dtype=torch.long,
                ),
            )
        valid_features = self._encode_image_native(
            owner, pixel_values.reshape(-1, *pixel_values.shape[2:])[flat_mask]
        )
        merged = valid_features.merged
        detail = valid_features.detail
        merged_visual = merged.new_zeros(
            (batch_size * num_views, merged.shape[1], merged.shape[2])
        )
        detail_visual = detail.new_zeros(
            (batch_size * num_views, detail.shape[1], detail.shape[2])
        )
        merged_visual[flat_mask] = merged
        detail_visual[flat_mask] = detail
        # 5-D path is single-frame per view; T is always 1 wherever valid.
        frames_per_view_flat = torch.zeros(
            batch_size * num_views,
            device=pixel_values.device,
            dtype=torch.long,
        )
        frames_per_view_flat[flat_mask] = 1
        return (
            merged_visual.view(
                batch_size, num_views, merged.shape[1], merged.shape[2]
            ),
            detail_visual.view(
                batch_size, num_views, detail.shape[1], detail.shape[2]
            ),
            mask,
            frames_per_view_flat.view(batch_size, num_views),
        )

    def precompute_policy_visual(
        self,
        owner,
        pixel_values: torch.Tensor,
        image_mask: torch.Tensor | None,
    ) -> _QwenPolicyVisualCache:
        keep_views = self._policy_keep_views(owner, pixel_values)
        merged, detail, mask, frames_per_view = self._encode_valid_views(
            owner,
            pixel_values,
            image_mask,
            require_visual=True,
            keep_per_frame_views=keep_views,
        )
        return _QwenPolicyVisualCache(
            merged=merged,
            detail=detail,
            mask=mask,
            frames_per_view=frames_per_view,
        )

    def _resolve_policy_visual(
        self,
        owner,
        *,
        pixel_values: torch.Tensor,
        image_mask: torch.Tensor | None,
        precomputed_features,
        require_visual: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        keep_views = self._policy_keep_views(owner, pixel_values)
        if precomputed_features is None:
            return self._encode_valid_views(
                owner,
                pixel_values,
                image_mask,
                require_visual=require_visual,
                keep_per_frame_views=keep_views,
            )
        if not isinstance(precomputed_features, _QwenPolicyVisualCache):
            raise TypeError(
                "Qwen3.5 precomputed_features must be a "
                "_QwenPolicyVisualCache from the same forward."
            )
        batch_size, num_views = pixel_values.shape[:2]
        requested_image_mask = image_mask
        if requested_image_mask is not None:
            requested_image_mask = torch.as_tensor(
                requested_image_mask,
                device=pixel_values.device,
                dtype=torch.bool,
            )
            if requested_image_mask.dim() == 3:
                requested_image_mask = requested_image_mask.any(dim=2)
        requested_mask = owner._normalize_view_mask(
            requested_image_mask,
            batch_size=batch_size,
            num_views=num_views,
            device=pixel_values.device,
            name="image_mask",
            require_non_empty=require_visual,
        )
        if (
            precomputed_features.merged.shape[:2]
            != (batch_size, num_views)
            or precomputed_features.detail.shape[:2]
            != (batch_size, num_views)
        ):
            raise ValueError(
                "Qwen shared visual cache shape does not match the current "
                f"batch/views {(batch_size, num_views)}."
            )
        if not torch.equal(requested_mask, precomputed_features.mask):
            raise ValueError(
                "Qwen shared visual cache mask differs from the current path; "
                "fall back to an independent visual encode."
            )
        cached_frames = precomputed_features.frames_per_view
        if cached_frames is None:
            # Legacy cache from a checkpoint that did not carry per-view T.
            cached_frames = precomputed_features.mask.to(dtype=torch.long)
        return (
            precomputed_features.merged,
            precomputed_features.detail,
            precomputed_features.mask,
            cached_frames,
        )

    def _encode_history_memory(
        self,
        owner,
        *,
        pixel_values: torch.Tensor,
        image_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode non-current frames as separately addressable spatial memory.

        The temporal Qwen path deliberately returns only each view's current
        frame. This branch keeps that path intact and independently encodes
        slots ``[:,-1]`` into a compact spatial bank. Token order is fixed as
        view, frame, pooled-row-major-position, so masks, relative timestamps,
        and view IDs remain aligned under variable per-view history lengths.
        """
        if pixel_values.dim() != 6:
            raise ValueError(
                "Qwen history memory requires images [B,V,K,3,H,W], got "
                f"{tuple(pixel_values.shape)}."
            )
        batch_size, num_views, num_frames, channels, height, width = (
            pixel_values.shape
        )
        if image_mask is None:
            frame_mask = torch.ones(
                (batch_size, num_views, num_frames),
                device=pixel_values.device,
                dtype=torch.bool,
            )
        else:
            frame_mask = torch.as_tensor(
                image_mask, device=pixel_values.device, dtype=torch.bool
            )
            if frame_mask.dim() == 2 and frame_mask.shape == (
                batch_size,
                num_views,
            ):
                frame_mask = frame_mask.unsqueeze(-1).expand(
                    batch_size, num_views, num_frames
                )
            elif frame_mask.shape != (batch_size, num_views, num_frames):
                raise ValueError(
                    "Qwen history-memory image_mask must be [B,V,K] or [B,V], "
                    f"got {tuple(frame_mask.shape)}."
                )
        history_mask = frame_mask.clone()
        history_mask[:, :, -1] = False
        pool_size = int(owner.config.qwen_history_memory_pool_size)
        detail_dim = int(owner.vlm.config.vision_config.hidden_size)
        tokens_per_frame = pool_size * pool_size
        output_shape = (
            batch_size,
            num_views,
            num_frames,
            tokens_per_frame,
            detail_dim,
        )
        pooled = pixel_values.new_zeros(output_shape)
        flat_history_mask = history_mask.reshape(-1)
        if bool(flat_history_mask.any()):
            native = self._encode_image_native(
                owner,
                pixel_values.reshape(
                    batch_size * num_views * num_frames,
                    channels,
                    height,
                    width,
                )[flat_history_mask],
            )
            detail = native.detail
            source_grid = int(round(detail.shape[1] ** 0.5))
            if source_grid * source_grid != detail.shape[1]:
                raise RuntimeError(
                    "Qwen history memory requires a square spatial detail grid, "
                    f"got {detail.shape[1]} tokens."
                )
            detail = detail.transpose(1, 2).reshape(
                detail.shape[0], detail.shape[2], source_grid, source_grid
            )
            detail = F.adaptive_avg_pool2d(
                detail, (pool_size, pool_size)
            ).flatten(2).transpose(1, 2)
            pooled.reshape(-1, tokens_per_frame, detail_dim)[
                flat_history_mask
            ] = detail
        memory_mask = history_mask.unsqueeze(-1).expand(
            batch_size, num_views, num_frames, tokens_per_frame
        ).reshape(batch_size, -1)
        offsets = (
            torch.arange(
                num_frames, device=pixel_values.device, dtype=torch.long
            )
            - (num_frames - 1)
        )
        time_offsets = offsets.view(1, 1, num_frames, 1).expand(
            batch_size, num_views, num_frames, tokens_per_frame
        ).reshape(batch_size, -1)
        view_ids = torch.arange(
            num_views, device=pixel_values.device, dtype=torch.long
        ).view(1, num_views, 1, 1).expand(
            batch_size, num_views, num_frames, tokens_per_frame
        ).reshape(batch_size, -1)
        return (
            pooled.reshape(batch_size, -1, detail_dim),
            memory_mask,
            time_offsets,
            view_ids,
        )

    def _forward_language_with_selected_states(
        self,
        owner,
        **language_kwargs,
    ) -> list[torch.Tensor]:
        layers = self._language_layers(owner.vlm)
        indices = tuple(owner.config.vlm_policy_semantic_layers)
        history_value = language_kwargs.pop("history_value", None)
        if history_value is not None and history_value["history"].shape[1] > 0:
            from XPolicyLab.policy.Era0_Preview.runtime.models.qwen_history_value import forward_history_value

            language_kwargs.pop("input_ids", None)
            return forward_history_value(
                owner.vlm.model.language_model,
                **language_kwargs, **history_value, selected_layers=indices,
                rope_theta=float(getattr(
                    owner.config, "qwen_history_value_rope_theta", 10000.0)),
            )
        captured: dict[int, torch.Tensor] = {}
        handles = []

        def _capture(index):
            def _hook(_module, _inputs, output):
                captured[index] = output[0] if isinstance(output, tuple) else output

            return _hook

        for index in indices[:-1]:
            handles.append(layers[index].register_forward_hook(_capture(index)))
        try:
            outputs = owner.vlm.model.language_model(
                **language_kwargs,
                use_cache=False,
                return_dict=True,
            )
        finally:
            for handle in handles:
                handle.remove()
        missing = [index for index in indices[:-1] if index not in captured]
        if missing:
            raise RuntimeError(
                f"Qwen semantic layer hooks did not run for layers {missing}."
            )
        # The public final state includes Qwen's terminal normalization and is
        # exactly the representation used by the previous XVLA-Qwen contract.
        return [captured[index] for index in indices[:-1]] + [
            outputs.last_hidden_state
        ]

    def _pack_multimodal(
        self,
        owner,
        *,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None,
        native_visual: torch.Tensor,
        visual_mask: torch.Tensor,
        visual_mask_host: torch.Tensor | None = None,
        frames_per_view: torch.Tensor | None = None,
        target_ids: list[torch.Tensor] | None = None,
        target_context_ids: list[torch.Tensor] | None = None,
        history_slot_mask: torch.Tensor | None = None,
        _history_value_current_only: bool = False,
    ) -> Dict[str, torch.Tensor]:
        batch_size, num_views, num_visual_tokens, native_dim = native_visual.shape
        if self._history_value_enabled(owner) and not _history_value_current_only:
            if target_ids is not None or target_context_ids is not None:
                raise ValueError("vlm_history_value_v1 currently supports action-only SFT, not LM cotrain.")
            if not bool(visual_mask[:, 0].all()):
                raise ValueError("vlm_history_value_v1 requires the raw head camera (view 0).")
            capacity = num_visual_tokens // QWEN_POLICY_LANGUAGE_TOKENS_PER_VIEW
            counts = frames_per_view if frames_per_view is not None else visual_mask.long()
            if num_visual_tokens % 49 or capacity < 1 or bool(
                (((counts < 1) | (counts > capacity)) & visual_mask).any()
            ):
                raise ValueError("History V fusion requires current image tokens.")
            frames = native_visual.reshape(batch_size, num_views, capacity, 49, native_dim)
            gather = (counts.clamp_min(1) - 1)[:, :, None, None, None].expand(-1, -1, 1, 49, native_dim)
            current = frames.gather(2, gather).squeeze(2)
            packed = self._pack_multimodal(
                owner, input_ids=input_ids, attention_mask=attention_mask,
                native_visual=current, visual_mask=visual_mask,
                visual_mask_host=visual_mask_host,
                frames_per_view=visual_mask.long(), _history_value_current_only=True,
            )
            h = capacity - 1
            slots = torch.arange(h, device=native_visual.device)[None].expand(batch_size, -1)
            history_mask = slots < (counts[:, :1] - 1)
            # Temporal offsets are in HISTORY-SLOT units, signed, 0 at the
            # current frame. One unit equals dataset.img_history_interval
            # dataset steps (train_xvla.py asserts the config match), so a
            # masked-out slot still shows up as a gap of 2 and the metric
            # spacing is preserved up to that fixed scale. Feeding raw step
            # counts (40 per slot) instead would push the frame code far out
            # of its well-conditioned range for no extra information.
            offsets = slots - (counts[:, :1] - 1)
            if history_slot_mask is not None and history_slot_mask.ndim == 3:
                head_mask = history_slot_mask[:, 0].to(device=native_visual.device, dtype=torch.bool)
                if head_mask.shape[1] != capacity or not bool(head_mask[:, -1].all()):
                    raise ValueError("History V fusion requires the final current slot.")
                if not torch.equal(head_mask.sum(dim=1), counts[:, 0]):
                    raise ValueError("History V frame mask and encoded frame counts differ.")
                original_slots = torch.arange(capacity, device=native_visual.device).expand(batch_size, -1)
                original_slots = original_slots.masked_fill(~head_mask, capacity).sort(dim=1).values
                offsets = original_slots[:, :h] - (capacity - 1)
            packed_ids = packed["input_ids"]
            positions = torch.arange(packed_ids.shape[1], device=packed_ids.device)
            positions = positions[None].expand(batch_size, -1).masked_fill(
                packed_ids != int(owner.vlm.config.image_token_id), packed_ids.shape[1],
            ).topk(49, dim=1, largest=False, sorted=True).values
            packed["history_value"] = {
                "history": frames[:, 0, :h],
                "history_mask": history_mask,
                "offsets": torch.cat((offsets, offsets.new_zeros(batch_size, 1)), dim=1),
                "head_positions": positions,
            }
            return packed
        ordered_visual, ordered_mask = owner._reorder_visual_features(
            native_visual, visual_mask
        )
        # frames_per_view flows through the same reordering as visual_mask so
        # emit_views can key it by the reordered view index.
        if frames_per_view is None:
            ordered_frames_per_view = ordered_mask.to(dtype=torch.long)
        else:
            if tuple(frames_per_view.shape) != (batch_size, num_views):
                raise ValueError(
                    "frames_per_view must have shape [B, V], got "
                    f"{tuple(frames_per_view.shape)} vs [{batch_size}, {num_views}]"
                )
            reordered_frames, _ = owner._reorder_visual_features(
                frames_per_view.unsqueeze(-1).unsqueeze(-1).to(
                    dtype=native_visual.dtype
                ),
                visual_mask,
            )
            # Undo the shape padding used to reuse the reorder helper.
            ordered_frames_per_view = (
                reordered_frames.squeeze(-1).squeeze(-1).to(dtype=torch.long)
            )
        text_mask = (
            torch.ones_like(input_ids, dtype=torch.bool)
            if attention_mask is None
            else attention_mask.to(device=input_ids.device, dtype=torch.bool)
        )
        cfg = owner.vlm.config
        semantics = qwen_policy_semantics(owner.config)
        current_only = self._vlm_history_current_enabled(owner)
        from ..qwen_image_attention import CURRENT_ONLY_IMAGE_BLOCKS, image_attention_mode
        restricted_attention = image_attention_mode(owner.config) == CURRENT_ONLY_IMAGE_BLOCKS
        pad_id = int(getattr(cfg, "pad_token_id", None) or cfg.text_config.eos_token_id)
        start_id = int(cfg.vision_start_token_id)
        image_id = int(cfg.image_token_id)
        end_id = int(cfg.vision_end_token_id)
        prompt_cfg = dict(
            getattr(owner.config, "vlm_prompt_config", None) or {}
        )
        user_prefix = input_ids.new_tensor(
            prompt_cfg.get("user_prefix_ids", [])
        )
        user_suffix = input_ids.new_tensor(
            prompt_cfg.get("user_suffix_ids", [])
        )
        assistant_prefix = input_ids.new_tensor(
            prompt_cfg.get("assistant_prefix_ids", [])
        )
        assistant_suffix = input_ids.new_tensor(
            prompt_cfg.get("assistant_suffix_ids", [])
        )
        timestamp_token_ids = getattr(
            owner.config, "qwen_timestamp_token_ids", None
        ) or {}
        timestamp_enabled = bool(
            getattr(owner.config, "qwen_native_video_history_enabled", False)
        ) and bool(timestamp_token_ids)

        def emit_timestamp(raw_view_idx: int, relative_slot: int) -> None:
            """Emit camera-qualified relative time before one visual block."""
            if not timestamp_enabled:
                return
            key = f"{int(raw_view_idx)}:{int(relative_slot)}"
            token_ids = timestamp_token_ids.get(key)
            if token_ids is None:
                raise RuntimeError(
                    "Missing Qwen timestamp token ids for "
                    f"camera={raw_view_idx}, relative_slot={relative_slot}."
                )
            emit(
                input_ids.new_tensor(token_ids),
                semantic=True,
                policy_prefix=True,
            )
        rows: list[torch.Tensor] = []
        mm_rows: list[torch.Tensor] = []
        label_rows: list[torch.Tensor] = []
        semantic_rows: list[torch.Tensor] = []
        wrist_rows: list[torch.Tensor] = []
        policy_prefix_rows: list[torch.Tensor] = []
        visual_rows: list[torch.Tensor] = []
        grids: list[list[int]] = []
        if num_visual_tokens % QWEN_POLICY_LANGUAGE_TOKENS_PER_VIEW != 0:
            raise RuntimeError(
                "Qwen language packing expects the visual token dim to be a "
                "multiple of 49 (per-frame merged tokens), got "
                f"{num_visual_tokens}."
            )
        tokens_per_frame = QWEN_POLICY_LANGUAGE_TOKENS_PER_VIEW
        max_frames_per_view = num_visual_tokens // tokens_per_frame
        time_spec = getattr(owner.config, "qwen_history_time_rope", None)
        time_rows = []
        time_slot_host = None
        if time_spec is not None:
            if history_slot_mask is None or history_slot_mask.ndim != 3:
                raise ValueError("Time RoPE requires original [B,V,K] history mask")
            time_slot_host = history_slot_mask.detach().to(device="cpu", dtype=torch.bool)
            if time_slot_host.shape != (batch_size, num_views, max_frames_per_view):
                raise ValueError("Time RoPE history mask capacity differs from encoded frames")
            if max_frames_per_view != time_spec["reference_history_slots"] + 1:
                raise ValueError("Time RoPE reference history span differs from image capacity")
        patch_grid_h, patch_grid_w = QWEN_POLICY_GEOMETRY_GRID
        history_grid_h, history_grid_w = QWEN_POLICY_HISTORY_GRID
        if visual_mask_host is None:
            # Legacy callers may only provide a device mask. The training
            # path passes the CPU history mask explicitly to avoid this sync.
            ordered_mask_host = ordered_mask.detach().cpu()
        else:
            host_mask = torch.as_tensor(visual_mask_host, dtype=torch.bool)
            primary_host = torch.argmax(host_mask.to(torch.int64), dim=1)
            if num_views == 1:
                order_host = primary_host.unsqueeze(1)
            else:
                tail_host = torch.arange(num_views - 1, dtype=torch.long).expand(
                    batch_size, -1
                )
                tail_host = tail_host + (
                    tail_host >= primary_host.unsqueeze(1)
                ).to(dtype=torch.long)
            order_host = torch.cat((primary_host.unsqueeze(1), tail_host), dim=1)
            ordered_mask_host = torch.gather(host_mask, 1, order_host)
        if visual_mask_host is None:
            # Derive raw camera IDs from the raw mask, not the already
            # reordered mask (whose first valid view is always index zero).
            host_mask = visual_mask.detach().cpu()
            primary_host = torch.argmax(host_mask.to(torch.int64), dim=1)
            if num_views == 1:
                order_host = primary_host.unsqueeze(1)
            else:
                tail_host = torch.arange(num_views - 1, dtype=torch.long).expand(
                    batch_size, -1
                )
                tail_host = tail_host + (
                    tail_host >= primary_host.unsqueeze(1)
                ).to(dtype=torch.long)
                order_host = torch.cat((primary_host.unsqueeze(1), tail_host), dim=1)
        ordered_raw_view_ids_host = order_host
        ordered_frames_host = ordered_frames_per_view.detach().cpu()
        for batch_idx in range(batch_size):
            ids: list[torch.Tensor] = []
            mm_types: list[torch.Tensor] = []
            labels: list[torch.Tensor] = []
            semantic_positions: list[torch.Tensor] = []
            wrist_positions: list[torch.Tensor] = []
            policy_prefix_positions: list[torch.Tensor] = []
            row_visual: list[torch.Tensor] = []
            time_positions = []

            def emit(
                segment: torch.Tensor,
                *,
                semantic: bool,
                policy_prefix: bool,
                label: torch.Tensor | None = None,
                mm_types_override: torch.Tensor | None = None,
            ) -> None:
                """Append one segment to every parallel per-row stream.

                All five streams have to stay index-aligned, so they are only
                ever grown here. Keeping this in one place is what makes the
                segment order below a two-line decision instead of an edit in
                seven duplicated blocks.
                """
                if not segment.numel():
                    return
                ids.append(segment)
                if time_spec is not None:
                    time_positions.append(torch.zeros_like(segment, dtype=torch.float32))
                mm_types.append(
                    torch.zeros_like(segment)
                    if mm_types_override is None
                    else mm_types_override
                )
                labels.append(
                    torch.full_like(segment, -100) if label is None else label
                )
                semantic_positions.append(
                    torch.full_like(segment, semantic, dtype=torch.bool)
                )
                if restricted_attention:
                    wrist_positions.append(torch.zeros_like(segment, dtype=torch.bool))
                policy_prefix_positions.append(
                    torch.full_like(segment, policy_prefix, dtype=torch.bool)
                )

            valid_view_indices = torch.nonzero(
                ordered_mask_host[batch_idx],
                as_tuple=False,
            ).flatten().tolist()
            prompt = input_ids[batch_idx][text_mask[batch_idx]]

            def emit_views() -> None:
                def emit_visual_block(
                    visual: torch.Tensor,
                    *,
                    block_t: int,
                    grid_h: int,
                    grid_w: int,
                    current: bool = True,
                    wrist: bool = False,
                    relative_time: float = 0.,
                ) -> None:
                    block_tokens = int(visual.shape[0])
                    if block_tokens != block_t * (
                        (grid_h // QWEN_POLICY_SPATIAL_MERGE_SIZE)
                        * (grid_w // QWEN_POLICY_SPATIAL_MERGE_SIZE)
                    ):
                        raise RuntimeError(
                            "Qwen visual block token/grid mismatch: "
                            f"tokens={block_tokens}, T={block_t}, "
                            f"grid=({grid_h},{grid_w})."
                        )
                    emit(
                        torch.tensor(
                            [start_id]
                            + [image_id] * block_tokens
                            + [end_id],
                            device=input_ids.device,
                            dtype=torch.long,
                        ),
                        # The formal policy keeps language-model outputs at
                        # image placeholders. Vision sentinels carry no patch.
                        semantic=False,
                        policy_prefix=False,
                        mm_types_override=torch.tensor(
                            [0] + [1] * block_tokens + [0],
                            device=input_ids.device,
                            dtype=torch.long,
                        ),
                    )
                    if semantics.visual_positions_semantic and current:
                        semantic_positions[-1][1:-1] = True
                    if time_spec is not None:
                        time_positions[-1][1:-1] = relative_time
                    if restricted_attention and current and wrist:
                        wrist_positions[-1][1:-1] = True
                    row_visual.append(visual)
                    grids.append([block_t, grid_h, grid_w])

                if current_only:
                    # All past images precede every current image, including
                    # wrists, so causal current states can read all history.
                    # Relative compacted slots align each camera at t=0.
                    counts = {
                        v: int(ordered_frames_host[batch_idx, v])
                        for v in valid_view_indices
                    }
                    if any(t < 1 or t > max_frames_per_view for t in counts.values()):
                        raise ValueError("Invalid frames_per_view for vlm_history_current_v1.")
                    original_slots = {}
                    if time_slot_host is not None:
                        for v, count in counts.items():
                            raw = int(ordered_raw_view_ids_host[batch_idx, v])
                            original_slots[v] = time_slot_host[batch_idx, raw].nonzero().flatten().tolist()
                            if len(original_slots[v]) != count or original_slots[v][-1] != max_frames_per_view - 1:
                                raise ValueError("Time RoPE history slots disagree with encoded frame counts/current")
                    for relative_slot in range(1 - max(counts.values(), default=1), 0):
                        for v, count in counts.items():
                            slot = count - 1 + relative_slot
                            if slot >= 0:
                                emit_visual_block(
                                    ordered_visual[
                                        batch_idx, v,
                                        slot * tokens_per_frame:(slot + 1) * tokens_per_frame,
                                    ],
                                    block_t=1,
                                    grid_h=patch_grid_h,
                                    grid_w=patch_grid_w,
                                    current=False,
                                    relative_time=(
                                        (original_slots[v][slot] - (max_frames_per_view - 1))
                                        / time_spec["reference_history_slots"]
                                        if time_spec is not None else 0.),
                                )
                    for v, count in counts.items():
                        slot = count - 1
                        emit_visual_block(
                            ordered_visual[
                                batch_idx, v,
                                slot * tokens_per_frame:(slot + 1) * tokens_per_frame,
                            ],
                            block_t=1,
                            grid_h=patch_grid_h,
                            grid_w=patch_grid_w,
                            wrist=int(ordered_raw_view_ids_host[batch_idx, v]) in (1, 2),
                        )
                    return

                for view_idx in valid_view_indices:
                    view_T = int(ordered_frames_host[batch_idx, view_idx].item())
                    if view_T < 1:
                        view_T = 1
                    if view_T > max_frames_per_view:
                        raise RuntimeError(
                            f"frames_per_view[{batch_idx},{view_idx}]={view_T} "
                            f"exceeds tokens_per_view // 49 = {max_frames_per_view}"
                        )
                    # ordered_visual is [B, V, F_max*49, D]; slice out the
                    # active frame tokens for this view. P3 keeps each past
                    # frame independent, then applies the DM05 2D 4x4 pool
                    # to each frame separately. The current frame remains a
                    # native 49-token image block. No temporal reduction is
                    # performed here.
                    view_visual = ordered_visual[batch_idx, view_idx]
                    raw_view_idx = int(
                        ordered_raw_view_ids_host[batch_idx, view_idx].item()
                    )
                    if view_T > 1:
                        history_visual = view_visual[: (view_T - 1) * tokens_per_frame]
                        history_visual = history_visual.reshape(
                            view_T - 1, tokens_per_frame, native_dim
                        )
                        history_visual = _pool_qwen_history_frame_tokens(
                            history_visual
                        )
                        # Keep one T=1 block per history frame, in temporal
                        # order. This mirrors DM05's independent <unused0>
                        # slots and avoids assigning a T>1 RoPE grid to a
                        # reduced 4x4 spatial grid (which would make the
                        # temporal ids overrun the block's end token).
                        first_slot = max_frames_per_view - view_T
                        for frame_idx, history_frame in enumerate(history_visual):
                            relative_slot = first_slot + frame_idx - (max_frames_per_view - 1)
                            emit_timestamp(raw_view_idx, relative_slot)
                            emit_visual_block(
                                history_frame,
                                block_t=1,
                                grid_h=history_grid_h,
                                grid_w=history_grid_w,
                            )
                        current_visual = view_visual[
                            (view_T - 1) * tokens_per_frame : view_T * tokens_per_frame
                        ]
                        emit_timestamp(raw_view_idx, 0)
                        emit_visual_block(
                            current_visual,
                            block_t=1,
                            grid_h=patch_grid_h,
                            grid_w=patch_grid_w,
                        )
                    else:
                        emit_timestamp(raw_view_idx, 0)
                        emit_visual_block(
                            view_visual[:tokens_per_frame],
                            block_t=1,
                            grid_h=patch_grid_h,
                            grid_w=patch_grid_w,
                        )

            def emit_prompt() -> None:
                emit(prompt, semantic=True, policy_prefix=True)

            emit(user_prefix, semantic=False, policy_prefix=False)
            if semantics.prompt_before_views:
                # The instruction precedes every camera, so causal image
                # positions can attend to it.
                emit_prompt()
                emit_views()
            else:
                emit_views()
                emit_prompt()
            for fixed in (user_suffix, assistant_prefix):
                emit(fixed, semantic=True, policy_prefix=True)
            if target_context_ids is not None:
                emit(
                    target_context_ids[batch_idx],
                    # FAST target context conditions the decoder CE path, but
                    # must never become action-expert policy context.
                    semantic=False,
                    policy_prefix=False,
                )
            if target_ids is not None:
                target = target_ids[batch_idx]
                emit(
                    target,
                    # Action labels and their suffix are decoder targets, not
                    # policy inputs. Keep them out of the semantic capture.
                    semantic=False,
                    policy_prefix=False,
                    label=target,
                )
                emit(
                    assistant_suffix,
                    semantic=False,
                    policy_prefix=False,
                    label=assistant_suffix,
                )
            rows.append(torch.cat(ids))
            if time_spec is not None:
                time_rows.append(torch.cat(time_positions))
            mm_rows.append(torch.cat(mm_types))
            label_rows.append(torch.cat(labels))
            semantic_rows.append(torch.cat(semantic_positions))
            if restricted_attention:
                wrist_rows.append(torch.cat(wrist_positions))
            policy_prefix_rows.append(torch.cat(policy_prefix_positions))
            visual_rows.append(
                torch.cat(row_visual, dim=0)
                if row_visual
                else native_visual.new_zeros((0, native_dim))
            )
        max_len = max(int(row.numel()) for row in rows)
        packed_ids = input_ids.new_full((batch_size, max_len), pad_id)
        packed_mask = input_ids.new_zeros((batch_size, max_len))
        packed_mm = input_ids.new_zeros((batch_size, max_len))
        packed_labels = input_ids.new_full((batch_size, max_len), -100)
        packed_semantic = torch.zeros(
            (batch_size, max_len),
            device=input_ids.device,
            dtype=torch.bool,
        )
        packed_policy_prefix = torch.zeros(
            (batch_size, max_len),
            device=input_ids.device,
            dtype=torch.bool,
        )
        packed_wrists = torch.zeros_like(packed_semantic) if restricted_attention else None
        for batch_idx, row in enumerate(rows):
            length = int(row.numel())
            packed_ids[batch_idx, :length] = row
            packed_mask[batch_idx, :length] = 1
            packed_mm[batch_idx, :length] = mm_rows[batch_idx]
            packed_labels[batch_idx, :length] = label_rows[batch_idx]
            packed_semantic[batch_idx, :length] = semantic_rows[batch_idx]
            if packed_wrists is not None:
                packed_wrists[batch_idx, :length] = wrist_rows[batch_idx]
            packed_policy_prefix[batch_idx, :length] = policy_prefix_rows[
                batch_idx
            ]
        embeddings = owner.vlm.get_input_embeddings()(packed_ids)
        for batch_idx, row_visual in enumerate(visual_rows):
            if row_visual.numel() == 0:
                continue
            slots = packed_ids[batch_idx] == image_id
            embeddings[batch_idx, slots] = row_visual.to(
                device=embeddings.device, dtype=embeddings.dtype
            )
        grid_tensor = torch.tensor(
            grids, device=input_ids.device, dtype=torch.long
        )
        position_ids = owner.vlm.model.compute_3d_position_ids(
            input_ids=packed_ids,
            inputs_embeds=embeddings,
            image_grid_thw=grid_tensor,
            attention_mask=packed_mask,
            mm_token_type_ids=packed_mm,
        )
        result = {
            "input_ids": packed_ids,
            "inputs_embeds": embeddings,
            "attention_mask": packed_mask,
            "mm_token_type_ids": packed_mm,
            "position_ids": position_ids,
            "labels": packed_labels,
            "policy_semantic_mask": packed_semantic,
            "policy_prefix_mask": packed_policy_prefix,
            "ordered_visual": ordered_visual,
            "ordered_mask": ordered_mask,
        }
        if time_spec is not None:
            time_offsets = torch.zeros_like(packed_mask, dtype=torch.float32)
            for b, times in enumerate(time_rows):
                time_offsets[b, :len(times)] = times
            result["history_time_offsets"] = time_offsets
        if restricted_attention:
            result["current_image_mask"] = packed_semantic & packed_ids.eq(image_id)
            result["wrist_image_mask"] = packed_wrists
        return result

    def _build_policy_output(
        self,
        owner,
        *,
        semantic_native_states: list[torch.Tensor] | None = None,
        policy_semantic_mask: torch.Tensor | None = None,
        geometry_native: torch.Tensor,
        visual_mask: torch.Tensor,
        history_memory_native: torch.Tensor | None = None,
        history_memory_mask: torch.Tensor | None = None,
        history_memory_time_offsets: torch.Tensor | None = None,
        history_memory_view_ids: torch.Tensor | None = None,
        prefix_kv_by_layer: list | None = None,
        prefix_kv_mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        if prefix_kv_by_layer is not None:
            # G0.5 prefix-KV expert. It consumes native per-layer K/V and
            # never reads the projected semantic/geometry streams, so the
            # legacy tensor keys are emitted as minimal placeholders purely to
            # keep the expert's argument contract unchanged.
            if prefix_kv_mask is None:
                raise ValueError("prefix_kv_by_layer requires prefix_kv_mask.")
            batch = int(prefix_kv_mask.shape[0])
            dim = int(self.output_hidden_size(owner.config, owner.vlm))
            ref = next(
                (kv[0] for kv in prefix_kv_by_layer if kv is not None), None
            )
            dtype = geometry_native.dtype if ref is None else ref.dtype
            device = prefix_kv_mask.device
            dummy = torch.zeros((batch, 1, dim), device=device, dtype=dtype)
            empty = dummy.new_zeros((batch, 0, dim))
            kv = prefix_kv_by_layer
            if bool(getattr(owner.config, "ki_detach_vlm_to_expert", False)):
                kv = [
                    None if item is None else (item[0].detach(), item[1].detach())
                    for item in kv
                ]
            return {
                "vlm_features": dummy,
                "vlm_feature_mask": torch.ones(
                    (batch, 1), device=device, dtype=torch.bool
                ),
                "aux_visual_inputs": empty,
                "aux_visual_mask": dummy.new_zeros((batch, 0), dtype=torch.bool),
                "dorsal_visual_inputs": empty,
                "prefix_kv_by_layer": kv,
                "prefix_kv_mask": prefix_kv_mask,
            }
        if semantic_native_states is None or policy_semantic_mask is None:
            raise ValueError(
                "Qwen policy output requires semantic states or G0.5 prefix K/V."
            )
        expert_detach = bool(
            getattr(owner.config, "ki_detach_vlm_to_expert", False)
        )
        if expert_detach:
            semantic_native_states = [
                state.detach() for state in semantic_native_states
            ]
            geometry_native = geometry_native.detach()
            if history_memory_native is not None:
                history_memory_native = history_memory_native.detach()
        # Keep the depths apart: [B, depths, L_full, D]. The expert's semantic
        # route decides which cross-attn layer reads which depth.
        semantic_native_full = torch.stack(semantic_native_states, dim=1)

        batch = semantic_native_full.shape[0]
        semantic_counts = policy_semantic_mask.sum(dim=1)
        max_semantic = int(semantic_counts.max().item())
        semantic_native = semantic_native_full.new_zeros(
            (
                batch,
                semantic_native_full.shape[1],
                max_semantic,
                semantic_native_full.shape[-1],
            )
        )
        semantic_mask = torch.zeros(
            (batch, max_semantic),
            device=semantic_native_full.device,
            dtype=torch.bool,
        )
        for batch_idx in range(batch):
            keep = policy_semantic_mask[batch_idx]
            selected = semantic_native_full[batch_idx][:, keep]
            semantic_native[batch_idx, :, : selected.shape[1]] = selected
            semantic_mask[batch_idx, : selected.shape[1]] = True
        semantic = owner.vlm_output_adapter(semantic_native)

        geometry = owner.vlm_geometry_adapter(geometry_native)
        batch_size, num_views, num_tokens, dim = geometry.shape
        if num_tokens != QWEN_POLICY_GEOMETRY_TOKENS_PER_VIEW:
            raise RuntimeError(
                "Qwen Action geometry requires 196 tokens per view, "
                f"got {num_tokens}."
            )
        view_embeddings = owner.vlm_policy_modules["view_embeddings"]
        if num_views > view_embeddings.num_embeddings:
            raise ValueError(
                f"Qwen received {num_views} views but "
                "vlm_max_view_embeddings="
                f"{view_embeddings.num_embeddings}."
            )
        view_ids = torch.arange(num_views, device=geometry.device)
        geometry = geometry + view_embeddings(view_ids).to(
            dtype=geometry.dtype
        ).view(1, num_views, 1, dim)
        geometry, ordered_geometry_mask = owner._reorder_visual_features(
            geometry,
            visual_mask,
        )
        geometry_mask = ordered_geometry_mask.unsqueeze(-1).expand(
            batch_size,
            num_views,
            num_tokens,
        )
        geometry = geometry * geometry_mask.unsqueeze(-1).to(
            dtype=geometry.dtype
        )
        geometry = geometry.reshape(batch_size, num_views * num_tokens, dim)
        geometry_mask = geometry_mask.reshape(batch_size, num_views * num_tokens)
        empty_dorsal = semantic.new_zeros((batch_size, 0, dim))
        output = {
            "vlm_features": semantic,
            "vlm_feature_mask": semantic_mask,
            "aux_visual_inputs": geometry,
            "aux_visual_mask": geometry_mask,
            "dorsal_visual_inputs": empty_dorsal,
        }
        if history_memory_native is not None:
            if any(value is None for value in (
                history_memory_mask,
                history_memory_time_offsets,
                history_memory_view_ids,
            )):
                raise RuntimeError("Qwen history memory metadata is incomplete.")
            output.update(
                {
                    "memory_features": owner.vlm_geometry_adapter(
                        history_memory_native
                    ),
                    "memory_mask": history_memory_mask,
                    "memory_time_offsets": history_memory_time_offsets,
                    "memory_view_ids": history_memory_view_ids,
                }
            )
        return output

    def forward_policy(
        self,
        owner,
        *,
        input_ids,
        pixel_values,
        image_mask,
        attention_mask,
        precomputed_features,
    ):
        expert_detach = bool(
            getattr(owner.config, "ki_detach_vlm_to_expert", False)
        )
        grad_context = torch.no_grad() if expert_detach else nullcontext()
        with grad_context:
            visual_mask_host = None
            if image_mask is not None and not image_mask.is_cuda:
                visual_mask_host = torch.as_tensor(image_mask, dtype=torch.bool)
                if visual_mask_host.dim() == 3:
                    visual_mask_host = visual_mask_host.any(dim=2)
            native_visual, detail_visual, visual_mask, frames_per_view = (
                self._resolve_policy_visual(
                    owner,
                    pixel_values=pixel_values,
                    image_mask=image_mask,
                    precomputed_features=precomputed_features,
                    require_visual=True,
                )
            )
            packed = self._pack_multimodal(
                owner,
                input_ids=input_ids,
                attention_mask=attention_mask,
                native_visual=native_visual,
                visual_mask=visual_mask,
                visual_mask_host=visual_mask_host,
                frames_per_view=frames_per_view,
                history_slot_mask=image_mask,
            )
            if is_vlm_fast_action_bridge_version(
                getattr(owner.config, "architecture_version", "legacy_cross_emb_v5")
            ):
                padded, prefix_mask = self._capture_prefix_kv(owner, packed=packed)
                result = self._build_policy_output(
                    owner, geometry_native=detail_visual, visual_mask=visual_mask,
                    prefix_kv_by_layer=padded, prefix_kv_mask=prefix_mask,
                )
                from ..qwen_image_attention import CURRENT_ONLY_IMAGE_BLOCKS, image_attention_mode
                if image_attention_mode(owner.config) == CURRENT_ONLY_IMAGE_BLOCKS:
                    # Selecting K/V must not renumber the suffix's rotary
                    # positions relative to the original packed LM prefix.
                    result["prefix_kv_position_offset"] = packed["attention_mask"].sum(1)
                return result
            semantic_native_states = (
                self._forward_language_with_selected_states(
                    owner,
                    input_ids=None,
                    inputs_embeds=packed["inputs_embeds"],
                    attention_mask=self._policy_attention_mask(owner, packed),
                    position_ids=packed["position_ids"],
                    history_value=packed.get("history_value"),
                )
            )
            history_memory = (
                self._encode_history_memory(
                    owner,
                    pixel_values=pixel_values,
                    image_mask=image_mask,
                )
                if bool(
                    getattr(owner.config, "qwen_history_memory_enabled", False)
                )
                else None
            )
        result = self._build_policy_output(
            owner,
            semantic_native_states=semantic_native_states,
            policy_semantic_mask=packed["policy_semantic_mask"],
            geometry_native=detail_visual,
            visual_mask=visual_mask,
            history_memory_native=(
                None if history_memory is None else history_memory[0]
            ),
            history_memory_mask=(
                None if history_memory is None else history_memory[1]
            ),
            history_memory_time_offsets=(
                None if history_memory is None else history_memory[2]
            ),
            history_memory_view_ids=(
                None if history_memory is None else history_memory[3]
            ),
        )
        return result

    @staticmethod
    def _policy_attention_mask(owner, packed):
        from ..qwen_image_attention import policy_attention_mask

        return policy_attention_mask(owner, packed)


















    def describe(self, config, model):
        state = super().describe(config, model)
        state["dtype"] = str(next(model.parameters()).dtype).removeprefix("torch.")
        state["attention_implementation"] = str(
            model.config._attn_implementation
        )
        state["qwen_image_attention_mode"] = getattr(config, "qwen_image_attention_mode", "causal")
        state["policy_modules"] = ["view_embeddings"]
        if getattr(config, "qwen_temporal_embedding", "off") == "vlm_history_value_v1":
            state["history_value"] = {
                "layers": "full_attention_only", "new_parameters": 0,
                "interval": config.qwen_history_value_interval,
                "main_sequence": "current_only_unchanged_positions",
                "history_stream": "shared_decoder_per_frame_no_text",
                "temporal_attention": "same_patch_causal_shared_qkv_gqa_mean_probability",
                "frame_code": "relative_rope_on_qk_host_rotary_width",
                "frame_code_theta": config.qwen_history_value_rope_theta,
                "frame_code_units": "history_slots",
            }
        state["policy_output_adapter"] = "per_depth_independent"
        state["action_context_semantic_route"] = getattr(
            config,
            "action_context_semantic_route",
            None,
        )
        state["policy_semantic_tokens"] = {
            "source": "post_vision_text_feature_preserving_pyramid",
            "layers": list(getattr(config, "vlm_policy_semantic_layers", [])),
        }
        state["policy_geometry_tokens"] = "all_views_premerge_patch_tokens"
        state["policy_vision_contract"] = qwen_policy_vision_contract()
        return state
