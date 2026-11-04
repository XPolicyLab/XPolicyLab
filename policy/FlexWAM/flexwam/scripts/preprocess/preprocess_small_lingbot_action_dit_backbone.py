#!/usr/bin/env python3
"""Interpolate a smaller LingBotActionDiT backbone from LingBot video weights.

Video LingBot is hidden=2048 / ffn=6144. This script shrinks the action expert
to hidden=1024 / ffn=3072 (same 3x MLP ratio), while keeping 16 heads x 128
dim so MoT mixed attention still matches the video expert — the same pattern
as Wan ActionDiT (hidden 3072→1024, attention space unchanged).

Default dtype is bfloat16. Alpha scaling is on, matching the Wan preprocess
script: when the last dim is resized, multiply by sqrt(old/new).

Example:
  python scripts/preprocess/preprocess_small_lingbot_action_dit_backbone.py \\
      --model-config configs/model/lingbot_video.yaml \\
      --lingbot-model-path lingbot-video \\
      --output lingbot-video/LingBotActionDiT_backbone_1024hdim.pt
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from flexwam.config import normalize_model_config
from safetensors.torch import load_file

from flexwam.models.backbone.lingbot_video.action_dit import LingBotActionDiT


def _parse_dtype(name: str) -> torch.dtype:
    value = str(name).strip().lower()
    if value == "float32":
        return torch.float32
    if value == "float16":
        return torch.float16
    if value == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}. Expected one of: float32, float16, bfloat16.")


def _parse_bool(name: str) -> bool:
    value = str(name).strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse bool value: {name}")


def _is_unresolved_interpolation(value: Any) -> bool:
    return isinstance(value, str) and "${" in value and "}" in value


def _resolve_from_video_cfg(value: Any, video_cfg: dict[str, Any]) -> Any:
    if not _is_unresolved_interpolation(value):
        return value
    text = str(value).strip()
    if not (text.startswith("${") and text.endswith("}")):
        return value
    expr = text[2:-1]
    if not expr.startswith("video_dit_config."):
        return value
    key = expr.split(".", 1)[1]
    if key not in video_cfg:
        return value
    resolved = video_cfg[key]
    return value if _is_unresolved_interpolation(resolved) else resolved


def _interpolate_last_dim(tensor: torch.Tensor, new_size: int) -> torch.Tensor:
    if tensor.shape[-1] == new_size:
        return tensor
    flat = tensor.reshape(-1, 1, tensor.shape[-1]).to(torch.float32)
    flat = F.interpolate(flat, size=new_size, mode="linear", align_corners=True)
    return flat.reshape(*tensor.shape[:-1], new_size)


def _resize_tensor_to_shape(src: torch.Tensor, target_shape: tuple[int, ...]) -> torch.Tensor:
    if tuple(src.shape) == tuple(target_shape):
        return src

    out = src.to(torch.float32)
    while out.ndim < len(target_shape):
        out = out.unsqueeze(0)
    while out.ndim > len(target_shape):
        if out.shape[0] != 1:
            raise ValueError(
                f"Cannot reduce tensor rank for resize: src shape={tuple(src.shape)}, target={target_shape}"
            )
        out = out.squeeze(0)

    for dim, new_size in enumerate(target_shape):
        if out.shape[dim] == new_size:
            continue
        perm = [i for i in range(out.ndim) if i != dim] + [dim]
        inv_perm = [0] * out.ndim
        for i, p in enumerate(perm):
            inv_perm[p] = i
        out_perm = out.permute(*perm).contiguous()
        prefix_shape = out_perm.shape[:-1]
        out_perm = _interpolate_last_dim(out_perm, new_size)
        out = out_perm.reshape(*prefix_shape, new_size).permute(*inv_perm).contiguous()

    if tuple(out.shape) != tuple(target_shape):
        raise ValueError(
            f"Resize produced wrong shape for tensor. src={tuple(src.shape)}, "
            f"target={target_shape}, got={tuple(out.shape)}"
        )
    return out.to(dtype=src.dtype)


def _load_model_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = normalize_model_config(OmegaConf.load(str(path)))
    if "video_dit_config" not in cfg or "action_dit_config" not in cfg:
        raise ValueError(f"`{path}` must contain both `video_dit_config` and `action_dit_config`.")
    video_cfg = OmegaConf.to_container(cfg.video_dit_config, resolve=False)
    action_cfg = OmegaConf.to_container(cfg.action_dit_config, resolve=False)
    if not isinstance(video_cfg, dict) or not isinstance(action_cfg, dict):
        raise ValueError("`video_dit_config` and `action_dit_config` must resolve to dicts.")
    if _is_unresolved_interpolation(action_cfg.get("action_dim")):
        print("[WARN] `action_dit_config.action_dim` is unresolved; defaulting to 14 for preprocessing.")
        action_cfg["action_dim"] = 14
    for key in [
        "hidden_size",
        "intermediate_size",
        "depth",
        "num_attention_heads",
        "attention_head_dim",
        "freq_dim",
        "axes_dims",
        "axes_lens",
        "rope_theta",
        "norm_eps",
        "qkv_bias",
        "out_bias",
        "timestep_mlp_bias",
    ]:
        if key in action_cfg:
            action_cfg[key] = _resolve_from_video_cfg(action_cfg[key], video_cfg)
    return video_cfg, action_cfg


def _action_init_kwargs(action_cfg: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        name
        for name, param in inspect.signature(LingBotActionDiT.__init__).parameters.items()
        if name != "self" and param.kind != inspect.Parameter.VAR_KEYWORD
    }
    return {key: value for key, value in action_cfg.items() if key in allowed}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess a smaller LingBotActionDiT backbone from LingBot video "
            "transformer weights (Wan-style 1D linear interp + alpha scaling, bf16)."
        )
    )
    parser.add_argument("--model-config", required=True, help="Path to configs/model/lingbot_video.yaml")
    parser.add_argument("--lingbot-model-path", default="lingbot-video", help="Local LingBot pipeline directory.")
    parser.add_argument("--output", required=True, help="Output .pt path for the small ActionDiT backbone.")
    parser.add_argument("--device", default="cpu", help="Device for building the target ActionDiT.")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=1024,
        help="Target action hidden size. Default: 1024 (half of LingBot video 2048).",
    )
    parser.add_argument(
        "--intermediate-size",
        type=int,
        default=3072,
        help="Target action FFN size. Default: 3072 (keep 3x hidden, half of video 6144).",
    )
    parser.add_argument(
        "--apply-alpha-scaling",
        default="true",
        help="Whether to apply alpha=sqrt(old/new) when the last dimension is resized (true/false). Default: true.",
    )
    args = parser.parse_args()

    model_config_path = Path(args.model_config)
    lingbot_model_path = Path(args.lingbot_model_path)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    apply_alpha_scaling = _parse_bool(args.apply_alpha_scaling)
    torch_dtype = _parse_dtype(args.dtype)

    video_cfg, action_cfg = _load_model_config(model_config_path)
    video_hidden = int(video_cfg.get("hidden_size", 2048))
    video_heads = int(video_cfg.get("num_attention_heads", action_cfg.get("num_attention_heads", 16)))
    video_head_dim = video_hidden // video_heads

    action_cfg["hidden_size"] = int(args.hidden_size)
    action_cfg["intermediate_size"] = int(args.intermediate_size)
    action_cfg["num_attention_heads"] = video_heads
    action_cfg["attention_head_dim"] = video_head_dim
    action_cfg["use_gradient_checkpointing"] = False

    print(
        f"[INFO] Loaded model config from {model_config_path}. "
        f"Small LingBotActionDiT: hidden={action_cfg['hidden_size']}, "
        f"ffn={action_cfg['intermediate_size']}, heads={video_heads}, "
        f"head_dim={video_head_dim} (video hidden={video_hidden}), "
        f"dtype={torch_dtype}, apply_alpha_scaling={apply_alpha_scaling}."
    )

    action_expert = LingBotActionDiT(**_action_init_kwargs(action_cfg)).to(device=args.device, dtype=torch_dtype)
    if int(action_expert.num_heads) != video_heads:
        raise ValueError("Action `num_heads` must match video expert for MoT mixed attention.")
    if int(action_expert.attn_head_dim) != video_head_dim:
        raise ValueError("Action `attn_head_dim` must match video expert for MoT mixed attention.")

    action_state = action_expert.state_dict()
    backbone_keys = LingBotActionDiT.backbone_key_set(action_state.keys())

    transformer_path = lingbot_model_path / "transformer" / "diffusion_pytorch_model.safetensors"
    if not transformer_path.is_file():
        raise FileNotFoundError(f"Missing LingBot transformer weights: {transformer_path}")
    video_state = load_file(str(transformer_path), device="cpu")

    backbone_state_dict: dict[str, torch.Tensor] = {}
    copied = 0
    interpolated = 0
    for key in sorted(backbone_keys):
        if key not in video_state:
            raise ValueError(f"Key `{key}` not found in LingBot video transformer state dict.")
        src = video_state[key]
        target = action_state[key]
        src_cpu = src.detach().to(device="cpu")
        if tuple(src.shape) == tuple(target.shape):
            value = src_cpu
            copied += 1
        else:
            value = _resize_tensor_to_shape(src_cpu, tuple(target.shape))
            if apply_alpha_scaling and src.ndim >= 2 and src.shape[-1] != target.shape[-1]:
                alpha = (float(src.shape[-1]) / float(target.shape[-1])) ** 0.5
                value = value.to(torch.float32) * alpha
            interpolated += 1
        backbone_state_dict[key] = value.detach().to(dtype=target.dtype, device="cpu").contiguous()

    payload = {
        "policy": {
            "source": str(transformer_path),
            "skip_prefixes": list(LingBotActionDiT.ACTION_BACKBONE_SKIP_PREFIXES),
            "alpha_scaling": bool(apply_alpha_scaling),
            "interpolation": "sequential_1d_linear_align_corners_true",
            "hidden_size": int(action_cfg["hidden_size"]),
            "intermediate_size": int(action_cfg["intermediate_size"]),
            "dtype": str(args.dtype),
        },
        "backbone_state_dict": backbone_state_dict,
        "meta": {
            "hidden_size": int(action_expert.hidden_size),
            "intermediate_size": int(action_expert.intermediate_size),
            "depth": int(len(action_expert.blocks)),
            "num_attention_heads": int(action_expert.num_attention_heads),
            "attention_head_dim": int(action_expert.attention_head_dim),
            "freq_dim": int(action_expert.freq_dim),
            "norm_eps": float(action_expert.norm_eps),
        },
    }
    torch.save(payload, str(output_path))
    skipped = len(action_state) - len(backbone_keys)
    print(
        "[INFO] Saved small LingBotActionDiT backbone to "
        f"{output_path} (copied={copied}, interpolated={interpolated}, skipped={skipped})."
    )


if __name__ == "__main__":
    main()
