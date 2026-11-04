from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from flexwam.config import normalize_model_config
from safetensors.torch import load_file

from flexwam.models.backbone.lingbot_video.action_dit import LingBotActionDiT


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
    if tuple(src.shape) == target_shape:
        return src
    out = src.to(torch.float32)
    while out.ndim < len(target_shape):
        out = out.unsqueeze(0)
    while out.ndim > len(target_shape):
        if out.shape[0] != 1:
            raise ValueError(f"Cannot reduce tensor rank: src={tuple(src.shape)}, target={target_shape}")
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
    if tuple(out.shape) != target_shape:
        raise ValueError(f"Resize produced wrong shape: src={tuple(src.shape)}, target={target_shape}, got={tuple(out.shape)}")
    return out.to(dtype=src.dtype)


def _load_model_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = normalize_model_config(OmegaConf.load(str(path)))
    if "video_dit_config" not in cfg or "action_dit_config" not in cfg:
        raise ValueError(f"`{path}` must contain `video_dit_config` and `action_dit_config`.")
    video_cfg = OmegaConf.to_container(cfg.video_dit_config, resolve=False)
    action_cfg = OmegaConf.to_container(cfg.action_dit_config, resolve=False)
    if not isinstance(video_cfg, dict) or not isinstance(action_cfg, dict):
        raise ValueError("`video_dit_config` and `action_dit_config` must resolve to dicts.")
    if _is_unresolved_interpolation(action_cfg.get("action_dim")):
        print("[WARN] `action_dit_config.action_dim` is unresolved; defaulting to 14 for preprocessing.")
        action_cfg["action_dim"] = 14
    for key in ["hidden_size", "intermediate_size", "depth", "num_attention_heads", "freq_dim", "axes_dims", "axes_lens", "rope_theta", "norm_eps", "qkv_bias", "out_bias", "timestep_mlp_bias"]:
        if key in action_cfg:
            action_cfg[key] = _resolve_from_video_cfg(action_cfg[key], video_cfg)
    return video_cfg, action_cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess LingBot video transformer weights for LingBotActionDiT.")
    parser.add_argument("--model-config", required=True, help="Path to configs/model/lingbot_video.yaml")
    parser.add_argument("--lingbot-model-path", default="lingbot-video", help="Local LingBot pipeline directory.")
    parser.add_argument("--output", required=True, help="Output .pt path.")
    parser.add_argument("--apply-alpha-scaling", action="store_true", help="Scale resized tensors by sqrt(old_dim/new_dim).")
    args = parser.parse_args()

    model_config_path = Path(args.model_config)
    lingbot_model_path = Path(args.lingbot_model_path)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    video_cfg, action_cfg = _load_model_config(model_config_path)
    action_expert = LingBotActionDiT(**action_cfg)
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
        if tuple(src.shape) == tuple(target.shape):
            value = src.detach().cpu()
            copied += 1
        else:
            value = _resize_tensor_to_shape(src.detach().cpu(), tuple(target.shape))
            if args.apply_alpha_scaling and src.ndim >= 2 and src.shape[-1] != target.shape[-1]:
                value = value.to(torch.float32) * (float(src.shape[-1]) / float(target.shape[-1])) ** 0.5
            interpolated += 1
        backbone_state_dict[key] = value.to(dtype=target.dtype).contiguous()

    payload = {
        "policy": {
            "source": str(transformer_path),
            "skip_prefixes": list(LingBotActionDiT.ACTION_BACKBONE_SKIP_PREFIXES),
            "alpha_scaling": bool(args.apply_alpha_scaling),
            "interpolation": "sequential_1d_linear_align_corners_true",
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
    print(
        f"[INFO] Saved LingBotActionDiT backbone to {output_path} "
        f"(copied={copied}, interpolated={interpolated}, skipped={len(action_state) - len(backbone_keys)})."
    )


if __name__ == "__main__":
    main()
