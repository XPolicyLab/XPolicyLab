#!/usr/bin/env python

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from lerobot.policies.fastwam.configuration_fastwam import FastWAMConfig
from lerobot.policies.fastwam.core.models.wan22.action_dit import ActionDiT
from lerobot.policies.fastwam.core.models.wan22.helpers.loader import load_wan22_ti2v_5b_components


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
        current_size = out.shape[dim]
        if current_size == new_size:
            continue
        perm = [i for i in range(out.ndim) if i != dim] + [dim]
        inv_perm = [0] * out.ndim
        for i, p in enumerate(perm):
            inv_perm[p] = i
        out_perm = out.permute(*perm).contiguous()
        prefix_shape = out_perm.shape[:-1]
        out_perm = _interpolate_last_dim(out_perm, new_size)
        out_perm = out_perm.reshape(*prefix_shape, new_size)
        out = out_perm.permute(*inv_perm).contiguous()

    if tuple(out.shape) != tuple(target_shape):
        raise ValueError(
            f"Resize produced wrong shape for tensor. src={tuple(src.shape)}, target={target_shape}, got={tuple(out.shape)}"
        )
    return out.to(dtype=src.dtype)


def _load_config(args: argparse.Namespace) -> FastWAMConfig:
    if args.policy_config_path:
        cfg_kwargs = asdict(FastWAMConfig.from_pretrained(args.policy_config_path))
    else:
        cfg_kwargs = {}

    if args.model_id is not None:
        cfg_kwargs["model_id"] = args.model_id
    if args.tokenizer_model_id is not None:
        cfg_kwargs["tokenizer_model_id"] = args.tokenizer_model_id
    if args.action_dim is not None:
        cfg_kwargs["action_dim"] = int(args.action_dim)
        cfg_kwargs["video_dit_config"] = None
        cfg_kwargs["action_dit_config"] = None
    if args.redirect_common_files is not None:
        cfg_kwargs["redirect_common_files"] = _parse_bool(args.redirect_common_files)
    if args.device is not None:
        cfg_kwargs["device"] = args.device
    if args.dtype is not None:
        cfg_kwargs["dtype"] = args.dtype

    return FastWAMConfig(**cfg_kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate FastWAM ActionDiT backbone weights from the Wan2.2 video DiT."
    )
    parser.add_argument(
        "--policy-config-path",
        default=None,
        help="Optional local FastWAM config directory/file saved by `save_pretrained`. CLI overrides still apply.",
    )
    parser.add_argument("--output", required=True, help="Output .pt path for the preprocessed ActionDiT backbone.")
    parser.add_argument("--device", default="cpu", help="Device for loading and preprocessing.")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--model-id", default=None, help="Override Wan video model id.")
    parser.add_argument("--tokenizer-model-id", default=None, help="Override Wan tokenizer model id.")
    parser.add_argument("--action-dim", type=int, default=None, help="Override action dimension before building configs.")
    parser.add_argument(
        "--redirect-common-files",
        default=None,
        help="Optional bool override for FastWAM redirect_common_files.",
    )
    parser.add_argument(
        "--apply-alpha-scaling",
        default="true",
        help="Whether to apply alpha=sqrt(dv/da) when resizing the last dimension.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    apply_alpha_scaling = _parse_bool(args.apply_alpha_scaling)
    torch_dtype = _parse_dtype(args.dtype)
    config = _load_config(args)

    print(
        "[INFO] Generating ActionDiT backbone from Wan2.2 with "
        f"model_id={config.model_id}, dtype={torch_dtype}, device={args.device}, "
        f"apply_alpha_scaling={apply_alpha_scaling}."
    )

    components = load_wan22_ti2v_5b_components(
        device=args.device,
        torch_dtype=torch_dtype,
        model_id=config.model_id,
        tokenizer_model_id=config.tokenizer_model_id,
        redirect_common_files=bool(config.redirect_common_files),
        dit_config=dict(config.video_dit_config),
        skip_dit_load_from_pretrain=False,
        load_text_encoder=False,
    )
    video_expert = components.dit

    action_cfg: dict[str, Any] = dict(config.action_dit_config)
    action_expert = ActionDiT(**action_cfg).to(device=args.device, dtype=torch_dtype)
    if int(action_cfg["num_heads"]) != int(video_expert.num_heads):
        raise ValueError("ActionDiT `num_heads` must match video expert for MoT mixed attention.")
    if int(action_cfg["attn_head_dim"]) != int(video_expert.attn_head_dim):
        raise ValueError("ActionDiT `attn_head_dim` must match video expert for MoT mixed attention.")
    if int(action_cfg["num_layers"]) != int(len(video_expert.blocks)):
        raise ValueError("ActionDiT `num_layers` must match video expert.")

    action_state = action_expert.state_dict()
    video_state = video_expert.state_dict()
    backbone_keys = ActionDiT.backbone_key_set(action_state.keys())

    backbone_state_dict: dict[str, torch.Tensor] = {}
    copied = 0
    interpolated = 0
    for key in sorted(backbone_keys):
        if key not in video_state:
            raise ValueError(f"Key `{key}` not found in video expert state dict.")
        src = video_state[key]
        target = action_state[key]
        if tuple(src.shape) == tuple(target.shape):
            value = src
            copied += 1
        else:
            value = _resize_tensor_to_shape(src, tuple(target.shape))
            if apply_alpha_scaling and src.ndim >= 2 and src.shape[-1] != target.shape[-1]:
                alpha = (float(src.shape[-1]) / float(target.shape[-1])) ** 0.5
                value = value.to(torch.float32) * alpha
            interpolated += 1
        backbone_state_dict[key] = value.detach().to(dtype=target.dtype, device="cpu").contiguous()

    payload = {
        "policy": {
            "skip_prefixes": list(ActionDiT.ACTION_BACKBONE_SKIP_PREFIXES),
            "alpha_scaling": bool(apply_alpha_scaling),
            "interpolation": "sequential_1d_linear_align_corners_true",
        },
        "backbone_state_dict": backbone_state_dict,
        "meta": {
            "hidden_dim": int(action_cfg["hidden_dim"]),
            "ffn_dim": int(action_cfg["ffn_dim"]),
            "num_layers": int(action_cfg["num_layers"]),
            "num_heads": int(action_cfg["num_heads"]),
            "attn_head_dim": int(action_cfg["attn_head_dim"]),
            "text_dim": int(action_cfg["text_dim"]),
            "freq_dim": int(action_cfg["freq_dim"]),
            "eps": float(action_cfg["eps"]),
        },
    }
    torch.save(payload, str(output_path))

    skipped = len(action_state) - len(backbone_keys)
    print(
        "[INFO] Saved ActionDiT backbone payload to "
        f"{output_path} (copied={copied}, interpolated={interpolated}, skipped={skipped})."
    )


if __name__ == "__main__":
    main()
