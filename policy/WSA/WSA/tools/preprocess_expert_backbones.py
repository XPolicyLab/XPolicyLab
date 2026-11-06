#!/usr/bin/env python

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch
import torch.nn.functional as F

from lerobot.policies.WSA_Large.configuration_wsa_large import WSALargeConfig as WSALargeConfig
from lerobot.policies.WSA_Large.core.models.wan22.action_dit import ActionDiT
from lerobot.policies.WSA_Large.core.models.wan22.future_3d_expert import Future3DExpert
from lerobot.policies.WSA_Large.core.models.wan22.helpers.loader import load_wan22_ti2v_5b_components


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


def _load_config(args: argparse.Namespace) -> WSALargeConfig:
    if args.policy_config_path:
        cfg_kwargs = asdict(WSALargeConfig.from_pretrained(args.policy_config_path))
    else:
        cfg_kwargs = {}

    for arg_name, cfg_name in (
        ("model_id", "model_id"),
        ("tokenizer_model_id", "tokenizer_model_id"),
        ("dtype", "dtype"),
        ("device", "device"),
        ("da3_variant", "da3_variant"),
        ("da3_model_path_or_name", "da3_model_path_or_name"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            cfg_kwargs[cfg_name] = value

    if args.redirect_common_files is not None:
        cfg_kwargs["redirect_common_files"] = _parse_bool(args.redirect_common_files)
    if args.action_dim is not None:
        cfg_kwargs["action_dim"] = int(args.action_dim)
        cfg_kwargs["video_dit_config"] = None
        cfg_kwargs["action_dit_config"] = None
    if args.da3_num_views is not None:
        cfg_kwargs["da3_num_views"] = int(args.da3_num_views)
        cfg_kwargs["future_3d_config"] = None
    if args.da3_tokens_per_view is not None:
        cfg_kwargs["da3_tokens_per_view"] = int(args.da3_tokens_per_view)
        cfg_kwargs["future_3d_config"] = None
    if args.da3_query_dim is not None:
        cfg_kwargs["da3_query_dim"] = int(args.da3_query_dim)
        cfg_kwargs["future_3d_config"] = None
    if args.future_3d_tokens_per_view is not None:
        cfg_kwargs["future_3d_tokens_per_view"] = int(args.future_3d_tokens_per_view)
        cfg_kwargs["future_3d_config"] = None

    return WSALargeConfig(**cfg_kwargs)


def _validate_mot_compatibility(expert_name: str, expert: Any, video_expert: Any) -> None:
    if int(expert.num_heads) != int(video_expert.num_heads):
        raise ValueError(f"{expert_name} `num_heads` must match video expert for MoT mixed attention.")
    if int(expert.attn_head_dim) != int(video_expert.attn_head_dim):
        raise ValueError(f"{expert_name} `attn_head_dim` must match video expert for MoT mixed attention.")
    if int(len(expert.blocks)) != int(len(video_expert.blocks)):
        raise ValueError(f"{expert_name} `num_layers` must match video expert.")


def _build_backbone_payload(
    *,
    expert_name: str,
    target_state: dict[str, torch.Tensor],
    video_state: dict[str, torch.Tensor],
    backbone_keys: set[str],
    skip_prefixes: tuple[str, ...],
    meta: dict[str, Any],
    apply_alpha_scaling: bool,
) -> dict[str, Any]:
    backbone_state_dict: dict[str, torch.Tensor] = {}
    copied = 0
    interpolated = 0

    for key in sorted(backbone_keys):
        if key not in video_state:
            raise ValueError(f"Key `{key}` not found in video expert state dict while building {expert_name}.")
        src = video_state[key]
        target = target_state[key]
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

    skipped = len(target_state) - len(backbone_keys)
    return {
        "policy": {
            "expert": expert_name,
            "skip_prefixes": list(skip_prefixes),
            "alpha_scaling": bool(apply_alpha_scaling),
            "interpolation": "sequential_1d_linear_align_corners_true",
        },
        "backbone_state_dict": backbone_state_dict,
        "meta": meta,
        "_summary": {
            "copied": copied,
            "interpolated": interpolated,
            "skipped": skipped,
        },
    }


def _save_payload(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = payload.pop("_summary")
    torch.save(payload, str(output_path))
    print(
        f"[INFO] Saved {payload['policy']['expert']} backbone payload to {output_path} "
        f"(copied={summary['copied']}, interpolated={summary['interpolated']}, skipped={summary['skipped']})."
    )


def _resolve_outputs(args: argparse.Namespace) -> tuple[Path | None, Path | None]:
    action_output = Path(args.action_output) if args.action_output else None
    future_3d_output = Path(args.future_3d_output) if args.future_3d_output else None
    generic_output = Path(args.output) if args.output else None

    if args.expert == "action":
        action_output = action_output or generic_output
    elif args.expert == "future_3d":
        future_3d_output = future_3d_output or generic_output
    elif generic_output is not None:
        raise ValueError("Use --action-output and --future-3d-output when --expert=both.")

    if args.expert in {"action", "both"} and action_output is None:
        raise ValueError("--action-output is required when generating the action expert.")
    if args.expert in {"future_3d", "both"} and future_3d_output is None:
        raise ValueError("--future-3d-output is required when generating the future-3D expert.")
    return action_output, future_3d_output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate WSA_Large expert backbone weights from the Wan2.2 video DiT."
    )
    parser.add_argument("--expert", choices=["action", "future_3d", "both"], default="both")
    parser.add_argument("--output", default=None, help="Output .pt path when generating a single expert.")
    parser.add_argument("--action-output", default=None, help="Output .pt path for the ActionDiT backbone.")
    parser.add_argument("--future-3d-output", default=None, help="Output .pt path for the Future3DExpert backbone.")
    parser.add_argument(
        "--policy-config-path",
        default=None,
        help="Optional local WSA_Large config directory/file saved by `save_pretrained`. CLI overrides still apply.",
    )
    parser.add_argument("--device", default="cpu", help="Device for loading and preprocessing.")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--model-id", default=None, help="Override Wan video model id.")
    parser.add_argument("--tokenizer-model-id", default=None, help="Override Wan tokenizer model id.")
    parser.add_argument("--action-dim", type=int, default=None, help="Override action dimension before building configs.")
    parser.add_argument("--da3-num-views", type=int, default=None)
    parser.add_argument("--da3-tokens-per-view", type=int, default=None)
    parser.add_argument("--da3-query-dim", type=int, default=None)
    parser.add_argument("--da3-variant", default=None)
    parser.add_argument("--da3-model-path-or-name", default=None)
    parser.add_argument("--future-3d-tokens-per-view", type=int, default=None)
    parser.add_argument(
        "--redirect-common-files",
        default=None,
        help="Optional bool override for WSA_Large redirect_common_files.",
    )
    parser.add_argument(
        "--apply-alpha-scaling",
        default="true",
        help="Whether to apply alpha=sqrt(dv/da) when resizing the last dimension.",
    )
    args = parser.parse_args()

    action_output, future_3d_output = _resolve_outputs(args)
    apply_alpha_scaling = _parse_bool(args.apply_alpha_scaling)
    torch_dtype = _parse_dtype(args.dtype)
    config = _load_config(args)

    print(
        "[INFO] Generating WSA_Large expert backbones from Wan2.2 with "
        f"model_id={config.model_id}, dtype={torch_dtype}, device={args.device}, "
        f"expert={args.expert}, apply_alpha_scaling={apply_alpha_scaling}."
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
    video_state = video_expert.state_dict()

    if args.expert in {"action", "both"}:
        action_cfg: dict[str, Any] = dict(config.action_dit_config)
        action_expert = ActionDiT(**action_cfg).to(device=args.device, dtype=torch_dtype)
        _validate_mot_compatibility("ActionDiT", action_expert, video_expert)
        action_state = action_expert.state_dict()
        action_payload = _build_backbone_payload(
            expert_name="action",
            target_state=action_state,
            video_state=video_state,
            backbone_keys=ActionDiT.backbone_key_set(action_state.keys()),
            skip_prefixes=ActionDiT.ACTION_BACKBONE_SKIP_PREFIXES,
            meta={
                "hidden_dim": int(action_cfg["hidden_dim"]),
                "ffn_dim": int(action_cfg["ffn_dim"]),
                "num_layers": int(action_cfg["num_layers"]),
                "num_heads": int(action_cfg["num_heads"]),
                "attn_head_dim": int(action_cfg["attn_head_dim"]),
                "text_dim": int(action_cfg["text_dim"]),
                "freq_dim": int(action_cfg["freq_dim"]),
                "eps": float(action_cfg["eps"]),
            },
            apply_alpha_scaling=apply_alpha_scaling,
        )
        _save_payload(action_payload, action_output)

    if args.expert in {"future_3d", "both"}:
        future_cfg: dict[str, Any] = dict(config.future_3d_config)
        future_expert = Future3DExpert(**future_cfg).to(device=args.device, dtype=torch_dtype)
        _validate_mot_compatibility("Future3DExpert", future_expert, video_expert)
        future_state = future_expert.state_dict()
        future_payload = _build_backbone_payload(
            expert_name="future_3d",
            target_state=future_state,
            video_state=video_state,
            backbone_keys=Future3DExpert.backbone_key_set(future_state.keys()),
            skip_prefixes=Future3DExpert.FUTURE_3D_BACKBONE_SKIP_PREFIXES,
            meta={
                "hidden_dim": int(future_cfg["hidden_dim"]),
                "ffn_dim": int(future_cfg["ffn_dim"]),
                "num_layers": int(future_cfg["num_layers"]),
                "num_heads": int(future_cfg["num_heads"]),
                "attn_head_dim": int(future_cfg["attn_head_dim"]),
                "freq_dim": int(future_cfg["freq_dim"]),
                "eps": float(future_cfg["eps"]),
            },
            apply_alpha_scaling=apply_alpha_scaling,
        )
        _save_payload(future_payload, future_3d_output)


if __name__ == "__main__":
    main()
