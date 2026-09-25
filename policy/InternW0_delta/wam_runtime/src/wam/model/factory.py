"""Construction of WAM and its pretrained backbones."""

from typing import Any, Optional

import torch
from omegaconf import DictConfig, OmegaConf

from .backbones.wan22.helpers.loader import load_wan22_ti2v_5b_components
from .modules.experts.action_dit import ActionDiT
from .modules.mot.mixture_of_transformers import MoT
from .world_action_model import WAM


def _resolve_action_dit_init_path(
    pretrained_path: str | None,
    *,
    use_interpolated_init: bool,
    skip_dit_load_from_pretrain: bool,
) -> str | None:
    if skip_dit_load_from_pretrain or not use_interpolated_init:
        return None
    return pretrained_path


def _as_dict(value: Any, *, name: str, required: bool = False) -> dict[str, Any]:
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if value is None:
        if required:
            raise ValueError(f"`{name}` is required for WAM.")
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"`{name}` must resolve to a dict, got {type(value)}")
    return value


def build_wam_from_wan22(
    *,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
    tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: Optional[int] = None,
    redirect_common_files: bool = True,
    video_dit_config: dict[str, Any] | None = None,
    action_dit_config: dict[str, Any] | None = None,
    action_dit_pretrained_path: str | None = None,
    action_dit_use_interpolated_init: bool = True,
    skip_dit_load_from_pretrain: bool = False,
    defer_vae_load: bool = False,
    mot_checkpoint_mixed_attn: bool = True,
    mot_checkpoint_layer_stride: int = 1,
    mot_checkpoint_extra_layers: Optional[list[int]] = None,
    mot_checkpoint_preserve_rng_state: bool = True,
    mot_attention_backend: str = "flex",
    mot_flex_block_size: int = 64,
    mot_compile_mode: str = "off",
    mot_compile_gradient_checkpointing: bool = False,
    mot_compile_action_context_pad_to: int = 640,
    video_train_shift: float = 5.0,
    video_infer_shift: float = 5.0,
    video_num_train_timesteps: int = 1000,
    action_train_shift: float = 5.0,
    action_infer_shift: float = 5.0,
    action_num_train_timesteps: int = 1000,
    loss_lambda_video: float = 1.0,
    loss_lambda_action: float = 1.0,
    understanding: Optional[dict[str, Any]] = None,
    memory: Optional[dict[str, Any]] = None,
    future_delta: Optional[dict[str, Any]] = None,
    proprio_encoding: Optional[dict[str, Any]] = None,
):
    if video_dit_config is None:
        raise ValueError("`video_dit_config` is required by build_wam_from_wan22().")
    if "text_dim" not in video_dit_config:
        raise ValueError("`video_dit_config['text_dim']` is required for WAM.")
    components = load_wan22_ti2v_5b_components(
        device=device,
        torch_dtype=torch_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=tokenizer_max_len,
        redirect_common_files=redirect_common_files,
        dit_config=video_dit_config,
        skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
        load_text_encoder=load_text_encoder,
        defer_vae_load=defer_vae_load,
    )

    video_expert = components.dit
    resolved_action_dit_path = _resolve_action_dit_init_path(
        action_dit_pretrained_path,
        use_interpolated_init=bool(action_dit_use_interpolated_init),
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
    )
    action_expert = ActionDiT.from_pretrained(
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=resolved_action_dit_path,
        skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
        device=device,
        torch_dtype=torch_dtype,
    )
    if int(action_expert.num_heads) != int(video_expert.num_heads):
        raise ValueError(
            "ActionDiT `num_heads` must match video expert for MoT mixed attention."
        )
    if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
        raise ValueError(
            "ActionDiT `attn_head_dim` must match video expert for MoT mixed attention."
        )
    if int(len(action_expert.blocks)) != int(len(video_expert.blocks)):
        raise ValueError("ActionDiT `num_layers` must match video expert.")

    mot = MoT(
        mixtures={"video": video_expert, "action": action_expert},
        mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        checkpoint_layer_stride=mot_checkpoint_layer_stride,
        checkpoint_extra_layers=mot_checkpoint_extra_layers,
        checkpoint_preserve_rng_state=mot_checkpoint_preserve_rng_state,
        attention_backend=mot_attention_backend,
        flex_block_size=mot_flex_block_size,
        compile_mode=mot_compile_mode,
        compile_gradient_checkpointing=mot_compile_gradient_checkpointing,
        compile_action_context_pad_to=mot_compile_action_context_pad_to,
    )

    model = WAM(
        video_expert=video_expert,
        action_expert=action_expert,
        mot=mot,
        vae=components.vae,
        text_encoder=components.text_encoder,
        tokenizer=components.tokenizer,
        text_dim=int(video_dit_config["text_dim"]),
        proprio_dim=proprio_dim,
        device=device,
        torch_dtype=torch_dtype,
        video_train_shift=video_train_shift,
        video_infer_shift=video_infer_shift,
        video_num_train_timesteps=video_num_train_timesteps,
        action_train_shift=action_train_shift,
        action_infer_shift=action_infer_shift,
        action_num_train_timesteps=action_num_train_timesteps,
        loss_lambda_video=loss_lambda_video,
        loss_lambda_action=loss_lambda_action,
        understanding=understanding,
        memory=memory,
        future_delta=future_delta,
        proprio_encoding=proprio_encoding,
    )
    model.model_paths = {
        "video_dit": components.dit_path,
        "vae": components.vae_path,
        "text_encoder": components.text_encoder_path,
        "tokenizer": components.tokenizer_path,
        "action_dit_backbone": (
            "SKIPPED_PRETRAIN"
            if skip_dit_load_from_pretrain
            else (
                resolved_action_dit_path
                if resolved_action_dit_path is not None
                else "RANDOM_INIT"
            )
        ),
    }
    return model


def create_wam(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    action_dit_use_interpolated_init: bool = True,
    skip_dit_load_from_pretrain: bool = False,
    defer_vae_load: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    understanding=None,
    memory=None,
    future_delta=None,
    proprio_encoding=None,
    mot_checkpoint_mixed_attn: bool = True,
    mot_checkpoint_layer_stride: int = 1,
    mot_checkpoint_extra_layers=None,
    mot_checkpoint_preserve_rng_state: bool = True,
    mot_attention_backend: str = "flex",
    mot_flex_block_size: int = 64,
    mot_compile_mode: str = "off",
    mot_compile_gradient_checkpointing: bool = False,
    mot_compile_action_context_pad_to: int = 640,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    video_dit_config = _as_dict(
        video_dit_config, name="video_dit_config", required=True
    )
    action_dit_config = _as_dict(action_dit_config, name="action_dit_config")
    video_scheduler = _as_dict(video_scheduler, name="video_scheduler")
    action_scheduler = _as_dict(
        action_scheduler, name="action_scheduler", required=True
    )
    loss = _as_dict(loss, name="loss")
    understanding = _as_dict(understanding, name="understanding")
    memory = _as_dict(memory, name="memory")
    future_delta = _as_dict(future_delta, name="future_delta")
    proprio_encoding = _as_dict(proprio_encoding, name="proprio_encoding")

    required_scheduler_keys = {"train_shift", "infer_shift", "num_train_timesteps"}
    missing_keys = required_scheduler_keys - set(action_scheduler)
    if missing_keys:
        raise ValueError(
            f"`action_scheduler` missing required keys: {sorted(missing_keys)}. "
            "Expected keys: train_shift, infer_shift, num_train_timesteps."
        )

    return build_wam_from_wan22(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=None if proprio_dim is None else int(proprio_dim),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        action_dit_use_interpolated_init=bool(
            action_dit_use_interpolated_init
        ),
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        defer_vae_load=bool(defer_vae_load),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        mot_checkpoint_layer_stride=int(mot_checkpoint_layer_stride),
        mot_checkpoint_extra_layers=[
            int(layer_idx) for layer_idx in (mot_checkpoint_extra_layers or ())
        ],
        mot_checkpoint_preserve_rng_state=bool(
            mot_checkpoint_preserve_rng_state
        ),
        mot_attention_backend=str(mot_attention_backend),
        mot_flex_block_size=int(mot_flex_block_size),
        mot_compile_mode=str(mot_compile_mode),
        mot_compile_gradient_checkpointing=bool(
            mot_compile_gradient_checkpointing
        ),
        mot_compile_action_context_pad_to=int(mot_compile_action_context_pad_to),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
        understanding=understanding,
        memory=memory,
        future_delta=future_delta,
        proprio_encoding=proprio_encoding,
    )
