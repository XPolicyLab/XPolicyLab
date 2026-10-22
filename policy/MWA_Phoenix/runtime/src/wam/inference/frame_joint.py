"""FastWAM-style joint sampling for one frame/action training window."""

from typing import Any, Optional

import torch

from wam.model.modules.codecs import video_latent_codec as video_codec
from wam.model.modules.codecs.utils import normalize_input_image_tensor
from wam.model.modules.conditioning.frame_attention import (
    build_frame_memory_attention_mask,
    build_joint96_attention_mask,
)
from wam.model.modules.conditioning.input_builder import joint96_camera_valid_mask
from wam.model.modules.conditioning.video_rope import sequence_start_video_rope_time_ids
from wam.model.modules.memory.proprio_encoder import append_proprio_to_context

from .online_action_policy import _encode_online_condition_frame
from .qwen_vl_cache import _get_online_vlm_pack


@torch.no_grad()
def infer_frame_joint_window(
    model,
    prompt: Optional[str],
    input_image: torch.Tensor,
    num_frames: int,
    action_horizon: int,
    *,
    action: Optional[torch.Tensor] = None,
    proprio: Optional[torch.Tensor] = None,
    memory_inputs: Optional[dict[str, Any]] = None,
    context: Optional[torch.Tensor] = None,
    context_mask: Optional[torch.Tensor] = None,
    vlm_current_images: Optional[torch.Tensor] = None,
    vlm_current_view_is_pad: Optional[torch.Tensor] = None,
    vlm_view_names: Any = None,
    understanding_prompt: Optional[str] = None,
    num_inference_steps: int = 20,
    video_sigma_shift: Optional[float] = None,
    action_sigma_shift: Optional[float] = None,
    seed: Optional[int] = None,
    rand_device: str = "cpu",
    tiled: bool = False,
    video_layout: Any = None,
    video_view_names: Any = None,
    camera_is_pad: Optional[torch.Tensor] = None,
    action_dim_is_pad: Optional[torch.Tensor] = None,
    proprio_dim_is_pad: Optional[torch.Tensor] = None,
    embodiment_ids: Optional[torch.Tensor] = None,
    decode_video: bool = True,
) -> dict[str, Any]:
    """Jointly denoise the future video latents and one action horizon.

    The real visual sequence matches frame training:
    ``[anchor, recent, current, noisy future]``. Video-side Future Delta
    queries are appended after patchification; they stay causally isolated
    from noisy future/action tokens while actions can consume them.

    Joint96 instead uses ``[current, noisy future]`` and lets action queries
    attend to both. Its camera/action/proprio padding masks must be explicit.
    """
    model.eval()
    input_image = normalize_input_image_tensor(input_image).to(
        device=model.device, dtype=model.torch_dtype
    )
    joint96 = bool(getattr(model, "joint96_three_view", False))
    if joint96:
        if tuple(input_image.shape) != (1, 3, 3, 256, 256):
            raise ValueError("Joint96 inference requires [1,3,3,256,256] current RGB views.")
        if memory_inputs is not None or vlm_current_images is not None:
            raise ValueError("Joint96 inference does not accept memory or VLM inputs.")
        for flag in ("memory_enabled", "memory_video_enabled", "understanding_enabled",
                     "future_delta_enabled", "semantic_future_alignment_enabled"):
            if bool(getattr(model, flag, False)):
                raise ValueError(f"Joint96 inference requires {flag}=false.")
        if model.video_expert.action_conditioned or action is not None:
            raise ValueError("Joint96 inference cannot condition on GT actions.")
        for name, value in (("action_dim_is_pad", action_dim_is_pad),
                            ("proprio_dim_is_pad", proprio_dim_is_pad)):
            if not isinstance(value, torch.Tensor) or value.dtype != torch.bool or tuple(value.shape) != (1, 80):
                raise ValueError(f"Joint96 inference requires bool {name}[1,80].")
            if bool(value.all()):
                raise ValueError(f"Joint96 inference requires valid dimensions in {name}.")
        action_dim_is_pad = action_dim_is_pad.to(model.device)
        proprio_dim_is_pad = proprio_dim_is_pad.to(model.device)
        if proprio is None:
            raise ValueError("Joint96 inference requires current proprio.")
    height, width = int(input_image.shape[-2]), int(input_image.shape[-1])
    if height % 16 != 0 or width % 16 != 0:
        raise ValueError(
            "`input_image` spatial dims must be multiples of 16, "
            f"got HxW=({height},{width})."
        )
    num_frames = int(num_frames)
    action_horizon = int(action_horizon)
    if num_frames <= 1:
        raise ValueError(f"`num_frames` must be greater than 1, got {num_frames}.")
    if int(num_inference_steps) <= 0:
        raise ValueError("`num_inference_steps` must be positive.")
    model._action_token_seq_len_for_mask(action_horizon)

    temporal_factor = int(model.vae.temporal_downsample_factor)
    if (num_frames - 1) % temporal_factor != 0:
        raise ValueError(
            "Future RGB frames must align with the VAE temporal factor: "
            f"num_frames={num_frames}, factor={temporal_factor}."
        )
    future_latent_frames = (num_frames - 1) // temporal_factor

    if proprio is not None:
        if model.proprio_dim is None:
            raise ValueError("`proprio` was provided but `proprio_dim=None`.")
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        if proprio.ndim != 2 or int(proprio.shape[1]) != int(model.proprio_dim):
            raise ValueError(
                f"`proprio` must be [D] or [1,D], got {tuple(proprio.shape)}."
            )
        proprio = proprio.to(device=model.device, dtype=model.torch_dtype)
        if joint96:
            proprio = proprio.masked_fill(proprio_dim_is_pad, 0.0)
    elif model.understanding_enabled:
        raise ValueError(
            "Action Qwen/state conditioning requires current `proprio`."
        )

    use_prompt = prompt is not None
    use_context = context is not None or context_mask is not None
    if use_prompt and use_context:
        raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
    if not use_prompt and not use_context:
        raise ValueError(
            "Video T5 conditioning requires either `prompt` or both "
            "`context/context_mask`."
        )
    if use_prompt:
        context, context_mask = model.encode_prompt(prompt)
    else:
        if context is None or context_mask is None:
            raise ValueError(
                "`context` and `context_mask` must be both provided together."
            )
        if context.ndim == 2:
            context = context.unsqueeze(0)
        if context_mask.ndim == 1:
            context_mask = context_mask.unsqueeze(0)
        context = context.to(
            device=model.device, dtype=model.torch_dtype, non_blocking=True
        )
        context_mask = context_mask.to(
            device=model.device, dtype=torch.bool, non_blocking=True
        )
    if proprio is not None and model.proprio_encoder is not None:
        context, context_mask = append_proprio_to_context(
            model,
            context=context,
            context_mask=context_mask,
            proprio=proprio,
        )
    video_context = context
    video_context_mask = context_mask
    action_context = video_context
    action_context_mask = video_context_mask

    first_frame_latents = video_codec.encode_input_image_latents_tensor(
        model,
        input_image=input_image,
        tiled=tiled,
        video_layout=video_layout,
        video_view_names=video_view_names,
        camera_is_pad=camera_is_pad,
    )
    if joint96:
        spatial_valid = joint96_camera_valid_mask(
            camera_is_pad, batch_size=1, height=int(first_frame_latents.shape[-2]),
            width=int(first_frame_latents.shape[-1]), device=model.device,
        )
        latent_valid = spatial_valid[:, None, None]
        first_frame_latents = first_frame_latents.masked_fill(~latent_valid, 0.0)
        anchor_latents = recent_latents = first_frame_latents[:, :, :0]
        anchor_valid = recent_valid = torch.zeros((1, 0), dtype=torch.bool, device=model.device)
    else:
        anchor_latents, anchor_valid = _encode_online_condition_frame(
            model,
            memory_inputs,
            name="memory_video_anchor",
            height=height,
            width=width,
            tiled=tiled,
            video_layout=video_layout,
            video_view_names=video_view_names,
        )
        recent_latents, recent_valid = _encode_online_condition_frame(
            model,
            memory_inputs,
            name="memory_video_recent",
            height=height,
            width=width,
            tiled=tiled,
            video_layout=video_layout,
            video_view_names=video_view_names,
        )

    generator = (
        None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
    )
    future_latents = torch.randn(
        (
            1,
            int(first_frame_latents.shape[1]),
            future_latent_frames,
            int(first_frame_latents.shape[3]),
            int(first_frame_latents.shape[4]),
        ),
        generator=generator,
        device=rand_device,
        dtype=torch.float32,
    ).to(device=model.device, dtype=model.torch_dtype)
    latents_action = torch.randn(
        (1, action_horizon, int(model.action_expert.action_dim)),
        generator=generator,
        device=rand_device,
        dtype=torch.float32,
    ).to(device=model.device, dtype=model.torch_dtype)
    if joint96:
        future_latents = future_latents.masked_fill(~latent_valid, 0.0)
        latents_action = latents_action.masked_fill(action_dim_is_pad[:, None], 0.0)

    action_condition = None
    if model.video_expert.action_conditioned:
        if action is None:
            raise ValueError(
                "Action-conditioned video evaluation requires the GT action window."
            )
        if action.ndim == 2:
            action = action.unsqueeze(0)
        action_condition = action.to(device=model.device, dtype=model.torch_dtype)

    vlm_pack = None
    if model.understanding_enabled:
        vlm_pack = _get_online_vlm_pack(
            model,
            vlm_current_images=vlm_current_images,
            proprio=proprio,
            prompt=(
                understanding_prompt
                if understanding_prompt is not None
                else (prompt or "")
            ),
            vlm_view_names=vlm_view_names,
            vlm_current_view_is_pad=vlm_current_view_is_pad,
        )
        action_context = vlm_pack["vlm_context"].to(
            device=model.device, dtype=model.torch_dtype
        )
        action_context_mask = vlm_pack["vlm_mask"].to(
            device=model.device, dtype=torch.bool
        )

    video_timesteps, video_deltas = model.infer_video_scheduler.build_inference_schedule(
        num_inference_steps=int(num_inference_steps),
        device=model.device,
        dtype=future_latents.dtype,
        shift_override=video_sigma_shift,
    )
    action_timesteps, action_deltas = model.infer_action_scheduler.build_inference_schedule(
        num_inference_steps=int(num_inference_steps),
        device=model.device,
        dtype=latents_action.dtype,
        shift_override=action_sigma_shift,
    )

    num_condition_frames = 1 if joint96 else 3
    for step_t_video, step_delta_video, step_t_action, step_delta_action in zip(
        video_timesteps,
        video_deltas,
        action_timesteps,
        action_deltas,
    ):
        window_latents = torch.cat(
            [first_frame_latents, future_latents], dim=2
        )
        video_latents = torch.cat(
            [anchor_latents, recent_latents, window_latents], dim=2
        )
        video_frame_timesteps = torch.cat(
            [
                torch.zeros(
                    (1, num_condition_frames - 1),
                    dtype=video_latents.dtype,
                    device=model.device,
                ),
                step_t_video.reshape(1, 1).expand(1, future_latent_frames),
            ],
            dim=1,
        )
        timestep_action = step_t_action.reshape(1).to(
            device=model.device, dtype=latents_action.dtype
        )

        video_pre = model.video_expert.pre_dit(
            x=video_latents,
            timestep=video_frame_timesteps,
            context=video_context,
            context_mask=video_context_mask,
            action=action_condition,
            fuse_vae_embedding_in_latents=joint96 or bool(
                getattr(model.video_expert, "fuse_vae_embedding_in_latents", False)
            ),
            video_layout=video_layout,
            video_view_names=video_view_names,
            **({"frame_ids": sequence_start_video_rope_time_ids(
                1, 1 + future_latent_frames, device=model.device,
            )} if joint96 else {}),
        )
        if model.future_delta_enabled:
            model.video_expert.append_future_delta_queries(
                video_pre,
                current_frame_index=num_condition_frames - 1,
            )
        action_pre = model.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=action_context,
            context_mask=action_context_mask,
            **({"embodiment_ids": embodiment_ids} if getattr(model.action_expert, "embodiment_prompt", None) is not None else {}),
        )
        video_tokens = video_pre["tokens"]
        action_tokens = action_pre["tokens"]
        tokens_per_frame = int(video_pre["meta"]["tokens_per_frame"])
        num_future_delta_tokens = int(
            video_pre["meta"].get("num_future_delta_tokens", 0)
        )
        mask_builder = build_joint96_attention_mask if joint96 else build_frame_memory_attention_mask
        attention_mask = mask_builder(
            video_seq_len=int(video_tokens.shape[1]),
            action_seq_len=int(action_tokens.shape[1]),
            video_tokens_per_frame=tokens_per_frame,
            device=video_tokens.device,
            **({} if joint96 else dict(
                num_condition_frames=num_condition_frames,
                num_future_delta_tokens=num_future_delta_tokens,
                num_anchor_condition_frames=1,
            )),
        )
        video_frame_valid = torch.cat(
            [
                anchor_valid,
                recent_valid,
                torch.ones(
                    (1, 1 + future_latent_frames),
                    dtype=torch.bool,
                    device=model.device,
                ),
            ],
            dim=1,
        )
        video_key_mask = video_frame_valid.repeat_interleave(
            tokens_per_frame, dim=1
        )
        if num_future_delta_tokens:
            video_key_mask = torch.cat(
                [
                    video_key_mask,
                    torch.ones(
                        (1, num_future_delta_tokens),
                        dtype=torch.bool,
                        device=video_key_mask.device,
                    ),
                ],
                dim=1,
            )
        action_key_mask = torch.ones(
            action_tokens.shape[:2], dtype=torch.bool, device=action_tokens.device
        )
        if joint96:
            patch_t, patch_h, patch_w = map(int, model.video_expert.patch_size)
            height, width = first_frame_latents.shape[-2:]
            if patch_t != 1 or height % patch_h or (width // 3) % patch_w:
                raise ValueError("Joint96 patches must align to each frame and camera slot.")
            spatial_tokens = joint96_camera_valid_mask(
                camera_is_pad, batch_size=1, height=height // patch_h, width=width // patch_w,
                device=model.device,
            ).flatten(1)
            if spatial_tokens.shape[1] != tokens_per_frame:
                raise ValueError("Joint96 camera mask does not match video token geometry.")
            video_key_mask &= spatial_tokens.repeat(1, 1 + future_latent_frames)
            for pre, valid in ((video_pre, video_key_mask), (action_pre, action_key_mask)):
                mask = pre["context_mask"]
                pre["context_mask"] = mask & (valid[:, :, None] if mask.ndim == 3 else valid[:, None, :, None])
        tokens_out = model.mot(
            embeds_all={"video": video_tokens, "action": action_tokens},
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
            key_masks_all={
                "video": video_key_mask,
                "action": action_key_mask,
            },
            **({"query_masks_all": {"video": video_key_mask, "action": action_key_mask}} if joint96 else {}),
        )
        pred_video_all = model.video_expert.post_dit(
            tokens_out["video"], video_pre
        )
        pred_future_video = pred_video_all[:, :, num_condition_frames:]
        pred_action = model.action_expert.post_dit(
            tokens_out["action"], action_pre
        )
        future_latents = model.infer_video_scheduler.step(
            pred_future_video, step_delta_video, future_latents
        )
        latents_action = model.infer_action_scheduler.step(
            pred_action, step_delta_action, latents_action
        )
        if joint96:
            future_latents = future_latents.masked_fill(~latent_valid, 0.0)
            latents_action = latents_action.masked_fill(action_dim_is_pad[:, None], 0.0)

    generated_window = torch.cat(
        [first_frame_latents, future_latents], dim=2
    )
    frames = video_codec.decode_latents(
        model,
        generated_window,
        tiled=tiled,
        video_layout=video_layout,
        video_view_names=video_view_names,
    ) if decode_video else None
    return {
        "video": frames,
        "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
    }
