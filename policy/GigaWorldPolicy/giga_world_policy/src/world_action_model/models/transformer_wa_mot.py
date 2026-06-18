from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from diffusers.configuration_utils import register_to_config
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from .action_state_dit import ActionStateDiT
from .mot import LayoutSegment, MoT, _slice_rotary
from .transformer_wa_casual import CasualWorldActionTransformer


def _additive_mask(
    allowed: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    mask = torch.zeros_like(allowed, dtype=dtype)
    return mask.masked_fill(~allowed, float("-inf"))


def _drop_unused_video_action_modules(video_expert: nn.Module) -> None:
    for attr in ("action_rope", "action_encoder", "action_decoder", "state_encoder", "condition_embedder_action"):
        if hasattr(video_expert, attr):
            delattr(video_expert, attr)


def _ensure_no_meta_tensors(module: nn.Module) -> None:
    meta_names = [name for name, tensor in module.named_parameters() if getattr(tensor, "is_meta", False)]
    meta_names.extend(name for name, tensor in module.named_buffers() if getattr(tensor, "is_meta", False))
    if meta_names:
        preview = ", ".join(meta_names[:8])
        suffix = "..." if len(meta_names) > 8 else ""
        raise RuntimeError(f"MoT model still contains meta tensors after cleanup: {preview}{suffix}")


class MoTWanVideoExpert(CasualWorldActionTransformer):
    """Wan video expert without the legacy GWP action/state branches."""

    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int, ...] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: Optional[int] = None,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        pos_embed_seq_len: Optional[int] = None,
    ) -> None:
        super().__init__(
            patch_size=patch_size,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            text_dim=text_dim,
            freq_dim=freq_dim,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            cross_attn_norm=cross_attn_norm,
            qk_norm=qk_norm,
            eps=eps,
            image_dim=image_dim,
            added_kv_proj_dim=added_kv_proj_dim,
            rope_max_seq_len=rope_max_seq_len,
            pos_embed_seq_len=pos_embed_seq_len,
        )
        _drop_unused_video_action_modules(self)


class MoTWorldActionTransformer(nn.Module):
    """GWP-compatible transformer wrapper using FastWAM-style MoT experts."""

    def __init__(
        self,
        video_expert: CasualWorldActionTransformer,
        action_expert: ActionStateDiT,
        mot_checkpoint_mixed_attn: bool = True,
        video_attention_mask_mode: str = "gwp_casual",
    ):
        super().__init__()
        if video_attention_mask_mode != "gwp_casual":
            raise ValueError(f"Only video_attention_mask_mode='gwp_casual' is supported, got {video_attention_mask_mode}")

        _drop_unused_video_action_modules(video_expert)
        video_expert.num_heads = int(video_expert.config.num_attention_heads)
        video_expert.attn_head_dim = int(video_expert.config.attention_head_dim)
        self.mot = MoT(
            {"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )
        self.video_attention_mask_mode = video_attention_mask_mode
        self.config = video_expert.config

    @property
    def video_expert(self) -> CasualWorldActionTransformer:
        return self.mot.mixtures["video"]

    @property
    def action_expert(self) -> ActionStateDiT:
        return self.mot.mixtures["action"]

    @classmethod
    def from_pretrained_video(
        cls,
        transformer_pretrained: str,
        torch_dtype: torch.dtype,
        action_dim: int,
        state_dim: int,
        action_expert: Optional[Dict[str, Any]] = None,
        mot_checkpoint_mixed_attn: bool = True,
        video_attention_mask_mode: str = "gwp_casual",
        unpretrain: bool = False,
    ) -> "MoTWorldActionTransformer":
        if unpretrain:
            video_expert = MoTWanVideoExpert.from_config(transformer_pretrained, torch_dtype=torch_dtype)
        else:
            video_expert = MoTWanVideoExpert.from_pretrained(transformer_pretrained, torch_dtype=torch_dtype)

        action_cfg = dict(action_expert or {})
        action_cfg.setdefault("action_dim", action_dim)
        action_cfg.setdefault("state_dim", state_dim)
        action_cfg.setdefault("hidden_dim", 1024)
        action_cfg.setdefault("ffn_dim", 4096)
        action_cfg.setdefault("text_dim", int(video_expert.config.text_dim))
        action_cfg.setdefault("freq_dim", int(video_expert.config.freq_dim))
        action_cfg.setdefault("num_heads", int(video_expert.config.num_attention_heads))
        action_cfg.setdefault("attn_head_dim", int(video_expert.config.attention_head_dim))
        action_cfg.setdefault("num_layers", int(video_expert.config.num_layers))
        action_cfg.setdefault("eps", float(video_expert.config.eps))
        action_cfg.setdefault("rope_max_seq_len", int(video_expert.config.rope_max_seq_len))
        action_module = ActionStateDiT(**action_cfg)

        model = cls(
            video_expert=video_expert,
            action_expert=action_module,
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
            video_attention_mask_mode=video_attention_mask_mode,
        )
        _ensure_no_meta_tensors(model)
        return model.to(dtype=torch_dtype)

    @staticmethod
    def _split_timesteps(
        timestep: torch.Tensor,
        batch_size: int,
        num_state_tokens: int,
        num_ref_tokens: int,
        num_action_tokens: int,
        num_noisy_tokens: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        if timestep.ndim == 1:
            if timestep.shape[0] == 1 and batch_size > 1:
                timestep = timestep.expand(batch_size)
            if timestep.shape[0] != batch_size:
                raise ValueError(f"Expected timestep length {batch_size}, got {tuple(timestep.shape)}")
            zero = torch.zeros((batch_size, 1), device=device, dtype=dtype)
            state_ts = zero.expand(batch_size, num_state_tokens)
            ref_ts = zero.expand(batch_size, num_ref_tokens)
            action_ts = timestep[:, None].to(device=device, dtype=dtype).expand(batch_size, num_action_tokens)
            noisy_ts = timestep[:, None].to(device=device, dtype=dtype).expand(batch_size, num_noisy_tokens)
            return state_ts, ref_ts, action_ts, noisy_ts

        expected = num_state_tokens + num_ref_tokens + num_action_tokens + num_noisy_tokens
        if timestep.ndim != 2 or timestep.shape[0] != batch_size or timestep.shape[1] < expected:
            raise ValueError(
                f"Expected timestep [B,{expected}] or longer, got {tuple(timestep.shape)}"
            )
        timestep = timestep[:, :expected].to(device=device, dtype=dtype)
        s_end = num_state_tokens
        r_end = s_end + num_ref_tokens
        a_end = r_end + num_action_tokens
        return (
            timestep[:, :s_end],
            timestep[:, s_end:r_end],
            timestep[:, r_end:a_end],
            timestep[:, a_end:expected],
        )

    def _build_video_pre(
        self,
        ref_latents: torch.Tensor,
        noisy_latents: Optional[torch.Tensor],
        video_timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
    ) -> dict:
        video_expert = self.video_expert
        if noisy_latents is None:
            hidden_states = ref_latents
        else:
            hidden_states = torch.cat([ref_latents, noisy_latents], dim=2)

        batch_size, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = video_expert.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w
        num_ref_tokens = (ref_latents.shape[2] // p_t) * post_patch_height * post_patch_width
        num_video_tokens = post_patch_num_frames * post_patch_height * post_patch_width
        num_noisy_tokens = num_video_tokens - num_ref_tokens

        rotary_emb = video_expert.rope(hidden_states)
        tokens = video_expert.patch_embedding(hidden_states.to(dtype=next(video_expert.patch_embedding.parameters()).dtype))
        tokens = tokens.flatten(2).transpose(1, 2)

        timestep_flat = video_timestep.flatten()
        temb, timestep_proj, context, encoder_hidden_states_image = video_expert.condition_embedder(
            timestep_flat,
            encoder_hidden_states,
            encoder_hidden_states_image,
            timestep_seq_len=video_timestep.shape[1],
        )
        timestep_proj = timestep_proj.unflatten(2, (6, -1))
        if encoder_hidden_states_image is not None:
            context = torch.concat([encoder_hidden_states_image, context], dim=1)

        return {
            "tokens": tokens,
            "rotary_emb": rotary_emb,
            "t_mod": timestep_proj,
            "temb": temb,
            "context": context,
            "meta": {
                "batch_size": batch_size,
                "num_frames": num_frames,
                "height": height,
                "width": width,
                "post_patch_num_frames": post_patch_num_frames,
                "post_patch_height": post_patch_height,
                "post_patch_width": post_patch_width,
                "num_ref_tokens": num_ref_tokens,
                "num_noisy_tokens": num_noisy_tokens,
                "patch_size": (p_t, p_h, p_w),
            },
        }

    def _post_video(self, video_tokens: torch.Tensor, video_pre: dict) -> torch.Tensor:
        video_expert = self.video_expert
        temb = video_pre["temb"]
        if temb.ndim == 3:
            shift, scale = (video_expert.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift, scale = shift.squeeze(2), scale.squeeze(2)
        else:
            shift, scale = (video_expert.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)

        hidden_states = (video_expert.norm_out(video_tokens.float()) * (1 + scale.to(video_tokens.device)) + shift.to(video_tokens.device)).type_as(video_tokens)
        hidden_states = video_expert.proj_out(hidden_states)

        meta = video_pre["meta"]
        p_t, p_h, p_w = meta["patch_size"]
        hidden_states = hidden_states.reshape(
            meta["batch_size"],
            meta["post_patch_num_frames"],
            meta["post_patch_height"],
            meta["post_patch_width"],
            p_t,
            p_h,
            p_w,
            -1,
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    @staticmethod
    def build_gwp_casual_mask(
        num_state_tokens: int,
        num_ref_tokens: int,
        num_action_tokens: int,
        num_noisy_tokens: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        total = num_state_tokens + num_ref_tokens + num_action_tokens + num_noisy_tokens
        allowed = torch.ones((total, total), dtype=torch.bool, device=device)
        s_r_end = num_state_tokens + num_ref_tokens
        action_end = s_r_end + num_action_tokens
        allowed[:s_r_end, s_r_end:] = False
        allowed[s_r_end:action_end, action_end:] = False
        return _additive_mask(allowed, dtype=dtype)

    @staticmethod
    def build_action_only_mask(
        num_state_tokens: int,
        num_ref_tokens: int,
        num_action_tokens: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        total = num_state_tokens + num_ref_tokens + num_action_tokens
        allowed = torch.ones((total, total), dtype=torch.bool, device=device)
        prefix_end = num_state_tokens + num_ref_tokens
        allowed[:prefix_end, prefix_end:] = False
        return _additive_mask(allowed, dtype=dtype)

    def _forward_full(
        self,
        noisy_latents: torch.Tensor,
        ref_latents: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor],
        return_dict: bool,
        state: torch.Tensor,
        action: torch.Tensor,
    ):
        hidden_states = torch.cat([ref_latents, noisy_latents], dim=2)
        batch_size, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_height = height // p_h
        post_patch_width = width // p_w
        num_state_tokens = state.shape[1]
        num_action_tokens = action.shape[1]
        num_ref_tokens = (ref_latents.shape[2] // p_t) * post_patch_height * post_patch_width
        num_noisy_tokens = (num_frames // p_t) * post_patch_height * post_patch_width - num_ref_tokens

        state_ts, ref_ts, action_ts, noisy_ts = self._split_timesteps(
            timestep,
            batch_size=batch_size,
            num_state_tokens=num_state_tokens,
            num_ref_tokens=num_ref_tokens,
            num_action_tokens=num_action_tokens,
            num_noisy_tokens=num_noisy_tokens,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        video_pre = self._build_video_pre(
            ref_latents=ref_latents,
            noisy_latents=noisy_latents,
            video_timestep=torch.cat([ref_ts, noisy_ts], dim=1),
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_image=encoder_hidden_states_image,
        )
        action_pre = self.action_expert.pre_dit(
            state=state.to(dtype=video_pre["tokens"].dtype),
            action=action.to(dtype=video_pre["tokens"].dtype),
            state_timestep=state_ts,
            action_timestep=action_ts,
            encoder_hidden_states=encoder_hidden_states,
        )

        layout = [
            LayoutSegment("action", 0, num_state_tokens),
            LayoutSegment("video", 0, num_ref_tokens),
            LayoutSegment("action", num_state_tokens, num_state_tokens + num_action_tokens),
            LayoutSegment("video", num_ref_tokens, num_ref_tokens + num_noisy_tokens),
        ]
        attention_mask = self.build_gwp_casual_mask(
            num_state_tokens=num_state_tokens,
            num_ref_tokens=num_ref_tokens,
            num_action_tokens=num_action_tokens,
            num_noisy_tokens=num_noisy_tokens,
            device=video_pre["tokens"].device,
            dtype=video_pre["tokens"].dtype,
        )
        tokens_out = self.mot(
            embeds_all={"video": video_pre["tokens"], "action": action_pre["tokens"]},
            attention_mask=attention_mask,
            rotary_all={"video": video_pre["rotary_emb"], "action": action_pre["rotary_emb"]},
            context_all={"video": video_pre["context"], "action": action_pre["context"]},
            t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]},
            layout=layout,
        )
        video_pred = self._post_video(tokens_out["video"], video_pre)
        action_tokens = tokens_out["action"][:, num_state_tokens : num_state_tokens + num_action_tokens]
        action_pred = self.action_expert.post_action(action_tokens)

        if not return_dict:
            return video_pred, action_pred
        return Transformer2DModelOutput(sample=video_pred)

    def clear_action_only_cache(self):
        self._action_only_prefix_cache = None

    def _forward_action_only(
        self,
        noisy_latents: torch.Tensor,
        ref_latents: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor],
        return_dict: bool,
        state: torch.Tensor,
        action: torch.Tensor,
    ):
        batch_size, _, _, height, width = ref_latents.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_height = height // p_h
        post_patch_width = width // p_w
        num_state_tokens = state.shape[1]
        num_action_tokens = action.shape[1]
        num_ref_tokens = (ref_latents.shape[2] // p_t) * post_patch_height * post_patch_width
        num_noisy_tokens = 0

        state_ts, ref_ts, action_ts, _ = self._split_timesteps(
            timestep,
            batch_size=batch_size,
            num_state_tokens=num_state_tokens,
            num_ref_tokens=num_ref_tokens,
            num_action_tokens=num_action_tokens,
            num_noisy_tokens=num_noisy_tokens,
            device=ref_latents.device,
            dtype=ref_latents.dtype,
        )
        action_dtype = next(self.action_expert.parameters()).dtype
        action_pre = self.action_expert.pre_dit(
            state=state.to(device=ref_latents.device, dtype=action_dtype),
            action=action.to(device=ref_latents.device, dtype=action_dtype),
            state_timestep=state_ts,
            action_timestep=action_ts,
            encoder_hidden_states=encoder_hidden_states,
        )

        image_key = None
        if encoder_hidden_states_image is not None:
            image_key = (
                int(encoder_hidden_states_image.data_ptr()),
                tuple(encoder_hidden_states_image.shape),
                str(encoder_hidden_states_image.device),
            )
        cache_key = (
            int(ref_latents.data_ptr()),
            tuple(ref_latents.shape),
            str(ref_latents.device),
            int(state.data_ptr()),
            tuple(state.shape),
            str(state.device),
            int(encoder_hidden_states.data_ptr()),
            tuple(encoder_hidden_states.shape),
            str(encoder_hidden_states.device),
            image_key,
            num_state_tokens,
            num_ref_tokens,
        )
        cached = getattr(self, "_action_only_prefix_cache", None)
        prefix_cache = None
        if not self.training and cached is not None and cached.get("key") == cache_key:
            prefix_cache = cached["prefix_cache"]

        if prefix_cache is None:
            video_pre = self._build_video_pre(
                ref_latents=ref_latents,
                noisy_latents=None,
                video_timestep=ref_ts,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_image=encoder_hidden_states_image,
            )
            prefix_mask = self.build_action_only_mask(
                num_state_tokens=num_state_tokens,
                num_ref_tokens=num_ref_tokens,
                num_action_tokens=0,
                device=video_pre["tokens"].device,
                dtype=video_pre["tokens"].dtype,
            )
            prefix_cache = self.mot.prefill_prefix_cache(
                embeds_all={
                    "action": action_pre["tokens"][:, :num_state_tokens],
                    "video": video_pre["tokens"],
                },
                attention_mask=prefix_mask,
                rotary_all={
                    "action": _slice_rotary(action_pre["rotary_emb"], 0, num_state_tokens),
                    "video": video_pre["rotary_emb"],
                },
                context_all={"action": action_pre["context"], "video": video_pre["context"]},
                t_mod_all={
                    "action": action_pre["t_mod"][:, :num_state_tokens],
                    "video": video_pre["t_mod"],
                },
                layout=[
                    LayoutSegment("action", 0, num_state_tokens),
                    LayoutSegment("video", 0, num_ref_tokens),
                ],
            )
            if not self.training:
                self._action_only_prefix_cache = {"key": cache_key, "prefix_cache": prefix_cache}

        action_attention_mask = torch.zeros(
            (num_action_tokens, num_state_tokens + num_ref_tokens + num_action_tokens),
            device=action_pre["tokens"].device,
            dtype=action_pre["tokens"].dtype,
        )
        action_tokens = self.mot.forward_action_with_prefix_cache(
            action_tokens=action_pre["tokens"][:, num_state_tokens:],
            action_rotary=_slice_rotary(action_pre["rotary_emb"], num_state_tokens, num_state_tokens + num_action_tokens),
            action_t_mod=action_pre["t_mod"][:, num_state_tokens:],
            action_context=action_pre["context"],
            prefix_kv_cache=prefix_cache,
            attention_mask=action_attention_mask,
        )
        action_pred = self.action_expert.post_action(action_tokens)
        if not return_dict:
            return action_pred
        return Transformer2DModelOutput(sample=None)

    def forward(
        self,
        noisy_latents: torch.Tensor = None,
        ref_latents: torch.Tensor = None,
        timestep: torch.Tensor = None,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        state: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        action_only: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        if attention_kwargs is not None and attention_kwargs.get("scale", None) not in (None, 1.0):
            raise ValueError("LoRA attention scaling is not supported by MoTWorldActionTransformer v1.")
        if ref_latents is None or timestep is None or encoder_hidden_states is None or state is None or action is None:
            raise ValueError("ref_latents, timestep, encoder_hidden_states, state, and action are required.")
        if action_only:
            return self._forward_action_only(
                noisy_latents=noisy_latents,
                ref_latents=ref_latents,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_image=encoder_hidden_states_image,
                return_dict=return_dict,
                state=state,
                action=action,
            )
        return self._forward_full(
            noisy_latents=noisy_latents,
            ref_latents=ref_latents,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_image=encoder_hidden_states_image,
            return_dict=return_dict,
            state=state,
            action=action,
        )


__all__ = ["MoTWorldActionTransformer"]
