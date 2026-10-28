# Copyright 2025 The NVIDIA Team and The HuggingFace Team. All rights reserved.
# Modified by Galaxea AI, 2026, for inference-only use in G07-preview.
# Licensed under the Apache License, Version 2.0; see LICENSE and THIRD_PARTY_NOTICES.md
# in the G07_preview directory.
"""Cosmos3 MoT parameter modules and token/layer primitives.

Submodule and parameter names match the published Cosmos3-Nano diffusers
checkpoint (``layers.N.self_attn.{to_q,add_q_proj,...}``, ``proj_in``,
``proj_out``, ``time_embedder.linear_{1,2}``, ``norm``, ``norm_moe_gen`` …), so
the checkpoint state dict loads directly. FastWAM variant classes own packed
sequence ordering, attention masks, mRoPE assembly, and end-to-end forwards.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn_blocks import (
    RMSNorm,
    Cosmos3VLTextMLP,
    Cosmos3VLTextRotaryEmbedding,
    TimestepEmbedding,
    Timesteps,
    rotate_half,
)


@dataclass
class Cosmos3BackboneConfig:
    hidden_size: int = 4096
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 12288
    latent_channel: int = 48
    latent_patch_size: int = 2
    patch_latent_dim: int = 192
    rope_theta: float = 5_000_000.0
    rope_axes_dim: tuple = (24, 20, 20)
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    timestep_scale: float = 0.001
    vocab_size: int = 151936
    action_dim: int = 7
    proprio_dim: int = 8
    # "cogen": video/action share the generator tower.
    # "split": video generator plus a parallel narrow action expert.
    action_path: str = "cogen"
    action_hidden: int = 1024
    action_ffn: int = 4096

    @classmethod
    def from_transformer_config_json(
        cls,
        cfg: dict,
        *,
        action_dim: int,
        proprio_dim: int,
        action_path: str = "cogen",
        action_hidden: int = 1024,
        action_ffn: int = 4096,
    ) -> "Cosmos3BackboneConfig":
        rope_axes = tuple(cfg.get("rope_scaling", {}).get("mrope_section", [24, 20, 20]))
        return cls(
            hidden_size=cfg["hidden_size"],
            num_hidden_layers=cfg["num_hidden_layers"],
            num_attention_heads=cfg["num_attention_heads"],
            num_key_value_heads=cfg["num_key_value_heads"],
            head_dim=cfg["head_dim"],
            intermediate_size=cfg["intermediate_size"],
            latent_channel=cfg["latent_channel"],
            latent_patch_size=cfg["latent_patch_size"],
            patch_latent_dim=cfg["patch_latent_dim"],
            rope_theta=cfg["rope_theta"],
            rope_axes_dim=rope_axes,
            rms_norm_eps=cfg["rms_norm_eps"],
            attention_bias=cfg.get("attention_bias", False),
            timestep_scale=cfg.get("timestep_scale", 0.001),
            vocab_size=cfg["vocab_size"],
            action_dim=action_dim,
            proprio_dim=proprio_dim,
            action_path=str(action_path),
            action_hidden=int(action_hidden),
            action_ffn=int(action_ffn),
        )


class Cosmos3PackedMoTAttention(nn.Module):
    """Dual-pathway attention with separate UND/generator projections."""

    def __init__(self, cfg: Cosmos3BackboneConfig):
        super().__init__()
        hidden_size, head_dim = cfg.hidden_size, cfg.head_dim
        num_heads, num_kv_heads = cfg.num_attention_heads, cfg.num_key_value_heads
        bias, eps = cfg.attention_bias, cfg.rms_norm_eps
        self.num_attention_heads = num_heads
        self.num_key_value_heads = num_kv_heads
        self.head_dim = head_dim
        self.groups = num_heads // num_kv_heads

        self.to_q = nn.Linear(hidden_size, num_heads * head_dim, bias=bias)
        self.to_k = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=bias)
        self.to_v = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=bias)
        self.to_out = nn.Linear(num_heads * head_dim, hidden_size, bias=bias)
        self.norm_q = RMSNorm(head_dim, eps=eps, elementwise_affine=True, bias=False)
        self.norm_k = RMSNorm(head_dim, eps=eps, elementwise_affine=True, bias=False)
        # The add_* names mirror the published checkpoint; these are full
        # generator-tower weights.
        self.add_q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=bias)
        self.add_k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=bias)
        self.add_v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=bias)
        self.to_add_out = nn.Linear(num_heads * head_dim, hidden_size, bias=bias)
        self.norm_added_q = RMSNorm(head_dim, eps=eps, elementwise_affine=True, bias=False)
        self.norm_added_k = RMSNorm(head_dim, eps=eps, elementwise_affine=True, bias=False)

    def _project(self, seq, q_proj, k_proj, v_proj, norm_q, norm_k, cos, sin):
        batch_size, seq_len, _ = seq.shape
        q = norm_q(
            q_proj(seq).view(
                batch_size,
                seq_len,
                self.num_attention_heads,
                self.head_dim,
            )
        )
        k = norm_k(
            k_proj(seq).view(
                batch_size,
                seq_len,
                self.num_key_value_heads,
                self.head_dim,
            )
        )
        v = v_proj(seq).view(
            batch_size,
            seq_len,
            self.num_key_value_heads,
            self.head_dim,
        )
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        return q, k, v

    def _attend(self, q, k, v, mask):
        # q [B,Sq,nh,hd]; k,v [B,Sk,nkv,hd]; mask [B,Sq,Sk], True=attend.
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.groups > 1:
            # TODO: Benchmark SDPA native GQA to avoid physically repeating K/V.
            k = k.repeat_interleave(self.groups, dim=1)
            v = v.repeat_interleave(self.groups, dim=1)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask[:, None, :, :],
        )
        batch_size, num_heads, query_len, head_dim = out.shape
        return out.transpose(1, 2).reshape(
            batch_size,
            query_len,
            num_heads * head_dim,
        )


class Cosmos3VLTextMoTDecoderLayer(nn.Module):
    def __init__(self, cfg: Cosmos3BackboneConfig):
        super().__init__()
        hidden_size, eps = cfg.hidden_size, cfg.rms_norm_eps
        self.self_attn = Cosmos3PackedMoTAttention(cfg)
        self.mlp = Cosmos3VLTextMLP(hidden_size, cfg.intermediate_size)
        self.mlp_moe_gen = Cosmos3VLTextMLP(hidden_size, cfg.intermediate_size)
        self.input_layernorm = RMSNorm(
            hidden_size,
            eps=eps,
            elementwise_affine=True,
            bias=False,
        )
        self.input_layernorm_moe_gen = RMSNorm(
            hidden_size,
            eps=eps,
            elementwise_affine=True,
            bias=False,
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size,
            eps=eps,
            elementwise_affine=True,
            bias=False,
        )
        self.post_attention_layernorm_moe_gen = RMSNorm(
            hidden_size,
            eps=eps,
            elementwise_affine=True,
            bias=False,
        )


class Cosmos3Backbone(nn.Module):
    """Checkpoint-compatible modules and modality token preparation primitives."""

    def __init__(self, cfg: Cosmos3BackboneConfig):
        super().__init__()
        self.cfg = cfg
        hidden_size = cfg.hidden_size
        self.embed_tokens = nn.Embedding(cfg.vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [
                Cosmos3VLTextMoTDecoderLayer(cfg)
                for _ in range(cfg.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(
            hidden_size,
            eps=cfg.rms_norm_eps,
            elementwise_affine=True,
            bias=False,
        )
        self.norm_moe_gen = RMSNorm(
            hidden_size,
            eps=cfg.rms_norm_eps,
            elementwise_affine=True,
            bias=False,
        )
        self.rotary_emb = Cosmos3VLTextRotaryEmbedding(
            cfg.head_dim,
            cfg.rope_theta,
            list(cfg.rope_axes_dim),
        )
        self.proj_in = nn.Linear(cfg.patch_latent_dim, hidden_size, bias=True)
        self.proj_out = nn.Linear(hidden_size, cfg.patch_latent_dim, bias=True)
        self.time_proj = Timesteps(
            num_channels=256,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )
        self.time_embedder = TimestepEmbedding(
            in_channels=256,
            time_embed_dim=hidden_size,
        )
        self.proprio_encoder = nn.Linear(cfg.proprio_dim, hidden_size, bias=True)
        if cfg.action_path == "cogen":
            self.action_modality_embed = nn.Parameter(torch.zeros(hidden_size))
            self.action_in = nn.Linear(cfg.action_dim, hidden_size, bias=True)
            self.action_out = nn.Linear(hidden_size, cfg.action_dim, bias=True)
            self.action_expert = None
        elif cfg.action_path == "split":
            from .action_expert import Cosmos3ActionExpert

            self.action_expert = Cosmos3ActionExpert(
                cfg,
                cfg.action_hidden,
                cfg.action_ffn,
                cfg.num_hidden_layers,
            )
        else:
            raise ValueError(
                f"action_path must be 'cogen' or 'split', got {cfg.action_path!r}"
            )

    def generator_parameter_groups(
        self,
    ) -> dict[str, list[nn.Parameter]]:
        """Classify parameters used by the ``generator`` training scope.

        Ownership is defined by the module tree instead of parameter-name
        substrings, so renaming a module cannot silently change what is trained.

        Returns:
            A dictionary with two optimizer groups. ``generator`` contains the
            video/shared generator tower and uses the base learning rate;
            ``action`` contains action/proprio modules and uses
            ``action_lr_mult``. Both belong to the single ``generator`` scope.
        """
        generator_modules = [
            self.norm_moe_gen,
            self.proj_in,
            self.proj_out,
            self.time_embedder,
        ]
        if hasattr(self, "fdm_action_encoder"):
            generator_modules.append(self.fdm_action_encoder)
        frozen_und_modules = [self.embed_tokens, self.norm]
        for layer in self.layers:
            generator_modules.extend(
                [
                    layer.self_attn.add_q_proj,
                    layer.self_attn.add_k_proj,
                    layer.self_attn.add_v_proj,
                    layer.self_attn.to_add_out,
                    layer.self_attn.norm_added_q,
                    layer.self_attn.norm_added_k,
                    layer.input_layernorm_moe_gen,
                    layer.post_attention_layernorm_moe_gen,
                    layer.mlp_moe_gen,
                ]
            )
            frozen_und_modules.extend(
                [
                    layer.self_attn.to_q,
                    layer.self_attn.to_k,
                    layer.self_attn.to_v,
                    layer.self_attn.to_out,
                    layer.self_attn.norm_q,
                    layer.self_attn.norm_k,
                    layer.input_layernorm,
                    layer.post_attention_layernorm,
                    layer.mlp,
                ]
            )

        action_modules = [self.proprio_encoder]
        action_parameters = []
        if self.cfg.action_path == "cogen":
            action_modules.extend([self.action_in, self.action_out])
            action_parameters.append(self.action_modality_embed)
        else:
            action_modules.append(self.action_expert)

        generator_ids = {
            id(parameter)
            for module in generator_modules
            for parameter in module.parameters()
        }
        action_ids = {
            id(parameter)
            for module in action_modules
            for parameter in module.parameters()
        }
        action_ids.update(id(parameter) for parameter in action_parameters)
        frozen_und_ids = {
            id(parameter)
            for module in frozen_und_modules
            for parameter in module.parameters()
        }
        overlaps = (
            (generator_ids & action_ids)
            | (generator_ids & frozen_und_ids)
            | (action_ids & frozen_und_ids)
        )
        if overlaps:
            raise RuntimeError(
                f"{len(overlaps)} Cosmos3 parameters belong to multiple "
                "training-scope groups."
            )

        all_parameter_ids = {id(parameter) for parameter in self.parameters()}
        classified_ids = generator_ids | action_ids | frozen_und_ids
        unclassified_ids = all_parameter_ids - classified_ids
        if unclassified_ids:
            names_by_id = {
                id(parameter): name
                for name, parameter in self.named_parameters()
            }
            unclassified = sorted(
                names_by_id[parameter_id]
                for parameter_id in unclassified_ids
            )
            raise RuntimeError(
                "Every Cosmos3 backbone parameter must have an explicit scope: "
                f"unclassified={unclassified}"
            )

        generator, action = [], []
        for parameter in self.parameters():
            parameter_id = id(parameter)
            if parameter_id in generator_ids:
                generator.append(parameter)
            elif parameter_id in action_ids:
                action.append(parameter)
        return {"generator": generator, "action": action}

    def _encode_action(self, action_latent):
        """Raw action ``[B,A,Da]`` to shared hidden tokens."""
        if action_latent.shape[-1] != self.cfg.action_dim:
            raise ValueError(
                f"action last dimension must be {self.cfg.action_dim}, "
                f"got {action_latent.shape[-1]}"
            )
        return self.action_in(action_latent)

    def _decode_action(self, hidden):
        """Shared hidden tokens to raw action velocity ``[B,A,Da]``."""
        return self.action_out(hidden)

    def _patchify(self, latent):
        """Patchify ``[batch, channels, latent_t, height, width]`` into video tokens."""
        patch_size = self.cfg.latent_patch_size
        channels = self.cfg.latent_channel
        batch_size, _, latent_t, height, width = latent.shape
        patch_h, patch_w = height // patch_size, width // patch_size
        latent = latent.reshape(
            batch_size,
            channels,
            latent_t,
            patch_h,
            patch_size,
            patch_w,
            patch_size,
        )
        latent = torch.einsum("bcthpwq->bthwpqc", latent)
        return (
            latent.reshape(
                batch_size,
                latent_t * patch_h * patch_w,
                patch_size * patch_size * channels,
            ),
            (latent_t, patch_h, patch_w),
        )

    def _unpatchify(self, tokens, latent_t, patch_h, patch_w):
        """Restore video tokens to ``[batch, channels, latent_t, height, width]``."""
        patch_size = self.cfg.latent_patch_size
        channels = self.cfg.latent_channel
        batch_size = tokens.shape[0]
        tokens = tokens.reshape(
            batch_size,
            latent_t,
            patch_h,
            patch_w,
            patch_size,
            patch_size,
            channels,
        )
        tokens = torch.einsum("bthwpqc->bcthpwq", tokens)
        return tokens.reshape(
            batch_size,
            channels,
            latent_t,
            patch_h * patch_size,
            patch_w * patch_size,
        )

    def _embed_und_tokens(self, input_ids, vision_embeds, image_token_mask):
        """Embed the text-only or first-frame-vision UND segment."""
        und = self.embed_tokens(input_ids)
        if vision_embeds is not None:
            if image_token_mask is None:
                raise ValueError(
                    "`image_token_mask` is required when `vision_embeds` is provided."
                )
            und = und.clone()
            und[image_token_mask] = vision_embeds.to(
                dtype=und.dtype,
                device=und.device,
            ).reshape(-1, und.shape[-1])
        return und

    def _clean_video_tokens(self, video_latent):
        """Patchify/project clean video without adding a timestep embedding.

        This is the canonical Cosmos3 encoding for condition video: clean
        frames receive no timestep embedding at all. In particular, clean does
        not mean adding ``time_embedder(0)``.
        """
        dtype = self.proj_in.weight.dtype
        video_tokens, (latent_t, patch_h, patch_w) = self._patchify(
            video_latent.to(dtype)
        )
        video_tokens = self.proj_in(video_tokens)
        tokens_per_frame = patch_h * patch_w
        return video_tokens, latent_t, patch_h, patch_w, tokens_per_frame

    def _video_tokens_with_time(
        self,
        video_latent,
        video_timesteps,
        noisy_frame_mask,
    ):
        """Encode video and add ``E(t)`` only to explicitly noisy frames.

        Args:
            video_latent: Video latents shaped ``[B, C, latent_t, H, W]``.
            video_timesteps: One diffusion timestep per sample, shaped ``[B]``.
            noisy_frame_mask: Boolean mask shaped ``[B, latent_t]``. ``True``
                frames receive ``time_embedder(t)``; ``False`` clean frames
                receive no timestep embedding.

        Cosmos3 clean condition frames must never be represented by ``E(0)``.
        Callers therefore identify noisy frames explicitly instead of encoding
        clean frames with a zero timestep.
        """
        (
            video_tokens,
            latent_t,
            patch_h,
            patch_w,
            tokens_per_frame,
        ) = self._clean_video_tokens(video_latent)
        expected_mask_shape = (video_latent.shape[0], latent_t)
        if noisy_frame_mask.shape != expected_mask_shape:
            raise ValueError(
                "`noisy_frame_mask` must match [batch, latent_t], "
                f"expected {expected_mask_shape}, got {tuple(noisy_frame_mask.shape)}."
            )
        dtype = self.proj_in.weight.dtype
        time_embedding = self.time_embedder(
            self.time_proj(video_timesteps * self.cfg.timestep_scale).to(dtype)
        ).to(dtype)
        token_time_mask = (
            noisy_frame_mask.to(device=video_tokens.device)
            .repeat_interleave(tokens_per_frame, dim=1)
            .unsqueeze(-1)
        )
        time_add = time_embedding[:, None] * token_time_mask.to(dtype)
        return (
            video_tokens + time_add,
            latent_t,
            patch_h,
            patch_w,
            tokens_per_frame,
        )

    def _action_tokens_cogen(self, action_latent, action_timesteps):
        """Encode shared-tower action tokens with modality and timestep embeddings."""
        dtype = self.proj_in.weight.dtype
        action_tokens = (
            self._encode_action(action_latent.to(dtype))
            + self.action_modality_embed
        )
        time_embedding = self.time_embedder(
            self.time_proj(action_timesteps * self.cfg.timestep_scale).to(dtype)
        ).to(dtype)
        return action_tokens + time_embedding[:, None]
