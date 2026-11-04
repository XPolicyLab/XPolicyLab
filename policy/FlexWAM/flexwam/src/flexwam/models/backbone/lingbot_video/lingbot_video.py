from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from torch import nn

from flexwam.models.backbone.common.cross_view_attention import CrossViewAttentionMixin
from flexwam.models.backbone.common.attention import sinusoidal_embedding_1d
from flexwam.models.architectures.state_conditioning import encode_state, state_config


class LingBotVideoRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


def _apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    # x: [B, S, H, D], freqs_cis: [B|1, S, D/2] complex64
    with torch.amp.autocast("cuda", enabled=False):
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        out = torch.view_as_real(x_complex * freqs_cis.unsqueeze(2)).flatten(3)
    return out.type_as(x)


class LingBotVideoRotaryEmbedding(nn.Module):
    def __init__(self, axes_dims: Tuple[int, ...], axes_lens: Tuple[int, ...], theta: float):
        super().__init__()
        self.axes_dims = tuple(int(x) for x in axes_dims)
        self.axes_lens = [int(x) for x in axes_lens]
        self.theta = float(theta)
        self.freqs_cis: list[torch.Tensor] | None = None

    @staticmethod
    def precompute_freqs_cis(dims: Tuple[int, ...], ends: Tuple[int, ...], theta: float) -> list[torch.Tensor]:
        freqs_cis = []
        for dim, end in zip(dims, ends):
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
            positions = torch.arange(end, dtype=torch.float64)
            phases = torch.outer(positions, freqs).float()
            freqs_cis.append(torch.polar(torch.ones_like(phases), phases).to(torch.complex64))
        return freqs_cis

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        device = position_ids.device
        if self.freqs_cis is None:
            max_vals = position_ids.max(dim=0).values.tolist()
            self.axes_lens = [
                max(axis_len, 2 * math.ceil((int(max_v) + 1) / 2)) for max_v, axis_len in zip(max_vals, self.axes_lens)
            ]
            self.freqs_cis = self.precompute_freqs_cis(self.axes_dims, tuple(self.axes_lens), self.theta)
        if self.freqs_cis[0].device != device:
            self.freqs_cis = [freqs.to(device=device) for freqs in self.freqs_cis]
        pieces = [freqs[position_ids[:, axis_idx]] for axis_idx, freqs in enumerate(self.freqs_cis)]
        return torch.cat(pieces, dim=-1)


class LingBotTextProjection(nn.Module):
    def __init__(self, text_dim: int, hidden_size: int, eps: float):
        super().__init__()
        self.norm = LingBotVideoRMSNorm(text_dim, eps=eps)
        self.linear_1 = nn.Linear(text_dim, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm(hidden_states)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = F.silu(hidden_states)
        return self.linear_2(hidden_states)


class LingBotVideoMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LingBotVideoSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, head_dim: int, eps: float, qkv_bias: bool, out_bias: bool):
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        inner_dim = self.num_heads * self.head_dim
        self.to_q = nn.Linear(hidden_size, inner_dim, bias=qkv_bias)
        self.to_k = nn.Linear(hidden_size, inner_dim, bias=qkv_bias)
        self.to_v = nn.Linear(hidden_size, inner_dim, bias=qkv_bias)
        self.to_out = nn.Linear(inner_dim, hidden_size, bias=out_bias)
        self.norm_q = LingBotVideoRMSNorm(self.head_dim, eps=eps)
        self.norm_k = LingBotVideoRMSNorm(self.head_dim, eps=eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        key_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        q = self.to_q(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim)
        k = self.to_k(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim)
        v = self.to_v(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim)

        q = self.norm_q(q)
        k = self.norm_k(k)
        q = _apply_rotary_emb(q, freqs_cis)
        k = _apply_rotary_emb(k, freqs_cis)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn_mask = None
        if key_mask is not None:
            attn_mask = key_mask[:, None, None, :].to(torch.bool)
        hidden_states = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        hidden_states = hidden_states.transpose(1, 2).reshape(bsz, seq_len, self.num_heads * self.head_dim)
        return self.to_out(hidden_states)


class LingBotVideoTransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int, head_dim: int, eps: float, qkv_bias: bool, out_bias: bool):
        super().__init__()
        self.norm1 = LingBotVideoRMSNorm(hidden_size, eps=eps)
        self.norm2 = LingBotVideoRMSNorm(hidden_size, eps=eps)
        self.norm_post_attn = LingBotVideoRMSNorm(hidden_size, eps=eps)
        self.norm_post_ffn = LingBotVideoRMSNorm(hidden_size, eps=eps)
        self.attn = LingBotVideoSelfAttention(hidden_size, num_heads, head_dim, eps, qkv_bias=qkv_bias, out_bias=out_bias)
        self.ffn = LingBotVideoMLP(hidden_size, intermediate_size)
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6 * hidden_size) / hidden_size**0.5)

    def _split_modulation(self, t_mod: torch.Tensor) -> tuple[torch.Tensor, ...]:
        hidden_size = int(self.scale_shift_table.shape[-1] // 6)
        if t_mod.ndim == 4:
            # Per-token AdaLN: [B, 6, S, D]. Prefix tokens can zero the t branch.
            if t_mod.shape[1] != 6 or t_mod.shape[-1] != hidden_size:
                raise ValueError(
                    f"`t_mod` per-token shape must be [B,6,S,{hidden_size}], got {tuple(t_mod.shape)}."
                )
            base = self.scale_shift_table.to(dtype=t_mod.dtype, device=t_mod.device).view(1, 6, 1, hidden_size)
            return tuple(chunk.squeeze(1) for chunk in (base + t_mod).chunk(6, dim=1))
        if t_mod.ndim == 3:
            base = self.scale_shift_table.to(dtype=t_mod.dtype, device=t_mod.device).view(1, 6, -1)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (base + t_mod).chunk(6, dim=1)
            return (
                shift_msa.squeeze(1),
                scale_msa.squeeze(1),
                gate_msa.squeeze(1),
                shift_mlp.squeeze(1),
                scale_mlp.squeeze(1),
                gate_mlp.squeeze(1),
            )
        if t_mod.ndim == 2:
            base = self.scale_shift_table.to(dtype=t_mod.dtype, device=t_mod.device)
            return (base + t_mod).chunk(6, dim=-1)
        raise ValueError(f"`t_mod` must be [B,6D], [B,6,D], or [B,6,S,D], got {tuple(t_mod.shape)}.")

    @staticmethod
    def _mod_for_seq(mod: torch.Tensor) -> torch.Tensor:
        if mod.ndim == 2:
            return mod.unsqueeze(1)
        if mod.ndim == 3:
            return mod
        raise ValueError(f"AdaLN chunk must be [B,D] or [B,S,D], got {tuple(mod.shape)}.")

    def _normalize_qk_for_mot(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = q.shape
        q = q.view(bsz, seq_len, self.attn.num_heads, self.attn.head_dim)
        k = k.view(bsz, seq_len, self.attn.num_heads, self.attn.head_dim)
        q = self.attn.norm_q(q)
        k = self.attn.norm_k(k)
        return q, k

    @staticmethod
    def _mot_freqs(freqs: torch.Tensor, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        if freqs.ndim == 3 and freqs.shape[0] == batch_size:
            return freqs.to(device=device)
        if freqs.ndim == 3 and freqs.shape[1] == 1:
            return freqs[:, 0, :].unsqueeze(0).expand(batch_size, -1, -1).to(device=device)
        if freqs.ndim == 2:
            return freqs.unsqueeze(0).expand(batch_size, -1, -1).to(device=device)
        raise ValueError(f"Unsupported LingBot RoPE shape for MoT: {tuple(freqs.shape)}.")

    def build_mot_attention_io(
        self,
        expert,
        x: torch.Tensor,
        freqs: torch.Tensor,
        t_mod: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        bool,
    ]:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._split_modulation(t_mod)
        attn_input = self.norm1(x)
        scale_msa = self._mod_for_seq(scale_msa)
        shift_msa = self._mod_for_seq(shift_msa)
        attn_input = attn_input * (1 + scale_msa) + shift_msa

        bsz, seq_len, _ = attn_input.shape
        q = self.attn.to_q(attn_input)
        k = self.attn.to_k(attn_input)
        v = self.attn.to_v(attn_input)
        q, k = self._normalize_qk_for_mot(q, k)
        freqs_cis = self._mot_freqs(freqs, bsz, seq_len, attn_input.device)
        q = _apply_rotary_emb(q, freqs_cis).reshape(bsz, seq_len, -1)
        k = _apply_rotary_emb(k, freqs_cis).reshape(bsz, seq_len, -1)

        use_gradient_checkpointing = bool(getattr(expert, "use_gradient_checkpointing", False))
        return q, k, v, x, gate_msa, shift_mlp, scale_mlp, gate_mlp, use_gradient_checkpointing

    def apply_mot_post_block(
        self,
        residual_x: torch.Tensor,
        mixed_attn_out: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        context_payload: Optional[dict],
    ) -> torch.Tensor:
        del context_payload
        gate_msa = self._mod_for_seq(gate_msa)
        scale_mlp = self._mod_for_seq(scale_mlp)
        shift_mlp = self._mod_for_seq(shift_mlp)
        gate_mlp = self._mod_for_seq(gate_mlp)
        x = residual_x + gate_msa * self.norm_post_attn(self.attn.to_out(mixed_attn_out))
        mlp_input = self.norm2(x)
        mlp_input = mlp_input * (1 + scale_mlp) + shift_mlp
        return x + gate_mlp * self.norm_post_ffn(self.ffn(mlp_input))

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        t_mod: torch.Tensor,
        key_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._split_modulation(t_mod)
        scale_msa = self._mod_for_seq(scale_msa)
        shift_msa = self._mod_for_seq(shift_msa)
        gate_msa = self._mod_for_seq(gate_msa)
        scale_mlp = self._mod_for_seq(scale_mlp)
        shift_mlp = self._mod_for_seq(shift_mlp)
        gate_mlp = self._mod_for_seq(gate_mlp)

        normed = self.norm1(hidden_states)
        normed = normed * (1 + scale_msa) + shift_msa
        hidden_states = hidden_states + gate_msa * self.norm_post_attn(
            self.attn(normed, freqs_cis=freqs_cis, key_mask=key_mask)
        )

        normed = self.norm2(hidden_states)
        normed = normed * (1 + scale_mlp) + shift_mlp
        hidden_states = hidden_states + gate_mlp * self.norm_post_ffn(self.ffn(normed))
        return hidden_states


class LingBotVideoTransformer3DModel(CrossViewAttentionMixin, ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True
    _no_split_modules = ["LingBotVideoTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        num_attention_heads: int = 16,
        hidden_size: int = 2048,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 2560,
        freq_dim: int = 256,
        intermediate_size: int = 6144,
        depth: int = 24,
        axes_dims: Tuple[int, int, int] = (32, 48, 48),
        axes_lens: Tuple[int, int, int] = (8192, 1024, 1024),
        rope_theta: float = 256.0,
        norm_eps: float = 1e-6,
        qkv_bias: bool = False,
        out_bias: bool = True,
        patch_embed_bias: bool = True,
        timestep_mlp_bias: bool = True,
        cross_view_attention_interval: int = 0,
        cross_view_attention_layers: Optional[Sequence[int]] = None,
        **kwargs,
    ):
        super().__init__()
        self.patch_size = tuple(int(x) for x in patch_size)
        self.hidden_size = int(hidden_size)
        self.out_channels = int(out_channels)
        self.num_attention_heads = int(num_attention_heads)
        self.attention_head_dim = self.hidden_size // self.num_attention_heads
        self.hidden_dim = self.hidden_size
        self.intermediate_size = int(intermediate_size)
        self.num_heads = self.num_attention_heads
        self.attn_head_dim = self.attention_head_dim
        self.in_dim = int(in_channels)
        self.freq_dim = int(freq_dim)
        self.text_dim = int(text_dim)

        patch_dim = int(in_channels) * math.prod(self.patch_size)
        self.patch_embedder = nn.Linear(patch_dim, self.hidden_size, bias=patch_embed_bias)
        self.time_embedder = TimestepEmbedding(
            in_channels=int(freq_dim),
            time_embed_dim=self.hidden_size,
            act_fn="silu",
            post_act_fn=None,
            out_dim=self.hidden_size,
            cond_proj_dim=None,
            sample_proj_bias=timestep_mlp_bias,
        )
        self.time_proj = Timesteps(num_channels=int(freq_dim), flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_modulation = nn.Sequential(nn.SiLU(), nn.Linear(self.hidden_size, 6 * self.hidden_size))
        self.text_embedder = LingBotTextProjection(int(text_dim), self.hidden_size, eps=norm_eps)
        self.blocks = nn.ModuleList(
            [
                LingBotVideoTransformerBlock(
                    self.hidden_size,
                    int(intermediate_size),
                    self.num_attention_heads,
                    self.attention_head_dim,
                    eps=norm_eps,
                    qkv_bias=bool(qkv_bias),
                    out_bias=bool(out_bias),
                )
                for _ in range(int(depth))
            ]
        )
        self.norm_out_modulation = nn.Sequential(nn.SiLU(), nn.Linear(self.hidden_size, 2 * self.hidden_size))
        self.proj_out = nn.Linear(self.hidden_size, int(out_channels) * math.prod(self.patch_size), bias=out_bias)
        self.rope = LingBotVideoRotaryEmbedding(tuple(axes_dims), tuple(axes_lens), theta=rope_theta)
        self.configure_cross_view_attention(cross_view_attention_interval, cross_view_attention_layers)
        self.gradient_checkpointing = False
        # One shared language prefix for the whole multiview sequence.
        self.prefix_layout = "global"
        self.num_views = int(kwargs.pop("num_views", 1))
        use_camera_embedding = kwargs.pop("use_camera_embedding", None)
        del kwargs
        self.configure_camera_embedding(self.num_views, enabled=use_camera_embedding)
        self.state_conditioning = state_config(None, video=True)
        self.state_proj = None
        self.proprio_dim = 0

    def configure_state_conditioning(self, proprio_dim: int, config: dict | None = None, *, shared: bool = False) -> None:
        """Configure before optimizer construction; disabled mode adds no parameters."""
        self.state_conditioning = state_config(config, video=True)
        self.proprio_dim = int(proprio_dim)
        if self.state_conditioning["enabled"] and not shared:
            reference = self.patch_embedder.weight
            self.state_proj = nn.Linear(self.proprio_dim, self.hidden_size).to(
                device=reference.device, dtype=reference.dtype,
            )
        else:
            self.state_proj = None

    def configure_camera_embedding(self, num_views: int, enabled: bool | None = None) -> None:
        """Create per-view camera embeddings, matching Wan's `camera_embedding`."""
        self.num_views = int(num_views)
        if enabled is None:
            enabled = self.num_views > 1
        self.use_camera_embedding = bool(enabled)
        if not self.use_camera_embedding:
            self.camera_embedding = None
            return
        ref = next(self.parameters())
        self.camera_embedding = nn.Parameter(
            torch.randn(
                1,
                self.num_views,
                1,
                self.hidden_size,
                device=ref.device,
                dtype=ref.dtype,
            )
            / (self.hidden_size ** 0.5)
        )

    def _patchify(self, latents: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
        bsz, channels, frames, height, width = latents.shape
        pt, ph, pw = self.patch_size
        if frames % pt or height % ph or width % pw:
            raise ValueError(f"Latent shape {(frames, height, width)} must be divisible by patch size {self.patch_size}.")
        grid = (frames // pt, height // ph, width // pw)
        latents = latents.reshape(bsz, channels, grid[0], pt, grid[1], ph, grid[2], pw)
        latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(bsz, grid[0] * grid[1] * grid[2], -1)
        return self.patch_embedder(latents), grid

    def _unpatchify(self, tokens: torch.Tensor, grid: tuple[int, int, int]) -> torch.Tensor:
        bsz = tokens.shape[0]
        pt, ph, pw = self.patch_size
        tokens = self.proj_out(tokens)
        tokens = tokens.reshape(bsz, grid[0], grid[1], grid[2], self.out_channels, pt, ph, pw)
        return tokens.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(
            bsz, self.out_channels, grid[0] * pt, grid[1] * ph, grid[2] * pw
        )

    def _build_fused_time_conditioning(
        self,
        *,
        tokens: torch.Tensor,
        grid: tuple[int, int, int],
        prefix_len: int,
        num_views: int,
        video_tokens_per_frame: int,
        video_tokens_per_view: int,
        timestep: torch.Tensor,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-token AdaLN: first latent frame stays t=0, later frames use `timestep`.

        Token layout must be global prefix: [language][view_0][view_1]...
        `t_mod` is [B, 6, S, D] to match the existing LingBot block/MoT path.
        """
        num_latent_frames = int(grid[0])
        if timestep.ndim == 0:
            timestep = timestep[None].expand(batch_size)
        elif timestep.ndim == 1 and timestep.shape[0] == 1 and batch_size > 1:
            timestep = timestep.expand(batch_size)

        token_timesteps = torch.ones(
            (batch_size, num_latent_frames, video_tokens_per_frame),
            dtype=timestep.dtype,
            device=timestep.device,
        ) * timestep.view(batch_size, 1, 1)
        token_timesteps[:, 0, :] = 0
        per_view_timesteps = token_timesteps.reshape(batch_size, video_tokens_per_view)
        visual_timesteps = per_view_timesteps.unsqueeze(1).expand(-1, num_views, -1).reshape(
            batch_size,
            num_views * video_tokens_per_view,
        )
        prefix_timesteps = torch.zeros(
            (batch_size, prefix_len),
            dtype=timestep.dtype,
            device=timestep.device,
        )
        all_timesteps = torch.cat([prefix_timesteps, visual_timesteps], dim=1)
        if all_timesteps.shape[1] != tokens.shape[1]:
            raise ValueError(
                "Fused LingBot timestep length mismatch: "
                f"timesteps={all_timesteps.shape[1]}, tokens={tokens.shape[1]}."
            )

        flat_t = all_timesteps.reshape(-1)
        time_emb = self.time_proj(flat_t).to(dtype=tokens.dtype, device=tokens.device)
        time_emb = self.time_embedder(time_emb)
        seq_len = all_timesteps.shape[1]
        time_emb = time_emb.reshape(batch_size, seq_len, self.hidden_size)
        t_mod = self.time_modulation(time_emb).view(batch_size, seq_len, 6, self.hidden_size)
        t_mod = t_mod.permute(0, 2, 1, 3).contiguous()
        return time_emb, t_mod

    def pre_dit(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        fuse_vae_embedding_in_latents: bool = True,
        control_camera_latents_input: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        state_keep_mask: Optional[torch.Tensor] = None,
        state_tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        del control_camera_latents_input
        # Shared with the Wan pre_dit interface; clean-first conditioning is mandatory.
        if not fuse_vae_embedding_in_latents:
            raise ValueError("LingBot requires clean-first-frame conditioning.")
        if x.ndim not in (5, 6):
            raise ValueError(f"`x` must be [B,C,T,H,W] or [B,V,C,T,H,W], got {tuple(x.shape)}.")
        if context.ndim != 3:
            raise ValueError(f"`context` must be [B,L,D], got {tuple(context.shape)}.")
        if context_mask is None:
            context_mask = torch.ones((context.shape[0], context.shape[1]), dtype=torch.bool, device=context.device)

        is_multiview = x.ndim == 6
        batch_size = x.shape[0]
        num_views = int(x.shape[1]) if is_multiview else 1
        if is_multiview:
            x_for_patch = x.reshape(batch_size * num_views, *x.shape[2:]).contiguous()
        else:
            x_for_patch = x

        tokens, grid = self._patchify(x_for_patch)
        video_tokens_per_frame = grid[1] * grid[2]
        video_tokens_per_view = tokens.shape[1]
        if is_multiview:
            tokens = tokens.reshape(batch_size, num_views, video_tokens_per_view, self.hidden_size)
        else:
            tokens = tokens.unsqueeze(1)
        camera_embedding = getattr(self, "camera_embedding", None)
        if camera_embedding is not None:
            if int(camera_embedding.shape[1]) < num_views:
                raise ValueError(
                    f"`camera_embedding` has {camera_embedding.shape[1]} views, need {num_views}."
                )
            tokens = tokens + camera_embedding[:, :num_views].to(
                dtype=tokens.dtype,
                device=tokens.device,
            )

        context_tokens = self.text_embedder(context)
        context_tokens = context_tokens * context_mask.to(dtype=context_tokens.dtype).unsqueeze(-1)
        prefix_tokens = [context_tokens]
        prefix_freqs = [
            self.rope(torch.zeros((context_tokens.shape[1], 3), device=tokens.device, dtype=torch.long))
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        ]

        shared_state = state_tokens is not None
        state_len = 0
        if state_tokens is None and self.state_proj is not None:
            state_tokens = encode_state(
                self.state_proj, proprio, batch_size=batch_size, proprio_dim=self.proprio_dim,
                reference=context_tokens, keep_mask=state_keep_mask,
            ).unsqueeze(1)
        if state_tokens is not None:
            # Shared mode: [state, language, video]; separate mode keeps legacy order.
            state_index = 0 if shared_state else 1
            prefix_tokens.insert(state_index, state_tokens.to(context_tokens))
            prefix_freqs.insert(state_index,
                self.rope(torch.zeros((1, 3), device=tokens.device, dtype=torch.long))
                .unsqueeze(0).expand(batch_size, -1, -1)
            )
            state_len = 1

        # State and language are clean, shared conditioning. No GT actions.
        del action
        action_len = 0

        prefix_tokens_cat = torch.cat(prefix_tokens, dim=1) if len(prefix_tokens) > 1 else prefix_tokens[0]
        prefix_freqs_cat = torch.cat(prefix_freqs, dim=1) if len(prefix_freqs) > 1 else prefix_freqs[0]
        prefix_len = int(prefix_tokens_cat.shape[1])
        prefix_layout = str(getattr(self, "prefix_layout", "global"))
        if prefix_layout != "global":
            raise ValueError(f"LingBot `prefix_layout` must be 'global', got {prefix_layout!r}.")

        single_view_pos = self._video_position_ids(grid, device=tokens.device)
        video_freqs = self.rope(single_view_pos).unsqueeze(0).expand(batch_size, -1, -1)
        # One shared language prefix for the whole multiview sequence:
        # [language][view_0 visual][view_1 visual]...
        tokens = torch.cat(
            [
                prefix_tokens_cat,
                tokens.reshape(batch_size, num_views * video_tokens_per_view, self.hidden_size),
            ],
            dim=1,
        )
        tokens_per_view = int(video_tokens_per_view)
        video_freqs_per_view = video_freqs.unsqueeze(1).expand(-1, num_views, -1, -1)
        freqs = torch.cat(
            [
                prefix_freqs_cat,
                video_freqs_per_view.reshape(batch_size, num_views * video_tokens_per_view, -1),
            ],
            dim=1,
        )

        time_emb, t_mod = self._build_fused_time_conditioning(
            tokens=tokens,
            grid=grid,
            prefix_len=prefix_len,
            num_views=num_views,
            video_tokens_per_frame=video_tokens_per_frame,
            video_tokens_per_view=video_tokens_per_view,
            timestep=timestep,
            batch_size=batch_size,
        )

        return {
            "tokens": tokens,
            "freqs": freqs,
            "t": time_emb,
            "t_mod": t_mod,
            "context": None,
            "context_mask": None,
            "meta": {
                "grid_size": grid,
                "tokens_per_frame": video_tokens_per_frame,
                "tokens_per_view": tokens_per_view,
                "video_tokens_per_view": video_tokens_per_view,
                "video_tokens_per_frame": video_tokens_per_frame,
                "num_views": num_views,
                "is_multiview": is_multiview,
                "batch_size": batch_size,
                "prefix_len": prefix_len,
                "context_len": int(context_tokens.shape[1]),
                "context_start": state_len if shared_state else 0,
                "state_len": state_len,
                "action_len": 0,
                "prefix_layout": "global",
                "visual_tokens_start": prefix_len,
                "visual_tokens_per_view": video_tokens_per_view,
                "fuse_vae_embedding_in_latents": True,
            },
        }

    def post_dit(self, tokens: torch.Tensor, pre_state: Dict[str, Any]) -> torch.Tensor:
        grid = pre_state["meta"]["grid_size"]
        t = pre_state["t"]
        shift, scale = self.norm_out_modulation(t).chunk(2, dim=-1)
        tokens = tokens * (1 + scale) + shift
        prefix_len = int(pre_state["meta"].get("prefix_len", 0))
        num_views = int(pre_state["meta"].get("num_views", 1))
        if bool(pre_state["meta"].get("is_multiview", False)):
            batch_size = int(pre_state["meta"]["batch_size"])
            video_tokens_per_view = int(pre_state["meta"]["video_tokens_per_view"])
            tokens = tokens[:, prefix_len:, :]
            tokens = tokens.reshape(batch_size, num_views, video_tokens_per_view, self.hidden_size)
            tokens = tokens.reshape(batch_size * num_views, video_tokens_per_view, self.hidden_size)
            video = self._unpatchify(tokens, grid)
            return video.reshape(batch_size, num_views, *video.shape[1:]).contiguous()
        tokens = tokens[:, prefix_len:]
        return self._unpatchify(tokens, grid)

    def build_video_to_video_mask(
        self,
        video_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
        num_views: int = 1,
        prefix_len: int = 0,
        prefix_layout: str = "global",
    ) -> torch.Tensor:
        num_views = int(num_views)
        if prefix_layout != "global":
            raise ValueError(f"LingBot video mask only supports `prefix_layout='global'`, got {prefix_layout!r}.")
        prefix_len = int(prefix_len)
        if prefix_len < 0 or prefix_len >= video_seq_len:
            raise ValueError(
                f"Global prefix length must be in [0,{video_seq_len}), got {prefix_len}."
            )
        visual_seq_len = video_seq_len - prefix_len
        if visual_seq_len % num_views != 0:
            raise ValueError(
                "Global-prefix video sequence must divide into equal view ranges: "
                f"visual_seq_len={visual_seq_len}, num_views={num_views}."
            )
        visual_tokens_per_view = visual_seq_len // num_views
        if video_tokens_per_frame <= 0 or video_tokens_per_frame > visual_tokens_per_view:
            raise ValueError(
                "Invalid visual `video_tokens_per_frame` for global-prefix layout: "
                f"tokens_per_frame={video_tokens_per_frame}, "
                f"visual_tokens_per_view={visual_tokens_per_view}."
            )
        mask = torch.zeros((video_seq_len, video_seq_len), dtype=torch.bool, device=device)
        # The shared language prefix is conditioning-only.
        mask[:prefix_len, :prefix_len] = True
        for view_idx in range(num_views):
            start = prefix_len + view_idx * visual_tokens_per_view
            end = start + visual_tokens_per_view
            first_end = start + video_tokens_per_frame
            mask[start:first_end, :prefix_len] = True
            mask[start:first_end, start:first_end] = True
            mask[first_end:end, :prefix_len] = True
            mask[first_end:end, start:end] = True
        return mask

    @staticmethod
    def _video_position_ids(grid: tuple[int, int, int], device: torch.device) -> torch.Tensor:
        t, h, w = grid
        ft = torch.arange(t, device=device).view(t, 1, 1).expand(t, h, w)
        fh = torch.arange(h, device=device).view(1, h, 1).expand(t, h, w)
        fw = torch.arange(w, device=device).view(1, 1, w).expand(t, h, w)
        return torch.stack([ft, fh, fw], dim=-1).reshape(-1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs,
    ):
        del kwargs
        if hidden_states.ndim != 5:
            raise ValueError(f"`hidden_states` must be [B,C,T,H,W], got {tuple(hidden_states.shape)}.")
        if encoder_hidden_states.ndim != 3:
            raise ValueError("`encoder_hidden_states` must be [B,L,D].")

        bsz = hidden_states.shape[0]
        video_tokens, grid = self._patchify(hidden_states)
        text_tokens = self.text_embedder(encoder_hidden_states)
        tokens = torch.cat([text_tokens, video_tokens], dim=1)

        text_len = text_tokens.shape[1]
        video_pos = self._video_position_ids(grid, device=tokens.device)
        text_pos = torch.zeros((text_len, 3), device=tokens.device, dtype=video_pos.dtype)
        pos_ids = torch.cat([text_pos, video_pos], dim=0).unsqueeze(0).expand(bsz, -1, -1)
        freqs_cis = self.rope(pos_ids.reshape(-1, 3)).reshape(bsz, pos_ids.shape[1], -1)

        if timestep.ndim == 0:
            timestep = timestep[None].expand(bsz)
        elif timestep.ndim == 1 and timestep.shape[0] == 1 and bsz > 1:
            timestep = timestep.expand(bsz)
        time_emb = self.time_proj(timestep).to(dtype=tokens.dtype, device=tokens.device)
        time_emb = self.time_embedder(time_emb)
        t_mod = self.time_modulation(time_emb)

        key_mask = None
        if encoder_attention_mask is not None:
            video_mask = torch.ones((bsz, video_tokens.shape[1]), dtype=torch.bool, device=tokens.device)
            key_mask = torch.cat([encoder_attention_mask.to(torch.bool), video_mask], dim=1)

        for block in self.blocks:
            if self.training and self.gradient_checkpointing:
                tokens = self._gradient_checkpointing_func(block, tokens, freqs_cis, t_mod, key_mask)
            else:
                tokens = block(tokens, freqs_cis=freqs_cis, t_mod=t_mod, key_mask=key_mask)

        video_tokens = tokens[:, text_len:]
        shift, scale = self.norm_out_modulation(time_emb).chunk(2, dim=-1)
        video_tokens = video_tokens * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        sample = self._unpatchify(video_tokens, grid)
        if not return_dict:
            return (sample,)
        return Transformer2DModelOutput(sample=sample)
