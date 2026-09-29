"""Causal current/future-DINO and action Mixture-of-Transformers.

The physical stream follows FastWAM's clean-prefix layout and is ordered as
``[z0, z_future_t, action_t]``.  ``z0`` is the full-resolution clean current
observation and is clamped throughout sampling.  Only the lower-resolution
future DINO tokens and action tokens are noised and denoised.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


def _cfg_get(config, key: str, default=None):
    if config is None:
        return default
    getter = getattr(config, "get", None)
    return getter(key, default) if callable(getter) else getattr(config, key, default)


def _sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    if dim <= 0 or dim % 2:
        raise ValueError("time_frequency_dim must be a positive even integer")
    sinusoid = torch.outer(
        position.to(torch.float64),
        torch.pow(
            10_000,
            -torch.arange(dim // 2, dtype=torch.float64, device=position.device)
            / (dim // 2),
        ),
    )
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1).to(
        position.dtype
    )


def _precompute_freqs_cis(
    dim: int,
    end: int = 1024,
    theta: float = 10_000.0,
    real_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if dim <= 0 or dim % 2:
        raise ValueError("RoPE dimensions must be positive even integers")
    frequencies = 1.0 / (
        theta
        ** (
            torch.arange(0, dim, 2, dtype=real_dtype)[: dim // 2]
            / float(dim)
        )
    )
    phase = torch.outer(
        torch.arange(end, dtype=real_dtype),
        frequencies,
    )
    return torch.polar(torch.ones_like(phase), phase)


def _precompute_freqs_cis_2d(
    dim: int,
    end: int = 1024,
    real_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    height_dim = dim // 2
    width_dim = dim - height_dim
    if min(height_dim, width_dim) <= 0 or height_dim % 2 or width_dim % 2:
        raise ValueError(
            "attention_head_dim must split into even height/width RoPE dimensions"
        )
    return (
        _precompute_freqs_cis(
            height_dim,
            end=end,
            real_dtype=real_dtype,
        ),
        _precompute_freqs_cis(
            width_dim,
            end=end,
            real_dtype=real_dtype,
        ),
    )


def _rope_apply(
    hidden: torch.Tensor,
    frequencies: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    batch, length, inner_dim = hidden.shape
    if inner_dim % num_heads:
        raise ValueError(
            f"attention inner dim {inner_dim} is not divisible by heads={num_heads}"
        )
    head_dim = inner_dim // num_heads
    values = hidden.reshape(batch, length, num_heads, head_dim)
    real_dtype = (
        torch.float64
        if frequencies.dtype == torch.complex128
        else torch.float32
    )
    complex_values = torch.view_as_complex(
        values.to(real_dtype).reshape(batch, length, num_heads, -1, 2)
    )
    frequencies = frequencies.to(
        device=hidden.device,
        dtype=complex_values.dtype,
    )
    complex_values = complex_values * frequencies
    return torch.view_as_real(complex_values).flatten(2).to(hidden.dtype)


def _attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    num_heads: int,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    batch, query_length, inner_dim = query.shape
    head_dim = inner_dim // num_heads

    def split(value_tensor: torch.Tensor) -> torch.Tensor:
        return value_tensor.view(
            batch,
            value_tensor.shape[1],
            num_heads,
            head_dim,
        ).transpose(1, 2)

    attended = F.scaled_dot_product_attention(
        split(query),
        split(key),
        split(value),
        attn_mask=attention_mask,
    )
    return attended.transpose(1, 2).reshape(batch, query_length, inner_dim)


@dataclass(frozen=True)
class _BaseActionCache:
    """Time-independent base-policy inputs reused across action denoising."""

    batch_size: int
    action_contexts: tuple[torch.Tensor, ...]
    world_keys: tuple[torch.Tensor, ...]
    world_values: tuple[torch.Tensor, ...]
    action_frequencies: torch.Tensor


class InnerRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        dtype = hidden.dtype
        normalized = hidden.float() * torch.rsqrt(
            hidden.float().pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        # Keep the activation contract stable when the learned scale is kept
        # in fp32 while the surrounding attention projections remain bf16.
        return (normalized.to(dtype) * self.weight).to(dtype)


class ExpertSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        attention_inner_dim: int,
        eps: float,
    ) -> None:
        super().__init__()
        self.q = nn.Linear(hidden_size, attention_inner_dim)
        self.k = nn.Linear(hidden_size, attention_inner_dim)
        self.v = nn.Linear(hidden_size, attention_inner_dim)
        self.o = nn.Linear(attention_inner_dim, hidden_size)
        self.norm_q = InnerRMSNorm(attention_inner_dim, eps)
        self.norm_k = InnerRMSNorm(attention_inner_dim, eps)


class ExpertCrossAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        attention_inner_dim: int,
        num_heads: int,
        eps: float,
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.q = nn.Linear(hidden_size, attention_inner_dim)
        self.k = nn.Linear(hidden_size, attention_inner_dim)
        self.v = nn.Linear(hidden_size, attention_inner_dim)
        self.o = nn.Linear(attention_inner_dim, hidden_size)
        self.norm_q = InnerRMSNorm(attention_inner_dim, eps)
        self.norm_k = InnerRMSNorm(attention_inner_dim, eps)

    def forward(
        self,
        hidden: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        query = self.norm_q(self.q(hidden))
        key = self.norm_k(self.k(context))
        value = self.v(context)
        return self.o(
            _attention(
                query,
                key,
                value,
                num_heads=self.num_heads,
                attention_mask=None,
            )
        )


class PhysicalExpertBlock(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        ffn_dim: int,
        attention_inner_dim: int,
        num_heads: int,
        eps: float,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.self_attn = ExpertSelfAttention(
            self.hidden_size,
            attention_inner_dim,
            eps,
        )
        self.cross_attn = ExpertCrossAttention(
            self.hidden_size,
            attention_inner_dim,
            self.num_heads,
            eps,
        )
        self.norm1 = nn.LayerNorm(
            self.hidden_size,
            eps=eps,
            elementwise_affine=False,
        )
        self.norm2 = nn.LayerNorm(
            self.hidden_size,
            eps=eps,
            elementwise_affine=False,
        )
        self.norm3 = nn.LayerNorm(self.hidden_size, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_size, int(ffn_dim)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(ffn_dim), self.hidden_size),
        )
        self.modulation = nn.Parameter(
            torch.randn(1, 6, self.hidden_size) / self.hidden_size**0.5
        )

    def attention_io(
        self,
        hidden: torch.Tensor,
        time_modulation: torch.Tensor,
        frequencies: torch.Tensor,
    ):
        has_token_time = time_modulation.ndim == 4
        chunk_dim = 2 if has_token_time else 1
        base = self.modulation.to(
            device=time_modulation.device,
            dtype=time_modulation.dtype,
        )
        modulation = base + time_modulation
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = modulation.chunk(6, dim=chunk_dim)
        if has_token_time:
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            shift_mlp = shift_mlp.squeeze(2)
            scale_mlp = scale_mlp.squeeze(2)
            gate_mlp = gate_mlp.squeeze(2)

        attention_input = self.norm1(hidden) * (1.0 + scale_msa) + shift_msa
        query = self.self_attn.norm_q(self.self_attn.q(attention_input))
        key = self.self_attn.norm_k(self.self_attn.k(attention_input))
        value = self.self_attn.v(attention_input)
        query = _rope_apply(query, frequencies, self.num_heads)
        key = _rope_apply(key, frequencies, self.num_heads)
        return (
            query,
            key,
            value,
            hidden,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        )

    def post_attention(
        self,
        *,
        residual_hidden: torch.Tensor,
        mixed_attention: torch.Tensor,
        context: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
    ) -> torch.Tensor:
        hidden = residual_hidden + gate_msa * self.self_attn.o(mixed_attention)
        # ``fp32_shell`` deliberately restores the action expert's affine
        # norm3 parameters to FP32 after the surrounding Transformer is cast
        # to BF16. CUDA LayerNorm requires its activation and parameters to
        # share a dtype, so compute this one boundary norm in the parameter
        # dtype and immediately return to the core activation dtype before the
        # BF16 cross-attention projections. Preserve the original LayerNorm
        # call exactly when the dtypes already match, which is the path used by
        # every non-fp32-shell model.
        core_dtype = hidden.dtype
        norm3_dtype = self.norm3.weight.dtype
        if core_dtype == norm3_dtype:
            norm3_hidden = self.norm3(hidden)
        else:
            norm3_hidden = self.norm3(hidden.to(dtype=norm3_dtype)).to(
                dtype=core_dtype
            )
        hidden = hidden + self.cross_attn(
            norm3_hidden,
            context,
        )
        ffn_input = self.norm2(hidden) * (1.0 + scale_mlp) + shift_mlp
        return hidden + gate_mlp * self.ffn(ffn_input)


class CausalDINOActionLayer(nn.Module):
    def __init__(
        self,
        *,
        world_hidden_size: int,
        action_hidden_size: int,
        world_ffn_dim: int,
        action_ffn_dim: int,
        attention_inner_dim: int,
        num_heads: int,
        eps: float,
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.world = PhysicalExpertBlock(
            hidden_size=world_hidden_size,
            ffn_dim=world_ffn_dim,
            attention_inner_dim=attention_inner_dim,
            num_heads=num_heads,
            eps=eps,
        )
        self.action = PhysicalExpertBlock(
            hidden_size=action_hidden_size,
            ffn_dim=action_ffn_dim,
            attention_inner_dim=attention_inner_dim,
            num_heads=num_heads,
            eps=eps,
        )

    def forward(
        self,
        action_hidden: torch.Tensor,
        world_hidden: torch.Tensor,
        action_context: torch.Tensor,
        world_context: torch.Tensor,
        action_time_modulation: torch.Tensor,
        world_time_modulation: torch.Tensor,
        action_frequencies: torch.Tensor,
        world_frequencies: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        world_io = self.world.attention_io(
            world_hidden,
            world_time_modulation,
            world_frequencies,
        )
        action_io = self.action.attention_io(
            action_hidden,
            action_time_modulation,
            action_frequencies,
        )
        world_query, world_key, world_value = world_io[:3]
        action_query, action_key, action_value = action_io[:3]
        world_length = world_hidden.shape[1]

        mixed = _attention(
            torch.cat([world_query, action_query], dim=1),
            torch.cat([world_key, action_key], dim=1),
            torch.cat([world_value, action_value], dim=1),
            num_heads=self.num_heads,
            attention_mask=attention_mask,
        )
        world_hidden = self.world.post_attention(
            residual_hidden=world_io[3],
            mixed_attention=mixed[:, :world_length],
            context=world_context,
            gate_msa=world_io[4],
            shift_mlp=world_io[5],
            scale_mlp=world_io[6],
            gate_mlp=world_io[7],
        )
        action_hidden = self.action.post_attention(
            residual_hidden=action_io[3],
            mixed_attention=mixed[:, world_length:],
            context=action_context,
            gate_msa=action_io[4],
            shift_mlp=action_io[5],
            scale_mlp=action_io[6],
            gate_mlp=action_io[7],
        )
        return action_hidden, world_hidden


class ShiftedFlowScheduler:
    def __init__(
        self,
        *,
        num_train_timesteps: int,
        shift: float,
        eps: float = 1.0e-10,
    ) -> None:
        if num_train_timesteps <= 0:
            raise ValueError("num_train_timesteps must be positive")
        if shift <= 0:
            raise ValueError("flow shift must be positive")
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.eps = float(eps)
        self._y_min, self._weight_norm_const = self._weight_statistics()

    @staticmethod
    def _phi(value: torch.Tensor, shift: float) -> torch.Tensor:
        return shift * value / (1.0 + (shift - 1.0) * value)

    def _weight_statistics(self) -> tuple[float, float]:
        steps = self.num_train_timesteps
        grid = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)[:-1]
        timestep = self._phi(grid, self.shift) * float(steps)
        values = torch.exp(
            -2.0 * ((timestep - (steps / 2.0)) / float(steps)).square()
        )
        minimum = float(values.min())
        normalization = float((values - minimum).mean())
        return minimum, normalization

    def sample_training_t(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        uniform = torch.rand(batch_size, device=device, dtype=torch.float32)
        sigma = self._phi(uniform, self.shift)
        return (sigma * float(self.num_train_timesteps)).to(dtype=dtype)

    def add_noise(
        self,
        clean: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        sigma = (timestep / float(self.num_train_timesteps)).to(
            device=clean.device,
            dtype=clean.dtype,
        )
        sigma = sigma.view(-1, *([1] * (clean.ndim - 1)))
        return (1.0 - sigma) * clean + sigma * noise

    @staticmethod
    def training_target(
        clean: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        return noise - clean

    def training_weight(self, timestep: torch.Tensor) -> torch.Tensor:
        value = timestep.float()
        steps = float(self.num_train_timesteps)
        score = torch.exp(-2.0 * ((value - steps / 2.0) / steps).square())
        return (score - self._y_min) / (self._weight_norm_const + self.eps)

    def inference_schedule(
        self,
        num_steps: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if num_steps <= 0:
            raise ValueError("num_inference_timesteps must be positive")
        uniform = torch.linspace(
            1.0,
            0.0,
            num_steps + 1,
            device=device,
            dtype=torch.float32,
        )
        sigma = self._phi(uniform, self.shift)
        timesteps = sigma[:-1] * float(self.num_train_timesteps)
        deltas = sigma[1:] - sigma[:-1]
        return timesteps.to(dtype=dtype), deltas.to(dtype=dtype)

    @staticmethod
    def step(
        prediction: torch.Tensor,
        delta: torch.Tensor,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        delta = delta.to(device=sample.device, dtype=sample.dtype)
        return sample + prediction * delta


class DINOFlowHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        eps: float,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(
            hidden_size,
            eps=eps,
            elementwise_affine=False,
        )
        self.output = nn.Linear(hidden_size, output_size)
        self.modulation = nn.Parameter(
            torch.randn(1, 2, hidden_size) / hidden_size**0.5
        )

    def forward(
        self,
        hidden: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:
        shift, scale = (
            self.modulation.unsqueeze(0).to(
                device=time_embedding.device,
                dtype=time_embedding.dtype,
            )
            + time_embedding.unsqueeze(2)
        ).chunk(2, dim=2)
        return self.output(
            self.norm(hidden) * (1.0 + scale.squeeze(2)) + shift.squeeze(2)
        )


class CausalDINOActionMoT(nn.Module):
    """FastWAM-style clean-current/future/action mixed-attention MoT."""

    def __init__(
        self,
        *,
        planner_dim: int,
        world_dim: int,
        action_config,
        mot_config,
        current_world_grid: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.action_horizon = int(_cfg_get(action_config, "action_horizon", 16))
        self.action_dim = int(_cfg_get(action_config, "action_dim", 14))
        self.state_dim = int(_cfg_get(action_config, "state_dim", 0) or 0)
        self.world_dim = int(world_dim)
        self.action_precision_mode = str(
            _cfg_get(mot_config, "action_precision_mode", "inherit")
        ).strip().lower()
        if self.action_precision_mode not in {"inherit", "fp32_shell"}:
            raise ValueError(
                "action_precision_mode must be 'inherit' or 'fp32_shell', got "
                f"{self.action_precision_mode!r}"
            )
        self.interaction_mode = str(
            _cfg_get(mot_config, "interaction_mode", "base")
        ).lower()
        if self.interaction_mode not in {"base", "joint"}:
            raise ValueError("interaction_mode must be 'base' or 'joint'")
        self.world_attention_mask_mode = str(
            _cfg_get(
                mot_config,
                "world_attention_mask_mode",
                "first_frame_causal",
            )
        ).lower()
        if self.world_attention_mask_mode != "first_frame_causal":
            raise ValueError(
                "CausalDINOActionMoT requires "
                "world_attention_mask_mode='first_frame_causal'"
            )

        self.world_hidden_size = int(
            _cfg_get(mot_config, "world_hidden_size", 3072)
        )
        self.action_hidden_size = int(
            _cfg_get(mot_config, "action_hidden_size", 1024)
        )
        self.world_ffn_dim = int(_cfg_get(mot_config, "world_ffn_dim", 14336))
        self.action_ffn_dim = int(_cfg_get(mot_config, "action_ffn_dim", 4096))
        self.num_heads = int(_cfg_get(mot_config, "num_attention_heads", 24))
        self.head_dim = int(_cfg_get(mot_config, "attention_head_dim", 128))
        self.attention_inner_dim = self.num_heads * self.head_dim
        self.num_layers = int(_cfg_get(mot_config, "num_layers", 30))
        self.layerwise_planner_coupling = bool(
            _cfg_get(mot_config, "layerwise_planner_coupling", False)
        )
        # Historical state-conditioned checkpoints are reconstructed from their
        # saved state_dim. Current layer-wise RoboDojo recipes set state_dim=0,
        # so neither expert receives a proprioceptive cross-attention token.
        # Non-layerwise 50k/80k checkpoints still append their saved state token
        # to both planner branches and therefore keep strict load/inference
        # compatibility without a new user-facing condition switch.
        self._legacy_world_condition_on_state = (
            self.state_dim > 0 and not self.layerwise_planner_coupling
        )
        self.norm_eps = float(_cfg_get(mot_config, "norm_eps", 1.0e-6))
        self.time_frequency_dim = int(
            _cfg_get(mot_config, "time_frequency_dim", 256)
        )
        self.grid_height = int(_cfg_get(mot_config, "world_grid_height", 12))
        self.grid_width = int(_cfg_get(mot_config, "world_grid_width", 10))
        # ``world_tokens`` is retained as the legacy public name; it denotes
        # only the future tokens that receive noise and a world loss.
        self.world_tokens = self.grid_height * self.grid_width
        self.future_world_tokens = self.world_tokens
        if current_world_grid is None:
            self.current_grid_height = self.grid_height
            self.current_grid_width = self.grid_width
        else:
            if len(current_world_grid) != 2:
                raise ValueError(
                    "current_world_grid must contain exactly (height, width)"
                )
            self.current_grid_height = int(current_world_grid[0])
            self.current_grid_width = int(current_world_grid[1])
        self.current_world_tokens = (
            self.current_grid_height * self.current_grid_width
        )
        self.multires_world_input = (
            self.current_grid_height != self.grid_height
            or self.current_grid_width != self.grid_width
        )
        if self.multires_world_input and (
            self.current_grid_height < self.grid_height
            or self.current_grid_width < self.grid_width
            or self.current_grid_height % self.grid_height
            or self.current_grid_width % self.grid_width
        ):
            raise ValueError(
                "the clean current grid must be an integer-resolution "
                "upsampling of the future grid, got current="
                f"{(self.current_grid_height, self.current_grid_width)}, "
                f"future={(self.grid_height, self.grid_width)}"
            )
        self.checkpoint_contract_version = (
            (
                "layerwise_query_only_multires_world_v3"
                if self.layerwise_planner_coupling
                else "legacy_planner_multires_world_v3"
            )
            if self.multires_world_input
            else (
                "layerwise_query_only_world_v2"
                if self.layerwise_planner_coupling
                else "legacy_shared_context_state_world_v1"
            )
        )
        self.inference_steps = int(
            _cfg_get(mot_config, "num_inference_timesteps", 20)
        )
        # This model predicts flow velocity directly, so inference-time RTC
        # can guide the clean-action estimate without retraining.
        self.rtc_guidance_supported = True
        self.gradient_checkpointing = bool(
            _cfg_get(mot_config, "enable_gradient_checkpointing", True)
        )
        # Keep legacy checkpoints backward compatible: historical causal-MoT
        # configs omitted these fields and directly predicted scheduler
        # velocity (noise - clean) with one noise draw.
        self.action_prediction_type = str(
            _cfg_get(mot_config, "action_prediction_type", "velocity")
        ).lower()
        self.action_velocity_target = str(
            _cfg_get(
                mot_config,
                "action_velocity_target",
                "noise_minus_clean",
            )
        ).lower()
        self.jit_t_eps = float(_cfg_get(mot_config, "jit_t_eps", 0.05))
        self.repeated_diffusion_steps = int(
            _cfg_get(mot_config, "repeated_diffusion_steps", 1)
        )
        self.action_loss_weight = float(
            _cfg_get(mot_config, "action_loss_weight", 1.0)
        )
        self.world_loss_weight = float(
            _cfg_get(mot_config, "world_loss_weight", 1.0)
        )
        if min(
            self.action_horizon,
            self.action_dim,
            self.world_hidden_size,
            self.action_hidden_size,
            self.world_ffn_dim,
            self.action_ffn_dim,
            self.num_heads,
            self.head_dim,
            self.num_layers,
            self.grid_height,
            self.grid_width,
            self.current_grid_height,
            self.current_grid_width,
            self.inference_steps,
            self.repeated_diffusion_steps,
        ) <= 0:
            raise ValueError("all architecture sizes must be positive")
        if self.head_dim % 2:
            raise ValueError("attention_head_dim must be even for RoPE")
        if self.action_prediction_type not in {"velocity", "jit_x"}:
            raise ValueError(
                "action_prediction_type must be 'velocity' or 'jit_x', got "
                f"{self.action_prediction_type!r}"
            )
        if self.action_velocity_target not in {
            "clean_minus_noise",
            "noise_minus_clean",
        }:
            raise ValueError(
                "action_velocity_target must be 'clean_minus_noise' or "
                f"'noise_minus_clean', got {self.action_velocity_target!r}"
            )
        if self.jit_t_eps <= 0:
            raise ValueError("jit_t_eps must be positive")
        if min(self.action_loss_weight, self.world_loss_weight) < 0:
            raise ValueError(
                "action_loss_weight and world_loss_weight must be non-negative"
            )

        self.world_input = nn.Linear(self.world_dim, self.world_hidden_size)
        self.action_input = nn.Linear(self.action_dim, self.action_hidden_size)

        def make_context(hidden_size: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(int(planner_dim), hidden_size),
                nn.GELU(approximate="tanh"),
                nn.Linear(hidden_size, hidden_size),
            )

        if self.layerwise_planner_coupling:
            self.world_context = nn.ModuleList(
                [make_context(self.world_hidden_size) for _ in range(self.num_layers)]
            )
            self.action_context = nn.ModuleList(
                [make_context(self.action_hidden_size) for _ in range(self.num_layers)]
            )
        else:
            self.world_context = make_context(self.world_hidden_size)
            self.action_context = make_context(self.action_hidden_size)
        # Compatibility-only for checkpoints/configs that were trained with
        # proprioception. New query-only layer-wise recipes instantiate no state
        # projection at all.
        self.state_to_planner = (
            nn.Linear(self.state_dim, int(planner_dim))
            if self.state_dim > 0
            else None
        )
        self.world_time_embedding = nn.Sequential(
            nn.Linear(self.time_frequency_dim, self.world_hidden_size),
            nn.SiLU(),
            nn.Linear(self.world_hidden_size, self.world_hidden_size),
        )
        self.world_time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.world_hidden_size, 6 * self.world_hidden_size),
        )
        self.action_time_embedding = nn.Sequential(
            nn.Linear(self.time_frequency_dim, self.action_hidden_size),
            nn.SiLU(),
            nn.Linear(self.action_hidden_size, self.action_hidden_size),
        )
        self.action_time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.action_hidden_size, 6 * self.action_hidden_size),
        )
        self.layers = nn.ModuleList(
            [
                CausalDINOActionLayer(
                    world_hidden_size=self.world_hidden_size,
                    action_hidden_size=self.action_hidden_size,
                    world_ffn_dim=self.world_ffn_dim,
                    action_ffn_dim=self.action_ffn_dim,
                    attention_inner_dim=self.attention_inner_dim,
                    num_heads=self.num_heads,
                    eps=self.norm_eps,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.action_output = nn.Linear(self.action_hidden_size, self.action_dim)
        self.world_output = DINOFlowHead(
            self.world_hidden_size,
            self.world_dim,
            self.norm_eps,
        )

        action_train_shift = float(
            _cfg_get(mot_config, "action_train_shift", 5.0)
        )
        action_infer_shift = float(
            _cfg_get(mot_config, "action_infer_shift", 5.0)
        )
        world_train_shift = float(
            _cfg_get(mot_config, "world_train_shift", 5.0)
        )
        world_infer_shift = float(
            _cfg_get(mot_config, "world_infer_shift", 5.0)
        )
        action_train_steps = int(
            _cfg_get(mot_config, "action_num_train_timesteps", 1000)
        )
        world_train_steps = int(
            _cfg_get(mot_config, "world_num_train_timesteps", 1000)
        )
        self.train_action_scheduler = ShiftedFlowScheduler(
            num_train_timesteps=action_train_steps,
            shift=action_train_shift,
        )
        self.infer_action_scheduler = ShiftedFlowScheduler(
            num_train_timesteps=action_train_steps,
            shift=action_infer_shift,
        )
        self.train_world_scheduler = ShiftedFlowScheduler(
            num_train_timesteps=world_train_steps,
            shift=world_train_shift,
        )
        self.infer_world_scheduler = ShiftedFlowScheduler(
            num_train_timesteps=world_train_steps,
            shift=world_infer_shift,
        )

        # Preserve the exact complex128 RoPE path used by released non-layerwise
        # 50k/80k checkpoints. New layer-wise recipes train and infer with the
        # faster standard complex64 path.
        rope_real_dtype = (
            torch.float32
            if self.layerwise_planner_coupling
            else torch.float64
        )
        action_frequencies = _precompute_freqs_cis(
            self.head_dim,
            end=max(1024, self.action_horizon),
            real_dtype=rope_real_dtype,
        )
        rope_cache_end = max(
            1024,
            self.current_grid_height,
            self.current_grid_width,
            self.grid_height,
            self.grid_width,
        )
        height, width = _precompute_freqs_cis_2d(
            self.head_dim,
            end=rope_cache_end,
            real_dtype=rope_real_dtype,
        )

        def spatial_grid_frequencies(
            grid_height: int,
            grid_width: int,
        ) -> torch.Tensor:
            return torch.cat(
                [
                    height[:grid_height]
                    .view(grid_height, 1, -1)
                    .expand(grid_height, grid_width, -1),
                    width[:grid_width]
                    .view(1, grid_width, -1)
                    .expand(grid_height, grid_width, -1),
                ],
                dim=-1,
            ).reshape(grid_height * grid_width, 1, -1)

        current_frequencies = spatial_grid_frequencies(
            self.current_grid_height,
            self.current_grid_width,
        )
        future_frequencies = spatial_grid_frequencies(
            self.grid_height,
            self.grid_width,
        )
        world_frequencies = torch.cat(
            [current_frequencies, future_frequencies],
            dim=0,
        )
        self.world_frame_embedding = nn.Parameter(
            torch.empty(1, 2, self.world_hidden_size)
        )
        nn.init.normal_(self.world_frame_embedding, std=0.02)
        # Keep complex RoPE caches as plain CPU tensors, as in the source
        # architecture.  Module-wide BF16 conversion must never discard their
        # imaginary component; _rope_apply moves them to the active device.
        self.action_frequencies = action_frequencies[: self.action_horizon].view(
            self.action_horizon,
            1,
            -1,
        )
        self.world_frequencies = world_frequencies
        self.register_buffer(
            "physical_attention_mask",
            self._build_attention_mask(),
            persistent=False,
        )
        self._restore_action_precision_policy()

    def _build_attention_mask(self) -> torch.Tensor:
        current = self.current_world_tokens
        future_end = current + self.world_tokens
        total = future_end + self.action_horizon
        mask = torch.zeros(total, total, dtype=torch.bool)
        # z0/zf video-style first-frame-causal block.
        mask[:future_end, :future_end] = True
        mask[:current, current:future_end] = False
        # Action tokens always mix with each other.
        mask[future_end:, future_end:] = True
        # Base mode reads z0 only; joint mode reads z0 and zf.
        world_key_end = current if self.interaction_mode == "base" else future_end
        mask[future_end:, :world_key_end] = True
        return mask

    @property
    def uses_fp32_action_shell(self) -> bool:
        return self.action_precision_mode == "fp32_shell"

    def _restore_action_precision_policy(self) -> None:
        """Restore opt-in fp32 parameters after a parent module-wide cast.

        Training and deployment both cast the complete framework to bf16.  A
        configuration-owned policy must therefore survive recursive
        ``module.to(torch.bfloat16)`` calls; otherwise checkpoint reload would
        silently erase the intended precision boundary.
        """

        if not self.uses_fp32_action_shell:
            return
        fp32_modules = [
            self.action_input,
            self.action_time_embedding,
            self.action_time_projection,
            self.action_output,
        ]
        if self.state_to_planner is not None:
            fp32_modules.append(self.state_to_planner)
        for layer in self.layers:
            fp32_modules.extend(
                [
                    layer.action.norm3,
                    layer.action.self_attn.norm_q,
                    layer.action.self_attn.norm_k,
                    layer.action.cross_attn.norm_q,
                    layer.action.cross_attn.norm_k,
                ]
            )
        for module in fp32_modules:
            module.float()

    def _apply(self, fn, recurse: bool = True):
        result = super()._apply(fn, recurse=recurse)
        self._restore_action_precision_policy()
        return result

    @staticmethod
    def _autocast_disabled(device: torch.device):
        if device.type in {"cpu", "cuda"}:
            return torch.autocast(device_type=device.type, enabled=False)
        return nullcontext()

    def _device_dtype(self) -> tuple[torch.device, torch.dtype]:
        # This is the physical Transformer dtype.  ``action_input`` is an fp32
        # boundary module in fp32_shell mode and can no longer serve as the
        # core-dtype anchor.
        weight = self.layers[0].action.self_attn.q.weight
        return weight.device, weight.dtype

    def action_flow_dtype(self) -> torch.dtype:
        """Dtype for action targets/noise/time/velocity and Euler state."""

        if self.uses_fp32_action_shell:
            return torch.float32
        return self._device_dtype()[1]

    def _project_state_token(
        self,
        current_state: torch.Tensor,
        *,
        device: torch.device,
        core_dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.state_to_planner is None:
            raise RuntimeError("state projection requested while state_dim is zero")
        if self.uses_fp32_action_shell:
            with self._autocast_disabled(device):
                token = self.state_to_planner(
                    current_state.to(device=device, dtype=torch.float32)
                )
            return token.to(dtype=core_dtype)
        return self.state_to_planner(
            current_state.to(device=device, dtype=core_dtype)
        )

    def _project_action_input(
        self,
        noisy_action: torch.Tensor,
        *,
        device: torch.device,
        core_dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.uses_fp32_action_shell:
            with self._autocast_disabled(device):
                hidden = self.action_input(
                    noisy_action.to(device=device, dtype=torch.float32)
                )
            return hidden.to(dtype=core_dtype)
        return self.action_input(
            noisy_action.to(device=device, dtype=core_dtype)
        )

    def _project_action_output(self, action_hidden: torch.Tensor) -> torch.Tensor:
        if self.uses_fp32_action_shell:
            with self._autocast_disabled(action_hidden.device):
                return self.action_output(action_hidden.float())
        return self.action_output(action_hidden)

    @staticmethod
    def _repeat_batch(
        tensor: torch.Tensor
        | list[torch.Tensor]
        | tuple[torch.Tensor, ...]
        | None,
        repeats: int,
    ) -> (
        torch.Tensor
        | list[torch.Tensor]
        | tuple[torch.Tensor, ...]
        | None
    ):
        if tensor is None or int(repeats) == 1:
            return tensor
        if isinstance(tensor, list):
            return [CausalDINOActionMoT._repeat_batch(value, repeats) for value in tensor]
        if isinstance(tensor, tuple):
            return tuple(
                CausalDINOActionMoT._repeat_batch(value, repeats) for value in tensor
            )
        return tensor.repeat(int(repeats), *([1] * (tensor.ndim - 1)))

    def _normalize_plan_layers(
        self,
        plan: torch.Tensor
        | list[torch.Tensor]
        | tuple[torch.Tensor, ...],
        *,
        name: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> list[torch.Tensor]:
        if self.layerwise_planner_coupling:
            if not isinstance(plan, (list, tuple)):
                raise TypeError(
                    f"{name} must be a list/tuple for layer-wise planner coupling"
                )
            if len(plan) != self.num_layers:
                raise ValueError(
                    f"{name} layer count must match physical layers: "
                    f"got={len(plan)}, expected={self.num_layers}"
                )
            layers = [value.to(device=device, dtype=dtype) for value in plan]
        else:
            if isinstance(plan, (list, tuple)):
                raise TypeError(
                    f"{name} must be a tensor when layer-wise coupling is disabled"
                )
            value = plan.to(device=device, dtype=dtype)
            layers = [value] * self.num_layers
        batch, length = layers[0].shape[:2]
        if any(tuple(layer.shape[:2]) != (batch, length) for layer in layers):
            raise ValueError(f"{name} layers must share batch and token dimensions")
        return layers

    def _project_plan_layers(
        self,
        plans: list[torch.Tensor],
        projectors: nn.Module,
    ) -> list[torch.Tensor]:
        if self.layerwise_planner_coupling:
            if not isinstance(projectors, nn.ModuleList):
                raise TypeError("layer-wise planner coupling requires per-layer projectors")
            return [
                projector(plan)
                for projector, plan in zip(projectors, plans, strict=True)
            ]
        return [projectors(plans[0])] * self.num_layers

    @staticmethod
    def _action_sigma(
        timestep: torch.Tensor,
        reference: torch.Tensor,
        *,
        scheduler: ShiftedFlowScheduler,
    ) -> torch.Tensor:
        timestep = torch.as_tensor(
            timestep,
            device=reference.device,
            dtype=reference.dtype,
        )
        if timestep.ndim == 0:
            timestep = timestep.expand(reference.shape[0])
        if timestep.ndim != 1 or timestep.shape[0] != reference.shape[0]:
            raise ValueError(
                "action timestep must be scalar or have shape [batch], got "
                f"{tuple(timestep.shape)} for batch={reference.shape[0]}"
            )
        sigma = timestep / float(scheduler.num_train_timesteps)
        return sigma.view(-1, *([1] * (reference.ndim - 1)))

    def _action_prediction_to_velocity(
        self,
        prediction: torch.Tensor,
        noisy_action: torch.Tensor,
        timestep: torch.Tensor,
        *,
        scheduler: ShiftedFlowScheduler,
    ) -> torch.Tensor:
        """Convert the configured action prediction into scheduler velocity.

        ``ShiftedFlowScheduler`` parameterizes the path by noise fraction
        ``sigma = 1 - tau``:

            a_sigma = (1 - sigma) * a_clean + sigma * noise

        and integrates from sigma=1 to sigma=0, so its velocity is
        ``d a / d sigma = noise - a_clean``.  A model configured to predict
        denoising velocity ``a_clean - noise`` is negated before the scheduler
        step. Legacy JiT-x predicts ``a_clean`` and is converted to velocity as
        ``(a_sigma - a_pred) / max(sigma, eps)``.
        """

        if self.action_prediction_type == "velocity":
            return (
                -prediction
                if self.action_velocity_target == "clean_minus_noise"
                else prediction
            )
        if prediction.shape != noisy_action.shape:
            raise ValueError(
                "action prediction and noisy action must have identical shapes, "
                f"got prediction={tuple(prediction.shape)} "
                f"noisy={tuple(noisy_action.shape)}"
            )
        sigma = self._action_sigma(
            timestep,
            prediction,
            scheduler=scheduler,
        )
        noisy_action = noisy_action.to(
            device=prediction.device,
            dtype=prediction.dtype,
        )
        return (noisy_action - prediction) / sigma.clamp_min(self.jit_t_eps)

    def _prepare_conditions(
        self,
        *,
        action_plan: torch.Tensor
        | list[torch.Tensor]
        | tuple[torch.Tensor, ...],
        world_plan: torch.Tensor
        | list[torch.Tensor]
        | tuple[torch.Tensor, ...],
        current_world: torch.Tensor,
        current_state: torch.Tensor | None,
    ):
        device, dtype = self._device_dtype()
        action_plan_layers = self._normalize_plan_layers(
            action_plan,
            name="action_plan",
            device=device,
            dtype=dtype,
        )
        world_plan_layers = self._normalize_plan_layers(
            world_plan,
            name="world_plan",
            device=device,
            dtype=dtype,
        )
        current_world = current_world.to(device=device, dtype=dtype)
        if (
            current_world.ndim != 3
            or current_world.shape[1] != self.current_world_tokens
            or current_world.shape[2] != self.world_dim
        ):
            raise ValueError(
                "current DINO must have shape "
                f"[B,{self.current_world_tokens},{self.world_dim}], "
                f"got {tuple(current_world.shape)}"
            )
        if self.state_to_planner is not None:
            if current_state is None:
                raise ValueError("current_state is required when state_dim > 0")
            if current_state.ndim == 2:
                current_state = current_state[:, None]
            state_token = self._project_state_token(
                current_state[:, :1],
                device=device,
                core_dtype=dtype,
            )
            action_plan_layers = [
                torch.cat([plan, state_token], dim=1)
                for plan in action_plan_layers
            ]
            if self._legacy_world_condition_on_state:
                world_plan_layers = [
                    torch.cat([plan, state_token], dim=1)
                    for plan in world_plan_layers
                ]
        elif current_state is not None:
            raise ValueError("current_state was provided but state_dim is zero")
        return (
            action_plan_layers,
            world_plan_layers,
            current_world,
        )

    def _time_features(
        self,
        action_time: torch.Tensor,
        world_time: torch.Tensor,
    ):
        batch = action_time.shape[0]
        _, core_dtype = self._device_dtype()
        action_sinusoid = _sinusoidal_embedding_1d(
            self.time_frequency_dim,
            action_time,
        )
        if self.uses_fp32_action_shell:
            with self._autocast_disabled(action_time.device):
                action_time_embedding = self.action_time_embedding(
                    action_sinusoid.float()
                )
                action_time_modulation = self.action_time_projection(
                    action_time_embedding
                ).unflatten(1, (6, self.action_hidden_size))
            action_time_modulation = action_time_modulation.to(dtype=core_dtype)
        else:
            action_time_embedding = self.action_time_embedding(action_sinusoid)
            action_time_modulation = self.action_time_projection(
                action_time_embedding
            ).unflatten(1, (6, self.action_hidden_size))

        token_times = torch.cat(
            [
                torch.zeros(
                    batch,
                    self.current_world_tokens,
                    device=world_time.device,
                    dtype=world_time.dtype,
                ),
                world_time[:, None].expand(-1, self.world_tokens),
            ],
            dim=1,
        )
        world_time_embedding = self.world_time_embedding(
            _sinusoidal_embedding_1d(
                self.time_frequency_dim,
                token_times.reshape(-1),
            )
        ).reshape(
            batch,
            self.current_world_tokens + self.world_tokens,
            self.world_hidden_size,
        )
        world_time_modulation = self.world_time_projection(
            world_time_embedding
        ).unflatten(2, (6, self.world_hidden_size))
        return (
            action_time_embedding,
            action_time_modulation,
            world_time_embedding,
            world_time_modulation,
        )

    def _predict(
        self,
        *,
        action_plan: torch.Tensor,
        world_plan: torch.Tensor,
        current_world: torch.Tensor,
        noisy_action: torch.Tensor,
        noisy_world: torch.Tensor,
        action_time: torch.Tensor,
        world_time: torch.Tensor,
        action_is_pad: torch.Tensor | None,
        current_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del action_is_pad
        (
            action_plan,
            world_plan,
            current_world,
        ) = self._prepare_conditions(
            action_plan=action_plan,
            world_plan=world_plan,
            current_world=current_world,
            current_state=current_state,
        )
        device, dtype = self._device_dtype()
        action_dtype = self.action_flow_dtype()
        noisy_action = noisy_action.to(device=device, dtype=action_dtype)
        noisy_world = noisy_world.to(device=device, dtype=dtype)
        action_time = action_time.to(device=device, dtype=action_dtype)
        world_time = world_time.to(device=device, dtype=dtype)
        expected_noisy_world = (
            current_world.shape[0],
            self.world_tokens,
            self.world_dim,
        )
        if noisy_world.shape != expected_noisy_world:
            raise ValueError(
                "noisy future DINO must have shape "
                f"{expected_noisy_world}, got {tuple(noisy_world.shape)}"
            )

        (
            _action_time_embedding,
            action_time_modulation,
            world_time_embedding,
            world_time_modulation,
        ) = self._time_features(action_time, world_time)
        action_hidden = self._project_action_input(
            noisy_action,
            device=device,
            core_dtype=dtype,
        )
        world_hidden = self.world_input(
            torch.cat([current_world, noisy_world], dim=1)
        )
        frame_embedding = torch.cat(
            [
                self.world_frame_embedding[:, :1].expand(
                    -1,
                    self.current_world_tokens,
                    -1,
                ),
                self.world_frame_embedding[:, 1:2].expand(
                    -1,
                    self.world_tokens,
                    -1,
                ),
            ],
            dim=1,
        )
        world_hidden = world_hidden + frame_embedding
        action_contexts = self._project_plan_layers(
            action_plan,
            self.action_context,
        )
        world_contexts = self._project_plan_layers(
            world_plan,
            self.world_context,
        )
        attention_mask = self.physical_attention_mask.to(device=device)
        action_frequencies = self.action_frequencies.to(device=device)
        world_frequencies = self.world_frequencies.to(device=device)

        for (
            layer,
            action_context,
            world_context,
        ) in zip(
            self.layers,
            action_contexts,
            world_contexts,
            strict=True,
        ):
            if (
                self.gradient_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                action_hidden, world_hidden = checkpoint(
                    layer,
                    action_hidden,
                    world_hidden,
                    action_context,
                    world_context,
                    action_time_modulation,
                    world_time_modulation,
                    action_frequencies,
                    world_frequencies,
                    attention_mask,
                    use_reentrant=False,
                )
            else:
                action_hidden, world_hidden = layer(
                    action_hidden,
                    world_hidden,
                    action_context,
                    world_context,
                    action_time_modulation,
                    world_time_modulation,
                    action_frequencies,
                    world_frequencies,
                    attention_mask,
                )
        action_prediction = self._project_action_output(action_hidden)
        future_hidden = world_hidden[:, self.current_world_tokens :]
        future_time_embedding = world_time_embedding[
            :,
            self.current_world_tokens :,
        ]
        world_prediction = self.world_output(
            future_hidden,
            future_time_embedding,
        )
        return action_prediction, world_prediction

    def _prepare_action_base_cache(
        self,
        *,
        action_plan: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
        world_plan: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
        current_world: torch.Tensor,
        current_state: torch.Tensor | None = None,
    ) -> _BaseActionCache:
        """Precompute the base policy's clean current-world layer trajectory.

        Base-mode action tokens can read only the clean current-world tokens,
        while world tokens cannot read action.  Consequently every world's
        per-layer K/V and planner projection is invariant across action
        denoising steps and can be computed once per replan.
        """
        if self.interaction_mode != "base":
            raise RuntimeError(
                "_prepare_action_base_cache requires interaction_mode='base'"
            )
        (
            action_plan,
            world_plan,
            current_world,
        ) = self._prepare_conditions(
            action_plan=action_plan,
            world_plan=world_plan,
            current_world=current_world,
            current_state=current_state,
        )
        device, dtype = self._device_dtype()
        batch = current_world.shape[0]
        current_times = torch.zeros(
            batch,
            self.current_world_tokens,
            device=device,
            dtype=dtype,
        )
        current_time_embedding = self.world_time_embedding(
            _sinusoidal_embedding_1d(
                self.time_frequency_dim,
                current_times.reshape(-1),
            )
        ).reshape(
            batch,
            self.current_world_tokens,
            self.world_hidden_size,
        )
        current_time_modulation = self.world_time_projection(
            current_time_embedding
        ).unflatten(2, (6, self.world_hidden_size))

        current_hidden = self.world_input(current_world)
        current_hidden = current_hidden + self.world_frame_embedding[:, :1]
        action_contexts = tuple(
            self._project_plan_layers(action_plan, self.action_context)
        )
        world_contexts = self._project_plan_layers(world_plan, self.world_context)
        action_frequencies = self.action_frequencies.to(device=device)
        current_frequencies = self.world_frequencies[
            : self.current_world_tokens
        ].to(device=device)

        world_keys = []
        world_values = []
        for layer, world_context in zip(
            self.layers,
            world_contexts,
            strict=True,
        ):
            world_io = layer.world.attention_io(
                current_hidden,
                current_time_modulation,
                current_frequencies,
            )
            world_query, world_key, world_value = world_io[:3]
            world_attention = _attention(
                world_query,
                world_key,
                world_value,
                num_heads=self.num_heads,
                attention_mask=None,
            )
            current_hidden = layer.world.post_attention(
                residual_hidden=world_io[3],
                mixed_attention=world_attention,
                context=world_context,
                gate_msa=world_io[4],
                shift_mlp=world_io[5],
                scale_mlp=world_io[6],
                gate_mlp=world_io[7],
            )
            world_keys.append(world_key)
            world_values.append(world_value)

        return _BaseActionCache(
            batch_size=batch,
            action_contexts=action_contexts,
            world_keys=tuple(world_keys),
            world_values=tuple(world_values),
            action_frequencies=action_frequencies,
        )

    def _predict_action_base(
        self,
        *,
        action_plan: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
        world_plan: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
        current_world: torch.Tensor,
        noisy_action: torch.Tensor,
        action_time: torch.Tensor,
        current_state: torch.Tensor | None = None,
        cache: _BaseActionCache | None = None,
    ) -> torch.Tensor:
        """Predict only action, reusing current-world features when available."""
        if self.interaction_mode != "base":
            raise RuntimeError("_predict_action_base requires interaction_mode='base'")
        if cache is None:
            cache = self._prepare_action_base_cache(
                action_plan=action_plan,
                world_plan=world_plan,
                current_world=current_world,
                current_state=current_state,
            )
        device, dtype = self._device_dtype()
        action_dtype = self.action_flow_dtype()
        noisy_action = noisy_action.to(device=device, dtype=action_dtype)
        action_time = action_time.to(device=device, dtype=action_dtype)
        if action_time.ndim == 0:
            action_time = action_time.expand(noisy_action.shape[0])
        if noisy_action.shape[0] != cache.batch_size or action_time.shape != (
            cache.batch_size,
        ):
            raise ValueError(
                "base action cache batch mismatch: "
                f"cache={cache.batch_size}, noisy_action={noisy_action.shape[0]}, "
                f"action_time={tuple(action_time.shape)}"
            )

        action_sinusoid = _sinusoidal_embedding_1d(
            self.time_frequency_dim,
            action_time,
        )
        if self.uses_fp32_action_shell:
            with self._autocast_disabled(device):
                action_time_embedding = self.action_time_embedding(
                    action_sinusoid.float()
                )
                action_time_modulation = self.action_time_projection(
                    action_time_embedding
                ).unflatten(1, (6, self.action_hidden_size))
            action_time_modulation = action_time_modulation.to(dtype=dtype)
        else:
            action_time_embedding = self.action_time_embedding(action_sinusoid)
            action_time_modulation = self.action_time_projection(
                action_time_embedding
            ).unflatten(1, (6, self.action_hidden_size))
        action_hidden = self._project_action_input(
            noisy_action,
            device=device,
            core_dtype=dtype,
        )

        for (
            layer,
            action_context,
            world_key,
            world_value,
        ) in zip(
            self.layers,
            cache.action_contexts,
            cache.world_keys,
            cache.world_values,
            strict=True,
        ):
            action_io = layer.action.attention_io(
                action_hidden,
                action_time_modulation,
                cache.action_frequencies,
            )
            action_query, action_key, action_value = action_io[:3]
            action_attention = _attention(
                action_query,
                torch.cat([world_key, action_key], dim=1),
                torch.cat([world_value, action_value], dim=1),
                num_heads=self.num_heads,
                attention_mask=None,
            )
            action_hidden = layer.action.post_attention(
                residual_hidden=action_io[3],
                mixed_attention=action_attention,
                context=action_context,
                gate_msa=action_io[4],
                shift_mlp=action_io[5],
                scale_mlp=action_io[6],
                gate_mlp=action_io[7],
            )
        return self._project_action_output(action_hidden)

    def forward_train(
        self,
        *,
        action_plan: torch.Tensor,
        world_plan: torch.Tensor,
        current_world: torch.Tensor,
        target_action: torch.Tensor,
        target_world: torch.Tensor,
        action_is_pad: torch.Tensor | None,
        future_valid: torch.Tensor,
        current_state: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        device, dtype = self._device_dtype()
        action_dtype = self.action_flow_dtype()
        target_action = target_action.to(device=device, dtype=action_dtype)
        target_world = target_world.to(device=device, dtype=dtype)
        if target_world.ndim != 3 or target_world.shape[1] != self.world_tokens:
            raise ValueError(
                "target future DINO must have shape "
                f"[B,{self.world_tokens},{self.world_dim}], got {tuple(target_world.shape)}"
            )
        repeats = self.repeated_diffusion_steps
        action_plan = self._repeat_batch(action_plan, repeats)
        world_plan = self._repeat_batch(world_plan, repeats)
        current_world = self._repeat_batch(current_world, repeats)
        target_action = self._repeat_batch(target_action, repeats)
        target_world = self._repeat_batch(target_world, repeats)
        current_state = self._repeat_batch(current_state, repeats)
        action_is_pad = self._repeat_batch(action_is_pad, repeats)
        future_valid = self._repeat_batch(future_valid, repeats)
        batch = target_action.shape[0]
        action_noise = torch.randn_like(target_action)
        world_noise = torch.randn_like(target_world)
        action_time = self.train_action_scheduler.sample_training_t(
            batch,
            device=device,
            dtype=action_dtype,
        )
        world_time = self.train_world_scheduler.sample_training_t(
            batch,
            device=device,
            dtype=dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(
            target_action,
            action_noise,
            action_time,
        )
        noisy_world = self.train_world_scheduler.add_noise(
            target_world,
            world_noise,
            world_time,
        )
        predicted_action, predicted_world = self._predict(
            action_plan=action_plan,
            world_plan=world_plan,
            current_world=current_world,
            noisy_action=noisy_action,
            noisy_world=noisy_world,
            action_time=action_time,
            world_time=world_time,
            action_is_pad=action_is_pad,
            current_state=current_state,
        )

        predicted_action_velocity = self._action_prediction_to_velocity(
            predicted_action,
            noisy_action,
            action_time,
            scheduler=self.train_action_scheduler,
        )
        target_action_velocity = (
            self._action_prediction_to_velocity(
                target_action,
                noisy_action,
                action_time,
                scheduler=self.train_action_scheduler,
            )
            if self.action_prediction_type == "jit_x"
            else self.train_action_scheduler.training_target(
                target_action,
                action_noise,
            )
        )
        action_error = (
            predicted_action_velocity.float()
            - target_action_velocity.float()
        ).square()
        action_per_step = action_error.mean(dim=-1)
        if action_is_pad is not None:
            valid_action = (~action_is_pad.to(device=device, dtype=torch.bool)).to(
                action_per_step.dtype
            )
            action_per_sample = (
                action_per_step * valid_action
            ).sum(dim=1) / valid_action.sum(dim=1).clamp_min(1.0)
        else:
            action_per_sample = action_per_step.mean(dim=1)
        action_weight = self.train_action_scheduler.training_weight(
            action_time
        ).to(device=device, dtype=action_per_sample.dtype)
        action_per_sample = action_per_sample * action_weight
        action_loss = action_per_sample.mean()

        world_per_sample = (
            predicted_world.float()
            - self.train_world_scheduler.training_target(
                target_world,
                world_noise,
            ).float()
        ).square().mean(dim=(1, 2))
        world_weight = self.train_world_scheduler.training_weight(
            world_time
        ).to(device=device, dtype=world_per_sample.dtype)
        world_per_sample = world_per_sample * world_weight
        valid_world = (
            future_valid.to(device=device).reshape(batch, -1)[:, 0] > 0.5
        )
        world_loss = (
            world_per_sample[valid_world].mean()
            if bool(valid_world.any())
            else world_per_sample.sum() * 0.0
        )
        action_objective = self.action_loss_weight * action_loss
        world_objective = self.world_loss_weight * world_loss
        total = action_objective + world_objective
        return {
            "loss": total,
            # Keep the weighted, graph-carrying objectives available to the
            # trainer's opt-in shared-VLM interface-gradient diagnostic.
            # ``autograd.grad`` is taken only with respect to the planner
            # interface tensors, so these values never populate/alter .grad.
            "action_objective": action_objective,
            "world_objective": world_objective,
            "action_loss_raw": action_loss.detach(),
            "world_loss_raw": world_loss.detach(),
        }

    @staticmethod
    def rtc_prefix_weights(
        *,
        inference_delay: int,
        execution_horizon: int,
        total_horizon: int,
        schedule: str,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Build the LeRobot/RTC soft prefix mask for one action chunk."""

        if total_horizon <= 0:
            raise ValueError("RTC total_horizon must be positive")
        if inference_delay < 0:
            raise ValueError("RTC inference_delay must be non-negative")
        if execution_horizon < 0:
            raise ValueError("RTC execution_horizon must be non-negative")
        schedule = str(schedule).strip().lower()
        if schedule not in {"exp", "linear", "ones", "zeros"}:
            raise ValueError(
                "RTC prefix_attention_schedule must be exp, linear, ones, "
                f"or zeros, got {schedule!r}"
            )

        end = min(int(execution_horizon), int(total_horizon))
        start = min(int(inference_delay), end)
        weights = torch.zeros(total_horizon, device=device, dtype=torch.float32)
        if end == 0:
            return weights
        if schedule == "ones":
            weights[:end] = 1.0
            return weights
        if schedule == "zeros":
            weights[:start] = 1.0
            return weights

        weights[:start] = 1.0
        transition = end - start
        if transition > 0:
            soft = torch.linspace(
                1.0,
                0.0,
                transition + 2,
                device=device,
                dtype=torch.float32,
            )[1:-1]
            if schedule == "exp":
                soft = soft * torch.expm1(soft) / (math.e - 1.0)
            weights[start:end] = soft
        return weights

    def _prepare_rtc_guidance(
        self,
        *,
        prev_actions,
        prefix_lengths,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        inference_delay: int,
        execution_horizon: int | None,
        prefix_attention_schedule: str,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if prev_actions is None:
            return None
        previous = torch.as_tensor(prev_actions, device=device, dtype=dtype)
        if previous.ndim == 2 and batch_size == 1:
            previous = previous.unsqueeze(0)
        if previous.ndim != 3:
            raise ValueError(
                "prev_action_chunk_normalized must have shape [B,T,A], got "
                f"{tuple(previous.shape)}"
            )
        if previous.shape[0] != batch_size or previous.shape[2] != self.action_dim:
            raise ValueError(
                "RTC previous-action batch/action dimensions must match the "
                f"sample: previous={tuple(previous.shape)}, "
                f"expected B={batch_size}, A={self.action_dim}"
            )

        available = min(int(previous.shape[1]), self.action_horizon)
        if prefix_lengths is None:
            lengths = torch.full(
                (batch_size,),
                available,
                device=device,
                dtype=torch.long,
            )
        else:
            lengths = torch.as_tensor(
                prefix_lengths,
                device=device,
                dtype=torch.long,
            ).reshape(-1)
            if lengths.shape != (batch_size,):
                raise ValueError(
                    "rtc_prefix_lengths must have shape [B], got "
                    f"{tuple(lengths.shape)} for B={batch_size}"
                )
            if bool((lengths < 0).any()) or bool((lengths > available).any()):
                raise ValueError(
                    "rtc_prefix_lengths must lie within the supplied previous "
                    f"chunk [0,{available}], got {lengths.tolist()}"
                )
        if not bool((lengths > 0).any()):
            return None

        if inference_delay < 0:
            raise ValueError("RTC inference_delay must be non-negative")
        if execution_horizon is not None and int(execution_horizon) <= 0:
            raise ValueError("RTC execution_horizon must be positive")

        target = torch.zeros(
            batch_size,
            self.action_horizon,
            self.action_dim,
            device=device,
            dtype=dtype,
        )
        target[:, :available] = previous[:, :available]
        weights = torch.zeros(
            batch_size,
            self.action_horizon,
            1,
            device=device,
            dtype=dtype,
        )
        for row, length_tensor in enumerate(lengths):
            length = int(length_tensor.item())
            if length == 0:
                continue
            horizon = (
                length
                if execution_horizon is None
                else min(length, int(execution_horizon))
            )
            weights[row, :, 0] = self.rtc_prefix_weights(
                inference_delay=min(int(inference_delay), horizon),
                execution_horizon=horizon,
                total_horizon=self.action_horizon,
                schedule=prefix_attention_schedule,
                device=device,
            ).to(dtype=dtype)
        return target, weights

    def _rtc_guided_action_velocity(
        self,
        *,
        noisy_action: torch.Tensor,
        action_time: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor,
        max_guidance_weight: float,
        velocity_fn,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply pseudo-inverse RTC guidance to one denoising evaluation."""

        if max_guidance_weight <= 0:
            raise ValueError("RTC max_guidance_weight must be positive")
        with torch.enable_grad():
            differentiable_action = noisy_action.detach().requires_grad_(True)
            velocity, auxiliary = velocity_fn(differentiable_action)
            sigma = self._action_sigma(
                action_time,
                differentiable_action,
                scheduler=self.infer_action_scheduler,
            )
            clean_estimate = differentiable_action - sigma * velocity
            error = (target - clean_estimate) * weights
            correction = torch.autograd.grad(
                clean_estimate,
                differentiable_action,
                grad_outputs=error.detach(),
                retain_graph=False,
            )[0]

        # LeRobot's implementation parameterizes denoising from sigma=1 to 0.
        # The cap keeps the endpoint singularities finite.
        sigma32 = sigma.detach().float()
        tau32 = 1.0 - sigma32
        tiny = torch.finfo(torch.float32).eps
        inv_r2 = (sigma32.square() + tau32.square()) / sigma32.square().clamp_min(tiny)
        coefficient = sigma32 / tau32.clamp_min(tiny)
        guidance_weight = torch.nan_to_num(
            coefficient * inv_r2,
            nan=0.0,
            posinf=float(max_guidance_weight),
            neginf=0.0,
        ).clamp(max=float(max_guidance_weight))
        guided = velocity.detach() - guidance_weight.to(velocity.dtype) * correction.detach()
        auxiliary = auxiliary.detach() if auxiliary is not None else None
        return guided, auxiliary

    @torch.no_grad()
    def sample(
        self,
        *,
        action_plan: torch.Tensor,
        world_plan: torch.Tensor,
        current_world: torch.Tensor,
        current_state: torch.Tensor | None = None,
        seed: int | None = None,
        num_inference_steps: int | None = None,
        prev_action_chunk_normalized=None,
        rtc_prefix_lengths=None,
        inference_delay: int = 0,
        execution_horizon: int | None = None,
        prefix_attention_schedule: str = "exp",
        max_guidance_weight: float = 10.0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        device, dtype = self._device_dtype()
        action_dtype = self.action_flow_dtype()
        inference_steps = (
            self.inference_steps
            if num_inference_steps is None
            else int(num_inference_steps)
        )
        if inference_steps <= 0:
            raise ValueError(
                f"num_inference_steps must be positive, got {inference_steps}"
            )
        current_world = current_world.to(device=device, dtype=dtype)
        if current_world.ndim != 3 or current_world.shape[1:] != (
            self.current_world_tokens,
            self.world_dim,
        ):
            raise ValueError(
                "current DINO must have shape "
                f"[B,{self.current_world_tokens},{self.world_dim}], "
                f"got {tuple(current_world.shape)}"
            )
        action_generator = None
        if seed is not None:
            action_generator = torch.Generator(device="cpu")
            action_generator.manual_seed(int(seed))
        batch = current_world.shape[0]
        rtc_guidance = self._prepare_rtc_guidance(
            prev_actions=prev_action_chunk_normalized,
            prefix_lengths=rtc_prefix_lengths,
            batch_size=batch,
            device=device,
            dtype=action_dtype,
            inference_delay=int(inference_delay),
            execution_horizon=execution_horizon,
            prefix_attention_schedule=prefix_attention_schedule,
        )
        action = torch.randn(
            batch,
            self.action_horizon,
            self.action_dim,
            device="cpu",
            dtype=torch.float32,
            generator=action_generator,
        ).to(device=device, dtype=action_dtype)
        action_times, action_deltas = self.infer_action_scheduler.inference_schedule(
            inference_steps,
            device=device,
            dtype=action_dtype,
        )
        if self.interaction_mode == "base":
            base_cache = self._prepare_action_base_cache(
                action_plan=action_plan,
                world_plan=world_plan,
                current_world=current_world,
                current_state=current_state,
            )
            for action_time, action_delta in zip(action_times, action_deltas):
                expanded_action_time = action_time.expand(batch)

                def base_velocity(candidate_action):
                    prediction = self._predict_action_base(
                        action_plan=action_plan,
                        world_plan=world_plan,
                        current_world=current_world,
                        noisy_action=candidate_action,
                        action_time=expanded_action_time,
                        current_state=current_state,
                        cache=base_cache,
                    )
                    return (
                        self._action_prediction_to_velocity(
                            prediction,
                            candidate_action,
                            expanded_action_time,
                            scheduler=self.infer_action_scheduler,
                        ),
                        None,
                    )

                if rtc_guidance is None:
                    action_velocity, _ = base_velocity(action)
                else:
                    action_velocity, _ = self._rtc_guided_action_velocity(
                        noisy_action=action,
                        action_time=expanded_action_time,
                        target=rtc_guidance[0],
                        weights=rtc_guidance[1],
                        max_guidance_weight=float(max_guidance_weight),
                        velocity_fn=base_velocity,
                    )
                action = self.infer_action_scheduler.step(
                    action_velocity,
                    action_delta,
                    action,
                ).to(action_dtype)
            return action, None

        world_generator = None
        if seed is not None:
            world_generator = torch.Generator(device="cpu")
            world_generator.manual_seed(int(seed))
        future_world = torch.randn(
            batch,
            self.world_tokens,
            self.world_dim,
            device="cpu",
            dtype=torch.float32,
            generator=world_generator,
        ).to(device=device, dtype=dtype)
        world_times, world_deltas = self.infer_world_scheduler.inference_schedule(
            inference_steps,
            device=device,
            dtype=dtype,
        )
        for action_time, action_delta, world_time, world_delta in zip(
            action_times,
            action_deltas,
            world_times,
            world_deltas,
        ):
            expanded_action_time = action_time.expand(batch)
            expanded_world_time = world_time.expand(batch)

            def joint_velocity(candidate_action):
                action_prediction, world_prediction = self._predict(
                    action_plan=action_plan,
                    world_plan=world_plan,
                    current_world=current_world,
                    noisy_action=candidate_action,
                    noisy_world=future_world,
                    action_time=expanded_action_time,
                    world_time=expanded_world_time,
                    action_is_pad=None,
                    current_state=current_state,
                )
                return (
                    self._action_prediction_to_velocity(
                        action_prediction,
                        candidate_action,
                        expanded_action_time,
                        scheduler=self.infer_action_scheduler,
                    ),
                    world_prediction,
                )

            if rtc_guidance is None:
                action_velocity, world_prediction = joint_velocity(action)
            else:
                action_velocity, world_prediction = self._rtc_guided_action_velocity(
                    noisy_action=action,
                    action_time=expanded_action_time,
                    target=rtc_guidance[0],
                    weights=rtc_guidance[1],
                    max_guidance_weight=float(max_guidance_weight),
                    velocity_fn=joint_velocity,
                )
            action = self.infer_action_scheduler.step(
                action_velocity,
                action_delta,
                action,
            ).to(action_dtype)
            future_world = self.infer_world_scheduler.step(
                world_prediction,
                world_delta,
                future_world,
            ).to(dtype)
        return action, future_world
