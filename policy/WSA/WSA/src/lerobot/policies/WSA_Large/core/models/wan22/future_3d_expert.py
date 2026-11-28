from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .wan_video_dit import GateModule, SelfAttention, precompute_freqs_cis, sinusoidal_embedding_1d


class Future3DPerceiverFeedForward(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim, bias=False),
            nn.SiLU(),
            nn.Linear(dim, dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Future3DPerceiverAttention(nn.Module):
    def __init__(self, dim: int, dim_head: int, heads: int):
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.memory_norm = nn.LayerNorm(dim)
        self.query_norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, width = x.shape
        x = x.view(batch_size, seq_len, self.heads, width // self.heads)
        return x.transpose(1, 2)

    def forward(self, memory: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        memory = self.memory_norm(memory)
        queries = self.query_norm(queries)

        q = self._reshape_heads(self.to_q(queries))
        k, v = self.to_kv(memory).chunk(2, dim=-1)
        k = self._reshape_heads(k)
        v = self._reshape_heads(v)

        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(queries.shape[0], queries.shape[1], -1)
        return self.to_out(attended)


class Future3DPerceiverResampler(nn.Module):
    def __init__(self, dim: int, num_heads: int, output_dim: int):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        dim_head = dim // num_heads
        self.attn = Future3DPerceiverAttention(dim=dim, dim_head=dim_head, heads=num_heads)
        self.ff = Future3DPerceiverFeedForward(dim=dim)
        self.output_norm = nn.LayerNorm(dim)
        self.output_proj = nn.Linear(dim, output_dim)

    def forward(self, output_queries: torch.Tensor, messenger_tokens: torch.Tensor) -> torch.Tensor:
        latents = output_queries
        latents = latents + self.attn(messenger_tokens, latents)
        latents = latents + self.ff(latents)
        return self.output_proj(self.output_norm(latents))


class Future3DBlock(nn.Module):
    """DiT-compatible block without text cross-attention.

    MoT only requires each expert block to expose `self_attn`, `norm1`, `norm2`,
    `ffn`, `modulation`, and `gate`. Keeping this block cross-attn-free makes the
    future-3D branch a compact auxiliary expert instead of another full DiT.
    """

    def __init__(
        self,
        hidden_dim: int,
        attn_head_dim: int,
        num_heads: int,
        ffn_dim: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn_head_dim = attn_head_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(hidden_dim, attn_head_dim, num_heads, eps)
        self.norm1 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, hidden_dim) / hidden_dim**0.5)
        self.gate = GateModule()


class Future3DExpert(nn.Module):
    QUERY_MODES = ("query_token", "noised_query_token", "slot_noise")
    QUERY_SIGMA_SOURCES = ("constant", "video")
    FUTURE_3D_BACKBONE_SKIP_PREFIXES = (
        "query_tokens",
        "inference_noise_tokens",
        "layer_norms.",
        "output_queries",
        "output_decoder.",
    )
    FUTURE_3D_BACKBONE_META_KEYS = (
        "hidden_dim",
        "ffn_dim",
        "num_layers",
        "num_heads",
        "attn_head_dim",
        "freq_dim",
        "eps",
    )

    def __init__(
        self,
        hidden_dim: int = 768,
        ffn_dim: int = 3072,
        text_dim: int = 4096,
        freq_dim: int = 256,
        eps: float = 1e-6,
        num_heads: int = 24,
        attn_head_dim: int = 128,
        num_layers: int = 30,
        num_query_tokens: int = 288,
        da3_num_views: int = 2,
        da3_tokens_per_view: int = 1296,
        da3_query_dim: int = 2048,
        query_layer_indices: tuple[int, ...] = (14, 19, 24, 29),
        future_query_init_std: float = 0.02,
        query_mode: str = "query_token",
        query_noise_scale: float = 1.0,
        query_noise_min_sigma: float = 0.0,
        query_noise_max_sigma: float = 0.5,
        query_sigma_source: str = "constant",
        slot_pos_scale: float = 1.0,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        del text_dim
        query_mode = str(query_mode)
        if query_mode not in self.QUERY_MODES:
            raise ValueError(f"`query_mode` must be one of {self.QUERY_MODES}, got {query_mode!r}")
        query_sigma_source = str(query_sigma_source)
        if query_sigma_source not in self.QUERY_SIGMA_SOURCES:
            raise ValueError(
                f"`query_sigma_source` must be one of {self.QUERY_SIGMA_SOURCES}, got {query_sigma_source!r}"
            )
        if num_query_tokens <= 0:
            raise ValueError(f"`num_query_tokens` must be positive, got {num_query_tokens}")
        if da3_num_views <= 0:
            raise ValueError(f"`da3_num_views` must be positive, got {da3_num_views}")
        if num_query_tokens % da3_num_views != 0:
            raise ValueError(
                f"`num_query_tokens` ({num_query_tokens}) must be divisible by da3_num_views ({da3_num_views})"
            )

        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.eps = eps
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.num_layers = num_layers
        self.num_query_tokens = num_query_tokens
        self.da3_num_views = da3_num_views
        self.da3_tokens_per_view = da3_tokens_per_view
        self.da3_query_dim = da3_query_dim
        self.query_layer_indices = tuple(int(idx) for idx in query_layer_indices)
        self.query_tokens_per_view = num_query_tokens // da3_num_views
        self.query_mode = query_mode
        self.query_noise_scale = float(query_noise_scale)
        self.query_noise_min_sigma = float(query_noise_min_sigma)
        self.query_noise_max_sigma = float(query_noise_max_sigma)
        self.query_sigma_source = query_sigma_source
        self.slot_pos_scale = float(slot_pos_scale)
        if self.query_noise_scale < 0:
            raise ValueError("`query_noise_scale` must be non-negative.")
        if not (0.0 <= self.query_noise_min_sigma <= self.query_noise_max_sigma <= 1.0):
            raise ValueError(
                "`query_noise_min_sigma` and `query_noise_max_sigma` must satisfy "
                "0 <= min <= max <= 1."
            )

        query_tokens = torch.randn(1, num_query_tokens, hidden_dim) * future_query_init_std
        if self.query_mode == "slot_noise":
            self.register_buffer("query_tokens", query_tokens)
        else:
            self.query_tokens = nn.Parameter(query_tokens)
        slot_positions = torch.arange(num_query_tokens, dtype=torch.float32)
        slot_pos_embed = sinusoidal_embedding_1d(hidden_dim, slot_positions).unsqueeze(0)
        self.register_buffer("slot_pos_embed", slot_pos_embed, persistent=False)
        inference_noise = torch.empty(1, num_query_tokens, hidden_dim)
        inference_noise.normal_(generator=torch.Generator().manual_seed(0))
        self.register_buffer("inference_noise_tokens", inference_noise, persistent=False)
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        self.blocks = nn.ModuleList(
            [
                Future3DBlock(
                    hidden_dim=hidden_dim,
                    attn_head_dim=attn_head_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    eps=eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.freqs = precompute_freqs_cis(attn_head_dim, end=max(1024, num_query_tokens))
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in self.query_layer_indices])
        self.output_queries = nn.Parameter(
            torch.randn(1, da3_tokens_per_view, hidden_dim) * future_query_init_std
        )
        self.output_decoder = Future3DPerceiverResampler(
            dim=hidden_dim,
            num_heads=num_heads,
            output_dim=da3_query_dim,
        )
        self.use_gradient_checkpointing = use_gradient_checkpointing

    @classmethod
    def backbone_key_set(cls, keys) -> set[str]:
        return {
            key
            for key in keys
            if not any(key.startswith(prefix) for prefix in cls.FUTURE_3D_BACKBONE_SKIP_PREFIXES)
        }

    @classmethod
    def _resolve_pretrained_path(cls, pretrained_path: str | Path) -> Path:
        p = Path(pretrained_path)
        if p.is_absolute():
            return p

        file_path = Path(__file__).resolve()
        candidate_roots = [Path.cwd()]
        if len(file_path.parents) > 7:
            candidate_roots.append(file_path.parents[7])
        if len(file_path.parents) > 4:
            candidate_roots.append(file_path.parents[4])

        for root in candidate_roots:
            candidate = (root / p).resolve()
            if candidate.is_file():
                return candidate
        return (candidate_roots[0] / p).resolve()

    @classmethod
    def from_pretrained(
        cls,
        future_3d_config: dict[str, Any],
        future_3d_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "Future3DExpert":
        if future_3d_config is None:
            raise ValueError("`future_3d_config` is required for Future3DExpert.from_pretrained().")
        if skip_dit_load_from_pretrain:
            return cls(**future_3d_config).to(device=device, dtype=torch_dtype)
        if not future_3d_pretrained_path:
            return cls(**future_3d_config).to(device=device, dtype=torch_dtype)

        p = cls._resolve_pretrained_path(future_3d_pretrained_path)
        future_3d_pretrained_path = str(p)
        if not os.path.isfile(future_3d_pretrained_path):
            raise FileNotFoundError(
                f"`future_3d_pretrained_path` does not exist: {future_3d_pretrained_path}"
            )

        future_cfg = dict(future_3d_config)
        future_expert = cls(**future_cfg).to(device=device, dtype=torch_dtype)
        future_state = future_expert.state_dict()
        expected_backbone_keys = cls.backbone_key_set(future_state.keys())

        payload = torch.load(future_3d_pretrained_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(
                f"Invalid future-3D backbone payload type from {future_3d_pretrained_path}: {type(payload)}"
            )

        meta = payload.get("meta")
        if not isinstance(meta, dict):
            raise ValueError(f"`meta` must be a dict in {future_3d_pretrained_path}.")
        expected_meta = {
            "hidden_dim": int(future_cfg["hidden_dim"]),
            "ffn_dim": int(future_cfg["ffn_dim"]),
            "num_layers": int(future_cfg["num_layers"]),
            "num_heads": int(future_cfg["num_heads"]),
            "attn_head_dim": int(future_cfg["attn_head_dim"]),
            "freq_dim": int(future_cfg["freq_dim"]),
            "eps": float(future_cfg["eps"]),
        }
        for key in cls.FUTURE_3D_BACKBONE_META_KEYS:
            if key not in meta:
                raise ValueError(f"`meta.{key}` missing in {future_3d_pretrained_path}")
            expected_value = expected_meta[key]
            got_value = meta[key]
            if key == "eps":
                if abs(float(got_value) - float(expected_value)) > 1e-12:
                    raise ValueError(
                        f"`meta.{key}` mismatch in {future_3d_pretrained_path}: "
                        f"expected {expected_value}, got {got_value}"
                    )
            elif int(got_value) != int(expected_value):
                raise ValueError(
                    f"`meta.{key}` mismatch in {future_3d_pretrained_path}: "
                    f"expected {expected_value}, got {got_value}"
                )

        backbone_state_dict = payload.get("backbone_state_dict")
        if not isinstance(backbone_state_dict, dict):
            raise ValueError(
                f"`backbone_state_dict` must be a dict in {future_3d_pretrained_path}, "
                f"got {type(backbone_state_dict)}"
            )

        provided_keys = set(backbone_state_dict.keys())
        missing_keys = sorted(expected_backbone_keys - provided_keys)
        unexpected_keys = sorted(provided_keys - expected_backbone_keys)
        if missing_keys or unexpected_keys:
            raise ValueError(
                "Future3D backbone key mismatch in preprocessed payload. "
                f"missing={missing_keys[:10]}{'...' if len(missing_keys) > 10 else ''}, "
                f"unexpected={unexpected_keys[:10]}{'...' if len(unexpected_keys) > 10 else ''}"
            )

        merged_state = dict(future_state)
        for key in expected_backbone_keys:
            value = backbone_state_dict[key]
            if not isinstance(value, torch.Tensor):
                raise ValueError(
                    f"`backbone_state_dict[{key}]` must be torch.Tensor in {future_3d_pretrained_path}, "
                    f"got {type(value)}"
                )
            target = merged_state[key]
            if tuple(value.shape) != tuple(target.shape):
                raise ValueError(
                    f"Shape mismatch for `{key}` in {future_3d_pretrained_path}: "
                    f"expected {tuple(target.shape)}, got {tuple(value.shape)}"
                )
            merged_state[key] = value.to(device=target.device, dtype=target.dtype)

        future_expert.load_state_dict(merged_state, strict=True)
        return future_expert.to(device=device, dtype=torch_dtype)

    def _query_noise(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.training and torch.is_grad_enabled():
            return torch.randn(batch_size, self.num_query_tokens, self.hidden_dim, device=device, dtype=dtype)
        return self.inference_noise_tokens.expand(batch_size, -1, -1).to(device=device, dtype=dtype)

    def _prepare_query_tokens(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        query_noise_sigma: torch.Tensor | None,
    ) -> torch.Tensor:
        clean_query = self.query_tokens.expand(batch_size, -1, -1).to(device=device, dtype=dtype)
        if self.query_mode == "query_token":
            return clean_query

        if query_noise_sigma is None:
            default_sigma = self.query_noise_max_sigma if self.training else self.query_noise_max_sigma
            sigma = torch.full((batch_size,), default_sigma, device=device, dtype=dtype)
        else:
            sigma = query_noise_sigma.to(device=device, dtype=dtype)
            if sigma.ndim == 0:
                sigma = sigma.expand(batch_size)
            if sigma.ndim != 1 or sigma.shape[0] not in (1, batch_size):
                raise ValueError(
                    "`query_noise_sigma` must be scalar or 1D [B], "
                    f"got shape {tuple(query_noise_sigma.shape)}"
                )
            if sigma.shape[0] == 1 and batch_size > 1:
                sigma = sigma.expand(batch_size)
        sigma = sigma.clamp(min=self.query_noise_min_sigma, max=self.query_noise_max_sigma)
        sigma = sigma.view(batch_size, 1, 1)

        noise = self._query_noise(batch_size=batch_size, device=device, dtype=dtype) * self.query_noise_scale
        if self.query_mode == "noised_query_token":
            return (1.0 - sigma) * clean_query + sigma * noise

        if self.query_mode == "slot_noise":
            slot_pos = self.slot_pos_embed.expand(batch_size, -1, -1).to(device=device, dtype=dtype)
            return slot_pos * self.slot_pos_scale + sigma * noise

        raise RuntimeError(f"Unsupported future-3D query mode: {self.query_mode!r}")

    def pre_dit(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        timestep: torch.Tensor | None = None,
        query_noise_sigma: torch.Tensor | None = None,
    ) -> Dict[str, Any]:
        if timestep is None:
            timestep = torch.zeros((batch_size,), device=device, dtype=dtype)
        if timestep.ndim != 1:
            raise ValueError(f"`timestep` must be 1D [B], got shape {tuple(timestep.shape)}")
        if timestep.shape[0] == 1 and batch_size > 1:
            if self.training:
                raise ValueError("During training, future-3D timestep length must match batch_size.")
            timestep = timestep.expand(batch_size)
        if timestep.shape[0] != batch_size:
            raise ValueError(f"`timestep` length must be {batch_size}, got {timestep.shape[0]}")

        tokens = self._prepare_query_tokens(
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            query_noise_sigma=query_noise_sigma,
        )
        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.hidden_dim))
        freqs = self.freqs[: self.num_query_tokens].view(self.num_query_tokens, 1, -1).to(device)
        return {
            "tokens": tokens,
            "freqs": freqs,
            "t": t,
            "t_mod": t_mod,
            "context": None,
            "context_mask": None,
            "meta": {
                "batch_size": batch_size,
                "seq_len": self.num_query_tokens,
                "query_mode": self.query_mode,
            },
        }

    def project_query_layers(self, layer_tokens: tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
        if len(layer_tokens) != len(self.query_layer_indices):
            raise ValueError(
                f"Expected {len(self.query_layer_indices)} future-3D layers, got {len(layer_tokens)}"
            )

        projected = []
        for layer_norm, tokens in zip(self.layer_norms, layer_tokens, strict=True):
            tokens = layer_norm(tokens)
            batch_size = tokens.shape[0]
            messenger = tokens.view(batch_size, self.da3_num_views, self.query_tokens_per_view, self.hidden_dim)
            messenger = messenger.reshape(batch_size * self.da3_num_views, self.query_tokens_per_view, self.hidden_dim)

            output_queries = self.output_queries.expand(batch_size, self.da3_tokens_per_view, -1)
            output_queries = output_queries[:, None, :, :].expand(
                batch_size, self.da3_num_views, self.da3_tokens_per_view, self.hidden_dim
            )
            output_queries = output_queries.reshape(
                batch_size * self.da3_num_views, self.da3_tokens_per_view, self.hidden_dim
            )

            decoded = self.output_decoder(output_queries, messenger)
            decoded = decoded.view(batch_size, self.da3_num_views * self.da3_tokens_per_view, self.da3_query_dim)
            projected.append(decoded)
        return projected
