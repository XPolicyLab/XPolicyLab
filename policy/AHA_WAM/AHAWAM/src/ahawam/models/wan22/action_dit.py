import os
import torch
import torch.nn as nn
from typing import Any, Dict, Optional

from ahawam.utils.logging_config import get_logger

from .helpers.gradient import gradient_checkpoint_forward
from .wan_video_dit import (
    DiTBlock,
    sinusoidal_embedding_1d,
    precompute_freqs_cis,
)

logger = get_logger(__name__)


class ActionHead(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int, eps: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.proj = nn.Linear(hidden_dim, out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, hidden_dim) / hidden_dim**0.5)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        shift, scale = (
            self.modulation.to(dtype=t.dtype, device=t.device) + t.unsqueeze(1)
        ).chunk(2, dim=1)
        shift = shift.squeeze(1)
        scale = scale.squeeze(1)
        return self.proj(self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1))


class ActionDiT(nn.Module):
    ACTION_BACKBONE_SKIP_PREFIXES = ("action_encoder.", "head.")
    ACTION_BACKBONE_META_KEYS = (
        "hidden_dim",
        "ffn_dim",
        "num_layers",
        "num_heads",
        "attn_head_dim",
        "text_dim",
        "freq_dim",
        "eps",
    )
    ACTION_TEACHER_FORCING_MASK_MODES = ("stage1_chunkwise",)

    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        ffn_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        num_heads: int,
        attn_head_dim: int,
        num_layers: int,
        autoregressive_teacher_forcing: bool = False,
        action_chunk_size: int = 1,
        action_teacher_forcing_mask_mode: str = "stage1_chunkwise",
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.ffn_dim = ffn_dim
        self.text_dim = text_dim
        self.freq_dim = freq_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.autoregressive_teacher_forcing = bool(autoregressive_teacher_forcing)
        self.action_chunk_size = int(action_chunk_size)
        self.action_teacher_forcing_mask_mode = str(action_teacher_forcing_mask_mode)

        if num_heads <= 0:
            raise ValueError(f"`num_heads` must be > 0, got {num_heads}")
        if attn_head_dim <= 0:
            raise ValueError(f"`attn_head_dim` must be > 0, got {attn_head_dim}")
        if attn_head_dim % 2 != 0:
            raise ValueError(
                f"`attn_head_dim` must be even for RoPE, got {attn_head_dim}"
            )
        if self.action_chunk_size <= 0:
            raise ValueError(
                f"`action_chunk_size` must be > 0, got {self.action_chunk_size}"
            )
        if (
            self.action_teacher_forcing_mask_mode
            not in self.ACTION_TEACHER_FORCING_MASK_MODES
        ):
            raise ValueError(
                "`action_teacher_forcing_mask_mode` must be one of "
                f"{self.ACTION_TEACHER_FORCING_MASK_MODES}, got {self.action_teacher_forcing_mask_mode}"
            )

        self.action_encoder = nn.Linear(action_dim, hidden_dim)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6)
        )
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_dim=hidden_dim,
                    attn_head_dim=attn_head_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    eps=eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.head = nn.Linear(hidden_dim, action_dim)
        self.freqs = precompute_freqs_cis(attn_head_dim, end=1024)

        self.use_gradient_checkpointing = use_gradient_checkpointing

    @classmethod
    def backbone_key_set(cls, keys) -> set[str]:
        return {
            key
            for key in keys
            if not any(
                key.startswith(prefix) for prefix in cls.ACTION_BACKBONE_SKIP_PREFIXES
            )
        }

    @classmethod
    def from_pretrained(
        cls,
        action_dit_config: dict[str, Any],
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "ActionDiT":
        if action_dit_config is None:
            raise ValueError(
                "`action_dit_config` is required for ActionDiT.from_pretrained()."
            )
        if skip_dit_load_from_pretrain:
            logger.info(
                "Skipping ActionDiT pretrained load (`skip_dit_load_from_pretrain=True`); "
                "initializing action expert randomly and expecting checkpoint override."
            )
            return cls(**action_dit_config).to(device=device, dtype=torch_dtype)
        if not action_dit_pretrained_path:
            logger.info(
                "No `action_dit_pretrained_path` provided, initializing ActionDiT with random weights."
            )
            return cls(**action_dit_config).to(device=device, dtype=torch_dtype)
        from pathlib import Path

        p = Path(action_dit_pretrained_path)
        if not p.is_absolute():
            p = Path(__file__).resolve().parents[4] / p
        action_dit_pretrained_path = str(p)
        if not os.path.isfile(action_dit_pretrained_path):
            raise FileNotFoundError(
                f"`action_dit_pretrained_path` does not exist: {action_dit_pretrained_path}"
            )

        action_cfg = dict(action_dit_config)
        action_expert = cls(**action_cfg)
        action_state = action_expert.state_dict()
        expected_backbone_keys = cls.backbone_key_set(action_state.keys())

        payload = torch.load(action_dit_pretrained_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(
                f"Invalid action backbone payload type from {action_dit_pretrained_path}: {type(payload)}"
            )

        policy = payload.get("policy", {})
        if policy:
            logger.info(f"ActionDiT backbone payload policy: {policy}")

        meta = payload.get("meta")
        if not isinstance(meta, dict):
            raise ValueError(
                f"`meta` must be a dict in {action_dit_pretrained_path}, got {type(meta)}"
            )
        expected_meta = {
            "hidden_dim": int(action_cfg["hidden_dim"]),
            "ffn_dim": int(action_cfg["ffn_dim"]),
            "num_layers": int(action_cfg["num_layers"]),
            "num_heads": int(action_cfg["num_heads"]),
            "attn_head_dim": int(action_cfg["attn_head_dim"]),
            "text_dim": int(action_cfg["text_dim"]),
            "freq_dim": int(action_cfg["freq_dim"]),
            "eps": float(action_cfg["eps"]),
        }
        for key in cls.ACTION_BACKBONE_META_KEYS:
            if key not in meta:
                raise ValueError(
                    f"`meta.{key}` missing in {action_dit_pretrained_path}"
                )
            expected_value = expected_meta[key]
            got_value = meta[key]
            if key == "eps":
                if abs(float(got_value) - float(expected_value)) > 1e-12:
                    raise ValueError(
                        f"`meta.{key}` mismatch in {action_dit_pretrained_path}: "
                        f"expected {expected_value}, got {got_value}"
                    )
            elif int(got_value) != int(expected_value):
                raise ValueError(
                    f"`meta.{key}` mismatch in {action_dit_pretrained_path}: "
                    f"expected {expected_value}, got {got_value}"
                )

        backbone_state_dict = payload.get("backbone_state_dict")
        if not isinstance(backbone_state_dict, dict):
            raise ValueError(
                f"`backbone_state_dict` must be a dict in {action_dit_pretrained_path}, "
                f"got {type(backbone_state_dict)}"
            )

        provided_keys = set(backbone_state_dict.keys())
        missing_keys = sorted(expected_backbone_keys - provided_keys)
        unexpected_keys = sorted(provided_keys - expected_backbone_keys)
        if missing_keys or unexpected_keys:
            raise ValueError(
                "Action backbone key mismatch in preprocessed payload. "
                f"missing={missing_keys[:10]}{'...' if len(missing_keys) > 10 else ''}, "
                f"unexpected={unexpected_keys[:10]}{'...' if len(unexpected_keys) > 10 else ''}"
            )

        merged_state = dict(action_state)
        for key in expected_backbone_keys:
            value = backbone_state_dict[key]
            if not isinstance(value, torch.Tensor):
                raise ValueError(
                    f"`backbone_state_dict[{key}]` must be torch.Tensor in {action_dit_pretrained_path}, "
                    f"got {type(value)}"
                )
            target = merged_state[key]
            if tuple(value.shape) != tuple(target.shape):
                raise ValueError(
                    f"Shape mismatch for `{key}` in {action_dit_pretrained_path}: "
                    f"expected {tuple(target.shape)}, got {tuple(value.shape)}"
                )
            merged_state[key] = value.to(device="cpu", dtype=target.dtype)

        action_expert.load_state_dict(merged_state, strict=True)
        logger.info(
            "Loaded ActionDiT backbone from %s (keys=%d; random_kept_prefixes=%s).",
            action_dit_pretrained_path,
            len(expected_backbone_keys),
            list(cls.ACTION_BACKBONE_SKIP_PREFIXES),
        )
        return action_expert.to(device=device, dtype=torch_dtype)

    def _normalize_tokenwise_timestep(
        self,
        timestep: torch.Tensor,
        *,
        batch_size: int,
        seq_len: int,
        chunk_size: int,
        name: str,
    ) -> tuple[torch.Tensor, bool]:
        """produce timesteps for causal/bidirectional timestep, support shared/per chunk/per token timesteps"""
        if timestep.ndim == 1:
            if timestep.shape[0] not in (1, batch_size):
                raise ValueError(
                    f"`{name}` length must be 1 or batch_size({batch_size}), got {timestep.shape[0]}"
                )
            if timestep.shape[0] == 1 and batch_size > 1:
                if self.training:
                    raise ValueError(
                        f"During training, `{name}` length must match batch_size."
                    )
                timestep = timestep.expand(batch_size)
            return timestep, False

        if timestep.ndim != 2:
            raise ValueError(
                f"`{name}` must be 1D [B]/[1] or 2D [B,T]/[B,num_chunks], got shape {tuple(timestep.shape)}"
            )
        if timestep.shape[0] != batch_size:
            raise ValueError(
                f"`{name}` batch size mismatch: expected {batch_size}, got {timestep.shape[0]}"
            )

        if timestep.shape[1] == seq_len:
            return timestep, True
        num_chunks = seq_len // chunk_size
        if timestep.shape[1] == num_chunks:
            return timestep.repeat_interleave(chunk_size, dim=1), True
        if timestep.shape[1] == 1:
            return timestep.expand(batch_size, seq_len), True
        raise ValueError(
            f"`{name}` second dim must be 1, seq_len({seq_len}), or num_chunks({num_chunks}), "
            f"got {timestep.shape[1]}"
        )

    def _merge_obs_context(
        self,
        *,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        batch_size: int,
        obs_context: Optional[torch.Tensor],
        obs_context_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], int, int]:
        """Validate and merge chunk-aligned obs_context into the text context.

        Returns (context, flat_obs_context_mask, obs_num_chunks, obs_tokens_per_chunk).
        """
        if obs_context is None:
            if obs_context_mask is not None:
                raise ValueError(
                    "`obs_context_mask` cannot be provided without `obs_context`."
                )
            return context, None, 0, 0

        if obs_context.ndim != 4:
            raise ValueError(
                "`obs_context` must be 4D [B, N, L, D] (chunk-aligned), "
                f"got shape {tuple(obs_context.shape)} with ndim={obs_context.ndim}. "
                "Global 3D obs_context (all chunks sharing one obs) is not supported."
            )
        if obs_context.shape[0] != batch_size:
            raise ValueError(
                "Batch mismatch between action tokens and chunked obs context: "
                f"{batch_size} vs {obs_context.shape[0]}"
            )
        if obs_context.shape[3] != context.shape[2]:
            raise ValueError(
                "`obs_context` last dim must match `context` last dim, "
                f"got {obs_context.shape[3]} vs {context.shape[2]}"
            )
        obs_num_chunks = int(obs_context.shape[1])
        obs_tokens_per_chunk = int(obs_context.shape[2])
        if obs_num_chunks <= 0 or obs_tokens_per_chunk <= 0:
            raise ValueError(
                "`obs_context` chunk dimensions must be positive, "
                f"got shape {tuple(obs_context.shape)}"
            )
        if obs_context_mask is None:
            obs_context_mask = torch.ones(
                (batch_size, obs_num_chunks, obs_tokens_per_chunk),
                dtype=torch.bool,
                device=obs_context.device,
            )
        else:
            if obs_context_mask.ndim != 3:
                raise ValueError(
                    "`obs_context_mask` for chunked obs context must be 3D [B, N, L], "
                    f"got shape {tuple(obs_context_mask.shape)}"
                )
            if tuple(obs_context_mask.shape) != (
                batch_size,
                obs_num_chunks,
                obs_tokens_per_chunk,
            ):
                raise ValueError(
                    "`obs_context_mask` shape must match chunked `obs_context` shape [B, N, L], "
                    f"got {tuple(obs_context_mask.shape)} vs {(batch_size, obs_num_chunks, obs_tokens_per_chunk)}"
                )
        flat_obs_context_mask = obs_context_mask.to(
            dtype=torch.bool, device=context_mask.device
        ).reshape(batch_size, obs_num_chunks * obs_tokens_per_chunk)
        merged_context = torch.cat(
            [
                context,
                obs_context.to(device=context.device, dtype=context.dtype).reshape(
                    batch_size,
                    obs_num_chunks * obs_tokens_per_chunk,
                    context.shape[2],
                ),
            ],
            dim=1,
        )
        return (
            merged_context,
            flat_obs_context_mask,
            obs_num_chunks,
            obs_tokens_per_chunk,
        )

    def _validate_positions(
        self,
        *,
        seq_len: int,
        clean_seq_len: int,
        effective_chunk_size: int,
        noisy_position_offset: int,
    ) -> None:
        """Raise ValueError if any position / length invariant is violated."""
        if seq_len > self.freqs.shape[0]:
            raise ValueError(
                f"Action token length {seq_len} exceeds RoPE cache {self.freqs.shape[0]}."
            )
        if seq_len % effective_chunk_size != 0:
            raise ValueError(
                f"Action token length {seq_len} must be divisible by chunk_size {effective_chunk_size}."
            )
        if clean_seq_len > 0 and clean_seq_len % effective_chunk_size != 0:
            raise ValueError(
                f"Clean action token length {clean_seq_len} must be divisible by chunk_size {effective_chunk_size}."
            )
        if noisy_position_offset < 0:
            raise ValueError(
                f"`noisy_position_offset` must be >= 0, got {noisy_position_offset}"
            )
        if noisy_position_offset % effective_chunk_size != 0:
            raise ValueError(
                "`noisy_position_offset` must align to chunk_size, "
                f"got offset={noisy_position_offset}, chunk_size={effective_chunk_size}."
            )
        if noisy_position_offset + seq_len > self.freqs.shape[0]:
            raise ValueError(
                "Action noisy positions exceed RoPE cache: "
                f"offset={noisy_position_offset}, seq_len={seq_len}, cache={self.freqs.shape[0]}."
            )
        if clean_seq_len > self.freqs.shape[0]:
            raise ValueError(
                f"Clean action token length {clean_seq_len} exceeds RoPE cache {self.freqs.shape[0]}."
            )

    def _build_context_attention_mask(
        self,
        *,
        base_context_attn_mask: torch.Tensor,
        total_seq_len: int,
        clean_seq_len: int,
        seq_len: int,
        effective_chunk_size: int,
        noisy_position_offset: int,
        flat_obs_context_mask: Optional[torch.Tensor],
        obs_num_chunks: int,
        obs_tokens_per_chunk: int,
        obs_chunk_offset: int = 0,
        obs_context_causal: bool = False,
        obs_proprio_tokens_per_chunk: int = 0,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Build the cross-attention mask over merged (text + obs) context tokens.

        This method supports three obs-routing modes:
        - `obs_context_causal=True` and `obs_proprio_tokens_per_chunk>0`:
          causal visual-obs routing + current-chunk-only proprio routing
        - `obs_context_causal=True` and no proprio split:
          causal obs routing
        - otherwise:
          local per-chunk obs routing

        Args:
            obs_chunk_offset: Global chunk index of the first chunk contained in
                `obs_context`.
                - Full-sequence training usually passes 0.
                - Local streaming inference passes the absolute chunk index of the
                  current local obs window.
                - Causal accumulated inference from chunk 0 also passes 0.
                - Future causal sliding windows would pass that window's absolute
                  start chunk, even though the routing is still causal.
            noisy_position_offset: Global action-token start offset of the current
                noisy chunk.
                - Training usually passes 0.
                - Streaming inference passes the absolute start position of the
                  current noisy action chunk.
        """
        if flat_obs_context_mask is None:
            return base_context_attn_mask

        assert obs_num_chunks > 0 and obs_tokens_per_chunk > 0
        clean_positions = torch.arange(clean_seq_len, device=device)
        noisy_positions = torch.arange(seq_len, device=device) + int(
            noisy_position_offset
        )
        absolute_positions = torch.cat([clean_positions, noisy_positions], dim=0)
        token_chunk_ids = (
            torch.div(
                absolute_positions,
                effective_chunk_size,
                rounding_mode="floor",
            ).long()
            - obs_chunk_offset
        )
        if bool((token_chunk_ids < 0).any().item()) or bool(
            (token_chunk_ids >= obs_num_chunks).any().item()
        ):
            raise ValueError(
                "Chunked obs context does not cover the requested action positions: "
                f"token chunk ids range [{int(token_chunk_ids.min().item())}, {int(token_chunk_ids.max().item())}], "
                f"obs window=[{obs_chunk_offset}, {obs_chunk_offset + obs_num_chunks}) "
                f"(obs_num_chunks={obs_num_chunks})."
            )
        if obs_proprio_tokens_per_chunk > 0 and obs_context_causal:
            visual_tokens_per_chunk = (
                obs_tokens_per_chunk - obs_proprio_tokens_per_chunk
            )
            if visual_tokens_per_chunk < 0:
                raise ValueError(
                    f"`obs_proprio_tokens_per_chunk` ({obs_proprio_tokens_per_chunk}) cannot exceed "
                    f"`obs_tokens_per_chunk` ({obs_tokens_per_chunk})."
                )

            chunk_id_range = torch.arange(obs_num_chunks, device=device)
            visual_block_mask = chunk_id_range.unsqueeze(
                0
            ) <= token_chunk_ids.unsqueeze(1)
            proprio_block_mask = torch.zeros(
                (total_seq_len, obs_num_chunks), dtype=torch.bool, device=device
            )
            proprio_block_mask[
                torch.arange(total_seq_len, device=device), token_chunk_ids
            ] = True

            visual_mask = visual_block_mask.unsqueeze(-1).expand(
                total_seq_len, obs_num_chunks, visual_tokens_per_chunk
            )
            proprio_mask = proprio_block_mask.unsqueeze(-1).expand(
                total_seq_len, obs_num_chunks, obs_proprio_tokens_per_chunk
            )
            per_token_obs_mask = torch.cat([visual_mask, proprio_mask], dim=-1)
            per_token_obs_mask = per_token_obs_mask.reshape(
                total_seq_len, obs_num_chunks * obs_tokens_per_chunk
            )
        elif obs_context_causal and obs_proprio_tokens_per_chunk == 0:
            chunk_id_range = torch.arange(obs_num_chunks, device=device)
            per_token_obs_block_mask = chunk_id_range.unsqueeze(
                0
            ) <= token_chunk_ids.unsqueeze(1)
            per_token_obs_mask = (
                per_token_obs_block_mask.unsqueeze(-1)
                .expand(total_seq_len, obs_num_chunks, obs_tokens_per_chunk)
                .reshape(total_seq_len, obs_num_chunks * obs_tokens_per_chunk)
            )
        else:
            per_token_obs_block_mask = torch.zeros(
                (total_seq_len, obs_num_chunks),
                dtype=torch.bool,
                device=device,
            )
            per_token_obs_block_mask[
                torch.arange(total_seq_len, device=device), token_chunk_ids
            ] = True
            per_token_obs_mask = (
                per_token_obs_block_mask.unsqueeze(-1)
                .expand(total_seq_len, obs_num_chunks, obs_tokens_per_chunk)
                .reshape(total_seq_len, obs_num_chunks * obs_tokens_per_chunk)
            )
        return torch.cat(
            [
                base_context_attn_mask,
                flat_obs_context_mask.unsqueeze(1) & per_token_obs_mask.unsqueeze(0),
            ],
            dim=2,
        )

    def _compute_teacher_forcing_modulations(
        self,
        *,
        batch_size: int,
        seq_len: int,
        clean_seq_len: int,
        timestep: torch.Tensor,
        timestep_is_tokenwise: bool,
        clean_timestep: Optional[torch.Tensor],
        noisy_position_offset: int,
        effective_chunk_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Compute t, t_mod, freqs, self_attn_mask for teacher-forced mode.

        Requires ``clean_seq_len > 0``.
        """
        if clean_timestep is None:
            raise ValueError(
                "`clean_timestep` is required when `clean_action_tokens` is provided."
            )
        clean_t = clean_timestep
        clean_t, clean_t_is_tokenwise = self._normalize_tokenwise_timestep(
            clean_t,
            batch_size=batch_size,
            seq_len=clean_seq_len,
            chunk_size=effective_chunk_size,
            name="clean_timestep",
        )
        if clean_t_is_tokenwise:
            clean_t_emb = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, clean_t.reshape(-1))
            ).view(batch_size, clean_seq_len, self.hidden_dim)
            clean_t_mod = self.time_projection(clean_t_emb).view(
                batch_size, clean_seq_len, 6, self.hidden_dim
            )
        else:
            clean_t_emb = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, clean_t)
            )
            clean_t_mod = (
                self.time_projection(clean_t_emb)
                .view(batch_size, 1, 6, self.hidden_dim)
                .expand(-1, clean_seq_len, -1, -1)
            )
        if timestep_is_tokenwise:
            noisy_t_emb = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, timestep.reshape(-1))
            ).view(batch_size, seq_len, self.hidden_dim)
            noisy_t_mod = self.time_projection(noisy_t_emb).view(
                batch_size, seq_len, 6, self.hidden_dim
            )
            t = noisy_t_emb
        else:
            noisy_t_emb = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, timestep)
            )
            noisy_t_mod = (
                self.time_projection(noisy_t_emb)
                .view(batch_size, 1, 6, self.hidden_dim)
                .expand(-1, seq_len, -1, -1)
            )
            t = noisy_t_emb
        t_mod = torch.cat([clean_t_mod, noisy_t_mod], dim=1)
        clean_freqs = self.freqs[:clean_seq_len].view(clean_seq_len, 1, -1).to(device)
        noisy_freqs = (
            self.freqs[noisy_position_offset : noisy_position_offset + seq_len]
            .view(seq_len, 1, -1)
            .to(device)
        )
        freqs = torch.cat([clean_freqs, noisy_freqs], dim=0)
        self_attn_mask = self.build_action_self_attention_mask(
            noisy_seq_len=seq_len,
            chunk_size=effective_chunk_size,
            device=device,
            clean_seq_len=clean_seq_len,
            noisy_position_offset=noisy_position_offset,
        )
        return t, t_mod, freqs, self_attn_mask

    def _compute_single_branch_modulations(
        self,
        *,
        batch_size: int,
        seq_len: int,
        timestep: torch.Tensor,
        timestep_is_tokenwise: bool,
        noisy_position_offset: int,
        effective_chunk_size: int,
        single_branch_chunk_causal: bool,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Compute t, t_mod, freqs, self_attn_mask for single-branch (no teacher forcing) mode."""
        if timestep_is_tokenwise:
            t = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, timestep.reshape(-1))
            ).view(batch_size, seq_len, self.hidden_dim)
            t_mod = self.time_projection(t).view(
                batch_size, seq_len, 6, self.hidden_dim
            )
        else:
            t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
            t_mod = self.time_projection(t).unflatten(1, (6, self.hidden_dim))
        freqs = (
            self.freqs[noisy_position_offset : noisy_position_offset + seq_len]
            .view(seq_len, 1, -1)
            .to(device)
        )
        self_attn_mask = (
            self.build_single_branch_chunk_causal_mask(
                seq_len=seq_len,
                chunk_size=effective_chunk_size,
                device=device,
            )
            if single_branch_chunk_causal
            else None
        )
        return t, t_mod, freqs, self_attn_mask

    def pre_dit(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        obs_context: Optional[torch.Tensor] = None,
        obs_context_mask: Optional[torch.Tensor] = None,
        clean_action_tokens: Optional[torch.Tensor] = None,
        clean_timestep: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
        noisy_position_offset: int = 0,
        single_branch_chunk_causal: bool = False,
        obs_chunk_offset: int = 0,
        obs_context_causal: bool = False,
        obs_proprio_tokens_per_chunk: int = 0,
        skip_context_embedding: bool = False,
    ) -> Dict[str, Any]:
        """
        predit mainly do following things:
        - MUSTDO:
          - encode noisy action tokens
          - project all context token
          - compute context mask
          - time modulations(support teacher forcing and single branch)
        - OPTIONAL
          - merge obs_context and text_context(context here) if have
          - encode clean actioon if have
        """
        if action_tokens.ndim != 3:
            raise ValueError(
                f"`action_tokens` must be 3D [B, T, action_dim], got shape {tuple(action_tokens.shape)}"
            )
        if action_tokens.shape[2] != self.action_dim:
            raise ValueError(
                f"`action_tokens` last dim must be {self.action_dim}, got {action_tokens.shape[2]}"
            )
        if context.ndim != 3:
            raise ValueError(
                f"`context` must be 3D [B, L, D], got shape {tuple(context.shape)}"
            )

        batch_size = action_tokens.shape[0]
        seq_len = action_tokens.shape[1]
        if context.shape[0] != batch_size:
            raise ValueError(
                f"Batch mismatch between action tokens and text context: {batch_size} vs {context.shape[0]}"
            )

        # --- context mask ---
        if context_mask is None:
            context_mask = torch.ones(
                (batch_size, context.shape[1]), dtype=torch.bool, device=context.device
            )
        else:
            if context_mask.ndim != 2:
                raise ValueError(
                    f"`context_mask` must be 2D [B, L], got shape {tuple(context_mask.shape)}"
                )
            if (
                context_mask.shape[0] != batch_size
                or context_mask.shape[1] != context.shape[1]
            ):
                raise ValueError(
                    f"`context_mask` shape must match `context` shape [B, L], got {tuple(context_mask.shape)} vs {tuple(context.shape)}"
                )

        # --- merge obs context ---
        context, flat_obs_context_mask, obs_num_chunks, obs_tokens_per_chunk = (
            self._merge_obs_context(
                context=context,
                context_mask=context_mask,
                batch_size=batch_size,
                obs_context=obs_context,
                obs_context_mask=obs_context_mask,
            )
        )

        # --- validate clean action tokens ---
        clean_seq_len = 0
        if clean_action_tokens is not None:
            if clean_action_tokens.ndim != 3:
                raise ValueError(
                    "`clean_action_tokens` must be 3D [B, T, action_dim], "
                    f"got shape {tuple(clean_action_tokens.shape)}"
                )
            if (
                clean_action_tokens.shape[0] != batch_size
                or clean_action_tokens.shape[2] != self.action_dim
            ):
                raise ValueError(
                    "`clean_action_tokens` must match batch/action_dim of `action_tokens`, "
                    f"got {tuple(clean_action_tokens.shape)} vs batch={batch_size}, action_dim={self.action_dim}"
                )
            clean_seq_len = int(clean_action_tokens.shape[1])

        # --- validate positions ---
        effective_chunk_size = int(chunk_size or self.action_chunk_size)
        if effective_chunk_size <= 0:
            raise ValueError(f"`chunk_size` must be > 0, got {effective_chunk_size}")
        self._validate_positions(
            seq_len=seq_len,
            clean_seq_len=clean_seq_len,
            effective_chunk_size=effective_chunk_size,
            noisy_position_offset=noisy_position_offset,
        )

        # --- encode tokens ---
        tokens = self.action_encoder(action_tokens)
        timestep, timestep_is_tokenwise = self._normalize_tokenwise_timestep(
            timestep,
            batch_size=batch_size,
            seq_len=seq_len,
            chunk_size=effective_chunk_size,
            name="timestep",
        )
        if clean_action_tokens is not None:
            clean_tokens = self.action_encoder(clean_action_tokens)
            tokens = torch.cat([clean_tokens, tokens], dim=1)
        context_emb = None if skip_context_embedding else self.text_embedding(context)

        # --- context attention mask ---
        total_seq_len = tokens.shape[1]
        assert context_mask is not None
        base_context_attn_mask = context_mask.unsqueeze(1).expand(-1, total_seq_len, -1)
        context_attn_mask = self._build_context_attention_mask(
            base_context_attn_mask=base_context_attn_mask,
            total_seq_len=total_seq_len,
            clean_seq_len=clean_seq_len,
            seq_len=seq_len,
            effective_chunk_size=effective_chunk_size,
            noisy_position_offset=noisy_position_offset,
            flat_obs_context_mask=flat_obs_context_mask,
            obs_num_chunks=obs_num_chunks,
            obs_tokens_per_chunk=obs_tokens_per_chunk,
            obs_chunk_offset=obs_chunk_offset,
            obs_context_causal=obs_context_causal,
            obs_proprio_tokens_per_chunk=obs_proprio_tokens_per_chunk,
            device=tokens.device,
        )

        # --- time modulations ---
        if clean_seq_len > 0:
            t, t_mod, freqs, self_attn_mask = self._compute_teacher_forcing_modulations(
                batch_size=batch_size,
                seq_len=seq_len,
                clean_seq_len=clean_seq_len,
                timestep=timestep,
                timestep_is_tokenwise=timestep_is_tokenwise,
                clean_timestep=clean_timestep,
                noisy_position_offset=noisy_position_offset,
                effective_chunk_size=effective_chunk_size,
                device=tokens.device,
            )
        else:
            t, t_mod, freqs, self_attn_mask = self._compute_single_branch_modulations(
                batch_size=batch_size,
                seq_len=seq_len,
                timestep=timestep,
                timestep_is_tokenwise=timestep_is_tokenwise,
                noisy_position_offset=noisy_position_offset,
                effective_chunk_size=effective_chunk_size,
                single_branch_chunk_causal=single_branch_chunk_causal,
                device=tokens.device,
            )

        return {
            "tokens": tokens,
            "freqs": freqs,
            "t": t,
            "t_mod": t_mod,
            "context": context_emb,
            "context_mask": context_attn_mask,
            "self_attn_mask": self_attn_mask,
            "meta": {
                "batch_size": batch_size,
                "seq_len": seq_len,
                "clean_seq_len": clean_seq_len,
                "total_seq_len": total_seq_len,
                "chunk_size": effective_chunk_size,
                "noisy_position_offset": int(noisy_position_offset),
                "single_branch_chunk_causal": bool(single_branch_chunk_causal),
            },
        }

    def post_dit(self, tokens: torch.Tensor, pre_state: Dict[str, Any]) -> torch.Tensor:
        clean_seq_len = int(pre_state["meta"].get("clean_seq_len", 0))
        if clean_seq_len > 0:
            tokens = tokens[:, clean_seq_len:]
        return self.head(tokens)

    def build_action_self_attention_mask(
        self,
        noisy_seq_len: int,
        chunk_size: int,
        device: torch.device,
        clean_seq_len: int = 0,
        noisy_position_offset: int = 0,
    ) -> torch.Tensor:
        if noisy_seq_len <= 0:
            raise ValueError(f"`noisy_seq_len` must be positive, got {noisy_seq_len}")
        if chunk_size <= 0:
            raise ValueError(f"`chunk_size` must be positive, got {chunk_size}")
        if noisy_seq_len % chunk_size != 0:
            raise ValueError(
                f"`noisy_seq_len` ({noisy_seq_len}) must be divisible by `chunk_size` ({chunk_size})."
            )
        if clean_seq_len % chunk_size != 0:
            raise ValueError(
                f"`clean_seq_len` ({clean_seq_len}) must be divisible by `chunk_size` ({chunk_size})."
            )
        if noisy_position_offset % chunk_size != 0:
            raise ValueError(
                "`noisy_position_offset` must align to chunk_size, "
                f"got {noisy_position_offset} and {chunk_size}."
            )

        if clean_seq_len == 0:
            return torch.ones(
                (noisy_seq_len, noisy_seq_len), dtype=torch.bool, device=device
            )

        num_clean_chunks = clean_seq_len // chunk_size
        num_noisy_chunks = noisy_seq_len // chunk_size
        total_seq_len = clean_seq_len + noisy_seq_len
        mask = torch.zeros(
            (total_seq_len, total_seq_len), dtype=torch.bool, device=device
        )

        clean_chunk_ids = torch.arange(
            num_clean_chunks, device=device
        ).repeat_interleave(chunk_size)
        noisy_chunk_start = noisy_position_offset // chunk_size
        noisy_chunk_ids = (
            torch.arange(num_noisy_chunks, device=device) + noisy_chunk_start
        ).repeat_interleave(chunk_size)

        clean_to_clean = clean_chunk_ids.unsqueeze(1) >= clean_chunk_ids.unsqueeze(0)
        mask[:clean_seq_len, :clean_seq_len] = clean_to_clean

        noisy_to_noisy = noisy_chunk_ids.unsqueeze(1) == noisy_chunk_ids.unsqueeze(0)
        mask[clean_seq_len:, clean_seq_len:] = noisy_to_noisy

        noisy_to_clean = noisy_chunk_ids.unsqueeze(1) > clean_chunk_ids.unsqueeze(0)
        mask[clean_seq_len:, :clean_seq_len] = noisy_to_clean
        return mask

    def build_single_branch_chunk_causal_mask(
        self,
        seq_len: int,
        chunk_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if seq_len <= 0:
            raise ValueError(f"`seq_len` must be positive, got {seq_len}")
        if chunk_size <= 0:
            raise ValueError(f"`chunk_size` must be positive, got {chunk_size}")
        if seq_len % chunk_size != 0:
            raise ValueError(
                f"`seq_len` ({seq_len}) must be divisible by `chunk_size` ({chunk_size})."
            )
        num_chunks = seq_len // chunk_size
        chunk_ids = torch.arange(num_chunks, device=device).repeat_interleave(
            chunk_size
        )
        return chunk_ids.unsqueeze(1) >= chunk_ids.unsqueeze(0)

    def forward(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        obs_context: Optional[torch.Tensor] = None,
        obs_context_mask: Optional[torch.Tensor] = None,
        clean_action_tokens: Optional[torch.Tensor] = None,
        clean_timestep: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
        noisy_position_offset: int = 0,
        single_branch_chunk_causal: bool = False,
    ) -> torch.Tensor:
        pre_state = self.pre_dit(
            action_tokens=action_tokens,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            obs_context=obs_context,
            obs_context_mask=obs_context_mask,
            clean_action_tokens=clean_action_tokens,
            clean_timestep=clean_timestep,
            chunk_size=chunk_size,
            noisy_position_offset=noisy_position_offset,
            single_branch_chunk_causal=single_branch_chunk_causal,
        )
        x = pre_state["tokens"]
        context = pre_state["context"]
        t_mod = pre_state["t_mod"]
        freqs = pre_state["freqs"]
        context_mask = pre_state["context_mask"]
        self_attn_mask = pre_state["self_attn_mask"]

        for block in self.blocks:
            if self.use_gradient_checkpointing:
                x = gradient_checkpoint_forward(
                    block,
                    self.use_gradient_checkpointing,
                    x,
                    context,
                    t_mod,
                    freqs,
                    context_mask=context_mask,
                    self_attn_mask=self_attn_mask,
                )
            else:
                x = block(
                    x,
                    context,
                    t_mod,
                    freqs,
                    context_mask=context_mask,
                    self_attn_mask=self_attn_mask,
                )

        return self.post_dit(x, pre_state)
