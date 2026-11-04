from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from diffusers.models.embeddings import TimestepEmbedding, Timesteps

from flexwam.utils.logging_config import get_logger

from flexwam.models.backbone.lingbot_video.lingbot_video import LingBotVideoRotaryEmbedding, LingBotVideoTransformerBlock

logger = get_logger(__name__)


class LingBotActionDiT(nn.Module):
    # These layers are task/input adapters and are intentionally initialized
    # outside the pretrained action backbone.
    ACTION_BACKBONE_SKIP_PREFIXES = ("action_encoder.", "proj_out.")
    ACTION_BACKBONE_META_KEYS = (
        "hidden_size",
        "intermediate_size",
        "depth",
        "num_attention_heads",
        "attention_head_dim",
        "freq_dim",
        "norm_eps",
    )

    def __init__(
        self,
        action_dim: int,
        hidden_size: int = 2048,
        intermediate_size: int = 6144,
        depth: int = 24,
        num_attention_heads: int = 16,
        attention_head_dim: Optional[int] = None,
        freq_dim: int = 256,
        axes_dims: tuple[int, int, int] = (32, 48, 48),
        axes_lens: tuple[int, int, int] = (8192, 1024, 1024),
        rope_theta: float = 256.0,
        norm_eps: float = 1e-6,
        qkv_bias: bool = False,
        out_bias: bool = True,
        timestep_mlp_bias: bool = True,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.hidden_size = int(hidden_size)
        self.hidden_dim = self.hidden_size
        self.intermediate_size = int(intermediate_size)
        self.num_attention_heads = int(num_attention_heads)
        self.num_heads = self.num_attention_heads
        if attention_head_dim is None:
            if self.hidden_size % self.num_attention_heads != 0:
                raise ValueError(
                    f"`hidden_size` must be divisible by `num_attention_heads`, "
                    f"got {hidden_size} and {num_attention_heads}."
                )
            self.attention_head_dim = self.hidden_size // self.num_attention_heads
        else:
            self.attention_head_dim = int(attention_head_dim)
        self.attn_head_dim = self.attention_head_dim
        self.freq_dim = int(freq_dim)
        self.norm_eps = float(norm_eps)

        if self.attention_head_dim != sum(int(x) for x in axes_dims):
            raise ValueError(
                "`axes_dims` must sum to action attention head dim: "
                f"sum={sum(int(x) for x in axes_dims)} vs head_dim={self.attention_head_dim}."
            )

        self.action_encoder = nn.Linear(self.action_dim, self.hidden_size)
        self.time_embedder = TimestepEmbedding(
            in_channels=self.freq_dim,
            time_embed_dim=self.hidden_size,
            act_fn="silu",
            post_act_fn=None,
            out_dim=self.hidden_size,
            cond_proj_dim=None,
            sample_proj_bias=timestep_mlp_bias,
        )
        self.time_proj = Timesteps(num_channels=self.freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_modulation = nn.Sequential(nn.SiLU(), nn.Linear(self.hidden_size, 6 * self.hidden_size))
        self.blocks = nn.ModuleList(
            [
                LingBotVideoTransformerBlock(
                    self.hidden_size,
                    self.intermediate_size,
                    self.num_attention_heads,
                    self.attention_head_dim,
                    eps=self.norm_eps,
                    qkv_bias=bool(qkv_bias),
                    out_bias=bool(out_bias),
                )
                for _ in range(int(depth))
            ]
        )
        self.proj_out = nn.Linear(self.hidden_size, self.action_dim, bias=out_bias)
        self.rope = LingBotVideoRotaryEmbedding(tuple(axes_dims), tuple(axes_lens), theta=rope_theta)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)

    @classmethod
    def backbone_key_set(cls, keys) -> set[str]:
        return {
            key
            for key in keys
            if not any(key.startswith(prefix) for prefix in cls.ACTION_BACKBONE_SKIP_PREFIXES)
        }

    @classmethod
    def from_pretrained(
        cls,
        action_dit_config: dict[str, Any],
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "LingBotActionDiT":
        if action_dit_config is None:
            raise ValueError("`action_dit_config` is required for LingBotActionDiT.from_pretrained().")
        if skip_dit_load_from_pretrain:
            logger.info(
                "Skipping LingBotActionDiT pretrained load (`skip_dit_load_from_pretrain=True`); "
                "initializing action expert randomly and expecting checkpoint override."
            )
            return cls(**action_dit_config).to(device=device, dtype=torch_dtype)
        if not action_dit_pretrained_path:
            logger.info("No `action_dit_pretrained_path` provided, initializing LingBotActionDiT randomly.")
            return cls(**action_dit_config).to(device=device, dtype=torch_dtype)

        path = Path(action_dit_pretrained_path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[4] / path
        if not os.path.isfile(path):
            raise FileNotFoundError(f"`action_dit_pretrained_path` does not exist: {path}")

        action_expert = cls(**dict(action_dit_config)).to(device=device, dtype=torch_dtype)
        action_state = action_expert.state_dict()
        expected_backbone_keys = cls.backbone_key_set(action_state.keys())

        payload = torch.load(str(path), map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid LingBot action backbone payload type from {path}: {type(payload)}")
        policy = payload.get("policy", {})
        if policy:
            logger.info("LingBotActionDiT backbone payload policy: %s", policy)

        meta = payload.get("meta")
        if not isinstance(meta, dict):
            raise ValueError(f"`meta` must be a dict in {path}")
        expected_meta = {
            "hidden_size": int(action_expert.hidden_size),
            "intermediate_size": int(action_expert.intermediate_size),
            "depth": int(len(action_expert.blocks)),
            "num_attention_heads": int(action_expert.num_attention_heads),
            "attention_head_dim": int(action_expert.attention_head_dim),
            "freq_dim": int(action_expert.freq_dim),
            "norm_eps": float(action_expert.norm_eps),
        }
        for key in cls.ACTION_BACKBONE_META_KEYS:
            if key not in meta:
                raise ValueError(f"`meta.{key}` missing in {path}")
            if key == "norm_eps":
                if abs(float(meta[key]) - float(expected_meta[key])) > 1e-12:
                    raise ValueError(f"`meta.{key}` mismatch: expected {expected_meta[key]}, got {meta[key]}")
            elif int(meta[key]) != int(expected_meta[key]):
                raise ValueError(f"`meta.{key}` mismatch: expected {expected_meta[key]}, got {meta[key]}")

        backbone_state_dict = payload.get("backbone_state_dict")
        if not isinstance(backbone_state_dict, dict):
            raise ValueError(f"`backbone_state_dict` must be a dict in {path}, got {type(backbone_state_dict)}")

        provided_keys = set(backbone_state_dict.keys())
        missing_keys = sorted(expected_backbone_keys - provided_keys)
        unexpected_keys = sorted(provided_keys - expected_backbone_keys)
        if missing_keys or unexpected_keys:
            raise ValueError(
                "LingBotActionDiT backbone key mismatch. "
                f"missing={missing_keys[:10]}{'...' if len(missing_keys) > 10 else ''}, "
                f"unexpected={unexpected_keys[:10]}{'...' if len(unexpected_keys) > 10 else ''}"
            )

        merged_state = dict(action_state)
        for key in expected_backbone_keys:
            value = backbone_state_dict[key]
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"`backbone_state_dict[{key}]` must be a tensor, got {type(value)}")
            target = merged_state[key]
            if tuple(value.shape) != tuple(target.shape):
                raise ValueError(f"Shape mismatch for `{key}`: expected {tuple(target.shape)}, got {tuple(value.shape)}")
            merged_state[key] = value.to(device=target.device, dtype=target.dtype)

        action_expert.load_state_dict(merged_state, strict=True)
        logger.info(
            "Loaded LingBotActionDiT backbone from %s (keys=%d; random_kept_prefixes=%s).",
            path,
            len(expected_backbone_keys),
            list(cls.ACTION_BACKBONE_SKIP_PREFIXES),
        )
        return action_expert.to(device=device, dtype=torch_dtype)

    @staticmethod
    def _action_position_ids(
        action_seq_len: int,
        prefix_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        # Keep the pretrained action-token RoPE positions unchanged.  The
        # newly introduced state/force prefix uses zero positions, matching
        # LingBot's video language-prefix convention.
        prefix_pos = torch.zeros((prefix_len,), device=device, dtype=torch.long)
        action_pos = torch.arange(action_seq_len, device=device, dtype=torch.long)
        pos = torch.cat([prefix_pos, action_pos], dim=0)
        zeros = torch.zeros_like(pos)
        return torch.stack([pos, zeros, zeros], dim=-1)

    def pre_dit(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        state_tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if action_tokens.ndim != 3:
            raise ValueError(f"`action_tokens` must be [B,T,D], got {tuple(action_tokens.shape)}.")
        if action_tokens.shape[2] != self.action_dim:
            raise ValueError(f"`action_tokens` last dim must be {self.action_dim}, got {action_tokens.shape[2]}.")
        batch_size, action_seq_len, _ = action_tokens.shape
        if context is not None and (context.ndim != 3 or context.shape[0] != batch_size):
            raise ValueError(
                "`context` must be [B,0,D] with matching batch dim, "
                f"got {tuple(context.shape)}, expected B={batch_size}."
            )
        if context is not None and context.shape[1] != 0:
            raise ValueError(
                "LingBotActionDiT does not accept a separate context prefix; "
                "language conditions the action branch through mixed attention with video tokens."
            )
        if context_mask is not None and context_mask.shape != (batch_size, 0):
            raise ValueError(
                "`context_mask` shape mismatch: "
                f"got {tuple(context_mask.shape)}, expected {(batch_size, 0)}."
            )
        if timestep.ndim == 0:
            timestep = timestep[None].expand(batch_size)
        elif timestep.ndim == 1 and timestep.shape[0] == 1 and batch_size > 1:
            timestep = timestep.expand(batch_size)
        if timestep.ndim != 1 or timestep.shape[0] != batch_size:
            raise ValueError(f"`timestep` must be [B] or scalar, got {tuple(timestep.shape)} for batch={batch_size}.")

        tokens = self.action_encoder(action_tokens)
        time_emb = self.time_proj(timestep).to(dtype=tokens.dtype, device=tokens.device)
        time_emb = self.time_embedder(time_emb)
        t_mod = self.time_modulation(time_emb).view(batch_size, 6, self.hidden_size)
        context_tokens = tokens.new_empty((batch_size, 0, self.hidden_size))
        if state_tokens is not None:
            if (
                state_tokens.ndim != 3
                or state_tokens.shape[0] != batch_size
                or state_tokens.shape[2] != self.hidden_size
            ):
                raise ValueError(
                    "`state_tokens` must be [B,S,hidden_size], "
                    f"got {tuple(state_tokens.shape)}, expected B={batch_size}, D={self.hidden_size}."
                )
            context_tokens = torch.cat(
                [context_tokens, state_tokens.to(device=tokens.device, dtype=tokens.dtype)],
                dim=1,
            )
        prefix_len = int(context_tokens.shape[1])
        tokens = torch.cat([context_tokens, tokens], dim=1)
        seq_len = int(tokens.shape[1])
        if prefix_len > 0:
            # Pi0: time conditions actions only. Keep the layer AdaLN table on
            # the state prefix, but zero the timestep residual.
            t_mod = t_mod.unsqueeze(2).expand(batch_size, 6, seq_len, self.hidden_size).clone()
            t_mod[:, :, :prefix_len].zero_()
        pos_ids = self._action_position_ids(
            action_seq_len=action_seq_len,
            prefix_len=prefix_len,
            device=tokens.device,
        )
        freqs = self.rope(pos_ids).unsqueeze(0).expand(batch_size, -1, -1)
        return {
            "tokens": tokens,
            "freqs": freqs,
            "t": time_emb,
            "t_mod": t_mod,
            # The context has already been embedded into the action prefix.
            # No separate post-block cross-attention is needed for LingBot.
            "context": None,
            "context_mask": None,
            "meta": {
                "batch_size": batch_size,
                "seq_len": seq_len,
                "action_prefix_len": prefix_len,
                "action_token_len": action_seq_len,
            },
        }

    def post_dit(self, tokens: torch.Tensor, pre_state: Dict[str, Any]) -> torch.Tensor:
        prefix_len = int(pre_state.get("meta", {}).get("action_prefix_len", 0))
        if prefix_len > 0:
            tokens = tokens[:, prefix_len:, :]
        if tokens.shape[1] != int(pre_state.get("meta", {}).get("action_token_len", tokens.shape[1])):
            raise ValueError(
                "Action token length mismatch after removing prefix: "
                f"got {tokens.shape[1]}, expected {pre_state.get('meta', {}).get('action_token_len')}."
            )
        return self.proj_out(tokens)

    def forward(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pre_state = self.pre_dit(
            action_tokens=action_tokens,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
        )
        x = pre_state["tokens"]
        freqs = pre_state["freqs"]
        t_mod = pre_state["t_mod"]
        for block in self.blocks:
            x = block(x, freqs_cis=freqs, t_mod=t_mod)
        return self.post_dit(x, pre_state)
