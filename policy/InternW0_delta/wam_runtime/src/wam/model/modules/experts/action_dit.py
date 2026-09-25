import os
import math
import torch
import torch.nn as nn
from typing import Any, Dict, Optional

from wam.utils.logging_config import get_logger

from ...backbones.wan22.helpers.gradient import gradient_checkpoint_forward
from ...backbones.wan22.wan_video_dit import (
    CrossAttention,
    DiTBlock,
    sinusoidal_embedding_1d,
    precompute_freqs_cis,
)

logger = get_logger(__name__)


class FeatureScaledLinear(nn.Linear):
    """Linear projection with a non-persistent task-defined input scale."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        dimension_valid_mask: Optional[list[bool]],
        valid_input_scale: float,
    ) -> None:
        super().__init__(int(in_features), int(out_features))
        scale_value = float(valid_input_scale)
        if not math.isfinite(scale_value) or scale_value <= 0.0:
            raise ValueError(
                f"valid_input_scale must be finite and positive, got {scale_value}."
            )
        if dimension_valid_mask is None:
            raise ValueError(
                "dimension_valid_mask is required when valid_input_scale != 1.0."
            )
        valid = torch.as_tensor(dimension_valid_mask, dtype=torch.bool)
        if valid.ndim != 1 or int(valid.numel()) != int(in_features):
            raise ValueError(
                f"dimension_valid_mask must be [{in_features}], got {tuple(valid.shape)}"
            )
        if not bool(valid.any().item()):
            raise ValueError("dimension_valid_mask must contain at least one valid dimension.")
        scale = torch.ones(int(in_features), dtype=torch.float32)
        scale[valid] = scale_value
        self.register_buffer("feature_scale", scale, persistent=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(
            input
            * self.feature_scale.to(device=input.device, dtype=input.dtype)
        )


class ActionHead(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int, eps: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.proj = nn.Linear(hidden_dim, out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, hidden_dim) / hidden_dim**0.5)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        shift, scale = (self.modulation.to(dtype=t.dtype, device=t.device) + t.unsqueeze(1)).chunk(2, dim=1)
        shift = shift.squeeze(1)
        scale = scale.squeeze(1)
        return self.proj(self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1))


class ActionDiT(nn.Module):
    ACTION_BACKBONE_SKIP_PREFIXES = (
        "action_encoder.",
        "head.",
    )
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
        use_gradient_checkpointing: bool = False,
        dimension_valid_mask: Optional[list[bool]] = None,
        valid_input_scale: float = 1.0,
        physical_time_rope_enabled: bool = True,
        rope_base_hz: float = 30.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.ffn_dim = ffn_dim
        self.text_dim = text_dim
        self.freq_dim = freq_dim
        self.eps = eps
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.physical_time_rope_enabled = bool(physical_time_rope_enabled)
        self.rope_base_hz = float(rope_base_hz)
        if not math.isfinite(self.rope_base_hz) or self.rope_base_hz <= 0.0:
            raise ValueError(
                f"`rope_base_hz` must be finite and positive, got {rope_base_hz}."
            )

        if num_heads <= 0:
            raise ValueError(f"`num_heads` must be > 0, got {num_heads}")
        if attn_head_dim <= 0:
            raise ValueError(f"`attn_head_dim` must be > 0, got {attn_head_dim}")
        if attn_head_dim % 2 != 0:
            raise ValueError(f"`attn_head_dim` must be even for RoPE, got {attn_head_dim}")

        if float(valid_input_scale) == 1.0:
            self.action_encoder = nn.Linear(action_dim, hidden_dim)
        else:
            self.action_encoder = FeatureScaledLinear(
                action_dim,
                hidden_dim,
                dimension_valid_mask=dimension_valid_mask,
                valid_input_scale=float(valid_input_scale),
            )
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
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
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
        rope_inv_freq = 1.0 / (
            10000.0
            ** (
                torch.arange(0, attn_head_dim, 2, dtype=torch.float64)
                / float(attn_head_dim)
            )
        )
        # Match dev/h-1@910c635 and the 0820-all pretraining runtime exactly:
        # this non-persistent buffer follows ActionDiT's execution dtype while
        # remaining absent from checkpoint state dictionaries.
        self.register_buffer(
            "rope_inv_freq", rope_inv_freq, persistent=False
        )

        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.vlm_context_dim: Optional[int] = None

    @property
    def uses_vlm_conditioning(self) -> bool:
        return self.vlm_context_dim is not None

    def configure_vlm_conditioning(self, context_dim: int) -> int:
        """Use raw Qwen hidden states in every existing action cross-attention.

        The action-side Q/output projections are retained from the pretrained
        ActionDiT. K/V are newly initialized for the Qwen hidden dimension.
        """
        context_dim = int(context_dim)
        if context_dim <= 0:
            raise ValueError(f"`context_dim` must be positive, got {context_dim}.")
        if self.uses_vlm_conditioning:
            if self.vlm_context_dim != context_dim:
                raise ValueError(
                    "Action VLM conditioning is already configured with "
                    f"context_dim={self.vlm_context_dim}; got {context_dim}."
                )
            return 0

        ref = next(self.parameters())
        replaced = 0
        for block in self.blocks:
            source = block.cross_attn
            vlm_cross_attn = CrossAttention(
                hidden_dim=int(self.hidden_dim),
                attn_head_dim=int(self.attn_head_dim),
                num_heads=int(self.num_heads),
                eps=float(self.eps),
                context_dim=context_dim,
            ).to(device=ref.device, dtype=ref.dtype)
            vlm_cross_attn.q.load_state_dict(source.q.state_dict(), strict=True)
            vlm_cross_attn.o.load_state_dict(source.o.state_dict(), strict=True)
            vlm_cross_attn.norm_q.load_state_dict(
                source.norm_q.state_dict(), strict=True
            )
            vlm_cross_attn.to(device=ref.device, dtype=ref.dtype)
            block.cross_attn = vlm_cross_attn
            replaced += 1

        self.vlm_context_dim = context_dim
        return replaced

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
    ) -> "ActionDiT":
        if action_dit_config is None:
            raise ValueError("`action_dit_config` is required for ActionDiT.from_pretrained().")
        if skip_dit_load_from_pretrain:
            logger.info(
                "Skipping ActionDiT pretrained load (`skip_dit_load_from_pretrain=True`); "
                "initializing action expert randomly and expecting checkpoint override."
            )
            return cls(**action_dit_config).to(device=device, dtype=torch_dtype)
        if not action_dit_pretrained_path:
            logger.info("No `action_dit_pretrained_path` provided, initializing ActionDiT with random weights.")
            return cls(**action_dit_config).to(device=device, dtype=torch_dtype)
        from pathlib import Path
        p = Path(action_dit_pretrained_path)
        if not p.is_absolute():
            current_file = Path(__file__).resolve()
            repo_root = next(
                (parent for parent in current_file.parents if (parent / "pyproject.toml").is_file()),
                current_file.parents[5],
            )
            p = repo_root / p
        action_dit_pretrained_path = str(p)
        if not os.path.isfile(action_dit_pretrained_path):
            raise FileNotFoundError(
                f"`action_dit_pretrained_path` does not exist: {action_dit_pretrained_path}"
            )

        action_cfg = dict(action_dit_config)
        action_expert = cls(**action_cfg).to(device=device, dtype=torch_dtype)
        action_state = action_expert.state_dict()
        expected_backbone_keys = cls.backbone_key_set(action_state.keys())

        payload = torch.load(action_dit_pretrained_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(
                f"Invalid action backbone payload type from {action_dit_pretrained_path}: {type(payload)}"
            )

        policy = payload.get("policy", {})
        if policy:
            logger.debug("ActionDiT backbone payload policy: %s", policy)

        meta = payload.get("meta")
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
                raise ValueError(f"`meta.{key}` missing in {action_dit_pretrained_path}")
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
            merged_state[key] = value.to(device=target.device, dtype=target.dtype)

        action_expert.load_state_dict(merged_state, strict=True)
        logger.info(
            "Loaded ActionDiT backbone from %s (keys=%d; random_kept_prefixes=%s).",
            action_dit_pretrained_path,
            len(expected_backbone_keys),
            list(cls.ACTION_BACKBONE_SKIP_PREFIXES),
        )
        return action_expert.to(device=device, dtype=torch_dtype)

    def pre_dit(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        action_step_ids: Optional[torch.Tensor] = None,
        action_hz: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if action_tokens.ndim != 3:
            raise ValueError(
                f"`action_tokens` must be 3D [B, T, action_dim], got shape {tuple(action_tokens.shape)}"
            )
        if action_tokens.shape[2] != self.action_dim:
            raise ValueError(
                f"`action_tokens` last dim must be {self.action_dim}, got {action_tokens.shape[2]}"
            )
        if timestep.ndim != 1:
            raise ValueError(f"`timestep` must be 1D [B] or [1], got shape {tuple(timestep.shape)}")
        if context.ndim != 3:
            raise ValueError(
                f"`context` must be 3D [B, L, D], got shape {tuple(context.shape)}"
            )

        batch_size = action_tokens.shape[0]
        if context.shape[0] != batch_size:
            raise ValueError(
                f"Batch mismatch between action tokens and context: {batch_size} vs {context.shape[0]}"
            )
        if self.uses_vlm_conditioning:
            if int(context.shape[2]) != int(self.vlm_context_dim):
                raise ValueError(
                    "Raw Qwen context width must match configured VLM width: "
                    f"{context.shape[2]} vs {self.vlm_context_dim}."
                )
        if timestep.shape[0] not in (1, batch_size):
            raise ValueError(
                f"`timestep` length must be 1 or batch_size({batch_size}), got {timestep.shape[0]}"
            )
        if timestep.shape[0] == 1 and batch_size > 1:
            if self.training:
                raise ValueError("During training, action timestep length must match batch_size.")
            timestep = timestep.expand(batch_size)

        if context_mask is None:
            context_mask = torch.ones(
                (batch_size, context.shape[1]), dtype=torch.bool, device=context.device
            )
        else:
            if context_mask.ndim != 2:
                raise ValueError(f"`context_mask` must be 2D [B, L], got shape {tuple(context_mask.shape)}")
            if context_mask.shape[0] != batch_size or context_mask.shape[1] != context.shape[1]:
                raise ValueError(
                    f"`context_mask` shape must match `context` shape [B, L], got {tuple(context_mask.shape)} vs {tuple(context.shape)}"
                )

        if action_step_ids is not None and action_hz is not None:
            raise ValueError(
                "`action_step_ids` and `action_hz` are mutually exclusive."
            )
        if action_hz is not None and not self.physical_time_rope_enabled:
            raise ValueError(
                "`action_hz` was provided while physical-time Action RoPE is disabled."
            )
        if action_step_ids is not None:
            if torch.is_tensor(action_step_ids):
                action_step_ids = action_step_ids.to(dtype=torch.long)
            else:
                action_step_ids = torch.as_tensor(action_step_ids, dtype=torch.long)

        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.hidden_dim))

        tokens = self.action_encoder(action_tokens)
        action_seq_len = int(tokens.shape[1])
        if self.uses_vlm_conditioning:
            context_emb = context.to(device=tokens.device, dtype=tokens.dtype)
        else:
            context_emb = self.text_embedding(context)
        seq_len = int(tokens.shape[1])
        context_attn_mask = context_mask.unsqueeze(1).expand(-1, seq_len, -1)
        if action_hz is not None:
            if torch.is_tensor(action_hz):
                action_hz = action_hz.detach().to(
                    device=tokens.device, dtype=torch.float32
                )
            else:
                action_hz = torch.as_tensor(
                    action_hz, device=tokens.device, dtype=torch.float32
                )
            if action_hz.ndim == 0:
                action_hz = action_hz.reshape(1)
            elif action_hz.ndim == 2 and int(action_hz.shape[1]) == 1:
                action_hz = action_hz[:, 0]
            elif action_hz.ndim != 1:
                raise ValueError(
                    "`action_hz` must be scalar, [B], or [B,1], "
                    f"got {tuple(action_hz.shape)}."
                )
            if int(action_hz.numel()) == 1 and batch_size > 1:
                action_hz = action_hz.expand(batch_size)
            if int(action_hz.numel()) != batch_size:
                raise ValueError(
                    f"`action_hz` must contain one value per sample ({batch_size}), "
                    f"got {int(action_hz.numel())}."
                )
            # Dataset/input and online-inference boundaries validate values on
            # CPU. Avoid a device synchronization in every training/denoise
            # call; retain direct-call validation on CPU for diagnostics.
            if action_hz.device.type == "cpu" and (
                not bool(torch.isfinite(action_hz).all().item())
                or not bool((action_hz > 0).all().item())
            ):
                raise ValueError(
                    "`action_hz` values must be finite and positive."
                )
            positions = torch.arange(
                action_seq_len,
                device=tokens.device,
                dtype=torch.float64,
            ).unsqueeze(0)
            positions = positions * (
                self.rope_base_hz / action_hz.to(dtype=torch.float64)
            ).unsqueeze(1)
            phases = positions.unsqueeze(-1) * self.rope_inv_freq.to(
                device=tokens.device
            ).view(1, 1, -1)
            action_freqs = torch.polar(
                torch.ones_like(phases), phases
            ).unsqueeze(2)
        elif action_step_ids is None:
            action_freqs = self.freqs[:action_seq_len].view(
                action_seq_len, 1, -1
            ).to(device=tokens.device)
        else:
            if action_step_ids.ndim == 1:
                if int(action_step_ids.shape[0]) != int(action_seq_len):
                    raise ValueError(
                        f"`action_step_ids` must have length {action_seq_len}, got {tuple(action_step_ids.shape)}"
                    )
                index_ids = action_step_ids.to(device=self.freqs.device, non_blocking=True)
                action_freqs = self.freqs[index_ids].unsqueeze(1).to(
                    device=tokens.device
                )
            elif action_step_ids.ndim == 2:
                if tuple(action_step_ids.shape) != (batch_size, action_seq_len):
                    raise ValueError(
                        f"`action_step_ids` must be [B,{action_seq_len}], got {tuple(action_step_ids.shape)}"
                    )
                freq_cache = self.freqs.to(device=tokens.device)
                action_freqs = freq_cache[
                    action_step_ids.to(device=tokens.device)
                ].unsqueeze(2)
            else:
                raise ValueError(f"`action_step_ids` must be [S] or [B,S], got {tuple(action_step_ids.shape)}")

        return {
            "tokens": tokens,
            "freqs": action_freqs,
            "t": t,
            "t_mod": t_mod,
            "context": context_emb,
            "context_mask": context_attn_mask,
            "meta": {
                "batch_size": batch_size,
                "seq_len": seq_len,
                "action_seq_len": action_seq_len,
            },
        }

    def post_dit(self, tokens: torch.Tensor, pre_state: Dict[str, Any]) -> torch.Tensor:
        return self.head(tokens)

    def forward(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        action_hz: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pre_state = self.pre_dit(
            action_tokens=action_tokens,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            action_hz=action_hz,
        )
        x = pre_state["tokens"]
        context = pre_state["context"]
        t_mod = pre_state["t_mod"]
        freqs = pre_state["freqs"]
        context_mask = pre_state["context_mask"]

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
                )
            else:
                x = block(x, context, t_mod, freqs, context_mask=context_mask)

        return self.post_dit(x, pre_state)
