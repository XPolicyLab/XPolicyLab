import os
import re
import torch
import torch.nn as nn
import torch.utils.checkpoint
from typing import Any, Dict, Optional, Sequence

from fastwam.utils.logging_config import get_logger

from .helpers.gradient import gradient_checkpoint_forward
from .wan_video_dit import (
    DiTBlock,
    flash_attention,
    modulate,
    rope_apply,
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
        shift, scale = (self.modulation.to(dtype=t.dtype, device=t.device) + t.unsqueeze(1)).chunk(2, dim=1)
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
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.ffn_dim = ffn_dim
        self.text_dim = text_dim
        self.freq_dim = freq_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim

        if num_heads <= 0:
            raise ValueError(f"`num_heads` must be > 0, got {num_heads}")
        if attn_head_dim <= 0:
            raise ValueError(f"`attn_head_dim` must be > 0, got {attn_head_dim}")
        if attn_head_dim % 2 != 0:
            raise ValueError(f"`attn_head_dim` must be even for RoPE, got {attn_head_dim}")

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

        self.use_gradient_checkpointing = use_gradient_checkpointing

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
        pretrained_layer_mapping: str | None = None,
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
            p = Path(__file__).resolve().parents[4] / p
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
            logger.info(f"ActionDiT backbone payload policy: {policy}")

        meta = payload.get("meta")
        if not isinstance(meta, dict):
            raise ValueError(f"`meta` must be a dict in {action_dit_pretrained_path}.")
        layer_mapping_mode = (
            None
            if pretrained_layer_mapping is None
            else str(pretrained_layer_mapping).strip().lower()
        )
        if layer_mapping_mode in {"", "none", "identity"}:
            layer_mapping_mode = None
        if layer_mapping_mode not in {None, "uniform"}:
            raise ValueError(
                "`pretrained_layer_mapping` must be null or 'uniform', got "
                f"{pretrained_layer_mapping!r}."
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
                raise ValueError(f"`meta.{key}` missing in {action_dit_pretrained_path}")
            expected_value = expected_meta[key]
            got_value = meta[key]
            if key == "eps":
                if abs(float(got_value) - float(expected_value)) > 1e-12:
                    raise ValueError(
                        f"`meta.{key}` mismatch in {action_dit_pretrained_path}: "
                        f"expected {expected_value}, got {got_value}"
                    )
            elif key == "num_layers" and layer_mapping_mode == "uniform":
                if int(got_value) <= 0:
                    raise ValueError(
                        f"`meta.num_layers` must be positive in {action_dit_pretrained_path}."
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

        source_num_layers = int(meta["num_layers"])
        target_num_layers = int(action_cfg["num_layers"])
        if layer_mapping_mode is None:
            layer_indices = tuple(range(target_num_layers))
            provided_keys = set(backbone_state_dict.keys())
            missing_keys = sorted(expected_backbone_keys - provided_keys)
            unexpected_keys = sorted(provided_keys - expected_backbone_keys)
            if missing_keys or unexpected_keys:
                raise ValueError(
                    "Action backbone key mismatch in preprocessed payload. "
                    f"missing={missing_keys[:10]}{'...' if len(missing_keys) > 10 else ''}, "
                    f"unexpected={unexpected_keys[:10]}{'...' if len(unexpected_keys) > 10 else ''}"
                )
        else:
            if target_num_layers == 1:
                layer_indices = (0,)
            else:
                # Endpoint-aligned uniform subsampling. For Qwen3.5 this maps
                # the 30-layer Wan ActionDiT initializer to 24 decoder-aligned
                # action blocks while retaining both the first and last block.
                layer_indices = tuple(
                    round(target_idx * (source_num_layers - 1) / (target_num_layers - 1))
                    for target_idx in range(target_num_layers)
                )
            if len(set(layer_indices)) != len(layer_indices):
                raise ValueError(
                    "Uniform pretrained layer mapping requires target depth <= source depth, "
                    f"got source={source_num_layers}, target={target_num_layers}."
                )

        merged_state = dict(action_state)
        block_key = re.compile(r"^blocks\.(\d+)\.(.+)$")
        for key in expected_backbone_keys:
            source_key = key
            match = block_key.match(key)
            if match is not None and layer_mapping_mode == "uniform":
                target_layer = int(match.group(1))
                source_key = f"blocks.{layer_indices[target_layer]}.{match.group(2)}"
            if source_key not in backbone_state_dict:
                raise ValueError(
                    "Action backbone key missing after pretrained layer mapping: "
                    f"target={key!r}, source={source_key!r}."
                )
            value = backbone_state_dict[source_key]
            if not isinstance(value, torch.Tensor):
                raise ValueError(
                    f"`backbone_state_dict[{source_key}]` must be torch.Tensor in {action_dit_pretrained_path}, "
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
            "Loaded ActionDiT backbone from %s (keys=%d; random_kept_prefixes=%s; "
            "layer_mapping=%s).",
            action_dit_pretrained_path,
            len(expected_backbone_keys),
            list(cls.ACTION_BACKBONE_SKIP_PREFIXES),
            (
                "identity"
                if layer_mapping_mode is None
                else f"uniform:{source_num_layers}->{target_num_layers}:{layer_indices}"
            ),
        )
        return action_expert.to(device=device, dtype=torch_dtype)

    @staticmethod
    def _forward_layerwise_mot_block(
        block: nn.Module,
        x: torch.Tensor,
        video_tokens: torch.Tensor,
        context: torch.Tensor,
        t_mod: torch.Tensor,
        freqs: torch.Tensor,
        context_mask: Optional[torch.Tensor],
        mixed_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply one causal Qwen/action mixed-attention block.

        Qwen has already advanced its own stream and never attends to action.
        The action query attends to the paired Qwen visual K/V and its own K/V
        in one softmax, matching the directional MoT mask used by FastWAM.
        Qwen visual K/V use the pretrained cross-attention projections because
        Qwen3.5's GatedDeltaNet layers do not expose standard Transformer K/V.
        """

        has_seq = t_mod.ndim == 4
        chunk_dim = 2 if has_seq else 1
        base_mod = block.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            base_mod + t_mod
        ).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2),
                scale_msa.squeeze(2),
                gate_msa.squeeze(2),
                shift_mlp.squeeze(2),
                scale_mlp.squeeze(2),
                gate_mlp.squeeze(2),
            )

        action_input = modulate(block.norm1(x), shift_msa, scale_msa)
        q_action = block.self_attn.norm_q(block.self_attn.q(action_input))
        k_action = block.self_attn.norm_k(block.self_attn.k(action_input))
        v_action = block.self_attn.v(action_input)
        q_action = rope_apply(q_action, freqs, block.num_heads)
        k_action = rope_apply(k_action, freqs, block.num_heads)

        # Keep image tokens as a distinct Qwen expert stream. The shared
        # visual adapter has already converted them to ActionDiT width.
        video_input = block.norm3(video_tokens)
        k_video = block.cross_attn.norm_k(block.cross_attn.k(video_input))
        v_video = block.cross_attn.v(video_input)

        mixed = flash_attention(
            q=q_action,
            k=torch.cat((k_video, k_action), dim=1),
            v=torch.cat((v_video, v_action), dim=1),
            num_heads=block.num_heads,
            ctx_mask=mixed_attention_mask,
        )
        x = block.gate(x, gate_msa, block.self_attn.o(mixed))

        # As in Wan/Ovis MoT, instruction/proprio context remains a separate
        # cross-attention path after the mixed expert attention.
        if context_mask is not None and context_mask.ndim == 3:
            context_mask = context_mask.unsqueeze(1)
        x = x + block.cross_attn(block.norm3(x), context, ctx_mask=context_mask)
        mlp_input = modulate(block.norm2(x), shift_mlp, scale_mlp)
        return block.gate(x, gate_mlp, block.ffn(mlp_input))

    def prepare_layerwise_mot_visual_tokens(
        self,
        visual_contexts: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
        """Embed one Qwen visual context per ActionDiT block."""

        if len(visual_contexts) != len(self.blocks):
            raise ValueError(
                "Layerwise MoT requires one Qwen visual context per ActionDiT block: "
                f"got {len(visual_contexts)} contexts and {len(self.blocks)} blocks."
            )
        embedded = []
        for layer_idx, visual_context in enumerate(visual_contexts):
            if visual_context.ndim != 3:
                raise ValueError(
                    f"`visual_contexts[{layer_idx}]` must be [B,V,D], got "
                    f"{tuple(visual_context.shape)}."
                )
            if visual_context.shape[-1] != self.text_dim:
                raise ValueError(
                    f"`visual_contexts[{layer_idx}]` last dim must be text_dim={self.text_dim}, "
                    f"got {visual_context.shape[-1]}."
                )
            embedded.append(self.text_embedding(visual_context))
        return tuple(embedded)

    def forward_layerwise_mot(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        layerwise_visual_tokens: Sequence[torch.Tensor],
        context_mask: Optional[torch.Tensor] = None,
        visual_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run 1:1 Qwen/ActionDiT layerwise causal mixed attention.

        ``layerwise_visual_tokens`` must already be embedded to ActionDiT
        ``hidden_dim``. Keeping this projection outside the denoising loop lets
        inference reuse the layerwise Qwen features at every flow-matching step.
        """

        if len(layerwise_visual_tokens) != len(self.blocks):
            raise ValueError(
                "Layerwise MoT depth mismatch: "
                f"visual={len(layerwise_visual_tokens)}, action={len(self.blocks)}."
            )
        pre_state = self.pre_dit(
            action_tokens=action_tokens,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
        )
        x = pre_state["tokens"]
        batch_size, action_seq_len = x.shape[:2]
        if not layerwise_visual_tokens:
            raise ValueError("`layerwise_visual_tokens` must not be empty.")
        video_seq_len = int(layerwise_visual_tokens[0].shape[1])
        for layer_idx, video_tokens in enumerate(layerwise_visual_tokens):
            if video_tokens.ndim != 3 or video_tokens.shape != (
                batch_size,
                video_seq_len,
                self.hidden_dim,
            ):
                raise ValueError(
                    f"`layerwise_visual_tokens[{layer_idx}]` must be "
                    f"[{batch_size},{video_seq_len},{self.hidden_dim}], got "
                    f"{tuple(video_tokens.shape)}."
                )
        if visual_mask is None:
            visual_mask = torch.ones(
                (batch_size, video_seq_len), dtype=torch.bool, device=x.device
            )
        elif visual_mask.shape != (batch_size, video_seq_len):
            raise ValueError(
                f"`visual_mask` must be [{batch_size},{video_seq_len}], got "
                f"{tuple(visual_mask.shape)}."
            )
        visual_mask = visual_mask.to(device=x.device, dtype=torch.bool)
        action_mask = torch.ones(
            (batch_size, action_seq_len), dtype=torch.bool, device=x.device
        )
        mixed_attention_mask = torch.cat((visual_mask, action_mask), dim=1)
        mixed_attention_mask = mixed_attention_mask[:, None, None, :].expand(
            -1, 1, action_seq_len, -1
        )

        for block, video_tokens in zip(self.blocks, layerwise_visual_tokens):
            if self.use_gradient_checkpointing and self.training:
                def _block_forward(
                    _x,
                    _video_tokens,
                    _context,
                    _t_mod,
                    _freqs,
                    _context_mask,
                    _mixed_mask,
                    _block=block,
                ):
                    return self._forward_layerwise_mot_block(
                        _block,
                        _x,
                        _video_tokens,
                        _context,
                        _t_mod,
                        _freqs,
                        _context_mask,
                        _mixed_mask,
                    )

                x = torch.utils.checkpoint.checkpoint(
                    _block_forward,
                    x,
                    video_tokens,
                    pre_state["context"],
                    pre_state["t_mod"],
                    pre_state["freqs"],
                    pre_state["context_mask"],
                    mixed_attention_mask,
                    use_reentrant=False,
                )
            else:
                x = self._forward_layerwise_mot_block(
                    block,
                    x,
                    video_tokens,
                    pre_state["context"],
                    pre_state["t_mod"],
                    pre_state["freqs"],
                    pre_state["context_mask"],
                    mixed_attention_mask,
                )

        return self.post_dit(x, pre_state)

    def pre_dit(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
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
                f"Batch mismatch between action tokens and text context: {batch_size} vs {context.shape[0]}"
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

        seq_len = action_tokens.shape[1]
        if seq_len > self.freqs.shape[0]:
            raise ValueError(
                f"Action token length {seq_len} exceeds RoPE cache {self.freqs.shape[0]}."
            )

        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.hidden_dim))

        tokens = self.action_encoder(action_tokens)
        context_emb = self.text_embedding(context)
        context_attn_mask = context_mask.unsqueeze(1).expand(-1, seq_len, -1)
        freqs = self.freqs[:seq_len].view(seq_len, 1, -1).to(tokens.device)

        return {
            "tokens": tokens,
            "freqs": freqs,
            "t": t,
            "t_mod": t_mod,
            "context": context_emb,
            "context_mask": context_attn_mask,
            "meta": {
                "batch_size": batch_size,
                "seq_len": seq_len,
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
    ) -> torch.Tensor:
        pre_state = self.pre_dit(
            action_tokens=action_tokens,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
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
