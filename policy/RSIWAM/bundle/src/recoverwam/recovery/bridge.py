"""Observation-conditioned recovery tokens inside the frozen world/action policy.

The pilot learns an intervention-state classifier, not an oracle failure or
object-relation detector. Its four tokens attend to pooled frozen video values
and modulate the final action features. A closed gate bypasses every change.
There is no task-ID routing, external agent, persistent batch state, or LoRA.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


FORMAT = "recoverwam-recovery-bridge-v1"
MODES = ("auto", "on", "off", "soft")


class RecoveryBridge(nn.Module):
    def __init__(self, action_dim: int, video_dim: int, context_dim: int,
                 hidden_dim: int = 128, token_count: int = 4,
                 pooled_tokens: int = 16, threshold: float = 0.9,
                 separate_proprio: bool = False):
        super().__init__()
        if min(action_dim, video_dim, context_dim, hidden_dim, token_count, pooled_tokens) < 1:
            raise ValueError("Bridge dimensions must be positive")
        if not 0 < threshold < 1 or hidden_dim % 4:
            raise ValueError("Invalid gate threshold or attention width")
        if not isinstance(separate_proprio, bool):
            raise ValueError("separate_proprio must be a bool")
        self.config = dict(action_dim=action_dim, video_dim=video_dim, context_dim=context_dim,
                           hidden_dim=hidden_dim, token_count=token_count,
                           pooled_tokens=pooled_tokens, threshold=threshold,
                           separate_proprio=separate_proprio)
        self.video_norm = nn.LayerNorm(video_dim)
        self.video_proj = nn.Linear(video_dim, hidden_dim)
        self.context_norm = nn.LayerNorm(context_dim)
        self.context_proj = nn.Linear(context_dim, hidden_dim)
        self.queries = nn.Parameter(torch.randn(1, token_count, hidden_dim) * 0.02)
        self.pool_position = nn.Parameter(torch.randn(1, pooled_tokens, hidden_dim) * 0.02)
        self.read_world = nn.MultiheadAttention(hidden_dim, 4, batch_first=True)
        self.state_norm = nn.LayerNorm(hidden_dim)
        self.gate_head = nn.Linear(hidden_dim, 1)
        self.action_norm = nn.LayerNorm(action_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.read_recovery = nn.MultiheadAttention(hidden_dim, 4, batch_first=True)
        self.residual = nn.Linear(hidden_dim, action_dim, bias=False)
        nn.init.zeros_(self.residual.weight)
        if separate_proprio:
            self.proprio_norm = nn.LayerNorm(context_dim)
            self.proprio_proj = nn.Linear(context_dim, hidden_dim)

    def encode(self, video_values, context, context_mask):
        if video_values.ndim != 3 or context.ndim != 3:
            raise ValueError("Expected [B,L,D] world values and context")
        if context_mask.shape != context.shape[:2] or context_mask.dtype != torch.bool:
            raise ValueError("Expected boolean context mask [B,L]")
        if video_values.shape[0] != context.shape[0] or not context_mask.any(1).all():
            raise ValueError("Batch mismatch or empty context")
        with torch.autocast(device_type=video_values.device.type, enabled=False):
            world = F.adaptive_avg_pool1d(video_values.detach().float().transpose(1, 2),
                                          self.config["pooled_tokens"]).transpose(1, 2)
            world = self.video_proj(self.video_norm(world)) + self.pool_position
            text, text_mask = context, context_mask
            if self.config["separate_proprio"]:
                # The frozen base appends proprio after the padded text sequence.
                if context.shape[1] < 2 or not context_mask[:, -1].all():
                    raise ValueError("Expected an appended, unmasked proprio token")
                text, text_mask = context[:, :-1], context_mask[:, :-1]
                if not text_mask.any(1).all():
                    raise ValueError("Expected text before the proprio token")
            masked = torch.where(text_mask[..., None], text.detach().float(), 0.0)
            text_state = masked.sum(1) / text_mask.sum(1, keepdim=True)
            text_state = self.context_proj(self.context_norm(text_state))
            queries = self.queries.expand(context.shape[0], -1, -1) + text_state[:, None]
            if self.config["separate_proprio"]:
                proprio = self.proprio_proj(self.proprio_norm(context[:, -1].detach().float()))
                queries = queries + proprio[:, None]
            tokens, _ = self.read_world(queries, world, world, need_weights=False)
            tokens = self.state_norm(tokens + queries)
            logits = self.gate_head(tokens.mean(1)).squeeze(-1)
        return {"tokens": tokens, "gate_logits": logits}

    def forward(self, action_tokens, state, *, mode="auto"):
        if mode not in MODES:
            raise ValueError(f"Unknown recovery mode: {mode}")
        if mode == "off":
            return action_tokens
        probability = state["gate_logits"].sigmoid()
        if probability.shape != (action_tokens.shape[0],):
            raise ValueError("Recovery state batch mismatch")
        if mode == "auto":
            gate = probability >= self.config["threshold"]
            if not gate.any():
                return action_tokens
        elif mode == "on":
            gate = torch.ones_like(probability)
        else:
            gate = probability
        with torch.autocast(device_type=action_tokens.device.type, enabled=False):
            query = self.action_proj(self.action_norm(action_tokens.detach().float()))
            update, _ = self.read_recovery(query, state["tokens"], state["tokens"], need_weights=False)
            update = self.residual(update) * gate[:, None, None]
        changed = action_tokens + update.to(action_tokens.dtype)
        # Preserve inactive rows exactly, including when another row is active.
        return torch.where((gate != 0)[:, None, None], changed, action_tokens)


def install_bridge(policy, *, gradient_checkpointing=False, **config):
    model = getattr(policy, "model", policy)
    if hasattr(model, "recovery_bridge") or hasattr(model, "_action_lora_config"):
        raise ValueError("The pilot requires an unadapted frozen base")
    if model.raw_action_dim != 16 or model.proprio_dim != 16:
        raise ValueError("The pilot supports official RoboDojo EE16 only")
    model.requires_grad_(False)
    dims = dict(action_dim=model.action_expert.hidden_dim,
                video_dim=model.video_expert.blocks[-1].self_attn.v.out_features,
                context_dim=model.action_expert.text_embedding[0].in_features)
    model.recovery_bridge = RecoveryBridge(**dims, **config).to(device=model.device, dtype=torch.float32)
    model.action_expert.use_gradient_checkpointing = gradient_checkpointing
    model.mot.mot_checkpoint_mixed_attn = gradient_checkpointing
    model.eval()
    assert_only_bridge_trainable(model)
    return dict(format=FORMAT, **model.recovery_bridge.config,
                trainable_parameters=sum(p.numel() for p in model.recovery_bridge.parameters()))


def assert_only_bridge_trainable(model):
    expected = {id(p) for p in model.recovery_bridge.parameters()}
    actual = {id(p) for p in model.parameters() if p.requires_grad}
    if expected != actual:
        raise RuntimeError("Only the recovery bridge may be trainable")


def encode_bridge(model, video_kv_cache, context, context_mask):
    return model.recovery_bridge.encode(video_kv_cache[-1]["v"], context, context_mask)


def apply_bridge(model, tokens, video_kv_cache, context, context_mask, *, mode="auto", state=None):
    if not hasattr(model, "recovery_bridge") or mode == "off":
        return tokens
    if state is None:
        state = encode_bridge(model, video_kv_cache, context, context_mask)
    return model.recovery_bridge(tokens, state, mode=mode)


def save_bridge(path, policy, *, metadata):
    model = getattr(policy, "model", policy)
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    payload = dict(format=FORMAT, config=model.recovery_bridge.config,
                   state_dict={k: v.detach().cpu().clone() for k, v in model.recovery_bridge.state_dict().items()},
                   metadata=dict(metadata))
    temp = path.with_name(path.name + ".pending")
    torch.save(payload, temp)
    temp.replace(path)


def load_bridge(path, policy, *, expected_base_identity):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != FORMAT or payload.get("metadata", {}).get("base_identity") != expected_base_identity:
        raise ValueError("Recovery bridge format or base identity mismatch")
    config = dict(payload["config"])
    dims = {k: config.pop(k) for k in ("action_dim", "video_dim", "context_dim")}
    model = getattr(policy, "model", policy)
    # Validate the whole payload before installing anything in the live model.
    candidate = RecoveryBridge(**dims, **config)
    for value in payload["state_dict"].values():
        if not torch.isfinite(value).all():
            raise ValueError("Nonfinite recovery bridge weights")
    candidate.load_state_dict(payload["state_dict"], strict=True)
    actual = install_bridge(policy, **config)
    if any(actual[k] != v for k, v in dims.items()):
        del model.recovery_bridge
        raise ValueError("Recovery bridge backbone dimensions differ")
    model.recovery_bridge.load_state_dict(candidate.state_dict(), strict=True)
    model.eval()
    return payload["metadata"]
