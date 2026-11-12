"""Parameter-free history time RoPE, with explicit checkpoint-replay inputs."""
from __future__ import annotations
import math
from types import MethodType
import torch


def resolve_spec(spec):
    if spec is None:
        return None
    spec = dict(spec)
    allowed = {"dim", "max_phase", "ramp_steps", "start_step", "completed_steps",
               "reference_history_slots", "frame_interval"}
    if set(spec) - allowed:
        raise ValueError(f"Unknown history time RoPE fields: {set(spec) - allowed}")
    for key, default in (("dim", 16), ("ramp_steps", 2000), ("start_step", 90000),
                         ("reference_history_slots", 7), ("frame_interval", 40)):
        value = spec.get(key, default)
        if isinstance(value, bool) or int(value) != value:
            raise ValueError(f"History time RoPE {key} must be an integer")
        spec[key] = int(value)
    if spec["dim"] == 0:
        return None
    if spec["dim"] < 2 or spec["dim"] % 2:
        raise ValueError("History time RoPE dim must be a positive even integer")
    if spec["ramp_steps"] < 1 or spec["start_step"] < 0:
        raise ValueError("Time RoPE requires positive ramp_steps and nonnegative start_step")
    if spec["reference_history_slots"] < 1 or spec["frame_interval"] < 1:
        raise ValueError("Time RoPE requires positive slot count and interval")
    spec["max_phase"] = float(spec.get("max_phase", .5))
    if not math.isfinite(spec["max_phase"]) or spec["max_phase"] <= 0:
        raise ValueError("Time RoPE max_phase must be finite and positive")
    completed = spec.get("completed_steps", spec["start_step"])
    if isinstance(completed, bool) or int(completed) != completed or completed < 0:
        raise ValueError("Time RoPE completed_steps must be a nonnegative integer")
    spec["completed_steps"] = int(completed)
    return spec


def validate_config(config):
    spec = resolve_spec(getattr(config, "qwen_history_time_rope", None))
    if spec is not None:
        from .architecture_contract import is_vlm_fast_action_bridge_version
        if (getattr(config, "vlm_backend", None) != "qwen3_5"
                or getattr(config, "qwen_temporal_embedding", None) != "vlm_history_current_v1"
                or getattr(config, "qwen_image_attention_mode", None) != "bidirectional_image_blocks_current_only"
                or not is_vlm_fast_action_bridge_version(config.architecture_version)):
            raise ValueError("Time RoPE requires Qwen G0.5 current-history/current-only image blocks")
    config.qwen_history_time_rope = spec


def phase(spec):
    return spec["max_phase"] * min(1., max(0.,
        (spec["completed_steps"] - spec["start_step"]) / spec["ramp_steps"]))


def set_completed_steps(model, completed_steps):
    spec = getattr(model.config, "qwen_history_time_rope", None)
    if spec is not None:
        spec["completed_steps"] = int(completed_steps)


def coefficients(times, spec):
    amplitude = phase(spec)
    if amplitude == 0:
        return None
    frequencies = torch.logspace(0, -1, spec["dim"] // 2, device=times.device)
    angles = times.float()[..., None] * amplitude * frequencies
    return angles.cos()[:, None], angles.sin()[:, None]


def rotate_subspace(value, cos, sin, start, dimensions):
    if start + dimensions > value.shape[-1] or start < 0:
        raise ValueError("Time RoPE exceeds non-spatial head dimensions")
    selected = value[..., start:start + dimensions].float()
    a, b = selected[..., 0::2], selected[..., 1::2]
    rotated = torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1)
    rotated = rotated.flatten(-2).to(value.dtype)
    return torch.cat((value[..., :start], rotated, value[..., start + dimensions:]), dim=-1)


def _forward(self, hidden_states, position_embeddings, attention_mask,
             past_key_values=None, **kwargs):
    temporal = kwargs.pop("xvla_time_rope", None)
    if temporal is None:
        return self._xvla_native_forward(hidden_states=hidden_states,
            position_embeddings=position_embeddings, attention_mask=attention_mask,
            past_key_values=past_key_values, **kwargs)
    if past_key_values is not None:
        raise ValueError("History time RoPE supports uncached policy prefill only")
    from transformers.models.qwen3_5 import modeling_qwen3_5 as native
    shape = hidden_states.shape[:-1]
    head_shape = (*shape, -1, self.head_dim)
    query, gate = torch.chunk(self.q_proj(hidden_states).view(*shape, -1, self.head_dim * 2), 2, dim=-1)
    gate = gate.reshape(*shape, -1)
    query = self.q_norm(query.view(head_shape)).transpose(1, 2)
    key = self.k_norm(self.k_proj(hidden_states).view(head_shape)).transpose(1, 2)
    value = self.v_proj(hidden_states).view(head_shape).transpose(1, 2)
    spatial_cos, spatial_sin = position_embeddings
    query, key = native.apply_rotary_pos_emb(query, key, spatial_cos, spatial_sin)
    time_cos, time_sin, dimensions = temporal
    start = spatial_cos.shape[-1]
    query = rotate_subspace(query, time_cos, time_sin, start, dimensions)
    key = rotate_subspace(key, time_cos, time_sin, start, dimensions)
    interface = native.ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, native.eager_attention_forward)
    output, weights = interface(self, query, key, value, attention_mask,
        dropout=0. if not self.training else self.attention_dropout,
        scaling=self.scaling, **kwargs)
    output = output.reshape(*shape, -1).contiguous() * torch.sigmoid(gate)
    return self.o_proj(output), weights


def install(model, spec):
    """Instance-local methods; no parameter/buffer/state-key additions."""
    if spec is None:
        return
    text = model.model.language_model
    factor = text.config.rope_parameters["partial_rotary_factor"]
    for layer in text.layers:
        if layer.block_type != "full_attention":
            continue
        attn = layer.self_attn
        if int(attn.head_dim * factor) + spec["dim"] > attn.head_dim:
            raise ValueError("Time RoPE does not fit outside native spatial RoPE")
        if not hasattr(attn, "_xvla_native_forward"):
            attn._xvla_native_forward = attn.forward
            attn.forward = MethodType(_forward, attn)
