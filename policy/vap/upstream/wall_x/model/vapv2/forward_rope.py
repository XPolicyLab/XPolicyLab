"""Opt-in, forward-local phase sharing; never cache layer-specific Q/K.

FP32 phases are built outside the block/checkpoint loop. Each stream binds
once to its actual projected Q/K dtype (not the residual/autocast input dtype).
Only this forward's closures retain the tables, including GC recomputation.
No module buffers, state-dict entries, RNG use or cross-batch cache are added.
"""

import os

import torch

from wall_x.model.vap.rope import RoPE4DPhaseCache, build_rope_4d_cache


def rope_forward_cache_enabled():
    value = os.environ.get("VAP_ROPE_FORWARD_CACHE", "0").strip().lower()
    if value not in {"0", "1", "false", "true", "off", "on"}:
        raise ValueError("VAP_ROPE_FORWARD_CACHE must be 0/1, false/true or off/on")
    return value in {"1", "true", "on"}


class ForwardRoPECache:
    def __init__(self, *, text, video, action, head_dim, axes_dim):
        self._positions = dict(text=text, video=video, action=action)
        self._phases = {}
        self._bound = set()
        with torch.no_grad():
            for name, positions in self._positions.items():
                if positions.requires_grad:
                    raise ValueError("Forward RoPE cache requires fixed position IDs")
                self._phases[name] = build_rope_4d_cache(
                    positions, head_dim=head_dim, axes_dim=axes_dim,
                    dtype=torch.float32,
                )

    def for_stream(self, name, q, k, positions):
        if positions is not self._positions[name]:
            raise ValueError("Forward RoPE cache used with different position IDs")
        if q.dtype != k.dtype:
            raise ValueError("Forward RoPE cache requires matching Q/K dtype")
        phase = self._phases[name]
        if name not in self._bound:
            with torch.no_grad():
                phase = RoPE4DPhaseCache(
                    cos=phase.cos.to(dtype=q.dtype),
                    sin=phase.sin.to(dtype=q.dtype),
                    axes_dim=phase.axes_dim, theta=phase.theta,
                )
            # Release transient FP32 tables when the stream uses BF16/FP16.
            self._phases[name] = phase
            self._bound.add(name)
        elif phase.cos.dtype != q.dtype:
            raise ValueError("Forward RoPE Q/K dtype changed between layers/recompute")
        # apply_rope_4d still validates batch/length/head/device/axes/theta.
        return phase
