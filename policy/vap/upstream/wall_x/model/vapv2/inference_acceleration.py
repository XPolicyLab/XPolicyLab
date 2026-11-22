"""Opt-in action-decode CUDA Graph + exact-rounding RoPE for joint VAPv2.

The graph owns static *storage*, not persistent semantic context. EVERY new
prefix refreshes all layer KV, validity (including FlexAttention's captured
score_mod tensor), and positions/phases. Prefill still runs the reference core
once per action chunk. Text/ego/history3 never carry stale per-layer KV between
windows or episodes. Noise, flow updates, time embedding and VAE stay outside
capture. No checkpoint parameters/buffers are added; no compact attention.
"""

from collections import OrderedDict, Counter
from dataclasses import replace
from threading import Lock, get_ident
import weakref

import torch

from wall_x.model.vap.kv_cache import VAPAttentionKVCache
from wall_x.model.vap.rope import build_rope_4d_cache
from .kv_cache import decode_impl
from .layout import JointConditionLayout


def _phase(core, cache, dtype):
    return build_rope_4d_cache(
        cache.full_positions, head_dim=core.head_dim,
        axes_dim=core.rope_axes_dim, dtype=dtype,
    )


def _time_pair(core, x, timestep, cache):
    # Keep the original shape and embedding arithmetic; no schedule caching.
    full_time = timestep.new_zeros(x.shape[0], cache.full_positions.shape[1])
    start = cache.active_start
    full_time[:, start:start + x.shape[1]] = (
        timestep[:, None] if timestep.ndim == 1 else timestep
    )
    embedding = core.action_time.embed(full_time, dtype=x.dtype)
    return embedding, core.action_time.project(embedding, dtype=x.dtype)


class _GraphLayout(JointConditionLayout):
    def __init__(self, source, *, dense):
        super().__init__(source.base, source.text_valid, source.memory_valid)
        # SDPA's reference mask builder creates CPU ranks, which cannot run
        # inside capture. FlexAttention uses its precompiled structural mask.
        self._dense = super().attention_mask() if dense else None
        self.graph_unpack_indices = self.base.physical_to_logical(self.valid.device)

    def attention_mask(self, device=None):
        if self._dense is None:
            raise RuntimeError("Dense graph mask was not prepared")
        if device is not None and torch.device(device) != self._dense.device:
            raise ValueError("Graph mask device changed")
        return self._dense


class _ActionGraph:
    def __init__(self, owner, x, timestep, source, pair):
        self.action, self.timestep = x.clone(), timestep.clone()
        self.pair = tuple(t.clone() for t in pair)
        layout = source.joint_layout
        static_layout = _GraphLayout(
            layout, dense=any(b.attention.attention_backend != "flex_attention"
                              for b in owner.core.blocks)
        )
        self.cache = replace(
            source,
            layers=tuple(VAPAttentionKVCache(c.key.clone(), c.value.clone(),
                                             c.active_capacity)
                         for c in source.layers),
            positions=source.positions.clone(),
            full_positions=source.full_positions.clone(),
            key_valid=source.key_valid.clone(),
            joint_layout=static_layout,
        )
        self.rope = _phase(owner.core, self.cache, x.dtype)
        self.source = source
        self.graph = torch.cuda.CUDAGraph()

        def forward():
            return decode_impl(owner.core, self.action, self.timestep, self.cache,
                               time_pair=self.pair, action_rope=self.rope)

        # Compile FlexAttention/Triton and create structural masks BEFORE capture.
        warm = torch.cuda.Stream(device=x.device)
        warm.wait_stream(torch.cuda.current_stream(x.device))
        with torch.cuda.stream(warm):
            for _ in range(3):
                self.output = forward()
        torch.cuda.current_stream(x.device).wait_stream(warm)
        with torch.cuda.graph(self.graph):
            self.output = forward()
        owner.stats["captures"] += 1

    def run(self, owner, x, timestep, source, pair):
        if self.source is not source:
            for dst, src in zip(self.cache.layers, source.layers):
                dst.key.copy_(src.key)
                dst.value.copy_(src.value)
            self.cache.positions.copy_(source.positions)
            self.cache.full_positions.copy_(source.full_positions)
            self.cache.key_valid.copy_(source.key_valid)
            dst, src = self.cache.joint_layout, source.joint_layout
            # Crucial: mutate the captured tensor in-place, not its attribute.
            dst.valid.copy_(src.valid)
            dst.text_valid.copy_(src.text_valid)
            if dst.memory_valid is not None:
                dst.memory_valid.copy_(src.memory_valid)
            if dst._dense is not None:
                dst._dense.copy_(src.attention_mask(x.device))
            phase = _phase(owner.core, source, x.dtype)
            self.rope.cos.copy_(phase.cos)
            self.rope.sin.copy_(phase.sin)
            self.source = source
            owner.stats["prefix_refreshes"] += 1
        self.action.copy_(x)
        self.timestep.copy_(timestep)
        for dst, src in zip(self.pair, pair):
            dst.copy_(src)
        self.graph.replay()
        owner.stats["replays"] += 1
        # Never return graph-owned storage that the next request will overwrite.
        return self.output.clone()


class VAPv2ActionAcceleration:
    def __init__(self, core, *, mode="graph_rope", max_graphs=2):
        if mode not in {"rope", "graph_rope"} or max_graphs < 1:
            raise ValueError("mode must be rope/graph_rope and max_graphs >= 1")
        p = next(core.parameters())
        if core.training or core.state_film is not None:
            raise ValueError("V2 acceleration requires eval and no State-FiLM")
        if not p.is_cuda or p.dtype != torch.bfloat16:
            raise ValueError("V2 acceleration requires CUDA BF16")
        if getattr(core, "_v2_action_accelerator", None) is not None:
            raise ValueError("Disable the previous V2 accelerator before replacing")
        self._core = weakref.ref(core)
        self.mode, self.max_graphs = mode, max_graphs
        self.graphs = OrderedDict()
        self.stats = Counter()
        self._lock, self._thread = Lock(), None
        self._signature = None
        self._last_source = None
        self._rope = None
        self._hook = core.register_load_state_dict_post_hook(lambda *_: self.clear())
        core._v2_action_accelerator = self

    @property
    def core(self):
        core = self._core()
        if core is None:
            raise RuntimeError("V2 model released")
        return core

    def clear(self):
        if self._lock.locked():
            raise RuntimeError("Cannot invalidate during graph replay")
        self.graphs.clear()
        self._last_source = self._rope = self._signature = None
        self.stats["invalidations"] += 1

    def disable(self):
        self.clear()
        self._hook.remove()
        self.core._v2_action_accelerator = None

    def metadata(self):
        return dict(mode=self.mode, graphs=len(self.graphs), stats=dict(self.stats),
                    scope="action_chunk", prefix_rebuilt_each_window=True)

    @torch.no_grad()
    def decode(self, x, timestep, source):
        if self.core.training or not x.is_cuda or x.dtype != torch.bfloat16:
            raise RuntimeError("V2 acceleration is CUDA BF16 eval-only")
        if self._thread not in (None, get_ident()):
            raise RuntimeError("One V2 graph runtime must stay on one worker thread")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Concurrent/reentrant V2 graph replay is unsupported")
        self._thread = get_ident()
        try:
            if self._last_source is not source:
                signature = tuple((p.data_ptr(), p._version, p.dtype, p.device)
                                  for p in self.core.parameters())
                if signature != self._signature:
                    self.graphs.clear()
                    self._signature = signature
                    self.stats["weight_invalidations"] += 1
                self._last_source = source
                self._rope = _phase(self.core, source, x.dtype)
            if self.mode == "rope":
                return decode_impl(self.core, x, timestep, source,
                                   action_rope=self._rope)
            layout = source.joint_layout
            key = (tuple(x.shape), x.dtype, x.device, tuple(timestep.shape),
                   timestep.dtype, layout.base, layout.text_len,
                   None if layout.memory_valid is None else tuple(layout.memory_valid.shape),
                   source.active_start, tuple(source.full_positions.shape),
                   tuple((tuple(c.key.shape), c.active_capacity) for c in source.layers))
            pair = _time_pair(self.core, x, timestep, source)
            entry = self.graphs.get(key)
            if entry is None:
                if len(self.graphs) >= self.max_graphs:
                    self.graphs.popitem(last=False)
                    self.stats["evictions"] += 1
                entry = _ActionGraph(self, x, timestep, source, pair)
                self.graphs[key] = entry
            self.graphs.move_to_end(key)
            return entry.run(self, x, timestep, source, pair)
        finally:
            self._lock.release()
