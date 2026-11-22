"""Instance-local history5 prefill acceleration; no M16/tile changes.

The installed model's original prefill keeps all seven KV fields, complete
reference shapes, Memory/text validity and cloned prefix storage. Only prefill
RoPE and integer pack/unpack index construction are replaced. Existing action
decode already uses inference_rope.rotate and remains untouched.
"""
from collections import Counter, OrderedDict
from dataclasses import fields
from functools import partial
from threading import Lock, get_ident
import json
import os
from pathlib import Path
import time
import weakref

import torch
from torch.nn import functional as F
from wall_x.model.vap.attention import _compiled_flex_attention, _flex_kernel_options
from wall_x.model.vap.layout import InterleaveLayout, ActionStageLayout
from wall_x.model.vap.rope import apply_rope_4d, build_rope_4d_cache
from wall_x.model.vapv2.inference_rope import rotate as exact_rotate
from wall_x.model.vapv2.kv_cache import JointActionPrefixCache


class PackedLayout:
    def __init__(self, layout, device):
        self.layout = layout
        self.permutation = layout.logical_to_physical(device=device)
        self.inverse = layout.physical_to_logical(device=device)

    def pack(self, video, action):
        if video.shape[-2] != self.layout.num_video_tokens or action.shape[-2] != self.layout.num_action_tokens:
            raise ValueError('cached pack shape differs from full layout')
        if video.device != self.permutation.device or action.device != video.device:
            raise ValueError('cached indices belong to a different device')
        return torch.cat((video, action), dim=-2).index_select(-2, self.permutation)

    def unpack(self, logical):
        if logical.shape[-2] != self.layout.sequence_length or logical.device != self.inverse.device:
            raise ValueError('cached unpack shape/device differs from full layout')
        physical = logical.index_select(-2, self.inverse)
        return physical[..., :self.layout.num_video_tokens, :], physical[..., self.layout.num_video_tokens:, :]


class LayoutIndexCache:
    """Only immutable integer layouts; never cache validity or semantic KV."""
    def __init__(self, capacity=2):
        if capacity < 1:
            raise ValueError('positive cache capacity required')
        self.capacity = capacity
        self.entries = OrderedDict()
        self.stats = Counter()

    def get(self, layout, device):
        if type(layout) not in (InterleaveLayout, ActionStageLayout):
            self.stats['unsupported_fallbacks'] += 1
            return layout
        device = torch.device(device)
        if device.type == 'cuda' and device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        key = (type(layout), tuple(getattr(layout, f.name) for f in fields(layout)), device)
        if key not in self.entries:
            if len(self.entries) >= self.capacity:
                self.entries.popitem(last=False)
                self.stats['evictions'] += 1
            self.entries[key] = PackedLayout(layout, device)
            self.stats['builds'] += 1
        else:
            self.stats['hits'] += 1
        self.entries.move_to_end(key)
        return self.entries[key]

    def clear(self):
        self.entries.clear()


class History5PrefillExact:
    def __init__(self, runtime, *, rope=True, layout=True):
        core = runtime.model.core
        tiles = _flex_kernel_options()
        if tiles is not None and tiles.get('BLOCK_M') == 16:
            raise ValueError('M16 Attention is not enabled by this adapter; remove the external M16 override')
        expected = {'layers','positions','key_valid','owner','full_positions','active_start','joint_layout'}
        if {f.name for f in fields(JointActionPrefixCache)} != expected:
            raise ValueError('requires current full-layout history5 VAPv2 KV ABI')
        if core.training or core.state_film is not None or not runtime.use_kv_cache:
            raise ValueError('requires eval/no State-FiLM and prefix KV')
        if getattr(core, '_history5_prefill_exact', None) is not None:
            raise ValueError('prefill adapter already installed')
        self._core = weakref.ref(core)
        self._original = core.forward_action_stage_with_prefix_cache
        self._core_override = 'forward_action_stage_with_prefix_cache' in core.__dict__
        self._refs = [(b.attention, b.attention.forward, 'forward' in b.attention.__dict__) for b in core.blocks]
        self._reference_by_id = {id(module):fn for module,fn,_ in self._refs}
        self.rope, self.layout = bool(rope), bool(layout)
        self.indices = LayoutIndexCache(2)
        self.stats = Counter()
        self._phases = {}
        self._lock = Lock()
        self._thread = None
        self._active = False
        self.enabled = True
        for module,_,_ in self._refs:
            module._history5_prefill_exact = self
            module.forward = partial(self.attention, module)
        core.forward_action_stage_with_prefix_cache = self.prefill
        core._history5_prefill_exact = self

    def configure(self, *, rope, layout):
        if self._active:
            raise RuntimeError('cannot change switches during a request')
        self.rope, self.layout = bool(rope), bool(layout)

    @torch.no_grad()
    def prefill(self, *args, **kwargs):
        if self._core().training:
            raise RuntimeError('prefill adapter is inference-only')
        if self._thread not in (None, get_ident()) or not self._lock.acquire(blocking=False):
            raise RuntimeError('requires one non-reentrant model owner thread')
        self._thread = get_ident()
        self._active = True
        self._phases.clear()
        self.stats['prefill_calls'] += 1
        try:
            return self._original(*args, **kwargs)
        finally:
            self._phases.clear()
            self._active = False
            self._lock.release()

    def attention(self, module, *args, **kwargs):
        if not self._active or not (self.rope or self.layout):
            return self._reference_by_id[id(module)](*args, **kwargs)
        if get_ident() != self._thread or module.training or torch.is_grad_enabled():
            raise RuntimeError('requires request-local inference context')
        return _attention_forward(module, *args, **kwargs)

    def rotate_prefill(self, stream, q, k, positions, axes, phase):
        if not self.rope or not q.is_cuda or q.dtype != torch.bfloat16 or q.stride(-1) != 1 or k.stride(-1) != 1:
            self.stats['rope_fallbacks'] += 1
            return apply_rope_4d(q, k, positions, axes_dim=axes, phase_cache=phase)
        if phase is None:
            key = (stream, tuple(axes), q.dtype, q.device, tuple(q.shape[:2]))
            try:
                version = positions._version
            except RuntimeError:
                version = None
            entry = self._phases.get(key)
            if version is not None and entry is not None and entry[0] is positions and entry[1] == version:
                phase = entry[2]
                self.stats['phase_hits'] += 1
            else:
                phase = build_rope_4d_cache(positions, head_dim=q.shape[-1], axes_dim=axes, dtype=q.dtype)
                self._phases[key] = (positions, version, phase)
                self.stats['phase_builds'] += 1
        phase.validate_for(q, k, position_ids=positions, axes_dim=axes, theta=10000.)
        self.stats['fused_prefill_rope_calls'] += 1
        return exact_rotate(q, k, phase.cos, phase.sin)

    def metadata(self):
        return dict(enabled=self.enabled, rope=self.rope, layout_cache=self.layout,
                    phase_entries=len(self._phases), layout_entries=len(self.indices.entries),
                    stats=dict(self.stats), layout_stats=dict(self.indices.stats),
                    scope='prefill_only', decode='existing exact Triton RoPE unchanged',
                    m16_enabled=False, cross_window_semantic_cache=False)

    def disable(self):
        if self._active:
            raise RuntimeError('cannot remove adapter during request')
        if not self.enabled:
            return
        for runtime, name, reference, had in getattr(self, '_sample_restore', []):
            if had:
                setattr(runtime, name, reference)
            else:
                delattr(runtime, name)
        for module,reference,had in self._refs:
            if had:
                module.forward = reference
            else:
                del module.forward
            del module._history5_prefill_exact
        core = self._core()
        if self._core_override:
            core.forward_action_stage_with_prefix_cache = self._original
        else:
            del core.forward_action_stage_with_prefix_cache
        del core._history5_prefill_exact
        self.indices.clear()
        self._phases.clear()
        self.enabled = False


def install(runtime, *, rope=True, layout=True):
    return History5PrefillExact(runtime, rope=rope, layout=layout)


def install_for_serving(runtime, *, rope=True, layout=True):
    """Fail closed if either first complete A0/A1 output differs from reference."""
    adapter = install(runtime, rope=rope, layout=layout)
    requested = dict(rope=bool(rope), layout=bool(layout))
    proof = dict(requested=requested, failed=False, action_checks=[])
    output = Path(os.environ['VAP_RUN_DIR'])/'history5_prefill_exact.json'
    def save():
        proof['adapter'] = adapter.metadata()
        tmp=output.with_suffix('.tmp');tmp.write_text(json.dumps(proof,indent=2));tmp.replace(output)
    adapter._sample_restore = []
    for name in ['sample_first_action','sample_next_action']:
        reference = getattr(runtime,name)
        adapter._sample_restore.append((runtime,name,reference,name in runtime.__dict__))
        def checked(*args,_name=name,_reference=reference,**kwargs):
            if proof['failed']:
                raise RuntimeError('prefill equality gate failed; refusing actions')
            if not any(c['method']==_name for c in proof['action_checks']):
                start=time.monotonic()
                try:
                    adapter.configure(rope=False,layout=False)
                    with torch.random.fork_rng(devices=[torch.device(runtime.device).index or 0]):
                        baseline=_reference(*args,**kwargs)
                finally:
                    adapter.configure(**requested)
                candidate=_reference(*args,**kwargs)
                equal=torch.equal(baseline,candidate)
                proof['action_checks'].append(dict(method=_name,bitwise_equal=equal,max_abs=float((baseline-candidate).abs().max()),steps=int(runtime.cfg.model.action_flow['inference_steps']),seconds=time.monotonic()-start))
                proof['failed']=not equal;save()
                if not equal:
                    raise RuntimeError('history5 prefill acceleration changed complete action')
                return candidate
            result=_reference(*args,**kwargs);save();return result
        setattr(runtime,name,checked)
    save()
    return adapter

# _attention_forward is appended from the audited current deployed implementation.
# Its diff only substitutes RoPE and pack/unpack indices; all mask/GEMM/KV logic
# stays verbatim, including clone(), capture_prefix_rank and seven-field ABI.


def _attention_forward(
    self,
    text,
    video,
    action,
    *,
    text_position_ids,
    video_position_ids,
    action_position_ids,
    joint_layout,
    capture_prefix_rank=None,
    rope_cache=None,
):
    layout = joint_layout.base
    adapter = self._history5_prefill_exact
    packed_layout = adapter.indices.get(layout, video.device) if adapter.layout else layout
    self._check_inputs(
        video, action, video_position_ids, action_position_ids, layout
    )
    if text.shape != (video.shape[0], joint_layout.text_len, self.video_dim):
        raise ValueError("Text must be [B,L,Dvideo] matching joint layout")
    qs, ks, values = [], [], []
    for stream, tensor, positions in (
        ("text", text, text_position_ids),
        ("video", video, video_position_ids),
        ("action", action, action_position_ids),
    ):
        q = self._heads(
            getattr(self, f"{stream}_norm_q")(getattr(self, f"{stream}_q")(tensor))
        )
        k = self._heads(
            getattr(self, f"{stream}_norm_k")(getattr(self, f"{stream}_k")(tensor))
        )
        v = self._heads(getattr(self, f"{stream}_v")(tensor))
        phase_cache = (
            None if rope_cache is None
            else rope_cache.for_stream(stream, q, k, positions)
        )
        q, k = adapter.rotate_prefill(
            stream, q, k, positions, self.rope_axes_dim, phase_cache
        )
        qs.append(q.transpose(1, 2))
        ks.append(k.transpose(1, 2))
        values.append(v.transpose(1, 2))

    def pack(parts):
        return torch.cat((parts[0], packed_layout.pack(parts[1], parts[2])), dim=-2)

    q, k, v = pack(qs), pack(ks), pack(values)
    if self.attention_backend == "flex_attention":
        if q.device.type != "cuda":
            raise RuntimeError("VAPv2 FlexAttention requires CUDA")
        output = _compiled_flex_attention()(
            q,
            k,
            v,
            block_mask=joint_layout.flex_block_mask(q.device),
            score_mod=joint_layout.flex_score_mod,
            kernel_options=_flex_kernel_options(),
        )
    else:
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=joint_layout.attention_mask(q.device)[:, None],
            dropout_p=self.dropout if self.training else 0.0,
        )
    t = output[:, :, : joint_layout.text_len]
    vo, ao = packed_layout.unpack(output[:, :, joint_layout.text_len :])
    outputs = tuple(
        getattr(self, f"{stream}_o")(tensor.transpose(1, 2).flatten(2))
        for stream, tensor in (("text", t), ("video", vo), ("action", ao))
    )
    if capture_prefix_rank is None:
        return outputs
    from wall_x.model.vap.kv_cache import VAPAttentionKVCache

    active = layout.action_tokens_per_chunk
    chunk = (capture_prefix_rank - 1) // 2
    prefix_end = (joint_layout.text_len + layout.condition_video_tokens
                  + chunk * (layout.video_tokens_per_chunk + active))
    # Preserve full-forward shapes at capture, retaining ONLY clean causal
    # prefix K/V. Neither active nor future tokens may enter the cache.
    cache = VAPAttentionKVCache(
        k[..., :prefix_end, :].detach().clone(),
        v[..., :prefix_end, :].detach().clone(), active
    )
    return outputs, cache
