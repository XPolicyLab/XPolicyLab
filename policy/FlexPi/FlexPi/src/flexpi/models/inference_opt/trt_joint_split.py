"""TensorRT runtime for the joint KV-split (prefill anchors once, decode noisy
per step). Drop-in replacement for ``mot._forward_joint_inner``, same contract
as ``trt_joint.py`` but backed by TWO engines built offline by
``scripts/inference_opt/trt_onnx_joint_split_engine.py``:

* prefill_core  — clean anchors → stacked per-layer K/V ``k_all/v_all``.
* decode_core   — noisy tokens + K/V cache → noisy_video/action outputs.

The 387 first-frame anchors (ff_v+ff_d+ff_p) attend only to anchors and are
modulated at t=0, so their per-layer K/V is step-invariant: the adapter runs
the prefill engine once per infer call (cached until the anchor tokens change)
and the decode engine every step. The anchor OUTPUT rows of ``merged_out`` are
step-invariant AND get frame-0-reclamped by the caller, so they are filled with
zeros (never read); only the noisy rows carry the decode output.

Off-shape calls fall back to the torch ``_forward_joint_inner``.

Memory: the two engines each bake a copy of the video-expert block weights
(prefill bakes video, decode bakes video+action — ~9.5 GB + ~11.9 GB), so
resident use is ~24.9 GB with ``free_video_blocks=True`` (both experts' blocks
freed) + one shared TRT activation scratch — measured E2E 413 ms/call (vs
553 ms single-engine, ~25% faster). The single engine sits at ~17 GB, so the
split's ~8 GB extra is the prefill engine's redundant video-expert copy. That
leaves too little headroom for ``encoder_cuda_graph`` captures (OOMs 32 GB) and
is tight beside a Vulkan-coresident sim renderer (SR runs ~8 episodes before the
sim's per-episode memory creep exhausts the reduced headroom on a 32 GB card).
Deploy with ``glue_cache`` alone, or use a >32 GB card for long sim runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from flexpi.utils.logging_config import get_logger

logger = get_logger(__name__)

_PREFILL_IN = ["anchor_tokens", "anchor_freqs_real", "anchor_t_mod", "anchor_ctx", "anchor_ctx_mask"]
_PREFILL_OUT = ["k_all", "v_all"]
_DECODE_IN = [
    "noisy_video", "action", "noisy_freqs_real", "action_freqs_real",
    "noisy_t_mod", "action_t_mod", "noisy_ctx", "noisy_ctx_mask",
    "action_ctx", "action_ctx_mask", "k_all", "v_all",
]
_DECODE_OUT = ["noisy_video_out", "action_out"]


def _engine_devmem_size(engine) -> int:
    for attr in ("device_memory_size_v2", "device_memory_size"):
        v = getattr(engine, attr, None)
        if isinstance(v, int) and v > 0:
            return v
    return 0


class _EngineRunner:
    """execute_async_v3 over torch-owned static buffers, bound by name.

    Context creation is deferred to :meth:`finalize` so the prefill and decode
    contexts can share ONE device-memory scratch (they never run concurrently:
    prefill once, then decode ×10) — halving the transient activation memory.
    """

    def __init__(self, engine_path, device):
        import tensorrt as trt
        self._trt = trt
        self.device = device
        rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = rt.deserialize_cuda_engine(Path(engine_path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TRT engine {engine_path}.")
        self.devmem_size = _engine_devmem_size(self.engine)

    def finalize(self, shared_devmem=None):
        trt = self._trt
        self.ctx = None
        if shared_devmem is not None:
            try:
                self.ctx = self.engine.create_execution_context(
                    trt.ExecutionContextAllocationStrategy.USER_MANAGED)
                self.ctx.set_device_memory(shared_devmem.data_ptr(), shared_devmem.numel())
            except Exception as exc:  # API skew → fall back to self-managed memory
                logger.warning("trt_joint_split: shared device memory unavailable (%s); self-managed.", exc)
                self.ctx = None
        if self.ctx is None:
            self.ctx = self.engine.create_execution_context()
        t2t = {
            trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16,
            trt.DataType.BF16: torch.bfloat16, trt.DataType.INT64: torch.int64,
            trt.DataType.INT32: torch.int32, trt.DataType.BOOL: torch.bool,
        }
        self.names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.is_in = {n: self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT for n in self.names}
        self.buffers = {}
        for n in self.names:
            shape = tuple(self.ctx.get_tensor_shape(n))
            self.buffers[n] = torch.empty(shape, dtype=t2t[self.engine.get_tensor_dtype(n)], device=self.device)
            self.ctx.set_tensor_address(n, self.buffers[n].data_ptr())
        self.in_shapes = {n: tuple(self.buffers[n].shape) for n in self.names if self.is_in[n]}
        return self

    def __call__(self, feed: dict, out_names):
        for n, t in feed.items():
            self.buffers[n].copy_(t.to(self.buffers[n].dtype), non_blocking=True)
        self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        return tuple(self.buffers[n] for n in out_names)


def _real(freqs: torch.Tensor) -> torch.Tensor:
    """complex [S,1,rope] -> real f32 [S,1,rope,2] (engine RoPE input)."""
    return torch.view_as_real(freqs.contiguous()).to(torch.float32).contiguous()


def install_trt_joint_split_adapter(
    model, prefill_engine_path, decode_engine_path, free_video_blocks: bool = False,
):
    device = model.device
    orig = model.mot._forward_joint_inner
    # Free the torch video-expert weights BEFORE loading the two engines
    # (~21 GB combined) so peak memory fits in 32 GB. With the engines active
    # the fixed-shape deploy path never touches the torch fallback.
    if free_video_blocks:
        # Both experts' BLOCK weights are baked into the engines; only the
        # pre_dit embeddings (time/patch/text) still run in torch. Free both.
        freed = 0
        for expert in ("video", "action"):
            for p in model.mot.mixtures[expert].blocks.parameters():
                freed += p.numel() * p.element_size()
                p.data = torch.empty(0, device=p.device, dtype=p.dtype)
        torch.cuda.empty_cache()
        logger.info("trt_joint_split: freed %.1f GB torch video+action block weights.", freed / 1e9)
    prefill = _EngineRunner(prefill_engine_path, device)
    decode = _EngineRunner(decode_engine_path, device)
    # Share ONE device-memory scratch between the two contexts (sequential use).
    shared = None
    scratch = max(prefill.devmem_size, decode.devmem_size)
    if scratch > 0:
        shared = torch.empty(scratch, dtype=torch.uint8, device=device)
        logger.info("trt_joint_split: shared %.1f GB device-memory scratch across both engines.", scratch / 1e9)
    prefill.finalize(shared)
    decode.finalize(shared)
    # Keep the shared scratch alive for the process lifetime (the contexts hold
    # a raw pointer into it — GC here would corrupt every engine call → NaN).
    prefill._shared_devmem = shared
    warned = [False]
    st = {"key": None}          # cached split geometry (masks/idx/real-freqs)
    pf = {"anchors": None, "k_all": None, "v_all": None}  # prefill cache

    def _build_split(video_t_mod, Sv, Sa):
        # Only the anchor/noisy token partition is needed at runtime; the
        # attention masks are baked into the two engines at export time, so the
        # adapter never feeds a mask (unlike the eager validation path).
        tm = video_t_mod[0]
        anchor_bool = (tm == tm[0]).all(dim=(-1, -2))
        noisy_bool = ~anchor_bool
        return dict(
            anchor_bool=anchor_bool, noisy_bool=noisy_bool,
            anchor_g=torch.where(anchor_bool)[0],
            noisy_video_g=torch.where(noisy_bool)[0],
            Sv=Sv,
        )

    def _slice_ctx_mask(m, sel_bool, Sv):
        d = [i for i, s in enumerate(m.shape) if s == Sv][0]
        return m.index_select(d, torch.where(sel_bool)[0])

    def _forward_joint_inner_split(
        video_tokens, action_tokens, video_freqs, action_freqs,
        video_t_mod, action_t_mod, video_context_payload, action_context_payload,
        attention_mask, sub_stream_lens=None, sub_stream_self_masks=None,
    ):
        Sv, Sa = video_tokens.shape[1], action_tokens.shape[1]
        ok = (
            attention_mask is not None and sub_stream_self_masks is not None
            and (Sv, Sa) == (decode.in_shapes["noisy_video"][1] + prefill.in_shapes["anchor_tokens"][1],
                             decode.in_shapes["action"][1])
        )
        if not ok:
            if free_video_blocks:
                raise RuntimeError("trt_joint_split: off-engine call but torch weights freed.")
            if not warned[0]:
                warned[0] = True
                logger.warning("trt_joint_split: shape mismatch — torch fallback (Sv=%d Sa=%d).", Sv, Sa)
            return orig(
                video_tokens=video_tokens, action_tokens=action_tokens,
                video_freqs=video_freqs, action_freqs=action_freqs,
                video_t_mod=video_t_mod, action_t_mod=action_t_mod,
                video_context_payload=video_context_payload,
                action_context_payload=action_context_payload,
                attention_mask=attention_mask, sub_stream_lens=sub_stream_lens,
                sub_stream_self_masks=sub_stream_self_masks,
            )

        key = (Sv, Sa)
        if st["key"] != key:
            st["key"] = key
            st.update(_build_split(video_t_mod, Sv, Sa))
            pf["anchors"] = None  # regime changed → invalidate prefill cache
        ab, nb = st["anchor_bool"], st["noisy_bool"]
        ag, nvg = st["anchor_g"], st["noisy_video_g"]

        # ---- prefill (once per infer call; anchors are step-invariant) ----
        anchor_tokens = video_tokens[:, ab, :]
        if pf["anchors"] is None or not torch.equal(anchor_tokens, pf["anchors"]):
            k_all, v_all = prefill({
                "anchor_tokens": anchor_tokens,
                "anchor_freqs_real": _real(video_freqs[ag]),
                "anchor_t_mod": video_t_mod[:, ab],
                "anchor_ctx": video_context_payload["context"],
                "anchor_ctx_mask": _slice_ctx_mask(video_context_payload["mask"], ab, Sv),
            }, _PREFILL_OUT)
            pf["anchors"] = anchor_tokens.clone()
            pf["k_all"] = k_all.clone()
            pf["v_all"] = v_all.clone()

        # ---- decode (every step) ----
        nvo, ao = decode({
            "noisy_video": video_tokens[:, nb, :],
            "action": action_tokens,
            "noisy_freqs_real": _real(video_freqs[nvg]),
            "action_freqs_real": _real(action_freqs),
            "noisy_t_mod": video_t_mod[:, nb],
            "action_t_mod": action_t_mod,
            "noisy_ctx": video_context_payload["context"],
            "noisy_ctx_mask": _slice_ctx_mask(video_context_payload["mask"], nb, Sv),
            "action_ctx": action_context_payload["context"],
            "action_ctx_mask": action_context_payload["mask"],
            "k_all": pf["k_all"],
            "v_all": pf["v_all"],
        }, _DECODE_OUT)

        # ---- reassemble merged_out (anchor rows zero = reclamped downstream) ----
        merged = video_tokens.new_zeros((video_tokens.shape[0], Sv, nvo.shape[-1]))
        merged.index_copy_(1, nvg, nvo.to(video_tokens.dtype))
        return merged, ao.to(action_tokens.dtype)

    model.mot._forward_joint_inner = _forward_joint_inner_split
    logger.info("trt_joint_split installed (prefill=%s, decode=%s).", prefill_engine_path, decode_engine_path)
    return _forward_joint_inner_split
