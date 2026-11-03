"""Manual CUDA-graph capture of the whole 10-step joint denoise loop (Lever 2).

The loop-scope inductor compile (``_run_joint_denoise_loop``) already captures
``_joint_denoise_loop_body`` as a CUDA graph in the NON-engine case — proving
the body (pre_dit → ``_forward_joint_inner`` → post_dit → x0→v → Euler steps →
frame-0 reclamps) is capture-clean. But inductor refuses to trace the opaque
TRT engine call, so with ``trt_joint_engine_path`` active the body runs eager,
paying the per-step host-launch cost ×10.

This module captures the SAME body MANUALLY: ``torch.cuda.graph`` CAN contain
the engine's ``execute_async_v3`` replay (the runner is capture-clean — static
buffers, current stream, no host sync; ``trt_joint.py:42-50``). Math is
bit-identical to the eager loop under Euler + step-skip-off (the only regime it
engages), verified by a warmup-vs-replay ``torch.equal`` self-check.

Guarded instance-shadow, same pattern as ``encoder_graphs`` / ``trt_joint``:
one graph is captured for the deploy shape/regime on the first call; different
shapes/regimes or any capture failure → permanent eager fallback via the saved
``_joint_denoise_loop_body``. The process-wide capture-poison flag is shared
with ``encoder_graphs`` — a failed capture in either poisons the allocator, so
neither may capture afterward.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from . import encoder_graphs as _eg
from flexpi.utils.logging_config import get_logger

logger = get_logger(__name__)


def _graph_out_equal(a, b) -> bool:
    """Bitwise-equal for the body's 4-tuple output (entries may be None)."""
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x is None and y is None:
            continue
        if x is None or y is None:
            return False
        if not torch.equal(x, y):
            return False
    return True


class _GraphedJointLoop:
    """Shadow for the joint denoise loop: capture once, replay per call."""

    def __init__(self, model):
        self._model = model
        self._body = model._joint_denoise_loop_body
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.failed = False
        self._static: Optional[Dict[str, torch.Tensor]] = None
        self._others: Dict[str, object] = {}
        self._sig: Optional[str] = None
        self._out: Optional[Tuple] = None

    # -- signature so a regime/shape change falls back instead of miscapturing
    @staticmethod
    def _sig_of(tensors: Dict[str, torch.Tensor], others: Dict[str, object]) -> str:
        t = ";".join(
            f"{k}:{tuple(v.shape)}:{v.dtype}" for k, v in sorted(tensors.items())
        )
        o = ";".join(f"{k}={v!r}" for k, v in sorted(others.items()))
        return t + "|" + o

    def __call__(self, **kwargs):
        tensors = {k: v for k, v in kwargs.items() if isinstance(v, torch.Tensor)}
        others = {k: v for k, v in kwargs.items() if not isinstance(v, torch.Tensor)}
        sig = self._sig_of(tensors, others)

        # off-signature (regime/shape change) or already-failed → eager
        if self.failed or (self._sig is not None and sig != self._sig):
            return self._body(**kwargs)
        if _eg._CAPTURE_POISONED and self.graph is None:
            self.failed = True
            return self._body(**kwargs)

        try:
            if self.graph is None:
                self._capture(tensors, others, sig)
            else:
                for k, buf in self._static.items():
                    buf.copy_(tensors[k], non_blocking=True)
                self.graph.replay()
            return self._out
        except Exception as exc:  # capture-time only; replay cannot raise
            self._fail(exc)
            return self._body(**kwargs)

    def _capture(self, tensors, others, sig) -> None:
        # Static input buffers (copy inputs OUT of the graph so pageable /
        # per-call tensors stay legal); non-tensor kwargs (bools, None) are
        # baked at capture — a change trips the signature guard above.
        self._static = {k: torch.empty_like(v) for k, v in tensors.items()}
        for k, v in tensors.items():
            self._static[k].copy_(v)
        self._others = dict(others)
        self._sig = sig

        def run():
            with self._model._sdpa_priority_ctx():
                return self._body(**self._static, **self._others)

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                ref = run()
        torch.cuda.current_stream().wait_stream(side)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = run()
        graph.replay()
        torch.cuda.synchronize()
        if not _graph_out_equal(out, ref):
            raise RuntimeError("joint_loop: replay differs from eager warmup")
        self.graph, self._out = graph, out
        logger.info(
            "joint_loop_cuda_graph: captured 10-step loop (single-launch replay)."
        )

    def _fail(self, exc: Exception) -> None:
        self.failed = True
        if self.graph is None:
            # Failure DURING capture may have poisoned the allocator's capture
            # bookkeeping — stop all further captures (shared with encoder_graphs).
            _eg._CAPTURE_POISONED = True
        self.graph = None
        logger.warning(
            "joint_loop_cuda_graph: capture failed — permanent eager fallback (%s)",
            exc,
        )


def install_joint_loop_cuda_graph(model) -> None:
    model._joint_loop_graph_runner = _GraphedJointLoop(model)
    logger.info(
        "joint_loop_cuda_graph: shadow installed (capture on first deploy call)."
    )
