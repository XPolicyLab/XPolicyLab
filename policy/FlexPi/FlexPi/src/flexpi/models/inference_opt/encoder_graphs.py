"""CUDA-graph capture for the three first-frame anchor encoders (report §9.7).

The joint-path encoders (VAE composite encode, DINO first-frame encode,
pointmap depth encode) are host-LAUNCH-rate-limited: ~7-12 ms of Python /
kernel-launch work each, roughly equal to their GPU time — which is why both
side-stream overlap and ``compile_encoders`` measured as no-ops on this path
(§9.6). Capturing each encoder as a CUDA graph replays it as a single launch,
removing the host cost.

Guarded instance-shadows, same pattern as ``trt_joint`` / ``trt_prefill``:

* ``install_encoder_cuda_graphs(model)`` shadows
  ``model._encode_input_image_latents_tensor``,
  ``model.dino_encoder.encode_video`` (deploy first-frame call only), and
  ``model._encode_first_frame_pointmap_raw``.
* The first deploy-signature call captures: 2 warmup runs on a side stream
  (cuDNN/cuBLAS workspace init), then ``torch.cuda.graph``, then a replay
  parity self-check (``torch.equal`` vs the warmup output — the encoders are
  deterministic, so replay must be bitwise). Any capture exception or parity
  mismatch → warn once, permanent eager fallback for that encoder.
* Off-signature calls (different shapes / kwargs — e.g. val-time full-video
  DINO encodes, tiled VAE) run eager through the saved original.
* Static-input staging happens OUTSIDE the graph (``copy_`` into fixed
  buffers), so CPU/pageable inputs stay legal. The pointmap graph stages
  camera intrinsics explicitly per call — ``set_camera_intrinsics()`` rebinds
  the attr per episode, so the graph must never bake that attr's address.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from flexpi.utils.logging_config import get_logger

logger = get_logger(__name__)

# A failed capture can leave the caching allocator's capture state poisoned
# (`captures_underway` internal assert on the NEXT capture attempt) — once any
# slot fails, no other slot may try to capture in this process.
_CAPTURE_POISONED = False


class _GraphSlot:
    """One captured graph: warmup → capture → bitwise replay self-check."""

    def __init__(self, label: str):
        self.label = label
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.out: Optional[torch.Tensor] = None
        self.failed = False

    def capture(self, run):
        global _CAPTURE_POISONED
        if _CAPTURE_POISONED:
            raise RuntimeError("a prior encoder capture failed (allocator capture state unsafe)")
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
        if not torch.equal(out, ref):
            raise RuntimeError(f"{self.label}: replay differs from eager warmup")
        self.graph, self.out = graph, out
        logger.info("encoder_cuda_graph: %s captured (single-launch replay).", self.label)

    def fail(self, exc: Exception):
        global _CAPTURE_POISONED
        self.failed = True
        if self.graph is None:
            # Failure DURING capture (not replay) may have poisoned the
            # allocator's capture bookkeeping — stop all further captures.
            _CAPTURE_POISONED = True
        self.graph = None
        logger.warning(
            "encoder_cuda_graph: %s capture failed — permanent eager fallback (%s)",
            self.label, exc,
        )


def _same_sig(buf: Dict[str, torch.Tensor], d: Dict[str, torch.Tensor]) -> bool:
    return set(buf) == set(d) and all(
        buf[k].shape == d[k].shape and buf[k].dtype == d[k].dtype for k in buf
    )


class _GraphedVaeEncode:
    def __init__(self, model):
        self._model = model
        self._orig = model._encode_input_image_latents_tensor
        self._slot = _GraphSlot("vae_encode")
        self._img: Optional[torch.Tensor] = None

    def __call__(self, input_image, tiled=False, **kwargs):
        slot = self._slot
        if (
            slot.failed or tiled or kwargs
            or not isinstance(input_image, torch.Tensor) or input_image.ndim != 4
            or (self._img is not None and (
                input_image.shape != self._img.shape or input_image.dtype != self._img.dtype))
        ):
            return self._orig(input_image=input_image, tiled=tiled, **kwargs)
        try:
            if slot.graph is None:
                self._img = torch.empty(
                    input_image.shape, dtype=input_image.dtype, device=self._model.device,
                )
                self._img.copy_(input_image)
                slot.capture(lambda: self._orig(input_image=self._img, tiled=False))
            else:
                self._img.copy_(input_image, non_blocking=True)
                slot.graph.replay()
            return slot.out
        except Exception as exc:  # capture-time only; replay cannot raise
            slot.fail(exc)
            return self._orig(input_image=input_image, tiled=tiled)


class _GraphedDinoFirstFrame:
    """Shadows ``dino_encoder.encode_video`` for the deploy first-frame call
    (``video=None, per_cam={...}, first_frame_only=True``). Anything else —
    composite path, full-video val encodes — runs eager."""

    def __init__(self, dino_encoder):
        self._enc = dino_encoder
        self._orig = dino_encoder.encode_video
        self._slot = _GraphSlot("dino_encode")
        self._bufs: Optional[Dict[str, torch.Tensor]] = None
        self._kwargs_sig = None

    def __call__(self, video=None, per_cam=None, first_frame_only=False, **kwargs):
        slot = self._slot
        graphable = (
            not slot.failed and video is None and isinstance(per_cam, dict)
            and first_frame_only
            and all(isinstance(v, torch.Tensor) and v.is_cuda for v in per_cam.values())
        )
        if graphable and self._bufs is not None:
            graphable = _same_sig(self._bufs, per_cam) and self._kwargs_sig == repr(sorted(kwargs.items()))
        if not graphable:
            return self._orig(
                video=video, per_cam=per_cam, first_frame_only=first_frame_only, **kwargs
            )
        try:
            if slot.graph is None:
                self._bufs = {
                    k: torch.empty_like(v) for k, v in per_cam.items()
                }
                for k in self._bufs:
                    self._bufs[k].copy_(per_cam[k])
                self._kwargs_sig = repr(sorted(kwargs.items()))
                slot.capture(lambda: self._orig(
                    video=None, per_cam=self._bufs, first_frame_only=True, **kwargs
                ))
            else:
                for k in self._bufs:
                    self._bufs[k].copy_(per_cam[k], non_blocking=True)
                slot.graph.replay()
            return slot.out
        except Exception as exc:
            slot.fail(exc)
            return self._orig(
                video=video, per_cam=per_cam, first_frame_only=first_frame_only, **kwargs
            )


class _GraphedPointmapEncode:
    def __init__(self, model):
        self._model = model
        self._orig = model._encode_first_frame_pointmap_raw
        self._slot = _GraphSlot("ptmap_encode")
        self._bufs: Optional[Dict[str, torch.Tensor]] = None
        self._k: Optional[torch.Tensor] = None

    def __call__(self, camera_intrinsics=None, tiled=False, per_cam_depth=None):
        slot = self._slot
        graphable = (
            not slot.failed and not tiled and isinstance(per_cam_depth, dict)
            and all(isinstance(v, torch.Tensor) for v in per_cam_depth.values())
        )
        if graphable and self._bufs is not None:
            graphable = _same_sig(self._bufs, per_cam_depth)
        if not graphable:
            return self._orig(
                camera_intrinsics=camera_intrinsics, tiled=tiled, per_cam_depth=per_cam_depth,
            )
        try:
            m = self._model
            k_now = m._resolve_camera_intrinsics(camera_intrinsics)
            if slot.graph is None:
                self._bufs = {
                    k: torch.empty(v.shape, dtype=v.dtype, device=m.device)
                    for k, v in per_cam_depth.items()
                }
                for k in self._bufs:
                    self._bufs[k].copy_(per_cam_depth[k])
                # Intrinsics change per episode (set_camera_intrinsics rebinds
                # the attr) — pass a static buffer EXPLICITLY so the graph
                # never bakes the rebound attr's address.
                self._k = k_now.detach().clone()
                slot.capture(lambda: self._orig(
                    camera_intrinsics=self._k, tiled=False, per_cam_depth=self._bufs,
                ))
            else:
                self._k.copy_(k_now, non_blocking=True)
                for k in self._bufs:
                    self._bufs[k].copy_(per_cam_depth[k], non_blocking=True)
                slot.graph.replay()
            return slot.out
        except Exception as exc:
            slot.fail(exc)
            return self._orig(
                camera_intrinsics=camera_intrinsics, tiled=tiled, per_cam_depth=per_cam_depth,
            )


def _hoist_capture_illegal_syncs(model) -> None:
    """Hoist the per-forward synced constants the §9.7 probe found — each was
    a capture-killer AND a per-call pageable H2D on the eager path.

    * WAN-VAE ``scale`` (mean/std) lives on CPU; ``single_encode`` does
      ``.to(device)`` every call. Pre-move once — the in-forward ``.to``
      becomes a device-side cast only (identical math).
    * timm EVA rope rebuilds its sin/cos table every forward in dynamic-size
      mode (``get_embed(shape=(H, W))`` bypasses timm's own cache). Deploy
      shapes are fixed — memoize per shape. Pure function at eval (coord augs
      are train-only); the memo bypasses itself under ``rope.training``.
    """
    vae = getattr(model, "vae", None)
    if (
        vae is not None and isinstance(getattr(vae, "scale", None), (list, tuple))
        and len(vae.scale) and isinstance(vae.scale[0], torch.Tensor)
    ):
        vae.scale = [s.to(device=model.device) for s in vae.scale]

    enc = getattr(model, "dino_encoder", None)
    rope = getattr(getattr(enc, "model", None), "rope", None) if enc is not None else None
    if rope is not None and not getattr(rope, "_flexpi_memo_get_embed", False):
        orig_get = rope.get_embed
        memo: Dict = {}

        def _memo_get_embed(shape=None):
            if rope.training:
                return orig_get(shape) if shape is not None else orig_get()
            key = None if shape is None else tuple(shape)
            hit = memo.get(key)
            if hit is None:
                hit = memo[key] = orig_get(shape) if shape is not None else orig_get()
            return hit

        rope.get_embed = _memo_get_embed
        rope._flexpi_memo_get_embed = True


def install_encoder_cuda_graphs(model) -> None:
    _hoist_capture_illegal_syncs(model)
    model._encode_input_image_latents_tensor = _GraphedVaeEncode(model)
    if getattr(model, "dino_encoder", None) is not None:
        model.dino_encoder.encode_video = _GraphedDinoFirstFrame(model.dino_encoder)
    if hasattr(model, "_encode_first_frame_pointmap_raw"):
        model._encode_first_frame_pointmap_raw = _GraphedPointmapEncode(model)
    logger.info("encoder_cuda_graph: shadows installed (capture on first deploy call).")
