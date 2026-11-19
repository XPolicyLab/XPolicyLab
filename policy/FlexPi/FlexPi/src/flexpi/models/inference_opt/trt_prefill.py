"""Raw TensorRT engine runtime for the action-only KV-cache PREFILL.

The action fast path's biggest phase after denoise is the one-shot 30-layer
video-expert prefill over the ~534 anchor tokens (~44 ms eager / ~25 ms
in-graph — larger than all 10 cached denoise steps combined). It is a
fixed-shape bidirectional pass, the same shape class where the joint core
engine beat inductor ~23%. This mirrors ``trt_joint.py`` for that pass.

The engine is built offline by ``scripts/inference_opt/trt_onnx_prefill_engine.py`` (trt29
env) from CAPTURED real inputs (no shape guessing) and emits the per-layer
K/V as two stacked tensors K_all/V_all [num_layers, B, Sv, H*Dh]; the adapter
unstacks them into the ``list[{"k","v"}]`` that ``prefill_video_cache``
returns. ``prepare_for_inference(trt_prefill_engine_path=...)`` shadows
``model.mot.prefill_video_cache``; off-shape calls fall back to torch.
"""

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

# Export contract of scripts/inference_opt/trt_onnx_prefill_engine.py (positional order).
_INPUT_NAMES = [
    "merged_tokens", "merged_freqs_real", "merged_t_mod",
    "video_context", "merged_ctx_mask", "prefill_attention_mask",
]
_OUTPUT_NAMES = ["k_all", "v_all"]


class TrtPrefillRunner:
    """execute_async_v3 runner over torch-owned buffers, bound by name.

    Runs on the caller's current CUDA stream (single-stream ordering, no host
    sync). Outputs are views into reused buffers, valid until the next call —
    which matches the one-prefill-per-infer-call cadence.
    """

    def __init__(self, engine_path: str | Path, device):
        import tensorrt as trt

        engine_path = Path(engine_path)
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(
                f"Failed to deserialize TRT prefill engine {engine_path} — "
                "engines are version-locked to the builder's tensorrt."
            )
        self.ctx = self.engine.create_execution_context()
        if self.ctx is None:
            raise RuntimeError(
                f"Failed to create TRT execution context for {engine_path} (OOM?)."
            )
        _trt2torch = {
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF: torch.float16,
            trt.DataType.BF16: torch.bfloat16,
            trt.DataType.INT64: torch.int64,
            trt.DataType.INT32: torch.int32,
            trt.DataType.BOOL: torch.bool,
        }
        io_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        missing = [n for n in _INPUT_NAMES + _OUTPUT_NAMES if n not in io_names]
        if missing:
            raise RuntimeError(
                f"TRT prefill engine {engine_path} missing IO tensors: {missing}."
            )
        self.buffers = {}
        for name in io_names:
            shape = tuple(self.ctx.get_tensor_shape(name))
            dtype = _trt2torch[self.engine.get_tensor_dtype(name)]
            self.buffers[name] = torch.empty(shape, dtype=dtype, device=device)
            self.ctx.set_tensor_address(name, self.buffers[name].data_ptr())
        self.input_shapes = {n: tuple(self.buffers[n].shape) for n in _INPUT_NAMES}
        self.num_layers = int(self.buffers["k_all"].shape[0])

    def matches(self, merged_tokens, video_context) -> bool:
        return (
            tuple(merged_tokens.shape) == self.input_shapes["merged_tokens"]
            and tuple(video_context.shape) == self.input_shapes["video_context"]
        )

    def __call__(self, merged_tokens, merged_freqs_real, merged_t_mod,
                 video_context, merged_ctx_mask, prefill_attention_mask):
        vals = [merged_tokens, merged_freqs_real, merged_t_mod,
                video_context, merged_ctx_mask, prefill_attention_mask]
        for name, t in zip(_INPUT_NAMES, vals):
            self.buffers[name].copy_(t.to(self.buffers[name].dtype), non_blocking=True)
        self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        k_all, v_all = self.buffers["k_all"], self.buffers["v_all"]
        return [{"k": k_all[i], "v": v_all[i]} for i in range(self.num_layers)]


def _freqs_to_real_f32(freqs: torch.Tensor) -> torch.Tensor:
    return torch.view_as_real(freqs.contiguous()).to(torch.float32).contiguous()


def install_trt_prefill_adapter(model, engine_path: str | Path) -> TrtPrefillRunner:
    """Shadow ``model.mot.prefill_video_cache`` with the TRT engine route.

    Instance-attribute shadowing; off-shape calls (or a non-None sub-mask
    config the engine didn't bake) fall back to the original method. The
    engine bakes any HBridge sub-stream self-masks for the build-time config,
    so ``video_sub_stream_*`` args are consumed only by the fallback.
    """
    runner = TrtPrefillRunner(engine_path, model.device)
    orig = model.mot.prefill_video_cache
    warned = [False]

    def prefill_video_cache_trt(
        video_tokens, video_freqs, video_t_mod, video_context_payload,
        video_attention_mask, video_sub_stream_lens=None,
        video_sub_stream_self_masks=None,
    ):
        ctx = video_context_payload["context"]
        if not runner.matches(video_tokens, ctx):
            if not warned[0]:
                warned[0] = True
                logger.warning(
                    "TRT prefill adapter: falling back to torch — merged %s / "
                    "ctx %s vs engine %s. Rebuild the engine at these shapes.",
                    tuple(video_tokens.shape), tuple(ctx.shape),
                    runner.input_shapes["merged_tokens"],
                )
            return orig(
                video_tokens=video_tokens, video_freqs=video_freqs,
                video_t_mod=video_t_mod, video_context_payload=video_context_payload,
                video_attention_mask=video_attention_mask,
                video_sub_stream_lens=video_sub_stream_lens,
                video_sub_stream_self_masks=video_sub_stream_self_masks,
            )
        mask = video_attention_mask
        if mask is None:  # auto backend drops the all-True prefill mask
            Sv = video_tokens.shape[1]
            mask = torch.ones(Sv, Sv, dtype=torch.bool, device=video_tokens.device)
        return runner(
            video_tokens, _freqs_to_real_f32(video_freqs), video_t_mod,
            ctx, video_context_payload["mask"], mask,
        )

    model.mot.prefill_video_cache = prefill_video_cache_trt
    logger.info(
        "TRT prefill engine installed from %s (merged %s, %d layers).",
        engine_path, runner.input_shapes["merged_tokens"], runner.num_layers,
    )
    return runner
