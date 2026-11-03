"""Raw TensorRT engine runtime for the flex joint denoise core.

VA2-style deploy split: the engine is built OFFLINE by
``scripts/inference_opt/trt_onnx_joint_engine.py`` (trt29 env — the torch>=2.12 dynamo
exporter) and serialized to disk; this module needs only the ``tensorrt``
runtime wheel (>=10.16, matching the builder) plus torch.
``prepare_for_inference(trt_joint_engine_path=...)`` calls
:func:`install_trt_joint_adapter`, which shadows
``model.mot._forward_joint_inner`` with an adapter routing the 30-layer
joint core through one TRT engine call. Encoders, pre/post_dit, schedulers,
regime-FiLM and the action-only fast path all stay in torch.

Baked into the engine at build time: real-valued RoPE (the adapter converts
the torch complex freqs via ``view_as_real``), the HBridge sub-stream
self-masks as graph constants, and static deploy shapes. The dense joint
attention mask is a genuine bool input (non-droppable, unlike the action
fast path). Off-shape calls — other regimes, other context lengths — fall
back to the original torch implementation via the shape guard, so the knob
never changes behavior outside the exact shape it was built for.
"""

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

# Export contract of scripts/inference_opt/trt_onnx_joint_engine.py (positional order).
_INPUT_NAMES = [
    "video_tokens", "action_tokens",
    "video_freqs_real", "action_freqs_real",
    "video_t_mod", "action_t_mod",
    "video_context", "video_context_mask",
    "action_context", "action_context_mask",
    "attention_mask",
]
_OUTPUT_NAMES = ["merged_out", "action_out"]


class TrtJointRunner:
    """execute_async_v3 runner over torch-owned buffers, bound by name.

    Everything runs on the caller's current CUDA stream — input copies, the
    engine launch, and downstream torch consumers are strictly ordered, so no
    cross-stream syncs (or host syncs) are needed inside the denoise loop.
    Returned outputs are views into reused buffers: valid until the next
    call, which matches the per-step consume-then-overwrite pattern of the
    denoise loop.
    """

    def __init__(self, engine_path: str | Path, device):
        import tensorrt as trt

        engine_path = Path(engine_path)
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(
                f"Failed to deserialize TRT engine {engine_path} — engines are "
                "version-locked; the installed tensorrt runtime must match the "
                "builder version (see scripts/inference_opt/trt_onnx_joint_engine.py)."
            )
        self.ctx = self.engine.create_execution_context()
        if self.ctx is None:
            raise RuntimeError(
                f"Failed to create TRT execution context for {engine_path} "
                "(usually out of GPU memory)."
            )
        _trt2torch = {
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF: torch.float16,
            trt.DataType.BF16: torch.bfloat16,
            trt.DataType.INT64: torch.int64,
            trt.DataType.INT32: torch.int32,
            trt.DataType.BOOL: torch.bool,
        }
        io_names = [
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
        ]
        missing = [n for n in _INPUT_NAMES + _OUTPUT_NAMES if n not in io_names]
        if missing:
            raise RuntimeError(
                f"TRT engine {engine_path} does not match the joint-core export "
                f"contract (missing IO tensors: {missing})."
            )
        # Static engine: shapes are known up front for inputs and outputs.
        self.buffers = {}
        for name in io_names:
            shape = tuple(self.ctx.get_tensor_shape(name))
            dtype = _trt2torch[self.engine.get_tensor_dtype(name)]
            self.buffers[name] = torch.empty(shape, dtype=dtype, device=device)
            self.ctx.set_tensor_address(name, self.buffers[name].data_ptr())
        self.input_shapes = {n: tuple(self.buffers[n].shape) for n in _INPUT_NAMES}

    def matches(self, video_tokens, action_tokens, video_context) -> bool:
        """Cheap routing guard on the shape-defining inputs."""
        return (
            tuple(video_tokens.shape) == self.input_shapes["video_tokens"]
            and tuple(action_tokens.shape) == self.input_shapes["action_tokens"]
            and tuple(video_context.shape) == self.input_shapes["video_context"]
        )

    def __call__(self, *inputs):
        for name, t in zip(_INPUT_NAMES, inputs):
            buf = self.buffers[name]
            buf.copy_(t.to(buf.dtype), non_blocking=True)
        self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        return self.buffers["merged_out"], self.buffers["action_out"]


def _freqs_to_real_f32(freqs: torch.Tensor) -> torch.Tensor:
    """Complex [S, 1, hd/2] -> real [S, 1, hd/2, 2] f32 (engine RoPE input)."""
    return torch.view_as_real(freqs.contiguous()).to(torch.float32).contiguous()


def install_trt_joint_adapter(
    model, engine_path: str | Path, free_video_blocks: bool = False
) -> TrtJointRunner:
    """Shadow ``model.mot._forward_joint_inner`` with the TRT engine route.

    Instance-attribute shadowing (same pattern as ``compile_encoders``): the
    class method is untouched, and any call whose shapes don't match the
    engine — or that carries no dense mask — falls back to it. The engine
    bakes the HBridge sub-stream masks for the build-time shapes/config, so
    ``sub_stream_*`` args are consumed by the fallback only; parity at real
    activations is validated by scripts/inference_opt/trt_joint_parity.py.

    ``free_video_blocks=True`` releases the torch-side video-expert BLOCK
    weights (~8.5 GB bf16) after the engine loads — the engine carries its
    own fp16 copy, and with the joint core routed to TRT those blocks never
    execute. Needed when the engine must coexist with a VRAM-hungry
    neighbor (e.g. the SAPIEN Vulkan renderer in sim evals). The torch
    fallback is then impossible, so off-shape calls RAISE instead of
    falling back, and the model can no longer serve action-regime prefill
    or be checkpointed. Deploy-only, fixed-shape use.
    """
    runner = TrtJointRunner(engine_path, model.device)
    orig = model.mot._forward_joint_inner
    fallback_warned = [False]
    if free_video_blocks:
        freed = 0
        for p in model.mot.mixtures["video"].blocks.parameters():
            freed += p.numel() * p.element_size()
            p.data = torch.empty(0, device=p.device, dtype=p.dtype)
        torch.cuda.empty_cache()
        logger.info(
            "TRT joint adapter: freed %.1f GB of torch-side video-expert "
            "block weights (engine carries its own copy; torch fallback for "
            "the joint core is now disabled).",
            freed / 1e9,
        )

    def _forward_joint_inner_trt(
        video_tokens, action_tokens, video_freqs, action_freqs,
        video_t_mod, action_t_mod,
        video_context_payload, action_context_payload,
        attention_mask, sub_stream_lens=None, sub_stream_self_masks=None,
    ):
        if attention_mask is None or not runner.matches(
            video_tokens, action_tokens, video_context_payload["context"]
        ):
            if free_video_blocks:
                raise RuntimeError(
                    "TRT joint adapter: off-engine call (mask "
                    f"{'None' if attention_mask is None else 'present'}, "
                    f"merged {tuple(video_tokens.shape)} / ctx "
                    f"{tuple(video_context_payload['context'].shape)} vs engine "
                    f"{runner.input_shapes['video_tokens']}) but the torch "
                    "fallback weights were freed (free_video_blocks=True). "
                    "Rebuild the engine at these shapes."
                )
            if not fallback_warned[0]:
                fallback_warned[0] = True
                logger.warning(
                    "TRT joint adapter: falling back to the torch path — "
                    "mask %s, merged %s / action %s / ctx %s vs engine %s. "
                    "Rebuild the engine at these shapes if TRT was intended.",
                    None if attention_mask is None else "present",
                    tuple(video_tokens.shape), tuple(action_tokens.shape),
                    tuple(video_context_payload["context"].shape),
                    runner.input_shapes["video_tokens"],
                )
            return orig(
                video_tokens=video_tokens, action_tokens=action_tokens,
                video_freqs=video_freqs, action_freqs=action_freqs,
                video_t_mod=video_t_mod, action_t_mod=action_t_mod,
                video_context_payload=video_context_payload,
                action_context_payload=action_context_payload,
                attention_mask=attention_mask,
                sub_stream_lens=sub_stream_lens,
                sub_stream_self_masks=sub_stream_self_masks,
            )
        merged_out, action_out = runner(
            video_tokens, action_tokens,
            _freqs_to_real_f32(video_freqs), _freqs_to_real_f32(action_freqs),
            video_t_mod, action_t_mod,
            video_context_payload["context"], video_context_payload["mask"],
            action_context_payload["context"], action_context_payload["mask"],
            attention_mask,
        )
        # Engine output dtype follows the export path (dynamo -> bf16,
        # TS -> fp32); downstream post_dit linears expect the stream dtype.
        return merged_out.to(video_tokens.dtype), action_out.to(action_tokens.dtype)

    model.mot._forward_joint_inner = _forward_joint_inner_trt
    logger.info(
        "TRT joint engine installed from %s (merged %s, action %s).",
        engine_path,
        runner.input_shapes["video_tokens"],
        runner.input_shapes["action_tokens"],
    )
    return runner
