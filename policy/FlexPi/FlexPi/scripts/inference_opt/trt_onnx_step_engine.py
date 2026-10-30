"""VA2-style hand-carved TensorRT engine for the flex action denoise step.

torch-tensorrt's dynamo conversion fragments this graph (complex RoPE,
synthesized 0-dim aten.any from bool-mask SDPA decomposition). This script
does what the LingBot-VA2 paper actually did instead: export a clean ONNX of
the hot step and hand it to the raw TRT builder — explicit KV-cache I/O,
no mask (all-True by construction ≡ attn_backend="auto"), real-valued RoPE,
fp32→ no float64 (sinusoidal embedding patched to fp32).

Benchmarks eager / inductor(reduce-overhead) / raw TRT engine on the same
inputs and reports ms/step + maxdiff vs the unpatched eager reference.

Usage (any env with torch + tensorrt + onnx; intended for the trt29 env):
  python scripts/inference_opt/trt_onnx_step_engine.py --ckpt /path/to/step.pt
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for p in (PROJECT_ROOT, PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from omegaconf import OmegaConf  # noqa: E402
from hydra.utils import instantiate  # noqa: E402

import trt_fp8_step_bench as sb  # noqa: E402


def sinusoidal_embedding_1d_f32(dim, position):
    """fp32 clone of wan_video_dit.sinusoidal_embedding_1d (fp64 original —
    TRT has no fp64). At bf16 model precision the delta is far below noise."""
    sinusoid = torch.outer(
        position.type(torch.float32),
        torch.pow(
            10000,
            -torch.arange(dim // 2, dtype=torch.float32, device=position.device).div(dim // 2),
        ),
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


class ExportStep(torch.nn.Module):
    """Maskless realrope ActionStep with a tensors-only signature for export.

    The self/context masks are dropped (all-True by construction ≡ None) but
    the action-only mask is kept as a graph constant — its *presence* is the
    HBridge outer-layer routing flag, not just a visibility mask.
    """

    def __init__(self, model, action_only_mask):
        super().__init__()
        self.step = sb.ActionStep(model, realrope=True)
        self.register_buffer("action_only_mask", action_only_mask)

    def forward(self, latents_action, timestep_action, context,
                cache_k_stacked, cache_v_stacked, action_freqs_real):
        return self.step(
            latents_action, timestep_action, context, None,
            cache_k_stacked, cache_v_stacked, None, action_freqs_real,
            self.action_only_mask,
        )


def bench(fn, inputs, iters=200, warmup=20):
    with torch.no_grad():
        for _ in range(warmup):
            fn(*inputs)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            out = fn(*inputs)
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3, out


def build_engine(onnx_path: Path, engine_path: Path, opt_level: int, extra_flags=None):
    import tensorrt as trt

    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, trt_logger)
    # parse_from_file (not parse(buffer)): the exporter stores the 2 GB of
    # weights as external data next to the .onnx, which only resolves
    # relative to the model's own path.
    if not parser.parse_from_file(str(onnx_path)):
        errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("ONNX parse failed:\n" + "\n".join(errs))
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.BF16)
    # extra_flags: sub-fp16 precision modes (FP8/FP4) whose Q/DQ nodes ride in
    # the ONNX graph but need the matching builder flag for TRT to honor the
    # low-precision tactics (e.g. MXFP8 -> FP8, NVFP4 -> FP4).
    for flag in (extra_flags or ()):
        config.set_flag(flag)
    config.builder_optimization_level = opt_level
    t0 = time.perf_counter()
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise RuntimeError("TRT engine build failed")
    engine_path.write_bytes(blob)
    return time.perf_counter() - t0


class TrtRunner:
    """Minimal execute_async_v3 runner over torch-owned buffers."""

    def __init__(self, engine_path: Path, example_inputs, device):
        import tensorrt as trt

        self.trt = trt
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        self.ctx = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream(device=device)
        _trt2torch = {
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF: torch.float16,
            trt.DataType.BF16: torch.bfloat16,
            trt.DataType.INT64: torch.int64,
            trt.DataType.INT32: torch.int32,
            trt.DataType.BOOL: torch.bool,
        }
        self.in_names, self.out_names, self.buffers = [], [], {}
        n_inputs = 0
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.in_names.append(name)
                buf = example_inputs[n_inputs].contiguous()
                dt = _trt2torch[self.engine.get_tensor_dtype(name)]
                if buf.dtype != dt:
                    buf = buf.to(dt)
                self.buffers[name] = buf.clone()
                n_inputs += 1
            else:
                self.out_names.append(name)
                shape = tuple(self.ctx.get_tensor_shape(name))
                dt = _trt2torch[self.engine.get_tensor_dtype(name)]
                self.buffers[name] = torch.empty(shape, dtype=dt, device=device)
            self.ctx.set_tensor_address(name, self.buffers[name].data_ptr())

    def __call__(self, *inputs):
        for name, t in zip(self.in_names, inputs):
            buf = self.buffers[name]
            buf.copy_(t.to(buf.dtype), non_blocking=True)
        with torch.cuda.stream(self.stream):
            self.ctx.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return self.buffers[self.out_names[0]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--opt-level", type=int, default=3)
    ap.add_argument("--workdir", default="/tmp/flex_trt_onnx")
    ap.add_argument("--reuse-onnx", action="store_true",
                    help="skip torch.onnx.export if the .onnx already exists")
    ap.add_argument("--loop-bench", action="store_true",
                    help="also time 10 engine steps with torch-side Euler updates")
    ap.add_argument("--output", default=str(PROJECT_ROOT / "benchmark_results.jsonl"))
    args = ap.parse_args()

    device, dtype = "cuda", torch.bfloat16
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.ckpt).resolve()
    trained_cfg = OmegaConf.load(sb._find_trained_config(ckpt_path))
    model_cfg = OmegaConf.create(OmegaConf.to_container(trained_cfg.model, resolve=True))
    model_cfg.load_text_encoder = False
    model = instantiate(model_cfg, model_dtype=dtype, device=device)
    model.load_checkpoint(str(ckpt_path))
    model = model.to(device).eval()

    inputs = sb.build_step_inputs(model, device, dtype)
    results, notes = {}, {}

    # Reference: unpatched eager with masks + complex rope.
    ref_step = sb.ActionStep(model).eval()
    with torch.no_grad():
        ms, ref = bench(lambda *a: ref_step(*inputs.values()), (), args.iters)
    results["eager_masked_complexrope"] = round(ms, 3)
    print(f"[trt-onnx] eager ref (masked, complex rope): {ms:.3f} ms/step")

    # Patch: real rope + fp32 sinusoidal embedding; maskless flat signature.
    import flexpi.models.mot as mot_mod
    import flexpi.models.backbone as fw_mod
    mot_mod.rope_apply = sb.rope_apply_real
    fw_mod.sinusoidal_embedding_1d = sinusoidal_embedding_1d_f32

    step = ExportStep(model, inputs["action_only_attention_mask"]).eval()
    flat = (
        inputs["latents_action"], inputs["timestep_action"], inputs["context"],
        inputs["cache_k_stacked"], inputs["cache_v_stacked"], inputs["action_freqs_real"],
    )
    with torch.no_grad():
        ms, out = bench(lambda *a: step(*a), flat, args.iters)
    results["eager_export_graph"] = round(ms, 3)
    notes["export_graph_maxdiff"] = round(float((out - ref).abs().max()), 5)
    print(f"[trt-onnx] eager export graph (maskless, realrope, f32 emb): {ms:.3f} ms/step "
          f"maxdiff={notes['export_graph_maxdiff']}")

    torch._dynamo.reset()
    cstep = torch.compile(step, mode="reduce-overhead", fullgraph=False)
    with torch.no_grad():
        ms, out = bench(lambda *a: cstep(*a), flat, args.iters)
    results["inductor"] = round(ms, 3)
    print(f"[trt-onnx] inductor: {ms:.3f} ms/step")

    onnx_path = workdir / "flex_action_step.onnx"
    if args.reuse_onnx and onnx_path.exists():
        print(f"[trt-onnx] reusing existing {onnx_path}")
    else:
        with torch.no_grad():
            torch.onnx.export(
                step, flat, str(onnx_path), dynamo=True,
                input_names=["latents_action", "timestep_action", "context",
                             "cache_k", "cache_v", "freqs"],
                output_names=["pred_action"],
            )
    notes["onnx_mb"] = round(sum(
        f.stat().st_size for f in workdir.glob("flex_action_step.onnx*")
    ) / 1e6, 1)
    print(f"[trt-onnx] exported {onnx_path} ({notes['onnx_mb']} MB)")

    engine_path = workdir / f"flex_action_step_o{args.opt_level}.engine"
    build_s = build_engine(onnx_path, engine_path, args.opt_level)
    notes["engine_build_s"] = round(build_s, 1)
    print(f"[trt-onnx] engine built in {build_s:.0f}s -> {engine_path}")

    runner = TrtRunner(engine_path, flat, device)
    ms, out = bench(runner, flat, args.iters)
    results[f"trt_engine_o{args.opt_level}"] = round(ms, 3)
    notes["trt_maxdiff"] = round(float((out.to(ref.dtype) - ref).abs().max()), 5)
    print(f"[trt-onnx] raw TRT engine (opt {args.opt_level}): {ms:.3f} ms/step "
          f"maxdiff={notes['trt_maxdiff']}")

    if args.loop_bench:
        # Deployable-number check: 10 denoise steps through the engine with a
        # torch-side Euler update between steps — includes all interop
        # (input copies, stream sync, dtype casts) the VA2 topology pays.
        la0 = flat[0]
        rest = flat[1:]

        def loop_once():
            x = la0
            for _ in range(10):
                v = runner(x, *rest)
                x = x - 0.1 * v.to(x.dtype)
            return x

        ms, _ = bench(lambda: loop_once(), (), max(20, args.iters // 4))
        results["trt_engine_10step_loop"] = round(ms, 3)
        print(f"[trt-onnx] engine 10-step loop (incl. interop): {ms:.3f} ms")

    import tensorrt as trt
    row = {
        "tag": "trt_onnx_step_engine",
        "ckpt": str(ckpt_path),
        "step_ms": results,
        "per_call_estimate_ms_10steps": {k: round(v * 10, 1) for k, v in results.items()},
        "notes": notes,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "tensorrt": trt.__version__,
    }
    with open(args.output, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
