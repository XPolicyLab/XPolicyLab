"""Standalone inference-latency benchmark for FlexPiUnifiedJoint (flex).

Loads a flex checkpoint (architecture auto-loaded from the config.yaml saved
next to it), builds deploy-shaped dummy inputs (composite 384x320 + per-cam
RGB + per-cam depth + intrinsics + proprio + precomputed text context), and
times `infer_action` for one (regime, optimization) configuration.

One configuration per process: quantization mutates weights in-place and
torch.compile caches would otherwise leak across configs. To sweep, invoke it
once per configuration from a shell loop.

Appends one JSON line per run to --output (default: benchmark_results.jsonl).

Usage:
  python scripts/inference_opt/benchmark_flex_latency.py \
      --ckpt /path/to/step_XXXX.pt \
      --joint-video true --joint-dino true --joint-pointmap true \
      --torch-compile --quantization fp8 --num-inference-steps 10 \
      --iters 20 --output /tmp/bench.jsonl --tag joint_fp8
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# All model components (WAN, DINOv3) are loaded from local caches; skip HF
# HEAD requests so benchmark startup is network-independent.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from flexpi.datasets.lerobot.robot_video_dataset import _PER_CAM_HW  # noqa: E402


def _parse_opt_bool(v):
    if v is None or str(v).strip().lower() in ("", "none", "null"):
        return None
    return str(v).strip().lower() in ("1", "true", "yes")


def _find_trained_config(ckpt_path: Path) -> Path:
    for parent in list(ckpt_path.parents)[:4]:
        cand = parent / "config.yaml"
        if cand.exists():
            return cand
    raise FileNotFoundError(f"No config.yaml found next to {ckpt_path}")


def build_dummy_inputs(model, num_video_frames: int, action_horizon: int, device, dtype,
                       ctx_len: int = 128):
    g = torch.Generator(device="cpu").manual_seed(1234)

    def _rand(shape):
        return torch.rand(shape, generator=g).to(device=device, dtype=dtype)

    inputs = {
        "input_image": _rand((1, 3, 384, 320)),
        "per_cam": {name: _rand((1, 3, h, w)) for name, (h, w) in _PER_CAM_HW},
        "per_cam_depth": {
            name: (torch.rand((1, 1, h, w), generator=g) * 1.2 + 0.3).to(device=device, dtype=dtype)
            for name, (h, w) in _PER_CAM_HW
        },
        "proprio": torch.randn((1, model.proprio_dim), generator=g, dtype=torch.float32),
        # Synthetic T5 context (the real deploy caches encode_prompt per task,
        # so prompt encoding is not part of steady-state latency). ctx_len=128
        # is the historical bench default (all §3/§7 rows); the REAL deploy
        # pads to tokenizer_max_len=512 (+1 proprio token appended inside
        # infer) — pass --ctx-len 512 for deploy-faithful rows (TRT engine).
        "context": (torch.randn((1, ctx_len, 4096), generator=g) * 0.1).to(device=device, dtype=dtype),
        "context_mask": torch.ones((1, ctx_len), device=device, dtype=torch.bool),
    }
    # Plausible RoboTwin-ish intrinsics per camera (fx~fy~290 @ 240x320).
    K = torch.tensor([[289.6, 0.0, 160.0], [0.0, 289.6, 120.0], [0.0, 0.0, 1.0]])
    inputs["camera_intrinsics"] = K.unsqueeze(0).expand(len(_PER_CAM_HW), -1, -1).contiguous().to(device)
    return inputs


class PhaseTimer:
    """CUDA-event wrapper around bound methods; stream-ordered, no host sync."""

    def __init__(self, model, enabled: bool, loop_scope: bool = False):
        self.records = {}
        self._events = []
        if not enabled:
            return
        for attr in (
            "_encode_input_image_latents_tensor",
            "_encode_first_frame_pointmap_raw",
            "_predict_joint_noise_unified",
            "_predict_action_noise_with_cache",
            "_run_joint_denoise_loop",
            "_run_action_prefill_denoise_loop",
        ):
            if hasattr(model, attr):
                self._wrap(model, attr)
        if hasattr(model, "dino_encoder") and model.dino_encoder is not None:
            self._wrap(model.dino_encoder, "encode_video", label="dino_encode")
        # Under torch_compile_scope="loop" the prefill runs INSIDE the compiled
        # loop body; wrapping it would put a Python closure in the traced code
        # and force a recompile every call (measured 30 s/call). Skip it there.
        if hasattr(model, "mot") and not loop_scope:
            self._wrap(model.mot, "prefill_video_cache", label="video_prefill")

    def _wrap(self, obj, attr, label=None):
        fn = getattr(obj, attr)
        label = label or attr.lstrip("_")
        events = self._events
        records = self.records
        records.setdefault(label, [])

        def wrapped(*args, **kwargs):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            out = fn(*args, **kwargs)
            e.record()
            events.append((label, s, e))
            return out

        setattr(obj, attr, wrapped)

    def flush(self):
        # Call after torch.cuda.synchronize(); accumulates per-label ms.
        # Events created while dynamo traces a wrapped method (e.g. prefill
        # inside the loop-scope compiled body) never record — skip those.
        for label, s, e in self._events:
            try:
                self.records[label].append(s.elapsed_time(e))
            except RuntimeError:
                pass
        self._events.clear()

    def reset(self):
        self._events.clear()
        for k in self.records:
            self.records[k] = []

    def summary(self, n_calls: int):
        out = {}
        for label, vals in self.records.items():
            if vals:
                out[label] = round(sum(vals) / n_calls, 2)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--joint-video", default=None)
    ap.add_argument("--joint-dino", default=None)
    ap.add_argument("--joint-pointmap", default=None)
    ap.add_argument("--present-video", default=None)
    ap.add_argument("--present-dino", default=None)
    ap.add_argument("--present-pointmap", default=None)
    ap.add_argument("--num-inference-steps", type=int, default=10)
    ap.add_argument("--sigma-shift", type=float, default=None)
    ap.add_argument("--torch-compile", action="store_true")
    ap.add_argument("--torch-compile-mode", default="reduce-overhead")
    ap.add_argument("--torch-compile-scope", default="step", choices=["step", "loop"])
    ap.add_argument("--attn-backend", default="sdpa", choices=["sdpa", "flex", "auto"])
    ap.add_argument("--compile-encoders", action="store_true")
    ap.add_argument("--torch-compile-backend", default="inductor", choices=["inductor", "tensorrt"])
    ap.add_argument("--quantization", default=None)
    ap.add_argument("--trt-joint-engine", default=None,
                    help="serialized TRT joint-core engine "
                         "(scripts/inference_opt/trt_onnx_joint_engine.py); joint regimes only")
    ap.add_argument("--trt-joint-prefill-split-engine", default=None,
                    help="KV-split prefill engine (scripts/inference_opt/trt_onnx_joint_split_engine.py)")
    ap.add_argument("--trt-joint-decode-split-engine", default=None,
                    help="KV-split decode engine; pair with prefill-split, replaces --trt-joint-engine")
    ap.add_argument("--trt-joint-free-video-blocks", action="store_true",
                    help="free torch video-expert weights after the engine loads "
                         "(needed to fit the split engines in 32 GB)")
    ap.add_argument("--ctx-len", type=int, default=128,
                    help="dummy text-context tokens; default 128 matches this "
                         "family's trained tokenizer_max_len (the real deploy "
                         "length — check the run's config.yaml)")
    ap.add_argument("--solver", default="euler", choices=["euler", "dpmpp_2m"],
                    help="inference ODE solver; dpmpp_2m engages on the eager/"
                         "engine joint path (see prepare_for_inference)")
    ap.add_argument("--trt-prefill-engine", default=None,
                    help="serialized TRT prefill engine "
                         "(scripts/inference_opt/trt_onnx_prefill_engine.py); action regime")
    ap.add_argument("--glue-cache", action="store_true",
                    help="memoize per-call-constant glue (freqs/masks); "
                         "bit-identical hits, see prepare_for_inference")
    ap.add_argument("--encoder-cuda-graph", action="store_true",
                    help="CUDA-graph the 3 anchor encoders (single-launch "
                         "replay; capture-fail falls back eager)")
    ap.add_argument("--joint-loop-cuda-graph", action="store_true",
                    help="manual CUDA-graph the whole 10-step joint denoise "
                         "loop (Euler-only; removes per-step host launch cost "
                         "on the TRT-engine path; capture-fail falls back eager)")
    ap.add_argument("--return-stream-latents", default="true",
                    help="false skips the D2H copies of video/dino/pointmap "
                         "latents in the infer output (deploy reads only action)")
    ap.add_argument("--dynamic-step-skip", action="store_true")
    ap.add_argument("--step-skip-thresholds", default="[[0.95, 4], [0.93, 2]]")
    ap.add_argument("--warmup-iters", type=int, default=3)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--profile-phases", action="store_true")
    ap.add_argument("--output", default=str(PROJECT_ROOT / "benchmark_results.jsonl"))
    args = ap.parse_args()

    device = "cuda"
    dtype = torch.bfloat16
    ckpt_path = Path(args.ckpt).resolve()
    trained_cfg = OmegaConf.load(_find_trained_config(ckpt_path))

    model_cfg = OmegaConf.create(OmegaConf.to_container(trained_cfg.model, resolve=True))
    model_cfg.load_text_encoder = False
    # Neutralize the training-time DiT bootstrap loads, exactly as the deploy
    # factories do (experiments/{yam,robotwin}/flexpi_policy/deploy_policy.py).
    # ``load_checkpoint`` below is the source of truth for every DiT parameter,
    # so these only cost I/O -- and on an inference-only machine they are a hard
    # dependency on ~20 GB of Wan2.2 base shards that would otherwise be
    # downloaded (ModelScope ignores HF_HUB_OFFLINE) just to be overwritten.
    model_cfg.action_dit_pretrained_path = None
    model_cfg.skip_dit_load_from_pretrain = True

    t0 = time.perf_counter()
    model = instantiate(model_cfg, model_dtype=dtype, device=device)
    model.load_checkpoint(str(ckpt_path))
    model = model.to(device).eval()
    load_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    model.prepare_for_inference(
        torch_compile=args.torch_compile,
        torch_compile_mode=args.torch_compile_mode,
        quantization=args.quantization,
        torch_compile_scope=args.torch_compile_scope,
        attn_backend=args.attn_backend,
        torch_compile_backend=args.torch_compile_backend,
        compile_encoders=args.compile_encoders,
        trt_joint_engine_path=args.trt_joint_engine,
        trt_joint_free_video_blocks=args.trt_joint_free_video_blocks,
        trt_joint_prefill_split_engine_path=args.trt_joint_prefill_split_engine,
        trt_joint_decode_split_engine_path=args.trt_joint_decode_split_engine,
        trt_prefill_engine_path=args.trt_prefill_engine,
        solver=args.solver,
        glue_cache=args.glue_cache,
        encoder_cuda_graph=args.encoder_cuda_graph,
        joint_loop_cuda_graph=args.joint_loop_cuda_graph,
    )
    prepare_s = time.perf_counter() - t0

    num_frames = int(trained_cfg.data.train.num_frames)
    ratio = int(trained_cfg.data.train.action_video_freq_ratio)
    num_video_frames = (num_frames - 1) // ratio + 1
    action_horizon = num_frames - 1

    inputs = build_dummy_inputs(model, num_video_frames, action_horizon, device, dtype,
                                ctx_len=args.ctx_len)
    model.set_camera_intrinsics(inputs["camera_intrinsics"].to(device))

    infer_kwargs = dict(
        prompt=None,
        context=inputs["context"],
        context_mask=inputs["context_mask"],
        input_image=inputs["input_image"],
        per_cam=inputs["per_cam"],
        per_cam_depth=inputs["per_cam_depth"],
        proprio=inputs["proprio"],
        action_horizon=action_horizon,
        num_video_frames=num_video_frames,
        num_inference_steps=args.num_inference_steps,
        sigma_shift=args.sigma_shift,
        seed=0,
        rand_device="cpu",
        text_cfg_scale=1.0,
        dynamic_step_skip=args.dynamic_step_skip,
        step_skip_thresholds=[
            (float(t), int(n)) for t, n in json.loads(args.step_skip_thresholds)
        ],
        return_stream_latents=_parse_opt_bool(args.return_stream_latents),
    )
    for key, raw in (
        ("joint_video", args.joint_video),
        ("joint_dino", args.joint_dino),
        ("joint_pointmap", args.joint_pointmap),
        ("present_video", args.present_video),
        ("present_dino", args.present_dino),
        ("present_pointmap", args.present_pointmap),
    ):
        val = _parse_opt_bool(raw)
        if val is not None:
            infer_kwargs[key] = val

    timer = PhaseTimer(
        model,
        enabled=args.profile_phases,
        loop_scope=(args.torch_compile and args.torch_compile_scope == "loop"),
    )

    # Warmup (first call pays torch.compile + CUDA-Graph capture).
    t0 = time.perf_counter()
    with torch.no_grad():
        model.infer_action(**infer_kwargs)
    torch.cuda.synchronize()
    first_call_s = time.perf_counter() - t0
    with torch.no_grad():
        for _ in range(max(0, args.warmup_iters - 1)):
            model.infer_action(**infer_kwargs)
    torch.cuda.synchronize()
    timer.flush()
    timer.reset()

    torch.cuda.reset_peak_memory_stats()
    times, action_out = [], None
    with torch.no_grad():
        for _ in range(args.iters):
            t0 = time.perf_counter()
            out = model.infer_action(**infer_kwargs)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1e3)
            timer.flush()
            action_out = out["action"]

    times_sorted = sorted(times)
    n = len(times_sorted)
    result = {
        "tag": args.tag,
        "ckpt": str(ckpt_path),
        "regime": {
            "joint_video": _parse_opt_bool(args.joint_video),
            "joint_dino": _parse_opt_bool(args.joint_dino),
            "joint_pointmap": _parse_opt_bool(args.joint_pointmap),
        },
        "opts": {
            "torch_compile": args.torch_compile,
            "torch_compile_mode": args.torch_compile_mode if args.torch_compile else None,
            "torch_compile_scope": args.torch_compile_scope if args.torch_compile else None,
            "torch_compile_backend": args.torch_compile_backend if args.torch_compile else None,
            "attn_backend": args.attn_backend,
            "compile_encoders": args.compile_encoders,
            "quantization": args.quantization,
            "trt_joint_engine": args.trt_joint_engine,
            "trt_joint_split": bool(args.trt_joint_decode_split_engine),
            "ctx_len": args.ctx_len,
            "num_inference_steps": args.num_inference_steps,
            "sigma_shift": args.sigma_shift,
            "dynamic_step_skip": args.dynamic_step_skip,
            "glue_cache": args.glue_cache,
            "encoder_cuda_graph": args.encoder_cuda_graph,
            "joint_loop_cuda_graph": args.joint_loop_cuda_graph,
            "return_stream_latents": _parse_opt_bool(args.return_stream_latents),
        },
        "mean_ms": round(sum(times) / n, 1),
        "p50_ms": round(times_sorted[n // 2], 1),
        "min_ms": round(times_sorted[0], 1),
        "max_ms": round(times_sorted[-1], 1),
        "iters": n,
        "first_call_s": round(first_call_s, 1),
        "load_s": round(load_s, 1),
        "prepare_s": round(prepare_s, 1),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "phases_ms_per_call": timer.summary(n),
        # Fixed-seed action fingerprint: compare across configs to gauge
        # numerical drift (success-rate impact needs a real sim eval).
        "action_fingerprint": [round(float(v), 5) for v in action_out.flatten()[:8]],
        "action_absmean": round(float(action_out.abs().mean()), 5),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
    }
    with open(args.output, "a") as f:
        f.write(json.dumps(result) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
