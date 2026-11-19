"""Real-activation parity gate for the TRT joint engine (world env).

Two checks against the eager torch joint path on identical deploy-shaped
inputs (benchmark_flex_latency.build_dummy_inputs, seed 0):

1. core-level — capture the step-0 ``mot._forward_joint_inner`` inputs and
   outputs during an eager joint ``infer_action``, run the engine on the
   captured inputs, and report relative errors on both output streams;
2. end-to-end — re-run the same ``infer_action`` with the adapter installed
   (``prepare_for_inference(trt_joint_engine_path=...)``) and compare the
   final action chunk.

The random-input relcheck bounds precision noise adversarially; this is the
tamer real-activation number the SR sweep rides on.

Usage:
    python scripts/inference_opt/trt_joint_parity.py --ckpt /path/step.pt \
        --engine /path/flex_joint_core_o3.engine
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for p in (PROJECT_ROOT, PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from omegaconf import OmegaConf  # noqa: E402
from hydra.utils import instantiate  # noqa: E402

from benchmark_flex_latency import _find_trained_config, build_dummy_inputs  # noqa: E402
from flexpi.models.inference_opt.trt_joint import (  # noqa: E402
    _INPUT_NAMES,
    TrtJointRunner,
    _freqs_to_real_f32,
)


def _rel(name, out, ref):
    out, ref = out.float(), ref.float()
    d = (out - ref).abs()
    stats = {
        "maxdiff": round(float(d.max()), 5),
        "ref_absmax": round(float(ref.abs().max()), 3),
        "rel_max": round(float(d.max() / ref.abs().max().clamp_min(1e-8)), 6),
        "relL2": round(float(d.norm() / ref.norm().clamp_min(1e-8)), 6),
    }
    print(f"[trt-parity] {name}: {stats}")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--num-inference-steps", type=int, default=10)
    ap.add_argument("--ctx-len", type=int, default=128,
                    help="dummy text-context tokens; set to the TRAINED "
                         "tokenizer_max_len (this family: 128 — check the "
                         "run's config.yaml, NOT the flexpi.py default of "
                         "512). +1 proprio token is appended inside infer, "
                         "so the core sees ctx_len+1 — build the engine "
                         "with --ctx-len <tokenizer_max_len + 1>")
    ap.add_argument("--output", default=str(PROJECT_ROOT / "benchmark_results.jsonl"))
    args = ap.parse_args()

    device, dtype = "cuda", torch.bfloat16
    ckpt_path = Path(args.ckpt).resolve()
    trained_cfg = OmegaConf.load(_find_trained_config(ckpt_path))
    model_cfg = OmegaConf.create(OmegaConf.to_container(trained_cfg.model, resolve=True))
    model_cfg.load_text_encoder = False
    # `load_checkpoint` below is the source of truth for every DiT parameter, so
    # loading either pretrained set costs only I/O -- and would make an
    # inference-only machine a hard dependency on ~20 GB of Wan2.2 base shards
    # just to overwrite them. Same as benchmark_flex_latency.py.
    model_cfg.action_dit_pretrained_path = None
    model_cfg.skip_dit_load_from_pretrain = True
    model = instantiate(model_cfg, model_dtype=dtype, device=device)
    model.load_checkpoint(str(ckpt_path))
    model = model.to(device).eval()

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
        seed=0,
        rand_device="cpu",
        text_cfg_scale=1.0,
        joint_video=True,
        joint_dino=True,
        joint_pointmap=True,
    )

    # --- Run A: eager torch, spy on every joint-core call ---
    # Step 0 feeds the parity check; the full per-step capture is saved as the
    # calibration set (real activations across the denoise-timestep range) that
    # scripts/inference_opt/trt_onnx_joint_split_engine.py consumes via --calib.
    orig = model.mot._forward_joint_inner
    captured = {"steps": []}

    def _cpu(v):
        if torch.is_tensor(v):
            return v.detach().to("cpu", copy=True)
        if isinstance(v, dict):
            return {k: _cpu(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return type(v)(_cpu(x) for x in v)
        return v

    def spy(**kw):
        out = orig(**kw)
        captured["steps"].append({k: _cpu(v) for k, v in kw.items()})
        if "in" not in captured:
            captured["in"] = {
                k: (v.detach().clone() if torch.is_tensor(v) else v)
                for k, v in kw.items()
            }
            captured["out"] = tuple(o.detach().clone() for o in out)
        return out

    model.mot._forward_joint_inner = spy
    with torch.no_grad():
        ref_actions = model.infer_action(**infer_kwargs)["action"].clone()
    model.mot._forward_joint_inner = orig
    kw = captured["in"]
    calib_path = Path(args.engine).parent / "captured_core_io.pt"
    torch.save(
        {"steps": captured["steps"], "step0_out": [o.cpu() for o in captured["out"]]},
        calib_path,
    )
    print(f"[trt-parity] saved {len(captured['steps'])}-step core IO capture -> {calib_path}")
    print(
        f"[trt-parity] captured step-0 core: merged {tuple(kw['video_tokens'].shape)} "
        f"action {tuple(kw['action_tokens'].shape)} "
        f"ctx {tuple(kw['video_context_payload']['context'].shape)}"
    )

    # --- core-level: engine on the captured step-0 inputs ---
    runner = TrtJointRunner(args.engine, device)
    flat = dict(zip(_INPUT_NAMES, [None] * len(_INPUT_NAMES)))
    flat["video_tokens"] = kw["video_tokens"]
    flat["action_tokens"] = kw["action_tokens"]
    flat["video_freqs_real"] = _freqs_to_real_f32(kw["video_freqs"])
    flat["action_freqs_real"] = _freqs_to_real_f32(kw["action_freqs"])
    flat["video_t_mod"] = kw["video_t_mod"]
    flat["action_t_mod"] = kw["action_t_mod"]
    flat["video_context"] = kw["video_context_payload"]["context"]
    flat["video_context_mask"] = kw["video_context_payload"]["mask"]
    flat["action_context"] = kw["action_context_payload"]["context"]
    flat["action_context_mask"] = kw["action_context_payload"]["mask"]
    flat["attention_mask"] = kw["attention_mask"]
    for name in _INPUT_NAMES:
        got = tuple(flat[name].shape)
        want = runner.input_shapes[name]
        if got != want:
            raise SystemExit(
                f"[trt-parity] FATAL shape mismatch {name}: model {got} vs "
                f"engine {want} — rebuild the engine at deploy shapes "
                f"(--ctx-len for the context length)."
            )
    trt_merged, trt_action = runner(*[flat[n] for n in _INPUT_NAMES])
    torch.cuda.synchronize()
    notes = {
        "core_merged": _rel("core step0 [merged]", trt_merged, captured["out"][0]),
        "core_action": _rel("core step0 [action]", trt_action, captured["out"][1]),
    }
    del runner
    torch.cuda.empty_cache()

    # --- Run B: end-to-end with the adapter installed ---
    model.prepare_for_inference(trt_joint_engine_path=args.engine)
    with torch.no_grad():
        t0 = time.perf_counter()
        trt_actions = model.infer_action(**infer_kwargs)["action"].clone()
        torch.cuda.synchronize()
        first_s = time.perf_counter() - t0
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            model.infer_action(**infer_kwargs)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1e3)
    notes["e2e_action"] = _rel("e2e final actions", trt_actions, ref_actions)
    notes["e2e_action"]["torch_absmean"] = round(float(ref_actions.float().abs().mean()), 5)
    notes["e2e_action"]["trt_absmean"] = round(float(trt_actions.float().abs().mean()), 5)
    print(
        f"[trt-parity] e2e adapter call: first {first_s:.1f}s, then "
        f"{sum(times) / len(times):.1f} ms/call (eager pre/post + engine core)"
    )

    row = {
        "tag": "trt_joint_parity",
        "ckpt": str(ckpt_path),
        "engine": str(args.engine),
        "adapter_call_ms": round(sum(times) / len(times), 1),
        "notes": notes,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
    }
    with open(args.output, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
