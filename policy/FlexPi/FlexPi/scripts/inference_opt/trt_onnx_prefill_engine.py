"""VA2-style raw TensorRT engine for the action-only KV-cache PREFILL.

The action fast path's biggest phase after denoise is the one-shot 30-layer
video-expert prefill over the ~534 anchor tokens (~44 ms eager / ~25 ms
in-graph). Same recipe as trt_onnx_joint_engine.py, different export unit:
``mot.prefill_video_cache`` (anchor tokens in -> per-layer K/V out).

Inputs are CAPTURED from a real action-only inference (spy on
prefill_video_cache) so the deploy shapes/masks are exact — no guessing (the
joint-core ctx saga cost three rebuilds). Real-valued RoPE for export; the
K/V ride out as two stacked tensors [num_layers, B, Sv, H*Dh].

Run in the trt29 env. Usage:
    python scripts/inference_opt/trt_onnx_prefill_engine.py --ckpt /path/step.pt
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
from trt_onnx_step_engine import TrtRunner, bench, build_engine  # noqa: E402
from benchmark_flex_latency import build_dummy_inputs  # noqa: E402


class PrefillCore(torch.nn.Module):
    """Flat-tensor wrapper around mot.prefill_video_cache.

    Context payload rebuilt inside; HBridge sub-stream metadata captured as
    constants (lens = python ints, masks = buffers). Emits the per-layer K/V
    stacked into two tensors so the ONNX graph has a fixed output arity.
    """

    def __init__(self, model, sub_lens, sub_masks):
        super().__init__()
        self.mot = model.mot
        self.sub_lens = sub_lens
        if sub_masks is not None:
            for i, m in enumerate(sub_masks):
                self.register_buffer(f"sub_mask_{i}", m)
            self.num_sub_masks = len(sub_masks)
        else:
            self.num_sub_masks = 0

    def forward(self, merged_tokens, merged_freqs_real, merged_t_mod,
                video_context, merged_ctx_mask, prefill_attention_mask):
        sub_masks = (
            [getattr(self, f"sub_mask_{i}") for i in range(self.num_sub_masks)]
            if self.num_sub_masks else None
        )
        cache = self.mot.prefill_video_cache(
            video_tokens=merged_tokens,
            video_freqs=merged_freqs_real,
            video_t_mod=merged_t_mod,
            video_context_payload={"context": video_context, "mask": merged_ctx_mask},
            video_attention_mask=prefill_attention_mask,
            video_sub_stream_lens=self.sub_lens,
            video_sub_stream_self_masks=sub_masks,
        )
        k_all = torch.stack([c["k"] for c in cache], dim=0)
        v_all = torch.stack([c["v"] for c in cache], dim=0)
        return k_all, v_all


def capture_prefill_io(model, trained_cfg, device, dtype):
    """Run one real action-only inference, spy on prefill_video_cache, return
    the exact deploy-shaped inputs + reference K/V + sub-stream metadata."""
    num_frames = int(trained_cfg.data.train.num_frames)
    ratio = int(trained_cfg.data.train.action_video_freq_ratio)
    num_video_frames = (num_frames - 1) // ratio + 1
    action_horizon = num_frames - 1
    inputs = build_dummy_inputs(model, num_video_frames, action_horizon, device, dtype)
    model.set_camera_intrinsics(inputs["camera_intrinsics"].to(device))

    orig = model.mot.prefill_video_cache
    cap = {}

    def spy(video_tokens, video_freqs, video_t_mod, video_context_payload,
            video_attention_mask, video_sub_stream_lens=None,
            video_sub_stream_self_masks=None):
        out = orig(
            video_tokens=video_tokens, video_freqs=video_freqs,
            video_t_mod=video_t_mod, video_context_payload=video_context_payload,
            video_attention_mask=video_attention_mask,
            video_sub_stream_lens=video_sub_stream_lens,
            video_sub_stream_self_masks=video_sub_stream_self_masks,
        )
        if "tokens" not in cap:
            Sv = video_tokens.shape[1]
            mask = video_attention_mask
            if mask is None:
                mask = torch.ones(Sv, Sv, dtype=torch.bool, device=video_tokens.device)
            cap["tokens"] = video_tokens.detach().clone()
            cap["freqs"] = video_freqs.detach().clone()
            cap["t_mod"] = video_t_mod.detach().clone()
            cap["context"] = video_context_payload["context"].detach().clone()
            cap["ctx_mask"] = video_context_payload["mask"].detach().clone()
            cap["mask"] = mask.detach().clone()
            cap["sub_lens"] = video_sub_stream_lens
            cap["sub_masks"] = (
                [m.detach().clone() for m in video_sub_stream_self_masks]
                if video_sub_stream_self_masks is not None else None
            )
            cap["ref_k"] = torch.stack([c["k"] for c in out], dim=0).detach().clone()
            cap["ref_v"] = torch.stack([c["v"] for c in out], dim=0).detach().clone()
        return out

    model.mot.prefill_video_cache = spy
    with torch.no_grad():
        model.infer_action(
            prompt=None, context=inputs["context"], context_mask=inputs["context_mask"],
            input_image=inputs["input_image"], per_cam=inputs["per_cam"],
            per_cam_depth=inputs["per_cam_depth"], proprio=inputs["proprio"],
            action_horizon=action_horizon, num_video_frames=num_video_frames,
            num_inference_steps=10, seed=0, rand_device="cpu", text_cfg_scale=1.0,
            joint_video=False, joint_dino=False, joint_pointmap=False,
        )
    model.mot.prefill_video_cache = orig
    if "tokens" not in cap:
        raise RuntimeError("prefill_video_cache was not called (wrong regime?).")
    return cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--opt-level", type=int, default=3)
    ap.add_argument("--workdir", default="/tmp/flex_trt_onnx_prefill")
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

    cap = capture_prefill_io(model, trained_cfg, device, dtype)
    Sv = cap["tokens"].shape[1]
    print(f"[trt-prefill] captured: merged {tuple(cap['tokens'].shape)} "
          f"ctx {tuple(cap['context'].shape)} mask {tuple(cap['mask'].shape)} Sv={Sv}")

    freqs_real = torch.view_as_real(cap["freqs"].contiguous()).to(torch.float32).contiguous()
    flat = (
        cap["tokens"], freqs_real, cap["t_mod"],
        cap["context"], cap["ctx_mask"], cap["mask"],
    )
    results, notes = {}, {"Sv": Sv}

    # Reference (complex rope) then patch to real rope for export/bench.
    core_ref = PrefillCore(model, cap["sub_lens"], cap["sub_masks"]).eval()
    complex_flat = (cap["tokens"], cap["freqs"], cap["t_mod"],
                    cap["context"], cap["ctx_mask"], cap["mask"])
    with torch.no_grad():
        ms, ref = bench(lambda *a: core_ref(*complex_flat), (), args.iters)
    results["eager_complexrope"] = round(ms, 3)
    ref_k = ref[0]
    print(f"[trt-prefill] eager (complex rope): {ms:.3f} ms")

    import flexpi.models.mot as mot_mod
    mot_mod.rope_apply = sb.rope_apply_real
    core = PrefillCore(model, cap["sub_lens"], cap["sub_masks"]).eval()
    with torch.no_grad():
        ms, out = bench(lambda *a: core(*a), flat, args.iters)
    results["eager_realrope"] = round(ms, 3)
    notes["realrope_k_maxdiff"] = round(float((out[0] - ref_k).abs().max()), 5)
    print(f"[trt-prefill] eager realrope: {ms:.3f} ms  k_maxdiff={notes['realrope_k_maxdiff']}")

    torch._dynamo.reset()
    ccore = torch.compile(core, mode="reduce-overhead", fullgraph=False)
    with torch.no_grad():
        ms, _ = bench(lambda *a: ccore(*a), flat, args.iters)
    results["inductor"] = round(ms, 3)
    print(f"[trt-prefill] inductor: {ms:.3f} ms")

    onnx_path = workdir / "flex_prefill.onnx"
    with torch.no_grad():
        torch.onnx.export(
            core, flat, str(onnx_path), dynamo=True,
            input_names=["merged_tokens", "merged_freqs_real", "merged_t_mod",
                         "video_context", "merged_ctx_mask", "prefill_attention_mask"],
            output_names=["k_all", "v_all"],
        )
    notes["onnx_mb"] = round(
        sum(f.stat().st_size for f in workdir.glob("flex_prefill.onnx*")) / 1e6, 1)
    print(f"[trt-prefill] exported {onnx_path} ({notes['onnx_mb']} MB)")

    engine_path = workdir / f"flex_prefill_o{args.opt_level}.engine"
    build_s = build_engine(onnx_path, engine_path, args.opt_level)
    notes["engine_build_s"] = round(build_s, 1)
    print(f"[trt-prefill] engine built in {build_s:.0f}s")

    runner = TrtRunner(engine_path, flat, device)
    ms, out = bench(runner, flat, args.iters)
    results[f"trt_engine_o{args.opt_level}"] = round(ms, 3)
    with torch.no_grad():
        eager_k = core(*flat)[0]
    notes["trt_k_maxdiff"] = round(float((out.to(eager_k.dtype) - eager_k).abs().max()), 5)
    notes["trt_k_relL2"] = round(
        float((out.to(eager_k.dtype) - eager_k).float().norm() / eager_k.float().norm()), 6)
    print(f"[trt-prefill] raw TRT engine (opt {args.opt_level}): {ms:.3f} ms  "
          f"k_maxdiff={notes['trt_k_maxdiff']} k_relL2={notes['trt_k_relL2']}")

    import tensorrt as trt
    row = {
        "tag": "trt_onnx_prefill",
        "ckpt": str(ckpt_path),
        "ms": results,
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
