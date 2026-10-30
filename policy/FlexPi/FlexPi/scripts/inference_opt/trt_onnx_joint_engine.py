"""VA2-style raw TensorRT engine for the flex JOINT denoise core.

The joint regime is where TRT kernels actually win on this GPU (raw GEMM
stack at M=1634: TRT fp16 1.36 vs inductor 2.03 ms — §5 of the latency
report), but the full step never survived torch-tensorrt conversion. Same
recipe as trt_onnx_step_engine.py, bigger export unit: the 30-layer
``mot._forward_joint_inner`` core (merged-stream tokens in → tokens out),
with pre/post_dit, embeddings and scheduler left in torch.

- real-valued RoPE (complex is not exportable) via the rope_apply patch;
- the dense joint attention mask is NON-trivial and rides in as a bool
  input (unlike the action step it cannot be dropped);
- HBridge sub-stream masks ride along as constants when enabled.

Benchmarks eager / inductor / raw TRT engine per joint step and reports the
10-step denoise-core estimate. Run in the trt29 env (torch 2.12 exporter).

Usage: python scripts/inference_opt/trt_onnx_joint_engine.py --ckpt /path/to/step.pt
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
from trt_onnx_step_engine import (  # noqa: E402
    TrtRunner,
    bench,
    build_engine,
    sinusoidal_embedding_1d_f32,
)


class JointCore(torch.nn.Module):
    """Flat-tensor wrapper around mot._forward_joint_inner.

    Context payload dicts are rebuilt inside; HBridge sub-stream metadata is
    captured as constants (lens are python ints; masks become buffers).
    """

    def __init__(self, model, sub_stream_lens, sub_stream_self_masks):
        super().__init__()
        self.mot = model.mot
        self.sub_stream_lens = sub_stream_lens
        if sub_stream_self_masks is not None:
            for i, m in enumerate(sub_stream_self_masks):
                self.register_buffer(f"sub_mask_{i}", m)
            self.num_sub_masks = len(sub_stream_self_masks)
        else:
            self.num_sub_masks = 0

    def forward(
        self,
        video_tokens, action_tokens,
        video_freqs_real, action_freqs_real,
        video_t_mod, action_t_mod,
        video_context, video_context_mask,
        action_context, action_context_mask,
        attention_mask,
    ):
        sub_masks = (
            [getattr(self, f"sub_mask_{i}") for i in range(self.num_sub_masks)]
            if self.num_sub_masks else None
        )
        merged_out, action_out = self.mot._forward_joint_inner(
            video_tokens=video_tokens,
            action_tokens=action_tokens,
            video_freqs=video_freqs_real,
            action_freqs=action_freqs_real,
            video_t_mod=video_t_mod,
            action_t_mod=action_t_mod,
            video_context_payload={"context": video_context, "mask": video_context_mask},
            action_context_payload={"context": action_context, "mask": action_context_mask},
            attention_mask=attention_mask,
            sub_stream_lens=self.sub_stream_lens,
            sub_stream_self_masks=sub_masks,
        )
        return merged_out, action_out


def build_joint_inputs(model, device, dtype, ctx_len=128):
    """Deploy-shaped joint-core inputs (§2 of the latency report): merged
    video stream = video 360 + dino + pointmap 360, action 32. The DINO
    stream length follows ``model.dino_cam_patches`` (robotwin 294/frame
    → 882; d7+pu2 folding 147/frame → 441). ``ctx_len`` must match deploy:
    trained tokenizer_max_len + 1 proprio token (this family: 128 + 1 = 129)."""
    g = torch.Generator(device="cpu").manual_seed(11)
    B = 1
    Sv, Sp, Sa = 360, 360, 32
    tpf, ptpf = 120, 120
    dtpf = sum(h * w for h, w in model.dino_cam_patches)
    Sd = dtpf * (Sv // tpf)
    S_merged = Sv + Sd + Sp
    D = model.video_expert.hidden_dim
    Da = model.action_expert.hidden_dim
    # freqs broadcast over heads: [S, 1, head_dim/2] (NOT heads*head_dim/2)
    rope_dim = model.video_expert.attn_head_dim // 2

    def r(*shape, scale=1.0, dt=dtype):
        return (torch.randn(*shape, generator=g) * scale).to(device=device, dtype=dt)
    mask = model._build_mot_attention_mask_unified(
        Sv, Sa, Sd, Sp, tpf, dtpf, ptpf, torch.device(device),
    )
    # merged-stream layout is [video | dino | pointmap], action separate; the
    # joint mask above is already in that layout (total = S_merged + Sa).
    video_freqs = torch.polar(
        torch.ones(S_merged, 1, rope_dim), r(S_merged, 1, rope_dim, dt=torch.float32).cpu().float()
    ).to(device)
    action_freqs = model.action_expert.freqs[:Sa].view(Sa, 1, -1).to(device)
    inputs = dict(
        video_tokens=r(B, S_merged, D, scale=0.5),
        action_tokens=r(B, Sa, Da, scale=0.5),
        video_freqs_real=torch.view_as_real(video_freqs).to(torch.float32).contiguous(),
        action_freqs_real=torch.view_as_real(action_freqs).to(torch.float32).contiguous(),
        # _split_modulation: 4-D = per-token [B, S, 6, D] (chunk dim 2);
        # 3-D = shared [B, 6, D] (chunk dim 1). Merged stream is per-token
        # (dino/pointmap anchors carry timestep 0), action is shared.
        video_t_mod=r(B, S_merged, 6, D, scale=0.2),
        action_t_mod=r(B, 6, Da, scale=0.2),
        video_context=r(B, ctx_len, D, scale=0.1),
        video_context_mask=torch.ones(B, S_merged, ctx_len, dtype=torch.bool, device=device),
        action_context=r(B, ctx_len, Da, scale=0.1),
        action_context_mask=torch.ones(B, Sa, ctx_len, dtype=torch.bool, device=device),
        attention_mask=mask,
    )
    return inputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--opt-level", type=int, default=3)
    ap.add_argument("--ctx-len", type=int, default=128,
                    help="text-context length baked into the engine = trained "
                         "tokenizer_max_len + 1 proprio token (this family "
                         "trains at 128 -> deploy engine needs 129)")
    ap.add_argument("--workdir", default="/tmp/flex_trt_onnx_joint")
    ap.add_argument("--reuse-onnx", action="store_true")
    ap.add_argument("--output", default=str(PROJECT_ROOT / "benchmark_results.jsonl"))
    args = ap.parse_args()

    device, dtype = "cuda", torch.bfloat16
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.ckpt).resolve()
    trained_cfg = OmegaConf.load(sb._find_trained_config(ckpt_path))
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

    inputs = build_joint_inputs(model, device, dtype, ctx_len=args.ctx_len)
    Sv, Sp, Sa = 360, 360, 32
    dtpf = sum(h * w for h, w in model.dino_cam_patches)
    Sd = dtpf * 3
    sub_lens, sub_masks = model._build_hbridge_self_masks_unified(
        Sv, Sa, Sd, Sp, 120, dtpf, 120, torch.device(device),
    )

    results, notes = {}, {"ctx_len": args.ctx_len}

    # Reference with complex rope (unpatched).
    core_ref = JointCore(model, sub_lens, sub_masks).eval()
    complex_inputs = dict(inputs)
    complex_inputs["video_freqs_real"] = torch.view_as_complex(
        inputs["video_freqs_real"].to(torch.float64).contiguous()
    )
    complex_inputs["action_freqs_real"] = torch.view_as_complex(
        inputs["action_freqs_real"].to(torch.float64).contiguous()
    )
    with torch.no_grad():
        ms, ref = bench(lambda *a: core_ref(*complex_inputs.values()), (), args.iters)
    ref = ref[1]  # compare on the action stream output
    results["eager_complexrope"] = round(ms, 3)
    print(f"[trt-joint] eager (complex rope): {ms:.3f} ms/step")

    # Patch to real rope + f32 embedding for export.
    import flexpi.models.mot as mot_mod
    import flexpi.models.backbone as fw_mod
    mot_mod.rope_apply = sb.rope_apply_real
    fw_mod.sinusoidal_embedding_1d = sinusoidal_embedding_1d_f32

    core = JointCore(model, sub_lens, sub_masks).eval()
    flat = tuple(inputs.values())
    with torch.no_grad():
        ms, out = bench(lambda *a: core(*a), flat, args.iters)
    results["eager_realrope"] = round(ms, 3)
    notes["export_graph_maxdiff"] = round(float((out[1] - ref).abs().max()), 5)
    print(f"[trt-joint] eager realrope: {ms:.3f} ms/step "
          f"maxdiff={notes['export_graph_maxdiff']}")

    torch._dynamo.reset()
    ccore = torch.compile(core, mode="reduce-overhead", fullgraph=False)
    with torch.no_grad():
        ms, _ = bench(lambda *a: ccore(*a), flat, args.iters)
    results["inductor"] = round(ms, 3)
    print(f"[trt-joint] inductor: {ms:.3f} ms/step")

    onnx_path = workdir / "flex_joint_core.onnx"
    if args.reuse_onnx and onnx_path.exists():
        print(f"[trt-joint] reusing {onnx_path}")
    else:
        with torch.no_grad():
            torch.onnx.export(
                core, flat, str(onnx_path), dynamo=True,
                input_names=list(inputs.keys()),
                output_names=["merged_out", "action_out"],
            )
    total_mb = sum(f.stat().st_size for f in workdir.glob("flex_joint_core.onnx*")) / 1e6
    notes["onnx_mb"] = round(total_mb, 1)
    print(f"[trt-joint] exported {onnx_path} ({notes['onnx_mb']} MB)")

    engine_path = workdir / f"flex_joint_core_o{args.opt_level}.engine"
    build_s = build_engine(onnx_path, engine_path, args.opt_level)
    notes["engine_build_s"] = round(build_s, 1)
    print(f"[trt-joint] engine built in {build_s:.0f}s")

    runner = TrtRunner(engine_path, flat, device)
    ms, out = bench(runner, flat, args.iters)
    results[f"trt_engine_o{args.opt_level}"] = round(ms, 3)
    # TrtRunner returns the first output; action_out is second — compare
    # merged_out against the realrope eager instead for the numeric note.
    with torch.no_grad():
        eager_merged = core(*flat)[0]
    notes["trt_merged_maxdiff"] = round(
        float((out.to(eager_merged.dtype) - eager_merged).abs().max()), 5
    )
    print(f"[trt-joint] raw TRT engine (opt {args.opt_level}): {ms:.3f} ms/step "
          f"merged_maxdiff={notes['trt_merged_maxdiff']}")

    import tensorrt as trt
    row = {
        "tag": "trt_onnx_joint_core",
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
