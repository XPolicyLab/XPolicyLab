"""TensorRT deep-dive (option b): ModelOpt FP8 PTQ + torch-tensorrt lowering
of the flex action denoise step, benchmarked against eager / inductor / TRT-bf16.

The action denoise step (`FlexPiBackbone._denoise_step_compiled` body) is the
cleanest TRT unit: fixed shapes, pure tensor I/O, called 10x per infer. RoPE
uses complex tensors which TRT cannot lower — torch-tensorrt partitions the
graph (linears/FFN/attention → TRT engines, complex ops stay in torch), so
this measures the realistic partitioned speedup, mirroring VA2's
"FP8 TensorRT compiled execution" on our architecture.

Usage:
  python scripts/inference_opt/trt_fp8_step_bench.py --ckpt /path/to/step_XXXX.pt \
      [--variants eager,inductor,trt_bf16,trt_fp8] [--iters 50]

Appends rows tagged trt_step_* to --output (default benchmark_results.jsonl).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate

os.environ.setdefault("HF_HUB_OFFLINE", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from flexpi.models.backbone import FlexPiBackbone  # noqa: E402


def _find_trained_config(ckpt_path: Path) -> Path:
    for parent in list(ckpt_path.parents)[:4]:
        cand = parent / "config.yaml"
        if cand.exists():
            return cand
    raise FileNotFoundError(f"No config.yaml found next to {ckpt_path}")


class ActionStep(torch.nn.Module):
    """Flat-tensor wrapper around the action denoise step for export/quant.

    K/V caches are stacked to [L, B, S, D] single tensors (TRT-friendly);
    unbound back to per-layer lists inside forward.
    """

    def __init__(self, model, realrope: bool = False):
        super().__init__()
        self.model = model  # registered submodule so quantization sees Linears
        self.realrope = realrope

    def forward(
        self,
        latents_action,
        timestep_action,
        context,
        context_mask,
        cache_k_stacked,
        cache_v_stacked,
        action_attention_mask,
        action_freqs_real,
        action_only_attention_mask,
    ):
        if self.realrope:
            # rope_apply is patched to rope_apply_real: pass the packed real
            # (cos,sin) tensor straight through — no complex dtype anywhere.
            action_freqs = action_freqs_real
        else:
            # torch-tensorrt rejects complex dtypes at the graph boundary; RoPE
            # freqs cross as view_as_real float64->float32 pairs and are rebuilt
            # here (the complex ops themselves get partitioned to torch fallback).
            action_freqs = torch.view_as_complex(
                action_freqs_real.to(torch.float64).contiguous()
            )
        cache_k = list(cache_k_stacked.unbind(0))
        cache_v = list(cache_v_stacked.unbind(0))
        return FlexPiBackbone._denoise_step_compiled(
            self.model,
            latents_action=latents_action,
            timestep_action=timestep_action,
            context=context,
            context_mask=context_mask,
            video_cache_k=cache_k,
            video_cache_v=cache_v,
            action_attention_mask=action_attention_mask,
            action_freqs=action_freqs,
            action_only_attention_mask=action_only_attention_mask,
        )


def build_step_inputs(model, device, dtype):
    """Deploy-shaped step inputs. Cache/mask shapes follow the action-only
    prefill layout: ff_v 120 + ff_d 294 + ff_p 120 = 534 video-seq tokens."""
    g = torch.Generator(device="cpu").manual_seed(7)
    B, Sa, Sv = 1, 32, 534
    L = len(model.mot.mixtures["video"].blocks)
    attn_dim = model.video_expert.num_heads * model.video_expert.attn_head_dim

    def r(*shape, scale=1.0):
        return (torch.randn(*shape, generator=g) * scale).to(device=device, dtype=dtype)

    total = Sv + Sa
    mask = torch.zeros((total, total), dtype=torch.bool, device=device)
    mask[:Sv, :Sv] = True
    mask[Sv:, :] = True
    action_mask = mask[Sv:total, :total]
    action_only_mask = model._build_action_only_attention_mask(Sa, device)
    freqs = model.action_expert.freqs[:Sa].view(Sa, 1, -1).to(device)
    # Complex -> interleaved-real float32 for the TRT-safe graph boundary.
    freqs_real = torch.view_as_real(freqs).to(torch.float32).contiguous()
    return dict(
        latents_action=r(B, Sa, model.action_expert.action_dim),
        timestep_action=torch.full((1,), 500.0, device=device, dtype=dtype),
        context=r(B, 128, 4096, scale=0.1),
        context_mask=torch.ones((B, 128), device=device, dtype=torch.bool),
        cache_k_stacked=r(L, B, Sv, attn_dim, scale=0.5),
        cache_v_stacked=r(L, B, Sv, attn_dim, scale=0.5),
        action_attention_mask=action_mask,
        action_freqs_real=freqs_real,
        action_only_attention_mask=action_only_mask,
    )


# Real-valued RoPE experiment: (a+bi)(c+di) expanded so no complex dtype ever
# enters the graph — TRT can't ingest complex, inductor can't codegen it
# (falls back to eager islands), and torch.export/AOTInductor reject it.
# cos/sin arrive packed AS the freqs argument ([S, 1, hd/2, 2] float32 =
# view_as_real of the complex freqs), which flows untouched through the step
# into rope_apply — no globals, export-clean.


def rope_apply_real(x, freqs, num_heads):
    B, S, D = x.shape
    xr = x.view(B, S, num_heads, -1).to(torch.float32).view(B, S, num_heads, -1, 2)
    a, b = xr[..., 0], xr[..., 1]
    c = freqs[..., 0].view(1, S, 1, -1)
    s = freqs[..., 1].view(1, S, 1, -1)
    out = torch.stack((a * c - b * s, a * s + b * c), dim=-1).flatten(3)
    return out.flatten(2).to(x.dtype)


def bench(fn, inputs, iters, warmup=10):
    with torch.no_grad():
        for _ in range(warmup):
            fn(**inputs)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            out = fn(**inputs)
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--variants", default="eager,inductor,trt_bf16,trt_fp8")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--output", default=str(PROJECT_ROOT / "benchmark_results.jsonl"))
    ap.add_argument(
        "--maskless", action="store_true",
        help="Pass None for the (all-True by construction) attention masks — "
        "matches attn_backend='auto' and removes bool-mask SDPA decomposition "
        "from the exported graph (suspected source of the 0-dim aten.any that "
        "poisons torch-tensorrt subgraph conversion).",
    )
    args = ap.parse_args()

    device, dtype = "cuda", torch.bfloat16
    ckpt_path = Path(args.ckpt).resolve()
    trained_cfg = OmegaConf.load(_find_trained_config(ckpt_path))
    model_cfg = OmegaConf.create(OmegaConf.to_container(trained_cfg.model, resolve=True))
    model_cfg.load_text_encoder = False
    model = instantiate(model_cfg, model_dtype=dtype, device=device)
    model.load_checkpoint(str(ckpt_path))
    model = model.to(device).eval()

    inputs = build_step_inputs(model, device, dtype)
    if args.maskless:
        # action_only_attention_mask is kept: its presence is the HBridge
        # outer-layer routing flag, not just a visibility mask.
        inputs["action_attention_mask"] = None
        inputs["context_mask"] = None
    step = ActionStep(model).eval()
    results = {}
    ref_out = None

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    with torch.no_grad():
        if "eager" in variants:
            ms, ref_out = bench(step.forward, inputs, args.iters)
            results["eager"] = ms
            print(f"[trt-bench] eager: {ms:.3f} ms/step")

        if "inductor" in variants:
            torch._dynamo.reset()
            cstep = torch.compile(step, mode="reduce-overhead", fullgraph=False)
            ms, out = bench(cstep.forward, inputs, args.iters)
            results["inductor"] = ms
            print(f"[trt-bench] inductor(reduce-overhead): {ms:.3f} ms/step "
                  f"maxdiff={float((out - ref_out).abs().max()) if ref_out is not None else -1:.5f}")

        if "trt_bf16" in variants:
            import torch_tensorrt
            torch._dynamo.reset()
            tstep = torch.compile(
                step,
                backend="tensorrt",
                options={
                    "enabled_precisions": {torch.float32, torch.float16, torch.bfloat16},
                    "min_block_size": 1,
                    "truncate_double": True,
                },
                fullgraph=False,
            )
            ms, out = bench(tstep.forward, inputs, args.iters)
            results["trt_bf16"] = ms
            print(f"[trt-bench] tensorrt bf16: {ms:.3f} ms/step "
                  f"maxdiff={float((out - ref_out).abs().max()) if ref_out is not None else -1:.5f}")

        if "inductor_realrope" in variants or "trt_bf16_realrope" in variants:
            # Swap MoT's rope_apply for the complex-free version; the packed
            # real (cos,sin) freqs pass through the realrope wrapper as-is.
            # Small float32-vs-float64 numeric delta.
            import flexpi.models.mot as mot_mod
            step = ActionStep(model, realrope=True).eval()
            _orig_rope = mot_mod.rope_apply
            mot_mod.rope_apply = rope_apply_real
            try:
                ms, out = bench(step.forward, inputs, args.iters)
                print(f"[trt-bench] eager realrope: {ms:.3f} ms/step "
                      f"maxdiff={float((out - ref_out).abs().max()) if ref_out is not None else -1:.5f}")
                if "inductor_realrope" in variants:
                    torch._dynamo.reset()
                    cstep = torch.compile(step, mode="reduce-overhead", fullgraph=False)
                    ms, out = bench(cstep.forward, inputs, args.iters)
                    results["inductor_realrope"] = ms
                    print(f"[trt-bench] inductor realrope: {ms:.3f} ms/step "
                          f"maxdiff={float((out - ref_out).abs().max()) if ref_out is not None else -1:.5f}")
                if "trt_bf16_realrope" in variants:
                    import torch_tensorrt
                    torch._dynamo.reset()
                    tstep = torch.compile(
                        step,
                        backend="tensorrt",
                        options={
                            "enabled_precisions": {torch.float32, torch.float16, torch.bfloat16},
                            "min_block_size": 1,
                            "truncate_double": True,
                        },
                        fullgraph=False,
                    )
                    ms, out = bench(tstep.forward, inputs, args.iters)
                    results["trt_bf16_realrope"] = ms
                    print(f"[trt-bench] tensorrt bf16 realrope: {ms:.3f} ms/step "
                          f"maxdiff={float((out - ref_out).abs().max()) if ref_out is not None else -1:.5f}")
            finally:
                mot_mod.rope_apply = _orig_rope

        if "trt_fp8" in variants:
            import torch_tensorrt
            import modelopt.torch.quantization as mtq

            def forward_loop(m):
                for _ in range(4):
                    m(**inputs)

            qstep = mtq.quantize(step, mtq.FP8_DEFAULT_CFG, forward_loop=forward_loop)
            torch._dynamo.reset()
            tqstep = torch.compile(
                qstep,
                backend="tensorrt",
                options={
                    "enabled_precisions": {
                        torch.float8_e4m3fn, torch.float32, torch.float16, torch.bfloat16,
                    },
                    "min_block_size": 1,
                    "truncate_double": True,
                },
                fullgraph=False,
            )
            ms, out = bench(tqstep.forward, inputs, args.iters)
            results["trt_fp8"] = ms
            print(f"[trt-bench] tensorrt fp8(modelopt): {ms:.3f} ms/step "
                  f"maxdiff={float((out - ref_out).abs().max()) if ref_out is not None else -1:.5f}")

    row = {
        "tag": "trt_step_action",
        "maskless": bool(args.maskless),
        "ckpt": str(ckpt_path),
        "step_ms": {k: round(v, 3) for k, v in results.items()},
        "per_call_estimate_ms_10steps": {k: round(v * 10, 1) for k, v in results.items()},
        "iters": args.iters,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
    }
    with open(args.output, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
