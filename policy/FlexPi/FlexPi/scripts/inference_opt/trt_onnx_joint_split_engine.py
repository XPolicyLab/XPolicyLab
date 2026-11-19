"""L3c spike: export the joint KV-split prefill + decode cores to TRT engines.

Prefill: clean anchors -> stacked per-layer K/V (k_all/v_all [L,B,Sanc,HD]).
Decode:  noisy tokens + k_all/v_all cache -> noisy_video/action outputs.

Mirrors scripts/inference_opt/trt_onnx_joint_engine.py: flat-tensor wrappers, real-valued
RoPE (mot.rope_apply -> sb.rope_apply_real), dynamo ONNX export, raw TRT build.
Inputs come from the REAL captured core I/O (scripts/inference_opt/trt_joint_parity.py) so the
split (anchor/noisy) matches deploy exactly.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for p in (PROJECT_ROOT, PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from omegaconf import OmegaConf  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
import trt_fp8_step_bench as sb  # noqa: E402
from trt_onnx_step_engine import build_engine  # noqa: E402
from benchmark_flex_latency import _find_trained_config  # noqa: E402


class PrefillCore(torch.nn.Module):
    def __init__(self, mot, inner_mask, outer_mask):
        super().__init__()
        self.mot = mot
        self.register_buffer("inner_mask", inner_mask)
        self.register_buffer("outer_mask", outer_mask)

    def forward(self, anchor_tokens, anchor_freqs_real, anchor_t_mod, anchor_ctx, anchor_ctx_mask):
        return self.mot.prefill_joint_anchor_kv(
            anchor_tokens=anchor_tokens, anchor_freqs=anchor_freqs_real,
            anchor_t_mod=anchor_t_mod,
            anchor_context_payload={"context": anchor_ctx, "mask": anchor_ctx_mask},
            anchor_inner_mask=self.inner_mask, anchor_outer_mask=self.outer_mask,
        )


class DecodeCore(torch.nn.Module):
    def __init__(self, mot, anchor_pos_idx, noisy_pos_idx, video_seq_len, inner_mask, outer_mask):
        super().__init__()
        self.mot = mot
        self.register_buffer("anchor_pos_idx", anchor_pos_idx)
        self.register_buffer("noisy_pos_idx", noisy_pos_idx)
        self.video_seq_len = int(video_seq_len)
        self.register_buffer("inner_mask", inner_mask)
        self.register_buffer("outer_mask", outer_mask)

    def forward(self, noisy_video, action, noisy_freqs_real, action_freqs_real,
                noisy_t_mod, action_t_mod, noisy_ctx, noisy_ctx_mask,
                action_ctx, action_ctx_mask, k_all, v_all):
        return self.mot.decode_joint_noisy(
            noisy_video_tokens=noisy_video, action_tokens=action,
            noisy_video_freqs=noisy_freqs_real, action_freqs=action_freqs_real,
            noisy_video_t_mod=noisy_t_mod, action_t_mod=action_t_mod,
            noisy_video_context_payload={"context": noisy_ctx, "mask": noisy_ctx_mask},
            action_context_payload={"context": action_ctx, "mask": action_ctx_mask},
            k_all=k_all, v_all=v_all,
            anchor_pos_idx=self.anchor_pos_idx, noisy_pos_idx=self.noisy_pos_idx,
            video_seq_len=self.video_seq_len,
            decode_inner_mask=self.inner_mask, decode_outer_mask=self.outer_mask,
        )


def real(freqs_complex):
    return torch.view_as_real(freqs_complex.contiguous()).to(torch.float32).contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--workdir", default="/tmp/flex_trt_onnx_joint_split")
    ap.add_argument("--opt-level", type=int, default=3)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--stage", default="both", choices=["prefill", "decode", "both"])
    args = ap.parse_args()
    device, dtype = "cuda", torch.bfloat16
    wd = Path(args.workdir); wd.mkdir(parents=True, exist_ok=True)

    tc = OmegaConf.load(_find_trained_config(Path(args.ckpt)))
    mc = OmegaConf.create(OmegaConf.to_container(tc.model, resolve=True)); mc.load_text_encoder = False
    # `load_checkpoint` below is the source of truth for every DiT parameter, so
    # loading either pretrained set costs only I/O -- and would make an
    # inference-only machine a hard dependency on ~20 GB of Wan2.2 base shards
    # just to overwrite them. Same as benchmark_flex_latency.py.
    mc.action_dit_pretrained_path = None; mc.skip_dit_load_from_pretrain = True
    model = instantiate(mc, model_dtype=dtype, device=device); model.load_checkpoint(str(args.ckpt))
    model = model.to(device).eval(); mot = model.mot

    cal = torch.load(args.calib, map_location="cpu", weights_only=False)
    def dev(v):
        if torch.is_tensor(v): return v.to(device)
        if isinstance(v, dict): return {k: dev(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)): return type(v)(dev(x) for x in v)
        return v
    s = {k: dev(v) for k, v in cal["steps"][0].items()}
    Sv = s["video_tokens"].shape[1]; Sa = s["action_tokens"].shape[1]; T = Sv + Sa

    tm = s["video_t_mod"][0]
    anchor_bool = (tm == tm[0]).all(dim=(-1, -2)); noisy_bool = ~anchor_bool
    anchor_g = torch.where(anchor_bool)[0]; noisy_video_g = torch.where(noisy_bool)[0]
    noisy_g = torch.cat([noisy_video_g, torch.arange(Sv, T, device=device)])
    full_mask = s["attention_mask"]
    outer_mask = torch.zeros((T, T), dtype=torch.bool, device=device); off = 0
    for L, M in zip(s["sub_stream_lens"], s["sub_stream_self_masks"]):
        outer_mask[off:off + L, off:off + L] = M; off += L

    def slice_ctx_mask(m, sel):
        d = [i for i, x in enumerate(m.shape) if x == Sv][0]
        return m.index_select(d, torch.where(sel)[0])

    # reference (complex rope) noisy positions
    with torch.no_grad():
        xv_ref, xa_ref = mot._forward_joint_inner(
            video_tokens=s["video_tokens"], action_tokens=s["action_tokens"],
            video_freqs=s["video_freqs"], action_freqs=s["action_freqs"],
            video_t_mod=s["video_t_mod"], action_t_mod=s["action_t_mod"],
            video_context_payload=s["video_context_payload"],
            action_context_payload=s["action_context_payload"],
            attention_mask=s["attention_mask"], sub_stream_lens=s["sub_stream_lens"],
            sub_stream_self_masks=s["sub_stream_self_masks"])
    ref_noisy_v = xv_ref[:, noisy_bool].float().cpu(); ref_a = xa_ref.float().cpu()
    del xv_ref, xa_ref; torch.cuda.empty_cache()

    # patch to real rope for export
    import flexpi.models.mot as mot_mod
    mot_mod.rope_apply = sb.rope_apply_real

    # ---- prefill inputs ----
    pre_inputs = dict(
        anchor_tokens=s["video_tokens"][:, anchor_bool, :],
        anchor_freqs_real=real(s["video_freqs"][anchor_g]),
        anchor_t_mod=s["video_t_mod"][:, anchor_bool],
        anchor_ctx=s["video_context_payload"]["context"],
        anchor_ctx_mask=slice_ctx_mask(s["video_context_payload"]["mask"], anchor_bool),
    )
    prefill_core = PrefillCore(mot, full_mask[anchor_g][:, anchor_g], outer_mask[anchor_g][:, anchor_g]).eval()
    with torch.no_grad():
        k_all, v_all = prefill_core(*pre_inputs.values())
    print(f"[split] anchors={int(anchor_bool.sum())} noisy_v={len(noisy_video_g)} k_all={tuple(k_all.shape)}")

    def bench(fn, *a, n=args.iters):
        for _ in range(4): fn(*a)
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(n): fn(*a)
        torch.cuda.synchronize(); return (time.perf_counter() - t) / n * 1e3

    if args.stage in ("prefill", "both"):
        with torch.no_grad():
            ms = bench(lambda *a: prefill_core(*a), *pre_inputs.values())
        print(f"[prefill] eager realrope {ms:.2f} ms")
        onnx_p = wd / "prefill_core.onnx"
        with torch.no_grad():
            torch.onnx.export(prefill_core, tuple(pre_inputs.values()), str(onnx_p), dynamo=True,
                              input_names=list(pre_inputs.keys()), output_names=["k_all", "v_all"])
        print(f"[prefill] onnx exported {onnx_p}")
        eng_p = wd / f"prefill_core_o{args.opt_level}.engine"
        bs = build_engine(onnx_p, eng_p, args.opt_level)
        print(f"[prefill] TRT engine built in {bs:.0f}s -> {eng_p}")

    if args.stage in ("decode", "both"):
        dec_inputs = dict(
            noisy_video=s["video_tokens"][:, noisy_bool, :],
            action=s["action_tokens"],
            noisy_freqs_real=real(s["video_freqs"][noisy_video_g]),
            action_freqs_real=real(s["action_freqs"]),
            noisy_t_mod=s["video_t_mod"][:, noisy_bool],
            action_t_mod=s["action_t_mod"],
            noisy_ctx=s["video_context_payload"]["context"],
            noisy_ctx_mask=slice_ctx_mask(s["video_context_payload"]["mask"], noisy_bool),
            action_ctx=s["action_context_payload"]["context"],
            action_ctx_mask=s["action_context_payload"]["mask"],
            k_all=k_all, v_all=v_all,
        )
        decode_core = DecodeCore(mot, anchor_g, noisy_video_g, Sv,
                                 full_mask[noisy_g][:, :], outer_mask[noisy_g][:, :]).eval()
        with torch.no_grad():
            dv, da = decode_core(*dec_inputs.values())
        pv = (dv.float().cpu() - ref_noisy_v).abs().mean() / ref_noisy_v.abs().mean()
        print(f"[decode] eager realrope parity: noisy rel={pv:.4g}")
        with torch.no_grad():
            ms = bench(lambda *a: decode_core(*a), *dec_inputs.values())
        print(f"[decode] eager realrope {ms:.2f} ms")
        onnx_d = wd / "decode_core.onnx"
        with torch.no_grad():
            torch.onnx.export(decode_core, tuple(dec_inputs.values()), str(onnx_d), dynamo=True,
                              input_names=list(dec_inputs.keys()), output_names=["noisy_video_out", "action_out"])
        print(f"[decode] onnx exported {onnx_d}")
        eng_d = wd / f"decode_core_o{args.opt_level}.engine"
        bs = build_engine(onnx_d, eng_d, args.opt_level)
        print(f"[decode] TRT engine built in {bs:.0f}s -> {eng_d}")

    # Provenance sidecar. The engines carry BAKED weights and nothing at serve
    # time can tell which checkpoint they came from, so serving step N's engine
    # against step M's checkpoint produces confidently wrong actions with no
    # error. Writing this here is what lets serve_yam_flexpi.py refuse that
    # pairing instead of discovering it on the robot.
    ckpt_path = Path(args.ckpt).expanduser().resolve()
    step = None
    for tok in ckpt_path.stem.split("_"):
        if tok.isdigit():
            step = int(tok)
    meta = {
        "ckpt": str(ckpt_path),
        "ckpt_name": ckpt_path.name,
        "ckpt_size_bytes": ckpt_path.stat().st_size if ckpt_path.exists() else None,
        "step": step,
        "ctx_len": int(cal.get("ctx_len")) if isinstance(cal, dict) and "ctx_len" in cal else None,
        "opt_level": args.opt_level,
        "stage": args.stage,
        "built": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "machine": os.uname().nodename,
    }
    (wd / "build_meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"[meta] wrote {wd / 'build_meta.json'}  (ckpt={ckpt_path.name} step={step})")

    print("SPLIT_ENGINE_DONE")


if __name__ == "__main__":
    main()
