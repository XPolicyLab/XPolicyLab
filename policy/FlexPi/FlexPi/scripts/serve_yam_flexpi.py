"""WebSocket policy server for YAM FlexPi.

Loads a ``YamFlexPiPolicy`` once, holds the GPU, and serves
``infer_action_chunk(obs, prompt) -> action_chunk[T_act, 32]`` over websocket
to one or more thin raiden bridges. Sister of
``yam_openpi/scripts/serve_policy.py`` but FlexPi-specific (no tyro union
config; flat argparse around our existing ``build_policy_from_checkpoint``
factory).

Lifecycle:
    1. Build the policy from the trained checkpoint (loads Wan2.2 + DINO +
       ActionDiT, ~15 s ckpt + ~5 s VAE + ~5 s DINO).
    2. If ``--torch-compile``, compile-warmup happens inside
       ``YamFlexPiPolicy.__init__`` (~45-60 s).
    3. Prewarm via one synthetic flat-schema ``infer`` call. Catches
       flat→nested glue bugs at boot, not on the bridge's first frame.
    4. Bind the websocket port.  /healthz responds 200 OK once the listener is
       up; clients see it green only after the prewarm completed.

Usage::

    python scripts/serve_yam_flexpi.py \\
        --ckpt-path /path/to/step_XXXX.pt \\
        --host 0.0.0.0 --port 8000 \\
        --num-inference-steps 4 \\
        --torch-compile --torch-compile-mode reduce-overhead \\
        --offload-text-encoder \\
        --action-horizon 32 \\
        --mixed-precision bf16 \\
        --default-prompt "Place the fork on the pink plate ..."
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Path bootstrap so ``python scripts/serve_yam_flexpi.py`` works from any cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np  # noqa: E402

from experiments.yam.flexpi_policy.deploy_policy import (  # noqa: E402
    build_policy_from_checkpoint,
    _CAM_ORDER,
    _D,
)
from experiments.yam.flexpi_policy.server_adapter import YamFlexPiServerPolicy  # noqa: E402
from experiments.yam.flexpi_policy.server_smoother import (  # noqa: E402
    add_smoother_args,
    build_smoother_from_args,
)
from experiments.yam.flexpi_policy.server_speed import (  # noqa: E402
    add_speed_args,
    build_adapter_from_args,
)
from experiments.yam.flexpi_policy._openpi_vendor.websocket_policy_server import (  # noqa: E402
    WebsocketPolicyServer,
)
from flexpi.datasets.lerobot.robot_video_dataset import _PER_CAM_HW  # noqa: E402


LOG_PREFIX = "[serve_yam_flexpi]"


class _SessionAwareWebsocketServer(WebsocketPolicyServer):
    """``WebsocketPolicyServer`` that fires ``start_session`` / ``end_session``
    on the underlying policy when a WS connection opens / closes.

    Used to scope per-session work to one bridge run: the bridge
    `rd infer` invocation establishes a connection; ctrl+c on the bridge
    closes it; the policy's session hooks fire in between (session-timer
    reset + final elapsed echo, ``PredictionRecorder`` open/close, etc.).

    Falls back to the parent's behavior when the policy doesn't expose
    ``start_session`` (defensive no-op).
    """

    async def _handler(self, websocket):  # type: ignore[override]
        # Reuse parent message-loop logic; just wrap with the lifecycle hooks.
        remote = str(getattr(websocket, "remote_address", ""))
        start_fn = getattr(self._policy, "start_session", None)
        end_fn = getattr(self._policy, "end_session", None)
        if callable(start_fn):
            try:
                start_fn(remote_addr=remote)
            except Exception as e:  # noqa: BLE001
                # A failed start_session must not block serving — log and
                # continue without recording.
                logging.getLogger(__name__).warning(
                    "start_session(%r) failed: %r", remote, e,
                )
        try:
            await super()._handler(websocket)
        finally:
            if callable(end_fn):
                try:
                    end_fn()
                except Exception as e:  # noqa: BLE001
                    logging.getLogger(__name__).warning(
                        "end_session() failed: %r", e,
                    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="YAM FlexPi websocket policy server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Server identity.
    p.add_argument("--ckpt-path", required=True, help="Path to FlexPi checkpoint (.pt).")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--default-prompt", default="",
                   help="Task instruction used if the bridge does not send 'prompt'.")
    p.add_argument("--no-prewarm", action="store_true",
                   help="Skip the synthetic infer at boot. Off-the-shelf risk: first "
                        "real bridge frame eats the torch.compile cost.")

    # Forwarded to build_policy_from_checkpoint.
    p.add_argument("--device", default=_D["device"])
    p.add_argument("--mixed-precision", default=_D["mixed_precision"], choices=["no", "fp16", "bf16"])
    p.add_argument("--num-inference-steps", type=int, default=_D["num_inference_steps"])
    p.add_argument("--action-horizon", type=int, default=_D["action_horizon"],
                   help="Override action horizon (default: from trained config).")
    p.add_argument("--dataset-stats-path", default=None,
                   help="Override; auto-walked from --ckpt-path if omitted.")
    p.add_argument("--intrinsics-json", default=None,
                   help="Fallback intrinsics when the bridge omits per-step K.")
    p.add_argument("--sigma-shift", type=float, default=_D["sigma_shift"])
    p.add_argument("--seed", type=int, default=_D["seed"])
    p.add_argument("--text-cfg-scale", type=float, default=_D["text_cfg_scale"])
    p.add_argument("--negative-prompt", default=_D["negative_prompt"])
    p.add_argument("--offload-text-encoder", action=argparse.BooleanOptionalAction,
                   default=_D["offload_text_encoder"],
                   help="Keep T5 on CPU to save ~10 GB VRAM.")
    p.add_argument("--torch-compile", action=argparse.BooleanOptionalAction,
                   default=_D["torch_compile"],
                   help="Compile the denoise step + capture CUDA graph (~3x speedup, "
                        "~45-60 s warmup at startup).")
    p.add_argument("--torch-compile-mode", default=_D["torch_compile_mode"],
                   choices=["default", "reduce-overhead", "max-autotune"])
    p.add_argument("--torch-compile-scope", default="loop",
                   choices=["step", "loop"],
                   help="'step' compiles one denoise step; 'loop' unrolls the "
                        "whole denoise loop (faster, longer first compile).")
    p.add_argument("--quantization", default=_D["quantization"], help='None or "int8".')
    p.add_argument("--attn-backend", default="auto",
                   choices=["sdpa", "flex", "auto"],
                   help="Attention kernel for the MoT blocks.")
    p.add_argument("--encoder-cuda-graph", action="store_true",
                   help="Capture the VAE/DINO encoders in a CUDA graph.")
    p.add_argument("--compile-encoders", action="store_true",
                   help="torch.compile the VAE/DINO encoders instead of capturing "
                        "them. Mutually exclusive with --encoder-cuda-graph; a "
                        "no-op under --no-torch-compile.")

    # TensorRT KV-split joint engines. Both must be given together; the pair is
    # baked for one checkpoint AND one token geometry (see
    # scripts/inference_opt/trt_onnx_joint_split_engine.py). The weights are baked in and the
    # engine does NOT verify which checkpoint it came from — serving step N's
    # engine against step M's checkpoint silently produces wrong actions.
    p.add_argument("--trt-joint-prefill-split-engine", default=None,
                   help="KV-split prefill engine; set together with "
                        "--trt-joint-decode-split-engine.")
    p.add_argument("--trt-joint-decode-split-engine", default=None,
                   help="KV-split decode engine; see "
                        "--trt-joint-prefill-split-engine.")
    p.add_argument("--trt-joint-free-video-blocks", action="store_true",
                   help="Free the torch video-block weights once the engines "
                        "are live (saves VRAM; off-shape calls then raise).")
    p.add_argument("--glue-cache", action="store_true",
                   help="Reuse the glue tensors between denoise steps. Pairs "
                        "with the TRT engine path; incompatible with "
                        "--torch-compile (joint boot hangs).")

    # Boot-level flex regime defaults. Unset = trained-config default. A
    # bridge-provided per-request value still wins (server_adapter merges with
    # setdefault). These also steer the WARMUP regime, so an action-only boot
    # compiles the action fast path at startup instead of on the robot's
    # first frame.
    _FLEX_HELP = "Default %s for every request (incl. warmup); unset = trained default."
    p.add_argument("--infer-joint-video", default=None, choices=["true", "false"],
                   help=_FLEX_HELP % "joint_video")
    p.add_argument("--infer-joint-dino", default=None, choices=["true", "false"],
                   help=_FLEX_HELP % "joint_dino")
    p.add_argument("--infer-joint-pointmap", default=None, choices=["true", "false"],
                   help=_FLEX_HELP % "joint_pointmap")
    p.add_argument("--infer-present-video", default=None, choices=["true", "false"],
                   help=_FLEX_HELP % "present_video")
    p.add_argument("--infer-present-dino", default=None, choices=["true", "false"],
                   help=_FLEX_HELP % "present_dino")
    p.add_argument("--infer-present-pointmap", default=None, choices=["true", "false"],
                   help=_FLEX_HELP % "present_pointmap")
    p.add_argument("--dynamic-step-skip", action="store_true",
                   help="Enable similarity-gated diffusion step skip in the joint "
                        "denoise loop (FlexPiUnifiedJoint only). Pair with the "
                        "model's default --num-inference-steps so there are steps "
                        "to skip.")

    # Post-processing (off by default; opt-in via flags).
    add_smoother_args(p)
    add_speed_args(p)


    # Prediction recording (off by default).
    p.add_argument(
        "--record-predictions-dir", default=None,
        help="When set, every chunk forces the joint-denoise regime "
             "(joint_video=joint_dino=joint_pointmap=True) and the server "
             "appends the decoded RGB + DINO PCA + pointmap PCA rows to an "
             "MP4 under <dir>/<timestamp>_<remote>/prediction.mp4. Each WS "
             "connection (= each `rd infer` invocation) gets its own subdir; "
             "ctrl+c on the bridge disconnects the WS and the server "
             "finalizes the file. Joint denoise is ~3x slower than the "
             "fast action-only path — the robot will visibly stall between "
             "chunks.",
    )
    p.add_argument(
        "--record-predictions-fps", type=int, default=8,
        help="MP4 playback FPS for the recording. Matches the trainer "
             "eval-video default of 8 (slow-motion relative to 32hz "
             "predicted future). Bump to 32 for real-time-ish playback.",
    )
    p.add_argument(
        "--record-predictions-frames-per-chunk", type=int, default=0,
        help="When > 0, keep only the first N decoded frames per chunk in "
             "prediction.mp4 instead of all ~9. Use this to align chunk "
             "boundaries seamlessly: each chunk's prediction covers ~0.66 s "
             "of model-imagined future but only chunk_size/action_horizon of "
             "it (~0.27 s for the YAM defaults of chunk_size=8, "
             "action_horizon=32) actually elapses before the next chunk "
             "fires. Pick N = ceil(F_dec * chunk_size / action_horizon) — "
             "with the YAM defaults that's 2 or 3. Default 0 = keep all 9 "
             "frames per chunk (the original training-eval-style layout, "
             "with the visible per-chunk anchor pattern).",
    )
    p.add_argument(
        "--record-predictions-pca-warmup-chunks", type=int, default=3,
        help="The 3 PCA rows (DINO, pointmap-VAE-PCA, RGB-VAE-PCA) fit a "
             "fixed PCA basis + global min/max over the first N chunks, then "
             "reuse them for the whole session so colors stay consistent "
             "across chunks (otherwise each chunk re-fits and the same "
             "feature shows different colors). The first N chunks are "
             "buffered and written once the warmup window closes. Default 3.",
    )
    return p


def _log_vram(stage: str) -> None:
    nvsmi = shutil.which("nvidia-smi")
    if nvsmi is None:
        return
    try:
        out = subprocess.check_output(
            [nvsmi, "--query-gpu=memory.free,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            timeout=5.0,
        ).decode().strip()
        print(f"{LOG_PREFIX} VRAM {stage}: {out} (free, used, total MiB)")
    except Exception as exc:  # noqa: BLE001
        print(f"{LOG_PREFIX} VRAM probe failed ({exc!r}); continuing.")


def _synthesize_prewarm_obs() -> dict:
    """A zero-filled flat obs at exactly the wire shapes the server expects."""
    obs: dict = {}
    for cam, (h, w) in _PER_CAM_HW:
        obs[f"observation/image_{cam}"] = np.zeros((h, w, 3), dtype=np.uint8)
        obs[f"observation/depth_{cam}"] = np.zeros((h, w), dtype=np.uint16)
        # Sane K placeholder so set_camera_intrinsics has a real matrix to use.
        obs[f"observation/intrinsics_{cam}"] = np.array(
            [200.0, 200.0, w / 2.0, h / 2.0], dtype=np.float32,
        )
    obs["observation/state"] = np.zeros(32, dtype=np.float32)
    obs["prompt"] = "warmup"
    return obs


def _verify_split_engine_provenance(prefill: str, decode: str, ckpt_path: str) -> None:
    """Refuse an engine pair that was baked for a different checkpoint.

    The KV-split engines carry BAKED weights and expose nothing at runtime that
    identifies their source, so pairing step N's engine with step M's checkpoint
    produces confidently wrong actions and no error. A stale `SPLIT_DIR` doing
    exactly that has driven a joint to 2.39 rad on a real arm. The builder
    writes `build_meta.json` beside the engines; this is the check that makes it
    load-bearing rather than documentation.

    Missing sidecar (engines baked before the builder wrote one) warns loudly
    rather than refusing -- the pairing may well be correct, and refusing would
    strand every pre-existing engine dir.
    """
    ckpt = Path(ckpt_path).expanduser().resolve()
    for role, engine in (("prefill", prefill), ("decode", decode)):
        meta_path = Path(engine).expanduser().resolve().parent / "build_meta.json"
        if not meta_path.is_file():
            print(f"{LOG_PREFIX} WARNING: no build_meta.json beside the {role} engine "
                  f"({meta_path.parent}). Nothing can verify it was baked for "
                  f"{ckpt.name} -- confirm by hand before touching the robot.")
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except Exception as exc:  # noqa: BLE001
            print(f"{LOG_PREFIX} WARNING: {meta_path} is unreadable ({exc!r}); skipping check.")
            continue
        baked = meta.get("ckpt_name") or Path(str(meta.get("ckpt", ""))).name
        if baked and baked != ckpt.name:
            raise SystemExit(
                f"{LOG_PREFIX} REFUSING TO START: the {role} engine in "
                f"{meta_path.parent} was baked for {baked!r}, but --ckpt-path is "
                f"{ckpt.name!r}. The engine's weights are baked in, so this pairing "
                f"would serve the wrong policy silently. Re-set the whole "
                f"SPLIT_DIR/CKPT group, or rebake."
            )
        size = meta.get("ckpt_size_bytes")
        if size and ckpt.exists() and int(size) != ckpt.stat().st_size:
            raise SystemExit(
                f"{LOG_PREFIX} REFUSING TO START: the {role} engine in "
                f"{meta_path.parent} records ckpt size {size} B but {ckpt} is "
                f"{ckpt.stat().st_size} B. Same filename, different file."
            )
        print(f"{LOG_PREFIX} {role} engine provenance OK "
              f"(baked for {baked}, step {meta.get('step')})")


def main() -> int:
    args = _build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, force=True)

    if args.glue_cache and args.torch_compile:
        raise SystemExit(
            f"{LOG_PREFIX} --glue-cache and --torch-compile are mutually "
            "exclusive (joint boot hangs). Use --glue-cache on the TRT engine "
            "path, --torch-compile on the engine-free path."
        )
    split_engines = (
        args.trt_joint_prefill_split_engine,
        args.trt_joint_decode_split_engine,
    )
    if any(split_engines) and not all(split_engines):
        raise SystemExit(
            f"{LOG_PREFIX} --trt-joint-prefill-split-engine and "
            "--trt-joint-decode-split-engine must be given together."
        )
    if all(split_engines):
        _verify_split_engine_provenance(*split_engines, args.ckpt_path)

    # Boot-level flex regime defaults: applied to every request the bridge
    # doesn't override, AND to the policy warmup (so the compile cost of the
    # served regime is paid at boot).
    flex_defaults = {
        key: val == "true"
        for key, val in (
            ("joint_video", args.infer_joint_video),
            ("joint_dino", args.infer_joint_dino),
            ("joint_pointmap", args.infer_joint_pointmap),
            ("present_video", args.infer_present_video),
            ("present_dino", args.infer_present_dino),
            ("present_pointmap", args.infer_present_pointmap),
        )
        if val is not None
    }
    if flex_defaults:
        print(f"{LOG_PREFIX} flex regime defaults: {flex_defaults}")

    print(f"{LOG_PREFIX} ckpt={args.ckpt_path}")
    _log_vram("pre-load")

    t0 = time.perf_counter()
    policy = build_policy_from_checkpoint(
        args.ckpt_path,
        dataset_stats_path=args.dataset_stats_path,
        intrinsics_json=args.intrinsics_json,
        device=args.device,
        mixed_precision=args.mixed_precision,
        action_horizon=args.action_horizon,
        num_inference_steps=args.num_inference_steps,
        sigma_shift=args.sigma_shift,
        seed=args.seed,
        text_cfg_scale=args.text_cfg_scale,
        negative_prompt=args.negative_prompt,
        offload_text_encoder=args.offload_text_encoder,
        torch_compile=args.torch_compile,
        torch_compile_mode=args.torch_compile_mode,
        torch_compile_scope=args.torch_compile_scope,
        quantization=args.quantization,
        attn_backend=args.attn_backend,
        trt_joint_free_video_blocks=args.trt_joint_free_video_blocks,
        trt_joint_prefill_split_engine_path=args.trt_joint_prefill_split_engine,
        trt_joint_decode_split_engine_path=args.trt_joint_decode_split_engine,
        glue_cache=args.glue_cache,
        encoder_cuda_graph=args.encoder_cuda_graph,
        compile_encoders=args.compile_encoders,
        dynamic_step_skip=args.dynamic_step_skip,
        warmup_flex=flex_defaults or None,
    )
    print(f"{LOG_PREFIX} policy loaded in {time.perf_counter() - t0:.1f} s")
    _log_vram("post-load")

    smoother = build_smoother_from_args(args)
    adapter = build_adapter_from_args(args)
    if smoother is not None:
        print(f"{LOG_PREFIX} smoother ON  cfg={smoother.cfg}")
    if adapter is not None:
        print(f"{LOG_PREFIX} speed adapter ON  mode={adapter.cfg.mode}")


    server_policy = YamFlexPiServerPolicy(
        policy,
        default_prompt=args.default_prompt,
        smoother=smoother,
        adapter=adapter,
        flex_defaults=flex_defaults or None,
        record_predictions_dir=args.record_predictions_dir,
        record_fps=args.record_predictions_fps,
        record_frames_per_chunk=args.record_predictions_frames_per_chunk,
        record_pca_warmup_chunks=args.record_predictions_pca_warmup_chunks,
    )

    if not args.no_prewarm:
        print(f"{LOG_PREFIX} prewarm: synthesizing dummy obs and running one infer...")
        prewarm_obs = _synthesize_prewarm_obs()
        t0 = time.perf_counter()
        out = server_policy.infer(prewarm_obs)
        prewarm_s = time.perf_counter() - t0
        assert out["actions"].shape == (policy.action_horizon, 32), (
            f"prewarm action shape {out['actions'].shape} != expected "
            f"({policy.action_horizon}, 32)"
        )
        print(f"{LOG_PREFIX} PREWARM DONE in {prewarm_s:.1f} s "
              f"(actions shape={out['actions'].shape})")
        _log_vram("post-prewarm")

    metadata = server_policy.metadata
    print(f"{LOG_PREFIX} serving on {args.host}:{args.port}  metadata={metadata}")
    if args.record_predictions_dir:
        print(
            f"{LOG_PREFIX} prediction recording ON  dir={args.record_predictions_dir}  "
            f"fps={args.record_predictions_fps}  (joint-denoise regime forced; "
            "expect ~3x slowdown per chunk)"
        )
    # Always use the session-aware subclass so the session timer (first
    # action → disconnect) runs whether or not recording is enabled; the
    # extra cost is one call to ``start_session`` / ``end_session`` per WS
    # connection — negligible compared to inference.
    server = _SessionAwareWebsocketServer(
        policy=server_policy,
        host=args.host,
        port=args.port,
        metadata=metadata,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
