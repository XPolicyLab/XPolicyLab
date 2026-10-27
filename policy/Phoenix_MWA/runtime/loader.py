"""Load Phoenix-MWA weights and create independent RoboTwin sessions."""

import os
import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parent
POLICY_DIR = SOURCE.parent


def activate_source():
    for name in ("wam", "eval.robotwin"):
        module = sys.modules.get(name)
        origin = getattr(module, "__file__", None)
        if origin and not Path(origin).resolve().is_relative_to(SOURCE):
            raise RuntimeError(
                f"{name} already comes from another checkout; use a fresh process"
            )
    for p in (SOURCE, SOURCE / "src"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def build_runtime(root, cfg):
    activate_source()
    root = Path(root).resolve()
    from .checkpoints import verify_local

    # Downloaded public files are checked once by download_checkpoint.sh.
    verify_local(root, full_hash=False, include_policy=False)
    os.environ["DINO_ENCODER_CHECKPOINT"] = str(root / "encoders/dino110k.pt")
    os.environ["STVAE_CHECKPOINT"] = str(root / "encoders/stvae150k.pt")
    os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(root)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"
    from eval.robotwin.joint96_robot37_policy import (
        build_runtime as upstream_build,
        Robot37Joint96Session,
    )

    checkpoint = root / str(cfg.get("policy_file") or "policy.pt")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    runtime, session_cfg = upstream_build(
        checkpoint,
        training_config=SOURCE / "configs/inference/phoenix_robotwin.yaml",
        stats_path=SOURCE / "assets/stats/robotwin.json",
        output_dir=Path(cfg.get("audit_dir") or POLICY_DIR / "audit"),
        device=str(cfg.get("device") or "cuda:0"),
        denoise=int(cfg.get("num_inference_steps", 10)),
        replan=int(cfg.get("replan_steps", 32)),
        seed=int(cfg.get("seed") or 0),
        current_visual_mode="repeat9_full_stvae",
    )
    return runtime, lambda: Robot37Joint96Session(runtime, session_cfg)
