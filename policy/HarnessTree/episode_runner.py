"""Step 2b: Direct Python episode runner for HarnessTree.

Drives the episode directly against the bridge + HTTP SAM3 client + skill file.
"""

from __future__ import annotations

import json
import math
import time
import traceback
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import skill_helpers
from .bridge import EpisodeBridge, EpisodeEnv, EpisodeOver
from .colour_toolkit import NativeToolkit
from .exceptions import EpisodeFinished, MotionUnacknowledged
from .robot_api import HarnessRobot

PROGRAMS_DIR = Path(__file__).resolve().parent / "programs"


def load_skill_source(skill: str) -> tuple[Path, str]:
    path = PROGRAMS_DIR / f"{skill}.py"
    if not path.is_file():
        raise FileNotFoundError(f"no skill program for {skill!r} (expected {path})")
    return path, path.read_text(encoding="utf-8")


def run_episode(
    *,
    bridge: EpisodeBridge,
    sam3_client: Any,
    skill: str,
    out_dir: Path,
    boot_timeout_s: float = 120.0,
) -> dict[str, Any]:
    """Execute one episode directly in Python against ``bridge`` and ``sam3_client``."""
    t0 = time.monotonic()
    out_dir.mkdir(parents=True, exist_ok=True)
    status = "completed"
    error: str | None = None
    try:
        bridge.first_obs(boot_timeout_s)
        env = EpisodeEnv(bridge)
        toolkit = NativeToolkit(env, sam3_client, out_dir, colour_only=True)
        robo = HarnessRobot(env, sam3_client, toolkit=toolkit)
        program_path, source = load_skill_source(skill)
        env_globals: dict[str, Any] = {
            "__name__": "__policy__",
            "__file__": str(program_path),
            "robo": robo,
            "skill_helpers": skill_helpers,
            "math": math,
            "time": time,
            "cv2": cv2,
            "np": np,
            "numpy": np,
            "display": lambda _img: None,
            "log": print,
        }
        code = compile(source, str(program_path), "exec")
        exec(code, env_globals)  # noqa: S102
    except (EpisodeFinished, EpisodeOver):
        status = "completed"
    except (Exception, MotionUnacknowledged) as exc:  # noqa: BLE001
        status = "error"
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()

    bridge.finish()
    verdict = bridge.verdict(timeout_s=bridge.poll_s)
    if bridge.motion_error is not None:
        status, error = "error", bridge.motion_error
    result = {
        "status": status,
        "environment_success": bool((verdict or {}).get("success")),
        "evaluator_verdict": verdict,
        "motion_receipts": bridge.receipts,
        "task": skill,
        "robot": "robodojo_xpl_rgb",
        "elapsed_s": round(time.monotonic() - t0, 3),
        "error": error,
    }
    try:
        (out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    return result
