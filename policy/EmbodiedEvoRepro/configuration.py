"""Validate configuration before starting any external model or simulator."""

import importlib
import os
from pathlib import Path
import re
import subprocess

UPSTREAM_SHA = "470278c478bdb5e4b6f17dde1ea0393761f230f0"


def interface_probe_enabled():
    value = os.environ.get("EMBODIEDEVO_INTERFACE_PROBE", "0")
    if value not in ("0", "1"):
        raise ValueError("EMBODIEDEVO_INTERFACE_PROBE must be 0 or 1")
    mode = os.environ.get("EVAL_ENV_TYPE", "sim") or "sim"
    if value == "1" and mode != "debug":
        raise ValueError("The interface probe is allowed only with EVAL_ENV_TYPE=debug")
    if mode == "debug" and value != "1":
        raise ValueError("Debug requires the explicit interface probe; no real model is called")
    if mode not in ("debug", "sim"):
        raise ValueError("This candidate supports simulation and interface debug only")
    return value == "1"


def validate_scope(cfg):
    expected = {"bench_name": "RoboDojo", "env_cfg_type": "arx_x5", "action_type": "ee"}
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(f"{key} must be {value!r}")
    if cfg.get("eval_batch", False) is not False or cfg.get("num_envs", 1) != 1:
        raise ValueError("Only num_envs=1 and eval_batch=false are supported")
    if cfg.get("phase", "test") != "test" or cfg.get("diagnostic", False):
        raise ValueError("The candidate adapter accepts frozen Test only")
    if cfg.get("protocol", "ws") != "ws":
        raise ValueError("Only the XPolicyLab WS protocol is supported")
    if cfg.get("seed") not in (0, 1, 2):
        raise ValueError("The evaluator seed must be 0, 1 or 2")
    if not isinstance(cfg.get("task_name"), str) or not re.fullmatch(r"[A-Za-z0-9_]+", cfg["task_name"]):
        raise ValueError("task_name must be an explicit RoboDojo task configuration")


def verified_upstream():
    """Import externally installed code, requiring its unchanged pinned Git checkout."""
    root_text = os.environ.get("EMBODIEDEVO_REPO", "")
    if not root_text:
        raise ValueError("Set EMBODIEDEVO_REPO to the pinned, separately installed upstream checkout")
    root = Path(root_text).expanduser().resolve(strict=True)
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if head != UPSTREAM_SHA:
        raise ValueError(f"EmbodiedEvo must be at the declared revision {UPSTREAM_SHA}")
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip()
    if dirty:
        raise ValueError("EmbodiedEvo tracked source differs from the declared revision")
    module = importlib.import_module("orchestrator.config")
    if Path(module.__file__).resolve().parents[1] != root or module.REPO != root:
        raise ValueError("Imported EmbodiedEvo does not match EMBODIEDEVO_REPO; use its editable install")
    return root
