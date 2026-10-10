"""Snapshot a task's inputs and create a fresh workspace for each Test episode."""

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from XPolicyLab.utils.checkpoint_resolver import DEFAULT_EXPLICIT_KEYS, resolve_checkpoint_root
from XPolicyLab.policy.EmbodiedRSI.runtime.config import setting
from XPolicyLab.policy.EmbodiedRSI.runtime.files import write_json

POLICY_DIR = Path(__file__).resolve().parents[1]
INPUTS = ("primitives/primitives.md", "scripts/env.py", "submission/solution.py")


def tree_hashes(root, *, components=None):
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"Missing or linked asset directory: {root}")
    files = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if components and relative.parts[0] not in components:
            continue
        if path.is_symlink():
            raise ValueError(f"Linked asset is forbidden: {relative}")
        if path.is_file():
            files[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return files


def verify_tree(root, expected, *, components=None):
    if tree_hashes(root, components=components) != expected:
        raise ValueError(f"Frozen asset hash mismatch in {root}")


def task_inputs(root, task):
    files = {name: (root / name).read_bytes() for name in INPUTS}
    instruction = (root / "instruction.md").read_bytes()
    recipe = root / "recipes" / f"{task}.md"
    if not recipe.is_file():
        raise ValueError(f"No task recipe: {recipe}")
    attribution = json.loads((root / "recipes/manifest.json").read_text())
    url = f"{attribution['repository']}/blob/{attribution['commit']}/{attribution['path']}/{task}.md"
    appendix = f"<!-- TASK RECIPE BEGIN -->\n<!-- Source: {url}; {attribution['license']} -->\nTASK RECIPE:\n"
    files["test.md"] = instruction + appendix.encode() + recipe.read_bytes() + b"\n<!-- TASK RECIPE END -->\n"
    return files


def prepare_run(model_cfg):
    cfg = dict(model_cfg)
    if cfg.get("bench_name") != "RoboDojo" or cfg.get("env_cfg_type") != "arx_x5":
        raise ValueError("Supported environment: RoboDojo / arx_x5")
    if cfg.get("protocol") != "ws" or type(cfg.get("eval_batch")) is not bool:
        raise ValueError("Use protocol=ws and a boolean eval_batch")
    if cfg.get("phase", "test") != "test" or cfg.get("action_type") not in {"ee", "joint"}:
        raise ValueError("Test only; action_type must be ee or joint")
    if type(cfg.get("diagnostic")) is not bool:
        raise ValueError("diagnostic must be boolean")
    if cfg.get("ckpt_name") == "diagnostic" and not cfg["diagnostic"]:
        raise ValueError("The diagnostic checkpoint is only for EVAL_ENV_TYPE=debug")
    if not cfg["diagnostic"] and not setting("CODEX_HOME"):
        raise ValueError("Set EMBODIEDRSI_CODEX_HOME to a provider/auth directory")
    for key in ("execution_budget", "submission_timeout_sec", "feedback_fps"):
        if type(cfg.get(key)) is not int or cfg[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")

    selection = dict(cfg)
    if not any(selection.get(k) for k in DEFAULT_EXPLICIT_KEYS) and cfg.get("ckpt_name") in {
        "embodiedrsi_astra_xhigh", "diagnostic",
    }:
        selection["checkpoint_path"] = str(POLICY_DIR / "Workspace")
    root = resolve_checkpoint_root(selection, POLICY_DIR / "checkpoints", policy_dir=POLICY_DIR)
    task = str(cfg.get("task_name", "")).removesuffix("_random")
    if not task or Path(task).name != task:
        raise ValueError("task_name must name a RoboDojo task")
    inputs = task_inputs(root, task)
    experience = {}
    for component in ("skills", "lessons"):
        source = root / component / task
        if source.exists() or source.is_symlink():
            experience[component] = tree_hashes(source)

    run_value = setting("RUN_DIR") or cfg.get("run_dir")
    if run_value:
        run = Path(run_value).expanduser().resolve()
        run.mkdir(parents=True, exist_ok=False)
    else:
        parent = POLICY_DIR.parent.parent / "runs/EmbodiedRSI"
        parent.mkdir(parents=True, exist_ok=True)
        run = Path(tempfile.mkdtemp(prefix=f"{task}-", dir=parent))
    for name, data in inputs.items():
        target = run / "inputs" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    for component in ("skills", "lessons"):
        target = run / "frozen_harness" / component
        if component in experience:
            shutil.copytree(root / component / task, target)
            verify_tree(target, experience[component])
        else:
            target.mkdir(parents=True)
    frozen = {f"{component}/{name}": digest for component, files in experience.items()
              for name, digest in files.items()}
    cfg.update(run_dir=str(run), checkpoint_path=str(root), phase="test", seconds=None)
    cfg["task_config"] = {
        "root": str(run), "name": "EmbodiedRSI/RoboDojo", "description": "Task via get_instruction().",
        "simulator": {"backend": "robodojo", "api_surface": "RoboDojoLowLevelApi",
                      "interpreter": "", "env_task": task},
        "agent": {"preset": "codex", "model": cfg["model"],
                  "reasoning_effort": cfg["reasoning_effort"], "timeout_sec": None,
                  "isolation": "docker", "image": cfg["agent_image"], "network": "filtered",
                  "cpus": int(cfg["agent_cpus"]), "memory_mb": int(cfg["agent_memory_mb"])},
        "test": {"scenes": 1, "budget": cfg["execution_budget"],
                 "submission_timeout_sec": cfg["submission_timeout_sec"],
                 "generation_timeout_sec": None, "feedback_fps": cfg["feedback_fps"],
                 "expose_success_feedback": True},
    }
    write_json(run / "resolved.json", cfg)
    write_json(run / "workspace-manifest.json", {
        "task": task, "inputs": tree_hashes(run / "inputs"), "experience": frozen,
    })
    print(f"[EmbodiedRSI] {task}: {len(frozen)} experience files; run_dir={run}", flush=True)
    return cfg, frozen


def create_workspace(run, inputs, harness):
    """Copy only this task's snapshot; episode programs and observations start fresh."""
    workspace = run / "workspace"
    shutil.copytree(inputs, workspace)
    (workspace / "test.md").rename(workspace / "instruction.md")
    for component in ("scratch", "observations"):
        (workspace / component).mkdir()
    for component in ("skills", "lessons"):
        shutil.copytree(harness / component, workspace / component)
    return workspace
