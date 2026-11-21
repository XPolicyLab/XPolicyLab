from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parent.parent


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def event(path, **value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps({"wall_time": time.time(), **value}, allow_nan=False) + "\n")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def inventory(root):
    rows = []
    for path in sorted(Path(root).rglob("*")):
        name = path.relative_to(root).as_posix()
        if path.is_symlink():
            rows.append({"path": name, "symlink": os.readlink(path)})
        elif path.is_file():
            rows.append({"path": name, "bytes": path.stat().st_size, "sha256": digest(path)})
    return rows


def snapshot(source, destination):
    shutil.copytree(source, destination, symlinks=True)
    return inventory(destination)


def stop_process(process):
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def load_config(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text())
    defaults = {"benchmark_python": "python3", "agent_python": "/usr/bin/python3",
                "codex_binary": "codex", "codex_home": "", "model": "gpt-6-astra",
                "reasoning_effort": "max", "explore_seconds": 3600,
                "episode_seconds": 900, "request_seconds": 600,
                "terminal_feedback": False, "options": {}, "public_instructions": "", "train_episodes": []}
    cfg = {**defaults, **cfg}
    def resolve(value):
        target = Path(os.path.expandvars(value)).expanduser()
        return str((path.parent / target).absolute() if not target.is_absolute() else target)
    for key in ("environment_dir", "training_set", "agent_workspace"):
        cfg[key] = resolve(cfg[key])
    for key in ("benchmark_python", "agent_python", "codex_binary"):
        cfg[key] = resolve(cfg[key]) if "/" in cfg[key] else (shutil.which(cfg[key]) or cfg[key])
    for key in ("codex_home", "public_instructions"):
        if cfg[key]:
            cfg[key] = resolve(cfg[key])
    file, name = cfg["adapter"].rsplit(":", 1)
    if not name.isidentifier():
        raise ValueError("adapter must be a file path followed by :ClassName")
    cfg["adapter"] = resolve(file) + ":" + name
    for key in ("train_episodes", "test_episodes"):
        episodes = cfg[key]
        if not isinstance(episodes, list) or (key == "test_episodes" and not episodes):
            raise ValueError(f"{key} must be a list; test_episodes must be nonempty")
        if any(not isinstance(e, dict) or not isinstance(e.get("id"), str) or not e["id"] for e in episodes):
            raise ValueError(f"Every {key} entry needs a nonempty string id")
        if len({e["id"] for e in episodes}) != len(episodes):
            raise ValueError(f"Duplicate ids in {key}")
    if {e["id"] for e in cfg["train_episodes"]} & {e["id"] for e in cfg["test_episodes"]}:
        raise ValueError("Training and test episode ids overlap")
    for key in ("explore_seconds", "episode_seconds", "request_seconds"):
        if type(cfg[key]) not in (int, float) or not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"Invalid {key}")
    if type(cfg["terminal_feedback"]) is not bool or not isinstance(cfg["options"], dict):
        raise ValueError("terminal_feedback must be boolean and options must be an object")
    return cfg


def validate_paths(cfg):
    root, data = Path(cfg["environment_dir"]).resolve(), Path(cfg["training_set"]).resolve()
    if not root.is_dir() or not data.is_dir():
        raise ValueError("environment_dir and training_set must be existing directories")
    if data == root or root.is_relative_to(data):
        raise ValueError("training_set must not contain the environment repository")
    workspace = Path(cfg["agent_workspace"]).resolve()
    if any(workspace == p or workspace.is_relative_to(p) or p.is_relative_to(workspace) for p in (root, data)):
        raise ValueError("agent_workspace must be separate from environment_dir and training_set")
    if not any(p.is_file() for p in data.rglob("*")):
        raise ValueError("training_set is empty")
    if any(p.is_symlink() for p in data.rglob("*")):
        raise ValueError("Materialize training-set symlinks before running (e.g. cp -Lr)")
    adapter = Path(cfg["adapter"].rsplit(":", 1)[0]).resolve()
    if not adapter.is_file() or adapter.is_relative_to(data):
        raise ValueError("The private adapter must exist outside training_set")
    for key in ("benchmark_python", "agent_python"):
        if not Path(cfg[key]).is_file():
            raise ValueError(f"{key} does not exist: {cfg[key]}")
    if cfg["public_instructions"] and not Path(cfg["public_instructions"]).is_file():
        raise ValueError("public_instructions does not exist")
    return {"adapter_sha256": digest(adapter), "environment_revision": cfg.get("environment_revision", "unspecified")}
