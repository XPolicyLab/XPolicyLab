"""Freeze the robosuite OSC_POSE scale that LIBERO actually loads.

LIBERO's ``ControlEnv`` (8f1084e ``libero/libero/envs/env_wrapper.py``; LIBERO-Plus
4976dc3 likewise) calls ``robosuite.load_controller_config(default_controller=
"OSC_POSE")``, which reads ``controllers/config/osc_pose.json`` of the robosuite
installed in the LIBERO environment. The vendored OpenWAM pins robosuite 1.4.0
(``benchmarks/libero{,-plus}/environment.yml``, ``setup_env.sh``). The file is
located through that environment's interpreter without importing robosuite,
hashed, and its input/output ranges are frozen. Nothing falls back to defaults.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess

import numpy as np

from .contracts import file_digest

PINNED_ROBOSUITE = "1.4.0"
CONFIG = Path("controllers/config/osc_pose.json")
LOCATE = ("import importlib.metadata, importlib.util, json; "
          "spec = importlib.util.find_spec('robosuite'); "
          "print(json.dumps({'package': list(spec.submodule_search_locations)[0], "
          "'version': importlib.metadata.version('robosuite')}))")


def locate(python) -> dict:
    """Ask the LIBERO interpreter where its robosuite package lives."""
    if not isinstance(python, str) or not Path(python).is_absolute():
        raise ValueError("absolute path of the LIBERO environment's python required")
    try:
        result = subprocess.run([python, "-c", LOCATE], check=True, capture_output=True,
                                text=True, timeout=120)
        found = json.loads(result.stdout.strip().splitlines()[-1])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
        raise ValueError(f"cannot locate robosuite through {python}: {type(exc).__name__}") from None
    if not isinstance(found, dict) or not isinstance(found.get("package"), str):
        raise ValueError("robosuite location report malformed")
    return found


def _range(value, name, size):
    try:
        array = None if isinstance(value, bool) or value is None else np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        array = None
    if array is None or array.shape not in ((), (size,)) or not np.isfinite(array).all():
        raise ValueError(f"OSC_POSE {name} must be a finite scalar or {size}-vector")
    return value


def read_osc_pose(path: Path) -> dict:
    """The scale-relevant fields of an OSC_POSE delta controller config."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise ValueError(f"robosuite OSC_POSE controller config unreadable: {path}") from None
    if not isinstance(data, dict) or data.get("type") != "OSC_POSE" or data.get("control_delta") is not True:
        raise ValueError("robosuite OSC_POSE delta controller config required")
    values = {"type": "OSC_POSE", "control_delta": True}
    for key in ("input_max", "input_min"):
        values[key] = _range(data.get(key), key, 6)
    for key in ("output_max", "output_min"):
        values[key] = _range(data.get(key), key, 6)
        if np.asarray(values[key]).shape != (6,):
            raise ValueError(f"OSC_POSE {key} must list the 6 pose channels")
    scale = OscScale(values)
    if np.any(scale.input_max <= scale.input_min) or np.any(scale.output_max <= scale.output_min):
        raise ValueError("OSC_POSE ranges must be increasing")
    return values


def freeze_osc_controller(block) -> dict:
    """Locate, hash and record the controller config; fail closed if anything is missing."""
    if not isinstance(block, dict) or not isinstance(block.get("python"), str):
        raise ValueError("osc_controller.python (the LIBERO environment's interpreter) required")
    # Keep the interpreter path as given: resolving a venv symlink would leave the venv.
    python = block["python"]
    found = locate(python)
    if found.get("version") != PINNED_ROBOSUITE:
        raise ValueError(f"LIBERO environment has robosuite {found.get('version')!r}, "
                         f"pinned {PINNED_ROBOSUITE}")
    path = (Path(found["package"]) / CONFIG).resolve()
    if not path.is_file():
        raise ValueError(f"robosuite OSC_POSE controller config missing: {path}")
    return {"python": python, "robosuite_version": found["version"], "path": str(path),
            "sha256": file_digest(path), **read_osc_pose(path)}


class OscScale:
    """robosuite ``Controller.scale_action`` for OSC_POSE, and its inverse."""

    def __init__(self, block):
        channel = lambda key: np.broadcast_to(np.asarray(block[key], dtype=float), (6,)).copy()
        self.input_max, self.input_min = channel("input_max"), channel("input_min")
        self.output_max, self.output_min = channel("output_max"), channel("output_min")
        self.scale = np.abs(self.output_max - self.output_min) / np.abs(self.input_max - self.input_min)
        self.output_offset = (self.output_max + self.output_min) / 2
        self.input_offset = (self.input_max + self.input_min) / 2
        if not np.isfinite(self.scale).all() or np.any(self.scale <= 0) or not math.isfinite(self.scale.sum()):
            raise ValueError("OSC_POSE scale must be finite and positive")

    def to_native(self, physical):
        """The clipped normalized command whose robosuite scaling is ``physical`` (m, rad)."""
        command = (np.asarray(physical, dtype=float) - self.output_offset) / self.scale + self.input_offset
        return np.clip(command, self.input_min, self.input_max)

    def to_physical(self, native):
        """What robosuite commands for a normalized 6D pose delta (m, rad)."""
        command = np.clip(np.asarray(native, dtype=float), self.input_min, self.input_max)
        return (command - self.input_offset) * self.scale + self.output_offset
