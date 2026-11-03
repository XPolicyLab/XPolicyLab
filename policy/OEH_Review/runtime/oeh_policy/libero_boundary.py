"""Pinned LIBERO-family OpenWAM wire boundary, independent of benchmark judgment.

The official clients construct their RGB/EEF10 payload before this boundary.
Only this projection is handed to the real OpenWAM PolicyServer.predict method.
The official client remains responsible for EEF10 -> 7D OSC conversion.
"""
from __future__ import annotations

import base64
import copy
import importlib.util
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

from .contracts import file_digest

CAMERA_KEYS = ("head_camera", "left_wrist_camera", "right_wrist_camera")
# Benchmark id (campaign and config naming) -> pinned OpenWAM benchmarks/ folder. OpenWAM ships no LIBERO-Pro folder:
# LIBERO-Pro's perturbed suites run through the LIBERO loop (oeh_eval.libero).
BENCHMARKS = {"libero": "libero", "libero_plus": "libero-plus", "libero_pro": "libero"}


def project_wire_observation(value: Mapping[str, Any], *, prompt: str | None = None) -> dict:
    """Copy the exact released client payload; discard evaluator-only fields.

    ``prompt`` may be replaced only by the separately frozen slow condition.
    Images stay as the original base64 PNG strings so the official server owns
    decoding, composition, crop and normalization for both conditions.
    """
    if not isinstance(value, Mapping) or not isinstance(value.get("images"), Mapping):
        raise ValueError("OpenWAM LIBERO images payload required")
    source = value["images"]
    images = {}
    for key in CAMERA_KEYS:
        image = source.get(key)
        if image is None and key != "head_camera":
            images[key] = None
            continue
        if not isinstance(image, str) or not image:
            raise ValueError(f"invalid {key} image")
        # Reject malformed transport before an OEH admission is requested.
        try:
            if not base64.b64decode(image, validate=True):
                raise ValueError(f"empty {key} image")
        except (ValueError, base64.binascii.Error) as exc:
            raise ValueError(f"invalid {key} base64 image") from exc
        images[key] = image
    text = value.get("prompt") if prompt is None else prompt
    if not isinstance(text, str) or not text.strip():
        raise ValueError("LIBERO task prompt required")
    raw_state = value.get("state")
    state = np.asarray(raw_state, dtype=np.float32)
    if state.shape != (10,) or not np.isfinite(state).all():
        raise ValueError("LIBERO achieved EEF10 state required")
    return {"type": "obs", "images": images, "prompt": text, "state": state.tolist()}


def validate_raw_action(value: Any) -> list[float]:
    """Validate released checkpoint's raw EEF10, without changing its values."""
    action = np.asarray(value, dtype=np.float32)
    if action.shape != (10,) or not np.isfinite(action).all():
        raise ValueError("OpenWAM LIBERO raw action must be finite EEF10")
    return action.tolist()


def validate_native_action(value: Any) -> list[float]:
    """Validate the official client's 7D OSC action after its conversion."""
    action = np.asarray(value, dtype=np.float32)
    if action.shape != (7,) or not np.isfinite(action).all():
        raise ValueError("LIBERO native action must be finite 7D OSC")
    if np.any(np.abs(action) > 1.000001):
        raise ValueError("LIBERO native action outside official [-1, 1] limits")
    return action.tolist()


def load_official_conversion(source: Path, benchmark: str, expected_sha256: str):
    """Load the pinned released action conversion rather than copying its math."""
    folder = BENCHMARKS.get(benchmark)
    if folder is None:
        raise ValueError(f"unsupported LIBERO benchmark {benchmark!r}")
    source = source.resolve()
    path = source / "benchmarks" / folder / "openwam2libero_interface.py"
    if file_digest(path) != expected_sha256:
        raise ValueError("official LIBERO action conversion identity mismatch")
    # benchmarks.utils comes from the same pinned source. Prevent a different
    # installed package from satisfying that import accidentally.
    for name in ("benchmarks", "benchmarks.utils"):
        loaded = sys.modules.get(name)
        if loaded is not None and not Path(loaded.__file__).resolve().is_relative_to(source):
            raise ValueError("another benchmarks package is already imported")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location(f"oeh_pinned_{folder.replace('-', '_')}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load pinned official LIBERO adapter")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.native_eef10_to_libero7d


class OfficialModelBoundary:
    """Last filter before the actual OpenWAM PolicyServer.predict call.

    A sidecar alone cannot establish input isolation. This object is installed
    in the serving process and owns the only call into the released model.
    """

    def __init__(self, official_server):
        self.official_server = official_server
        self.last_model_input = None

    def reset(self):
        self.official_server.reset()

    def predict(self, wire_observation: Mapping[str, Any], *, slow_prompt: str | None = None) -> dict:
        model_input = project_wire_observation(wire_observation, prompt=slow_prompt)
        self.last_model_input = copy.deepcopy(model_input)
        result = self.official_server.predict(model_input)
        if not isinstance(result, Mapping):
            raise ValueError("official OpenWAM server returned no action mapping")
        action = validate_raw_action(result.get("action"))
        latency = result.get("latency_ms")
        if latency is not None and (isinstance(latency, bool) or
                                    not isinstance(latency, (int, float)) or
                                    not math.isfinite(latency) or latency < 0):
            raise ValueError("invalid official model latency")
        return {"type": "action", "action": action, "step": result.get("step"),
                "latency_ms": latency}
