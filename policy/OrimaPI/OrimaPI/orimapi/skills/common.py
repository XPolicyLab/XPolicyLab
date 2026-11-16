# coding=utf-8
"""Camera decoding, explicit robot layouts, and observation lifecycle checks."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

from ..state import Observation, RGBImage

if TYPE_CHECKING:
    from ..policy import PolicyConfig


def numpy_module():
    try:
        import numpy as np
    except ImportError:
        raise RuntimeError(
            "Install skill dependencies: pip install -e '.[skills]'"
        ) from None
    return np


def encode_observation(observation: Observation, config: PolicyConfig) -> dict:
    np = numpy_module()
    try:
        from PIL import ImageFile
    except ImportError:
        raise RuntimeError(
            "Install skill dependencies: pip install -e '.[skills]'"
        ) from None
    images = {}
    for model_name, camera_name in config.camera_map.items():
        if camera_name not in observation.images:
            raise ValueError(f"Missing observed camera: {camera_name}")
        raw = observation.images[camera_name]
        if isinstance(raw, RGBImage):
            images[model_name] = (
                np.asarray(memoryview(raw.pixels), dtype=np.uint8)
                .reshape(raw.height, raw.width, 3)
                .transpose(2, 0, 1)
                .copy()
            )
            continue
        # Observation images use standard RGB PNG/JPEG. Legacy XPolicyLab byte
        # streams must be decoded/re-encoded correctly by its future adapter.
        parser = ImageFile.Parser()
        parser.feed(observation.images[camera_name])
        with parser.close() as image:
            images[model_name] = (
                np.asarray(image.convert("RGB"), dtype=np.uint8)
                .transpose(2, 0, 1)
                .copy()
            )
    state = observation.robot_state
    if isinstance(state, dict):
        parts = []
        for name, width in zip(config.state_fields, config.state_dims, strict=True):
            if name not in state:
                raise ValueError(f"Missing robot-state field: {name}")
            part = np.asarray(state[name], dtype=np.float32)
            if part.shape != (width,):
                raise ValueError(
                    f"State {name} must have shape ({width},), got {part.shape}"
                )
            parts.append(part)
        packed = np.concatenate(parts)
    else:
        packed = np.asarray(state, dtype=np.float32)
    expected = sum(config.state_dims)
    if packed.shape != (expected,) or not np.isfinite(packed).all():
        raise ValueError(f"Robot state must be a finite vector of dimension {expected}")
    return {"images": images, "state": packed.copy()}


def action_limit(max_actions: int | None) -> int | None:
    if max_actions is not None and (type(max_actions) is not int or max_actions < 1):
        raise ValueError("max_actions must be a positive integer")
    return max_actions


def decode_actions(
    actions, config: PolicyConfig, max_actions: int | None = None
) -> list[dict]:
    np = numpy_module()
    action_limit(max_actions)
    values = np.asarray(actions, dtype=np.float32)
    width = sum(config.action_dims)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] != width:
        raise ValueError(
            f"Expected action shape (horizon, {width}), got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("Action values must be finite")
    rows = []
    for vector in values[:max_actions]:
        row, offset = {}, 0
        for name, size in zip(config.action_fields, config.action_dims, strict=True):
            row[name] = vector[offset : offset + size].tolist()
            offset += size
        rows.append(row)
    return rows


def validate_code_actions(
    actions, config: PolicyConfig, max_actions: int
) -> list[dict]:
    np = numpy_module()
    action_limit(max_actions)
    if not isinstance(actions, list) or not 1 <= len(actions) <= max_actions:
        raise ValueError(
            "Code policy must return 1..max_actions actions; its sequence cannot be silently truncated"
        )
    packed = []
    for row in actions:
        if not isinstance(row, dict) or set(row) != set(config.action_fields):
            raise ValueError(
                "Code action fields differ from the configured action layout"
            )
        parts = []
        for name, size in zip(config.action_fields, config.action_dims, strict=True):
            value = np.asarray(row[name], dtype=np.float32)
            if value.shape != (size,):
                raise ValueError(f"Code action {name} must have dimension {size}")
            parts.append(value)
        packed.append(np.concatenate(parts))
    return decode_actions(packed, config, max_actions)


def is_new_observation(previous: Observation | None, observation: Observation) -> bool:
    """Validate ordering against the caller's last owned observation."""
    if previous is None:
        return True
    if observation.instruction != previous.instruction:
        raise ValueError("Task instruction changed; reset the skill first")
    if observation.step < previous.step:
        raise ValueError("Observation step moved backwards; reset the skill first")
    if observation.step == previous.step:
        if observation != previous:
            raise ValueError("The same observation step has different contents")
        return False
    return True
