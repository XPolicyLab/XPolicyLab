"""Checkpoint discovery and weight loading for Soma inference."""

import logging
import os
import re

import megfile
import safetensors.torch
import torch
from safetensors import safe_open

from soma.inference.runtime.artifact_cache import cache_locally

logger = logging.getLogger(__name__)

_DCP_STEP_PATTERN = re.compile(r"(?:^|/)step-(\d+)/?$")


def _read_checkpoint(
    path: str,
    names: set[str] | None = None,
) -> dict[str, torch.Tensor]:
    if path.endswith((".pt", ".pth")):
        with megfile.smart_open(path, "rb") as source:
            state_dict = torch.load(source, map_location="cpu", weights_only=True)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if not isinstance(state_dict, dict):
            raise TypeError("checkpoint state_dict must be a mapping, got " f"{type(state_dict).__name__}")
        if names is None:
            return state_dict
        return {name: value for name, value in state_dict.items() if name in names}
    if "://" in path:
        with megfile.smart_open(path, "rb") as source:
            state_dict = safetensors.torch.load(source.read())
        if names is None:
            return state_dict
        return {name: value for name, value in state_dict.items() if name in names}
    with safe_open(path, framework="pt") as checkpoint:
        checkpoint_names = checkpoint.keys()
        if names is not None:
            checkpoint_names = [name for name in checkpoint_names if name in names]
        return {name: checkpoint.get_tensor(name) for name in checkpoint_names}


def resolve_checkpoint_path(path: str) -> str:
    """Resolve a Falcon experiment root to its latest DCP step."""
    if _DCP_STEP_PATTERN.search(path):
        return path.rstrip("/")
    if "://" in path:
        from megfile import smart_exists, smart_open

        tracker = f"{path.rstrip('/')}/ckpts/latest_checkpointed_iteration.txt"
        if not smart_exists(tracker):
            return path
        with smart_open(tracker) as tracker_file:
            step = int(tracker_file.read().strip())
        return f"{path.rstrip('/')}/step-{step}"
    tracker = os.path.join(path, "ckpts", "latest_checkpointed_iteration.txt")
    if os.path.isfile(tracker):
        with open(tracker) as tracker_file:
            step = int(tracker_file.read().strip())
        return os.path.join(path, "ckpts", f"step-{step}")
    return path


def is_dcp_checkpoint(path: str) -> bool:
    """Return whether a path identifies a distributed checkpoint step."""
    if "://" in path:
        return _DCP_STEP_PATTERN.search(path) is not None
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, ".metadata"))


def load_model_weights(
    model: torch.nn.Module,
    checkpoint_path: str,
    names: set[str] | None = None,
) -> None:
    """Load all or a selected subset of model weights."""
    checkpoint_path = resolve_checkpoint_path(checkpoint_path)
    if is_dcp_checkpoint(checkpoint_path):
        raise ValueError(
            "this deployment loads safetensors checkpoints only; "
            f"got a distributed checkpoint at {checkpoint_path}"
        )
    if not os.path.isfile(checkpoint_path):
        checkpoint_path = cache_locally(checkpoint_path)
    incompatible = model.load_state_dict(
        _read_checkpoint(checkpoint_path, names),
        strict=False,
    )
    missing = [name for name in incompatible.missing_keys if names is None or name in names]
    if missing:
        raise RuntimeError(
            "Falcon Motubrain checkpoint is incomplete; missing " f"{len(missing)} model weights: {missing[:8]}"
        )
    if incompatible.unexpected_keys:
        logger.warning("Unexpected model keys: %s", incompatible.unexpected_keys)
