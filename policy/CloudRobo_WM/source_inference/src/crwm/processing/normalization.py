"""Frozen native RoboTwin joint/gripper normalization in the S96/A90 layout."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


STATE_DIM = 96
ACTION_DIM = 90
STATE_INDICES = np.asarray([*range(80, 86), 9, *range(88, 94), 43])
ACTION_INDICES = np.asarray([*range(74, 80), 6, *range(82, 88), 37])


class Joint14Normalizer:
    """Read-only affine maps; grippers keep their original native units."""

    def __init__(self, config_path: str | Path):
        self.config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        if self.config.get("format") != "crwm.joint14.normalization.v1":
            raise ValueError("Unsupported CR-WM normalization format")
        self.stats_key = self.config["stats_key"]
        self._affine = {}
        for kind, dim, indices in (
            ("state", STATE_DIM, STATE_INDICES),
            ("action", ACTION_DIM, ACTION_INDICES),
        ):
            item = self.config[kind]
            mean, std = (np.asarray(item[name], dtype=np.float32) for name in ("mean", "std"))
            if (mean.shape != (14,) or std.shape != (14,)
                    or not np.isfinite(mean).all() or not np.isfinite(std).all()
                    or (std < 0).any()):
                raise ValueError(f"Invalid frozen joint14 {kind} statistics")
            # Match the original SingleFieldLinearNormalizer's FP32 operation
            # order, including the epsilon even for nonconstant dimensions.
            denom = std + np.float32(1e-8)
            scale = np.ones(dim, dtype=np.float32)
            offset = np.zeros(dim, dtype=np.float32)
            scale[indices] = np.float32(1) / denom
            offset[indices] = -mean / denom
            known = np.zeros(dim, dtype=bool)
            known[indices] = True
            self._affine[kind] = (scale, offset, known)
        self.action_valid = self._affine["action"][2].copy()

    def validate_checkpoint(self, contract: dict) -> int:
        """Reject a different normalizer or native joint order before loading GPU weights."""
        groups = contract["statistics"]["groups"]
        if set(groups) != {self.stats_key}:
            raise ValueError("Checkpoint must contain exactly the native RoboTwin statistics group")
        group = groups[self.stats_key]
        if group.get("mode") != "z-score":
            raise ValueError("Checkpoint must use native joint14 z-score normalization")
        if group.get("native_stats_sha256") != self.config["source_statistics_sha256"]:
            raise ValueError("Checkpoint and CR-WM normalization fingerprints differ")
        for kind, indices in (("state", STATE_INDICES), ("action", ACTION_INDICES)):
            if group.get("native_indices", {}).get(kind) != indices.tolist():
                raise ValueError(f"Checkpoint native {kind} joint order differs")
            known = np.asarray(group[kind]["count"]) > 0
            if not np.array_equal(known, self._affine[kind][2]):
                raise ValueError(f"Checkpoint must enable exactly the native 14 {kind} slots")
            for name in ("mean", "std"):
                saved = np.asarray(group[kind][name], dtype=np.float32)[indices]
                expected = np.asarray(self.config[kind][name], dtype=np.float32)
                if not np.array_equal(saved, expected):
                    raise ValueError(f"Checkpoint and bundled {kind}/{name} statistics differ")
        choices = [int(index) for index, condition in contract["conditions"].items()
                   if condition["stats_key"] == self.stats_key
                   and condition["embodiment"] == "aloha_agilex"]
        if len(choices) != 1:
            raise ValueError("Checkpoint requires one unambiguous RoboTwin condition")
        return choices[0]

    def encode_state(self, native_state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(native_state, dtype=np.float32)
        if x.shape != (14,) or not np.isfinite(x).all():
            raise ValueError("Expected finite native 14D drive-target state")
        packed = np.zeros(STATE_DIM, dtype=np.float32)
        packed[STATE_INDICES] = x
        scale, offset, valid = self._affine["state"]
        state = np.where(valid, np.clip(packed * scale + offset, -5, 5), 0).astype(np.float32)
        return state, valid.copy()

    def decode_action(self, prediction: np.ndarray) -> np.ndarray:
        x = np.asarray(prediction, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != ACTION_DIM:
            raise ValueError(f"Expected [T,{ACTION_DIM}] action prediction, got {x.shape}")
        scale, offset, known = self._affine["action"]
        decoded = np.where(known, (x - offset) / scale, 0).astype(np.float32)
        result = decoded[:, ACTION_INDICES]
        if not np.isfinite(result).all():
            raise ValueError("CR-WM produced a nonfinite native joint command")
        return result
