"""MoPA RGB episode dataset and action-chunk sampling."""
from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from ..common import DATASET_FORMAT, normalize, validate_statistics
from .prepare import validate_contract


class EpisodeDataset(Dataset):
    """Every real frame starts a chunk, repeating the final action at episode ends."""

    def __init__(self, root: str | Path, action_horizon: int = 4, cache_episodes: int = 2):
        self.root = Path(root)
        self.metadata = json.loads((self.root / "metadata.json").read_text(encoding="utf-8"))
        self.statistics = json.loads((self.root / "dataset_statistics.json").read_text(encoding="utf-8"))
        if self.metadata.get("format_version") != DATASET_FORMAT:
            raise ValueError("Unsupported dataset format; prepare it with mopa.data.prepare.")
        self.dim = validate_contract(self.metadata)
        if self.metadata.get("state_dim") != self.dim or self.metadata.get("action_dim") != self.dim:
            raise ValueError("Dataset dimensions disagree with robot metadata.")
        for kind in ("state", "action"):
            validate_statistics(self.statistics[kind], self.dim)
        self.cameras = list(self.metadata["cameras"])
        self.episodes = self.metadata["episodes"]
        lengths = [record["length"] for record in self.episodes]
        if not lengths or any(not isinstance(length, int) or isinstance(length, bool) or length <= 0 for length in lengths):
            raise ValueError("Dataset must contain nonempty episodes.")
        self.ends = np.cumsum(lengths)
        if int(self.ends[-1]) != self.metadata["num_frames"]:
            raise ValueError("Dataset frame count does not match episode lengths.")
        if action_horizon <= 0 or cache_episodes <= 0:
            raise ValueError("Action horizon and episode cache size must be positive.")
        self.action_horizon = action_horizon
        self.cache_episodes = cache_episodes
        self._cache = OrderedDict()

    def __len__(self):
        return int(self.ends[-1])

    def _episode(self, index: int) -> dict:
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]
        record = self.episodes[index]
        path = self.root / record["file"]
        with np.load(path, allow_pickle=False) as archive:
            episode = {key: archive[key] for key in archive.files}
        expected = (record["length"], self.dim)
        for kind in ("state", "action"):
            if episode[kind].shape != expected or not np.isfinite(episode[kind]).all():
                raise ValueError(f"{path}: invalid {kind} data, expected finite {expected}.")
        image_shape = (record["length"], *self.metadata["image_size"], 3)
        for camera_index in range(len(self.cameras)):
            image = episode[f"image_{camera_index}"]
            if image.shape != image_shape or image.dtype != np.uint8:
                raise ValueError(f"{path}: images must be uint8 RGB {image_shape}.")
        instruction = episode["instruction"]
        if instruction.ndim != 0 or instruction.dtype.kind not in ("U", "S"):
            raise ValueError(f"{path}: expected a scalar instruction string.")
        instruction = instruction.item()
        if isinstance(instruction, bytes):
            instruction = instruction.decode("utf-8")
        if not instruction.strip():
            raise ValueError(f"{path}: expected a nonempty instruction.")
        episode["instruction"] = np.asarray(instruction)
        self._cache[index] = episode
        while len(self._cache) > self.cache_episodes:
            self._cache.popitem(last=False)
        return episode

    def __getitem__(self, index: int) -> dict:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode_index = int(np.searchsorted(self.ends, index, side="right"))
        offset = index - (int(self.ends[episode_index - 1]) if episode_index else 0)
        episode = self._episode(episode_index)
        action_indices = np.minimum(np.arange(offset, offset + self.action_horizon), len(episode["action"]) - 1)
        return {
            "images": [episode[f"image_{i}"][offset] for i in range(len(self.cameras))],
            "instruction": str(episode["instruction"].item()),
            "state": normalize(episode["state"][offset], self.statistics["state"]),
            "actions": normalize(episode["action"][action_indices], self.statistics["action"]),
        }
