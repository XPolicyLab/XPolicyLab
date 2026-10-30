"""LeRobot v2/v3 readers with episode-aligned full-frame history."""

from __future__ import annotations

import bisect
import json
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset

from focus_vlwa.data import head_history

try:
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
    LEROBOT_V3 = False
except ModuleNotFoundError as exc:
    if exc.name not in {"lerobot.common", "lerobot.common.datasets", "lerobot.common.datasets.lerobot_dataset"}:
        raise
    import lerobot.datasets.lerobot_dataset as lerobot_dataset
    LEROBOT_V3 = True

class _MetadataProjectedDataset:
    """Read initialization columns without converting unrelated supervision."""

    def __init__(self, dataset):
        self._dataset = dataset
        self._column_views = {}

    def query_column(self, key, indices):
        """Reuse column metadata and gather only the requested rows."""
        if key not in self._column_views:
            self._column_views[key] = self._dataset.select_columns([key])
        return self._column_views[key][list(indices)][key]

    def __len__(self):
        return len(self._dataset)

    def __getattr__(self, name):
        dataset = self.__dict__.get("_dataset")
        if dataset is None:
            raise AttributeError(name)
        return getattr(dataset, name)

    def __getitem__(self, key):
        if isinstance(key, str) and key in ("timestamp", "episode_index"):
            return self._dataset.select_columns([key]).with_format("arrow")[:].column(key)
        return self._dataset[key]


class _HeadHistoryLeRobotDataset(lerobot_dataset.LeRobotDataset):
    """Query episode-grid head history without changing actions or metadata."""

    def load_hf_dataset(self):
        return _MetadataProjectedDataset(super().load_hf_dataset())

    def _get_query_indices(self, abs_idx, ep_idx):
        queries, pads = super()._get_query_indices(abs_idx, ep_idx)
        if LEROBOT_V3:
            start = int(self.meta.episodes[ep_idx]["dataset_from_index"])
        else:
            start = int(self.episode_data_index["from"][ep_idx])
        indices, valid = head_history.history_indices(abs_idx - start)
        image_key = self._history_image_key
        queries[image_key] = [start + max(0, int(index)) for index in indices] + [abs_idx]
        pads[f"{image_key}_is_pad"] = torch.tensor([not bool(value) for value in valid] + [False])
        return queries, pads

    def _query_column(self, key, indices):
        # v3 temporal queries are absolute, even when only selected episodes are loaded.
        mapping = getattr(self, "_absolute_to_relative_idx", None)
        if mapping is not None:
            indices = [mapping[int(index)] for index in indices]
        return self.hf_dataset.query_column(key, indices)

    def _get_query_timestamps(self, current_ts, query_indices=None):
        timestamps = {}
        for key in self.meta.video_keys:
            if query_indices is not None and key in query_indices:
                column = self._query_column("timestamp", query_indices[key])
                timestamps[key] = [float(value) for value in column]
            else:
                timestamps[key] = [current_ts]
        return timestamps

    def _query_hf_dataset(self, query_indices):
        return {
            key: torch.stack(list(self._query_column(key, indices)))
            for key, indices in query_indices.items()
            if key not in self.meta.video_keys
        }


def dataset_root(repo_id):
    """Resolve both local dataset paths and cached repository identifiers."""
    path = Path(repo_id).expanduser()
    return path if (path / "meta/info.json").is_file() else Path(lerobot_dataset.HF_LEROBOT_HOME) / repo_id


def resolve_repo_ids(pattern):
    pattern = os.environ.get("FOCUS_VLWA_LEROBOT_REPOS") or pattern
    if "," in pattern:
        return [part.strip() for part in pattern.split(",") if part.strip()]
    if not pattern.endswith("*"):
        return [pattern]
    import glob
    paths = sorted(glob.glob(str(Path(pattern).expanduser())))
    if not paths:
        paths = sorted(glob.glob(str(Path(lerobot_dataset.HF_LEROBOT_HOME) / pattern)))
    result = [path for path in paths if (Path(path) / "meta/info.json").is_file()]
    if not result:
        raise FileNotFoundError(f"No LeRobot datasets match {pattern}")
    return result


class LeRobotShards(Dataset):
    """Open shards lazily and retain native or pre-chunked action semantics."""

    def __init__(self, repo_ids, model_config):
        self.repo_ids = repo_ids
        self.model_config = model_config
        self.datasets = [None] * len(repo_ids)
        self.metadata = []
        self.cumulative_lengths = []
        total = 0
        for repo_id in repo_ids:
            info = json.loads((dataset_root(repo_id) / "meta/info.json").read_text())
            version = info.get("codebase_version", "")
            if version.startswith("v3") != LEROBOT_V3:
                raise ValueError(f"Dataset {repo_id} uses {version}; install the matching LeRobot reader")
            if info["fps"] != head_history.RUNTIME_FPS:
                raise ValueError("Full head history requires original 25 FPS data")
            self.metadata.append(info)
            total += int(info["total_frames"])
            self.cumulative_lengths.append(total)

    def __len__(self):
        return self.cumulative_lengths[-1] if self.cumulative_lengths else 0

    def _open(self, shard):
        info = self.metadata[shard]
        config = self.model_config
        shape = tuple(info["features"]["action"]["shape"])
        if shape == (14,):
            delta = {"action": [step / info["fps"] for step in range(config.action_horizon)]}
        elif shape == (config.action_horizon, 14):
            delta = {}
        else:
            raise ValueError(f"Expected native [14] or pre-chunked [{config.action_horizon},14] actions, got {shape}")
        delta["observation.images.cam_high"] = head_history.history_offsets() + [0.0]
        repo_id = self.repo_ids[shard]
        kwargs = {"delta_timestamps": delta or None}
        backend = os.environ.get("FOCUS_VLWA_VIDEO_BACKEND", "pyav" if LEROBOT_V3 else "")
        if backend:
            kwargs["video_backend"] = backend
        if Path(repo_id).expanduser().is_dir():
            kwargs["root"] = dataset_root(repo_id)
            repo_id = dataset_root(repo_id).name
        dataset = _HeadHistoryLeRobotDataset(repo_id, **kwargs)
        dataset._history_image_key = "observation.images.cam_high"
        return dataset

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard = bisect.bisect_right(self.cumulative_lengths, index)
        previous = self.cumulative_lengths[shard - 1] if shard else 0
        if self.datasets[shard] is None:
            self.datasets[shard] = self._open(shard)
        sample = self.datasets[shard][index - previous]
        sample = head_history.SplitLeRobotHeadHistory("observation.images.cam_high")(sample)
        return sample
