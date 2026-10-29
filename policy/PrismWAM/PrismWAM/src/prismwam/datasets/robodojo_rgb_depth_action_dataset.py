"""RoboDojo RGB/Depth/joint-action samples for PrismWAM.

The RoboDojo simulation data uses the directory layout
``<task>/arx_x5/data/episode_*.hdf5`` and an HDF5 schema that differs from
RoboTwin. This adapter translates that schema onto the RoboTwin sample format.
"""

from __future__ import annotations
from collections import Counter
import os
import hashlib
import json
import re
from pathlib import Path
from typing import Any
import h5py
import numpy as np
import torch
from .robotwin_rgb_depth_action_dataset import RobotWinRGBDepthActionVideoDataset

ROBODOJO_INDEX_VERSION = "robodojo_episode_index_v1"
DEFAULT_XPOLICYLAB_PARENT = os.environ.get(
    "PRISMWAM_ROBODOJO_ROOT", "external/RoboDojo"
)
_ACTION_FIELDS = (
    "left_arm_joint_states",
    "left_ee_joint_states",
    "right_arm_joint_states",
    "right_ee_joint_states",
)


def _natural_key(path: Path) -> list[int | str]:
    return [
        int(part) if part.isdigit() else part for part in re.split("(\\d+)", path.name)
    ]


def _decode_instruction(value: Any) -> str:
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, (bytes, np.bytes_)):
        value = bytes(value).rstrip(b"\x00").decode("utf-8")
    text = str(value).strip()
    if not text:
        raise ValueError("RoboDojo instruction must be a non-empty UTF-8 string.")
    return text


class RoboDojoRGBDepthActionVideoDataset(RobotWinRGBDepthActionVideoDataset):
    """RGB/depth/action samples from RoboDojo simulation's dual-ARX5 trajectories.

    ``episode_split='full'`` trains on every episode of the index; ``'val'`` reads
    the fixed validation partition (``holdout/val.json``, 10 episodes per task)
    and requires action statistics fitted on the matching training partition.
    """

    def __init__(
        self,
        *args,
        episode_index_path: str | None = None,
        xpolicylab_parent: str = DEFAULT_XPOLICYLAB_PARENT,
        validate_index_files: bool = False,
        episode_split: str = "full",
        **kwargs,
    ) -> None:
        if episode_split not in ("full", "val"):
            raise ValueError("episode_split must be full or val")
        self.episode_split = episode_split
        self.split_metadata = None
        if episode_split == "val" and (
            episode_index_path is None or kwargs.get("max_episodes_per_dir") is not None
        ):
            raise ValueError(
                "Validation requires the complete partition index without episode truncation"
            )
        self.episode_index_path = (
            None
            if episode_index_path is None
            else Path(episode_index_path).expanduser()
        )
        self.validate_index_files = bool(validate_index_files)
        self.manifest_fingerprint: str | None = None
        super().__init__(*args, xpolicylab_parent=xpolicylab_parent, **kwargs)
        if self.episode_split == "val":
            counts = Counter((e["task"] for e in self.episodes))
            if len(counts) != self.split_metadata["expected_tasks"] or set(
                counts.values()
            ) != {10}:
                raise ValueError(
                    "Validation dataset silently dropped tasks or episodes"
                )
            payload = torch.load(
                self.action_stats_cache_path, map_location="cpu", weights_only=True
            )
            if (
                payload.get("train_manifest_fingerprint")
                != self.split_metadata["partition_fingerprints"]["train"]
            ):
                raise ValueError(
                    "Action normalization was not fitted to this training partition"
                )

    @staticmethod
    def _episode_index(path: Path) -> int:
        match = re.search("episode_(\\d+)", path.stem)
        return int(match.group(1)) if match is not None else -1

    @staticmethod
    def _depth_key(camera: str) -> str:
        return f"vision/{camera}/approximate_depths"

    @staticmethod
    def _rgb_key(camera: str) -> str:
        return f"vision/{camera}/colors"

    def _build_index(self) -> None:
        if self.episode_index_path is not None:
            self._build_index_from_manifest(self.episode_index_path)
            return
        self._scan_episode_index()

    def _append_episode(self, episode: dict[str, Any], total: int) -> int:
        task = str(episode["task"])
        phase = str(episode["embodiment"])
        if self.task_names is not None and task not in self.task_names:
            return total
        if task in self.exclude_task_names or phase not in self.phases:
            return total
        length = int(episode["length"])
        num_samples = self._num_samples_for_episode(length)
        if num_samples <= 0:
            return total
        record = {
            **episode,
            "path": self.root_dir / str(episode["relative_path"]),
            "phase": phase,
            "length": length,
            "num_samples": num_samples,
        }
        self.episodes.append(record)
        total += num_samples
        self.cumulative_sizes.append(total)
        return total

    def _build_index_from_manifest(self, index_path: Path) -> None:
        if not index_path.is_file():
            raise FileNotFoundError(f"RoboDojo episode index not found: {index_path}")
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        if payload.get("version") != ROBODOJO_INDEX_VERSION:
            raise ValueError(
                f"Unsupported RoboDojo episode index version: {payload.get('version')!r}."
            )
        expected_root = str(self.root_dir.resolve())
        actual_root = str(Path(str(payload.get("root_dir", ""))).resolve())
        if actual_root != expected_root:
            raise ValueError(
                f"RoboDojo episode index root_dir mismatch: expected {expected_root}, got {actual_root}."
            )
        episodes = payload.get("episodes")
        if not isinstance(episodes, list):
            raise ValueError("RoboDojo episode index must contain an episodes list.")
        self.manifest_fingerprint = str(payload.get("manifest_fingerprint", ""))
        if not self.manifest_fingerprint:
            raise ValueError("RoboDojo episode index lacks manifest_fingerprint.")
        if self.episode_split == "val":
            from prismwam.robodojo_split import validate_partition

            self.split_metadata = validate_partition(payload, self.episode_split)
        total = 0
        per_task_phase: dict[tuple[str, str], int] = {}
        for raw_episode in episodes:
            if not isinstance(raw_episode, dict):
                raise ValueError("RoboDojo episode index entries must be mappings.")
            task_phase = (
                str(raw_episode.get("task", "")),
                str(raw_episode.get("embodiment", "")),
            )
            seen = per_task_phase.get(task_phase, 0)
            if self.max_episodes_per_dir is not None and seen >= int(
                self.max_episodes_per_dir
            ):
                continue
            per_task_phase[task_phase] = seen + 1
            required = {
                "relative_path",
                "task",
                "embodiment",
                "episode_index",
                "length",
                "instruction",
            }
            missing = sorted(required - set(raw_episode))
            if missing:
                raise ValueError(f"RoboDojo episode index entry is missing {missing}.")
            episode = dict(raw_episode)
            episode["instruction"] = _decode_instruction(episode["instruction"])
            if self.validate_index_files:
                path = self.root_dir / str(episode["relative_path"])
                if not path.is_file():
                    raise FileNotFoundError(
                        f"Indexed RoboDojo episode is missing: {path}"
                    )
                stat = path.stat()
                if (
                    "size_bytes" in episode
                    and int(episode["size_bytes"]) != stat.st_size
                ):
                    raise ValueError(f"Indexed RoboDojo episode size changed: {path}")
            total = self._append_episode(episode, total)
        self._require_nonempty_index()

    def _scan_episode_index(self) -> None:
        total = 0
        identity = hashlib.sha256()
        for task_dir in sorted(self.root_dir.iterdir(), key=_natural_key):
            if not task_dir.is_dir() or task_dir.name.startswith("."):
                continue
            if self.task_names is not None and task_dir.name not in self.task_names:
                continue
            if task_dir.name in self.exclude_task_names:
                continue
            for embodiment in self.phases:
                data_dir = task_dir / embodiment / "data"
                if not data_dir.is_dir():
                    continue
                paths = sorted(data_dir.glob("episode_*.hdf5"), key=_natural_key)
                if self.max_episodes_per_dir is not None:
                    paths = paths[: int(self.max_episodes_per_dir)]
                for path in paths:
                    metadata = self._read_episode_metadata(path)
                    relative = path.relative_to(self.root_dir).as_posix()
                    stat = path.stat()
                    episode = {
                        "relative_path": relative,
                        "task": task_dir.name,
                        "embodiment": str(embodiment),
                        "episode_index": self._episode_index(path),
                        "length": metadata["length"],
                        "instruction": metadata["instruction"],
                        "frequency": metadata["frequency"],
                        "size_bytes": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                    }
                    identity.update(
                        json.dumps(
                            episode, sort_keys=True, separators=(",", ":")
                        ).encode("utf-8")
                    )
                    total = self._append_episode(episode, total)
        self.manifest_fingerprint = identity.hexdigest()
        self._require_nonempty_index()

    def _require_nonempty_index(self) -> None:
        if not self.episodes:
            raise ValueError(
                f"No RoboDojo HDF5 episodes found under {self.root_dir} for embodiments={self.phases}."
            )

    def _read_episode_metadata(self, path: Path) -> dict[str, Any]:
        required_sequences = [
            *(self._depth_key(camera) for camera in self.cameras),
            *(self._rgb_key(camera) for camera in self.cameras),
            *(f"action/{field}" for field in _ACTION_FIELDS),
            *(f"state/{field}" for field in _ACTION_FIELDS),
        ]
        with h5py.File(path, "r") as handle:
            missing = [key for key in required_sequences if key not in handle]
            if "instruction" not in handle:
                missing.append("instruction")
            if missing:
                raise KeyError(f"RoboDojo episode {path} is missing {missing}.")
            lengths = {key: int(handle[key].shape[0]) for key in required_sequences}
            unique_lengths = set(lengths.values())
            if len(unique_lengths) != 1:
                raise ValueError(
                    f"RoboDojo episode fields are not frame-aligned in {path}: {lengths}."
                )
            instruction = _decode_instruction(handle["instruction"][()])
            frequency = int(
                handle["additional_info/frequency"][()]
                if "additional_info/frequency" in handle
                else 25
            )
        return {
            "length": unique_lengths.pop(),
            "instruction": instruction,
            "frequency": frequency,
        }

    def _read_episode_length(self, path: Path) -> int:
        return int(self._read_episode_metadata(path)["length"])

    def _attach_instruction_candidates_to_episodes(self) -> None:
        for episode in self.episodes:
            instruction = _decode_instruction(episode.get("instruction", ""))
            episode["instruction_candidates"] = (instruction,)

    def _read_depth_window(
        self, h5_file: h5py.File, camera: str, start: int, length: int
    ) -> torch.Tensor:
        depth_ds = h5_file[self._depth_key(camera)]
        stop = min(int(start) + self.window_span, int(length))
        depth = np.asarray(
            depth_ds[int(start) : stop : self.frame_stride], dtype=np.float32
        )
        if depth.shape[0] == 0:
            raise ValueError(
                f"Empty RoboDojo depth window: camera={camera}, start={start}."
            )
        if depth.shape[0] < self.num_video_frames:
            depth = np.concatenate(
                (
                    depth,
                    np.repeat(
                        depth[-1:], self.num_video_frames - depth.shape[0], axis=0
                    ),
                ),
                axis=0,
            )
        return self._normalize_depth(depth)

    def _read_rgb_frames(
        self, h5_file: h5py.File, camera: str, indices: list[int]
    ) -> torch.Tensor:
        dataset = h5_file[self._rgb_key(camera)]
        frames = [self._decode_rgb_frame(dataset[index]) for index in indices]
        video = np.stack(frames, axis=0).astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(video).permute(0, 3, 1, 2).contiguous()

    def _read_rgb_video(self, episode: dict[str, Any], start: int) -> torch.Tensor:
        length = int(episode["length"])
        indices = list(
            range(
                int(start),
                min(int(start) + self.window_span, length),
                self.frame_stride,
            )
        )
        if not indices:
            raise ValueError(f"Empty RoboDojo RGB window at start={start}.")
        indices.extend([indices[-1]] * (self.num_video_frames - len(indices)))
        with h5py.File(episode["path"], "r") as handle:
            camera_videos = [
                self._read_rgb_frames(handle, camera, indices)
                for camera in self.cameras
            ]
        return self._layout_video(camera_videos).permute(1, 0, 2, 3).contiguous()

    def _read_rgb_history(self, episode: dict[str, Any], start: int) -> torch.Tensor:
        length = int(episode["length"])
        indices = self.rgb_history.indices(start, last=length - 1)
        with h5py.File(episode["path"], "r") as handle:
            camera_videos = [
                self._read_rgb_frames(handle, camera, indices)
                for camera in self.cameras
            ]
        return self._layout_video(camera_videos).permute(1, 0, 2, 3).contiguous()

    @staticmethod
    def _joint_vector(
        handle: h5py.File, prefix: str, selection: slice = slice(None)
    ) -> np.ndarray:
        fields = [
            np.asarray(handle[f"{prefix}/{field}"][selection], dtype=np.float32)
            for field in _ACTION_FIELDS
        ]
        vector = np.concatenate(fields, axis=-1)
        if vector.ndim != 2 or vector.shape[1] != 14:
            raise ValueError(
                f"RoboDojo {prefix} joint vector must be [T,14], got {vector.shape}."
            )
        for gripper_index in (6, 13):
            gripper = vector[:, gripper_index]
            if np.any(gripper < -1e-06) or np.any(gripper > 1.0 + 1e-06):
                raise ValueError(f"RoboDojo {prefix} gripper must lie in [0,1].")
            vector[:, gripper_index] = np.clip(gripper, 0.0, 1.0)
        return vector

    @staticmethod
    def _vector_prefix(key: str) -> str:
        if key == "action/joint_vector":
            return "action"
        if key == "state/joint_vector":
            return "state"
        raise KeyError(
            f"RoboDojo virtual vector key must be action/joint_vector or state/joint_vector, got {key!r}."
        )

    def _read_vector_window(
        self,
        h5_file: h5py.File,
        key: str,
        start: int,
        length: int,
        start_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prefix = self._vector_prefix(key)
        source_length = min(
            (int(h5_file[f"{prefix}/{field}"].shape[0]) for field in _ACTION_FIELDS)
        )
        sequence_length = min(int(length), source_length)
        read_start = int(start) + int(start_offset)
        if sequence_length <= 0:
            raise ValueError("Cannot read a RoboDojo vector from an empty episode.")
        if read_start < sequence_length:
            values = self._joint_vector(
                h5_file,
                prefix,
                slice(
                    read_start, min(read_start + self.action_horizon, sequence_length)
                ),
            )
        else:
            values = self._joint_vector(
                h5_file, prefix, slice(sequence_length - 1, sequence_length)
            )
        pad_mask = torch.tensor(
            [
                read_start + offset >= sequence_length
                for offset in range(self.action_horizon)
            ],
            dtype=torch.bool,
        )
        if values.shape[0] < self.action_horizon:
            values = np.concatenate(
                (
                    values,
                    np.repeat(
                        values[-1:], self.action_horizon - values.shape[0], axis=0
                    ),
                ),
                axis=0,
            )
        tensor = torch.from_numpy(values[: self.action_horizon]).contiguous()
        if self.normalize_actions:
            tensor = (tensor - self.action_mean) / self.action_std
        return (tensor, pad_mask)

    def _compute_action_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        count = 0
        total = np.zeros((self.action_dim,), dtype=np.float64)
        total_sq = np.zeros((self.action_dim,), dtype=np.float64)
        for episode in self.episodes:
            with h5py.File(episode["path"], "r") as handle:
                action = self._joint_vector(handle, "action").astype(
                    np.float64, copy=False
                )
            total += action.sum(axis=0)
            total_sq += np.square(action).sum(axis=0)
            count += int(action.shape[0])
        if count <= 0:
            raise ValueError("Cannot compute RoboDojo stats from an empty index.")
        mean = total / count
        variance = np.maximum(total_sq / count - np.square(mean), 0.0)
        std = np.maximum(np.sqrt(variance), self.action_std_floor)
        return (
            torch.from_numpy(mean.astype(np.float32)),
            torch.from_numpy(std.astype(np.float32)),
        )
