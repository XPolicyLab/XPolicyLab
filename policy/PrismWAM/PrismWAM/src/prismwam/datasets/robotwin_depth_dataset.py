import bisect
import re
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import torch
import torch.nn.functional as F


def _natural_key(path: Path):
    return [
        int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)
    ]


class RobotWinDepthVideoDataset(torch.utils.data.Dataset):
    """Depth video windows from raw RoboTwin HDF5 episodes."""

    def __init__(
        self,
        root_dir: str,
        phases: Sequence[str] = ("demo_clean", "demo_randomized"),
        cameras: Sequence[str] = ("head_camera", "left_camera", "right_camera"),
        task_names: Sequence[str] | None = None,
        exclude_task_names: Sequence[str] | None = None,
        num_video_frames: int = 33,
        frame_stride: int = 1,
        sample_stride: int = 1,
        drop_incomplete_windows: bool = True,
        video_size: Sequence[int] = (384, 320),
        layout: str = "robotwin",
        depth_min: float = 0.0,
        depth_max: float = 1500.0,
        max_episodes_per_dir: int | None = None,
    ):
        self.root_dir = Path(root_dir).expanduser()
        self.phases = tuple(phases)
        self.cameras = tuple(cameras)
        self.task_names = None if task_names is None else set(task_names)
        self.exclude_task_names = set(exclude_task_names or ())
        if self.task_names is not None:
            overlap = self.task_names.intersection(self.exclude_task_names)
            if overlap:
                raise ValueError(
                    "RoboTwin task_names and exclude_task_names overlap: "
                    f"{sorted(overlap)}."
                )
        self.num_video_frames = int(num_video_frames)
        self.frame_stride = int(frame_stride)
        self.sample_stride = int(sample_stride)
        self.drop_incomplete_windows = bool(drop_incomplete_windows)
        self.video_size = (int(video_size[0]), int(video_size[1]))
        self.layout = str(layout)
        self.depth_min = float(depth_min)
        self.depth_max = float(depth_max)
        self.max_episodes_per_dir = max_episodes_per_dir

        if not self.root_dir.exists():
            raise FileNotFoundError(f"RoboTwin root_dir not found: {self.root_dir}")
        if self.num_video_frames <= 1:
            raise ValueError(
                f"`num_video_frames` must be > 1, got {self.num_video_frames}"
            )
        if self.frame_stride <= 0:
            raise ValueError(
                f"`frame_stride` must be positive, got {self.frame_stride}"
            )
        self.frame_offsets = [
            i * self.frame_stride for i in range(self.num_video_frames)
        ]
        self.window_span = self.frame_offsets[-1] + 1
        if self.num_video_frames % 4 != 1:
            raise ValueError(
                "Depth VAE video length must satisfy T % 4 == 1 for WAN VAE tokenization, "
                f"got T={self.num_video_frames}."
            )
        if self.depth_max <= self.depth_min:
            raise ValueError(
                f"`depth_max` must be greater than `depth_min`, got {self.depth_min}, {self.depth_max}"
            )
        if self.sample_stride <= 0:
            raise ValueError(
                f"`sample_stride` must be positive, got {self.sample_stride}"
            )

        self.episodes: list[dict] = []
        self.cumulative_sizes: list[int] = []
        self._build_index()

    def _num_samples_for_episode(self, length: int) -> int:
        if self.drop_incomplete_windows:
            if length < self.window_span:
                return 0
            return (length - self.window_span) // self.sample_stride + 1
        return (length + self.sample_stride - 1) // self.sample_stride

    def _build_index(self):
        total = 0
        for task_dir in sorted(self.root_dir.iterdir(), key=_natural_key):
            if not task_dir.is_dir() or task_dir.name.startswith("."):
                continue
            if self.task_names is not None and task_dir.name not in self.task_names:
                continue
            if task_dir.name in self.exclude_task_names:
                continue
            for phase in self.phases:
                data_dir = task_dir / phase / "data"
                if not data_dir.exists():
                    continue
                episode_paths = sorted(data_dir.glob("episode*.hdf5"), key=_natural_key)
                if self.max_episodes_per_dir is not None:
                    episode_paths = episode_paths[: int(self.max_episodes_per_dir)]
                for path in episode_paths:
                    length = self._read_episode_length(path)
                    if length <= 0:
                        continue
                    num_samples = self._num_samples_for_episode(length)
                    if num_samples <= 0:
                        continue
                    episode_index = self._episode_index(path)
                    self.episodes.append(
                        {
                            "path": path,
                            "task": task_dir.name,
                            "phase": phase,
                            "episode_index": episode_index,
                            "length": length,
                            "num_samples": num_samples,
                        }
                    )
                    total += num_samples
                    self.cumulative_sizes.append(total)
        if not self.episodes:
            raise ValueError(
                f"No RoboTwin HDF5 episodes found under {self.root_dir} for phases={self.phases}."
            )

    def _read_episode_length(self, path: Path) -> int:
        key = f"observation/{self.cameras[0]}/depth"
        with h5py.File(path, "r") as f:
            if key not in f:
                raise KeyError(f"Missing depth dataset `{key}` in {path}")
            return int(f[key].shape[0])

    @staticmethod
    def _episode_index(path: Path) -> int:
        match = re.search(r"episode(\d+)", path.stem)
        return int(match.group(1)) if match is not None else -1

    def __len__(self):
        return self.cumulative_sizes[-1]

    def task_index_groups(self) -> dict[str, list[int]]:
        """Return the exact dataset-index ownership of every active task."""
        groups: dict[str, list[int]] = {}
        previous = 0
        for episode, cumulative in zip(self.episodes, self.cumulative_sizes):
            task = str(episode["task"])
            groups.setdefault(task, []).extend(range(previous, int(cumulative)))
            previous = int(cumulative)
        if previous != len(self):
            raise RuntimeError(
                "RoboTwin task index groups do not cover the dataset length: "
                f"covered={previous}, length={len(self)}."
            )
        return groups

    def _locate(self, idx: int):
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"Index {idx} out of bounds for dataset of length {len(self)}"
            )
        episode_pos = bisect.bisect_right(self.cumulative_sizes, idx)
        prev = 0 if episode_pos == 0 else self.cumulative_sizes[episode_pos - 1]
        local_idx = idx - prev
        episode = self.episodes[episode_pos]
        start = int(local_idx * self.sample_stride)
        return episode, start

    def _normalize_depth(self, depth: np.ndarray) -> torch.Tensor:
        depth = np.asarray(depth, dtype=np.float32)
        depth = np.nan_to_num(
            depth,
            nan=self.depth_max,
            posinf=self.depth_max,
            neginf=self.depth_min,
        )
        depth = np.clip(depth, self.depth_min, self.depth_max)
        depth = (depth - self.depth_min) / (self.depth_max - self.depth_min)
        depth = depth * 2.0 - 1.0
        return torch.from_numpy(depth).unsqueeze(1).contiguous()

    def _read_depth_window(
        self, h5_file: h5py.File, camera: str, start: int, length: int
    ) -> torch.Tensor:
        depth_ds = h5_file[f"observation/{camera}/depth"]
        stop = min(start + self.window_span, length)
        depth = depth_ds[start : stop : self.frame_stride]
        if depth.shape[0] == 0:
            raise ValueError(
                f"Empty depth window for camera={camera}, start={start}, length={length}"
            )
        if depth.shape[0] < self.num_video_frames:
            pad = np.repeat(depth[-1:], self.num_video_frames - depth.shape[0], axis=0)
            depth = np.concatenate([depth, pad], axis=0)
        return self._normalize_depth(depth)

    @staticmethod
    def _resize(video: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(video, size=size, mode="bilinear", align_corners=False)

    def _layout_video(self, camera_videos: list[torch.Tensor]) -> torch.Tensor:
        height, width = self.video_size
        if self.layout == "robotwin":
            if len(camera_videos) != 3:
                raise ValueError(
                    "`layout=robotwin` expects exactly three cameras: head, left, right."
                )
            top_h = int(round(height * 2.0 / 3.0))
            bottom_h = height - top_h
            left_w = width // 2
            right_w = width - left_w
            head = self._resize(camera_videos[0], (top_h, width))
            left = self._resize(camera_videos[1], (bottom_h, left_w))
            right = self._resize(camera_videos[2], (bottom_h, right_w))
            return torch.cat([head, torch.cat([left, right], dim=-1)], dim=-2)
        if self.layout == "horizontal":
            per_camera_width = width // len(camera_videos)
            resized = [
                self._resize(
                    video,
                    (
                        height,
                        per_camera_width
                        if i < len(camera_videos) - 1
                        else width - per_camera_width * i,
                    ),
                )
                for i, video in enumerate(camera_videos)
            ]
            return torch.cat(resized, dim=-1)
        if self.layout == "single":
            if len(camera_videos) != 1:
                raise ValueError("`layout=single` expects exactly one camera.")
            return self._resize(camera_videos[0], (height, width))
        raise ValueError(f"Unsupported depth video layout: {self.layout}")

    def __getitem__(self, idx: int):
        episode, start = self._locate(idx)
        length = int(episode["length"])
        with h5py.File(episode["path"], "r") as f:
            camera_videos = [
                self._read_depth_window(f, camera=camera, start=start, length=length)
                for camera in self.cameras
            ]
        video_tchw = self._layout_video(camera_videos)
        image_is_pad = torch.tensor(
            [start + offset >= length for offset in self.frame_offsets],
            dtype=torch.bool,
        )
        return {
            "video": video_tchw.permute(1, 0, 2, 3).contiguous(),
            "image_is_pad": image_is_pad,
            "idx": int(idx),
            "task": episode["task"],
            "phase": episode["phase"],
            "episode_index": int(episode["episode_index"]),
            "start_index": int(start),
        }
