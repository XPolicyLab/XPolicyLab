import bisect
import hashlib
import os
import re
from pathlib import Path
from typing import Sequence
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from .robotwin_instructions import RobotWinRawInstructionMixin
from .xpolicylab_images import decode_rgb_frame

PROMPT_TEMPLATE = "A video recorded from a robot's point of view executing the following instruction: {task}"
# RoboTwin checkout that contains XPolicyLab/, whose helper decodes the RGB frames.
DEFAULT_ROBOTWIN_ROOT = os.environ.get("PRISMWAM_ROBOTWIN_ROOT", "external/RoboTwin")


def _natural_key(path: Path):
    return [
        int(part) if part.isdigit() else part for part in re.split("(\\d+)", path.name)
    ]


class RobotWinRGBVideoDataset(RobotWinRawInstructionMixin, torch.utils.data.Dataset):
    """RGB video windows from raw RoboTwin HDF5 episodes."""

    def __init__(
        self,
        root_dir: str,
        phases: Sequence[str] = ("demo_clean", "demo_randomized"),
        cameras: Sequence[str] = ("head_camera", "left_camera", "right_camera"),
        task_names: Sequence[str] | None = None,
        num_video_frames: int = 9,
        frame_stride: int = 4,
        sample_stride: int = 1,
        drop_incomplete_windows: bool = True,
        video_size: Sequence[int] = (384, 320),
        layout: str = "robotwin",
        max_episodes_per_dir: int | None = None,
        prompt_template: str = PROMPT_TEMPLATE,
        text_embedding_cache_dir: str | None = None,
        context_len: int = 128,
        text_encoder_id: str = "wan22ti2v5b",
        humanize_task_name: bool = False,
        instruction_source: str = "task",
        instruction_type: str = "seen",
        instruction_selection: str = "episode_hash",
        task_instruction_dir: str | None = None,
        objects_description_dir: str | None = None,
        max_instructions_per_episode: int | None = None,
        dataset_uid: str | None = None,
        prompt_policy_id: str = "raw_robotwin_prompt",
        xpolicylab_parent: str = DEFAULT_ROBOTWIN_ROOT,
    ):
        self.xpolicylab_parent = str(Path(xpolicylab_parent).expanduser())
        self.root_dir = Path(root_dir).expanduser()
        self.phases = tuple(phases)
        self.cameras = tuple(cameras)
        self.task_names = None if task_names is None else set(task_names)
        self.num_video_frames = int(num_video_frames)
        self.frame_stride = int(frame_stride)
        self.sample_stride = int(sample_stride)
        self.drop_incomplete_windows = bool(drop_incomplete_windows)
        self.video_size = (int(video_size[0]), int(video_size[1]))
        self.layout = str(layout)
        self.max_episodes_per_dir = max_episodes_per_dir
        self.text_embedding_cache_dir = (
            None if text_embedding_cache_dir is None else Path(text_embedding_cache_dir)
        )
        self.context_len = int(context_len)
        self.text_encoder_id = str(text_encoder_id)
        self.dataset_uid = (
            str(dataset_uid)
            if dataset_uid is not None
            else str(self.root_dir.resolve())
        )
        self.prompt_policy_id = str(prompt_policy_id)
        self._init_instruction_config(
            prompt_template=prompt_template,
            humanize_task_name=humanize_task_name,
            instruction_source=instruction_source,
            instruction_type=instruction_type,
            instruction_selection=instruction_selection,
            task_instruction_dir=task_instruction_dir,
            objects_description_dir=objects_description_dir,
            max_instructions_per_episode=max_instructions_per_episode,
        )
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
        if self.sample_stride <= 0:
            raise ValueError(
                f"`sample_stride` must be positive, got {self.sample_stride}"
            )
        if self.num_video_frames % 4 != 1:
            raise ValueError(
                f"RGB VAE video length must satisfy T % 4 == 1 for WAN VAE tokenization, got T={self.num_video_frames}."
            )
        self.frame_offsets = [
            i * self.frame_stride for i in range(self.num_video_frames)
        ]
        self.window_span = self.frame_offsets[-1] + 1
        self.episodes: list[dict] = []
        self.cumulative_sizes: list[int] = []
        self._build_index()
        self._attach_instruction_candidates_to_episodes()

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
        key = f"observation/{self.cameras[0]}/rgb"
        with h5py.File(path, "r") as f:
            if key not in f:
                raise KeyError(f"Missing RGB dataset `{key}` in {path}")
            return int(f[key].shape[0])

    @staticmethod
    def _episode_index(path: Path) -> int:
        match = re.search("episode(\\d+)", path.stem)
        return int(match.group(1)) if match is not None else -1

    def __len__(self):
        return self.cumulative_sizes[-1]

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
        return (episode, start)

    def _decode_rgb_frame(self, encoded) -> np.ndarray:
        return decode_rgb_frame(encoded, self.xpolicylab_parent)

    def _read_rgb_window(
        self, h5_file: h5py.File, camera: str, start: int, length: int
    ) -> torch.Tensor:
        rgb_ds = h5_file[f"observation/{camera}/rgb"]
        stop = min(start + self.window_span, length)
        encoded_frames = rgb_ds[start : stop : self.frame_stride]
        if encoded_frames.shape[0] == 0:
            raise ValueError(
                f"Empty RGB window for camera={camera}, start={start}, length={length}"
            )
        frames = [self._decode_rgb_frame(frame) for frame in encoded_frames]
        while len(frames) < self.num_video_frames:
            frames.append(frames[-1])
        video = np.stack(frames, axis=0).astype(np.float32) / 127.5 - 1.0
        video = torch.from_numpy(video).permute(0, 3, 1, 2).contiguous()
        return video

    @staticmethod
    def _resize(video: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(video, size=size, mode="bilinear", align_corners=False)

    def _layout_video(self, camera_videos: list[torch.Tensor]) -> torch.Tensor:
        (height, width) = self.video_size
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
                for (i, video) in enumerate(camera_videos)
            ]
            return torch.cat(resized, dim=-1)
        if self.layout == "single":
            if len(camera_videos) != 1:
                raise ValueError("`layout=single` expects exactly one camera.")
            return self._resize(camera_videos[0], (height, width))
        raise ValueError(f"Unsupported RGB video layout: {self.layout}")

    def _get_cached_text_context(self, prompt: str):
        if self.text_embedding_cache_dir is None:
            return (None, None)
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = (
            self.text_embedding_cache_dir
            / f"{hashed}.t5_len{self.context_len}.{self.text_encoder_id}.pt"
        )
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Missing text embedding cache: {cache_path}. Build it with `scripts/prepare_data.py text`."
            )
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"]
        context_mask = payload["mask"].bool()
        if context.ndim != 2:
            raise ValueError(
                f"Cached `context` must be 2D [L,D], got {tuple(context.shape)} in {cache_path}"
            )
        if context_mask.ndim != 1:
            raise ValueError(
                f"Cached `mask` must be 1D [L], got {tuple(context_mask.shape)} in {cache_path}"
            )
        if context.shape[0] != self.context_len:
            raise ValueError(
                f"Cached context length mismatch in {cache_path}: expected {self.context_len}, got {context.shape[0]}"
            )
        context = context.clone()
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)
        return (context, context_mask)

    def __getitem__(self, idx: int):
        (episode, start) = self._locate(idx)
        length = int(episode["length"])
        with h5py.File(episode["path"], "r") as f:
            camera_videos = [
                self._read_rgb_window(f, camera=camera, start=start, length=length)
                for camera in self.cameras
            ]
        video_tchw = self._layout_video(camera_videos)
        image_is_pad = torch.tensor(
            [start + offset >= length for offset in self.frame_offsets],
            dtype=torch.bool,
        )
        prompt = self._prompt_for_episode(episode, start=start)
        (context, context_mask) = self._get_cached_text_context(prompt)
        sample = {
            "video": video_tchw.permute(1, 0, 2, 3).contiguous(),
            "image_is_pad": image_is_pad,
            "prompt": prompt,
            "idx": int(idx),
            "task": episode["task"],
            "phase": episode["phase"],
            "episode_index": int(episode["episode_index"]),
            "start_index": int(start),
        }
        if context is not None:
            sample["context"] = context
            sample["context_mask"] = context_mask
        return sample
