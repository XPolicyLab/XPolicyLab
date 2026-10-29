"""Aligned RGB/Depth/Action RoboTwin samples for PrismWAM Stage 2.

The Depth dataset defines the sample boundary, actions, proprioception and text
context. This adapter adds the RGB clip for the same anchor: either the full
``F0:8`` video, or a causal RGB history ending at F0 followed by the F8 goal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from prismwam.rgb_history import RGBHistory

from .robotwin_depth_joint_dataset import RobotWinDepthJointVideoDataset
from .robotwin_rgb_dataset import DEFAULT_ROBOTWIN_ROOT, RobotWinRGBVideoDataset
from .xpolicylab_images import decode_rgb_frame


RGB_TARGET_F8 = "f8"


class RobotWinRGBDepthActionVideoDataset(RobotWinDepthJointVideoDataset):
    """Depth-first RoboTwin dataset with an aligned RGB diffusion target."""

    def __init__(
        self,
        *args,
        rgb_target_mode: str = RGB_TARGET_F8,
        rgb_history_num_frames: int = 1,
        rgb_history_frame_stride: int = 4,
        xpolicylab_parent: str = DEFAULT_ROBOTWIN_ROOT,
        **kwargs,
    ) -> None:
        self.xpolicylab_parent = str(Path(xpolicylab_parent).expanduser())
        self.rgb_target_mode = str(rgb_target_mode).strip().lower()
        if self.rgb_target_mode != RGB_TARGET_F8:
            raise ValueError(
                f"`rgb_target_mode` must be {RGB_TARGET_F8!r}, got {rgb_target_mode!r}."
            )
        self.rgb_history_num_frames = int(rgb_history_num_frames)
        self.rgb_history_frame_stride = int(rgb_history_frame_stride)
        if (
            self.rgb_history_num_frames <= 0
            or (self.rgb_history_num_frames - 1) % 4 != 0
        ):
            raise ValueError(
                "`rgb_history_num_frames` must satisfy T = 4n + 1, got "
                f"{self.rgb_history_num_frames}."
            )
        if self.rgb_history_frame_stride <= 0:
            raise ValueError("`rgb_history_frame_stride` must be positive.")
        self.rgb_history = RGBHistory.dense(
            self.rgb_history_num_frames, self.rgb_history_frame_stride
        )
        super().__init__(*args, **kwargs)

    def _decode_rgb_frame(self, encoded) -> np.ndarray:
        return decode_rgb_frame(encoded, self.xpolicylab_parent)

    def _read_rgb_video(self, episode: dict[str, Any], start: int) -> torch.Tensor:
        length = int(episode["length"])
        with h5py.File(episode["path"], "r") as h5_file:
            camera_videos = [
                RobotWinRGBVideoDataset._read_rgb_window(
                    self,
                    h5_file,
                    camera=camera,
                    start=start,
                    length=length,
                )
                for camera in self.cameras
            ]
        return (
            RobotWinRGBVideoDataset._layout_video(
                self,
                camera_videos,
            )
            .permute(1, 0, 2, 3)
            .contiguous()
        )

    def _read_rgb_history(
        self,
        episode: dict[str, Any],
        start: int,
    ) -> torch.Tensor:
        """Read history ending at ``start`` and left-pad at episode start."""

        length = int(episode["length"])
        if length <= 0:
            raise ValueError("Cannot read RGB history from an empty episode.")
        indices = self.rgb_history.indices(start, last=length - 1)
        with h5py.File(episode["path"], "r") as h5_file:
            camera_videos = []
            for camera in self.cameras:
                rgb_ds = h5_file[f"observation/{camera}/rgb"]
                frames = [self._decode_rgb_frame(rgb_ds[index]) for index in indices]
                video = np.stack(frames, axis=0).astype(np.float32) / 127.5 - 1.0
                camera_videos.append(
                    torch.from_numpy(video).permute(0, 3, 1, 2).contiguous()
                )
        return (
            RobotWinRGBVideoDataset._layout_video(
                self,
                camera_videos,
            )
            .permute(1, 0, 2, 3)
            .contiguous()
        )

    def __getitem__(self, idx: int) -> dict[str, Any]:
        episode, start = self._locate(idx)
        sample = super().__getitem__(idx)
        future_video = self._read_rgb_video(episode, start)
        if self.rgb_history_num_frames == 1:
            sample["rgb_video"] = future_video
        else:
            sample["rgb_video"] = torch.cat(
                (self._read_rgb_history(episode, start), future_video[:, -1:]),
                dim=1,
            ).contiguous()
        sample["rgb_target_is_valid"] = torch.tensor(
            int(start) + int(self.frame_offsets[-1]) < int(episode["length"]),
            dtype=torch.bool,
        )
        sample["rgb_target_mode"] = self.rgb_target_mode
        return sample
