"""Observed sparse RGB contexts with checkpoint-bound action windows."""

from __future__ import annotations
from typing import Any
import h5py
import torch
from prismwam.rgb_history import RGBHistory
from .robodojo_rgb_depth_action_dataset import RoboDojoRGBDepthActionVideoDataset


def read_memory_video(
    dataset, episode: dict[str, Any], indices: list[int]
) -> torch.Tensor:
    with h5py.File(episode["path"], "r") as handle:
        cameras = [
            dataset._read_rgb_frames(handle, camera, indices)
            for camera in dataset.cameras
        ]
    return dataset._layout_video(cameras).permute(1, 0, 2, 3).contiguous()


class RoboDojoSparseRGBMemoryDataset(RoboDojoRGBDepthActionVideoDataset):
    """Encode episode start, t-32 and current separately; supervise an F8 endpoint.

    Action, state, Depth, instruction and sampling use the standard dataset.
    """

    def __init__(
        self,
        *,
        sparse_recent_action_offset: int = 32,
        sparse_frame_encoding: str = "independent",
        rgb_history_num_frames: int = 3,
        rgb_history_frame_stride: int = 4,
        **kwargs,
    ):
        if sparse_frame_encoding != "independent":
            raise ValueError("Unknown sparse dataset encoding")
        expected_frames = 3
        if (
            rgb_history_num_frames != expected_frames
            or sparse_recent_action_offset <= 0
        ):
            raise ValueError(
                f"Sparse RGB requires {expected_frames} observations and a positive offset"
            )
        super().__init__(rgb_history_num_frames=1, rgb_history_frame_stride=1, **kwargs)
        if self.rgb_target_mode != "f8":
            raise ValueError("Sparse RGB requires the standard PrismWAM F8 target")
        self.sparse_recent_action_offset = int(sparse_recent_action_offset)
        self.rgb_history = RGBHistory.sparse(self.sparse_recent_action_offset)
        self.sparse_frame_encoding = sparse_frame_encoding
        self.rgb_history_num_frames = expected_frames
        self.rgb_history_frame_stride = int(rgb_history_frame_stride)

    def _read_rgb_history(self, episode: dict[str, Any], start: int) -> torch.Tensor:
        indices = self.rgb_history.indices(start)
        return read_memory_video(self, episode, indices)

    def _read_rgb_video(self, episode: dict[str, Any], start: int) -> torch.Tensor:
        last_offset = min(
            int(self.frame_offsets[-1]),
            (int(episode["length"]) - 1 - start)
            // self.frame_stride
            * self.frame_stride,
        )
        return read_memory_video(self, episode, [start + last_offset])
