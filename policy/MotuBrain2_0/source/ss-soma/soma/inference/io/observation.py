"""Camera-view selection, composition, and resizing for inference."""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
import logging

import numpy as np
import torch
from morph.convs import image as image_convs

from soma.data import visual_history
from soma.inference.config import MultiviewConfig
from soma.inference.config import ObservationConfig
from soma.inference.config import SparseMemoryConfig
from soma.inference.io import video_encoding
from soma.inference.runtime.backend import ModelInferenceRuntime
from soma.inference.types import DiffusionCondition
from soma.inference.types import ImageInput
from soma.inference.types import SparseVideoMemory

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _SparseMemoryFrames:
    """Selected RGB frames for one sparse-memory inference request."""

    history: tuple[np.ndarray, ...]
    current: tuple[np.ndarray, ...]


class _SparseVisualHistory:
    """Retain the configured past observations plus the episode anchor."""

    def __init__(self, view_names: Sequence[str], frame_stride: int) -> None:
        self._view_names = tuple(view_names)
        self._offsets = visual_history.history_offsets(frame_stride)
        self._span = -self._offsets[0]
        self.reset()

    def reset(self) -> None:
        """Clear all episode-scoped frames."""
        self._anchor: tuple[np.ndarray, ...] | None = None
        self._recent_frames: list[tuple[np.ndarray, ...]] = []
        self._frame_count = 0

    def update(
        self,
        views: Sequence[np.ndarray],
    ) -> _SparseMemoryFrames:
        """Consume every captured observation, then select history and current."""
        if len(views) != len(self._view_names):
            raise ValueError(
                "sparse-memory views must match configured view names: " f"{len(views)} != {len(self._view_names)}"
            )
        temporal_views = [_as_temporal_frames(view)[0] for view in views]
        frame_counts = {view.shape[0] for view in temporal_views}
        if len(frame_counts) != 1:
            raise ValueError(
                "sparse-memory views must contain equal frame counts, got "
                f"{[view.shape[0] for view in temporal_views]}"
            )
        frame_count = next(iter(frame_counts))
        if frame_count <= 0:
            raise ValueError("sparse-memory observation chunks must not be empty")

        for frame_index in range(frame_count):
            frame = tuple(view[frame_index].copy() for view in temporal_views)
            if self._anchor is None:
                self._anchor = frame
            self._recent_frames.append(frame)
            if len(self._recent_frames) > self._span + 1:
                self._recent_frames.pop(0)
            self._frame_count += 1

        if self._anchor is None:
            raise RuntimeError("sparse-memory history has no anchor frame")
        current = self._recent_frames[-1]
        _, indices = visual_history.select_history_indices(self._frame_count - 1, self._offsets)
        buffer_start = self._frame_count - len(self._recent_frames)
        selected = [
            self._anchor if index == 0 else self._recent_frames[index - buffer_start]
            for index in indices
        ]
        return _SparseMemoryFrames(
            history=tuple(
                np.stack([frame[view_index] for frame in selected])
                for view_index in range(len(self._view_names))
            ),
            current=current,
        )


def _as_temporal_frames(frames: object) -> tuple[np.ndarray, bool]:
    """Normalize one image or temporal image chunk to ``THWC``."""
    array = np.asarray(frames, dtype=np.uint8)
    if array.ndim == 3 and array.shape[-1] == 3:
        return array[np.newaxis, ...], True
    if array.ndim == 4 and array.shape[-1] == 3:
        return array, False
    raise ValueError("camera frames must have shape [H,W,3] or [T,H,W,3], got " f"{array.shape}")


def resize_frames(
    frames: object,
    resize: image_convs.ResizeConfig,
) -> np.ndarray:
    """Resize one frame or temporal observation chunk with Morph."""
    array = np.asarray(frames, dtype=np.uint8)
    if array.ndim not in {3, 4} or array.shape[-1] != 3:
        raise ValueError("view must have shape [H,W,3] or [T,H,W,3], got " f"{array.shape}")
    return resize.apply(array, spatial_axes=(-3, -2))


class ObservationWrapper:
    """Prepare configured camera inputs as an ordered list of video views."""

    def __init__(
        self,
        observation_cfg: ObservationConfig,
        multiview_cfg: MultiviewConfig,
        runtime: ModelInferenceRuntime,
        *,
        sparse_memory: SparseMemoryConfig | None = None,
    ) -> None:
        self.cfg = observation_cfg
        self.runtime = runtime
        self.num_views = multiview_cfg.num_views
        self.view_names = list(multiview_cfg.view_names)
        self.view_resizes = dict(multiview_cfg.view_resizes)
        if len(self.view_names) != self.num_views:
            raise ValueError(
                "multiview.view_names must contain num_views entries, got "
                f"num_views={self.num_views}, view_names={self.view_names}"
            )
        self._sparse_history = None
        if sparse_memory is not None and sparse_memory.enabled:
            if self.cfg.video_length != -1:
                raise ValueError("sparse memory requires observation.video_length=-1")
            self._sparse_history = _SparseVisualHistory(
                self.view_names,
                frame_stride=sparse_memory.frame_stride,
            )

    def encode(
        self,
        images: ImageInput,
        instruction: str,
        *,
        video_views: Sequence[np.ndarray] | None = None,
    ) -> DiffusionCondition:
        """Prepare and encode all visual and text conditioning inputs."""
        views = self.prepare(images, video_views=video_views)
        return self.encode_prepared(views, instruction)

    def encode_prepared(
        self,
        views: Sequence[np.ndarray],
        instruction: str,
    ) -> DiffusionCondition:
        """Encode views already prepared by this wrapper."""
        if len(views) != self.num_views:
            raise ValueError(f"expected {self.num_views} prepared views, got {len(views)}")
        if self._sparse_history is not None:
            return self._encode_sparse_prepared(views, instruction)
        latents = []
        for frames, view_name in zip(
            views,
            self.view_names,
            strict=True,
        ):
            video = self._preprocess_video(frames)
            latents.append(self.runtime.encode_video(video, stream_id=view_name))
        return DiffusionCondition(
            video_latents=latents,
            text_embedding=self.runtime.encode_text(instruction),
        )

    def reset(self) -> None:
        """Clear causal observation features accumulated for this episode."""
        if self._sparse_history is not None:
            self._sparse_history.reset()
        self.runtime.reset_observation()

    def _encode_sparse_prepared(
        self,
        views: Sequence[np.ndarray],
        instruction: str,
    ) -> DiffusionCondition:
        """Encode the configured causal history and current-frame layout."""
        if self._sparse_history is None:
            raise RuntimeError("sparse-memory history is not configured")
        frames = self._sparse_history.update(views)
        history_latents = self._encode_independent_views(
            frames.history,
            role="history",
        )
        current_latents = self._encode_independent_views(
            frames.current,
            role="current",
        )
        return DiffusionCondition(
            video_latents=current_latents,
            text_embedding=self.runtime.encode_text(instruction),
            sparse_memory=SparseVideoMemory(latents=history_latents),
        )

    def _encode_independent_views(
        self,
        views: Sequence[np.ndarray],
        *,
        role: str,
    ) -> list[torch.Tensor]:
        """Encode each view's complete sequence without a stream cache."""
        latents = []
        for frame, view_name in zip(views, self.view_names, strict=True):
            latents.append(
                self.runtime.encode_video(
                    self._preprocess_video(frame),
                    stream_id=f"sparse_memory/{role}/{view_name}",
                    incremental=False,
                )
            )
        return latents

    def prepare(
        self,
        images: ImageInput,
        *,
        video_views: Sequence[np.ndarray] | None = None,
    ) -> list[np.ndarray]:
        """Resolve, compose, and resize configured camera views."""
        views = self._resolve_views(images, video_views)
        prepared = []
        for frames, view_name in zip(views, self.view_names, strict=True):
            resized = resize_frames(
                frames,
                self.view_resizes.get(view_name, self.cfg.resize),
            )
            if self.cfg.post_resize_h264_crf is not None:
                resized = video_encoding.h264_roundtrip(
                    resized,
                    self.cfg.post_resize_h264_crf,
                    repeat_frames=self.cfg.post_resize_h264_repeat_frames,
                    threads=self.cfg.post_resize_h264_threads,
                )
            prepared.append(resized)
        return prepared

    def decode_video_latents(
        self,
        latents: Sequence[torch.Tensor],
        *,
        ground_truth_latents: Sequence[torch.Tensor] | None = None,
        head_split_num: int = 0,
    ) -> list[np.ndarray]:
        """Decode predicted view latents to uint8 videos."""
        if ground_truth_latents is not None and len(ground_truth_latents) != len(latents):
            raise ValueError("ground_truth_latents and latents must contain the same views")
        videos = []
        for index, (latent, view_name) in enumerate(zip(latents, self.view_names, strict=True)):
            if ground_truth_latents is None:
                if head_split_num > 0:
                    latent = latent[:, head_split_num:]
                video = self.runtime.decode_video(latent)
            else:
                prediction_start = max(1, head_split_num)
                video = self.runtime.decode_video_incremental(
                    ground_truth_latents[index],
                    latent[:, prediction_start:],
                    stream_id=view_name,
                    include_condition=head_split_num == 0,
                )
            videos.append(video.numpy())
        return videos

    def _resolve_views(
        self,
        images: ImageInput,
        video_views: Sequence[np.ndarray] | None,
    ) -> list[np.ndarray]:
        if self.num_views == 1 and isinstance(images, Mapping):
            if self.cfg.use_cam_torso:
                views = [self._compose_torso_main(images)]
            elif video_views is not None:
                views = list(video_views)
            else:
                views = [self._compose_head_wrist(images)]
        elif video_views is not None:
            views = list(video_views)
        elif self.num_views == 1:
            views = [np.asarray(images, dtype=np.uint8)]
        elif isinstance(images, Mapping) and all(view_name in images for view_name in self.view_names):
            views = [np.asarray(images[name], dtype=np.uint8) for name in self.view_names]
        else:
            available = list(images) if isinstance(images, Mapping) else []
            missing = [name for name in self.view_names if name not in available]
            raise ValueError("Multiview input is missing configured cameras: " f"{missing}; available={available}")

        if len(views) != self.num_views:
            raise ValueError(f"expected {self.num_views} video views, got {len(views)}")
        return views

    def _compose_torso_main(
        self,
        images: Mapping[str, object],
    ) -> np.ndarray:
        torso = self._pick_frames(images, self.cfg.torso_camera_keys)
        main = self._pick_frames(images, self.cfg.main_camera_keys)
        if main is None:
            main = self._pick_frames(images, tuple(images))
        if main is None:
            raise ValueError("images must contain at least one camera frame")

        main, main_is_single = _as_temporal_frames(main)
        assert self.cfg.main_resize is not None
        main = resize_frames(main, self.cfg.main_resize)
        if torso is None:
            logger.warning(
                "use_cam_torso=True but none of %s are present; using main view",
                self.cfg.torso_camera_keys,
            )
            return main[0] if main_is_single else main
        torso, torso_is_single = _as_temporal_frames(torso)
        if torso.shape[0] != main.shape[0]:
            raise ValueError(
                "torso and main cameras must have the same frame count, got " f"{torso.shape[0]} and {main.shape[0]}"
            )
        assert self.cfg.torso_resize is not None
        torso = resize_frames(torso, self.cfg.torso_resize)
        if torso.shape[-2] != main.shape[-2]:
            raise ValueError(
                "torso_resize and main_resize must produce equal widths, got " f"{torso.shape[-2]} and {main.shape[-2]}"
            )
        composed = np.concatenate([torso, main], axis=1)
        return composed[0] if torso_is_single and main_is_single else composed

    def _compose_head_wrist(
        self,
        images: Mapping[str, object],
    ) -> np.ndarray:
        head = self._pick_frames(
            images,
            ("head_camera", "head", "cam_high"),
        )
        left = self._pick_frames(
            images,
            ("left_camera", "left_wrist", "cam_left_wrist"),
        )
        right = self._pick_frames(
            images,
            ("right_camera", "right_wrist", "cam_right_wrist"),
        )
        if head is None:
            return np.asarray(next(iter(images.values())), dtype=np.uint8)
        if left is None or right is None:
            return head

        temporal = [_as_temporal_frames(value) for value in (head, left, right)]
        frame_counts = [frames.shape[0] for frames, _ in temporal]
        if len(set(frame_counts)) != 1:
            raise ValueError("head and wrist cameras must have the same frame count, got " f"{frame_counts}")
        head, left, right = [frames for frames, _ in temporal]
        height, width = head.shape[1:3]
        left = resize_frames(
            left,
            image_convs.ResizeConfig(
                width=width // 2,
                height=height // 2,
                mode="direct",
                interpolation=self.cfg.resize.interpolation,
            ),
        )
        right = resize_frames(
            right,
            image_convs.ResizeConfig(
                width=width // 2,
                height=height // 2,
                mode="direct",
                interpolation=self.cfg.resize.interpolation,
            ),
        )
        bottom = np.concatenate([left, right], axis=2)
        composed = np.concatenate([head, bottom], axis=1)
        return composed[0] if all(single for _, single in temporal) else composed

    @staticmethod
    def _pick_frames(
        images: Mapping[str, object],
        camera_keys: tuple[str, ...],
    ) -> np.ndarray | None:
        for camera_key in camera_keys:
            if camera_key in images:
                return np.asarray(images[camera_key], dtype=np.uint8)
        return None

    @staticmethod
    def _preprocess_video(frames: np.ndarray) -> torch.Tensor:
        """Convert uint8 frames from ``THWC`` to normalized ``CTHW``."""
        frames = np.asarray(frames, dtype=np.uint8)
        if frames.ndim == 3:
            frames = frames[np.newaxis, ...]
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError("video must have shape [H,W,3] or [T,H,W,3], got " f"{frames.shape}")
        normalized = frames.astype(np.float32) / 255.0 * 2.0 - 1.0
        return torch.from_numpy(normalized).permute(3, 0, 1, 2)
