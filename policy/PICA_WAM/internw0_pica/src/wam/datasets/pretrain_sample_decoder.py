import math
import os
import time
from typing import Any

import pyarrow as pa
import torch
import torch.nn.functional as F

from wam.datasets import pretrain_lerobot_loader as pretrain_loader


class PretrainSampleDecoderMixin:
    """Video decoding and conversion into InternW0-delta training samples."""

    def _video_indices(self, length: int) -> list[int]:
        return list(range(0, int(length), self.action_video_freq_ratio))

    def _video_concat_mode(self, ds: pretrain_loader.PretrainLeRobotDataset) -> str:
        if self.concat_multi_camera is not None:
            return self.concat_multi_camera
        modalities = ds.spec.modalities or {}
        mode = modalities.get("video_concat") or modalities.get("concat_multi_camera")
        video_spec = modalities.get("video")
        if mode is None and isinstance(video_spec, dict):
            mode = video_spec.get("concat") or video_spec.get("concat_mode")
        if mode is None:
            return "single" if len(ds.video_keys) <= 1 else "horizontal"
        return str(mode)

    def _resize_clip(self, clip: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(
            clip.to(torch.float32), size=size, mode="bilinear", align_corners=False
        )

    def _resize_clip_to_slot(
        self, clip: torch.Tensor, size: tuple[int, int]
    ) -> torch.Tensor:
        return self._resize_clip(clip, size)

    def _single_repeat_video(self, clip: torch.Tensor) -> torch.Tensor:
        height, width = self.video_size
        top_h = int(height) // 2
        bottom_h = int(height) - top_h
        top = self._resize_clip_to_slot(clip, (top_h, width))
        bottom = (
            top
            if bottom_h == top_h
            else self._resize_clip_to_slot(clip, (bottom_h, width))
        )
        return torch.cat([top, bottom], dim=-2)

    def _single_masked_video(self, clip: torch.Tensor) -> torch.Tensor:
        """Place one physical view in the upper half of the shared canvas."""
        height, width = self.video_size
        valid_h = max(1, int(height) // 2)
        masked_h = int(height) - valid_h
        valid = self._resize_clip_to_slot(clip, (valid_h, width))
        if masked_h <= 0:
            return valid
        # 0.5 becomes zero after the later [0, 1] -> [-1, 1]
        # normalization. The value is still explicitly excluded by the
        # spatial mask; using the neutral value only reduces VAE boundary
        # leakage at the valid/masked seam.
        masked = valid.new_full(
            (*valid.shape[:-2], masked_h, int(width)), 0.5
        )
        return torch.cat([valid, masked], dim=-2)

    def _video_spatial_valid_mask(
        self, ds: pretrain_loader.PretrainLeRobotDataset
    ) -> torch.Tensor:
        """Return the model-canvas pixel mask shared by all frames."""
        height, width = self.video_size
        mask = torch.ones((int(height), int(width)), dtype=torch.bool)
        mode = self._video_concat_mode(ds)
        if mode in {"head_right_wrist_masked", "head_left_wrist_masked"}:
            top_h, _, left_w, _ = self._robotwin_slot_sizes()
            if mode == "head_right_wrist_masked":
                mask[top_h:, :left_w] = False
            else:
                mask[top_h:, left_w:] = False
        elif mode == "single_masked":
            valid_h = max(1, int(height) // 2)
            mask[valid_h:] = False
        return mask

    def _robotwin_slot_sizes(self) -> tuple[int, int, int, int]:
        height, width = self.video_size
        top_h = max(1, (int(height) * 2) // 3)
        bottom_h = max(1, int(height) - top_h)
        left_w = int(width) // 2
        right_w = int(width) - left_w
        return top_h, bottom_h, left_w, right_w

    def _head_one_wrist_video(self, clips: list[torch.Tensor], *, right: bool) -> torch.Tensor:
        if len(clips) != 2:
            raise ValueError("Head + one physical wrist requires exactly two views")
        top_h, bottom_h, left_w, right_w = self._robotwin_slot_sizes()
        top = self._resize_clip_to_slot(clips[0], (top_h, self.video_size[1]))
        wrist_width = right_w if right else left_w
        wrist = self._resize_clip_to_slot(clips[1], (bottom_h, wrist_width))
        missing_width = left_w if right else right_w
        missing = wrist.new_full((*wrist.shape[:-2], bottom_h, missing_width), 0.5)
        bottom = torch.cat([missing, wrist], dim=-1) if right else torch.cat([wrist, missing], dim=-1)
        return torch.cat([top, bottom], dim=-2)

    def _horizontal_video(self, clips: list[torch.Tensor]) -> torch.Tensor:
        if not clips:
            raise RuntimeError("No video clips to concatenate.")
        height, width = self.video_size
        num_cameras = len(clips)
        base_width = width // num_cameras
        widths = [base_width] * num_cameras
        widths[-1] += width - sum(widths)
        resized = [
            self._resize_clip_to_slot(clip, (height, widths[idx]))
            for idx, clip in enumerate(clips)
        ]
        return torch.cat(resized, dim=-1)

    def _vertical_video(self, clips: list[torch.Tensor]) -> torch.Tensor:
        if not clips:
            raise RuntimeError("No video clips to concatenate.")
        if len(clips) == 1:
            return self._single_repeat_video(clips[0])
        height, width = self.video_size
        num_cameras = len(clips)
        base_height = height // num_cameras
        heights = [base_height] * num_cameras
        heights[-1] += height - sum(heights)
        resized = [
            self._resize_clip_to_slot(clip, (heights[idx], width))
            for idx, clip in enumerate(clips)
        ]
        return torch.cat(resized, dim=-2)

    def _grid2x2_video(self, clips: list[torch.Tensor]) -> torch.Tensor:
        if len(clips) not in {3, 4}:
            raise ValueError(f"grid2x2 expects 3 or 4 cameras, got {len(clips)}")
        height, width = self.video_size
        top_h = height // 2
        bottom_h = height - top_h
        left_w = width // 2
        right_w = width - left_w
        empty = (
            clips[0]
            .to(torch.float32)
            .new_zeros((*clips[0].shape[:-2], bottom_h, right_w))
        )
        cells = [
            self._resize_clip_to_slot(clips[0], (top_h, left_w)),
            self._resize_clip_to_slot(clips[1], (top_h, right_w)),
            self._resize_clip_to_slot(clips[2], (bottom_h, left_w)),
            self._resize_clip_to_slot(clips[3], (bottom_h, right_w))
            if len(clips) == 4
            else empty,
        ]
        top = torch.cat(cells[:2], dim=-1)
        bottom = torch.cat(cells[2:], dim=-1)
        return torch.cat([top, bottom], dim=-2)

    def _robotwin_video(self, clips: list[torch.Tensor]) -> torch.Tensor:
        if len(clips) not in {1, 2, 3}:
            raise ValueError(f"robotwin expects 1, 2, or 3 cameras, got {len(clips)}")
        if len(clips) == 1:
            return self._single_repeat_video(clips[0])
        top_h, bottom_h, left_w, right_w = self._robotwin_slot_sizes()
        _, width = self.video_size
        top = self._resize_clip_to_slot(clips[0], (top_h, width))
        if len(clips) == 2:
            bottom = self._resize_clip_to_slot(clips[1], (bottom_h, width))
        else:
            left = self._resize_clip_to_slot(clips[1], (bottom_h, left_w))
            right = self._resize_clip_to_slot(clips[2], (bottom_h, right_w))
            bottom = torch.cat([left, right], dim=-1)
        return torch.cat([top, bottom], dim=-2)

    def _latent_horizontal_video(self, clips: list[torch.Tensor]) -> torch.Tensor:
        if not clips:
            raise RuntimeError("No video clips to concatenate.")
        resized = [self._resize_clip_to_slot(clip, self.video_size) for clip in clips]
        video = torch.stack(resized, dim=0).mul(2.0).sub(1.0)
        return video.permute(0, 2, 1, 3, 4).contiguous()

    def _video_sample_indices_for_item(self, item: dict[str, Any]) -> list[int]:
        if "timestamp" in item:
            query_len = int(item["timestamp"].shape[0])
        elif "frame_index" in item:
            query_len = int(item["frame_index"].shape[0])
        else:
            query_len = int(item["observation.state"].shape[0])
        return self._video_indices(query_len)

    def _video_timestamps_for_item(
        self,
        ds: pretrain_loader.PretrainLeRobotDataset,
        item: dict[str, Any],
        video_key: str,
        sample_indices: list[int] | tuple[int, ...],
    ) -> list[float]:
        return ds.resolve_video_timestamps(item, video_key, sample_indices)

    def _log_bad_video(self, message: str) -> None:
        if self._bad_video_logs < self._bad_video_log_limit:
            rank = os.environ.get("RANK", "?")
            print(f"[pretrain-dataset][rank{rank}] {message}", flush=True)
        self._bad_video_logs += 1

    def _item_context(self, item: dict[str, Any]) -> str:
        ds_pos = int(item["dataset_index"].item())
        ds = self.datasets[ds_pos]
        episode_index = int(item["episode_index"].item())
        frame = "?"
        frame_index = item.get("frame_index")
        if frame_index is not None:
            try:
                frame = str(int(frame_index.reshape(-1)[0].item()))
            except Exception:
                frame = str(frame_index)
        dataset_name = str(item.get("dataset_name") or ds.name)
        return f"dataset={dataset_name} ds_pos={ds_pos} episode={episode_index} frame={frame}"

    def _video_decode_context(
        self,
        ds: pretrain_loader.PretrainLeRobotDataset,
        episode_index: int,
        video_key: str,
        *,
        positions: list[int] | None = None,
        items: list[dict[str, Any]] | None = None,
        raw_bytes: int | None = None,
        timestamps: list[float] | None = None,
    ) -> str:
        try:
            rel_path = ds._video_rel_path(int(episode_index), video_key)
            remote_path = ds._remote_path(rel_path)
        except Exception as exc:
            rel_path = f"<unresolved: {type(exc).__name__}: {exc}>"
            remote_path = rel_path
        sample_bits: list[str] = []
        if positions is not None and items is not None:
            for pos in positions[:4]:
                try:
                    sample_bits.append(self._item_context(items[pos]))
                except Exception:
                    sample_bits.append(f"local_pos={pos}")
        ts_text = ""
        if timestamps:
            ts_text = f" timestamps=[{min(timestamps):.3f},{max(timestamps):.3f}] n_ts={len(timestamps)}"
        bytes_text = f" bytes={raw_bytes}" if raw_bytes is not None else ""
        samples_text = f" samples={sample_bits}" if sample_bits else ""
        return (
            f"dataset={ds.name} root={ds.remote_root} episode={int(episode_index)} "
            f"video_key={video_key} rel_path={rel_path} remote_path={remote_path}"
            f"{bytes_text}{ts_text}{samples_text}"
        )

    def _replacement_index(self, idx: int, attempt: int) -> int:
        if self.mixed_dataset.training and hasattr(self.mixed_dataset, "optimizer_stratified_replacement_index"):
            return self.mixed_dataset.optimizer_stratified_replacement_index(int(idx), int(attempt))
        length = max(len(self), 1)
        samples_per_segment = max(1, int(getattr(self.mixed_dataset, "training_block_size", 1)))
        return int(
            (
                int(idx)
                + 104729 * int(attempt) * samples_per_segment
            )
            % length
        )

    def _item_video_to_training_tensor_with_context(
        self, item: dict[str, Any]
    ) -> torch.Tensor:
        try:
            return self._item_video_to_training_tensor(item)
        except Exception as exc:
            ds_pos = int(item["dataset_index"].item())
            ds = self.datasets[ds_pos]
            raise RuntimeError(
                f"Pretrain video decode failed while loading {self._item_context(item)} keys={list(ds.video_keys)}"
            ) from exc

    def _pack_single_index_with_retries(self, idx: int) -> dict[str, Any]:
        last_exc: Exception | None = None
        attempts = self.max_decode_retries + 1 if self.skip_bad_videos else 1
        for attempt in range(attempts):
            sample_idx = (
                int(idx) if attempt == 0 else self._replacement_index(int(idx), attempt)
            )
            item_t0 = time.perf_counter()
            item = self.mixed_dataset[sample_idx]
            item_s = time.perf_counter() - item_t0
            video_t0 = time.perf_counter()
            try:
                video = self._item_video_to_training_tensor_with_context(item)
            except Exception as exc:
                last_exc = exc
                self._log_bad_video(
                    f"bad video sample attempt={attempt + 1}/{attempts} original_idx={int(idx)} "
                    f"sample_idx={sample_idx} {self._item_context(item)} error={type(exc).__name__}: {exc}"
                )
                continue
            video_s = time.perf_counter() - video_t0
            try:
                packed = self._pack_training_item(
                    int(idx), item, video, item_s=item_s, video_s=video_s
                )
            except Exception as exc:
                last_exc = exc
                self._log_bad_video(
                    f"bad packed sample attempt={attempt + 1}/{attempts} original_idx={int(idx)} "
                    f"sample_idx={sample_idx} {self._item_context(item)} error={type(exc).__name__}: {exc}"
                )
                continue
            if attempt > 0:
                self._log_bad_video(
                    f"replaced bad video original_idx={int(idx)} with sample_idx={sample_idx} after {attempt} retry"
                )
            return packed
        raise RuntimeError(
            f"Failed to fetch a decodable Pretrain sample for idx={int(idx)} after {attempts} attempts"
        ) from last_exc

    def _clips_to_training_tensor(
        self,
        ds: pretrain_loader.PretrainLeRobotDataset,
        clips: list[torch.Tensor],
    ) -> torch.Tensor:
        if not clips:
            raise RuntimeError(f"Dataset {ds.name} has no selected video keys.")
        clips = [clip.to(torch.float32) for clip in clips]
        mode = self._video_concat_mode(ds)
        if mode == "single":
            video = self._single_repeat_video(clips[0])
        elif mode == "single_masked":
            if len(clips) != 1:
                raise ValueError(
                    "single_masked expects exactly one camera, "
                    f"got {len(clips)} for dataset {ds.name}"
                )
            video = self._single_masked_video(clips[0])
        elif mode == "latent_horizontal":
            return self._latent_horizontal_video(clips)
        elif mode == "horizontal":
            video = self._horizontal_video(clips)
        elif mode == "vertical":
            video = self._vertical_video(clips)
        elif mode == "grid2x2":
            video = self._grid2x2_video(clips)
        elif mode == "robotwin":
            video = self._robotwin_video(clips)
        elif mode in {"head_right_wrist_masked", "head_left_wrist_masked"}:
            video = self._head_one_wrist_video(clips, right=mode == "head_right_wrist_masked")
        else:
            raise ValueError(
                f"Unsupported Pretrain video_concat mode {mode!r} for dataset {ds.name}"
            )
        if tuple(video.shape[-2:]) != self.video_size:
            video = self._resize_clip_to_slot(video, self.video_size)
        video = video.mul(2.0).sub(1.0)
        return video.permute(1, 0, 2, 3).contiguous()

    def _current_vlm_images(self, clips: list[torch.Tensor]) -> torch.Tensor:
        """Letterbox each current camera frame onto a fixed square canvas."""
        frames = [clip[0].clamp(0.0, 1.0) for clip in clips]
        canvas_size = math.isqrt(self.vlm_max_pixels)
        padded = []
        for frame in frames:
            height, width = map(int, frame.shape[-2:])
            scale = min(float(canvas_size) / height, float(canvas_size) / width)
            frame_height = max(1, int(round(height * scale)))
            frame_width = max(1, int(round(width * scale)))
            if tuple(frame.shape[-2:]) != (frame_height, frame_width):
                frame = F.interpolate(
                    frame.unsqueeze(0),
                    size=(frame_height, frame_width),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            pad_height = canvas_size - frame_height
            pad_width = canvas_size - frame_width
            padded.append(
                F.pad(
                    frame,
                    (
                        pad_width // 2,
                        pad_width - pad_width // 2,
                        pad_height // 2,
                        pad_height - pad_height // 2,
                    ),
                )
            )
        return torch.stack(padded, dim=0).mul(255.0).round().to(torch.uint8)

    def _item_video_to_training_tensor(self, item: dict[str, Any]) -> dict[str, Any]:
        return self._batch_frame_videos_to_training_tensors([item])[0]

    def _batch_frame_videos_to_training_tensors(
        self, items: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Decode current window, anchor, and previous-decision recent frame."""
        bundles: list[dict[str, Any] | None] = [None] * len(items)
        grouped: dict[tuple[int, int], list[int]] = {}
        for local_pos, item in enumerate(items):
            ds_pos = int(item["dataset_index"].item())
            episode_index = int(item["episode_index"].item())
            grouped.setdefault((ds_pos, episode_index), []).append(local_pos)

        for (ds_pos, episode_index), positions in grouped.items():
            ds = self.datasets[ds_pos]
            clips_by_pos = {
                pos: {"video": [], "anchor": [], "recent": []}
                for pos in positions
            }
            recent_valid_by_pos: dict[int, bool] = {}
            for video_key in ds.video_keys:
                timestamps_by_pos: dict[int, dict[str, list[float]]] = {}
                decode_keys_by_pos: dict[
                    int, dict[str, tuple[str, int]]
                ] = {}
                unique_timestamps: dict[
                    tuple[str, int], dict[int, float]
                ] = {}
                video_offset = ds._video_timestamp_offset(
                    episode_index, video_key
                )
                segment_start_timestamp = (
                    float(video_offset)
                    + float(ds.episode_start_frame(episode_index))
                    / float(ds.fps)
                )
                for pos in positions:
                    item = items[pos]
                    segment_value = item.get("decoded_segment_index")
                    segment_index = (
                        int(segment_value.reshape(-1)[0].item())
                        if segment_value is not None
                        else int(pos)
                    )
                    sample_indices = self._video_sample_indices_for_item(item)
                    current_timestamps = self._video_timestamps_for_item(
                        ds, item, video_key, sample_indices
                    )
                    start_value = item.get("sample_frame_index")
                    start_frame = (
                        int(start_value.reshape(-1)[0].item())
                        if start_value is not None
                        else 0
                    )
                    recent_offset = (
                        float(self.memory_recent_frame_offset)
                        / float(ds.slow_motion_factor)
                    )
                    recent_valid = float(start_frame) >= recent_offset
                    recent_frame = max(0.0, float(start_frame) - recent_offset)
                    recent_valid_by_pos[pos] = recent_valid
                    groups = {
                        "video": current_timestamps,
                        "anchor": [segment_start_timestamp],
                        "recent": [
                            segment_start_timestamp
                            + recent_frame / float(ds.fps)
                        ],
                    }
                    timestamps_by_pos[pos] = groups
                    decode_keys = {
                        "video": ("video", segment_index),
                        "anchor": ("anchor", 0),
                        "recent": ("recent", segment_index),
                    }
                    decode_keys_by_pos[pos] = decode_keys
                    for name, timestamps in groups.items():
                        decode_group = decode_keys[name]
                        group_timestamps = unique_timestamps.setdefault(
                            decode_group, {}
                        )
                        for timestamp in timestamps:
                            timestamp_key = int(round(timestamp * 1_000_000))
                            group_timestamps.setdefault(
                                timestamp_key, float(timestamp)
                            )

                decoded_groups: dict[tuple[str, int], torch.Tensor] = {}
                timestamp_positions: dict[
                    tuple[str, int], dict[int, int]
                ] = {}
                for decode_group in sorted(unique_timestamps):
                    sorted_items = sorted(
                        unique_timestamps[decode_group].items()
                    )
                    sorted_timestamps = [
                        value for _, value in sorted_items
                    ]
                    timestamp_positions[decode_group] = {
                        timestamp_key: index
                        for index, (timestamp_key, _) in enumerate(sorted_items)
                    }
                source = None
                try:
                    block_cache = ds._episode_video_block_cache(
                        episode_index, video_key
                    )
                    source = ds._open_episode_video(
                        episode_index,
                        video_key,
                        block_cache=block_cache,
                    )
                    grouped_timestamps = {
                        decode_group: [
                            value
                            for _, value in sorted(
                                unique_timestamps[decode_group].items()
                            )
                        ]
                        for decode_group in unique_timestamps
                    }
                    with source:
                        decoded_groups = (
                            pretrain_loader._decode_mp4_groups_lerobot(
                                source,
                                grouped_timestamps,
                                fps=ds.fps,
                            )
                        )
                except Exception as exc:
                    all_timestamps = [
                        timestamp
                        for timestamps in unique_timestamps.values()
                        for timestamp in timestamps.values()
                    ]
                    context = self._video_decode_context(
                        ds,
                        episode_index,
                        video_key,
                        positions=positions,
                        items=items,
                        raw_bytes=(
                            getattr(source, "bytes_read", None)
                            if source is not None
                            else None
                        ),
                        timestamps=all_timestamps,
                    )
                    raise RuntimeError(
                        "Pretrain batched frame video decode failed: "
                        f"groups={list(unique_timestamps)} {context}"
                    ) from exc
                for pos in positions:
                    for name, timestamps in timestamps_by_pos[pos].items():
                        decode_group = decode_keys_by_pos[pos][name]
                        take = [
                            timestamp_positions[decode_group][
                                int(round(timestamp * 1_000_000))
                            ]
                            for timestamp in timestamps
                        ]
                        clips_by_pos[pos][name].append(
                            decoded_groups[decode_group][take].contiguous()
                        )
                del decoded_groups

            for pos in positions:
                current_clips = clips_by_pos[pos]["video"]
                bundle = {
                    name: self._clips_to_training_tensor(ds, clips)
                    for name, clips in clips_by_pos[pos].items()
                }
                bundle["vlm_current_images"] = self._current_vlm_images(
                    current_clips
                )
                bundle["spatial_valid_mask"] = self._video_spatial_valid_mask(ds)
                bundle["anchor_is_pad"] = torch.zeros(
                    (1,), dtype=torch.bool
                )
                bundle["recent_is_pad"] = torch.tensor(
                    [not recent_valid_by_pos[pos]], dtype=torch.bool
                )
                bundles[pos] = bundle

        if any(bundle is None for bundle in bundles):
            missing = [
                idx for idx, bundle in enumerate(bundles) if bundle is None
            ]
            raise RuntimeError(
                f"Failed to build Pretrain frame video bundles for {missing}"
            )
        return [bundle for bundle in bundles if bundle is not None]

    def _batch_videos_to_training_tensors(
        self, items: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return self._batch_frame_videos_to_training_tensors(items)

    def _pack_training_item(
        self,
        idx: int,
        item: dict[str, Any],
        video: dict[str, Any],
        *,
        item_s: float = 0.0,
        video_s: float = 0.0,
        total_t0: float | None = None,
    ) -> dict[str, Any]:
        profile = bool(getattr(self, "benchmark_stage_times", False))
        if not isinstance(video, dict):
            raise TypeError("Pretrain frame training expects a video bundle.")
        memory_video_anchor = video["anchor"]
        memory_video_recent = video["recent"]
        memory_video_anchor_is_pad = video["anchor_is_pad"]
        memory_video_recent_is_pad = video["recent_is_pad"]
        vlm_current_images = video["vlm_current_images"]
        video_spatial_valid_mask = video["spatial_valid_mask"]
        video = video["video"]

        t0 = time.perf_counter() if profile else 0.0
        state = item["observation.state"].to(torch.float32)
        action = item["action"].to(torch.float32)
        state_pad = item["observation.state_is_pad"].to(torch.bool)
        action_pad = item["action_is_pad"].to(torch.bool)
        state_dim_pad = item["observation.state_dim_is_pad"].to(torch.bool)
        action_dim_pad = item["action_dim_is_pad"].to(torch.bool)
        action_dim_loss_weight = item.get("action_dim_loss_weight")
        action_loss_weight = item.get("action_loss_weight")
        action_loss_weighted_valid_cells = item.get(
            "action_loss_weighted_valid_cells"
        )
        state_mask = item.get("observation.state_mask")
        action_mask = item.get("action_mask")
        if state_mask is not None:
            state_mask = state_mask.to(torch.bool)
            state_mask &= ~state_pad[:, None]
            state_mask &= ~state_dim_pad[None, :]
            state[~state_mask] = 0.0
        if action_mask is not None:
            action_mask = action_mask.to(torch.bool)
            action_mask &= ~action_pad[:, None]
            action_mask &= ~action_dim_pad[None, :]
            action[~action_mask] = 0.0
        stats_group = str(
            item.get("stats_group")
            or self.stats_groups[int(item["dataset_index"].item())]
        )
        prep_s = time.perf_counter() - t0 if profile else 0.0

        t0 = time.perf_counter() if profile else 0.0
        if self.grouped_normalizer is not None:
            state = self.grouped_normalizer.forward(
                state, group=stats_group, kind="state"
            )
            action = self.grouped_normalizer.forward(
                action, group=stats_group, kind="action"
            )
            state = self._apply_post_normalize_transforms(
                state, group=stats_group, kind="state"
            )
            action = self._apply_post_normalize_transforms(
                action, group=stats_group, kind="action"
            )
            if bool(state_dim_pad.any().item()):
                state[:, state_dim_pad] = 0.0
            if bool(action_dim_pad.any().item()):
                action[:, action_dim_pad] = 0.0
            if bool(state_pad.any().item()):
                state[state_pad] = 0.0
            if bool(action_pad.any().item()):
                action[action_pad] = 0.0
            if state_mask is not None:
                state[~state_mask] = 0.0
            if action_mask is not None:
                action[~action_mask] = 0.0
        norm_s = time.perf_counter() - t0 if profile else 0.0

        t0 = time.perf_counter() if profile else 0.0
        action, action_dim_pad = self._project_feature_dim(
            action,
            action_dim_pad,
            output_dim=self.action_output_dim,
        )
        state, state_dim_pad = self._project_feature_dim(
            state,
            state_dim_pad,
            output_dim=self.proprio_output_dim,
        )
        context = item.get("context")
        context_mask = item.get("context_mask")
        if context is None or context_mask is None:
            if self.text_context_required:
                raise FileNotFoundError(
                    "Pretrain sample is missing cached context/context_mask. "
                    "Precompute text embeddings for configs/pretrain/dataset.yaml first."
                )
            context = torch.zeros((128, 4096), dtype=torch.bfloat16)
            context_mask = torch.zeros((128,), dtype=torch.bool)

        action_len = action.shape[0]
        proprio_pad = state_pad[:action_len]
        proprio = state[:action_len]
        proprio_dim_pad = state_dim_pad
        video_frames = int(video.shape[1] if video.ndim == 4 else video.shape[2])
        image_pad_indices = self._video_indices(state_pad.shape[0])[
            :video_frames
        ]
        prompt = str(item.get("prompt", ""))
        dataset = self.datasets[int(item["dataset_index"].item())]
        canvas_layout = self._video_concat_mode(dataset)
        weighted_pool_families = tuple(
            getattr(
                self,
                "action_loss_weighted_valid_cells_families",
                (),
            )
        )
        action_loss_weighted_valid_cells_pool = None
        if weighted_pool_families:
            action_loss_weighted_valid_cells_pool = torch.zeros(
                len(weighted_pool_families), dtype=torch.bool
            )
            is_weighted = bool(
                action_loss_weighted_valid_cells is not None
                and action_loss_weighted_valid_cells.to(torch.bool).item()
            )
            if is_weighted:
                family = str(
                    dataset.spec.sampling_family
                    or "__weighted_default__"
                )
                pool_index = getattr(
                    self,
                    "action_loss_weighted_valid_cells_family_to_pool",
                    {},
                ).get(family)
                if pool_index is None:
                    raise ValueError(
                        "Weighted action-loss sample has no configured family "
                        f"pool: dataset={dataset.name} family={family!r} "
                        f"pools={weighted_pool_families}"
                    )
                action_loss_weighted_valid_cells_pool[int(pool_index)] = True
        pack_s = time.perf_counter() - t0 if profile else 0.0

        out = {
            "video": video,
            "memory_video_anchor": memory_video_anchor,
            "memory_video_anchor_is_pad": memory_video_anchor_is_pad,
            "memory_video_recent": memory_video_recent,
            "memory_video_recent_is_pad": memory_video_recent_is_pad,
            "action": action,
            "proprio": proprio,
            "prompt": prompt,
            "context": context,
            "context_mask": context_mask,
            "vlm_current_images": vlm_current_images,
            "video_layout": "single",
            "video_view_names": "canvas",
            "video_canvas_layout": canvas_layout,
            "video_canvas_view_names": "|".join(dataset.video_keys),
            "video_spatial_valid_mask": video_spatial_valid_mask,
            "image_is_pad": state_pad[image_pad_indices],
            "action_is_pad": action_pad,
            "proprio_is_pad": proprio_pad,
            "action_dim_is_pad": action_dim_pad,
            "proprio_dim_is_pad": proprio_dim_pad,
            "idx": torch.tensor(int(idx), dtype=torch.long),
            "dataset_name": item.get("dataset_name", ""),
            "stats_group": stats_group,
        }
        if action_dim_loss_weight is not None:
            # Canonical action dimension weights: (D,).
            out["action_dim_loss_weight"] = action_dim_loss_weight.to(
                torch.float32
            )
        if action_loss_weight is not None:
            out["action_loss_weight"] = action_loss_weight.to(torch.float32)
        if action_loss_weighted_valid_cells is not None:
            out["action_loss_weighted_valid_cells"] = (
                action_loss_weighted_valid_cells.to(torch.bool)
            )
        if action_loss_weighted_valid_cells_pool is not None:
            out["action_loss_weighted_valid_cells_pool"] = (
                action_loss_weighted_valid_cells_pool
            )
        if action_mask is not None:
            out["action_mask"] = action_mask
        if profile:
            total_s = (
                time.perf_counter() - total_t0
                if total_t0 is not None
                else item_s + video_s + prep_s + norm_s + pack_s
            )
            out["_bench_times"] = torch.tensor(
                [item_s, video_s, prep_s, norm_s, pack_s, total_s],
                dtype=torch.float32,
            )
        return out

    def __getitem__(self, idx: int) -> dict[str, Any]:
        idx = int(idx)

        profile = bool(getattr(self, "benchmark_stage_times", False))
        total_t0 = time.perf_counter() if profile else None

        t0 = time.perf_counter() if profile else 0.0
        item = self.mixed_dataset[int(idx)]
        item_s = time.perf_counter() - t0 if profile else 0.0

        t0 = time.perf_counter() if profile else 0.0
        try:
            video = self._item_video_to_training_tensor_with_context(item)
        except Exception:
            if self.skip_bad_videos:
                return self._pack_single_index_with_retries(int(idx))
            raise
        video_s = time.perf_counter() - t0 if profile else 0.0
        try:
            return self._pack_training_item(
                int(idx),
                item,
                video,
                item_s=item_s,
                video_s=video_s,
                total_t0=total_t0,
            )
        except Exception:
            if self.skip_bad_videos:
                return self._pack_single_index_with_retries(int(idx))
            raise

    def __getitems__(self, indices: list[int]) -> list[dict[str, Any]]:
        if not indices:
            return []
        profile = bool(getattr(self, "benchmark_stage_times", False))
        t0 = time.perf_counter() if profile else 0.0
        items = self.mixed_dataset.get_items([int(idx) for idx in indices])
        item_s = (
            (time.perf_counter() - t0) / max(len(indices), 1)
            if profile
            else 0.0
        )
        item_times = [item_s] * len(indices)

        t0 = time.perf_counter() if profile else 0.0
        try:
            videos = self._batch_videos_to_training_tensors(items)
        except Exception as exc:
            if not self.skip_bad_videos:
                raise
            self._log_bad_video(
                f"batched video decode failed for indices={list(map(int, indices[:8]))} "
                f"fallback_to_individual_retries error={type(exc).__name__}: {exc}"
            )
            return [self._pack_single_index_with_retries(int(idx)) for idx in indices]
        video_total_s = time.perf_counter() - t0 if profile else 0.0
        video_s = video_total_s / max(len(indices), 1)

        out: list[dict[str, Any]] = []
        for idx, item, video, item_time in zip(indices, items, videos, item_times):
            try:
                out.append(
                    self._pack_training_item(
                        int(idx),
                        item,
                        video,
                        item_s=item_time,
                        video_s=video_s,
                    )
                )
            except Exception as exc:
                if not self.skip_bad_videos:
                    raise
                self._log_bad_video(
                    f"packed sample failed for idx={int(idx)} {self._item_context(item)} "
                    f"fallback_to_individual_retries error={type(exc).__name__}: {exc}"
                )
                out.append(self._pack_single_index_with_retries(int(idx)))
        pa.default_memory_pool().release_unused()
        return out
