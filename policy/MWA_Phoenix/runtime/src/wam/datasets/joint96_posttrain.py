"""Current-only LIBERO/RoboDojo windows using the native canonical80 processor."""

from pathlib import Path
from bisect import bisect_right

import torch

from .lerobot.robot_video_dataset import DEFAULT_PROMPT, RobotVideoDataset


def pack_camera_slots(video, slots):
    slots = tuple(int(slot) for slot in slots)
    if (video.ndim != 5 or len(slots) != video.shape[0] or len(set(slots)) != len(slots)
            or 0 not in slots or any(slot not in (0, 1, 2) for slot in slots)):
        raise ValueError("Expected independent [views,C,T,H,W] images and unique main/left/right slots")
    result = video.new_zeros((3, *video.shape[1:]))
    mask = torch.ones(3, dtype=torch.bool, device=video.device)
    result[list(slots)] = video
    mask[list(slots)] = False
    return result, mask


def decode_joint96_actions(processor, action, proprio):
    """Inverse the native 80D mapping without the old memory-prefix crop."""
    if action.ndim != 3 or proprio.ndim != 3 or action.shape[-1] != 80 or proprio.shape[-1] != 80:
        raise ValueError("Expected batched action/proprio with last dimension 80")
    data = processor.action_state_merger.backward({"action": action, "state": proprio})
    data = processor.normalizer.backward(data)
    for transform in reversed(processor.action_state_transforms or []):
        data = transform.backward(data)
    return data["action"]["default"]


class Joint96PosttrainDataset(RobotVideoDataset):
    """Reuse native action normalization/inverse mapping, without memory queries."""

    def __init__(self, *, camera_slots, domain, validation_panel_size=128, **kwargs):
        if domain not in ("libero", "robodojo", "robotwin"):
            raise ValueError("Joint96 posttraining supports libero, robodojo or robotwin")
        expected_slots = (0, 2) if domain == "libero" else (0, 1, 2)
        if tuple(camera_slots) != expected_slots:
            raise ValueError(f"{domain} requires camera_slots={expected_slots}")
        if not Path(str(kwargs.get("pretrained_norm_stats", ""))).is_file():
            raise FileNotFoundError("Prepare explicit posttraining normalization with tools/posttrain_stats.py first")
        for key, value in {"num_frames": 33, "action_video_freq_ratio": 4, "global_sample_stride": 1,
                           "memory_video_anchor_size": 0, "memory_recent_frame_offset": 0,
                           "concat_multi_camera": "latent_horizontal", "single_canvas": False,
                           "return_vlm_current_images": False}.items():
            if kwargs.get(key, value) != value:
                raise ValueError(f"Joint96 posttraining requires {key}={value!r}")
            kwargs[key] = value
        if list(kwargs.get("video_size", [])) != [256, 256]:
            raise ValueError("Joint96 requires independent 256x256 views")
        self.camera_slots, self.domain = expected_slots, domain
        super().__init__(**kwargs)
        # The parent clamps this to >=1 for its historical memory path. Our
        # __getitem__ queries explicit current-only offsets instead.
        self.memory_recent_frame_offset = 0
        # At least the first future sampled image (offset 4) must be real.
        # Keep other tail padding; never resample it into an unrelated episode.
        self._valid_raw_starts, self._valid_ranges = [], []
        cursor = 0
        for start, end in zip(self._episode_starts, self._episode_ends):
            count = max(0, end - start - self.action_video_freq_ratio)
            if count:
                self._valid_raw_starts.append(start)
                self._valid_ranges.append((cursor, cursor + count))
                cursor += count
        self._valid_ends = [end for _, end in self._valid_ranges]
        if not cursor:
            raise ValueError("No windows contain a real future video frame")
        if not self.is_training_set:
            count = min(int(validation_panel_size), len(self))
            self.fixed_validation_panel = [
                {"index": int((i + .5) * len(self) / count), "source": domain, "seed": 20260929 + i}
                for i in range(count)
            ]

    def __len__(self):
        return self._valid_ends[-1] if hasattr(self, "_valid_ends") else super().__len__()

    @property
    def sample_episode_ranges(self):
        return tuple(self._valid_ranges)

    def _raw_index(self, index):
        index = int(index)
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode = bisect_right(self._valid_ends, index)
        return self._valid_raw_starts[episode] + index - self._valid_ranges[episode][0]

    def _read_window(self, raw_index):
        return self.lerobot_dataset.get_item_with_offsets(
            raw_index, image_offsets=list(range(0, 33, 4)),
            state_offsets=list(range(33)), action_offsets=list(range(32)),
        )

    def _visual_inputs(self, sample, raw_index):
        video = self._format_video_window(
            sample["pixel_values"], color_jitter_params=self._sample_color_jitter_params(),
            camera_jitter_params=self._sample_camera_jitter_params(),
            sensor_noise_params=self._sample_sensor_noise_params(),
        )
        video, camera_is_pad = pack_camera_slots(video, self.camera_slots)
        if video.shape != (3, 3, 9, 256, 256):
            raise ValueError(f"Invalid Joint96 video shape: {tuple(video.shape)}")
        return {"video": video, "camera_is_pad": camera_is_pad}

    def __getitem__(self, index):
        raw_index = self._raw_index(index)
        sample = self._read_window(raw_index)
        if int(sample["idx"]) != raw_index:
            raise RuntimeError("Posttraining reader refuses random replacement of a failed sample")
        visual = self._visual_inputs(sample, raw_index)
        action = sample["action"]
        proprio = sample["proprio"][:1]
        if action.shape != (32, 80) or proprio.shape != (1, 80):
            raise ValueError("Posttraining requires native processor output action[32,80], proprio[1,80]")
        action_pad = sample["action_is_pad"].bool()
        action_dim_pad = sample["action_dim_is_pad"].bool()
        proprio_dim_pad = sample["proprio_dim_is_pad"].bool()
        valid = ~action_pad[:, None] & ~action_dim_pad[None, :]
        prompt = DEFAULT_PROMPT.format(task=self.override_instruction or sample["instruction"])
        context, context_mask = self._get_cached_text_context(prompt)
        proprio = self._apply_proprio_noise(proprio, sample["proprio_is_pad"][:1], proprio_dim_pad)
        return {
            "sample_id": raw_index, **visual,
            "action": action.masked_fill(~valid, 0),
            "action_mask": valid, "action_is_pad": action_pad, "action_dim_is_pad": action_dim_pad,
            "proprio": proprio.masked_fill(proprio_dim_pad[None, :], 0),
            "proprio_is_pad": sample["proprio_is_pad"][:1], "proprio_dim_is_pad": proprio_dim_pad,
            "image_is_pad": sample["image_is_pad"].bool(),
            "prompt": prompt, "context": context, "context_mask": context_mask,
            "video_layout": "latent_horizontal", "video_view_names": "main|left|right",
        }
