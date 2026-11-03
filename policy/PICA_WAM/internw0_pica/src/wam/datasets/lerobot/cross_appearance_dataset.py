"""Paired nominal/perturbed views (PICA-WAM) with InternW0-delta's frame windows and state noise."""
import torch
from hydra.utils import instantiate

from .robot_video_dataset import RobotVideoDataset


class CrossAppearanceDataset(RobotVideoDataset):
    def __init__(self, *args, video_augmentation, emit_canonical_video=True,
                 image_domain_jitter=None, **kwargs):
        # RoboTwin's base recipe supplies this domain-only option. The paired views replace it,
        # and RobotVideoDataset does not accept it even when it is disabled.
        if image_domain_jitter and image_domain_jitter.get("enabled", False):
            raise ValueError("Cross-appearance views replace image_domain_jitter; disable it in the data config.")
        super().__init__(*args, **kwargs)
        self.emit_canonical_video = bool(emit_canonical_video)
        if any((self.image_color_jitter, self.image_camera_jitter, self.image_sensor_noise)):
            raise ValueError("Cross-appearance views replace the legacy image augmentation settings.")
        self.video_augmentation = (
            instantiate(video_augmentation) if not isinstance(video_augmentation, torch.nn.Module)
            else video_augmentation
        )
        if self.video_augmentation.canonical_geometry != "none":
            raise ValueError("PICA alignment requires the untouched nominal teacher view.")

    def _prepare_training_views(self, sample):
        raw = sample["pixel_values"]
        if not self.is_training_set:
            return sample, {}
        if self.emit_canonical_video:
            augmented, canonical = self.video_augmentation.forward_split(raw)
        else:
            augmented = self.video_augmentation(raw)
        sample = dict(sample, pixel_values=augmented)
        memory_count = self.memory_video_anchor_size + 1
        current = slice(memory_count, memory_count + len(self.video_sample_indices))
        # Reuse the actual student pixels; no second draw or conflicting VLM augmentation.
        if self.return_vlm_current_images:
            sample["vlm_current_images"] = augmented[:, memory_count].float() * 255.0
        if not self.emit_canonical_video:
            return sample, {}
        views = {
            "video_canonical": self._format_video_window(self._slice_video_time(canonical, current)),
            "memory_video_anchor_canonical": self._format_video_window(
                self._slice_video_time(canonical, slice(0, self.memory_video_anchor_size))
            ),
            "memory_video_recent_canonical": self._format_video_window(
                self._slice_video_time(canonical, slice(self.memory_video_anchor_size, memory_count))
            ),
        }
        return sample, views
