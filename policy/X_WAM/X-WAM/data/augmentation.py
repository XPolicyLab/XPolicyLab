"""
Video augmentation for multi-view robot datasets.

Data format convention:
    video / depths: Tensor[V, T, C, H, W]
        V - number of camera views
        T - sequence length (temporal dimension)
        C - channels (3)
        H, W - spatial dimensions
    Value range: [-1, 1] (as produced by RobocasaDataset)
"""

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from typing import Tuple


class VideoAugmentation:
    """
    Applies two augmentations to multi-view RGB+depth data:

    1. **Per-view random crop + resize**
       - Each view gets its own independently sampled crop region.
       - The same crop region is applied to *all T frames* within a view
         (temporal consistency), and to both the RGB and depth of that view.
       - ``crop_ratio`` controls the side-length of the crop relative to the
         original image; the crop is then resized back to the original size.

    2. **Global colour jitter (RGB only)**
       - A single set of jitter parameters is sampled once and applied
         identically to *every* view and *every* timestep, so colour
         statistics remain consistent across the sequence.
    """

    def __init__(
        self,
        crop_ratio: float = 0.95,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.05,
    ):
        """
        Args:
            crop_ratio:  Fraction of the spatial dimensions kept after cropping.
                         E.g. 0.95 means a 242×242 crop from a 256×256 image.
            brightness:  Maximum additive brightness factor (symmetric around 1).
            contrast:    Maximum additive contrast factor (symmetric around 1).
            saturation:  Maximum additive saturation factor (symmetric around 1).
            hue:         Maximum absolute hue shift in [-hue, hue] (must be ≤ 0.5).
        """
        assert 0.0 < crop_ratio <= 1.0, "crop_ratio must be in (0, 1]"
        assert 0.0 <= hue <= 0.5, "hue must be in [0, 0.5]"

        self.crop_ratio = crop_ratio
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __call__(self, data: dict) -> dict:
        """
        Apply augmentations in-place on a copy of *data*.

        Expects ``data`` to contain at minimum:
            ``'video'``  – Tensor[V, T, C, H, W] in [-1, 1]
            ``'depths'`` – Tensor[V, T, C, H, W] in [-1, 1]

        Returns a new dict with augmented ``'video'`` and ``'depths'``.
        """
        video = data["video"]  # [V, T, C, H, W]
        depths = data["depths"]  # [V, T, C, H, W]

        H, W = video.shape[-2], video.shape[-1]

        video, depths = self._apply_random_crop(video, depths, H, W)
        video = self._apply_color_jitter(video)

        out = dict(data)
        out["video"] = video
        out["depths"] = depths
        return out

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_random_crop(
        self,
        video: torch.Tensor,
        depths: torch.Tensor,
        H: int,
        W: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        For each view independently, sample one crop region and apply it to
        all T frames of that view (both RGB and depth).

        The crop is a rectangle of size (crop_h × crop_w) that is bilinearly
        resized back to (H × W) for RGB and nearest-neighbour resized for depth
        to avoid blending depth discontinuities.
        """
        V, T = video.shape[:2]
        crop_h = max(1, int(H * self.crop_ratio))
        crop_w = max(1, int(W * self.crop_ratio))

        if crop_h == H and crop_w == W:
            return video, depths

        aug_video = []
        aug_depths = []

        for v in range(V):
            # Sample crop origin once per view and resize all frames together.
            top = torch.randint(0, H - crop_h + 1, ()).item()
            left = torch.randint(0, W - crop_w + 1, ()).item()

            rgb_crop = video[v, :, :, top : top + crop_h, left : left + crop_w]
            depth_crop = depths[v, :, :, top : top + crop_h, left : left + crop_w]

            rgb_crop = F.interpolate(
                rgb_crop,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
                antialias=False,
            )
            depth_crop = F.interpolate(
                depth_crop,
                size=(H, W),
                mode="nearest",
            )

            aug_video.append(rgb_crop)  # [T, C, H, W]
            aug_depths.append(depth_crop)

        return torch.stack(aug_video), torch.stack(aug_depths)  # [V, T, C, H, W]

    def _apply_color_jitter(self, video: torch.Tensor) -> torch.Tensor:
        """
        Sample a single set of colour-jitter parameters and apply them
        uniformly to every (view, time) frame.

        ``video`` is expected in [-1, 1]; it is converted to [0, 1] for the
        TF operations and converted back afterwards.
        """
        V, T, C, H, W = video.shape

        # Sample factors once for the whole batch.
        brightness_factor = self._sample_factor(self.brightness)
        contrast_factor = self._sample_factor(self.contrast)
        saturation_factor = self._sample_factor(self.saturation)
        hue_factor = float(torch.empty(1).uniform_(-self.hue, self.hue)) if self.hue > 0 else None

        # Convert [-1, 1] → [0, 1].
        imgs = (video + 1.0) / 2.0  # [V, T, C, H, W]

        # Flatten views and time so that each 2-D image is processed the same way.
        flat = imgs.flatten(0, 1)  # [V*T, C, H, W]

        # Apply the four transforms in a random order (matching ColorJitter behaviour).
        for fn_id in torch.randperm(4).tolist():
            if fn_id == 0 and brightness_factor is not None:
                flat = TF.adjust_brightness(flat, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                flat = TF.adjust_contrast(flat, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                flat = TF.adjust_saturation(flat, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                flat = TF.adjust_hue(flat, hue_factor)

        imgs = flat.reshape(V, T, C, H, W).clamp(0.0, 1.0)

        # Convert [0, 1] → [-1, 1].
        return imgs * 2.0 - 1.0

    @staticmethod
    def _sample_factor(magnitude: float):
        """Sample a multiplicative factor uniformly in [1-magnitude, 1+magnitude]."""
        if magnitude <= 0:
            return None
        lo = max(0.0, 1.0 - magnitude)
        hi = 1.0 + magnitude
        return float(torch.empty(1).uniform_(lo, hi))
