from types import SimpleNamespace
from .required_modalities import RequiredModalitiesDataset

class AugmentedRequiredModalitiesDataset(RequiredModalitiesDataset):
    def __init__(self, *, augment_rgb_dino_depth=False, augment_camera_keys=None, augment_crop_scale=0.95, augment_rotate_degrees=5.0, augment_color_jitter_brightness=0.3, augment_color_jitter_contrast=0.4, augment_color_jitter_saturation=0.5, augment_color_jitter_hue=0.0, **kwargs):
        if augment_rgb_dino_depth:
            raise ValueError("AS2 uses only the admitted rich RGBD augmentation")
        self.image_augmentation=SimpleNamespace(enabled=False)
        super().__init__(**kwargs)
