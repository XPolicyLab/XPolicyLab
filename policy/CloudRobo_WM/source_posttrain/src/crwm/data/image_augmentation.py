import torch
import torch.nn.functional as F
from .required_modalities import RequiredModalitiesDataset
from .schema import CAMERAS
from .image_transform import ClipImageAugmentation
from crwm.pointmap_validity import depth_valid_mask

class AugmentedRequiredModalitiesDataset(RequiredModalitiesDataset):

    def __init__(self, *, augment_rgb_dino_depth=False, augment_camera_keys=None, augment_crop_scale=0.95, augment_rotate_degrees=5.0, augment_color_jitter_brightness=0.3, augment_color_jitter_contrast=0.4, augment_color_jitter_saturation=0.5, augment_color_jitter_hue=0.0, **kwargs):
        split = kwargs.get('split', 'train')
        self.image_augmentation = ClipImageAugmentation(enabled=augment_rgb_dino_depth, is_training=split == 'train', camera_keys=augment_camera_keys, crop_scale=augment_crop_scale, rotate_degrees=augment_rotate_degrees, brightness=augment_color_jitter_brightness, contrast=augment_color_jitter_contrast, saturation=augment_color_jitter_saturation, hue=augment_color_jitter_hue)
        super().__init__(**kwargs)
        if self.image_augmentation.enabled:
            for adapter in self.adapters:
                if adapter.source['adapter'] != 'robotwin':
                    raise ValueError('This image augmentation recipe is audited for RoboTwin only')

    @property
    def checkpoint_contract(self):
        result = dict(super().checkpoint_contract)
        if self.image_augmentation.enabled:
            result['image_augmentation'] = {**self.image_augmentation.metadata(), 'full_K_per_example': True, 'pointmap_mask': 'recomputed_from_transformed_depth'}
        return result

    def __getitem__(self, item):
        sample = super().__getitem__(item)
        if not self.image_augmentation.enabled:
            return sample
        sample = self.image_augmentation(sample, camera_order=CAMERAS)
        mask = torch.zeros_like(sample['pointmap_pixel_valid'])
        for i, cam in enumerate(CAMERAS):
            value = sample['per_cam_depth'][cam]
            valid = depth_valid_mask(value, mode=self.pointmap_depth_validity)
            hw = (256, 320) if i == 0 else (128, 160)
            m = F.interpolate(valid[:, None].float(), size=hw, mode='nearest')[:, 0].bool()
            if i == 0:
                mask[:, :256] = m
            else:
                mask[:, 256:, (i - 1) * 160:i * 160] = m
        sample['pointmap_pixel_valid'] = mask
        return sample
