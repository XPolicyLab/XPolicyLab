import torch
import torch.nn.functional as F
from .prompt_augmentation import PromptAugmentedInstructionDataset
from .schema import CAMERAS
from cloudrobo_wm.pointmap_validity import depth_valid_mask
from cloudrobo_wm.datasets.robotwin_image_augmentation import resolve_image_augmentation, augmentation_contract, augment_rgbd, reject_unaugmented_feature_caches

class RichPromptAugmentedInstructionDataset(PromptAugmentedInstructionDataset):

    def __init__(self, *, image_augmentation=None, **kwargs):
        self.rich_augmentation = resolve_image_augmentation(image_augmentation)
        if kwargs.get('augment_rgb_dino_depth', False):
            raise ValueError('Old and new augmentation must never both run')
        self._native_rgb = {}
        self._capture_native = False
        super().__init__(**kwargs)
        if self.rich_augmentation:
            for adapter in self.adapters:
                reject_unaugmented_feature_caches(adapter.source)

    def __getstate__(self):
        state = super().__getstate__()
        state['_native_rgb'] = {}
        state['_capture_native'] = False
        return state

    def _decoded_rgb_observer(self, slot, frames):
        if self._capture_native:
            self._native_rgb[slot] = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255

    def _augment_sample(self, sample, native):
        if not self.rich_augmentation:
            return sample
        if set(native) != set(CAMERAS) or not bool(sample['camera_valid'].all()):
            raise ValueError('This RobotWin recipe requires all three native cameras')
        rgb, depth, K, applied, masks = augment_rgbd(native, sample['per_cam_depth'], sample['camera_intrinsics'], self.rich_augmentation, index=int(sample['window_index']), occurrence=int(sample['sample_position']), return_valid_masks=True, replay=getattr(self, '_augmentation_replay', None))
        if applied:
            sample['per_cam'] = {cam: F.interpolate(rgb[cam] * 2 - 1, size=(256, 320) if i == 0 else (224, 224), mode='bilinear', align_corners=False, antialias=True).permute(1, 0, 2, 3) for i, cam in enumerate(CAMERAS)}
            sample['per_cam_depth'] = depth
            sample['camera_intrinsics'] = K
            canvas = torch.zeros_like(sample['pointmap_pixel_valid'])
            for i, cam in enumerate(CAMERAS):
                valid = depth_valid_mask(depth[cam], mode=self.pointmap_depth_validity)
                hw = (256, 320) if i == 0 else (128, 160)
                valid = F.interpolate(valid[:, None].float(), size=hw, mode='nearest')[:, 0].bool()
                if i == 0:
                    canvas[:, :256] = valid
                else:
                    canvas[:, 256:, (i - 1) * 160:i * 160] = valid
            sample['pointmap_pixel_valid'] = canvas
        else:
            masks = {cam: torch.ones(v.shape[-2:], dtype=torch.bool) for cam, v in native.items()}
        sample['image_augmentation_applied'] = torch.tensor(applied)
        sample['per_cam_rgb_valid'] = masks
        return sample

    def __getitem__(self, item):
        self._native_rgb = {}
        self._capture_native = self.rich_augmentation is not None
        try:
            sample = super().__getitem__(item)
            return self._augment_sample(sample, self._native_rgb)
        finally:
            self._capture_native = False
            self._native_rgb = {}

    @property
    def checkpoint_contract(self):
        value = dict(super().checkpoint_contract)
        if self.rich_augmentation:
            value['training_image_augmentation'] = augmentation_contract(self.rich_augmentation)
        return value
