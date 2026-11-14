"""Six native RGB observations with optional past-only lighting."""
import copy
import numpy as np
import torch
import torch.nn.functional as F
from .rich_image_augmentation import RichPromptAugmentedInstructionDataset
from .schema import CAMERAS
from cloudrobo_wm.datasets.robotwin_image_augmentation import photometric, warp
from cloudrobo_wm.datasets.history_light_augmentation import resolve_history_light_augmentation, history_light_contract, sample_history_light, apply_history_light
from cloudrobo_wm.models.history_attention import HistoryConfig
from cloudrobo_wm.history_sampling import history_frame_indices
from cloudrobo_wm.per_cam_compose import compose_from_per_cam
from cloudrobo_wm.composite_layouts import get_layout

def history_indices(anchor, sampling='uniform_stride4_v1'):
    if anchor < 0:
        raise ValueError('Episode anchor must be nonnegative')
    indices = np.asarray(history_frame_indices(anchor, sampling), dtype=np.int64)
    return (indices, indices - anchor)

class HistoryRichPromptDataset(RichPromptAugmentedInstructionDataset):

    def __init__(self, *, history=None, history_light_augmentation=None, **kwargs):
        self.history_light_augmentation = resolve_history_light_augmentation(history_light_augmentation)
        self._last_history_lighting = None
        self.history_config = HistoryConfig.parse(history)
        if self.history_config.mode == 'none':
            raise ValueError('Use the original dataset for the no-history control')
        if kwargs.get('stride', 1) != 1:
            raise ValueError('History v1 is defined in native steps and requires stride=1')
        super().__init__(**kwargs)
        if any((a.source['adapter'] != 'robotwin' for a in self.adapters)):
            raise ValueError('History v1 is a RobotWin-only recipe')

    def _augment_sample(self, sample, native):
        index = int(sample['window_index'])
        leaf, local = self.locate(index)
        adapter = self.adapters[leaf]
        ep, anchor, _ = adapter.index.locate(local)
        indices, offsets = history_indices(anchor, self.history_config.sampling)
        cfg = getattr(self, 'history_light_augmentation', None)
        lighting = sample_history_light(cfg, index=index, occurrence=int(sample['sample_position']), frame_indices=indices, anchor=anchor) if cfg is not None else dict(gate_selected=False, by_frame={})
        by_frame = lighting['by_frame']
        lighting['applied'] = bool(by_frame)
        lighting['cameras'] = {}
        replay = {}
        self._augmentation_replay = replay
        try:
            sample = super()._augment_sample(sample, native)
        finally:
            self._augmentation_replay = None
        saved_native, saved_capture = (self._native_rgb, self._capture_native)
        self._native_rgb, self._capture_native = ({}, True)
        try:
            if by_frame:
                resized = {}
                for cam in CAMERAS:
                    frames = self._decode_rgb(adapter, ep['cameras'][cam], indices, cam, ep)
                    self._decoded_rgb_observer(cam, frames)
            else:
                resized = {cam: self._video(adapter, ep['cameras'][cam], indices, cam, ep) for cam in CAMERAS}
            raw_history = self._native_rgb
        finally:
            self._native_rgb, self._capture_native = (saved_native, saved_capture)
        if replay:
            for i, cam in enumerate(CAMERAS):
                row = replay[cam]
                image = raw_history[cam]
                if row['matrix'] is not None:
                    image, _ = warp(image, row['matrix'], mode='bilinear')
                image = photometric(image, row['params'], self.rich_augmentation, statistics=row['statistics'])
                if by_frame:
                    stats = lighting['cameras'].setdefault(cam, {})
                    image = apply_history_light(image, indices, anchor, by_frame, record=stats)
                if row['valid'] is not None:
                    image = torch.where(row['valid'][None, None], image, 0)
                resized[cam] = F.interpolate(image * 2 - 1, size=(256, 320) if i == 0 else (224, 224), mode='bilinear', align_corners=False, antialias=True).permute(1, 0, 2, 3)
        elif by_frame:
            for i, cam in enumerate(CAMERAS):
                stats = lighting['cameras'].setdefault(cam, {})
                image = apply_history_light(raw_history[cam], indices, anchor, by_frame, record=stats)
                resized[cam] = F.interpolate(image * 2 - 1, size=(256, 320) if i == 0 else (224, 224), mode='bilinear', align_corners=False, antialias=True).permute(1, 0, 2, 3)
        self._last_history_lighting = lighting
        for cam in CAMERAS:
            for j, frame in enumerate(indices):
                if frame == anchor:
                    resized[cam][:, j] = sample['per_cam'][cam][:, 0]
        sample['history_video'] = compose_from_per_cam({cam: value[None] for cam, value in resized.items()}, layout=get_layout('tshape_robotwin_384x320_uniform'))[0]
        sample['history_frame_offsets'] = torch.from_numpy(offsets.copy()).float()
        return sample

    @property
    def last_history_lighting(self):
        """Worker-local audit of the last sample, outside model input fields."""
        return copy.deepcopy(getattr(self, '_last_history_lighting', None))

    @property
    def checkpoint_contract(self):
        value = dict(super().checkpoint_contract, training_history=self.history_config.contract())
        cfg = getattr(self, 'history_light_augmentation', None)
        if cfg is not None:
            value['training_history_lighting'] = history_light_contract(cfg)
        return value
