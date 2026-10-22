"""Train-only RGB illumination proxies; no simulator or action-label changes."""
import math

import torch
from omegaconf import OmegaConf

from .robot_video_dataset import RobotVideoDataset


def sample_domain(config, training, views):
    if not training or not config or not config.get('enabled', False):
        return None
    def uniform(lo, hi):
        return float(torch.empty(()).uniform_(lo, hi).item())
    params = {}
    if float(torch.rand(())) < config.get('rgb_prob', 0.):
        lo, hi = config['rgb_gain_range']
        gains = [uniform(lo, hi) for _ in range(3)]
        # Separate illuminant color from the existing brightness augmentation.
        mean = sum(gains) / 3
        params['rgb'] = [g / mean for g in gains]
    if float(torch.rand(())) < config.get('shadow_prob', 0.):
        # Each view has a stable screen-space light field throughout this clip.
        params['shadows'] = [dict(x=uniform(0., 1.), y=uniform(0., 1.),
                                  sigma=uniform(*config['shadow_sigma_range']),
                                  depth=uniform(*config['shadow_depth_range'])) for _ in views]
    return params or None


def apply_domain(video, params):
    """N,T,C,H,W float RGB in [0,1], normalized coordinates across resolutions."""
    if not params:
        return video
    assert video.ndim == 5 and video.shape[2] == 3 and video.is_floating_point()
    out = video
    if 'rgb' in params:
        out = out * out.new_tensor(params['rgb']).view(1, 1, 3, 1, 1)
    if 'shadows' in params:
        assert len(params['shadows']) == out.shape[0]
        h, w = out.shape[-2:]
        ys = torch.linspace(0, 1, h, device=out.device, dtype=out.dtype).view(h, 1)
        xs = torch.linspace(0, 1, w, device=out.device, dtype=out.dtype).view(1, w)
        fields = []
        for p in params['shadows']:
            distance = (xs-p['x']).square() + (ys-p['y']).square()
            fields.append(1-p['depth']*torch.exp(-distance/(2*p['sigma']**2)))
        out = out * torch.stack(fields).view(out.shape[0], 1, 1, h, w)
    return out.clamp(0, 1)


class DomainAugmentedRobotVideoDataset(RobotVideoDataset):
    def __init__(self, image_domain_jitter=None, **kwargs):
        config = OmegaConf.to_container(image_domain_jitter, resolve=True) if OmegaConf.is_config(image_domain_jitter) else dict(image_domain_jitter or {})
        for key in ('rgb_prob', 'shadow_prob'):
            value = float(config.get(key, 0.))
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'Invalid {key}: {value}')
        for key, bounds in [('rgb_gain_range', (0., 2.)), ('shadow_sigma_range', (0., 1.)), ('shadow_depth_range', (0., 1.))]:
            if key in config:
                lo, hi = config[key]
                if not (math.isfinite(lo) and math.isfinite(hi) and bounds[0] < lo <= hi <= bounds[1]):
                    raise ValueError(f'Invalid {key}: {config[key]}')
        self.image_domain_jitter = config
        self._domain_params = None
        super().__init__(**kwargs)
        if config.get('enabled', False) and not self.apply_color_jitter_to_vlm:
            raise ValueError('Domain augmentation requires apply_to_vlm color path for paired RGB.')

    def _get(self, *args, **kwargs):
        previous = self._domain_params
        self._domain_params = sample_domain(self.image_domain_jitter, self.is_training_set, self.video_view_names)
        try:
            return super()._get(*args, **kwargs)
        finally:
            self._domain_params = previous

    def _format_video_window(self, video, color_jitter_params=None, camera_jitter_params=None, sensor_noise_params=None):
        if self._domain_params is None:
            return super()._format_video_window(video, color_jitter_params, camera_jitter_params, sensor_noise_params)
        if video.ndim != 5:
            raise ValueError('RoboTwin domain augmentation requires explicit camera axis')
        # Same ordering as VLM: crop -> illumination -> color jitter.
        video = self._apply_camera_jitter_to_video(video, camera_jitter_params)
        video = apply_domain(video, self._domain_params)
        return super()._format_video_window(video, color_jitter_params, None, sensor_noise_params)

    def _apply_color_jitter_to_vlm_images(self, images, params):
        if self._domain_params is None:
            return super()._apply_color_jitter_to_vlm_images(images, params)
        if images.dtype == torch.uint8:
            pixels = images.float().div(255.).unsqueeze(1)
            pixels = apply_domain(pixels, self._domain_params).squeeze(1)
            # Keep float until both transforms finish, avoiding extra quantization.
            pixels = self._apply_color_jitter_to_video(pixels, params)
            return pixels.mul(255.).round().clamp(0, 255).to(torch.uint8)
        images = apply_domain(images.unsqueeze(1), self._domain_params).squeeze(1)
        return super()._apply_color_jitter_to_vlm_images(images, params)
