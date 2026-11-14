import hashlib
import json
import math
from collections.abc import Mapping, Sequence
import numpy as np
import torch
import torch.nn.functional as F
CAMERAS = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')
DEFAULTS = dict(enabled=False, probability=0.8, seed=42, crop_linear_scale=[0.95, 1.0], rotation_degrees=[-5.0, 5.0], brightness=[0.6, 1.4], contrast=[0.6, 1.4], saturation=[0.6, 1.4], hue_degrees=[-36.0, 36.0], gamma=[0.7, 1.4], exposure_stops=[-0.5, 0.5], red_blue_gain=[0.85, 1.15], blur_probability=0.3, blur_kernel=5, blur_sigma=[0.1, 2.0], noise_sigma=0.03, adain_probability=0.5, adain_target_mean=[0.2, 0.8], adain_target_std=[0.05, 0.3], adain_blend=[0.2, 0.5], fft_probability=0.5, fft_amplitude=[0.5, 1.5])

def resolve_image_augmentation(value=None):
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError('image_augmentation must be a mapping')
    enabled = value.get('enabled', False)
    if not isinstance(enabled, bool):
        raise ValueError('image_augmentation.enabled must be Boolean')
    if not enabled:
        return None
    unknown = set(value) - set(DEFAULTS)
    if unknown:
        raise ValueError('Unknown augmentation configuration: ' + str(sorted(unknown)))
    cfg = {**DEFAULTS, **dict(value), 'enabled': True}
    for key, default in DEFAULTS.items():
        if isinstance(default, list):
            pair = cfg[key]
            if isinstance(pair, (str, bytes)) or not isinstance(pair, Sequence) or len(pair) != 2:
                raise ValueError('Augmentation range must contain two values: ' + key)
            lo, hi = map(float, pair)
            if not (math.isfinite(lo) and math.isfinite(hi) and (lo <= hi)):
                raise ValueError('Invalid augmentation range: ' + key)
            cfg[key] = [lo, hi]
    for key in ('probability', 'blur_probability', 'adain_probability', 'fft_probability'):
        if not 0 <= float(cfg[key]) <= 1:
            raise ValueError('Invalid augmentation probability: ' + key)
        cfg[key] = float(cfg[key])
    if not 0 < cfg['crop_linear_scale'][0] <= cfg['crop_linear_scale'][1] <= 1:
        raise ValueError('Crop linear scale must be in (0,1]')
    for key in ('brightness', 'contrast', 'saturation', 'gamma', 'red_blue_gain', 'blur_sigma', 'adain_target_std', 'fft_amplitude'):
        if cfg[key][0] <= 0:
            raise ValueError('Augmentation scale must be positive: ' + key)
    if not -180 <= cfg['hue_degrees'][0] <= cfg['hue_degrees'][1] <= 180:
        raise ValueError('Hue must be in [-180,180] degrees')
    if not (0 <= cfg['adain_target_mean'][0] <= cfg['adain_target_mean'][1] <= 1 and 0 <= cfg['adain_blend'][0] <= cfg['adain_blend'][1] <= 1):
        raise ValueError('AdaIN mean/blend must be in [0,1]')
    cfg['blur_kernel'] = int(cfg['blur_kernel'])
    if cfg['blur_kernel'] < 1 or cfg['blur_kernel'] % 2 != 1:
        raise ValueError('Blur kernel must be a positive odd integer')
    if not math.isfinite(float(cfg['noise_sigma'])) or float(cfg['noise_sigma']) < 0:
        raise ValueError('Noise sigma must be finite and nonnegative')
    cfg['noise_sigma'] = float(cfg['noise_sigma'])
    cfg['seed'] = int(cfg['seed'])
    return cfg

def augmentation_contract(value=None):
    cfg = resolve_image_augmentation(value)
    if cfg is None:
        return {'enabled': False}
    result = dict(enabled=True, version='egowm.robotwin.clean_to_random.v1', parameters=cfg, paper='https://arxiv.org/html/2609.25961v1#A6', geometry_cameras=['cam_high'], color_cameras=list(CAMERAS), RGB_domain='float32_[0,1]', application='one_Bernoulli_per_sample', cameras='independent_parameter_streams', temporal_frames=9, temporal_sharing='parameters_and_noise_pattern_and_FFT_multiplier', RNG='SHA256(seed,window_index,sample_position,camera); no_global_RNG', geometry='linear_FOV_crop_then_pixel_rigid_rotation; A_output_to_source; nearest_depth; Kprime=solve(A,K)', RGB_depth_registration='pixel_centres_scaled_between_native_grids', outside_fill=dict(RGB=0, depth=0, valid=False), order=['head_geometry', 'brightness', 'contrast', 'saturation', 'hue', 'gamma', 'exposure', 'red_blue_gain', 'blur', 'Gaussian_noise', 'AdaIN', 'FFT'], engineering_choices=dict(contrast_anchor='clip-wide_grayscale_mean', blur='Gaussian_separable_kernel5_sigma_uniform_.1_to_2_default', noise='always_in_selected_sample; sigma_in_RGB_[0,1]; shared_2D_pattern', adain='per_channel_target_mean_and_std_uniform; blend_uniform; clip-wide_source_stats', fft='independent_channel_frequency_multiplier; conjugate_pairs_share_factor; phase_preserved_before_output_clamp', output='clamp_[0,1]_after_each_operation; geometry_invalid_RGB_reset_to_zero'), state_action_language='unchanged', physical_time='unchanged')
    result['sha256'] = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return result

def reject_unaugmented_feature_caches(source):

    def visit(value, path='source'):
        if not isinstance(value, Mapping):
            return
        for key, child in value.items():
            label = str(key).lower()
            location = path + '.' + str(key)
            if child and (('dino' in label or 'perception' in label or 'feature' in label or ('latent' in label)) and ('cache' in label or 'root' in label or 'path' in label) or ('flow' in label and ('cache' in label or 'root' in label))):
                raise ValueError('Augmentation rejects original-RGB feature/flow cache: ' + location)
            visit(child, location)
    visit(source)
    auxiliary = source.get('auxiliary', {})
    if auxiliary.get('flow'):
        raise ValueError('Augmentation profile requires flow labels disabled, not cached original flow')

def _rng(cfg, index, occurrence, stream):
    encoded = json.dumps([cfg['seed'], int(index), int(occurrence), stream], separators=(',', ':')).encode()
    seed = int.from_bytes(hashlib.sha256(encoded).digest()[:16], 'little')
    return np.random.default_rng(seed)

def sample_parameters(cfg, index, occurrence):
    cfg = resolve_image_augmentation(cfg)
    if cfg is None or _rng(cfg, index, occurrence, 'sample').random() >= cfg['probability']:
        return None
    cameras = {}
    for cam in CAMERAS:
        rng = _rng(cfg, index, occurrence, cam)
        row = {k: float(rng.uniform(*cfg[k])) for k in ('brightness', 'contrast', 'saturation', 'hue_degrees', 'gamma', 'exposure_stops', 'blur_sigma', 'adain_blend')}
        row['red_blue_gain'] = rng.uniform(*cfg['red_blue_gain'], size=2).tolist()
        row['adain_target_mean'] = rng.uniform(*cfg['adain_target_mean'], size=3).tolist()
        row['adain_target_std'] = rng.uniform(*cfg['adain_target_std'], size=3).tolist()
        row['blur'] = bool(rng.random() < cfg['blur_probability'])
        row['adain'] = bool(rng.random() < cfg['adain_probability'])
        row['fft'] = bool(rng.random() < cfg['fft_probability'])
        row['noise_seed'] = int(rng.integers(0, 2 ** 63))
        row['fft_seed'] = int(rng.integers(0, 2 ** 63))
        if cam == 'cam_high':
            row['crop_scale'] = float(rng.uniform(*cfg['crop_linear_scale']))
            row['rotation_degrees'] = float(rng.uniform(*cfg['rotation_degrees']))
            row['crop_offset_fraction'] = rng.uniform(-1, 1, size=2).tolist()
        cameras[cam] = row
    return cameras

def pixel_output_to_input(height, width, params, *, device='cpu'):
    angle = math.radians(params['rotation_degrees'])
    scale = params['crop_scale']
    c, s = (math.cos(angle), math.sin(angle))
    linear = torch.tensor([[c, -s], [s, c]], dtype=torch.float32, device=device) * scale
    centre = torch.tensor([(width - 1) / 2, (height - 1) / 2], device=device)
    offset = torch.tensor(params['crop_offset_fraction'], device=device) * (1 - scale) * centre
    matrix = torch.eye(3, dtype=torch.float32, device=device)
    matrix[:2, :2] = linear
    matrix[:2, 2] = centre + offset - linear @ centre
    return matrix

def _transport_grid(matrix, from_hw, to_hw):
    fh, fw = from_hw
    th, tw = to_hw
    scale = matrix.new_tensor([[tw / fw, 0, (tw / fw - 1) / 2], [0, th / fh, (th / fh - 1) / 2], [0, 0, 1]])
    return scale @ matrix @ torch.linalg.inv(scale)

def transform_intrinsics(matrix, intrinsics):
    A = matrix.to(device=intrinsics.device, dtype=torch.float64)
    K = intrinsics.to(dtype=torch.float64)
    canonical = A.new_tensor([0.0, 0.0, 1.0])
    if A.shape != (3, 3) or K.shape != (3, 3) or (not torch.isfinite(A).all()) or (not torch.isfinite(K).all()):
        raise ValueError('Image-plane transform requires finite 3x3 A and K')
    if not torch.equal(A[2], canonical) or not torch.equal(K[2], canonical):
        raise ValueError('Image-plane transform requires calibrated affine A/K last rows')
    top = torch.linalg.solve(A[:2, :2], K[:2] - A[:2, 2, None] * K[2, None])
    result = torch.cat([top, canonical[None]], dim=0).to(intrinsics)
    if not torch.isfinite(result).all():
        raise ValueError('Image-plane transform produced nonfinite K')
    return result

def warp(image, matrix, *, mode):
    t, _, height, width = image.shape
    yy, xx = torch.meshgrid(torch.arange(height, device=image.device), torch.arange(width, device=image.device), indexing='ij')
    pixels = torch.stack([xx, yy, torch.ones_like(xx)], -1).float()
    source = pixels @ matrix.T
    x, y = (source[..., 0], source[..., 1])
    valid = (x >= -0.5) & (x < width - 0.5) & (y >= -0.5) & (y < height - 0.5)
    grid = torch.stack([(x + 0.5) * 2 / width - 1, (y + 0.5) * 2 / height - 1], -1)
    result = F.grid_sample(image.float(), grid[None].expand(t, -1, -1, -1), mode=mode, padding_mode='zeros', align_corners=False)
    return (result, valid)

def _hue(image, degrees):
    if degrees == 0:
        return image
    maximum, channel = image.max(1, keepdim=True)
    minimum = image.min(1, keepdim=True).values
    delta = maximum - minimum
    safe = delta.clamp_min(1e-08)
    r, g, b = image.unbind(1)
    hue = torch.where(channel[:, 0] == 0, (g - b) / safe[:, 0], torch.where(channel[:, 0] == 1, (b - r) / safe[:, 0] + 2, (r - g) / safe[:, 0] + 4))
    hue = torch.where(delta[:, 0] > 0, (hue / 6 + degrees / 360) % 1, torch.zeros_like(hue))
    saturation = torch.where(maximum > 0, delta / maximum.clamp_min(1e-08), torch.zeros_like(delta))[:, 0]
    value = maximum[:, 0]
    sector = torch.floor(hue * 6).long() % 6
    f = hue * 6 - torch.floor(hue * 6)
    p, q, v = (value * (1 - saturation), value * (1 - f * saturation), value * (1 - (1 - f) * saturation))
    candidates = torch.stack([torch.stack(row, 1) for row in ((value, v, p), (q, value, p), (p, value, v), (p, q, value), (v, p, value), (value, p, q))], 1)
    return candidates.gather(1, sector[:, None, None].expand(-1, 1, 3, -1, -1)).squeeze(1)

def fft_amplitude_transform(image, factors):
    transformed = torch.fft.fft2(image, dim=(-2, -1)) * factors[None]
    return torch.fft.ifft2(transformed, dim=(-2, -1)).real

def _fft_factors(height, width, seed, bounds):
    rng = np.random.default_rng(seed)
    y, x = np.indices((height, width))
    here = y * width + x
    conjugate = -y % height * width + -x % width
    paired = np.minimum(here, conjugate)
    factors = rng.uniform(*bounds, size=(3, height * width))[:, paired]
    return torch.from_numpy(factors.astype(np.float32))

def photometric(image, params, cfg, *, statistics=None, record=None):
    """Clip [T,3,H,W] with camera-specific, time-shared parameters."""
    x = (image * params['brightness']).clamp(0, 1)
    coefficients = x.new_tensor([0.2989, 0.587, 0.114])[None, :, None, None]
    anchor = (x * coefficients).sum(1, keepdim=True).mean() if statistics is None else statistics['contrast_anchor']
    if record is not None:
        record['contrast_anchor'] = anchor.detach().clone()
    x = (params['contrast'] * x + (1 - params['contrast']) * anchor).clamp(0, 1)
    grey = (x * coefficients).sum(1, keepdim=True)
    x = (params['saturation'] * x + (1 - params['saturation']) * grey).clamp(0, 1)
    x = _hue(x, params['hue_degrees']).clamp(0, 1)
    x = x.pow(params['gamma']).clamp(0, 1)
    x = (x * 2 ** params['exposure_stops']).clamp(0, 1)
    x = (x * x.new_tensor([params['red_blue_gain'][0], 1, params['red_blue_gain'][1]])[None, :, None, None]).clamp(0, 1)
    if params['blur']:
        kernel = cfg['blur_kernel']
        radius = kernel // 2
        if min(x.shape[-2:]) <= radius:
            raise ValueError('Image too small for configured reflect blur')
        axis = torch.arange(-radius, radius + 1, dtype=x.dtype, device=x.device)
        weights = torch.exp(-axis.square() / (2 * params['blur_sigma'] ** 2))
        weights /= weights.sum()
        x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode='reflect'), weights[None, None, None, :].expand(3, 1, 1, kernel), groups=3)
        x = F.conv2d(F.pad(x, (0, 0, radius, radius), mode='reflect'), weights[None, None, :, None].expand(3, 1, kernel, 1), groups=3).clamp(0, 1)
    noise = np.random.default_rng(params['noise_seed']).normal(0, cfg['noise_sigma'], size=(3, *x.shape[-2:])).astype(np.float32)
    x = (x + torch.from_numpy(noise).to(x)[None]).clamp(0, 1)
    if params['adain']:
        mean = x.mean((0, 2, 3), keepdim=True) if statistics is None else statistics['adain_mean']
        std = x.std((0, 2, 3), keepdim=True, correction=0).clamp_min(1e-06) if statistics is None else statistics['adain_std']
        if record is not None:
            record.update(adain_mean=mean.detach().clone(), adain_std=std.detach().clone())
        target_mean = x.new_tensor(params['adain_target_mean'])[None, :, None, None]
        target_std = x.new_tensor(params['adain_target_std'])[None, :, None, None]
        styled = (x - mean) / std * target_std + target_mean
        alpha = params['adain_blend']
        x = ((1 - alpha) * x + alpha * styled).clamp(0, 1)
    if params['fft']:
        factors = _fft_factors(*x.shape[-2:], params['fft_seed'], cfg['fft_amplitude']).to(x)
        x = fft_amplitude_transform(x, factors).clamp(0, 1)
    return x

def augment_rgbd(rgb, depths, intrinsics, cfg, *, index, occurrence, return_valid_masks=False, replay=None):
    """Joint native-grid head RGB-D/K geometry, all-camera color/style.

    Inputs are decoded RGB [9,3,H,W] in [0,1], depth [9,H,W] in mm, and
    per-camera K [3,3,3]. The disabled/unsampled path returns input objects.
    """
    cfg = resolve_image_augmentation(cfg)
    parameters = sample_parameters(cfg, index, occurrence) if cfg else None
    if parameters is None:
        result = (rgb, depths, intrinsics, False)
        return (*result, None) if return_valid_masks else result
    if tuple(rgb) != CAMERAS or tuple(depths) != CAMERAS:
        raise ValueError('Augmentation camera ordering differs')
    if intrinsics.shape != (3, 3, 3) or not torch.isfinite(intrinsics).all():
        raise ValueError('Augmentation requires finite per-camera native-depth K')
    images = {}
    transformed_depths = dict(depths)
    K = intrinsics.clone()
    valid_masks = {}
    for ci, cam in enumerate(CAMERAS):
        image = rgb[cam]
        if image.ndim != 4 or image.shape[:2] != (9, 3):
            raise ValueError('Augmentation expects one complete nine-frame RGB clip')
        if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError('Augmentation RGB domain must be finite [0,1]')
        valid = None
        image_matrix = None
        if cam == 'cam_high':
            depth = depths[cam]
            if depth.ndim != 3 or depth.shape[0] != 9:
                raise ValueError('Augmentation expects nine depth frames')
            matrix = pixel_output_to_input(*depth.shape[-2:], parameters[cam], device=image.device)
            image_matrix = _transport_grid(matrix, depth.shape[-2:], image.shape[-2:])
            image, valid = warp(image, image_matrix, mode='bilinear')
            warped, depth_valid = warp(depth[:, None], matrix, mode='nearest')
            transformed_depths[cam] = torch.where(depth_valid[None], warped[:, 0], 0).round().to(depth.dtype)
            K[ci] = transform_intrinsics(matrix, K[ci])
        stats = {} if replay is not None else None
        image = photometric(image, parameters[cam], cfg, record=stats)
        if replay is not None:
            replay[cam] = dict(params=parameters[cam], statistics=stats, matrix=image_matrix, valid=valid)
        images[cam] = image if valid is None else torch.where(valid[None, None], image, 0)
        valid_masks[cam] = torch.ones(image.shape[-2:], dtype=torch.bool, device=image.device) if valid is None else valid
    result = (images, transformed_depths, K, True)
    return (*result, valid_masks) if return_valid_masks else result
