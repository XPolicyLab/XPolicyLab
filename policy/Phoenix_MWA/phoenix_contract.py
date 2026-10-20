"""Explicit new-real-training IO; never reinterpret legacy z-score weights."""
import json
import types

import numpy as np
import torch
from torchvision.transforms.functional import to_tensor

from .joint96_assets import sha256

ACTIVE = list(range(6)) + [16] + list(range(40, 46)) + [56]
ROBOTS = {'arx_x5': (2, 'arx_x5', 30), 'piper': (4, 'piper100', 30),
          'piper_x': (15, 'piperx', 25)}
PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"
MODE = 'repeat9_full_joint96'


def preprocessing_contract(family):
    if family not in ROBOTS:
        raise ValueError('Unknown real robot family')
    applied = family != 'arx_x5'
    # This serialized training identifier must match existing checkpoints.
    return dict(mode='openwam_real_clip01_v1', applied=applied,
                native_dims=[6, 13] if applied else [], raw_clip=[0., 1.] if applied else None,
                sensor_tolerance=.05 if applied else None,
                stage='before_action_and_state_normalization')


def preprocess_state(state, family):
    result = np.asarray(state, dtype=np.float32)
    if result.shape != (14,) or not np.isfinite(result).all():
        raise ValueError('Expected finite native joint14 state')
    if preprocessing_contract(family)['applied']:
        gripper = result[[6, 13]]
        if np.any((gripper < -.05) | (gripper > 1.05)):
            raise ValueError('Gripper outside sensor tolerance [-0.05, 1.05]')
        result = result.copy()
        result[[6, 13]] = np.clip(gripper, 0., 1.)
    return result


def current_rgb(observation):
    """Explicit direct-resize variant; retain RobotWin three-view boundaries."""
    from .phoenix_resize_concat import current_rgb as resize_current_rgb

    return resize_current_rgb(observation)


def install_repeat9_joint(visual):
    """Wrap this instance only; both branches see nine RGB frames before slicing."""
    if visual.current_conditioning != 'legacy_clip_prefix':
        raise ValueError('New real adapter requires legacy_clip_prefix')
    if (hasattr(visual, '_phoenix_repeat9_original')
            or hasattr(visual.stvae_encoder, '_current_repeat9_original')):
        raise RuntimeError('Conflicting current-frame encoder adapter')
    original = visual.encode
    evidence = {'calls': 0, 'input_frames': None, 'encoded_frames': None,
                'full_latent_shape': None, 'first_latent_shape': None}

    def encode(self, video, *, camera_is_pad):
        if video.ndim != 6 or video.shape[0] == 0:
            raise ValueError('Expected nonempty [B,V,C,T,H,W]')
        if video.shape[3] != 1:
            return original(video, camera_is_pad=camera_is_pad)
        repeated = video.repeat(1, 1, 1, 9, 1, 1)
        full = original(repeated, camera_is_pad=camera_is_pad)
        if full.shape != (video.shape[0], 96, 3, 16, 48) or not torch.isfinite(full).all():
            raise ValueError('Expected finite full Joint96 nine-frame latents')
        result = full[:, :, :1].contiguous()
        evidence.update(calls=evidence['calls'] + 1, input_frames=1, encoded_frames=9,
                        full_latent_shape=list(full.shape), first_latent_shape=list(result.shape))
        return result

    visual._phoenix_repeat9_original = original
    visual.encode = types.MethodType(encode, visual)
    return evidence


class CachedText:
    def __init__(self, path, expected_sha256):
        if sha256(path) != expected_sha256:
            raise ValueError('Text cache SHA256 mismatch')
        cache = torch.load(path, map_location='cpu', weights_only=True)
        if cache.get('prompt_template') != PROMPT:
            raise ValueError('Text template differs from training')
        self.rows = {}
        for prompt, row in cache['rows'].items():
            context, mask = row['context'], row['mask']
            if context.shape != (128, 4096) or mask.shape != (128,):
                raise ValueError('Invalid T5 cache shape')
            context = context.masked_fill(~mask.bool()[:, None], 0)
            if not torch.isfinite(context).all():
                raise ValueError('Nonfinite T5 cache')
            self.rows[prompt] = (context, torch.ones(128, dtype=torch.bool))

    def get(self, instruction, *, device, dtype):
        prompt = PROMPT.format(task=instruction)
        if prompt not in self.rows:
            raise ValueError('Unknown cached instruction: use text_mode=online_t5 for new instructions')
        context, mask = self.rows[prompt]
        return context.unsqueeze(0).to(device=device, dtype=dtype), mask.unsqueeze(0).to(device)


def validate_training(cfg, payload, package, family, expected_step, *, packaged=False):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from wam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
    from .joint96_backend import _static_dimension_is_pad
    from .joint96_checkpoint import _joint96_identity
    from .robot_profiles import checkpoint_robot_support

    if family not in ROBOTS or type(expected_step) is not int or not 0 < expected_step <= 40000:
        raise ValueError('Require a real family and actual posttraining step in [1, 40000]')
    if payload.get('step') != expected_step:
        raise ValueError('Actual checkpoint step mismatch')
    get = lambda key, default=None: OmegaConf.select(cfg, key, default=default)
    slot, registry_name, fps = ROBOTS[family]
    identity = _joint96_identity(payload.get('joint96_metadata'))
    contract = payload['joint96_metadata'].get('soft_prompt', {}).get('training_contract', {})
    from .training_routes import validate_route

    route = validate_route(cfg, contract, family)
    cached = route['cached']
    from .joint96_assets import read_release

    release = read_release()
    expected_sources = release.get('resize_training_source_sha256')
    if not expected_sources or contract.get('geometry_source_sha256') != expected_sources:
        raise ValueError('Unreviewed resize training source identity')
    for key, expected in {
        'channel_order': ['wan_vae48', 'stvae48'],
        'camera_order': ['main', 'left', 'right'],
        'action_dim': 80,
        'proprio_dim': 80,
    }.items():
        if identity.get(key) != expected:
            raise ValueError('Joint96 checkpoint identity mismatch: ' + key)
    required = dict(domain='robodojo_real', embodiment_ids={family: slot},
                    action_space='native_joint14_to_canonical80', cached_visuals=cached,
                    cached_text=True, dataloader=route['dataloader'],
                    normalization='frame_weighted_per_family_q01_q99_clip5',
                    action_offsets=list(range(32)), video_offsets=list(range(0, 33, 4)),
                    rgb_geometry='direct_bicubic_resize_256_per_view_horizontal_concat',
                    rgb_camera_order=['cam_head', 'cam_left_wrist', 'cam_right_wrist'],
                    rgb_view_size=[256, 256], rgb_crop=None,
                    rgb_interpolation='PIL_torchvision_BICUBIC_antialias_true',
                    visual_encoding='three_view_boundaries_preserved_latent_horizontal',
                    rgb_decode='robodojo_cv2_no_channel_swap',
                    loss_weights={'vae': .5, 'stvae': .5, 'action': 1.})
    for key, value in required.items():
        if key not in contract or contract[key] != value:
            raise ValueError('New real checkpoint contract mismatch: ' + key)
    required_cfg = {'joint96_posttrain.enabled': True,
                    'joint96_posttrain.rgb_geometry': 'direct_bicubic_resize_256_per_view_horizontal_concat',
                    'joint96_posttrain.cached_visuals': cached,
                    'joint96_posttrain.robodojo_openwam_joint14': True,
                    'model.joint96_three_view': True, 'model.proprio_dim': 80,
                    'model.action_dit_config.action_dim': 80,
                    'model.action_dit_config.num_embodiments': 28,
                    'model.action_dit_config.soft_prompt_tokens': 8,
                    'model.video_dit_config.in_dim': 96, 'model.video_dit_config.out_dim': 96,
                    'model.tokenizer_max_len': 128,
                    'model.joint_visual_codec.expected_stvae_step': 150000,
                    'data.train._target_': route['reader'],
                    'data.train.embodiment': family,
                    'data.train.gripper_preprocessing': 'openwam_real_clip01_v1',
                    'data.train.processor.norm_default_mode': 'q01/q99',
                    'data.train.processor.norm_exception_mode': None,
                    'data.train.processor.use_stepwise_action_norm': False,
                    'data.train.processor.normalization_passthrough_dims': None,
                    'data.train.processor.action_state_transforms': None}
    for key, value in required_cfg.items():
        if get(key) != value:
            raise ValueError('Saved training config mismatch: ' + key)
    for key in ('model.memory.enabled', 'model.memory.video.enabled', 'model.understanding.enabled',
                'model.future_delta.enabled', 'model.distillation.enabled'):
        if get(key, False):
            raise ValueError('Unsupported conditioning: ' + key)
    if (get('model.joint_visual_codec.current_conditioning', 'legacy_clip_prefix') != 'legacy_clip_prefix'
            or identity.get('current_conditioning', 'legacy_clip_prefix') != 'legacy_clip_prefix'):
        raise ValueError('Expected legacy_clip_prefix conditioning')
    source = payload['joint96_metadata']['model_paths'].get('posttrain_initialization', {})
    if source.get('step') != 256013 or int(get('joint96_posttrain.expected_pretrain_step', 0)) != 256013:
        raise ValueError('Expected pretraining lineage step 256013')
    if identity['action_video_attention'] != 'current_and_noisy_future_v1':
        raise ValueError('Incorrect action/video attention')
    index = json.loads((package / 'index.json').read_text())
    index_sha = sha256(package / 'index.json')
    if packaged:
        # The exported index is pinned by phoenix_source/manifest.json in the
        # release loader. It contains provenance, no training episode paths.
        if index.get('schema') != 'phoenix-inference-index-v1':
            raise ValueError('Incorrect inference metadata schema')
        index_sha = index.get('source_index_sha256')
    if (index_sha != contract.get('raw_index_sha256')
            or index['prompt_template'] != PROMPT
            or index['joint14_preprocessing'][family] != preprocessing_contract(family)):
        raise ValueError('Raw index/text/preprocessing identity mismatch')
    norm = contract.get('normalization_contract', {})
    stats = package / f'stats_slot{slot}.json'
    norm_expected = dict(version='robodojo_joint14_q01_q99_clip5_v2', mode='q01/q99',
                         forward_clip=[-5., 5.], inverse_clip=None, family=family,
                         canonical_active_dims=ACTIVE, stats_filename=stats.name,
                         stats_sha256=sha256(stats), joint14_preprocessing=preprocessing_contract(family))
    for key, value in norm_expected.items():
        if key not in norm or norm[key] != value:
            raise ValueError('Normalization contract mismatch: ' + key)
    if index['package_sha256'][stats.name] != norm['stats_sha256']:
        raise ValueError('Index and checkpoint statistics differ')
    processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(stats))
    for field in ('action', 'state'):
        if (~_static_dimension_is_pad(processor, field)).nonzero().flatten().tolist() != ACTIVE:
            raise ValueError('Canonical80 active slots mismatch')
        normalizer = processor.normalizer.normalizers[field]['default']
        for key in ('scale', 'offset', 'q01', 'q99'):
            actual = getattr(normalizer, key) if key in ('scale', 'offset') else normalizer.get_stats()[key]
            expected = torch.tensor(norm['fields'][field][key], dtype=actual.dtype)
            if actual.shape != (14,) or not torch.isfinite(actual).all() or not torch.equal(actual, expected):
                raise ValueError(f'Saved {field} {key} differs from selected statistics')
        # An active-slot check alone cannot detect a wrong source permutation.
        sample = torch.arange(14, dtype=torch.float32)[None]
        merged = processor.action_state_merger.forward({'action': {'default': sample.clone()},
                                                       'state': {'default': sample.clone()}})
        if not torch.equal(merged[field][..., ACTIVE], sample):
            raise ValueError('Canonical source ordering mismatch')
    support = checkpoint_robot_support(payload, dict(robot_type=family, embodiment_id=slot,
                                                      registry_name=registry_name))
    if not support['posttrained']:
        raise ValueError('Checkpoint is not posttrained for selected family')
    return dict(identity=identity, contract=contract, processor=processor, stats=stats,
                index=index, support=support, fps=fps, cached_visuals=cached)
