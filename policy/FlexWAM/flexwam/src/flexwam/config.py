"""Compile the public data/model/task configuration into stable runtime inputs.

Camera frames are stretched from their raw size to ``observation.image_size`` in
a single PIL BILINEAR resize inside the processor. This module owns configuration
translation; numerical model code never needs to know the public YAML layout.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re
from typing import Any

from omegaconf import DictConfig, OmegaConf

from flexwam.models.architectures.view_scoring_head import HEAD_NETWORK_KEYS, normalize_head_arch
from flexwam.models.architectures.state_conditioning import conditioning_config, state_config

MODEL_KEYS = {
    'video': 'video_dit_config',
    'action_dit': 'action_dit_config',
    'router': 'action_router',
    'critic': 'action_critic',
}
ACTION_ATTENTION_MODES = ('additive', 'logit_bias')
TRAIN_KEYS = {'epochs': 'num_epochs', 'stages': 'training_stages'}


def action_attention_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize the action-to-video attention mode.

    ``additive`` keeps one softmax per camera and adds ``scale_v * Attn_v`` to
    action self-attention. ``logit_bias`` runs one softmax over video and action
    keys, adding ``bias_strength * log(scale_v)`` to that camera's visible keys.
    """
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)
    raw = dict(config or {})
    unknown = set(raw) - {'mode', 'bias_strength'}
    if unknown:
        raise ValueError(f'action_attention has unknown keys: {sorted(unknown)}')
    mode = str(raw.get('mode', 'additive'))
    if mode not in ACTION_ATTENTION_MODES:
        raise ValueError(
            f"action_attention.mode must be one of {ACTION_ATTENTION_MODES}, got {mode!r}."
        )
    strength = float(raw.get('bias_strength', 1.0))
    if strength < 0.0 or strength != strength or strength == float('inf'):
        raise ValueError(f'action_attention.bias_strength must be finite and >= 0, got {strength}.')
    return {'mode': mode, 'bias_strength': strength}


def normalize_model_config(cfg: DictConfig) -> DictConfig:
    """Also usable by backbone-preparation scripts before interpolations resolve."""
    obj = OmegaConf.to_container(cfg, resolve=False)
    for public, internal in MODEL_KEYS.items():
        if public in obj:
            obj[internal] = obj.pop(public)
    obj.pop('name', None)
    return OmegaConf.create(obj)


def _dataset_configs(data: dict[str, Any], spatial_multiple: int = 16) -> dict[str, Any]:
    robot, observation = data['robot'], data['observation']
    cameras = observation['cameras']
    names = [cam['name'] for cam in cameras]
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError('data.observation.cameras must contain at least two distinct camera names.')
    h, w = observation['image_size']
    if min(h, w) <= 0 or h % spatial_multiple or w % spatial_multiple:
        raise ValueError(
            f'data.observation.image_size must contain positive multiples of {spatial_multiple} (H, W).'
        )
    if 'preprocess_size' in observation:
        raise ValueError('data.observation.preprocess_size was removed; cameras resize raw -> image_size in one step.')
    image_size = [int(h), int(w)]
    sampling = data['sampling']
    horizon, stride = int(sampling['action_horizon']), int(sampling['video_stride'])
    if horizon <= 0 or stride <= 0 or horizon % stride or (horizon // stride) % 4:
        raise ValueError('action_horizon / video_stride must be a positive multiple of 4 for the current VAE.')
    if not data.get('roots'):
        raise ValueError('Set data.roots to one or more LeRobot dataset directories.')
    shape_meta = {
        'images': [{'key': c['name'], 'lerobot_key': c['source'],
                    'raw_shape': c.get('raw_shape', [3, *image_size]),
                    'shape': [3, *image_size]} for c in cameras],
        'action': [{'key': 'default', 'lerobot_key': robot['action_source'],
                    'raw_shape': robot['action_dim'], 'shape': robot['action_dim']}],
        'state': [{'key': 'default', 'lerobot_key': robot['state_source'],
                   'raw_shape': robot['state_dim'], 'shape': robot['state_dim']}],
    }
    action = data['action']
    representation = action['representation']
    if representation == 'joint_delta':
        absolute = set(action.get('absolute_indices', []))
        if any(i < 0 or i >= robot['action_dim'] for i in absolute):
            raise ValueError('action.absolute_indices must be within robot.action_dim.')
        transforms = [{'_target_': 'flexwam.datasets.lerobot.transforms.relative_action.MaskedRelativeJointTransform',
                       'keys': ['default'], 'mask': [i not in absolute for i in range(robot['action_dim'])]}]
    elif representation == 'pose6d_delta':
        transforms = [{'_target_': 'flexwam.datasets.lerobot.transforms.relative_action.FlexivPose6DDeltaTransform', 'keys': ['default']}]
    elif representation == 'pose6d_delta_bimanual':
        transforms = [{'_target_': 'flexwam.datasets.lerobot.transforms.relative_action.BimanualPose6DDeltaTransform', 'keys': ['default']}]
    elif representation == 'absolute':
        transforms = []
    else:
        raise ValueError(f'Unsupported action.representation: {representation!r}')
    norm = data['normalization']
    proc = {
        '_target_': 'flexwam.datasets.lerobot.processors.flexwam_processor.FlexWAMProcessor',
        'shape_meta': shape_meta, 'num_obs_steps': horizon + 1,
        'num_output_cameras': len(cameras), 'action_output_dim': robot['action_dim'],
        'proprio_output_dim': robot['state_dim'], 'action_state_transforms': transforms,
        'use_stepwise_action_norm': norm['stepwise'], 'norm_default_mode': norm['mode'],
        'norm_exception_mode': norm.get('exceptions'),
        'action_state_merger': {'_target_': 'flexwam.datasets.lerobot.transforms.action_state_merger.ConcatLeftAlign'},
        'train_transforms': [{'_target_': 'flexwam.datasets.lerobot.transforms.image.PILBilinearResize', 'size': image_size},
                             {'_target_': 'flexwam.datasets.lerobot.transforms.image.ToTensor'}],
        'val_transforms': [{'_target_': 'flexwam.datasets.lerobot.transforms.image.PILBilinearResize', 'size': image_size},
                           {'_target_': 'flexwam.datasets.lerobot.transforms.image.ToTensor'}],
    }
    # Missing/default keys remain missing to preserve old resolved configurations.
    if norm.get('clip') is not None:
        proc['norm_clip_to_unit_range'] = norm['clip']
    augmentation = data.get('augmentation', {}).get('color_jitter')
    if augmentation is not None:
        proc['color_jitter'] = {'_target_': 'flexwam.datasets.lerobot.transforms.image.VideoColorJitter', **augmentation}
    text = data['text']
    common = {
        '_target_': 'flexwam.datasets.lerobot.robot_video_dataset.RobotVideoDataset',
        'dataset_dirs': data['roots'], 'shape_meta': shape_meta, 'num_frames': horizon + 1,
        'global_sample_stride': sampling['global_stride'], 'action_video_freq_ratio': stride,
        'video_size': observation['image_size'], 'camera_key': None,
        'val_set_proportion': data['split']['val_ratio'],
        'pretrained_norm_stats': norm.get('stats'), 'concat_multi_camera': 'independent',
        'text_embedding_cache_encoder_id': text['encoder_id'], 'processor': proc,
        'text_embedding_cache_dir': text['cache_dir'], 'context_len': text['context_len'],
    }
    train, val = deepcopy(common), deepcopy(common)
    train['is_training_set'], val['is_training_set'] = True, False
    val['dataset_dirs'] = data.get('val_roots') or data['roots']
    if norm.get('validation_mode') is not None:
        val['processor']['norm_default_mode'] = norm['validation_mode']
    if augmentation is not None:
        val['processor']['color_jitter'] = None
    return {'train': train, 'val': val}


def _video_depth(video: dict[str, Any]) -> int | None:
    if video.get('depth') is not None:
        return int(video['depth'])
    if video.get('num_layers') is not None:
        return int(video['num_layers'])
    return None


def _copy_router_network_config(router: dict[str, Any]) -> dict[str, Any]:
    copied = {key: deepcopy(router[key]) for key in HEAD_NETWORK_KEYS if key in router}
    copied['arch'] = normalize_head_arch(copied.get('arch'))
    return copied


def _validate_head_layer(label: str, cfg: dict[str, Any], depth: int | None) -> None:
    layer = int(cfg.get('feature_layer_index', 18))
    if layer < 0:
        raise ValueError(f'{label}.feature_layer_index must be non-negative, got {layer}.')
    if depth is not None and not 0 <= layer < depth:
        raise ValueError(f'{label}.feature_layer_index={layer} is outside video depth={depth}.')
    cfg['feature_layer_index'] = layer
    cfg['arch'] = normalize_head_arch(cfg.get('arch'))


def _resolve_camera_heads(model: dict[str, Any], nviews: int) -> None:
    video = model['video_dit_config']
    if 'state_conditioning' in video:
        video['state_conditioning'] = state_config(video['state_conditioning'], video=True)
    model['state_conditioning'] = conditioning_config(model.get('state_conditioning'))
    model['action_attention'] = action_attention_config(model.get('action_attention'))
    depth = _video_depth(video)
    router = model.get('action_router') or {}
    router.setdefault('num_views', nviews)
    router['arch'] = normalize_head_arch(router.get('arch'))
    _validate_head_layer('router', router, depth)
    model['action_router'] = router

    critic_raw = model.get('action_critic')
    if critic_raw is None:
        critic = _copy_router_network_config(router)
    else:
        critic = dict(critic_raw)
        critic.pop('same_router', None)
        critic['arch'] = normalize_head_arch(critic.get('arch'))
    critic.setdefault('num_views', nviews)
    _validate_head_layer('critic', critic, depth)
    model['action_critic'] = critic
    for head in (router, critic):
        if 'state_conditioning' in head:
            head['state_conditioning'] = state_config(head['state_conditioning'])

    for label, count in [
        ('video', video.get('num_views', nviews)),
        ('router', router.get('num_views', nviews)),
        ('critic', critic.get('num_views', nviews)),
    ]:
        if count != nviews:
            raise ValueError(f'{label}.num_views={count} does not match the {nviews} configured cameras.')


def resolve_config(cfg: DictConfig) -> DictConfig:
    """Resolve a public config once; accept saved runtime configs unchanged."""
    if 'data' in cfg and 'train' in cfg.data and 'val' in cfg.data:
        return cfg
    obj = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    is_wan = obj['model']['_target_'].rsplit('.', 1)[-1] == 'create_flexwam'
    data = _dataset_configs(obj['data'], spatial_multiple=32 if is_wan else 16)
    model = OmegaConf.to_container(normalize_model_config(OmegaConf.create(obj['model'])), resolve=True)
    train = obj.get('train', {})
    cf = train.get('counterfactual')
    if cf is not None:
        model['router_cf'] = cf
    nviews = len(obj['data']['observation']['cameras'])
    video = model['video_dit_config']
    _resolve_camera_heads(model, nviews)
    if model['_target_'].rsplit('.', 1)[-1] == 'create_flexwam':
        # Tasks inherit LingBot's cache ID. Native WAN must look up UMT5 files,
        # whose encoder-specific filenames can coexist in the same directory.
        native_text_dim = int(video['text_dim'])
        if int(model.get('context_input_dim') or native_text_dim) == native_text_dim:
            encoder_id = re.sub(r'[^a-z0-9]+', '', model['model_id'].split('/')[-1].lower())
            for dataset in data.values():
                if dataset['text_embedding_cache_encoder_id'] == 'lingbotqwen3vl':
                    dataset['text_embedding_cache_encoder_id'] = encoder_id
    if 'lingbot' in model['_target_']:
        action = model['action_dit_config']
        for key in ['depth', 'num_attention_heads']:
            if video[key] != action[key]:
                raise ValueError(f'Video and Action DiT {key} must match for MoT.')
        if video['hidden_size'] // video['num_attention_heads'] != sum(action['axes_dims']):
            raise ValueError('Video and Action DiT attention head dimensions must match.')
    out = {k: v for k, v in obj.items() if k not in {'data', 'model', 'train', 'runtime', 'task', 'config_version'}}
    out.update({TRAIN_KEYS.get(k, k): v for k, v in train.items() if k != 'counterfactual'})
    out.update(obj.get('runtime', {}))
    out['data'], out['model'] = data, model
    return OmegaConf.create(out)


def compose_task(task: str, *, config_dir: str | Path | None = None, overrides=()) -> DictConfig:
    from hydra import compose, initialize_config_dir
    from flexwam.utils.config_resolvers import register_default_resolvers
    register_default_resolvers()
    directory = Path(config_dir) if config_dir else Path(__file__).resolve().parents[2] / 'configs'
    with initialize_config_dir(config_dir=str(directory.resolve()), version_base='1.3'):
        return resolve_config(compose(config_name='train', overrides=[f'task={task}', *overrides]))
