"""Inference for raw and audited cached-visual q01/q99 Phoenix real weights."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from .joint96_assets import POLICY_DIR, read_release, read_source, relative_path, sha256, prepare_tokenizer
from .joint96_backend import RoboDojoRealJoint96Session, _static_dimension_is_pad
from .joint96_checkpoint import _load_joint96_checkpoint
from .phoenix_contract import (ACTIVE, MODE, ROBOTS, CachedText, current_rgb,
                               install_repeat9_joint, preprocess_state, validate_training)
from wam.utils.config_resolvers import register_default_resolvers


def explicit_path(config, key):
    value = config.get(key)
    if not value:
        raise ValueError('New real inference requires explicit ' + key)
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else POLICY_DIR / path).resolve(strict=True)


def validate_options(config):
    family = config.get('env_cfg_type')
    if family not in ROBOTS or config.get('action_type', 'joint') != 'joint':
        raise ValueError('Select arx_x5, piper or piper_x / joint')
    if int(config.get('action_horizon', 32)) != 32:
        raise ValueError('Require action_horizon=32')
    replan, denoise = int(config.get('replan_steps', 32)), int(config.get('num_inference_steps', 10))
    if not 1 <= replan <= 32 or denoise <= 0:
        raise ValueError('Require replan in [1,32] and positive denoise steps')
    if (config.get('current_visual_mode', MODE) != MODE
            or config.get('mixed_precision', 'bf16') != 'bf16'
            or config.get('rand_device', 'cpu') != 'cpu'
            or float(config.get('text_cfg_scale', 1.)) != 1.
            or config.get('negative_prompt') not in (None, '')
            or str(config.get('tiled', False)).lower() not in ('false', '0', 'none')
            or config.get('sigma_shift') is not None):
        raise ValueError('Unsupported new-real inference option')
    hz = float(config.get('action_hz') or ROBOTS[family][2])
    if not np.isfinite(hz) or hz <= 0:
        raise ValueError('Require a finite positive requested action_hz')
    return dict(family=family, replan=replan, denoise=denoise, hz=hz,
                seed=int(config.get('seed') or 0))


def prepare_inputs(config):
    """CPU identity gate. No training imports, downloads or device motion."""
    if not config.get('training_config'):
        return prepare_release_inputs(config)
    options = validate_options(config)
    register_default_resolvers()
    checkpoint = explicit_path(config, 'checkpoint_path')
    config_path = explicit_path(config, 'training_config')
    package = explicit_path(config, 'stats_package')
    root = explicit_path(config, 'resource_root')
    expected_hash = str(config.get('checkpoint_sha256') or '').lower()
    if len(expected_hash) != 64 or any(c not in '0123456789abcdef' for c in expected_hash):
        raise ValueError('Pin checkpoint_sha256 from the actual posttraining file')
    if not config.get('expected_step'):
        raise ValueError('Pin the actual posttraining expected_step, not 256013')
    step = int(config['expected_step'])
    signature = checkpoint.stat()
    cfg = OmegaConf.load(config_path)
    payload = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=True)
    checked = validate_training(cfg, payload, package, options['family'], step)
    if sha256(checkpoint) != expected_hash:
        raise ValueError('Checkpoint SHA256 mismatch')
    after = checkpoint.stat()
    if (signature.st_ino, signature.st_size, signature.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError('Checkpoint changed during loading; use an immutable step archive')
    _, assets = read_source()
    release = read_release()
    weights = {}
    # Reuse only the pinned frozen resources, NEVER the old release's policy or stats.
    for key in ('wan_vae', 'dino_checkpoint', 'stvae_checkpoint'):
        spec = release['weights'][key]
        path = relative_path(root, spec['path'])
        if sha256(path) != spec['sha256']:
            raise ValueError('Frozen resource SHA256 mismatch: ' + key)
        assets[key] = path
        weights[key] = spec['sha256']
    identity = checked['identity']
    if (identity['teacher_sha256'] != weights['dino_checkpoint']
            or identity['stvae_checkpoint_sha256'] != weights['stvae_checkpoint']):
        raise ValueError('Checkpoint encoder resources differ from this source release')
    from wam.model.modules.codecs.joint_visual_encoder import encoder_source_sha256
    if encoder_source_sha256(assets['encoder_source_root']) != identity['source_sha256']:
        raise ValueError('Frozen encoder source identity mismatch')
    text = CachedText(package / 'text.pt', checked['index']['package_sha256']['text.pt'])
    assets['base_model_root'] = root / 'base_models'
    return dict(cfg=cfg, payload=payload, checked=checked, assets=assets, text=text,
                checkpoint=checkpoint, config_path=config_path, package=package,
                step=step, checkpoint_sha256=expected_hash, resources_sha256=weights,
                options=options)


def prepare_release_inputs(config):
    """Portable public release: code/config/stats in Git, only weights downloaded."""
    from .checkpoint_paths import resolve_policy_checkpoint
    options = validate_options(config)
    register_default_resolvers()
    release = read_release()
    spec = release['checkpoints'][options['family']]
    root = resolve_policy_checkpoint(config).resolve(strict=True)
    if not root.is_dir():
        raise ValueError('Public release checkpoint_path must be a downloaded weights directory')
    if config.get('expected_step') is not None and int(config['expected_step']) != spec['step']:
        raise ValueError('expected_step differs from released checkpoint')
    if config.get('checkpoint_sha256') and config['checkpoint_sha256'] != spec['sha256']:
        raise ValueError('checkpoint_sha256 differs from released checkpoint')
    _, assets = read_source()
    package = POLICY_DIR / 'phoenix_source'
    manifest = json.loads((package / 'manifest.json').read_text())
    if manifest.get('schema') != 'phoenix-inference-source-v1':
        raise ValueError('Incorrect public inference source schema')
    for name, expected in manifest['sha256'].items():
        if sha256(relative_path(package, name)) != expected:
            raise ValueError('Changed inference metadata: ' + name)
    config_path = package / (options['family'] + '.json')
    cfg = OmegaConf.load(config_path)
    checkpoint = relative_path(root, spec['path'])
    if checkpoint.stat().st_size != spec['size'] or sha256(checkpoint) != spec['sha256']:
        raise ValueError('Released checkpoint size/SHA256 mismatch')
    payload = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=True)
    checked = validate_training(cfg, payload, package, options['family'], spec['step'], packaged=True)
    weights = {}
    for key, resource in release['weights'].items():
        path = relative_path(root, resource['path'])
        if sha256(path) != resource['sha256']:
            raise ValueError('Frozen resource SHA256 mismatch: ' + key)
        assets[key], weights[key] = path, resource['sha256']
    identity = checked['identity']
    if (identity['teacher_sha256'] != weights['dino_checkpoint']
            or identity['stvae_checkpoint_sha256'] != weights['stvae_checkpoint']):
        raise ValueError('Checkpoint encoder resources differ from release')
    from wam.model.modules.codecs.joint_visual_encoder import encoder_source_sha256
    if encoder_source_sha256(assets['encoder_source_root']) != identity['source_sha256']:
        raise ValueError('Frozen encoder source identity mismatch')
    prepare_tokenizer(root, assets['tokenizer_root'])
    assets['base_model_root'] = root / 'base_models'
    return dict(cfg=cfg, payload=payload, checked=checked, assets=assets, text=None,
                checkpoint=checkpoint, config_path=config_path, package=package,
                step=spec['step'], checkpoint_sha256=spec['sha256'], resources_sha256=weights,
                options=options)


def build_phoenix_runtime(config):
    data = prepare_inputs(config)
    cfg, assets, checked = data['cfg'], data['assets'], data['checked']
    opts = data['options']
    device = str(config.get('device') or 'cuda:0')
    device = 'cuda:0' if device == 'cuda' else device
    if not device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('CUDA required; use preflight_phoenix for the CPU identity check')
    os.environ['DIFFSYNTH_SKIP_DOWNLOAD'] = 'true'
    os.environ['DIFFSYNTH_MODEL_BASE_PATH'] = str(assets['base_model_root'])
    torch.set_num_threads(1)
    torch.manual_seed(opts['seed'])
    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    model_cfg.load_text_encoder = data['text'] is None
    model_cfg.skip_dit_load_from_pretrain = True
    model_cfg.joint96_checkpoint_restore = True
    model_cfg.mot_checkpoint_mixed_attn = False
    model_cfg.mot_compile_gradient_checkpointing = False
    model_cfg.mot_compile_mode = 'off'
    model_cfg.video_dit_config.use_gradient_checkpointing = False
    model_cfg.action_dit_config.use_gradient_checkpointing = False
    model_cfg.action_dit_config.embodiment_registry_path = str(assets['registry'])
    codec = model_cfg.joint_visual_codec
    codec.source_root = str(assets['encoder_source_root'])
    codec.stvae_checkpoint_path = str(assets['stvae_checkpoint'])
    codec.dino_kwargs.checkpoint_path = str(assets['dino_checkpoint'])
    codec.dino_kwargs.starvla_source_path = str(assets['encoder_source_root'] / 'third_party/dino90k_runtime')
    codec.dino_kwargs.dinov3_source_path = str(Path(codec.dino_kwargs.starvla_source_path) / 'dinov3')
    model = instantiate(model_cfg, model_dtype=torch.bfloat16, device=device)
    paths = model.model_paths
    actual = {'wan_vae': paths['vae'],
              'dino_checkpoint': paths['joint_visual_encoder']['stvae']['teacher_checkpoint_path'],
              'stvae_checkpoint': paths['joint_visual_encoder']['stvae']['stvae_checkpoint_path']}
    if data['text'] is None:
        actual['t5_encoder'] = paths['text_encoder']
    if any(Path(path).resolve() != assets[key].resolve() for key, path in actual.items()):
        raise ValueError('Factory loaded a different frozen resource')
    _load_joint96_checkpoint(model, data['payload'])
    counts = {}
    for name in ('mot', 'proprio_encoder', 'action_proprio_encoder'):
        module = getattr(model, name, None)
        if module is None:
            continue
        tensors = module.state_dict()
        for key, value in tensors.items():
            if not torch.equal(value, data['payload'][name][key].to(value)):
                raise ValueError('Restored tensor mismatch: ' + name + '.' + key)
        counts[name] = len(tensors)
    model.eval().requires_grad_(False)
    visual_evidence = install_repeat9_joint(model.joint_visual_encoder)
    audit_dir = Path(str(config.get('audit_dir') or POLICY_DIR / 'logs/phoenix'))
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit = dict(checkpoint=str(data['checkpoint']), checkpoint_sha256=data['checkpoint_sha256'],
                 step=data['step'], pretrain_step=256013, family=opts['family'],
                 training_config=str(data['config_path']), training_config_sha256=sha256(data['config_path']),
                 normalization_contract=checked['contract']['normalization_contract'],
                 training_dataloader=checked['contract']['dataloader'],
                 training_rgb_geometry=checked['contract']['rgb_geometry'],
                 training_rgb_decode=checked['contract']['rgb_decode'],
                 cache_contract_sha256=checked['contract'].get('cache_contract_sha256'),
                 raw_cache_binding_sha256=checked['contract'].get('raw_cache_binding_sha256'),
                 training_text_cache_sha256=checked['index']['package_sha256']['text.pt'],
                 resources_sha256=data['resources_sha256'], strict_exact_tensor_counts=counts,
                 current_visual_mode=MODE, num_frames=9, action_horizon=32,
                 replan_steps=opts['replan'], num_inference_steps=opts['denoise'], seed=opts['seed'],
                 native_data_hz=checked['fps'], requested_action_hz=opts['hz'],
                 controller_rate_verified=False, input_color='RGB_uint8',
                 input_geometry='PIL_aspect_bicubic_center_crop_256_float32',
                 cameras=['main', 'left', 'right'], canonical_active_dims=ACTIVE,
                 text_padding='zero_features_all_128_keys_visible',
                 text_mode='online_t5' if data['text'] is None else 'training_cache_only',
                 output_limits='device_adapter_gripper_clip01_only_with_log',
                 video_scheduler=OmegaConf.to_container(cfg.model.video_scheduler, resolve=True),
                 action_scheduler=OmegaConf.to_container(cfg.model.action_scheduler, resolve=True),
                 sigma_shift_override=None, real_robot_rollout_verified=False,
                 train_inference_numerical_equivalence_verified=False)
    (audit_dir / 'eval_contract.json').write_text(json.dumps(audit, indent=2) + '\n')
    runtime = SimpleNamespace(model=model, processor=checked['processor'],
                              action_dim_is_pad=_static_dimension_is_pad(checked['processor'], 'action'),
                              device=device, model_dtype=torch.bfloat16,
                              robot_type=opts['family'], embodiment_id=ROBOTS[opts['family']][0],
                              posttrain_step=data['step'], pretrain_step=256013,
                              robot_support=checked['support'],
                              first_action_recorded=False, audit_dir=audit_dir,
                              current_visual_mode=MODE, text=data['text'], visual_evidence=visual_evidence)
    session_cfg = SimpleNamespace(num_inference_steps=opts['denoise'], replan_steps=opts['replan'],
                                  seed=opts['seed'], rand_device='cpu')
    return runtime, session_cfg


class PhoenixSession(RoboDojoRealJoint96Session):
    def _normalize_state(self, state):
        return super()._normalize_state(preprocess_state(state, self.runtime.robot_type))

    def _build_robotwin_image_tensor(self, observation):
        return current_rgb(observation).to(device=self.runtime.device)

    def _text_condition(self, instruction):
        if self.runtime.text is None:
            return super()._text_condition(instruction)
        return self.runtime.text.get(instruction, device=self.runtime.device, dtype=self.runtime.model_dtype)

    def _denormalize_action(self, action):
        if torch.count_nonzero(action[..., self.action_dim_is_pad.to(action.device)]):
            raise ValueError('Sampler returned nonzero padded action dimensions')
        return super()._denormalize_action(action)


def predict_phoenix(adapter, session, observation, instruction):
    from XPolicyLab.utils.process_data import unpack_robot_state
    if session.pending_model_actions:
        raise RuntimeError('Previous actions have not all been acknowledged')
    raw = np.asarray(session.get_action({'observation': observation, 'instruction': instruction}), dtype=np.float32)
    if raw.shape != (adapter.replan_steps, 14) or not np.isfinite(raw).all():
        raise ValueError('Invalid native absolute joint14 output')
    limited = raw.copy()
    limited[:, [6, 13]] = np.clip(raw[:, [6, 13]], 0., 1.)
    record = dict(family=adapter.env_cfg_type, instruction=instruction,
                  inference_call=session.get_timing_rollout()['replan_calls'],
                  gripper_before=raw[:, [6, 13]].tolist(), gripper_after=limited[:, [6, 13]].tolist(),
                  changed_values=int(np.count_nonzero(raw != limited)),
                  visual_encoding=dict(session.runtime.visual_evidence))
    with (session.runtime.audit_dir / 'device_adapter.jsonl').open('a') as stream:
        stream.write(json.dumps(record) + '\n')
    return unpack_robot_state(limited, adapter.action_type, adapter.robot_action_dim_info, source_type='obs')
