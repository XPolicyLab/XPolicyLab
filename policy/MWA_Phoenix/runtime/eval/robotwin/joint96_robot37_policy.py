"""Opt-in Robot37 Joint96 policy. Never route these weights through memory inference."""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from eval.robotwin.policy import RobotWinPolicySession, RobotWinSessionConfig, RobotWinSharedRuntime, _static_dimension_is_pad
from eval.robotwin.current_visual_adapter import configure_current_visual
from eval.robotwin.config import ROOT, inference_paths, resource_path
from wam.datasets.dataset_utils import ResizeSmallestSideAspectPreserving, CenterCrop, Normalize
from wam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from wam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from wam.inference.frame_joint import infer_frame_joint_window
from wam.training.checkpoint import _joint96_identity, _load_joint96_checkpoint
from wam.utils.config_resolvers import register_default_resolvers

CAMERAS = ('head_camera', 'left_camera', 'right_camera')
DATA_CAMERAS = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')
VALID_DIMS = list(range(6)) + [16] + list(range(40, 46)) + [56]


def current_rgb(processor, observation):
    """Exactly the clean training processor + dataset resize/crop/normalize.

    Keep floating pixels between stages; no PIL/uint8 roundtrip or RGB canvas.
    Evaluation deliberately omits stochastic training-only augmentation.
    """
    images = {}
    for camera, meta in zip(CAMERAS, processor.shape_meta['images']):
        rgb = np.asarray(observation['observation'][camera]['rgb'])
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError('Expected uint8 HWC RGB: ' + camera)
        images[meta['key']] = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0)
    views = processor._transform_images({'images': images})
    resize = ResizeSmallestSideAspectPreserving({'img_h': 256, 'img_w': 256})
    crop = CenterCrop({'img_h': 256, 'img_w': 256})
    norm = Normalize({'mean': .5, 'std': .5})
    result = torch.stack([norm(crop(resize(v))).clamp(-1, 1)[0] for v in views]).unsqueeze(0)
    if result.shape != (1, 3, 3, 256, 256) or not torch.isfinite(result).all():
        raise ValueError('Invalid current RGB geometry or values')
    return result


def validate_contract(cfg, payload, stats_path, expected_step=None):
    get = lambda key, default=None: OmegaConf.select(cfg, key, default=default)
    identity = _joint96_identity(payload.get('joint96_metadata'))
    conditioning = get('model.joint_visual_codec.current_conditioning', 'legacy_clip_prefix')
    if conditioning not in ('legacy_clip_prefix', 'independent_single_frame_v1'):
        raise ValueError('Unknown current-frame conditioning')
    if identity.get('current_conditioning', 'legacy_clip_prefix') != conditioning:
        raise ValueError('Checkpoint/config current-frame conditioning mismatch')
    contract = payload['joint96_metadata'].get('soft_prompt', {}).get('training_contract', {})
    for key, expected in {'domain': 'robotwin', 'embodiment_id': 2, 'embodiment_name': 'arx_x5',
                          'action_space': 'native_joint14_to_canonical80', 'cached_visuals': False}.items():
        if contract.get(key) != expected:
            raise ValueError('Not a Robot37 RoboTwin posttraining checkpoint: ' + key)
    # Explicit statistics resource; validate its dimensions through the processor.
    # Runtime file fingerprints are intentionally not an admission criterion.
    required = {'model.joint96_three_view': True, 'model.proprio_dim': 80,
                'model.action_dit_config.action_dim': 80, 'model.video_dit_config.in_dim': 96,
                'model.video_dit_config.out_dim': 96, 'model.action_dit_config.num_embodiments': 28,
                'model.action_dit_config.soft_prompt_tokens': 8,
                'joint96_posttrain.robotwin_softprompt.enabled': True,
                'joint96_posttrain.robotwin_softprompt.embodiment_id': 2,
                'joint96_posttrain.action_video_attention': 'current_and_noisy_future_v1',
                'data.train.domain': 'robotwin', 'data.train.concat_multi_camera': 'latent_horizontal',
                'data.train.single_canvas': False, 'data.train.num_frames': 33,
                'data.train.action_video_freq_ratio': 4, 'data.train.context_len': 128,
                'data.train.processor.norm_default_mode': 'z-score',
                'data.train.processor.action_state_transforms': None,
                'data.train.processor.use_stepwise_action_norm': False}
    for key, expected in required.items():
        if get(key) != expected:
            raise ValueError('Saved training contract mismatch: ' + key)
    if list(cfg.data.train.video_size) != [256, 256]:
        raise ValueError('Expected per-view 256 resolution')
    if [x.key for x in cfg.data.train.shape_meta.images] != list(DATA_CAMERAS):
        raise ValueError('Camera ordering differs from RoboTwin training')
    for key in ('model.memory.enabled', 'model.memory.video.enabled', 'model.understanding.enabled',
                'model.future_delta.enabled', 'model.distillation.enabled'):
        if get(key, False):
            raise ValueError('Unsupported conditioning: ' + key)
    if identity['action_video_attention'] != required['joint96_posttrain.action_video_attention']:
        raise ValueError('Checkpoint attention mismatch')
    return identity


def load_inference_config(path):
    register_default_resolvers()
    cfg = OmegaConf.load(resource_path(path))
    # Only relocate frozen resources; never override architecture or noise schedules.
    codec = cfg.model.joint_visual_codec
    codec.stvae_checkpoint_path = str(resource_path(
        os.environ.get('STVAE_CHECKPOINT') or codec.stvae_checkpoint_path))
    codec.dino_kwargs.checkpoint_path = str(resource_path(
        os.environ.get('DINO_ENCODER_CHECKPOINT') or codec.dino_kwargs.checkpoint_path))
    from wam.model.modules.codecs.dino_stvae.provenance import bundled_config
    cfg.model.joint_visual_codec = bundled_config(OmegaConf.to_container(codec, resolve=True))
    registry = str(resource_path(os.environ.get('EMBODIMENT_REGISTRY')
        or ROOT / 'configs/embodiment_registry/local_robot37_arm18_capacity28_v1.json'))
    cfg.local_robot_softprompt.registry_path = registry
    cfg.model.action_dit_config.embodiment_registry_path = registry
    return cfg


load_training_config = load_inference_config  # Compatibility for existing integrations.


def instantiate_checkpoint_model(model_config, device):
    """Skip only Wan DiT initialization weights, retaining factory contract checks.

    Called once per policy server before worker threads start. The full strict
    checkpoint restore in build_runtime is mandatory before the model is usable.
    Training factories and their default initialization behavior stay unchanged.
    """
    import wam.model.factory as factory
    original_loader = factory.load_wan22_ti2v_5b_components

    def checkpoint_only_components(**kwargs):
        kwargs['skip_dit_load_from_pretrain'] = True
        return original_loader(**kwargs)

    factory.load_wan22_ti2v_5b_components = checkpoint_only_components
    try:
        return instantiate(model_config, model_dtype=torch.bfloat16, device=device)
    finally:
        factory.load_wan22_ti2v_5b_components = original_loader


def build_runtime(checkpoint, training_config=None, stats_path=None, *, expected_step=None, output_dir,
                  device='cuda:0', denoise=10, replan=10, seed=0,
                  current_visual_mode='auto'):
    from wam.posttrain.guard import install_import_guard
    install_import_guard()
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    config_path, stats_path = inference_paths(training_config, stats_path)
    cfg = load_inference_config(config_path)
    print('ROBOTWIN_CHECKPOINT_LOAD', str(checkpoint), flush=True)
    payload = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=True)
    identity = validate_contract(cfg, payload, stats_path, expected_step)
    processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(stats_path)))
    for field in ('action', 'state'):
        if (~_static_dimension_is_pad(processor, field)).nonzero().flatten().tolist() != VALID_DIMS:
            raise ValueError('Canonical80 mapping mismatch: ' + field)
    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    model_cfg.load_text_encoder = True
    model_cfg.mot_checkpoint_mixed_attn = False
    model_cfg.mot_compile_gradient_checkpointing = False
    model_cfg.mot_compile_mode = 'off'
    model_cfg.video_dit_config.use_gradient_checkpointing = False
    model_cfg.action_dit_config.use_gradient_checkpointing = False
    # Use the exact training factory, then restore EVERY policy tensor strictly.
    print('ROBOTWIN_MODEL_BUILD', flush=True)
    model = instantiate_checkpoint_model(model_cfg, device)
    print('ROBOTWIN_STRICT_RESTORE', flush=True)
    _load_joint96_checkpoint(model, payload)
    counts = {}
    for group in ('mot', 'proprio_encoder'):
        actual = getattr(model, group).state_dict()
        for key, tensor in actual.items():
            if not torch.equal(tensor, payload[group][key].to(device=tensor.device, dtype=tensor.dtype)):
                raise ValueError('Restored tensor mismatch: ' + group + '.' + key)
        counts[group] = len(actual)
    model.eval().requires_grad_(False)
    print('ROBOTWIN_TENSORS_VERIFIED', counts, flush=True)
    current_visual_audit = configure_current_visual(model, current_visual_mode)
    runtime = RobotWinSharedRuntime(model, processor, str(checkpoint), Path(stats_path), device,
                                   torch.bfloat16, _static_dimension_is_pad(processor, 'action'))
    config = RobotWinSessionConfig(action_horizon=32, action_hz=50., replan_steps=replan,
        num_inference_steps=denoise, sigma_shift=None, seed=seed, rand_device='cpu', tiled=False,
        timing_enabled=True, num_video_frames=9, video_size=(256, 256),
        video_layout='latent_horizontal', video_view_names='main|left|right')
    from wam.utils.checkpoint_steps import checkpoint_steps
    audit = {**checkpoint_steps(payload), 'checkpoint': str(checkpoint),
             'inference_config': str(config_path),
             'training_config': str(config_path),  # Legacy receipt key.
             'wan_dit_initialization_weights_loaded': False,
             'normalization_path': str(stats_path), 'strict_exact_tensor_counts': counts,
             'identity': identity, 'embodiment_id': 2, 'cameras': list(CAMERAS),
             'canonical_valid_indices': VALID_DIMS, 'prompt_template': DEFAULT_PROMPT,
             'text_padding': 'zero_padded_features_all_keys_visible',
             'current_conditioning': identity.get('current_conditioning', 'legacy_clip_prefix'),
             'inference_current_visual': current_visual_audit,
             'train_inference_numerical_equivalence_verified': False,
             'num_frames': 9, 'action_horizon': 32, 'replan_steps': replan, 'num_inference_steps': denoise,
             'video_scheduler': OmegaConf.to_container(cfg.model.video_scheduler, resolve=True),
             'action_scheduler': OmegaConf.to_container(cfg.model.action_scheduler, resolve=True)}
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / 'MODEL_LOADED.json').write_text(json.dumps(audit, indent=2)+'\n')
    return runtime, config


class Robot37Joint96Session(RobotWinPolicySession):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.use_memory or self.use_ar_action_history or not self.model.joint96_three_view:
            raise ValueError('Robot37 Joint96 requires current-only conditioning')
        self._text_cache = {}
        self._state_pad = _static_dimension_is_pad(self.processor, 'state').unsqueeze(0).to(self.runtime.device)

    def _build_robotwin_image_tensor(self, observation):
        return current_rgb(self.processor, observation).to(device=self.runtime.device, dtype=self.runtime.model_dtype)

    @torch.no_grad()
    def _infer_action_chunk(self, observation, instruction):
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError('A nonempty raw instruction is required')
        start = time.perf_counter()
        rgb = self._build_robotwin_image_tensor(observation)
        joint = np.asarray(observation['joint_action']['vector'], dtype=np.float32)
        if joint.shape != (14,) or not np.isfinite(joint).all():
            raise ValueError('Expected finite native joint14 state')
        proprio = self._normalize_state(joint).to(device=self.runtime.device, dtype=self.runtime.model_dtype)
        prompt = DEFAULT_PROMPT.format(task=instruction)
        if prompt not in self._text_cache:
            context, mask = self.model.encode_prompt(prompt)
            if context.shape != (1, 128, 4096) or mask.shape != (1, 128) or not mask.bool().all():
                raise ValueError('Online T5 differs from official training padding semantics')
            self._text_cache[prompt] = (context, mask)
        context, mask = self._text_cache[prompt]
        result = infer_frame_joint_window(self.model, None, rgb, num_frames=9, action_horizon=32,
            proprio=proprio, context=context, context_mask=mask, num_inference_steps=self.num_inference_steps,
            seed=self.seed, rand_device=self.rand_device, tiled=False,
            video_layout='latent_horizontal', video_view_names='main|left|right',
            camera_is_pad=torch.zeros((1, 3), dtype=torch.bool, device=self.runtime.device),
            action_dim_is_pad=self.action_dim_is_pad.unsqueeze(0).to(self.runtime.device),
            proprio_dim_is_pad=self._state_pad,
            embodiment_ids=torch.tensor([2], dtype=torch.long, device=self.runtime.device))
        normalized = result['action']
        if normalized.shape != (32, 80) or not torch.isfinite(normalized).all():
            raise ValueError('Invalid normalized action chunk')
        action = self._denormalize_action(normalized)[0].astype(np.float32)
        if action.shape != (32, 14) or not np.isfinite(action).all():
            raise ValueError('Invalid native action chunk')
        self._timing_rollout['infer_s'] += time.perf_counter() - start
        self._timing_rollout['replan_calls'] += 1
        self._timing_rollout['predicted_action_steps'] += 32
        return action, normalized.detach().float().cpu()
