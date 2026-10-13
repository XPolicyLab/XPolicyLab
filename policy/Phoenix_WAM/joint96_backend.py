"""RoboDojo three-robot Joint96 policy, adapted from the audited A100 evaluator."""
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize as resize_rgb
from hydra.utils import instantiate
from omegaconf import OmegaConf

from collections import deque
from types import SimpleNamespace
from .current_visual_adapter import configure_current_visual
from wam.datasets.dataset_utils import ResizeSmallestSideAspectPreserving, CenterCrop, Normalize
DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"
from wam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from wam.inference.frame_joint import infer_frame_joint_window
from .joint96_checkpoint import _joint96_identity, _load_joint96_checkpoint
from wam.utils.config_resolvers import register_default_resolvers

def _static_dimension_is_pad(processor, field: str) -> torch.Tensor:
    """Derive a fixed canonical-dimension padding mask from the merger."""

    if field not in {"action", "state"}:
        raise ValueError(f"Unsupported merged field: {field}")
    merger = processor.action_state_merger
    meta = processor.shape_meta[field]
    source_dim = sum(int(item["shape"]) for item in meta)
    target_dim = getattr(merger, f"{field}_target_dim")
    target_dim = source_dim if target_dim is None else int(target_dim)
    target_slices = getattr(merger, f"{field}_target_slices")
    mask = torch.ones(target_dim, dtype=torch.bool)
    if target_slices is None:
        mask[:source_dim] = False
    else:
        for entry in target_slices:
            start, end = (int(value) for value in entry["target_slice"])
            mask[start:end] = False
    return mask


CAMERAS = ('head_camera', 'left_camera', 'right_camera')
DATA_CAMERAS = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')
VALID_DIMS = list(range(6)) + [16] + list(range(40, 46)) + [56]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def current_rgb(processor, observation):
    """Adapt decoded RGB to stored training resolution, then preprocess it.

    The saved processor consumes uint8 256x256 source videos. Live cameras
    may be rectangular: resize the entire view at this boundary (no crop).
    After ToTensor, keep floating pixels between the training stages.
    This does not reproduce the training videos' H264 resampling numerically.
    Evaluation deliberately omits stochastic training-only augmentation.
    """
    images = {}
    for camera, meta in zip(CAMERAS, processor.shape_meta['images']):
        rgb = np.asarray(observation['observation'][camera]['rgb'])
        if (rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3
                or min(rgb.shape[:2]) <= 0):
            raise ValueError('Expected uint8 HWC RGB: ' + camera)
        frame = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0)
        target_hw = list(meta['raw_shape'][-2:])
        if list(frame.shape[-2:]) != target_hw:
            frame = resize_rgb(frame, target_hw, interpolation=InterpolationMode.BILINEAR,
                               antialias=True)
        images[meta['key']] = frame
    views = processor._transform_images({'images': images})
    resize = ResizeSmallestSideAspectPreserving({'img_h': 256, 'img_w': 256})
    crop = CenterCrop({'img_h': 256, 'img_w': 256})
    norm = Normalize({'mean': .5, 'std': .5})
    result = torch.stack([norm(crop(resize(v))).clamp(-1, 1)[0] for v in views]).unsqueeze(0)
    if result.shape != (1, 3, 3, 256, 256) or not torch.isfinite(result).all():
        raise ValueError('Invalid current RGB geometry or values')
    return result


def validate_contract(cfg, payload, stats_path, expected_step):
    """Bind an explicit posttraining checkpoint to its real-data cache and stats."""
    get = lambda key, default=None: OmegaConf.select(cfg, key, default=default)
    if type(expected_step) is not int or expected_step <= 0 or payload.get('step') != expected_step:
        raise ValueError('Evaluation checkpoint step mismatch')
    if expected_step > int(cfg.checkpoint_contract.max_posttrain_steps):
        raise ValueError('Checkpoint exceeds the saved posttraining budget')
    identity = _joint96_identity(payload.get('joint96_metadata'))
    conditioning = get('model.joint_visual_codec.current_conditioning', 'legacy_clip_prefix')
    if conditioning != 'legacy_clip_prefix' or identity.get('current_conditioning', 'legacy_clip_prefix') != conditioning:
        raise ValueError('This repeat9 evaluator requires legacy_clip_prefix training identity')
    contract = payload['joint96_metadata'].get('soft_prompt', {}).get('training_contract', {})
    recipes = {
        'robodojo_real_half_video_v1': {'vae': .25, 'stvae': .25, 'action': 1.0},
        'robodojo_real_standard_video_v1': {'vae': .5, 'stvae': .5, 'action': 1.0},
    }
    loss_weights = recipes.get(get('checkpoint_contract.loss_recipe'))
    if loss_weights is None or dict(cfg.checkpoint_contract.loss_weights) != loss_weights:
        raise ValueError('Unsupported or inconsistent saved posttraining loss recipe')
    required_contract = dict(domain='robodojo_real',
        embodiment_ids={str(cfg.robot_profile.robot_type): int(cfg.robot_profile.embodiment_id)},
        action_space='native_joint14_to_canonical80', cached_visuals=True,
        normalization='frame_weighted_per_mechanical_family_joint14',
        loss_weights=loss_weights)
    for key, expected in required_contract.items():
        if contract.get(key) != expected:
            raise ValueError(f'Checkpoint does not match {cfg.robot_profile.robot_type} posttraining: {key}')
    required = {'model.joint96_three_view': True, 'model.proprio_dim': 80,
        'model.action_dit_config.action_dim': 80, 'model.video_dit_config.in_dim': 96,
        'model.video_dit_config.out_dim': 96, 'model.action_dit_config.num_embodiments': 28,
        'model.action_dit_config.soft_prompt_tokens': 8,
        'checkpoint_contract.enabled': True, 'checkpoint_contract.cached_visuals': True,
        'checkpoint_contract.robodojo_real_family_cache': True,
        'checkpoint_contract.robodojo_real_softprompt.enabled': True,
        'checkpoint_contract.action_video_attention': 'current_and_noisy_future_v1',
        'observation.domain': 'robodojo', 'observation.embodiment': cfg.robot_profile.robot_type,
        'observation.concat_multi_camera': 'latent_horizontal', 'observation.single_canvas': False,
        'observation.num_frames': 33, 'observation.action_video_freq_ratio': 4,
        'observation.context_len': 128, 'observation.processor.norm_default_mode': 'z-score',
        'observation.processor.action_state_transforms': None,
        'observation.processor.use_stepwise_action_norm': False,
        'model.joint_visual_codec.expected_stvae_step': 150000}
    for key, expected in required.items():
        if get(key) != expected:
            raise ValueError('Saved training contract mismatch: ' + key)
    if list(cfg.observation.video_size) != [256, 256]:
        raise ValueError('Expected per-view 256 resolution')
    if [x.key for x in cfg.observation.shape_meta.images] != list(DATA_CAMERAS):
        raise ValueError('Camera ordering differs from real-data training')
    for key in ('model.memory.enabled', 'model.memory.video.enabled', 'model.understanding.enabled',
                'model.future_delta.enabled', 'model.distillation.enabled'):
        if get(key, False):
            raise ValueError('Unsupported conditioning: ' + key)
    if identity['action_video_attention'] != required['checkpoint_contract.action_video_attention']:
        raise ValueError('Checkpoint attention mismatch')
    source_step = payload['joint96_metadata']['model_paths'].get('posttrain_initialization', {}).get('step')
    expected_pretrain_step = get('checkpoint_contract.expected_pretrain_step')
    if type(expected_pretrain_step) is not int or expected_pretrain_step <= 0 or source_step != expected_pretrain_step:
        raise ValueError('Saved config and checkpoint pretraining lineage differ')
    # The training checkpoint binds the cache contract. That contract hashes the
    # package manifest, which in turn hashes each family's normalization file.
    # Follow this chain instead of accepting unrelated simulator statistics.
    cache_root = Path(cfg.cache_metadata)
    manifest = json.loads((cache_root / 'manifest.json').read_text())
    receipt = json.loads((cache_root / 'COMPLETE.json').read_text())
    signature = hashlib.sha256(json.dumps(manifest['contract'], sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if (manifest['contract_sha256'] != signature or receipt['contract_sha256'] != signature
            or contract.get('cache_contract_sha256') != signature
            or receipt.get('validation') != 'distributed_full_sha256_rowhash_and_finite'):
        raise ValueError('Checkpoint/cache contract mismatch')
    encoder_identity = dict(identity)
    encoder_identity.pop('soft_prompt', None)
    if manifest['contract']['encoder_identity'] != encoder_identity:
        raise ValueError('Checkpoint/cache encoder mismatch')
    package_root = Path(cfg.normalization_metadata)
    package_file = package_root / 'manifest.json'
    if sha256(package_file) != manifest['contract']['package_sha256']:
        raise ValueError('Training package manifest changed')
    package = json.loads(package_file.read_text())
    if (package['contract'] != 'robodojo.real.v3.joint14.robot37.v1'
            or sha256(stats_path) != package['normalization_sha256'][str(cfg.robot_profile.embodiment_id)]):
        raise ValueError('Selected robot normalization differs from the training package')
    families = {(t['robot_type'], t['embodiment_id']) for t in package['tasks']}
    if (cfg.robot_profile.robot_type, cfg.robot_profile.embodiment_id) not in families:
        raise ValueError('Robot profile does not belong to the normalization package')
    return identity


def load_inference_config(path, assets):
    register_default_resolvers()
    cfg = OmegaConf.load(path)
    cfg.robot_profile = OmegaConf.create(assets['robot_profile'])
    cfg.observation.embodiment = cfg.robot_profile.robot_type
    release = assets['checkpoint_spec']
    cfg.checkpoint_contract.expected_pretrain_step = int(release['pretrain_step'])
    cfg.checkpoint_contract.loss_recipe = release['loss_recipe']
    cfg.checkpoint_contract.loss_weights = OmegaConf.create(release['loss_weights'])
    # Relocate the inference configuration without changing model parameters.
    codec = cfg.model.joint_visual_codec
    codec.source_root = str(assets['encoder_source_root'])
    codec.stvae_checkpoint_path = str(assets['stvae_checkpoint'])
    codec.dino_kwargs.checkpoint_path = str(assets['dino_checkpoint'])
    codec.dino_kwargs.starvla_source_path = str(assets['encoder_source_root'] / 'third_party/dino90k_runtime')
    codec.dino_kwargs.dinov3_source_path = str(Path(codec.dino_kwargs.starvla_source_path) / 'dinov3')
    cfg.model.action_dit_config.embodiment_registry_path = str(assets['registry'])
    cfg.cache_metadata = str(assets['cache_metadata'])
    cfg.normalization_metadata = str(assets['normalization_metadata'])
    return cfg


def verify_loaded_weight_paths(model, assets):
    """Bind the frozen models actually loaded by the factory to the release files."""
    actual = model.model_paths
    encoder = actual['joint_visual_encoder']['stvae']
    loaded = dict(wan_vae=actual['vae'], t5_encoder=actual['text_encoder'],
                  dino_checkpoint=encoder['teacher_checkpoint_path'],
                  stvae_checkpoint=encoder['stvae_checkpoint_path'],
                  tokenizer_root=actual['tokenizer'])
    for key, value in loaded.items():
        if value is None or Path(value).resolve() != Path(assets[key]).resolve():
            raise ValueError('Model factory loaded a different release resource: ' + key)
    return {key: str(Path(value).resolve()) for key, value in loaded.items()}


def build_runtime(checkpoint, inference_config, stats_path, *, expected_step, output_dir, assets,
                  device='cuda:0', denoise=10, replan=10, seed=0,
                  current_visual_mode='repeat9_full_stvae'):
    os.environ['DIFFSYNTH_SKIP_DOWNLOAD'] = 'true'
    os.environ['DIFFSYNTH_MODEL_BASE_PATH'] = str(assets['base_model_root'])
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    if not 1 <= replan <= 32 or denoise <= 0:
        raise ValueError('Require 1 <= replan <= 32 and denoise > 0')
    cfg = load_inference_config(inference_config, assets)
    payload = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=True)
    identity = validate_contract(cfg, payload, stats_path, expected_step)
    from .robot_profiles import checkpoint_robot_support
    support = checkpoint_robot_support(payload, assets['robot_profile'])
    if not support['posttrained']:
        raise ValueError('Selected checkpoint was not posttrained for ' + support['robot_type'])
    processor = instantiate(cfg.observation.processor).eval()
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(stats_path)))
    for field in ('action', 'state'):
        if (~_static_dimension_is_pad(processor, field)).nonzero().flatten().tolist() != VALID_DIMS:
            raise ValueError('Canonical80 mapping mismatch: ' + field)
    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    model_cfg.load_text_encoder = True
    # Both DiT experts are fully restored below; no base DiT download is needed.
    model_cfg.skip_dit_load_from_pretrain = True
    model_cfg.joint96_checkpoint_restore = True
    model_cfg.mot_checkpoint_mixed_attn = False
    model_cfg.mot_compile_gradient_checkpointing = False
    model_cfg.mot_compile_mode = 'off'
    model_cfg.video_dit_config.use_gradient_checkpointing = False
    model_cfg.action_dit_config.use_gradient_checkpointing = False
    # Use the exact training factory, then restore EVERY policy tensor strictly.
    print('[Joint96] Loading frozen backbones and constructing the model...', flush=True)
    model = instantiate(model_cfg, model_dtype=torch.bfloat16, device=device)
    loaded_weights = verify_loaded_weight_paths(model, assets)
    print('[Joint96] Restoring and checking policy tensors...', flush=True)
    _load_joint96_checkpoint(model, payload)
    counts = {}
    for group in ('mot', 'proprio_encoder', 'action_proprio_encoder'):
        if getattr(model, group, None) is None:
            continue
        actual = getattr(model, group).state_dict()
        for key, tensor in actual.items():
            if not torch.equal(tensor, payload[group][key].to(device=tensor.device, dtype=tensor.dtype)):
                raise ValueError('Restored tensor mismatch: ' + group + '.' + key)
        counts[group] = len(actual)
    model.eval().requires_grad_(False)
    current_visual_audit = configure_current_visual(model, current_visual_mode)
    runtime = SimpleNamespace(model=model, processor=processor, checkpoint_path=str(checkpoint),
        dataset_stats_path=Path(stats_path), device=device, model_dtype=torch.bfloat16,
        robot_type=support['robot_type'], embodiment_id=support['embodiment_id'], robot_support=support,
        action_dim_is_pad=_static_dimension_is_pad(processor, 'action'))
    config = SimpleNamespace(action_horizon=32, action_hz=25., replan_steps=replan,
        num_inference_steps=denoise, sigma_shift=None, seed=seed, rand_device='cpu', tiled=False,
        timing_enabled=True, num_video_frames=9, video_size=(256, 256),
        video_layout='latent_horizontal', video_view_names='main|left|right')
    audit = {'pretrain_step': payload['joint96_metadata']['model_paths']['posttrain_initialization']['step'], 'checkpoint': str(checkpoint), 'step': expected_step, 'checkpoint_sha256': sha256(checkpoint),
             'inference_config': str(inference_config), 'inference_config_sha256': sha256(inference_config),
             'normalization_sha256': sha256(stats_path), 'strict_exact_tensor_counts': counts,
             'identity': identity, 'training_domain': 'robodojo_real', 'embodiment_id': support['embodiment_id'], 'robot_support': support,
             'robot_profile': str(assets['robot_profile_path']), 'cameras': list(CAMERAS),
             'canonical_valid_indices': VALID_DIMS, 'prompt_template': DEFAULT_PROMPT,
             'text_padding': 'zero_padded_features_all_keys_visible',
             'current_conditioning': identity.get('current_conditioning', 'legacy_clip_prefix'),
             'inference_current_visual': current_visual_audit,
             'skip_dit_load_from_pretrain': True,
             'loaded_resource_paths': loaded_weights,
             'input_rgb_resize': 'uint8_full_view_bilinear_antialias_to_256x256_if_needed',
             'train_inference_numerical_equivalence_verified': False,
             'num_frames': 9, 'action_horizon': 32, 'replan_steps': replan, 'num_inference_steps': denoise,
             'video_scheduler': OmegaConf.to_container(cfg.model.video_scheduler, resolve=True),
             'action_scheduler': OmegaConf.to_container(cfg.model.action_scheduler, resolve=True)}
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / 'MODEL_LOADED.json').write_text(json.dumps(audit, indent=2)+'\n')
    runtime.audit_dir = Path(output_dir)
    runtime.posttrain_step = expected_step
    runtime.pretrain_step = audit['pretrain_step']
    runtime.first_action_recorded = False
    return runtime, config


class RoboDojoRealJoint96Session:
    def __init__(self, runtime, config):
        self.runtime = runtime
        self.model = runtime.model
        self.processor = runtime.processor
        self.action_dim_is_pad = runtime.action_dim_is_pad
        self.num_inference_steps = config.num_inference_steps
        self.replan_steps = config.replan_steps
        self.seed = config.seed
        self.rand_device = config.rand_device
        self._text_cache = {}
        self._state_pad = _static_dimension_is_pad(self.processor, 'state').unsqueeze(0).to(runtime.device)
        self.reset_model()

    def reset_model(self):
        self.pending_model_actions = deque()
        self._timing_rollout = dict(infer_s=0., replan_calls=0, predicted_action_steps=0,
                                   scheduled_action_steps=0, executed_action_steps=0)
        return True

    def get_timing_rollout(self):
        return dict(self._timing_rollout)

    def update_obs(self, observation):
        if not self.pending_model_actions:
            raise RuntimeError('Observation without a pending action')
        self.pending_model_actions.popleft()
        self._timing_rollout['executed_action_steps'] += 1

    def get_action(self, request):
        if self.pending_model_actions:
            raise RuntimeError('Previous actions have not all been acknowledged')
        actions, normalized = self._infer_action_chunk(request['observation'], request['instruction'])
        self.pending_model_actions.extend(normalized[:self.replan_steps])
        self._timing_rollout['scheduled_action_steps'] += self.replan_steps
        return actions[:self.replan_steps]

    def _normalize_state(self, state: np.ndarray) -> torch.Tensor:
        state_meta = self.processor.shape_meta["state"]
        if len(state_meta) != 1:
            raise ValueError("Expected exactly one merged state key in shape_meta['state'].")
        state_key = state_meta[0]["key"]

        state_batch = {"state": {state_key: torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)}}
        state_batch = self.processor.action_state_transform(state_batch)
        state_batch = self.processor.normalizer.forward(state_batch)
        state_batch = self.processor.action_state_merger.forward(state_batch)
        return state_batch["state"]

    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3:
            raise ValueError(f"Expected action tensor [B,T,D], got {tuple(action.shape)}")

        action_meta = self.processor.shape_meta["action"]
        if len(action_meta) != 1:
            raise ValueError("Expected exactly one merged action key in shape_meta['action'].")

        action = action.to(dtype=torch.float32, device="cpu")
        batch = {"action": action, "state": action.new_zeros(action.shape)}
        batch = self.processor.action_state_merger.backward(batch)
        batch = self.processor.normalizer.backward(batch)
        return batch["action"][action_meta[0]["key"]].numpy()

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
            embodiment_ids=torch.tensor([self.runtime.embodiment_id], dtype=torch.long, device=self.runtime.device))
        normalized = result['action']
        if normalized.shape != (32, 80) or not torch.isfinite(normalized).all():
            raise ValueError('Invalid normalized action chunk')
        action = self._denormalize_action(normalized)[0].astype(np.float32)
        if action.shape != (32, 14) or not np.isfinite(action).all():
            raise ValueError('Invalid native action chunk')
        if not self.runtime.first_action_recorded:
            record = dict(step=self.runtime.posttrain_step, pretrain_step=self.runtime.pretrain_step,
                robot_type=self.runtime.robot_type, embodiment_id=self.runtime.embodiment_id,
                inference_current_visual_mode='repeat9_full_stvae', action_horizon=32,
                replan_steps=self.replan_steps, finite_native_joint14=True)
            (self.runtime.audit_dir / 'FIRST_ACTION.json').write_text(json.dumps(record, indent=2) + '\n')
            self.runtime.first_action_recorded = True
        self._timing_rollout['infer_s'] += time.perf_counter() - start
        self._timing_rollout['replan_calls'] += 1
        self._timing_rollout['predicted_action_steps'] += 32
        return action, normalized.detach().float().cpu()
