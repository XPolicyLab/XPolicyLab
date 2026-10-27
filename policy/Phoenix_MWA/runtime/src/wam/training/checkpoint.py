import copy
import hashlib
import json
from pathlib import Path
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from wam.utils.logging_config import get_logger
from wam.model.modules.conditioning.frame_attention import JOINT96_ACTION_VIDEO_ATTENTION
from wam.training.multiview_recipe import MULTIVIEW_RESUME_SCHEMA, immutable_multiview_contract, multiview_enabled, validate_multiview_budget
logger = get_logger(__name__)

def joint96_resume_recipe(cfg):
    recipe = OmegaConf.to_container(cfg, resolve=True)
    for key in ('output_dir', 'resume', 'wandb'):
        recipe.pop(key, None)
    recipe['local_robot_joint96'].pop('stop_after_step', None)
    recipe['local_robot_joint96'].pop('integration_trace', None)
    if 'local_robot_softprompt' in recipe:
        recipe['local_robot_softprompt'].pop('init_checkpoint', None)
        recipe['local_robot_softprompt'].pop('expected_init_step', None)
    if multiview_enabled(cfg):
        recipe.pop('max_steps', None)
        recipe['local_robot_multiview'].pop('resume_mask_change', None)
    return recipe

def joint96_json_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def joint96_file_sha256(path):
    """File existence only; runtime content hashing is disabled."""
    Path(path).resolve(strict=True)
    return None

def _resume_mask_transition(root, payload, cfg, current_recipe):
    """Authorize only pinned mask, cadence, language and block-policy transitions."""
    policy = OmegaConf.select(cfg, 'local_robot_multiview.resume_mask_change')
    if policy is None:
        return None
    policy = OmegaConf.to_container(policy, resolve=True)
    required = {'source_state', 'source_step', 'from_prob', 'to_prob'}
    cadence_fields = {'from_save_every', 'to_save_every'}
    language_fields = {'from_language_prob', 'to_language_prob'}
    block_fields = {'from_block_policy', 'to_block_policy'}
    accepted_fields = (required, required | cadence_fields, required | cadence_fields | language_fields, required | block_fields)
    if not isinstance(policy, dict) or set(policy) not in accepted_fields:
        raise ValueError('Invalid resume_mask_change policy fields')
    change_cadence = cadence_fields.issubset(policy)
    change_language = language_fields.issubset(policy)
    change_block = block_fields.issubset(policy)
    if change_language and (policy['from_language_prob'] != 0.5 or policy['to_language_prob'] != 0.7):
        raise ValueError('Only task-language augmentation 0.5 -> 0.7 is supported')
    from_prob, to_prob = (policy['from_prob'], policy['to_prob'])
    if change_block:
        if not multiview_enabled(cfg) or (from_prob, to_prob) != (0.075, 0.1225) or policy['from_block_policy'] != 'none' or (policy['to_block_policy'] != 'reserve_clean_v1'):
            raise ValueError('Only mask 0.075 -> 0.1225 with reserve_clean_v1 is supported')
    elif change_cadence:
        if not multiview_enabled(cfg) or (from_prob, to_prob) != (0.055, 0.075) or type(policy['from_save_every']) is not int or (type(policy['to_save_every']) is not int) or (policy['from_save_every'] != 2500) or (policy['to_save_every'] != 5000):
            raise ValueError('Only mask 0.055 -> 0.075 with save_every 2500 -> 5000 is supported')
    elif not multiview_enabled(cfg) or (from_prob, to_prob) != (0.03, 0.055):
        raise ValueError('Only the explicit multiview mask 0.03 -> 0.055 transition is supported')
    source = Path(str(policy['source_state']))
    step = policy['source_step']
    if not source.is_absolute() or source.resolve() != root or cfg.resume is None or (Path(str(cfg.resume)).resolve() != root) or (type(step) is not int) or (step <= 0) or (payload.get('global_step') != step) or (root.name != f'step_{step:06d}'):
        raise ValueError('resume_mask_change source path/step does not match the checkpoint')
    saved = copy.deepcopy(payload['resume_contract']['recipe'])
    try:
        augmentation = saved['data']['train']['main_view_augmentation']
        target = current_recipe['data']['train']['main_view_augmentation']
    except KeyError as exc:
        raise ValueError('resume_mask_change requires explicit augmentation configurations') from exc
    if augmentation.get('mask_prob') != from_prob or target.get('mask_prob') != to_prob:
        raise ValueError('resume_mask_change saved/requested mask does not match its policy')
    augmentation['mask_prob'] = to_prob
    if change_block:
        saved_train = saved['data']['train']
        target_train = current_recipe['data']['train']
        if saved_train.get('main_view_block_policy', 'none') != 'none' or target_train.get('main_view_block_policy') != 'reserve_clean_v1' or augmentation.get('sensor_noise_prob') != 0.075 or (augmentation.get('enabled') is not True):
            raise ValueError('Block transition requires legacy enabled augmentation and noise 0.075')
        saved_train['main_view_block_policy'] = 'reserve_clean_v1'
    if change_cadence:
        if type(saved.get('save_every')) is not int or type(current_recipe.get('save_every')) is not int or saved['save_every'] != policy['from_save_every'] or (current_recipe['save_every'] != policy['to_save_every']):
            raise ValueError('resume_mask_change saved/requested save_every does not match its policy')
        saved['save_every'] = policy['to_save_every']
    if change_language:
        saved_train = saved['data']['train']
        target_train = current_recipe['data']['train']
        if saved_train.get('language_augmentation_prob', 0.5) != 0.5 or target_train.get('language_augmentation_prob') != 0.7:
            raise ValueError('resume_mask_change saved/requested language probability mismatch')
        saved_train['language_augmentation_prob'] = 0.7
    if saved != current_recipe:
        raise ValueError('resume_mask_change cannot change any other recipe field')
    return {**policy, 'source_state': str(root), 'source_contract_sha256': joint96_json_sha256(payload['resume_contract']), 'source_trainer_state_sha256': joint96_file_sha256(root / 'trainer_state.json')}

def validate_joint96_resume_directory(path, *, cfg, world_size, expected_contract=None):
    """Validate an explicit trusted full-state directory before mutating the model."""
    root = Path(str(path)).expanduser()
    if not root.is_absolute() or not root.is_dir():
        raise ValueError('Joint96 resume requires an explicit absolute full-state directory, not weights/latest/auto')
    root = root.resolve()
    payload = json.loads((root / 'trainer_state.json').read_text())
    complete = json.loads((root / 'COMPLETE.json').read_text())
    contract = payload.get('resume_contract', {})
    multiview = multiview_enabled(cfg)
    required_schema = MULTIVIEW_RESUME_SCHEMA if multiview else 'phoenix.local_joint96.resume.v1'
    if contract.get('schema') != required_schema:
        raise ValueError('Checkpoint lacks the exact Joint96 resume contract; legacy state cannot resume safely')
    current_recipe = joint96_resume_recipe(cfg)
    transition = _resume_mask_transition(root, payload, cfg, current_recipe)
    if transition is None and contract.get('recipe') != current_recipe:
        raise ValueError('Joint96 resume recipe mismatch (including schedule, batch, seed and augmentation)')
    if multiview:
        validate_multiview_budget(contract, cfg)
        saved_identity = copy.deepcopy(immutable_multiview_contract(contract))
        if transition is not None:
            saved_identity['recipe'] = current_recipe
        expected_identity = immutable_multiview_contract(expected_contract) if expected_contract is not None else None
    else:
        saved_identity, expected_identity = (contract, expected_contract)
    if expected_identity is not None and saved_identity != expected_identity:
        raise ValueError('Joint96 resume data/norm/encoder/sampler identity mismatch')
    step = payload.get('global_step')
    if multiview and (type(step) is not int or step > contract['run_budget']['max_steps']):
        raise ValueError('Multiview saved step exceeds its recorded stopping budget')
    if type(step) is not int or not 0 < step < int(cfg.max_steps) or root.name != f'step_{step:06d}' or (payload.get('optimizer_updates') != step) or (complete.get('step') != step) or (complete.get('optimizer_updates') != step) or (payload.get('world_size') != world_size) or (complete.get('world_size') != world_size) or (contract.get('world_size') != world_size):
        raise ValueError('Joint96 checkpoint step/world geometry mismatch')
    if Path(str(cfg.output_dir)).resolve() == root.parents[2]:
        raise ValueError('Joint96 resume must use a new output branch; preserve the parent run')
    batches = int(contract['batches_per_epoch'])
    epoch, consumed = (payload.get('epoch'), payload.get('batch_in_epoch'))
    if type(epoch) is not int or type(consumed) is not int or epoch < 0 or (batches <= 0) or (not 0 <= consumed <= batches) or (epoch * batches + consumed != step):
        raise ValueError('Joint96 consumed batch cursor is inconsistent with optimizer updates')
    required = ['trainer_state.json', 'scheduler.bin', *[f'random_states_{rank}.pkl' for rank in range(world_size)]]
    if bool(cfg.local_robot_joint96.get('require_deepspeed', True)):
        models = list(root.rglob('*model_states.pt'))
        optimizers = list(root.rglob('*optim_states.pt'))
        expected_names = {f'bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt' for rank in range(world_size)}
        if len(models) != 1 or len(optimizers) != world_size or {p.name for p in optimizers} != expected_names:
            raise ValueError('Incomplete Joint96 ZeRO-1 model/optimizer rank shards')
        required.extend((str(p.relative_to(root)) for p in [*models, *optimizers]))
    if not set(required).issubset(complete.get('files', [])):
        raise ValueError('Required resume files missing from COMPLETE manifest')
    sizes = complete.get('file_sizes', {})
    for name in complete['files']:
        file = root / name
        if not file.resolve().is_relative_to(root) or not file.is_file() or file.stat().st_size <= 0 or (sizes.get(name) != file.stat().st_size):
            raise ValueError(f'Missing/truncated Joint96 resume file: {name}')
    if transition is not None:
        payload['applied_resume_mask_change'] = transition
    return (root, payload)

def joint96_checkpoint_metadata(model):
    if not bool(getattr(model, 'joint96_three_view', False)):
        return None
    paths = copy.deepcopy(getattr(model, 'model_paths', {}))
    encoder = paths.get('joint_visual_encoder', {})
    if not encoder.get('stvae'):
        raise ValueError('Joint96 checkpoint requires frozen encoder source and checkpoint metadata')
    metadata = {'schema': 'phoenix.local_joint96.weights.v1', 'model_paths': paths, 'channel_order': ['wan_vae48', 'stvae48'], 'camera_order': ['main', 'left', 'right'], 'action_dim': 80, 'proprio_dim': 80, 'action_video_attention': JOINT96_ACTION_VIDEO_ATTENTION}
    prompt = getattr(getattr(model, 'action_expert', None), 'embodiment_prompt', None)
    if prompt is not None:
        metadata['soft_prompt'] = prompt.metadata()
        contract = getattr(model, 'soft_prompt_training_contract', None)
        if contract is not None:
            metadata['soft_prompt']['training_contract'] = copy.deepcopy(contract)
    json.dumps(metadata, allow_nan=False)
    return metadata

def _joint96_identity(metadata):
    if not isinstance(metadata, dict) or metadata.get('schema') != 'phoenix.local_joint96.weights.v1':
        raise ValueError('Missing or incompatible Joint96 checkpoint metadata')
    attention = metadata.get('action_video_attention')
    if not isinstance(attention, str) or not attention:
        raise ValueError('Joint96 checkpoint identity missing a valid action_video_attention; legacy attention semantics must be verified before migration')
    encoder = metadata['model_paths']['joint_visual_encoder']
    stvae = encoder['stvae']
    identity = {key: metadata[key] for key in ('channel_order', 'camera_order', 'action_dim', 'proprio_dim', 'action_video_attention')}
    for key in ('source_sha256', 'posterior', 'per_camera_input_hw', 'temporal_compression', 'extra_latent_scaling'):
        identity[key] = encoder.get(key)
    if encoder.get('source_root') == 'bundled://dino-stvae-v1':
        from wam.model.modules.codecs.dino_stvae.provenance import canonical_source_identity
        identity['source_sha256'] = canonical_source_identity(encoder)
    if 'current_conditioning' in encoder:
        if encoder['current_conditioning'] != 'independent_single_frame_v1':
            raise ValueError('Unknown Joint96 current_conditioning')
        identity['current_conditioning'] = encoder['current_conditioning']
    for key in ('stvae_checkpoint_sha256', 'teacher_sha256', 'stvae_step', 'teacher_checkpoint_state', 'teacher_output_source', 'teacher_preprocessing', 'dinov3_source_commit'):
        identity[key] = stvae.get(key)
    if 'soft_prompt' in metadata:
        from wam.embodiment_registry import EmbodimentRegistry
        prompt = metadata['soft_prompt']
        registry = prompt.get('registry', {})
        EmbodimentRegistry.validate(registry)
        if prompt.get('schema') != 'wam.action_soft_prompt.v1' or prompt.get('num_embodiments') != registry['num_embodiments'] or any((type(prompt.get(key)) is not int or prompt[key] <= 0 for key in ('num_tokens', 'context_dim'))):
            raise ValueError('Invalid soft prompt checkpoint identity')
        identity['soft_prompt'] = copy.deepcopy(prompt)
        identity['soft_prompt'].pop('training_contract', None)
    return identity

def _load_joint96_checkpoint(model, payload, *, initialize_soft_prompt=False, initialize_independent_current=False):
    incoming_identity = _joint96_identity(payload.get('joint96_metadata'))
    expected_identity = _joint96_identity(joint96_checkpoint_metadata(model))
    if initialize_independent_current:
        if 'current_conditioning' in incoming_identity or expected_identity.get('current_conditioning') != 'independent_single_frame_v1' or payload['joint96_metadata'].get('soft_prompt', {}).get('training_contract'):
            raise ValueError('Invalid independent-current pretraining transfer')
        expected_identity.pop('current_conditioning')
    incoming_attention = incoming_identity['action_video_attention']
    expected_attention = expected_identity['action_video_attention']
    if incoming_attention != expected_attention:
        raise ValueError(f'Joint96 action_video_attention mismatch: checkpoint={incoming_attention!r}, model={expected_attention!r}')
    prompt_key = 'mixtures.action.embodiment_prompt.embedding.weight'
    if initialize_soft_prompt:
        if 'soft_prompt' in incoming_identity or 'soft_prompt' not in expected_identity:
            raise ValueError('Prompt migration only accepts an old model without prompts')
        expected_identity.pop('soft_prompt')
    from wam.utils.runtime_contract import semantic_contract
    if semantic_contract(incoming_identity) != semantic_contract(expected_identity):
        raise ValueError('Joint96 encoder/channel/action/camera identity mismatch')
    modules = (('mot', model.mot), ('proprio_encoder', model.proprio_encoder), ('action_proprio_encoder', model.action_proprio_encoder))
    for name, module in modules:
        incoming = payload.get(name)
        if module is None:
            if incoming is not None:
                raise ValueError(f'Joint96 unexpected module: {name}')
            continue
        expected = module.state_dict()
        allowed_missing = {prompt_key} if initialize_soft_prompt and name == 'mot' else set()
        if not isinstance(incoming, dict) or set(incoming) != set(expected) - allowed_missing or (not allowed_missing.issubset(expected)):
            raise ValueError(f'Joint96 strict state keys mismatch: {name}')
        for key, tensor in incoming.items():
            if not torch.is_tensor(tensor) or tensor.shape != expected[key].shape:
                raise ValueError(f'Joint96 strict tensor shape mismatch: {name}.{key}')
    for name, module in modules:
        if module is not None:
            state = payload[name]
            if initialize_soft_prompt and name == 'mot':
                state = {**state, prompt_key: module.state_dict()[prompt_key]}
            module.load_state_dict(state, strict=True)
    return payload

def load_joint96_soft_prompt_initialization(model, path, *, expected_step=None, frame_ablation=None):
    """Explicit old-policy to prompt-policy transfer before AdamW/ZeRO creation."""
    path = Path(str(path)).expanduser()
    if not path.is_absolute() or not path.is_file():
        raise ValueError('Soft prompt initialization requires an absolute weights file')
    config_path = path.parents[2] / 'config.yaml'
    source_config = OmegaConf.load(config_path)
    if not OmegaConf.select(source_config, 'local_robot_joint96.enabled', default=False):
        raise ValueError('Soft prompt initialization requires a local Robot Joint96 parent run')
    if OmegaConf.select(source_config, 'data.frame_ablation') != frame_ablation:
        raise ValueError("Soft prompt migration must preserve the parent's A/B treatment")
    payload = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
    expected_step = payload.get('step')  # legacy argument ignored; record actual progress
    _load_joint96_checkpoint(model, payload, initialize_soft_prompt=True)
    return {'checkpoint': str(path.resolve()), 'pretrain_step': expected_step, 'new_global_step': 0, 'weights_only': True, 'new_parameters': ['mot.mixtures.action.embodiment_prompt.embedding.weight'], 'parent_config_sha256': joint96_file_sha256(config_path), 'frame_ablation': frame_ablation, 'soft_prompt': joint96_checkpoint_metadata(model)['soft_prompt']}

def load_joint96_initialization(model, path, *, expected_step=None):
    """Strict weights-only transfer, called before AdamW/ZeRO master creation."""
    path = Path(str(path)).expanduser()
    if not path.is_absolute() or not path.is_file():
        raise ValueError('Joint96 initialization requires an explicit absolute weights file')
    if not bool(getattr(model, 'joint96_three_view', False)):
        raise ValueError('Joint96 initialization requires a Joint96 model')
    payload = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
    payload.setdefault('step', None)
    conditioning = getattr(getattr(model, 'joint_visual_encoder', None), 'current_conditioning', 'legacy_clip_prefix')
    migrate_current = conditioning == 'independent_single_frame_v1'
    _load_joint96_checkpoint(model, payload, initialize_independent_current=migrate_current)
    model.model_paths['posttrain_initialization'] = {'path': str(path.resolve()), 'step': payload['step']}
    if migrate_current:
        model.model_paths['posttrain_initialization']['current_conditioning_transfer'] = {'from': 'legacy_clip_prefix', 'to': conditioning, 'weights_unchanged': True}
    return {'checkpoint': str(path.resolve()), 'pretrain_step': payload['step'], 'weights_only': True, 'new_global_step': 0, 'tensor_counts': {name: len(payload[name]) for name in ('mot', 'proprio_encoder', 'action_proprio_encoder') if name in payload}, 'joint96_identity': _joint96_identity(payload['joint96_metadata'])}

def save_wam_checkpoint(model, path, step=None):
    payload = {'mot': model.mot.state_dict(), 'step': step, 'torch_dtype': str(model.torch_dtype)}
    if model.proprio_encoder is not None:
        payload['proprio_encoder'] = model.proprio_encoder.state_dict()
    if model.action_proprio_encoder is not None:
        payload['action_proprio_encoder'] = model.action_proprio_encoder.state_dict()
    if model.understanding is not None:
        payload['understanding'] = model.understanding.adapter_state_dict()
    if model.track_bridge is not None:
        payload['track_bridge'] = model.track_bridge.state_dict()
    metadata = joint96_checkpoint_metadata(model)
    if metadata is not None:
        payload['joint96_metadata'] = metadata
    torch.save(payload, path)

def format_skipped_state_preview(skipped: list[tuple[str, tuple[int, ...], tuple[int, ...]]]) -> str:
    preview = [f'{key}: ckpt{ckpt_shape} != model{model_shape}' for key, ckpt_shape, model_shape in skipped[:8]]
    if len(skipped) > len(preview):
        preview.append(f'... +{len(skipped) - len(preview)} more')
    return '; '.join(preview)

def load_compatible_state_dict(module: nn.Module, checkpoint_state: dict[str, torch.Tensor], module_name: str, *, skip_all_on_shape_mismatch: bool=False) -> dict[str, int]:
    model_state = module.state_dict()
    compatible = {}
    skipped_shape: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    skipped_type: list[str] = []
    unexpected: list[str] = []
    for key, value in checkpoint_state.items():
        target_key = key
        if target_key not in model_state:
            if '.base.' in key:
                alt_key = key.replace('.base.', '.')
                if alt_key in model_state:
                    target_key = alt_key
            elif key.endswith('.weight'):
                alt_key = f"{key[:-len('.weight')]}.base.weight"
                if alt_key in model_state:
                    target_key = alt_key
            elif key.endswith('.bias'):
                alt_key = f"{key[:-len('.bias')]}.base.bias"
                if alt_key in model_state:
                    target_key = alt_key
        if target_key not in model_state:
            unexpected.append(key)
            continue
        target = model_state[target_key]
        if not torch.is_tensor(value) or not torch.is_tensor(target):
            skipped_type.append(key)
            continue
        if tuple(value.shape) != tuple(target.shape):
            skipped_shape.append((target_key, tuple(value.shape), tuple(target.shape)))
            continue
        compatible[target_key] = value
    if skip_all_on_shape_mismatch and skipped_shape:
        logger.warning('Skipped all %s checkpoint tensors because this module has incompatible shapes: %s', module_name, format_skipped_state_preview(skipped_shape))
        return {'compatible': 0, 'missing': len(model_state), 'skipped_shape': len(skipped_shape), 'skipped_type': len(skipped_type), 'unexpected': len(unexpected)}
    incompat = module.load_state_dict(compatible, strict=False)
    missing = list(incompat.missing_keys)
    logger.info('Loaded %s checkpoint tensors: compatible=%d/%d missing=%d skipped_shape=%d unexpected=%d', module_name, len(compatible), len(checkpoint_state), len(missing), len(skipped_shape), len(unexpected))
    if skipped_shape:
        logger.warning('Skipped %s checkpoint tensors with incompatible shapes: %s', module_name, format_skipped_state_preview(skipped_shape))
    if skipped_type:
        logger.warning('Skipped %d non-tensor %s checkpoint entries: %s', len(skipped_type), module_name, ', '.join(skipped_type[:8]))
    if unexpected:
        suffix = '' if len(unexpected) <= 8 else f', ... +{len(unexpected) - 8} more'
        logger.warning('Ignored %d unexpected %s checkpoint tensors: %s%s', len(unexpected), module_name, ', '.join(unexpected[:8]), suffix)
    return {'compatible': len(compatible), 'missing': len(missing), 'skipped_shape': len(skipped_shape), 'skipped_type': len(skipped_type), 'unexpected': len(unexpected)}

def load_wam_checkpoint(model, path):
    payload = torch.load(path, map_location='cpu')
    if bool(getattr(model, 'joint96_three_view', False)):
        return _load_joint96_checkpoint(model, payload)
    if model.track_bridge is not None and 'track_bridge' in payload:
        state = {key: value for key, value in payload['track_bridge'].items() if not key.startswith('gate.')}
        model.track_bridge.load_state_dict(state, strict=True)
    mot_state = payload.get('mot')
    if not isinstance(mot_state, dict):
        raise ValueError(f'Checkpoint has no valid `mot` state: {path}')
    load_compatible_state_dict(model.mot, mot_state, 'mot')
    optional_modules = (('proprio_encoder', model.proprio_encoder), ('action_proprio_encoder', model.action_proprio_encoder))
    for name, module in optional_modules:
        state = payload.get(name)
        if module is not None and isinstance(state, dict):
            load_compatible_state_dict(module, state, name, skip_all_on_shape_mismatch=True)
    understanding_state = payload.get('understanding')
    if model.understanding is not None and isinstance(understanding_state, dict):
        model.understanding.load_adapter_state_dict(understanding_state, strict=False)
        logger.info('Loaded understanding adapter checkpoint.')
    return payload
