from wam.utils.runtime_contract import semantic_contract
'Opt-in RoboCasa recipe; strict full-weight transfer is owned by WAMTrainer.'
import copy
from pathlib import Path
from omegaconf import OmegaConf
import torch
from wam.datasets.robocasa_delta import CONTRACT
from wam.datasets.robocasa_abs47 import CONTRACT as ABS47_CONTRACT
from wam.datasets.robocasa_delta80_slot18 import CONTRACT as DELTA80_SLOT18_CONTRACT
from wam.datasets.robocasa_visual_cache import file_hash

def activate_robocasa_slot(source, *, action_space='abs47'):
    """Version the registry while assigning exactly one previously unused row."""
    from wam.embodiment_registry import EmbodimentRegistry
    EmbodimentRegistry.validate(source)
    if action_space not in ('abs47', 'delta80'):
        raise ValueError('Unsupported RoboCasa prompt identity')
    if source['num_embodiments'] != 28 or source['slot_status'][18] != 'reserved' or source['embodiments'][18] != 'reserved_18':
        raise ValueError('RoboCasa requires the unused Robot37 slot 18')
    result = copy.deepcopy(source)
    result['version'] += f'.robocasa_{action_space}_slot18.v1'
    result['embodiments'][18] = f'robocasa_gr1_{action_space}'
    result['slot_status'][18] = 'assigned'
    result['assigned_count'] += 1
    result['profile_to_embodiment_id'][f'robocasa_gr1_{action_space}'] = 18
    EmbodimentRegistry.validate(result)
    return result

def validate_abs47_identity(incoming, expected):
    from wam.embodiment_registry import EmbodimentRegistry
    migrated = copy.deepcopy(incoming)
    if (migrated.get('action_dim'), migrated.get('proprio_dim')) != (80, 80):
        raise ValueError('absolute47 transfer requires the canonical80 Robot37 parent')
    migrated.update(action_dim=47, proprio_dim=47)
    prompt = migrated['soft_prompt']
    prompt['registry'] = activate_robocasa_slot(prompt['registry'])
    prompt['registry_sha256'] = EmbodimentRegistry.content_hash(prompt['registry'])
    if semantic_contract(migrated) != semantic_contract(expected):
        raise ValueError('absolute47 transfer changed an unrelated encoder/model/prompt identity')

def load_robocasa_abs47_initialization(model, path, *, expected_step=None):
    """Explicit 80->47 weights-only migration before AdamW/ZeRO construction."""
    from wam.training.checkpoint import _joint96_identity, joint96_checkpoint_metadata, _load_joint96_checkpoint
    path = Path(path)
    if not path.is_absolute() or not path.is_file():
        raise ValueError('absolute47 initialization requires an absolute checkpoint file')
    payload = torch.load(path, mmap=True, map_location='cpu', weights_only=True)
    expected_step = payload.get('step')
    metadata = joint96_checkpoint_metadata(model)
    validate_abs47_identity(_joint96_identity(payload.get('joint96_metadata')), _joint96_identity(metadata))
    prompt_key = 'mixtures.action.embodiment_prompt.embedding.weight'
    if bool(torch.count_nonzero(payload['mot'][prompt_key][18])):
        raise ValueError('Parent prompt slot18 is not empty; refusing to repurpose a trained slot')
    adapted = {**payload, 'mot': dict(payload['mot']), 'joint96_metadata': metadata}
    new_keys = []
    fresh_mot = model.mot.state_dict()
    prompt = payload['mot'][prompt_key].clone()
    prompt[18] = fresh_mot[prompt_key][18].to(prompt)
    adapted['mot'][prompt_key] = prompt
    for prefix in ('mixtures.action.action_encoder.', 'mixtures.action.head.'):
        for key in (prefix + 'weight', prefix + 'bias'):
            if key not in payload['mot'] or key not in fresh_mot:
                raise ValueError(f'Missing action IO tensor: {key}')
            adapted['mot'][key] = fresh_mot[key]
            new_keys.append('mot.' + key)
    for name in ('proprio_encoder', 'action_proprio_encoder'):
        module = getattr(model, name)
        if module is not None:
            if not isinstance(payload.get(name), dict) or set(payload[name]) != set(module.state_dict()):
                raise ValueError(f'Parent proprio module mismatch: {name}')
            adapted[name] = module.state_dict()
            new_keys.extend((name + '.' + key for key in adapted[name]))
    _load_joint96_checkpoint(model, adapted)
    model.model_paths['posttrain_initialization'] = {'path': str(path), 'step': expected_step}
    return {'checkpoint': str(path), 'pretrain_step': expected_step, 'new_global_step': 0, 'weights_only': True, 'new_parameters': new_keys, 'new_prompt_slot': 18, 'new_prompt_initialization': 'fresh_normal_std0.02_after_empty_parent_check', 'action_space': ABS47_CONTRACT}

def validate_delta80_slot18_identity(incoming, expected):
    from wam.embodiment_registry import EmbodimentRegistry
    migrated = copy.deepcopy(incoming)
    if (migrated.get('action_dim'), migrated.get('proprio_dim')) != (80, 80):
        raise ValueError('delta80 slot18 requires an unchanged canonical80 parent')
    prompt = migrated['soft_prompt']
    prompt['registry'] = activate_robocasa_slot(prompt['registry'], action_space='delta80')
    prompt['registry_sha256'] = EmbodimentRegistry.content_hash(prompt['registry'])
    if semantic_contract(migrated) != semantic_contract(expected):
        raise ValueError('delta80 slot18 transfer changed an unrelated model identity')

def load_robocasa_delta80_slot18_initialization(model, path, *, expected_step=None):
    """Strictly inherit every weight except the newly activated prompt row."""
    from wam.training.checkpoint import _joint96_identity, joint96_checkpoint_metadata, _load_joint96_checkpoint
    path = Path(path)
    if not path.is_absolute() or not path.is_file():
        raise ValueError('delta80 slot18 initialization requires an absolute checkpoint file')
    payload = torch.load(path, mmap=True, map_location='cpu', weights_only=True)
    expected_step = payload.get('step')
    metadata = joint96_checkpoint_metadata(model)
    validate_delta80_slot18_identity(_joint96_identity(payload.get('joint96_metadata')), _joint96_identity(metadata))
    key = 'mixtures.action.embodiment_prompt.embedding.weight'
    if bool(torch.count_nonzero(payload['mot'][key][18])):
        raise ValueError('Parent prompt slot18 is not empty; refusing to overwrite it')
    prompt = payload['mot'][key].clone()
    prompt[18] = model.mot.state_dict()[key][18].to(prompt)
    adapted = {**payload, 'mot': {**payload['mot'], key: prompt}, 'joint96_metadata': metadata}
    _load_joint96_checkpoint(model, adapted)
    model.model_paths['posttrain_initialization'] = {'path': str(path), 'step': expected_step}
    return {'checkpoint': str(path), 'pretrain_step': expected_step, 'new_global_step': 0, 'weights_only': True, 'new_parameters': [], 'new_prompt_slot': 18, 'new_prompt_initialization': 'fresh_normal_std0.02_after_empty_parent_check', 'action_space': DELTA80_SLOT18_CONTRACT}

def validate_robocasa_config(cfg):
    get = lambda key, default=None: OmegaConf.select(cfg, key, default=default)
    absolute = get('joint96_posttrain.robocasa.frame_contract') == ABS47_CONTRACT['version']
    slot18 = get('joint96_posttrain.robocasa.frame_contract') == DELTA80_SLOT18_CONTRACT['version']
    contract = ABS47_CONTRACT if absolute else DELTA80_SLOT18_CONTRACT if slot18 else CONTRACT
    dim = 47 if absolute else 80
    required = {'joint96_posttrain.enabled': True, 'joint96_posttrain.cached_visuals': True, 'joint96_posttrain.robocasa.frame_contract': contract['version'], 'joint96_posttrain.robocasa.embodiment_id': contract['embodiment_id'], 'local_robot_joint96.enabled': False, 'local_robot_softprompt.enabled': True, 'local_robot_softprompt.init_checkpoint': None, 'resume': None, 'model.action_dit_config.num_embodiments': 28, 'model.action_dit_config.soft_prompt_tokens': 8, 'model.action_dit_config.action_dim': dim, 'model.proprio_dim': dim, 'model._target_': 'wam.training.robocasa_posttrain.create_robocasa_wam'}
    for key, expected in required.items():
        if get(key) != expected:
            raise ValueError(f'RoboCasa requires {key}={expected!r}')
    for split in ('train', 'val'):
        if get(f'data.{split}.domain') != 'robocasa' or get(f'data.{split}.split') != split:
            raise ValueError('RoboCasa needs explicit train/val readers')
        for key in ('package_root', 'cache_root', 'pretrained_norm_stats', 'allow_smoke', 'smoke_selection'):
            if get(f'data.{split}.{key}') != get(f'data.train.{key}'):
                raise ValueError(f'RoboCasa splits must share {key}')

def validate_cached_initialization(payload, plan, expected_step=None):
    from wam.training.checkpoint import _joint96_identity
    identity = _joint96_identity(payload.get('joint96_metadata'))
    for key, value in plan['encoder_identity'].items():
        if 'sha256' in key or key in ('dinov3_source_commit', 'stvae_step', 'teacher_step', 'pretrain_step'):
            continue
        if identity.get(key) != value:
            raise ValueError(f'Initialization/cache visual identity mismatch: {key}')
    return identity

def wrap_robocasa_prompt(model, train, val, cfg, registry):
    absolute = cfg.joint96_posttrain.robocasa.frame_contract == ABS47_CONTRACT['version']
    slot18 = cfg.joint96_posttrain.robocasa.frame_contract == DELTA80_SLOT18_CONTRACT['version']
    contract = ABS47_CONTRACT if absolute else DELTA80_SLOT18_CONTRACT if slot18 else CONTRACT
    slot, name = (18, 'robocasa_gr1_abs47') if absolute else (18, 'robocasa_gr1_delta80') if slot18 else (13, 'galaxea_r1pro')
    if registry.content['embodiments'][slot] != name or slot not in registry.assigned_ids:
        raise ValueError('RoboCasa prompt registry assignment mismatch')
    if val is None or train is val:
        raise ValueError('Require separate train/val datasets from the same package')
    for ds in (train, val):
        ds.embodiment_registry = registry
        if ds.package['contract'] != contract:
            raise ValueError('Training recipe/package action space mismatch')
        require_remarks = absolute or slot18 or cfg.joint96_posttrain.robocasa.get('require_episode_remarks', False)
        if require_remarks and ds.package.get('prompt_source') != 'episodes.jsonl:remarks':
            raise ValueError('This RoboCasa recipe requires corrected episode instruction labels')
    plan = train.visual.plan
    init = Path(cfg.init_checkpoint).resolve()
    from wam.training.checkpoint import _joint96_identity, joint96_checkpoint_metadata
    identity = _joint96_identity(joint96_checkpoint_metadata(model))
    payload = torch.load(init, map_location='cpu', mmap=True, weights_only=True)
    step = payload.get('step')
    incoming = validate_cached_initialization(payload, plan, step)
    if absolute:
        validate_abs47_identity(incoming, identity)
    elif slot18:
        validate_delta80_slot18_identity(incoming, identity)
    elif semantic_contract(incoming) != semantic_contract(identity):
        raise ValueError('RoboCasa initialization/model identity mismatch')
    del payload
    model.soft_prompt_training_contract = {'domain': 'robocasa', 'embodiment_id': slot, 'embodiment_name': name, 'frame_contract': copy.deepcopy(contract), 'package_sha256': train.package['sha256'], 'normalization_sha256': train.package['norm_sha256'], 'visual_plan_sha256': plan['sha256'], 'visual_attention_backend': 'math_sdpa', 'pretrain_step': step, 'pretrain_checkpoint': str(init), 'text_manifest_sha256': file_hash(train.root / 'text/COMPLETE.json')}
    if getattr(train, 'smoke_selection_sha256', None):
        model.soft_prompt_training_contract['smoke_selection_sha256'] = train.smoke_selection_sha256
    return (train, val)

def install_robocasa_runtime(model):
    """Match PPU VAE scaling without changing the existing LIBERO runtime."""
    move = model.set_runtime_device
    mean = model.vae.mean.detach().float().clone()
    std = model.vae.std.detach().float().clone()
    if model.vae.mean.dtype != torch.float32 or model.vae.std.dtype != torch.float32:
        raise ValueError('RoboCasa requires original FP32 VAE normalization constants')

    def move_with_fp32_scale(device):
        move(device)
        model.vae.mean = mean.to(device=device)
        model.vae.std = std.to(device=device)
        model.vae.scale = [model.vae.mean, 1.0 / model.vae.std]
    model.set_runtime_device = move_with_fp32_scale

def create_robocasa_wam(**kwargs):
    from wam.utils.torch24_compat import install_safe_serialization_compat
    from wam.model.factory import create_wam
    install_safe_serialization_compat()
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    loss = kwargs.get('loss', {})
    if any((float(loss.get(name, default)) != default for name, default in (('lambda_video', 1.0), ('lambda_action', 1.0), ('lambda_vae', 0.5), ('lambda_stvae', 0.5)))):
        raise ValueError('This RoboCasa port preserves action + 0.5 VAE + 0.5 STVAE loss')
    model = create_wam(**kwargs)
    install_robocasa_runtime(model)
    encode = model.joint_visual_encoder.encode

    def encode_math(*args, **kw):
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            return encode(*args, **kw)
    model.joint_visual_encoder.encode = encode_math
    return model
