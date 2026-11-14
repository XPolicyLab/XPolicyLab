from collections.abc import Mapping
import torch
from .dual_cooperative_action import ACTION_TOPOLOGY, JG_INDICES, EEF_INDICES
_TRAINING_STATE_KEYS = frozenset(('optimizer', 'optimizers', 'scheduler', 'schedulers', 'rng', 'rng_state', 'random_states', 'sampler', 'coverage_sampler'))

def dual_action_contract():
    return dict(version=1, topology=ACTION_TOPOLOGY, external_action_dim=90, joint_gripper_indices=list(JG_INDICES), eef_indices=list(EEF_INDICES), token_order=['action_joint', 'action_eef'], independent_parameters=True, action_horizon=32, rope='each_tower_0_to_H_minus_1', attention='outer_independent_middle_cooperative_visual_reads_no_action', noise='single_A90_draw_shared_diffusion_time', loss='A90_valid_element_mean_per_sample_then_global_valid_sample_mean')

def _checkpoint_modules(model):
    names = tuple(dict.fromkeys(('mot', 'proprio_encoder', *model._dino_ckpt_keys(), *model._mode_ckpt_keys())))
    return {name: getattr(model, name) for name in names if getattr(model, name, None) is not None}

def validate_dual_checkpoint_contract(payload, model=None):
    if payload.get('action_topology') != ACTION_TOPOLOGY or payload.get('dual_action_contract') != dual_action_contract():
        raise ValueError('Strict dual resume rejected: action topology/channel contract mismatch; use original single A90 step150000 only as a weights-only warm start')
    if model is not None:
        for key, module in _checkpoint_modules(model).items():
            saved = payload.get(key)
            current = module.state_dict()
            if not isinstance(saved, Mapping) or set(saved) != set(current):
                raise ValueError(f'Strict dual resume requires complete parameter keys for {key}')
            for name, target in current.items():
                source = saved[name]
                if not torch.is_tensor(source) or source.shape != target.shape:
                    raise ValueError(f'Strict dual resume parameter shape mismatch: {key}.{name}')

def _action_mapping(name):
    for prefix, indices in (('mixtures.action.', JG_INDICES), ('mixtures.action_eef.', EEF_INDICES)):
        if name.startswith(prefix):
            suffix = name[len(prefix):]
            source_name = 'mixtures.action.' + suffix
            axis = 1 if suffix == 'action_encoder.weight' else 0 if suffix in ('head.weight', 'head.bias') else None
            return (source_name, axis, indices if axis is not None else None)
    return (name, None, None)

def _conditioning_mapping(name):
    for branch, indices in (('joint', JG_INDICES), ('eef', EEF_INDICES)):
        if name == f'action_valid_{branch}.weight':
            return ('action_valid.weight', 1, indices)
        if name == f'time_projection.action_{branch}.weight':
            return ('time_projection.action.weight', None, None)
    return (name, None, None)

def migrate_single_action_payload(model, payload, checkpoint='<in-memory>', *, expected_source_step=170000):
    if not isinstance(payload, Mapping):
        raise ValueError('Warm-start checkpoint must be a mapping')
    if payload.get('action_topology') not in (None, 'single', 'single_action_v1') or 'dual_action_contract' in payload:
        raise ValueError('Dual warm initialization requires the original single ActionDiT checkpoint; dual checkpoints require strict resume')
    if payload.get('step') != expected_source_step:
        raise ValueError(f"Dual warm initialization requires original step{expected_source_step}, got {payload.get('step')!r}")
    source_mot = payload.get('mot', {})
    source_encoder = source_mot.get('mixtures.action.action_encoder.weight')
    if not torch.is_tensor(source_encoder) or source_encoder.ndim != 2 or source_encoder.shape[1] != 90:
        raise ValueError('Dual warm initialization requires an A90 action encoder')
    if any((name.startswith('mixtures.action_eef.') for name in source_mot)):
        raise ValueError('Source checkpoint already contains a second action tower')
    model._validate_checkpoint_payload(payload, strict_shape=False)
    mapped = {key: value for key, value in payload.items() if key not in _TRAINING_STATE_KEYS}
    mappings, allowed_missing = ([], [])
    modules = _checkpoint_modules(model)
    excluded_perception = {}
    if getattr(getattr(model, 'perception_config', None), 'enabled', None) is False:
        assert 'perception' not in modules and (not hasattr(model, 'perception'))
        auxiliary = mapped.pop('perception', {})
        if not isinstance(auxiliary, Mapping) or not all((torch.is_tensor(v) for v in auxiliary.values())):
            raise ValueError('Invalid disabled perception source state')
        excluded_perception = {name: dict(shape=list(value.shape), dtype=str(value.dtype), numel=value.numel()) for name, value in auxiliary.items()}
    for key, module in modules.items():
        current = module.state_dict()
        saved = payload.get(key)
        if saved is None and key == 'perception' and getattr(model, 'initialize_missing_perception_from_rgb', False):
            allowed_missing.extend((f'{key}.{name}' for name in current))
            continue
        if not isinstance(saved, Mapping):
            raise ValueError(f'Dual migration requires checkpoint module {key}')
        result, consumed = ({}, set())
        for name, target in current.items():
            source_name, axis, indices = _action_mapping(name) if key == 'mot' else _conditioning_mapping(name) if key == 'multisource_conditioning' else (name, None, None)
            source = saved.get(source_name)
            if not torch.is_tensor(source):
                raise ValueError(f'Dual migration missing source tensor: {key}.{source_name}')
            consumed.add(source_name)
            value = source
            if axis is not None:
                if source.ndim <= axis or source.shape[axis] != 90:
                    raise ValueError(f'Dual migration requires A90 source dimension: {key}.{source_name}')
                value = source.index_select(axis, torch.tensor(indices, device=source.device))
            if value.shape != target.shape:
                raise ValueError(f'Dual migration shape mismatch for {key}.{name}: {tuple(value.shape)} != {tuple(target.shape)}')
            result[name] = value
            mappings.append(dict(source=f'{key}.{source_name}', destination=f'{key}.{name}', operation='semantic_condition_rows' if key == 'multisource_conditioning' and name == 'conditions.weight' else 'copy' if axis is None else 'index_select', axis=axis, indices=list(indices) if indices is not None else None, source_shape=list(source.shape), target_shape=list(target.shape), source_dtype=str(source.dtype), target_dtype=str(target.dtype), numel=target.numel()))
        unexpected = set(saved) - consumed
        if unexpected:
            raise ValueError(f'Dual migration refuses unexpected {key} tensors: {sorted(unexpected)}')
        mapped[key] = result
    audit = dict(version=1, checkpoint=str(checkpoint), source_step=payload['step'], action_topology=ACTION_TOPOLOGY, contract=dual_action_contract(), mappings=mappings, permitted_missing_perception=allowed_missing, excluded_disabled_perception_tensors=excluded_perception, ignored_training_state_keys=sorted(set(payload) & _TRAINING_STATE_KEYS), weights_only=True, fresh_optimizer_scheduler_step=True, after_cast_exact=False, independent_action_storage=False)
    return (mapped, audit)

def _verify_after_cast(model, mapped, audit):
    modules = _checkpoint_modules(model)
    state = {key: module.state_dict() for key, module in modules.items()}
    for item in audit['mappings']:
        key, name = item['destination'].split('.', 1)
        if item['operation'] == 'semantic_condition_rows':
            for old_id, new_id in model.multisource_initialization_report.get('matched_condition_rows', []):
                actual = state[key][name][new_id]
                expected = mapped[key][name][old_id].to(actual)
                if not torch.equal(actual, expected):
                    raise RuntimeError('Dual migration condition row failed after-cast equality')
            item['after_cast_exact'] = True
            item['verified_scope'] = 'matched_semantic_rows_only'
            continue
        actual = state[key][name]
        expected = mapped[key][name].to(actual)
        if not torch.equal(actual, expected):
            raise RuntimeError(f"Dual migration after-cast equality failed: {item['destination']}")
        item['after_cast_exact'] = True
    joint = model.mot.mixtures['action']
    eef = model.mot.mixtures['action_eef']
    first = {parameter.untyped_storage().data_ptr() for parameter in joint.parameters()}
    second = {parameter.untyped_storage().data_ptr() for parameter in eef.parameters()}
    if first & second:
        raise RuntimeError('Dual action towers unexpectedly share parameter storage')
    audit['after_cast_exact'] = True
    audit['independent_action_storage'] = True
    audit['verified_tensor_count'] = len(audit['mappings'])

def load_dual_stage_payload(model, payload, checkpoint='<in-memory>', *, expected_source_step=170000):
    from .stage_initialization import load_stage_payload
    mapped, audit = migrate_single_action_payload(model, payload, checkpoint, expected_source_step=expected_source_step)
    load_stage_payload(model, mapped, checkpoint)
    report = model.multisource_initialization_report
    unexpected_skips = [item for item in report['skipped'] if item['key'] not in audit['permitted_missing_perception'] and item['reason'] != 'no_matching_condition_semantics']
    if unexpected_skips or report['unexpected']:
        raise RuntimeError(f"Unexpected skipped tensors in dual migration: {unexpected_skips}; {report['unexpected']}")
    _verify_after_cast(model, mapped, audit)
    report['dual_action_migration'] = audit
    report['source_step'] = payload['step']
    report['fresh_optimizer_scheduler_step'] = True
    return mapped

def load_dual_stage_checkpoint(model, path):
    payload = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    return load_dual_stage_payload(model, payload, path)
