"""Fresh Action-LoRA on a frozen, SHA-bound recovery parent.

Load the official base, then call load_frozen_parent(), then
install_recovery_action_lora(). Training uses the existing recovery-aware
predictor; the ordinary success133 predictor must not be substituted.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import torch

from recoverwam.recovery import lora
from recoverwam.recovery.bridge import MODES, RecoveryBridge, load_bridge


FORMAT = 'recoverwam-frozen-recovery-action-lora-v1'
QUERY_AUTOGRAD = 'frozen-recovery-action-query-autograd-v1'
REGISTERED_PARAMETERS = 3_932_160
RANK = 8
ALPHA = 8.0


class ActionQueryAutogradBridge(RecoveryBridge):
    """Same recovery values, with the frozen residual's input Jacobian retained."""

    def forward(self, action_tokens, state, *, mode='auto'):
        if mode not in MODES:
            raise ValueError(f'Unknown recovery mode: {mode}')
        if mode == 'off':
            return action_tokens
        probability = state['gate_logits'].sigmoid()
        if probability.shape != (action_tokens.shape[0],):
            raise ValueError('Recovery state batch mismatch')
        if mode == 'auto':
            gate = probability >= self.config['threshold']
            if not gate.any():
                return action_tokens
        elif mode == 'on':
            gate = torch.ones_like(probability)
        else:
            gate = probability
        with torch.autocast(device_type=action_tokens.device.type, enabled=False):
            # Only this detach differs from the frozen parent's forward: LoRA
            # changes upstream action tokens, so retain their residual Jacobian.
            query = self.action_proj(self.action_norm(action_tokens.float()))
            update, _ = self.read_recovery(query, state['tokens'], state['tokens'], need_weights=False)
            update = self.residual(update) * gate[:, None, None]
        changed = action_tokens + update.to(action_tokens.dtype)
        return torch.where((gate != 0)[:, None, None], changed, action_tokens)


def _model(policy):
    return getattr(policy, 'model', policy)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_frozen_parent(policy, checkpoint, *, expected_sha256, expected_base_identity):
    """Load an explicitly selected r5 parent before any Action-LoRA installation."""
    model = _model(policy)
    if hasattr(model, 'recovery_bridge') or hasattr(model, '_action_lora_config'):
        raise ValueError('Load the frozen recovery parent before installing fresh Action-LoRA')
    if not expected_base_identity or file_sha256(checkpoint) != expected_sha256:
        raise ValueError('Recovery parent checkpoint SHA256 or base identity is missing/mismatched')
    metadata = load_bridge(checkpoint, policy, expected_base_identity=expected_base_identity)
    if file_sha256(checkpoint) != expected_sha256:
        raise ValueError('Recovery parent checkpoint changed while loading')
    model.requires_grad_(False)
    model.eval()
    parent = dict(checkpoint_sha256=expected_sha256, base_identity=expected_base_identity,
                  bridge_config=copy.deepcopy(model.recovery_bridge.config), metadata=copy.deepcopy(metadata))
    model._frozen_recovery_parent_identity = parent
    return copy.deepcopy(parent)


def _targets(model):
    result = {}
    for index, block in enumerate(model.action_expert.blocks):
        for kind in ('q', 'v'):
            layer = getattr(block.self_attn, kind)
            if not isinstance(layer, (torch.nn.Linear, lora.LoRALinear)):
                raise ValueError('Registered Action-LoRA targets must be q/v Linear layers')
            base = layer.base if isinstance(layer, lora.LoRALinear) else layer
            result[f'blocks.{index}.self_attn.{kind}'] = (base.in_features, base.out_features)
    return result


def install_recovery_action_lora(policy, *, lora_seed, gradient_checkpointing=True,
                                 expected_trainable_parameters=REGISTERED_PARAMETERS):
    """Install rank-8/alpha-8 q/v LoRA; preserve the caller's RNG stream.

    lora_seed is the schedule's initialization RNG seed, not its user-facing
    experiment seed. The initializer is reseeded AFTER load_bridge, which consumes random
    numbers. Equal LoRA seeds therefore reproduce the ordinary group's A
    matrices regardless of parent recipe or recovery construction history.
    The parameter-count override is only for explicitly dimensioned CPU tests.
    """
    model = _model(policy)
    if not hasattr(model, '_frozen_recovery_parent_identity') or type(model.recovery_bridge) is not RecoveryBridge:
        raise ValueError('A registered frozen recovery parent must be loaded first')
    if list(lora.lora_modules(policy)) or hasattr(model, '_action_lora_config'):
        raise ValueError('Fresh Action-LoRA cannot be installed over an existing adapter')
    targets = _targets(model)
    count = sum(RANK * (inputs + outputs) for inputs, outputs in targets.values())
    if type(expected_trainable_parameters) is not int or count != expected_trainable_parameters:
        raise ValueError(f'Action-LoRA parameter count {count} differs from registered {expected_trainable_parameters}')
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise ValueError('Parent/base must have no accumulated gradients before composition')
    device = torch.device(model.device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    # Reuse every loaded module/Parameter object; do not initialize or recopy the parent.
    model.recovery_bridge.__class__ = ActionQueryAutogradBridge
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(int(lora_seed))
        if devices:
            with torch.cuda.device(devices[0]):
                torch.cuda.manual_seed(int(lora_seed))
        config = lora.install_action_lora(policy, rank=RANK, alpha=ALPHA,
                                          gradient_checkpointing=gradient_checkpointing)
    model._recovery_action_lora_config = dict(format=FORMAT, query_autograd=QUERY_AUTOGRAD,
        lora_seed=int(lora_seed), expected_trainable_parameters=count,
        parent=copy.deepcopy(model._frozen_recovery_parent_identity), action_lora=copy.deepcopy(config))
    assert_trainable_contract(policy)
    return copy.deepcopy(model._recovery_action_lora_config)


def assert_trainable_contract(policy):
    """Enforce an explicit A/B-only whitelist; frozen tensors cannot acquire grads."""
    model = _model(policy)
    config = model._recovery_action_lora_config
    if type(model.recovery_bridge) is not ActionQueryAutogradBridge:
        raise RuntimeError('The recovery-aware training bridge is missing')
    if (model.recovery_bridge.config != config['parent']['bridge_config']
            or model._frozen_recovery_parent_identity != config['parent']):
        raise RuntimeError('Frozen recovery parent identity/config changed')
    expected = {}
    for name, module in lora.lora_modules(policy):
        if module.rank != RANK or module.alpha != ALPHA:
            raise RuntimeError('Unregistered Action-LoRA rank/alpha')
        for suffix in ('lora_A', 'lora_B'):
            expected['action_expert.' + name + '.' + suffix] = getattr(module, suffix)
    expected_names = {'action_expert.' + name + '.' + suffix
                      for name in _targets(model) for suffix in ('lora_A', 'lora_B')}
    if set(expected) != expected_names or not expected:
        raise RuntimeError('Action-LoRA target whitelist differs from q/v A/B')
    actual = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    if set(actual) != expected_names or any(actual[name] is not value for name, value in expected.items()):
        raise RuntimeError('Only registered action q/v LoRA A/B tensors may train')
    if sum(value.numel() for value in actual.values()) != config['expected_trainable_parameters']:
        raise RuntimeError('Trainable parameter count changed')
    allowed = {id(value) for value in expected.values()}
    if any(parameter.grad is not None for parameter in model.parameters() if id(parameter) not in allowed):
        raise RuntimeError('Frozen base or recovery parent accumulated gradients')
    lora.assert_only_lora_trainable(policy)
    return tuple(sorted(expected_names))


def prepare_condition(policy, prepared_inputs=None, **kwargs):
    """Accept the native RecoverWAM format only; no silent producer relabeling."""
    assert_trainable_contract(policy)
    return lora.prepare_condition(policy, prepared_inputs, **kwargs)


def predict_action_velocity(policy, condition, noisy_actions, timesteps):
    assert_trainable_contract(policy)
    return lora.predict_action_velocity(policy, condition, noisy_actions, timesteps)


def action_flow_loss(policy, condition, normalized_actions, action_valid_mask, **kwargs):
    assert_trainable_contract(policy)
    return lora.action_flow_loss(policy, condition, normalized_actions, action_valid_mask, **kwargs)


def infer_from_condition(policy, condition, **kwargs):
    assert_trainable_contract(policy)
    return lora.infer_from_condition(policy, condition, **kwargs)


def lora_parameters(policy):
    assert_trainable_contract(policy)
    return lora.lora_parameters(policy)


def save_adapter(path, policy, *, metadata=None):
    """Persist fresh A/B tensors and parent SHA; never embed or overwrite the parent."""
    whitelist = assert_trainable_contract(policy)
    payload = dict(format=FORMAT, config=copy.deepcopy(_model(policy)._recovery_action_lora_config),
                   trainable_whitelist=list(whitelist), state_dict=lora.adapter_state_dict(policy),
                   metadata=copy.deepcopy(metadata or {}))
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.pending')
    with temporary.open('xb') as stream:
        torch.save(payload, stream)
    if path.exists():
        raise FileExistsError(path)
    temporary.replace(path)


def load_adapter(path, policy, *, expected_parent_sha256, expected_base_identity):
    """Reload A/B only onto the same independently loaded frozen recovery parent."""
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('format') != FORMAT:
        raise ValueError('Unregistered composed adapter format')
    config = payload['config']
    model = _model(policy)
    parent = getattr(model, '_frozen_recovery_parent_identity', None)
    if (parent is None or config['parent'] != parent
            or parent['checkpoint_sha256'] != expected_parent_sha256
            or parent['base_identity'] != expected_base_identity
            or config.get('query_autograd') != QUERY_AUTOGRAD):
        raise ValueError('Adapter does not belong to this frozen recovery parent/base')
    target_shapes = {}
    for name, (inputs, outputs) in _targets(model).items():
        target_shapes[name + '.lora_A'] = (RANK, inputs)
        target_shapes[name + '.lora_B'] = (outputs, RANK)
    saved = payload['state_dict']
    if saved.keys() != target_shapes.keys():
        raise ValueError('Composed adapter tensor whitelist mismatch')
    for name, shape in target_shapes.items():
        tensor = saved[name]
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape or tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
            raise ValueError('Invalid composed adapter tensor: ' + name)
    if not hasattr(model, '_recovery_action_lora_config'):
        install_recovery_action_lora(policy, lora_seed=config['lora_seed'],
            gradient_checkpointing=config['action_lora']['gradient_checkpointing'],
            expected_trainable_parameters=config['expected_trainable_parameters'])
    if model._recovery_action_lora_config != config:
        raise ValueError('Installed composed adapter configuration differs')
    if list(assert_trainable_contract(policy)) != payload['trainable_whitelist']:
        raise ValueError('Saved composed adapter trainable whitelist differs')
    with torch.no_grad():
        for name, module in lora.lora_modules(policy):
            for suffix in ('lora_A', 'lora_B'):
                getattr(module, suffix).copy_(saved[name + '.' + suffix])
    return copy.deepcopy(payload.get('metadata', {}))
