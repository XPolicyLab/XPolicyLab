PRETRAIN_FORMAT = 'liber0-pretrain-v1'
HEAD_KEYS = frozenset({'action_head.weight', 'action_head.bias'})
INTERFACE_KEYS = frozenset(f'{module}.{name}' for module in
                           ('action_encoder', 'proprio_encoder', 'action_head') for name in ('weight', 'bias'))
CONTRACT = dict(action_expert_stability_fp32=True, action_backbone='shared',
                target_image_mode='mosaic', semantic_image_mode='legacy_384',
                joint_layout='legacy', numerical_mode='legacy_bf16')


def load_weights(dit, payload, *, pretrained=False, assign=False):
    expected = dict(CONTRACT)
    if pretrained:
        if payload.get('format') != PRETRAIN_FORMAT:
            raise ValueError('Expected a head-free pretraining checkpoint.')
        expected['action_expert_stability_fp32'] = False
    for name, value in expected.items():
        if payload[name] != value:
            raise ValueError(f'Incompatible checkpoint: {name}')
    state = payload['dit']
    if pretrained:
        target = dit.state_dict()
        if set(target) - set(state) != HEAD_KEYS or set(state) - set(target):
            raise ValueError('Pretraining checkpoint must omit only the action output head.')
        if (state['action_encoder.weight'].shape[1] != 326
                or state['proprio_encoder.weight'].shape[1] != 346
                or target['action_encoder.weight'].shape[1] != 14
                or target['proprio_encoder.weight'].shape[1] != 14
                or target['action_head.weight'].shape[0] != 14):
            raise ValueError('Unsupported pretraining interface dimensions.')
        hidden = target['action_encoder.weight'].shape[0]
        for name, width in (('action_encoder', 326), ('proprio_encoder', 346)):
            if state[f'{name}.weight'].shape != (hidden, width) or state[f'{name}.bias'].shape != (hidden,):
                raise ValueError(f'Invalid source interface: {name}')
        state = {**state, **{key: target[key] for key in INTERFACE_KEYS}}
    return dit.load_state_dict(state, strict=True, assign=assign)
