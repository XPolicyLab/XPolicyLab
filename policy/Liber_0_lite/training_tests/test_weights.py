import copy

import pytest
import torch
from torch import nn

from XPolicyLab.policy.Liber_0_lite.liber0.weights import CONTRACT, HEAD_KEYS, INTERFACE_KEYS, PRETRAIN_FORMAT, load_weights


class Network(nn.Module):
    def __init__(self, action, state):
        super().__init__()
        self.body = nn.Linear(8, 8)
        self.action_encoder = nn.Linear(action, 8)
        self.proprio_encoder = nn.Linear(state, 8)
        self.action_head = nn.Linear(8, action)


def payload():
    source = Network(326, 346)
    return dict(format=PRETRAIN_FORMAT, **{**CONTRACT, 'action_expert_stability_fp32': False},
                dit={key: value for key, value in source.state_dict().items() if key not in HEAD_KEYS})


def test_pretraining_preserves_body_and_fresh_interfaces():
    value = payload()
    target = Network(14, 14)
    fresh = copy.deepcopy(target.state_dict())
    loaded = load_weights(target, value, pretrained=True)
    assert not loaded.missing_keys and not loaded.unexpected_keys
    for name, tensor in target.state_dict().items():
        assert torch.equal(tensor, fresh[name] if name in INTERFACE_KEYS else value['dit'][name])


@pytest.mark.parametrize('issue', ['missing_body', 'extra', 'head', 'shape', 'input_shape', 'format', 'precision'])
def test_bad_pretraining_checkpoint_rejected(issue):
    value = payload()
    if issue == 'missing_body': del value['dit']['body.bias']
    if issue == 'extra': value['dit']['extra'] = torch.zeros(1)
    if issue == 'head': value['dit']['action_head.bias'] = torch.zeros(326)
    if issue == 'shape': value['dit']['body.weight'] = torch.zeros(7, 8)
    if issue == 'input_shape': value['dit']['action_encoder.bias'] = torch.zeros(7)
    if issue == 'format': value['format'] = 'other'
    if issue == 'precision': value['action_expert_stability_fp32'] = True
    with pytest.raises((ValueError, RuntimeError)):
        load_weights(Network(14, 14), value, pretrained=True)


def test_regular_checkpoint_remains_strict():
    target = Network(14, 14)
    source = Network(14, 14)
    value = dict(dit=source.state_dict(), **CONTRACT)
    load_weights(target, value)
    assert all(torch.equal(a, b) for a, b in zip(source.parameters(), target.parameters()))
    del value['dit']['action_head.bias']
    with pytest.raises(RuntimeError): load_weights(target, value)
    with pytest.raises(ValueError): load_weights(target, payload())


def test_policy_routes_explicit_initialization(monkeypatch, tmp_path):
    import json
    import yaml
    import XPolicyLab.policy.Liber_0_lite.liber0.runtime as runtime
    config = dict(model={'_target_': 'unused', 'model_path': 'unused', 'source_path': 'unused'},
                  inference=dict(robotwin_camera_layout='four_grid_288x384', image_crop_scale=.95,
                                 action_horizon=32, processor=dict(norm_default_mode='z-score',
                                 use_stepwise_action_norm=False, val_transforms=[{'_target_':'ToTensor'},
                                 {'_target_':'torchvision.transforms.Resize','size':[240,320]}])))
    (tmp_path / 'config.yaml').write_text(yaml.safe_dump(config))
    entry = dict(default=dict(global_mean=[0]*14, global_std=[1]*14))
    (tmp_path / 'dataset_stats.json').write_text(json.dumps(dict(action=entry, state=entry)))
    calls = []
    monkeypatch.setattr(runtime, 'load_model', lambda *a, **kw: calls.append((a, kw)))
    pretrained = tmp_path / 'pretrain/model.pt'
    runtime.Policy(tmp_path, tmp_path, 'cpu', 'model.pt', 10, initialization_weights=pretrained)
    assert calls[-1][0][2] == pretrained and calls[-1][1]['pretrained'] is True
    runtime.Policy(tmp_path, tmp_path, 'cpu', 'model.pt', 10)
    assert calls[-1][0][2] == tmp_path / 'model.pt' and calls[-1][1]['pretrained'] is False
