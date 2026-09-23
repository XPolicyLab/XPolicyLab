"""CPU regression checks for the real adapter/session, with only weights mocked."""
import importlib
import sys
import types

import numpy as np
import pytest

from XPolicyLab.policy.EchoPolicy.model import Model


@pytest.fixture
def model(monkeypatch, tmp_path):
    for name in ('openpi', 'openpi.policies', 'openpi.policies.policy_config',
                 'openpi.shared', 'openpi.shared.normalize', 'openpi.training',
                 'openpi.training.config'):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    pm = importlib.import_module('XPolicyLab.policy.Pi_05.model')
    batches = []

    class Policy:
        def infer(self, obs):
            batches.append(obs)
            return {'actions': np.repeat(obs['state'][:, None, :], 50, axis=1)}

    def init(self, cfg):
        self.policy = self.model = Policy()
        self.robot_action_dim_info = {'arm_dim': [6, 6], 'ee_dim': [1, 1]}

    monkeypatch.setattr(pm.Model, '__init__', init)
    from XPolicyLab.utils import process_data
    monkeypatch.setattr(process_data, 'get_robot_action_dim_info',
                        lambda _: {'arm_dim': [6, 6], 'ee_dim': [1, 1]})
    monkeypatch.delenv('VLM_API_KEY', raising=False)
    monkeypatch.setenv('ECHO_ALLOW_PASSTHROUGH', '1')
    model = Model({'env_cfg_type': 'arx_x5', 'model_path': str(tmp_path),
                   'echo_vla_image_noise_std': 0, 'echo_vla_joint_noise_std': 0})
    yield model, batches
    model.close()


def obs(i, value=0):
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    return {'env_idx': i, 'instruction': 'observe', 'state': {
        'left_arm_joint_state': np.full(6, value), 'left_ee_joint_state': np.zeros(1),
        'right_arm_joint_state': np.full(6, value), 'right_ee_joint_state': np.zeros(1),
    }, 'vision': {cam: {'color': image} for cam in
                  ['cam_head', 'cam_left_wrist', 'cam_right_wrist']}}


def test_updates_do_not_infer_and_active_order_is_preserved(model):
    m, batches = model
    for _ in range(10):
        m.update_obs_batch([obs(7, 0.1), obs(3, 0.2)])
    assert not batches
    actions = m.get_action_batch([3, 7])
    assert len(batches) == 1 and batches[0]['state'].shape == (160, 14)
    assert [len(a) for a in actions] == [10, 10]
    np.testing.assert_allclose(actions[0][0]['left_arm_joint_state'], 0.2)
    np.testing.assert_allclose(actions[1][0]['left_arm_joint_state'], 0.1)
    m.update_obs_batch([obs(7)])
    assert len(m.get_action_batch()) == 1
    assert len(batches) == 2
    assert m.get_action_batch([]) == []
    m.reset()
    with pytest.raises(RuntimeError, match='update_obs'):
        m.get_action()
    m.update_obs(obs(9))
    assert len(m.get_action()) == 10


def test_worker_error_propagates_and_reset_recovers(model):
    m, _ = model
    infer = m._pi05.policy.infer
    m._pi05.policy.infer = lambda _: {'actions': np.full((160, 50, 14), np.nan)}
    m.update_obs(obs(0))
    with pytest.raises(RuntimeError, match='invalid batched actions'):
        m.get_action()
    m.reset()
    m._pi05.policy.infer = infer
    m.update_obs(obs(0))
    assert len(m.get_action()) == 10


def test_candidates_over_limit_are_split(model):
    m, batches = model
    canonical, _ = m._adapter.to_canonical(obs(0))
    actions = m._infer_candidates([canonical] * 161)
    assert len(actions) == 161
    assert [b['state'].shape[0] for b in batches] == [160, 160]


def test_missing_key_fails_before_loading_weights(monkeypatch):
    monkeypatch.delenv('VLM_API_KEY', raising=False)
    monkeypatch.delenv('ECHO_ALLOW_PASSTHROUGH', raising=False)
    with pytest.raises(RuntimeError, match='VLM_API_KEY'):
        Model({})
