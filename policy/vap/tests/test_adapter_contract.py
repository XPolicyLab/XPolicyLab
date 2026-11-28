"""CPU contract tests with a fake backend; these are not model-quality rollouts."""
import asyncio
import os
import threading
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
from XPolicyLab.policy.vap.model import Model
from XPolicyLab.policy.vap.server import ExclusiveServer
from XPolicyLab.policy.vap.deploy import (
    eval_one_episode, STATE_DIMS, CAMERA_NAMES, _configure_transport_timeouts,
)
from XPolicyLab.client_server.ws.model_server import PolicyServerConfig
from XPolicyLab.client_server.ws.model_client import WsModelClient
from XPolicyLab.client_server.ws.protocol.client import PolicyEvalClientConfig
from XPolicyLab.utils.process_data import encode_image_bit


class Backend:
    def __init__(self):
        self.owner = threading.get_ident()
        self.resets = 0
        self.windows = []
    def reset(self):
        assert threading.get_ident() == self.owner
        self.resets += 1
        self.window = 0
    def update_obs(self, obs):
        assert threading.get_ident() == self.owner
        assert obs['_vap']['window_index'] == self.window
        assert obs['vision']['cam_head']['color'].shape == (32, 40, 3)
        pixel = obs['vision']['cam_head']['color'][0, 0].astype(int)
        assert pixel[0] > 180 and pixel[2] < 40, pixel
        if self.window:
            assert obs['_vap']['executed_video']['cam_head'].shape == (16, 32, 40, 3)
            assert obs['_vap']['executed_state']['left_ee_pose'].shape == (32, 7)
        else:
            assert 'executed_video' not in obs['_vap']
        self.windows.append(self.window)
        self.window += 1
    def get_action(self):
        assert threading.get_ident() == self.owner
        action = {side+'_ee_pose': np.array([0, 0, 1, 1, 0, 0, 0], dtype=np.float32) for side in ('left','right')}
        action.update({side+'_ee_joint_state': np.ones(1,dtype=np.float32) for side in ('left','right')})
        return [action.copy() for _ in range(32)]


class TestModel(Model):
    __test__ = False
    def _initialize(self):
        return Backend()


class Env:
    task_name = 'stack_bowls'
    eval_seed = 0
    episode_num = 0
    def __init__(self, encoded):
        self.encoded = encoded
        self.step = 0
    def get_obs(self):
        image = np.full((32, 40, 3), [230, 40, 10],dtype=np.uint8)
        if self.encoded:
            image = encode_image_bit(image)
        return {'vision': {key: {'color':image} for key in CAMERA_NAMES},
                'state': {key: np.ones(dim) for key,dim in STATE_DIMS.items()}}
    def take_action(self,action):self.step += 1
    def is_episode_end(self):return self.step >= 321


def test_timeout_configuration_precedes_unchanged_rpc_sequence():
    config = PolicyEvalClientConfig(url='ws://test.invalid', evaluation_id='test')
    client = SimpleNamespace(_client=SimpleNamespace(config=config), calls=[])

    def call(*, func_name, obs=None):
        assert config.request_timeout_s == 600.0
        assert config.max_connect_seconds == 1800.0
        client.calls.append((func_name, obs))
        if func_name == 'get_action':
            return [{} for _ in range(32)]

    client.call = call
    env = Env(encoded=False)
    with patch.dict(os.environ, {'EVAL_ENV_TYPE': 'debug'}):
        eval_one_episode(env, client)
    names = [name for name, _ in client.calls]
    assert names == ['reset'] + ['update_obs', 'get_action'] * 11
    assert all(obs is None for name, obs in client.calls if name != 'update_obs')
    assert client.calls[1][1]['_vap']['window_index'] == 0
    assert config.ws_ping_interval_s == 20.0
    assert config.ws_ping_timeout_s == 20.0


def test_timeout_does_not_replay_stateful_action():
    client = SimpleNamespace(_client=SimpleNamespace(config=PolicyEvalClientConfig(
        url='ws://test.invalid', evaluation_id='test')), calls=[])

    def call(*, func_name, obs=None):
        client.calls.append(func_name)
        if func_name == 'get_action':
            raise TimeoutError('test timeout')
        return []

    client.call = call
    with patch.dict(os.environ, {'EVAL_ENV_TYPE': 'debug'}):
        with pytest.raises(TimeoutError, match='test timeout'):
            eval_one_episode(Env(encoded=False), client)
    assert client.calls == ['reset', 'update_obs', 'get_action']


def test_debug_client_without_protocol_config_is_unchanged():
    client = SimpleNamespace(calls=[])
    client.call = lambda *, func_name, obs=None: client.calls.append(func_name)
    _configure_transport_timeouts(client)
    client.call(func_name='reset')
    assert client.calls == ['reset']


@pytest.mark.parametrize('value', [True, -1, float('nan'), '600'])
def test_invalid_timeout_fails_before_any_rpc(value, monkeypatch, tmp_path):
    from XPolicyLab.policy.vap import deploy

    (tmp_path / 'deploy.yml').write_text(
        f'request_timeout_s: 600.0\nmax_connect_seconds: {value!r}\n')
    config = PolicyEvalClientConfig(url='ws://test.invalid', evaluation_id='test')
    client = SimpleNamespace(_client=SimpleNamespace(config=config), calls=[])
    client.call = lambda *, func_name, obs=None: client.calls.append(func_name)
    monkeypatch.setattr(deploy, '__file__', str(tmp_path / 'deploy.py'))
    with pytest.raises(ValueError, match='max_connect_seconds'):
        _configure_transport_timeouts(client)
    assert config.request_timeout_s == 120.0
    assert config.max_connect_seconds == 900.0
    assert client.calls == []


def test_official_transport_rgb_histories_reset_thread_owner():
    model=TestModel({'env_cfg_type':'arx_x5','action_type':'ee','eval_batch':False})
    loop=asyncio.new_event_loop()
    worker=threading.Thread(target=loop.run_forever,daemon=True)
    worker.start()
    server=ExclusiveServer(model,PolicyServerConfig(host='127.0.0.1',port=0))
    asyncio.run_coroutine_threadsafe(server.start(),loop).result(10)
    port=server._server.sockets[0].getsockname()[1]
    client=WsModelClient(url='ws://127.0.0.1:'+str(port),evaluation_id='contract-test',trial_id='contract-test',request_timeout_s=10)
    try:
        with patch.dict(os.environ,{'EVAL_ENV_TYPE':'debug'}):
            for encoded in (False,True):
                env=Env(encoded)
                eval_one_episode(env,client)
                assert env.step == 321
        assert model.backend.resets==2
        assert model.backend.windows==list(range(11))*2
        assert model.backend.owner != threading.get_ident()
    finally:
        client.close()
        asyncio.run_coroutine_threadsafe(server.stop(),loop).result(10)
        loop.call_soon_threadsafe(loop.stop)
        worker.join(10)
        loop.close()
        model.executor.shutdown()
