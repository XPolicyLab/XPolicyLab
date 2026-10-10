"""CPU contract tests; the neural inference engine is explicitly mocked."""
import asyncio
import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import yaml

adapter = importlib.import_module("XPolicyLab.policy.Liber_0_lite.model")
from XPolicyLab.utils.process_data import decode_obs_images, encode_image_bit, pack_robot_state


class Engine:
    action_horizon = 32
    robotwin_camera_layout = "four_grid_288x384"
    model = SimpleNamespace(dit=SimpleNamespace(action_dim=14), action_expert_stability_fp32=True)

    def __init__(self):
        self.calls = []
        self.batch_calls = []
        self.actions = np.tile(np.arange(14, dtype=np.float32), (32, 1))

    def infer_action_chunk(self, observation, instruction, request_seed):
        self.calls.append((observation, instruction, request_seed))
        return self.actions.copy()

    def infer_action_batch(self, observations, instructions, request_seeds):
        self.batch_calls.append((observations, instructions, request_seeds))
        return np.stack([self.infer_action_chunk(obs, instruction, seed)
                         for obs, instruction, seed in zip(observations, instructions, request_seeds)])


@pytest.fixture
def policy(monkeypatch, tmp_path):
    config = yaml.safe_load((Path(__file__).parent / "deploy.yml").read_text())
    config.update(device="cpu", action_type="joint", env_cfg_type="arx_x5", seed=0,
                  ckpt_name=str(tmp_path), bench_name="RoboDojo", task_name="unused")
    engine = Engine()
    monkeypatch.setattr(adapter, "_load_policy", lambda cfg, root: engine)
    monkeypatch.setattr(adapter, "get_robot_action_dim_info", lambda env: {"arm_dim": [6, 6], "ee_dim": [1, 1]})
    return adapter.Model(config)


def observation(value=10):
    return {
        "instruction": "Place the object in the box.",
        "vision": {name: {"color": np.full((24, 32, 3), value + i, dtype=np.uint8)}
                   for i, name in enumerate(("cam_head", "cam_left_wrist", "cam_right_wrist"))},
        "state": {"left_arm_joint_state": np.arange(6), "left_ee_joint_state": np.array([6]),
                  "right_arm_joint_state": np.arange(7, 13), "right_ee_joint_state": np.array([13])},
    }


def test_cue_is_copied_and_fixed_until_reset(policy):
    first = observation(10)
    policy.update_obs(first)
    first["vision"]["cam_head"]["color"][:] = 80
    policy.update_obs(observation(30))
    assert np.all(policy.observation["visual_cue_rgb"] == 10)
    assert np.all(policy.observation["head_camera_rgb"] == 30)
    policy.reset()
    policy.update_obs(observation(40))
    assert np.all(policy.observation["visual_cue_rgb"] == 40)


def test_joint_order_horizon_and_seed_sequence(policy):
    obs = observation()
    obs.update(task_name="not used", layout_id=123, _external_request_seed=999)
    policy.update_obs(obs)
    actions = policy.get_action()
    assert len(actions) == 16
    assert np.array_equal(policy.observation["joint_action_vector"], np.arange(14))
    for item in actions:
        assert np.array_equal(pack_robot_state({"state": item}, "joint", policy.robot_action_dim_info), np.arange(14))
    policy.get_action()
    assert [call[2] for call in policy.policy.calls] == [0, 1]
    policy.reset()
    policy.update_obs(observation())
    policy.get_action()
    assert policy.policy.calls[-1][2] == 0


def test_readonly_rgb_observations_are_supported(policy):
    obs = observation()
    for view in obs["vision"].values():
        view["color"].flags.writeable = False
    policy.update_obs(obs)
    assert policy.episode_cue.flags.writeable
    assert len(policy.get_action()) == 16


def test_server_decoded_encoded_rgb_has_no_extra_channel_swap(policy):
    obs = observation()
    colors = ((210, 20, 40), (20, 210, 40), (20, 40, 210))
    for i, (view, color) in enumerate(zip(obs["vision"].values(), colors)):
        image = np.full((24, 32, 3), color, dtype=np.uint8)
        encoded = encode_image_bit(image)
        view["color"] = np.asarray(bytearray(encoded), dtype=np.uint8) if i == 0 else encoded if i == 1 else image
    decoded = decode_obs_images(obs)
    policy.update_obs(decoded)
    for key, channel in (("head_camera_rgb", 0), ("left_camera_rgb", 1), ("right_camera_rgb", 2)):
        assert policy.observation[key].mean(axis=(0, 1)).argmax() == channel
    assert np.array_equal(policy.episode_cue, decoded["vision"]["cam_head"]["color"])


def test_instruction_list_and_missing_instruction(policy):
    obs = observation()
    obs["instructions"] = [obs.pop("instruction")]
    policy.update_obs(obs)
    assert policy.instruction == "Place the object in the box."
    obs.pop("instructions")
    with pytest.raises(ValueError, match="instruction"):
        policy.update_obs(obs)


def test_no_actions_before_first_observation(policy):
    with pytest.raises(RuntimeError, match="update_obs"):
        policy.get_action()


@pytest.mark.parametrize("kind", ["shape", "nan"])
def test_invalid_predictions_surface(policy, kind):
    policy.update_obs(observation())
    if kind == "shape":
        policy.policy.actions = np.zeros((31, 14), dtype=np.float32)
    else:
        policy.policy.actions[0, 0] = np.nan
    with pytest.raises(ValueError, match="Invalid predicted"):
        policy.get_action()


def test_batch_identity_cue_seed_subset_and_reset(policy):
    first, second = observation(10), observation(50)
    first['env_idx'], second['env_idx'] = 7, 2
    policy.update_obs_batch([first, second])
    first['vision']['cam_head']['color'][:] = 99
    assert len(policy.get_action_batch([2, 7])) == 2
    observations, _, seeds = policy.policy.batch_calls[-1]
    assert seeds == [0, 0]
    assert np.all(observations[0]['visual_cue_rgb'] == 50)
    assert np.all(observations[1]['visual_cue_rgb'] == 10)
    second['vision']['cam_head']['color'][:] = 80
    policy.update_obs_batch([second])
    assert len(policy.get_action_batch()) == 1
    assert policy.policy.batch_calls[-1][2] == [1]
    assert policy._batch[7]['replan_index'] == 1
    policy.reset_envs([2])
    policy.update_obs_batch([second, first])
    policy.get_action_batch()
    observations, _, seeds = policy.policy.batch_calls[-1]
    assert seeds == [0, 1]
    assert np.all(observations[0]['visual_cue_rgb'] == 80)
    assert np.all(observations[1]['visual_cue_rgb'] == 10)
    policy.reset()
    assert not policy._batch and not policy._batch_order


def test_batch_requires_explicit_unique_ids(policy):
    with pytest.raises(KeyError):
        policy.update_obs_batch([observation()])
    obs = dict(observation(), env_idx=0)
    with pytest.raises(ValueError, match='unique'):
        policy.update_obs_batch([obs, obs])
    with pytest.raises(ValueError, match='update_obs_batch'):
        policy.get_action_batch()
    policy.update_obs_batch([obs])
    with pytest.raises(KeyError):
        policy.get_action_batch([1])


@pytest.mark.parametrize('kind', ['shape', 'nan'])
def test_invalid_batch_does_not_advance_counters(policy, kind):
    policy.update_obs_batch([dict(observation(), env_idx=2), dict(observation(20), env_idx=7)])
    if kind == 'shape':
        policy.policy.actions = np.zeros((31, 14), dtype=np.float32)
    else:
        policy.policy.actions[0, 0] = np.nan
    with pytest.raises(ValueError, match='Invalid predicted batch'):
        policy.get_action_batch()
    assert all(state['replan_index'] == 0 for state in policy._batch.values())


def test_loader_uses_bundled_runtime(monkeypatch, tmp_path):
    assets = tmp_path / 'assets'
    assets.mkdir()
    constructor = Mock(return_value=object())
    monkeypatch.setattr(adapter, 'Policy', constructor)
    config = dict(model_assets_path=str(assets), weights_file='model.pt',
                  device='cpu', num_inference_steps=10, seed=0)
    adapter._load_policy(config, tmp_path)
    constructor.assert_called_once_with(checkpoint_root=tmp_path, assets=assets,
        device='cpu', weights_file='model.pt', num_inference_steps=10)


@pytest.mark.parametrize("encoded", [False, True])
def test_official_websocket_and_deploy_loop_with_mock_engine(policy, encoded):
    from client_server.ws.model_client import WsModelClient
    from client_server.ws.model_server import PolicyServer, PolicyServerConfig
    eval_one_episode = importlib.import_module("XPolicyLab.policy.Liber_0_lite.deploy").eval_one_episode

    class Environment:
        step = 0

        def is_episode_end(self):
            return self.step >= 18

        def get_obs(self):
            obs = observation(10 + self.step)
            if encoded:
                for view in obs["vision"].values():
                    view["color"] = encode_image_bit(view["color"])
            return obs

        def take_action(self, action):
            assert len(action) == 4
            self.step += 1

    async def exercise():
        server = PolicyServer(policy, PolicyServerConfig(host="127.0.0.1", port=0))
        await server.start()
        port = server._server.sockets[0].getsockname()[1]

        def run_client():
            client = WsModelClient(url=f"ws://127.0.0.1:{port}", evaluation_id="mock-contract",
                                   trial_id="episode", max_connect_attempts=1)
            env = Environment()
            try:
                eval_one_episode(env, client)
            finally:
                client.close()
            assert env.step == 18

        try:
            await asyncio.to_thread(run_client)
        finally:
            await server.stop()

    asyncio.run(exercise())
    assert len(policy.policy.calls) == 2
    assert policy.replan_index == 2
    assert np.all(policy.episode_cue == 10)


@pytest.mark.parametrize('encoded', [False, True])
def test_official_batch_loop_handles_finished_environments(policy, encoded):
    from client_server.ws.model_client import WsModelClient
    from client_server.ws.model_server import PolicyServer, PolicyServerConfig
    deploy = importlib.import_module('XPolicyLab.policy.Liber_0_lite.deploy')

    class Environment:
        def __init__(self):
            self.steps = {7: 0, 2: 0}
            self.limits = {7: 5, 2: 18}

        def get_running_env_idx_list(self):
            return [key for key in self.steps if self.steps[key] < self.limits[key]]

        def is_episode_end(self):
            return not self.get_running_env_idx_list()

        def get_obs_batch(self, env_ids):
            result = []
            for env_id in env_ids:
                obs = dict(observation(10 * env_id + self.steps[env_id]), env_idx=env_id)
                if encoded:
                    for view in obs['vision'].values():
                        view['color'] = encode_image_bit(view['color'])
                result.append(obs)
            return result

        def take_action_batch(self, actions, env_ids):
            assert len(actions) == len(env_ids)
            for env_id, action in zip(env_ids, actions):
                assert len(action) == 4
                self.steps[env_id] += 1

    async def exercise():
        server = PolicyServer(policy, PolicyServerConfig(host='127.0.0.1', port=0))
        await server.start()
        port = server._server.sockets[0].getsockname()[1]

        def run_client():
            client = WsModelClient(url=f'ws://127.0.0.1:{port}', evaluation_id='mock-batch',
                                   trial_id='episode', max_connect_attempts=1)
            env = Environment()
            try:
                deploy.eval_one_episode_batch(env, client)
                assert env.steps == env.limits
                assert policy._batch[7]['replan_index'] == 1
                assert policy._batch[2]['replan_index'] == 2
                assert np.all(policy._batch[7]['cue'] == 70)
                assert np.all(policy._batch[2]['cue'] == 20)
                client.call(func_name='reset_envs', obs=[7])
                assert 7 not in policy._batch and 2 in policy._batch
            finally:
                client.close()

        try:
            await asyncio.to_thread(run_client)
        finally:
            await server.stop()

    asyncio.run(exercise())
    assert [len(call[0]) for call in policy.policy.batch_calls] == [2, 1]
