"""CPU transport regression with an explicit neural test double, never real weights."""
import asyncio
import copy
from pathlib import Path
import sys
import unittest

import numpy as np

POLICY_DIR = Path(__file__).resolve().parents[1]
WORKSPACE = POLICY_DIR.parents[2]
sys.path[:0] = [str(WORKSPACE), str(WORKSPACE / 'XPolicyLab')]

from XPolicyLab.policy.JAM.delivery import load_deployment
from XPolicyLab.policy.JAM.model import Model
from XPolicyLab.utils.process_data import encode_image_bit, get_robot_action_dim_info
from client_server.ws.model_client import WsModelClient
from client_server.ws.model_server import PolicyServer, PolicyServerConfig


class NeuralTestDouble:
    action_horizon = 32

    def predict(self, obs, instruction):
        assert instruction == 'cap the pen'
        assert obs['cam_high'].dtype == np.uint8
        assert obs['cam_high'].shape == (480, 640, 3)
        # JPEG may round by a few levels, but a channel swap must fail.
        np.testing.assert_allclose(obs['cam_high'][0, 0], [230, 40, 15], atol=3)
        return np.tile(obs['qpos'], (32, 1)), 0.0


def observation(env_idx=0):
    image = np.empty((480, 640, 3), dtype=np.uint8)
    image[:] = [230, 40, 15]
    return {
        'env_idx': env_idx,
        'instruction': 'cap the pen',
        'vision': {camera: {'color': image.copy()} for camera in
                   ('cam_head', 'cam_left_wrist', 'cam_right_wrist')},
        'state': {
            'left_arm_joint_state': np.arange(6, dtype=np.float32) + env_idx,
            'left_ee_joint_state': np.array([6], dtype=np.float32),
            'right_arm_joint_state': np.arange(7, 13, dtype=np.float32),
            'right_ee_joint_state': np.array([13], dtype=np.float32),
        },
    }


def transported(obs, encoded):
    obs = copy.deepcopy(obs)
    if encoded:
        # Exercise standard JPEG buffer, bytes, and already-decoded RGB together.
        obs['vision']['cam_head']['color'] = np.array(list(encode_image_bit(obs['vision']['cam_head']['color'])), dtype=np.uint8)
        obs['vision']['cam_left_wrist']['color'] = encode_image_bit(obs['vision']['cam_left_wrist']['color'])
    return obs


class TransportTests(unittest.IsolatedAsyncioTestCase):
    async def check_transport(self, encoded):
        for method in ('jam', 'fastwam'):
            with self.subTest(method=method, encoded=encoded):
                manifest = load_deployment(WORKSPACE / 'models' / method, verify_checkpoint=False)
                model = Model.__new__(Model)
                model.policy = NeuralTestDouble()
                model.dimensions = get_robot_action_dim_info('piper_x')
                model.replan_steps = manifest['replan_steps']
                model.reset()
                server = PolicyServer(model, PolicyServerConfig(host='127.0.0.1', port=0))
                await server.start()
                client = None
                try:
                    client = await asyncio.to_thread(WsModelClient, url=server.url,
                        evaluation_id='cpu-test-double', trial_id='cpu-interface-only',
                        connect_timeout_s=5, request_timeout_s=10, max_connect_attempts=1)
                    await asyncio.to_thread(client.call, 'reset')
                    await asyncio.to_thread(client.call, 'update_obs', transported(observation(), encoded))
                    actions = await asyncio.to_thread(client.call, 'get_action')
                    self.assertEqual(len(actions), 24)
                    self.assertEqual(set(actions[0]), set(observation()['state']))
                    np.testing.assert_array_equal(actions[0]['left_arm_joint_state'], np.arange(6))
                    obs = [transported(observation(idx), encoded) for idx in (9, 3)]
                    await asyncio.to_thread(client.call, 'update_obs_batch', obs)
                    batch = await asyncio.to_thread(client.call, 'get_action_batch', [3, 9])
                    self.assertEqual([len(chunk) for chunk in batch], [24, 24])
                    np.testing.assert_array_equal(batch[0][0]['left_arm_joint_state'], np.arange(6) + 3)
                    np.testing.assert_array_equal(batch[1][0]['left_arm_joint_state'], np.arange(6) + 9)
                    await asyncio.to_thread(client.call, 'reset')
                    self.assertIsNone(model.current)
                    self.assertEqual(model.batch, {})
                    self.assertNotIn('torch', sys.modules)
                finally:
                    if client is not None:
                        await asyncio.to_thread(client.close)
                    await server.stop()

    async def test_raw_rgb_and_batch(self):
        await self.check_transport(encoded=False)

    async def test_encoded_rgb_and_batch(self):
        await self.check_transport(encoded=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
