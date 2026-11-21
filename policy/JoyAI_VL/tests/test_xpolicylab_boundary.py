"""CPU boundary checks with shared codecs; no model construction or GPU execution."""
from pathlib import Path
import sys
import types
import unittest

import numpy as np

# The package lives in the benchmark workspace, above the XPolicyLab checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.policy.JoyAI_VL.model import Model
from XPolicyLab.policy.JoyAI_VL.deploy import eval_one_episode
from XPolicyLab.utils.process_data import decode_obs_images, encode_image_bit


class Boundary(unittest.TestCase):
    def test_actual_model_contract(self):
        self.assertTrue(issubclass(Model, ModelTemplate))
        for method in ('__init__', 'reset', 'update_obs', 'update_obs_batch',
                       'get_action', 'get_action_batch'):
            self.assertTrue(callable(getattr(Model, method, None)), method)

    def test_shared_server_decode_reaches_model_as_rgb(self):
        rgb = np.zeros((12, 16, 3), dtype=np.uint8)
        rgb[:] = [220, 30, 10]
        rgb.setflags(write=False)
        encoded = encode_image_bit(rgb, quality=100)
        obs = {'instruction': 'synthetic task', 'vision': {
            'cam_head': {'color': rgb},
            'cam_left_wrist': {'color': encoded},
            'cam_right_wrist': {'color': memoryview(encoded)},
        }}
        # Use the same helper as the shared server, without opening a socket.
        decoded = decode_obs_images(obs)
        model = Model.__new__(Model)
        model.executor = types.SimpleNamespace(obs=None)
        model.update_obs(decoded)
        self.assertIs(model.executor.obs, decoded)
        self.assertIs(decoded['vision']['cam_head']['color'], rgb)
        for camera in decoded['vision'].values():
            color = camera['color']
            self.assertEqual(color.shape, rgb.shape)
            self.assertGreater(int(color[0, 0, 0]), 200)
            self.assertLess(int(color[0, 0, 2]), 20)
        model.update_obs_batch([decoded])
        self.assertIs(model.executor.obs, decoded)

    def test_wait_does_not_step_environment_and_verdict_is_forwarded(self):
        action = {'left_arm_joint_state': np.zeros(6), 'left_ee_joint_state': np.ones(1),
                  'right_arm_joint_state': np.zeros(6), 'right_ee_joint_state': np.ones(1)}

        class Environment:
            task_name = 'synthetic'
            step_lim = 100
            take_action_cnt = [0]
            env_seeds = [19321]
            success = [False]

            def is_episode_end(self):
                return self.take_action_cnt[0] == 1

            def get_obs(self):
                return {'instruction': 'synthetic task'}

            def take_action(self, actual):
                self.assert_action = actual
                self.take_action_cnt[0] += 1
                self.success[0] = True

        env = Environment()
        calls = []
        replies = iter([[{'hexashell': 'wait'}], [action]])

        def call(func_name, **kwargs):
            calls.append((func_name, kwargs))
            if func_name == 'get_action':
                self.assertEqual(env.take_action_cnt[0], 0)
                return next(replies)

        eval_one_episode(env, types.SimpleNamespace(call=call))
        self.assertEqual([name for name, _ in calls],
                         ['reset', 'update_obs', 'get_action', 'get_action', 'trial_end'])
        self.assertIs(env.assert_action, action)
        verdict = calls[-1][1]['obs']
        self.assertEqual(verdict['layout'], 19321)
        self.assertEqual(verdict['step'], 1)
        self.assertTrue(verdict['success'])
        self.assertEqual(verdict['progress'], 100.0)


if __name__ == '__main__':
    unittest.main()
