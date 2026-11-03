"""Image-only lag regressions using real adapter observation updates on CPU."""
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

POLICY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POLICY.parents[2]))
from XPolicyLab.policy.Era0_Preview.obs_lag import ImageObservationLag, observation_lag, legacy_render_setting
from XPolicyLab.policy.Era0_Preview.native_protocol import default_tasks


def observation(env_idx, value):
    image = np.full((4, 5, 3), value, dtype=np.uint8)
    return dict(env_idx=env_idx,
        vision={name: dict(color=image.copy()) for name in ['cam_head','cam_left_wrist','cam_right_wrist']},
        state=dict(left_ee_pose=[value,0,0,1,0,0,0], right_ee_pose=[0,0,0,1,0,0,0],
                   left_ee_joint_state=[value / 100], right_ee_joint_state=[.4]),
        instruction=f'current {value}')


class ObservationLagTests(unittest.TestCase):
    def adapter(self, lag, cadence='chunk'):
        from XPolicyLab.policy.Era0_Preview.model import Model
        obj = Model.__new__(Model)
        obj.policy = SimpleNamespace(reset=lambda: None)
        obj.gripper_close_hold = SimpleNamespace(reset=lambda: None)
        obj._image_lag = ImageObservationLag(lag)
        obj.use_video_encoder = True
        obj.history_cadence = cadence
        obj._obs_by_env = {};obj._top_by_env = {};obj._left_by_env = {};obj._right_by_env = {}
        return obj

    def test_delay_is_per_update_with_current_state_and_all_three_cameras(self):
        obj = self.adapter(1)
        for value, expected in [(10,10),(20,10),(30,20)]:
            obs = observation(7,value)
            obj.update_obs_batch([obs])
            batch = obj._obs_by_env[7]
            self.assertEqual(batch['task'], [f'current {value}'])
            self.assertEqual(batch['observation.state'][0,0], value)
            self.assertAlmostEqual(float(batch['observation.state'][0,7]), value/100, places=6)
            for camera in ['top_head','hand_left','hand_right']:
                self.assertTrue(np.all(batch['raw_images.'+camera] == expected))
                np.testing.assert_allclose(batch['observation.images.'+camera],expected/255,atol=1e-7)
            # Mutating the caller's frame cannot corrupt any delayed snapshot.
            for item in obs['vision'].values():item['color'][:] = 255
        obj._append_history_frames(7, obj._obs_by_env[7])
        self.assertTrue(np.all(obj._top_by_env[7][-1] == 20))

    def test_batch_compaction_keeps_environment_queues_independent(self):
        obj = self.adapter(1,'step')
        obj.update_obs_batch([observation(5,10),observation(9,50)])
        obj.update_obs_batch([observation(9,60),observation(5,20)])
        obj.update_obs_batch([observation(9,70)])
        self.assertTrue(np.all(obj._obs_by_env[9]['raw_images.top_head'] == 60))
        self.assertTrue(np.all(obj._obs_by_env[5]['raw_images.top_head'] == 10))
        self.assertEqual(len(obj._top_by_env[9]),3)
        self.assertEqual(len(obj._top_by_env[5]),2)
        obj.reset()
        self.assertFalse(obj._image_lag._by_env)
        obj.update_obs(observation(9,90))
        self.assertTrue(np.all(obj._obs_by_env[9]['raw_images.top_head'] == 90))

    def test_multi_update_warmup_and_disabled_path(self):
        obj = self.adapter(2)
        for value, expected in [(10,10),(20,10),(30,10),(40,20)]:
            obj.update_obs(observation(0,value))
            self.assertTrue(np.all(obj._obs_by_env[0]['raw_images.top_head'] == expected))
        batch = {'raw_images.top_head': np.zeros((1,1,3),np.uint8)}
        disabled = ImageObservationLag(0)
        self.assertIs(disabled.apply(0,batch),batch)
        self.assertFalse(disabled._by_env)

    def test_settings_and_double_delay_rejection(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(observation_lag(),1)
            self.assertFalse(legacy_render_setting(0))
            self.assertFalse(legacy_render_setting(1))
        with patch.dict(os.environ, {'HY_OBS_LAG':'0', 'ROBODOJO_EVAL_LEGACY_RENDER':'1'}):
            self.assertEqual(observation_lag(),0)
            self.assertTrue(legacy_render_setting(observation_lag()))
        for value in [-1,1.5,True,'x','1.0']:
            with self.assertRaises(ValueError):observation_lag(value)
        with patch.dict(os.environ, {'ROBODOJO_EVAL_LEGACY_RENDER':'1'}):
            with self.assertRaisesRegex(ValueError,'cannot be combined'):legacy_render_setting(1)

    def test_default_task_inventory_follows_sdk(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);script=root/'scripts/internal/task_inventory.py'
            script.parent.mkdir(parents=True)
            script.write_text("import sys\nassert sys.argv[1:] == ['--only-runnable', '--dimension', 'all']\nprint('task_a\\ntask_b')\n")
            self.assertEqual(default_tasks(root),['task_a','task_b'])
            script.write_text(script.read_text().replace('task_a\\ntask_b','task_a\\ntask_b\\ntask_added'))
            self.assertEqual(default_tasks(root),['task_a','task_b','task_added'])


if __name__ == '__main__':unittest.main()
