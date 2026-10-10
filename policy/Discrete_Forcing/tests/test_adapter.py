import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from XPolicyLab.policy.Discrete_Forcing import deploy, model, prepare_data
from XPolicyLab.utils.process_data import decode_obs_images, encode_image_bit


class SyntheticPolicy:
    def __init__(self):
        self.calls = []

    def predict_action(self, **kwargs):
        self.calls.append(kwargs)
        values = np.linspace(-1, 1, 14, dtype=np.float32)
        values[-2:] = [0.48, 0.5]
        output = np.tile(values, (len(kwargs["examples"]), 50, 1))
        return {"normalized_actions": output}


def observation(index=0):
    image = np.full((24, 32, 3), [20, 80, 160], dtype=np.uint8)
    image.flags.writeable = False
    return {
        "env_idx": index,
        "instruction": f"task {index}",
        "vision": {camera: {"color": image} for camera in
                   ("cam_head", "cam_left_wrist", "cam_right_wrist")},
    }


class AdapterTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.model_root = root / "source"
        marker = self.model_root / "starVLA/model/framework/QwenPILF_v3.py"
        marker.parent.mkdir(parents=True)
        marker.touch()
        self.run = root / "run"
        self.weight = self.run / "checkpoints/steps_60000_pytorch_model.pt"
        self.weight.parent.mkdir(parents=True)
        self.weight.touch()
        self.config = {
            "framework": {"name": "QwenPILF_v3", "action_model": {
                "action_dim": 14, "action_horizon": 50,
                "isolate_branch_tokens_before_shared": True,
            }},
            "datasets": {"vla_data": {"include_state": False}},
        }
        self.write_config()
        self.low = np.arange(14, dtype=np.float32)
        self.high = self.low + 2
        self.stats = {"new_embodiment": {"action": {
            "min": self.low.tolist(), "max": self.high.tolist(),
            "mask": [True] * 12 + [False] * 2,
        }}}
        (self.run / "dataset_statistics.json").write_text(json.dumps(self.stats), encoding="utf-8")
        self.options = {"env_cfg_type": "test_robot", "action_type": "joint",
                        "checkpoint_path": str(self.weight), "model_root": str(self.model_root)}
        self.policy = SyntheticPolicy()
        self.patchers = [
            patch.object(model, "get_robot_action_dim_info", return_value={"arm_dim": [6, 6], "ee_dim": [1, 1]}),
            patch.object(model, "_load_policy", return_value=self.policy),
            patch.dict(os.environ, {"DF_ROOT": str(self.model_root)}),
        ]
        for patcher in self.patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    def write_config(self):
        (self.run / "config.yaml").write_text(yaml.safe_dump(self.config), encoding="utf-8")

    def test_rgb_scaling_grippers_and_reset(self):
        adapter = model.Model(self.options)
        obs = observation()
        adapter.update_obs(obs)
        actions = adapter.get_action()
        self.assertEqual(len(actions), 50)
        vector = np.linspace(-1, 1, 14, dtype=np.float32)
        expected_joints = 0.5 * (vector + 1) * (self.high - self.low) + self.low
        np.testing.assert_allclose(actions[0]["left_arm_joint_state"], expected_joints[:6])
        np.testing.assert_allclose(actions[0]["right_arm_joint_state"], expected_joints[6:12])
        np.testing.assert_array_equal(actions[0]["left_ee_joint_state"], [0])
        np.testing.assert_array_equal(actions[0]["right_ee_joint_state"], [1])
        inputs = self.policy.calls[-1]
        self.assertFalse(inputs["do_sample"])
        self.assertTrue(inputs["use_ddim"])
        self.assertNotIn("state", inputs["examples"][0])
        for image in inputs["examples"][0]["image"]:
            self.assertEqual(image.shape, (224, 224, 3))
            np.testing.assert_array_equal(image[0, 0], [20, 80, 160])
        self.assertFalse(obs["vision"]["cam_head"]["color"].flags.writeable)
        adapter.reset()
        self.assertEqual(adapter.get_action_batch(), [])
        with self.assertRaises(ValueError):
            adapter.get_action()

    def test_batch_selection_and_instruction_refresh(self):
        adapter = model.Model(self.options)
        adapter.update_obs_batch([observation(7), observation(3)])
        chunks = adapter.get_action_batch([3, 7])
        self.assertEqual([len(chunk) for chunk in chunks], [50, 50])
        self.assertEqual([ex["lang"] for ex in self.policy.calls[-1]["examples"]], ["task 3", "task 7"])
        obs = observation(3)
        obs["instruction"] = "new task"
        adapter.update_obs(obs)
        adapter.get_action()
        self.assertEqual(self.policy.calls[-1]["examples"][0]["lang"], "new task")
        with self.assertRaises(ValueError):
            adapter.update_obs_batch([observation(3), observation(3)])

    def test_server_decoded_standard_and_legacy_images(self):
        adapter = model.Model(self.options)
        obs = observation()
        obs["vision"]["cam_head"]["color"] = encode_image_bit(obs["vision"]["cam_head"]["color"])
        # Marker-free RGB JPEGs reproduce the legacy trajectory encoding in memory only.
        ok, legacy = cv2.imencode(".jpg", obs["vision"]["cam_left_wrist"]["color"])
        self.assertTrue(ok)
        obs["vision"]["cam_left_wrist"]["color"] = legacy.tobytes()
        adapter.update_obs(decode_obs_images(obs))
        adapter.get_action()
        for image in self.policy.calls[-1]["examples"][0]["image"]:
            np.testing.assert_allclose(image[0, 0], [20, 80, 160], atol=3)

    def test_reject_historical_checkpoint(self):
        self.config["framework"]["action_model"]["isolate_branch_tokens_before_shared"] = False
        self.write_config()
        with self.assertRaisesRegex(ValueError, "branch-isolated"):
            model.Model(self.options)

    def test_run_directory_resolution(self):
        self.options["checkpoint_path"] = str(self.run)
        adapter = model.Model(self.options)
        self.assertEqual(adapter.action_chunk_size, 50)

    def test_nonfinite_actions_rejected(self):
        adapter = model.Model(self.options)
        adapter.update_obs(observation())
        with patch.object(self.policy, "predict_action", return_value={"normalized_actions": np.full((1, 50, 14), np.nan)}):
            with self.assertRaisesRegex(ValueError, "finite"):
                adapter.get_action()

    def test_reject_wrong_robot_and_statistics(self):
        with patch.object(model, "get_robot_action_dim_info", return_value={"arm_dim": [7, 7], "ee_dim": [1, 1]}):
            with self.assertRaisesRegex(ValueError, "6-joint"):
                model.Model(self.options)
        self.stats["new_embodiment"]["action"]["mask"] = [True] * 14
        (self.run / "dataset_statistics.json").write_text(json.dumps(self.stats), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "binary grippers"):
            model.Model(self.options)

    def test_deploy_replans_after_50_actions(self):
        adapter = model.Model(self.options)

        class Environment:
            steps = 0

            def is_episode_end(self):
                return self.steps >= 55

            def get_obs(self):
                return observation()

            def take_action(self, action):
                self.steps += 1

        class Client:
            def call(self, func_name, obs=None):
                return getattr(adapter, func_name)() if obs is None else getattr(adapter, func_name)(obs)

        env = Environment()
        deploy.eval_one_episode(env, Client())
        self.assertEqual(env.steps, 55)
        self.assertEqual(len(self.policy.calls), 2)

    def test_batch_deploy_removes_finished_environments(self):
        adapter = model.Model(self.options)

        class Environment:
            steps = {7: 0, 3: 0}
            limits = {7: 2, 3: 55}

            def get_running_env_idx_list(self):
                return [index for index in self.steps if self.steps[index] < self.limits[index]]

            def is_episode_end(self):
                return not self.get_running_env_idx_list()

            def get_obs_batch(self, indices):
                return [observation(index) for index in indices]

            def take_action_batch(self, actions, indices):
                for index in indices:
                    self.steps[index] += 1

        class Client:
            def call(self, func_name, obs=None):
                return getattr(adapter, func_name)() if obs is None else getattr(adapter, func_name)(obs)

        env = Environment()
        deploy.eval_one_episode_batch(env, Client())
        self.assertEqual(env.steps, {7: 2, 3: 55})
        self.assertEqual([len(call["examples"]) for call in self.policy.calls], [2, 1])


class PreparedDataTest(unittest.TestCase):
    def test_validate_then_link_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mixture = root / "model/starVLA/dataloader/gr00t_lerobot/mixtures.py"
            mixture.parent.mkdir(parents=True)
            mixture.write_text("ROBOTWIN_CLEAN_TASKS = ('task_one', 'task_two')\n", encoding="utf-8")
            template = root / "model/examples/Robotwin/train_files/modality.json"
            template.parent.mkdir(parents=True)
            modality = {"action": {"left_joints": {"start": 0, "end": 6, "original_key": "action"}}}
            template.write_text(json.dumps(modality), encoding="utf-8")
            for task in ("task_one", "task_two"):
                meta = root / "data/Clean" / task / "meta"
                meta.mkdir(parents=True)
                (meta / "info.json").write_text(json.dumps({"codebase_version": "v2.1"}), encoding="utf-8")
                (meta / "modality.json").write_text(json.dumps(modality), encoding="utf-8")
            destination = root / "linked"
            with patch.object(Path, "symlink_to") as link:
                prepare_data.prepare(root / "model", root / "data", destination)
            link.assert_called_once_with((root / "data").resolve(), target_is_directory=True)
            destination.mkdir()
            with self.assertRaises(FileExistsError):
                prepare_data.prepare(root / "model", root / "data", destination)
            (root / "data/Clean/task_two/meta/modality.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "modality"):
                prepare_data.prepare(root / "model", root / "data", root / "other_link")
            self.assertFalse((root / "other_link").exists())


if __name__ == "__main__":
    unittest.main()
