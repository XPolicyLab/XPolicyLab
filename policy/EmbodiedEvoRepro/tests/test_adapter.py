"""CPU contract/failure tests; none imports or calls the external model runtime."""

import io
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from XPolicyLab.policy.EmbodiedEvoRepro import configuration
from XPolicyLab.policy.EmbodiedEvoRepro.download_assets import extract_checked
from XPolicyLab.policy.EmbodiedEvoRepro.model import Model

MODULE = "XPolicyLab.policy.EmbodiedEvoRepro.model"
CFG = {"bench_name": "RoboDojo", "task_name": "stack_bowls", "env_cfg_type": "arx_x5",
       "action_type": "ee", "seed": 1, "protocol": "ws", "num_envs": 1, "eval_batch": False}
DIMS = {"arm_dim": [6, 6], "ee_dim": [1, 1]}


def observation():
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    rgb[:, :, 0], rgb[:, :, 2] = 211, 17
    rgb.setflags(write=False)
    state = {}
    for prefix in ("left_", "right_"):
        state[prefix + "ee_pose"] = np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32)
        state[prefix + "ee_joint_state"] = np.zeros(1, dtype=np.float32)
    return {"vision": {"head": {"color": rgb}}, "state": state}


class AdapterBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"EVAL_ENV_TYPE": "debug", "EMBODIEDEVO_INTERFACE_PROBE": "1"})
        self.env.start()
        self.dims = patch(MODULE + ".get_robot_action_dim_info", return_value=DIMS)
        self.dims.start()

    def tearDown(self):
        self.dims.stop()
        self.env.stop()

    def test_probe_cannot_run_in_sim(self):
        with patch.dict(os.environ, {"EVAL_ENV_TYPE": "sim"}):
            with self.assertRaisesRegex(ValueError, "only"):
                Model(CFG)

    def test_debug_requires_explicit_probe(self):
        with patch.dict(os.environ, {"EMBODIEDEVO_INTERFACE_PROBE": "0"}):
            with self.assertRaisesRegex(ValueError, "explicit"):
                Model(CFG)

    def test_native_never_silently_falls_back_to_probe(self):
        with patch.dict(os.environ, {"EVAL_ENV_TYPE": "sim", "EMBODIEDEVO_INTERFACE_PROBE": "0"}), \
                patch(MODULE + ".verified_upstream", side_effect=ValueError("missing source")):
            with self.assertRaisesRegex(ValueError, "missing source"):
                Model(CFG)

    def test_real_mode_refuses_missing_frozen_assets_before_run_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "frozen-source"
            source.mkdir()
            run = root / "202610270000_attempt"
            with patch.dict(os.environ, {
                "EVAL_ENV_TYPE": "sim", "EMBODIEDEVO_INTERFACE_PROBE": "0",
                "EMBODIEDEVO_SOURCE_RUN": str(source), "EMBODIEDEVO_RUN_DIR": str(run)
            }), patch(MODULE + ".verified_upstream", return_value=root):
                with self.assertRaisesRegex(FileNotFoundError, "Required frozen source"):
                    Model(CFG)
            self.assertFalse(run.exists())

    def test_rejects_unsupported_scope(self):
        for changed in ({"action_type": "joint"}, {"num_envs": 2}, {"phase": "playground"},
                        {"eval_batch": True}, {"diagnostic": True}, {"seed": 4}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                Model({**CFG, **changed})

    def test_rgb_and_readonly_observation_remain_unchanged(self):
        model = Model(CFG)
        obs = observation()
        model.update_obs(obs)
        np.testing.assert_array_equal(model.obs["vision"]["head"]["color"], obs["vision"]["head"]["color"])
        actions = model.get_action()
        self.assertEqual(len(actions), 1)
        self.assertEqual(set(actions[0]), set(model.action_shapes))
        model.obs["vision"]["head"]["color"][0, 0, 0] = 0
        self.assertEqual(obs["vision"]["head"]["color"][0, 0, 0], 211)

    def test_encoded_color_is_rejected_at_model_boundary(self):
        model = Model(CFG)
        obs = observation()
        obs["vision"]["head"]["color"] = b"encoded bytes must be decoded by server"
        with self.assertRaisesRegex(TypeError, "already-decoded"):
            model.update_obs(obs)

    def test_invalid_action_values_are_not_repaired(self):
        for bad in (np.zeros(5), np.array([0, 0, 0, np.nan, 0, 0, 0])):
            model = Model(CFG)
            obs = observation()
            obs["state"]["left_ee_pose"] = bad
            model.update_obs(obs)
            with self.assertRaisesRegex(ValueError, "Invalid action"):
                model.get_action()

    def test_single_environment_batch_contract(self):
        model = Model(CFG)
        model.update_obs_batch([observation()])
        self.assertEqual(len(model.get_action_batch([0])), 1)
        with self.assertRaises(ValueError):
            model.update_obs_batch([observation(), observation()])
        with self.assertRaises(ValueError):
            model.get_action_batch([1])

    def test_reset_clears_observation_but_cannot_reset_simulation(self):
        model = Model(CFG)
        model.update_obs(observation())
        model.reset()
        with self.assertRaises(RuntimeError):
            model.get_action()
        with self.assertRaises(PermissionError):
            model.complete_reset(observation())

    def test_native_delegate_receives_unmodified_feedback(self):
        model = Model(CFG)
        model.delegate = Mock()
        model.spec = {"episodes": 1}
        result = {"success": False, "terminated": False, "truncated": True}
        with patch.object(model, "_verify_harness"):
            model.on_trial_end(result)
        model.delegate.on_trial_end.assert_called_once_with(result)
        with self.assertRaisesRegex(RuntimeError, "episode count"):
            model.prepare_case({"native_step_limit": 100})

    def test_native_propagates_worker_error(self):
        model = Model(CFG)
        model.delegate = Mock()
        model.delegate.get_action.side_effect = RuntimeError("worker failed")
        with self.assertRaisesRegex(RuntimeError, "worker failed"):
            model.get_action()

    def test_missing_upstream_is_rejected_without_importing_model(self):
        with patch.dict(os.environ, {"EMBODIEDEVO_REPO": ""}):
            with self.assertRaisesRegex(ValueError, "EMBODIEDEVO_REPO"):
                configuration.verified_upstream()


class AssetExtractionTests(unittest.TestCase):
    def archive(self, root, extra=None):
        archive = root / "bundle.tar.gz"
        payload = {"protocol.json": b"{}", "inputs/test.md": b"test",
                   "inputs/primitives/primitives.md": b"primitives", "frozen_harness/skills/example.md": b"skill"}
        with tarfile.open(archive, "w:gz") as tar:
            for name, data in payload.items():
                item = tarfile.TarInfo(name)
                item.size = len(data)
                tar.addfile(item, io.BytesIO(data))
            if extra is not None:
                tar.addfile(extra, io.BytesIO(b""))
        return archive

    def test_extracts_only_expected_bundle_shape(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            extract_checked(self.archive(root), root / "output")
            self.assertEqual((root / "output/frozen_harness/skills/example.md").read_text(), "skill")
            with self.assertRaises(FileExistsError):
                extract_checked(root / "bundle.tar.gz", root / "output")

    def test_rejects_path_traversal_and_credentials(self):
        for name in ("../escape", "inputs/../../escape", "/tmp/escape", "inputs/.codex/auth.json"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                item = tarfile.TarInfo(name)
                with self.assertRaises(ValueError):
                    extract_checked(self.archive(root, item), root / "output")
                self.assertFalse((root / "output").exists())

    def test_rejects_symlinks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            item = tarfile.TarInfo("inputs/link")
            item.type, item.linkname = tarfile.SYMTYPE, "/tmp/elsewhere"
            with self.assertRaises(ValueError):
                extract_checked(self.archive(root, item), root / "output")


if __name__ == "__main__":
    unittest.main()
