import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from XPolicyLab.utils.process_data import encode_image_bit


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


utils = load_module("act_utils", "policy/ACT/utils.py")
process_data = load_module("act_process_data", "policy/ACT/detr/process_data.py")


class ActionAlignmentTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.length = 80
        self.dim = 14
        self.states = np.arange((self.length + 1) * self.dim, dtype=np.float32).reshape(-1, self.dim)
        self.stats = {
            "qpos_mean": np.zeros(self.dim, dtype=np.float32),
            "qpos_std": np.ones(self.dim, dtype=np.float32),
            "action_mean": np.zeros(self.dim, dtype=np.float32),
            "action_std": np.ones(self.dim, dtype=np.float32),
        }

    def write_episode(self, attrs, length=None):
        length = self.length if length is None else length
        with h5py.File(self.directory / "episode_0.hdf5", "w") as root:
            for key, value in attrs.items():
                root.attrs[key] = value
            root.create_dataset("observations/qpos", data=self.states[:length])
            root.create_dataset("action", data=self.states[1:length + 1])
            root.create_dataset("observations/images/cam_head", data=np.zeros((length, 8, 8, 3), dtype=np.uint8))

    def sample(self, start, directory=None):
        directory = self.directory if directory is None else directory
        # The constructor also samples once to initialize is_sim.
        with patch.object(utils.np.random, "choice", return_value=start):
            dataset = utils.EpisodicDataset([0], str(directory), ["cam_head"], self.stats, self.length)
            return dataset, dataset[0]

    def assert_sample(self, start, action_start, length=None, directory=None, expected_actions=None):
        length = self.length if length is None else length
        expected_actions = self.states[1:length + 1] if expected_actions is None else expected_actions
        dataset, (images, qpos, actions, is_pad) = self.sample(start, directory)
        valid = length - action_start
        np.testing.assert_allclose(qpos.numpy() * self.stats["qpos_std"] + self.stats["qpos_mean"],
                                   self.states[start])
        restored = actions.numpy() * self.stats["action_std"] + self.stats["action_mean"]
        np.testing.assert_allclose(restored[:valid], expected_actions[action_start:length], rtol=1e-6)
        np.testing.assert_allclose(restored[valid:], 0, atol=1e-6)
        np.testing.assert_array_equal(is_pad.numpy(), np.arange(self.length) >= valid)
        self.assertEqual(tuple(images.shape)[0:2], (1, 3))
        return dataset

    def test_next_state_chunk_at_nonzero_timestamp(self):
        self.write_episode({"action_alignment": "next_state", "sim": True})
        self.assert_sample(20, 20)  # Includes every target in the first 50-step chunk.

    def test_explicit_alignment_overrides_sim(self):
        for alignment, action_start in (("next_state", 20), ("legacy_previous", 19)):
            for sim in (None, False, True):
                with self.subTest(alignment=alignment, sim=sim):
                    attrs = {"action_alignment": alignment}
                    if sim is not None:
                        attrs["sim"] = sim
                    self.write_episode(attrs)
                    dataset = self.assert_sample(20, action_start)
                    self.assertIs(dataset.is_sim, sim)

    def test_sim_fallback(self):
        for sim, action_start in ((True, 20), (False, 19), (1, 20), (0, 19)):
            with self.subTest(sim=sim):
                self.write_episode({"sim": sim})
                dataset = self.assert_sample(20, action_start)
                self.assertIs(dataset.is_sim, bool(sim))

    def test_missing_metadata_preserves_legacy(self):
        self.write_episode({})
        dataset = self.assert_sample(20, 19)
        self.assertIsNone(dataset.is_sim)

    def test_byte_alignment(self):
        for alignment, action_start in ((b"next_state", 20), (b"legacy_previous", 19)):
            with self.subTest(alignment=alignment):
                self.write_episode({"action_alignment": np.bytes_(alignment)})
                self.assert_sample(20, action_start)

    def test_boundaries_and_short_episodes(self):
        for alignment in ("next_state", "legacy_previous"):
            for length, start in ((80, 0), (80, 79), (3, 0), (3, 1), (3, 2), (1, 0)):
                with self.subTest(alignment=alignment, length=length, start=start):
                    self.write_episode({"action_alignment": alignment}, length)
                    action_start = start if alignment == "next_state" else max(0, start - 1)
                    self.assert_sample(start, action_start, length)

    def test_nontrivial_normalization(self):
        self.stats = {
            "qpos_mean": np.arange(self.dim, dtype=np.float32),
            "qpos_std": np.full(self.dim, 2, dtype=np.float32),
            "action_mean": np.arange(self.dim, dtype=np.float32) + 10,
            "action_std": np.full(self.dim, 4, dtype=np.float32),
        }
        for alignment, action_start in (("next_state", 20), ("legacy_previous", 19)):
            with self.subTest(alignment=alignment):
                self.write_episode({"action_alignment": alignment})
                self.assert_sample(20, action_start)

    def test_invalid_alignment_fails(self):
        for value in ("unknown", np.bytes_(b"unknown"), 1, ["next_state"]):
            with self.subTest(value=value):
                self.write_episode({"action_alignment": value})
                with self.assertRaisesRegex(ValueError, "action_alignment"):
                    self.sample(20)

    def test_invalid_sim_fails_instead_of_using_truthiness(self):
        for value in ("False", "True", np.bytes_(b"False"), 2, [True]):
            with self.subTest(value=value):
                self.write_episode({"sim": value})
                with self.assertRaisesRegex(ValueError, "sim"):
                    self.sample(20)

    def test_standard_converter_to_loader(self):
        raw_dir = self.directory / "data"
        raw_dir.mkdir()
        length = 4
        robot_info = {"arm_dim": [6, 6], "ee_dim": [1, 1]}
        image = np.full((8, 8, 3), [20, 80, 160], dtype=np.uint8)
        encoded = encode_image_bit(image)
        # Commands need not equal the next measured state (target_offset=100).
        for sim, target_offset in ((None, 0), (False, 0), (True, 0), (False, 100)):
            with self.subTest(sim=sim, target_offset=target_offset):
                targets = self.states[1:length + 1] + target_offset
                with h5py.File(raw_dir / "episode_0000000.hdf5", "w") as root:
                    if sim is not None:
                        root.attrs["sim"] = sim
                    for group, values in (("state", self.states[:length]), ("action", targets)):
                        root.create_dataset(f"{group}/left_arm_joint_states", data=values[:, :6])
                        root.create_dataset(f"{group}/left_ee_joint_states", data=values[:, 6:7])
                        root.create_dataset(f"{group}/right_arm_joint_states", data=values[:, 7:13])
                        root.create_dataset(f"{group}/right_ee_joint_states", data=values[:, 13:14])
                    for camera in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
                        root.create_dataset(f"vision/{camera}/colors", data=np.stack([encoded] * length))
                output = self.directory / "converted"
                with patch.object(process_data, "action_type", "joint", create=True):
                    count = process_data.data_transform(str(raw_dir), 1, str(self.directory), str(output), robot_info)
                self.assertEqual(count, 1)
                with h5py.File(output / "episode_0.hdf5", "r") as root:
                    self.assertEqual(root.attrs["action_alignment"], "next_state")
                    self.assertEqual(root.attrs.get("sim"), sim)
                    np.testing.assert_array_equal(root["action"][()], targets)
                    np.testing.assert_array_equal(root["observations/qpos"][()], self.states[:length])
                self.assert_sample(2, 2, length, output, targets)


if __name__ == "__main__":
    unittest.main()
