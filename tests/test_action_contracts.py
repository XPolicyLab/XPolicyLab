"""Regression checks for the shared action contract, without model weights."""

import ast
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from XPolicyLab.utils import process_data
from XPolicyLab.utils.debug_env_client import TestEnv, validate_robot_state_dict
from XPolicyLab.policy.AHA_WAM import debug_env_client as aha_debug
from XPolicyLab.policy.AHA_WAM.debug_env_client import (
    TestEnv as AhaTestEnv,
    _validate_robot_state_dict as validate_aha_action,
)


ROOT = Path(__file__).resolve().parents[1]
SINGLE = {"arm_dim": [6], "ee_dim": [1]}
DUAL = {"arm_dim": [6, 6], "ee_dim": [1, 1]}


class ActionContractTests(unittest.TestCase):
    def test_single_arm_aliases_pack_identically_without_mutating_input(self):
        for source_type, suffix, shape in (("obs", "", (6,)), ("dataset", "s", (2, 6))):
            with self.subTest(source_type=source_type):
                arm = np.arange(np.prod(shape)).reshape(shape)
                gripper = np.ones(shape[:-1] + (1,))
                canonical = {f"arm_joint_state{suffix}": arm, f"ee_joint_state{suffix}": gripper}
                legacy = {f"joint_state{suffix}": arm, f"ee_joint_state{suffix}": gripper}
                expected = np.concatenate([arm, gripper], axis=-1)
                for state in (canonical, legacy):
                    packed = process_data.pack_robot_state({"state": state}, "joint", SINGLE, source_type)
                    np.testing.assert_array_equal(packed, expected)
                self.assertIn(f"joint_state{suffix}", legacy)
                self.assertNotIn(f"arm_joint_state{suffix}", legacy)
                unpacked = process_data.unpack_robot_state(expected, "joint", SINGLE, source_type)
                self.assertEqual(set(unpacked), set(canonical))
                for key, value in canonical.items():
                    np.testing.assert_array_equal(unpacked[key], value)

    def test_ambiguous_aliases_are_rejected(self):
        for source_type, suffix in (("obs", ""), ("dataset", "s")):
            state = {f"joint_state{suffix}": np.zeros(6), f"arm_joint_state{suffix}": np.zeros(6),
                     f"ee_joint_state{suffix}": np.zeros(1)}
            with self.subTest(source_type=source_type), self.assertRaisesRegex(ValueError, "Ambiguous"):
                process_data.pack_robot_state({"state": state}, "joint", SINGLE, source_type)

    def test_aha_boundary_preserves_legacy_input_and_action_chunks(self):
        # Execute only the adapter's pure boundary wrappers. This checks its
        # integration with shared helpers without claiming to load the model.
        source = ROOT / "policy/AHA_WAM/model.py"
        tree = ast.parse(source.read_text())
        wrappers = {"_pack_robot_state", "_unpack_robot_state"}
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wrappers]
        namespace = {"np": np, "pack_robot_state": process_data.pack_robot_state,
                     "unpack_robot_state": process_data.unpack_robot_state}
        exec(compile(tree, str(source), "exec"), namespace)
        state = {"joint_state": np.arange(6), "ee_joint_state": np.array([0.5])}
        packed = namespace["_pack_robot_state"]({"state": state}, "joint", SINGLE)
        np.testing.assert_array_equal(packed, [0, 1, 2, 3, 4, 5, 0.5])
        for values, chunk_size in ((packed, 1), (np.stack([packed, packed]), 2)):
            chunk = namespace["_unpack_robot_state"](values, "joint", SINGLE)
            self.assertIsInstance(chunk, list)
            self.assertEqual(len(chunk), chunk_size)
            for action in chunk:
                validate_robot_state_dict(action, SINGLE)
                self.assertIn("arm_joint_state", action)

    def test_complete_actions_match_both_robot_and_action_types(self):
        for robot in (SINGLE, DUAL):
            for action_type in ("joint", "ee"):
                with self.subTest(robot=robot, action_type=action_type):
                    width = sum(robot["arm_dim"]) if action_type == "joint" else 7 * len(robot["arm_dim"])
                    values = np.arange(width + sum(robot["ee_dim"]), dtype=np.float32)
                    action = process_data.unpack_robot_state(values, action_type, robot)
                    validate_robot_state_dict(action, robot, action_type)
                    np.testing.assert_array_equal(
                        process_data.pack_robot_state({"state": action}, action_type, robot), values
                    )
        validate_robot_state_dict({"joint_state": [0] * 6, "ee_joint_state": [0]}, SINGLE)

    def test_invalid_actions_cannot_pass_debug_validation(self):
        valid = {"arm_joint_state": np.zeros(6), "ee_joint_state": np.zeros(1)}
        invalid = [
            {}, {"arm_joint_state": np.zeros(6)},
            dict(valid, unexpected=[0]),
            dict(valid, arm_joint_state=np.zeros(7)),
            dict(valid, arm_joint_state=np.zeros((1, 6))),
            dict(valid, arm_joint_state=np.full(6, np.nan)),
            dict(valid, arm_joint_state=np.full(6, np.inf)),
            dict(valid, arm_joint_state=["0"] * 6),
            dict(valid, arm_joint_state=np.zeros(6, dtype=complex)),
            dict(valid, arm_joint_state=np.zeros(6, dtype=bool)),
        ]
        for index, action in enumerate(invalid):
            with self.subTest(index=index), self.assertRaises((TypeError, ValueError, KeyError)):
                validate_robot_state_dict(action, SINGLE)
        with self.assertRaises(KeyError):
            validate_robot_state_dict(valid, SINGLE, "ee")

    def test_failed_single_or_batch_actions_do_not_advance_debug_steps(self):
        for env_class in (TestEnv, AhaTestEnv):
            with self.subTest(client=env_class.__module__):
                # Avoid connecting a client: exercise the actual action boundary.
                env = env_class.__new__(env_class)
                env.robot_action_dim_info = SINGLE
                env.action_type = "joint"
                env.episode_step = 0
                env.episode_step_limit = 20
                valid = {"arm_joint_state": np.zeros(6), "ee_joint_state": np.zeros(1)}
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(KeyError):
                        env.take_action({})
                    with self.assertRaises(ValueError):
                        env.take_action(dict(valid, arm_joint_state=np.full(6, np.nan)))
                    with self.assertRaises(KeyError):
                        env.take_action_batch([valid, {}], [0, 1])
                    with self.assertRaises(ValueError):
                        env.take_action_batch([], [])
                    with self.assertRaises(ValueError):
                        env.take_action_batch([valid], [0, 1])
                    self.assertEqual(env.episode_step, 0)
                    env.take_action(valid)
                    env.take_action_batch([valid, valid], [0, 1])
                    self.assertEqual(env.episode_step, 2)

    def test_aha_debug_requires_complete_finite_joint_actions(self):
        for robot in (SINGLE, DUAL):
            with self.subTest(robot=robot):
                width = sum(robot["arm_dim"]) + sum(robot["ee_dim"])
                action = process_data.unpack_robot_state(np.zeros(width), "joint", robot)
                validate_aha_action(action, robot)
                with self.assertRaises(KeyError):
                    validate_aha_action({}, robot)
                first_key = next(iter(action))
                for invalid in (np.nan, np.inf):
                    broken = dict(action)
                    broken[first_key] = np.full_like(action[first_key], invalid)
                    with self.assertRaises(ValueError):
                        validate_aha_action(broken, robot)
                pose_action = process_data.unpack_robot_state(
                    np.zeros(8 * len(robot["arm_dim"])), "ee", robot
                )
                with self.assertRaises(KeyError):
                    validate_aha_action(pose_action, robot)

    def test_aha_debug_uses_unique_evaluations_and_preserves_explicit_id(self):
        config = {"env_cfg_type": "single", "env_cfg_root": "/unused", "host": "localhost", "port": 19000}
        with patch.object(aha_debug, "_load_robot_action_dim_info", return_value=SINGLE), \
             patch.object(aha_debug, "WsModelClient") as client:
            AhaTestEnv(config)
            first_id = client.call_args.kwargs["evaluation_id"]
            AhaTestEnv(config)
            second_id = client.call_args.kwargs["evaluation_id"]
            self.assertNotEqual(first_id, second_id)
            self.assertTrue(first_id.startswith("aha-wam-debug-"))
            AhaTestEnv(dict(config, evaluation_id="resume-this-evaluation"))
            self.assertEqual(client.call_args.kwargs["evaluation_id"], "resume-this-evaluation")

    def test_python_shell_and_packed_widths_agree(self):
        # Both metadata files deliberately describe the same synthetic robots.
        robots = {"single": SINGLE, "dual": DUAL}
        with tempfile.TemporaryDirectory() as directory:
            metadata = Path(directory) / "XPolicyLab/utils/robot/_robot_info.json"
            metadata.parent.mkdir(parents=True)
            metadata.write_text(json.dumps(robots))
            env = dict(os.environ, PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""))
            for name, robot in robots.items():
                with patch.object(process_data, "load_yaml", return_value={"config": {"robot": name}}), \
                     patch.object(process_data, "load_json", return_value=robots):
                    for action_type, expected in (("joint", 7 if name == "single" else 14),
                                                  ("ee", 8 if name == "single" else 16)):
                        with self.subTest(robot=name, action_type=action_type):
                            self.assertEqual(process_data.get_action_dim(name, action_type), expected)
                            shell = subprocess.run(
                                ["bash", str(ROOT / "utils/get_action_dim.sh"), directory, name, action_type],
                                env=env, check=True, capture_output=True, text=True,
                            )
                            self.assertEqual(int(shell.stdout), expected)
                            self.assertEqual(sum(process_data.get_robot_state_schema(action_type, robot).values()), expected)
                    self.assertEqual(process_data.get_action_dim(name), sum(robot["arm_dim"]) + sum(robot["ee_dim"]))
                    with self.assertRaises(ValueError):
                        process_data.get_action_dim(name, "invalid")
            invalid = subprocess.run(
                ["bash", str(ROOT / "utils/get_action_dim.sh"), directory, "single", "invalid"],
                env=env, capture_output=True, text=True,
            )
            self.assertNotEqual(invalid.returncode, 0)


if __name__ == "__main__":
    unittest.main()
