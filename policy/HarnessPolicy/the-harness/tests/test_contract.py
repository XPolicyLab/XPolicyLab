"""Observation contracts: what an evaluator lets a policy see (core/contract.py), and the
prompt and system resolution that follow it."""

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from harness_policy.config.agent import resolve_agent  # noqa: E402
from harness_policy.core.contract import ContractFilter, _matmul, pose_matrix  # noqa: E402
from harness_policy.embodiments import load_embodiment  # noqa: E402

TEST_DEPLOY = ROOT / "tests" / "fixtures" / "deploy-test.toml"
CONFIGS = ROOT / "configs"
EMBODIMENT = load_embodiment("arx_x5")


class ContractFilterTests(unittest.TestCase):
    def setUp(self):
        self.filter = ContractFilter("xpolicylab", EMBODIMENT)
        self.joints = [0.1] * 14
        self.joints[6], self.joints[13] = 0.19, 0.5   # measured openings

    def test_profile_reports_the_contract_and_no_step_limit(self):
        profile = self.filter.apply("profile", {}, {"control_hz": 25, "step_limit": 800})
        self.assertNotIn("step_limit", profile)
        self.assertEqual(profile["observation_contract"], "xpolicylab")
        self.assertEqual(profile["gripper_feedback"], "commanded")

    def test_grippers_read_the_last_command(self):
        reset = self.filter.apply("reset", {}, {"state": {"joint_position": [0.0] * 6 + [1.0] + [0.0] * 6 + [1.0],
                                                          "step_limit": 800}})
        self.assertNotIn("step_limit", reset["state"])
        command = [0.0] * 14
        command[6] = 0.0   # close the left gripper
        command[13] = 1.0
        result = self.filter.apply("tick", {"command": command}, {
            "executed": True, "shield": {"command": command}, "state": {"joint_position": self.joints}})
        self.assertEqual(result["state"]["joint_position"][6], 0.0)
        self.assertEqual(result["state"]["joint_position"][13], 1.0)
        self.assertEqual(result["state"]["joint_position"][0], 0.1)   # arm joints stay measured
        state = self.filter.apply("state", {}, {"joint_position": self.joints, "step_limit": 800})
        self.assertEqual(state["joint_position"][6], 0.0)
        self.assertNotIn("step_limit", state)

    def test_calibration_comes_from_the_embodiment(self):
        pose = [0.2, -0.3, 1.0, 1.0, 0.0, 0.0, 0.0]
        data = {"state": {"joint_position": self.joints, "left_ee_pose": pose, "right_ee_pose": pose},
                "intrinsics": {"cam_head": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "cam_left_wrist": [[1]]},
                "camera_to_world": {"cam_head": [[0]], "cam_left_wrist": [[0]]}}
        observed = self.filter.apply("observe", {}, data)
        calibration = EMBODIMENT["calibration"]
        self.assertEqual(observed["intrinsics"]["cam_head"], calibration["cam_head"]["intrinsics"])
        self.assertEqual(observed["camera_to_world"]["cam_head"], calibration["cam_head"]["camera_to_world"])
        expected = _matmul(pose_matrix(pose), calibration["cam_left_wrist"]["camera_on_link6"])
        self.assertEqual(observed["camera_to_world"]["cam_left_wrist"], expected)

    def test_native_changes_nothing(self):
        native = ContractFilter("native", EMBODIMENT)
        data = {"joint_position": self.joints, "step_limit": 800}
        self.assertIs(native.apply("state", {}, data), data)


class ContractAgentTests(unittest.TestCase):
    def resolve(self, name, **options):
        path = CONFIGS / ("default.toml" if name == "default" else f"ablations/{name}.toml")
        return resolve_agent(path, deploy=str(TEST_DEPLOY), name=name, **options)

    def test_prompt_follows_the_contract(self):
        contract, native = self.resolve("default"), self.resolve("default", runtime_overrides={"contract": "native"})
        self.assertIn('get_state()["step_limit"]', native.system_prompt)
        self.assertIn("measured opening of an empty closed", native.system_prompt)
        self.assertNotIn('["step_limit"]', contract.system_prompt)
        self.assertIn("not reported", contract.system_prompt)
        self.assertIn("last commands, not the measured openings", contract.system_prompt)
        self.assertNotEqual(native.identity, contract.identity)
        self.assertEqual(contract.lock["runtime"]["contract"], "xpolicylab")
        self.assertEqual(native.lock["skills"], contract.lock["skills"])

    def test_direct_control_follows_the_contract(self):
        contract, native = self.resolve("r0"), self.resolve("r0", runtime_overrides={"contract": "native"})
        self.assertIn("measured opening of an empty closed", native.system_prompt)
        self.assertIn("last commands, not the measured openings", contract.system_prompt)
        self.assertNotIn("<!--", contract.system_prompt)

    def test_task_descriptions_need_the_runner_not_the_policy(self):
        agent = self.resolve("with-task-description")
        self.assertEqual(agent.runtime["contract"], "xpolicylab")
        self.assertTrue(agent.runtime["task_descriptions"].endswith("robodojo_tasks"))


if __name__ == "__main__":
    unittest.main()
