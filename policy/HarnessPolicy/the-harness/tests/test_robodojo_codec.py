import os
import sys
import unittest

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benches.robodojo.codec import (
    CAMERAS,
    action_dict,
    clamp_position,
    episode_outcome,
    hold_position,
    instruction_text,
    nudge_position,
    policy_observation,
    tool_schema,
    validate_position,
)


def _terms():
    joints = [f"joint{index}" for index in range(1, 7)]
    return [
        {
            "term": "left_arm",
            "key": "left_arm_joint_state",
            "joints": joints,
            "size": 6,
            "command": "absolute_joint",
        },
        {
            "term": "left_ee",
            "key": "left_ee_joint_state",
            "joints": ["normalized_opening"],
            "size": 1,
            "command": "gripper_normalized",
        },
        {
            "term": "right_arm",
            "key": "right_arm_joint_state",
            "joints": joints,
            "size": 6,
            "command": "absolute_joint",
        },
        {
            "term": "right_ee",
            "key": "right_ee_joint_state",
            "joints": ["normalized_opening"],
            "size": 1,
            "command": "gripper_normalized",
        },
    ]


LOW = [-2.0] * 6 + [0.0] + [-2.0] * 6 + [0.0]
HIGH = [2.0] * 6 + [1.0] + [2.0] * 6 + [1.0]


class RoboDojoCodecTests(unittest.TestCase):
    def setUp(self):
        self.terms = _terms()
        self.position = [0.1] * 6 + [0.2] + [0.3] * 6 + [0.4]

    def test_hold_opens_both_grippers_without_collapsing_duplicate_joint_names(self):
        command = hold_position(self.terms, self.position)
        self.assertEqual(command[:6], [0.1] * 6)
        self.assertEqual(command[6], 1.0)
        self.assertEqual(command[7:13], [0.3] * 6)
        self.assertEqual(command[13], 1.0)
        payload = action_dict(self.terms, command)
        self.assertEqual(set(payload), {
            "left_arm_joint_state",
            "left_ee_joint_state",
            "right_arm_joint_state",
            "right_ee_joint_state",
        })
        self.assertEqual(payload["left_ee_joint_state"], [1.0])
        self.assertEqual(payload["right_arm_joint_state"], [0.3] * 6)
        self.assertEqual(validate_position({"position": command}, list(zip(LOW, HIGH, strict=True))), command)

    def test_nudge_moves_one_arm_joint_and_crosses_to_the_right_arm(self):
        even = nudge_position(self.terms, self.position, LOW, HIGH, 0, 0.05)
        self.assertAlmostEqual(even[0], 0.15)
        self.assertEqual(even[1:6], [0.1] * 5)
        self.assertEqual(even[6], 1.0)
        self.assertEqual(even[13], 1.0)
        odd = nudge_position(self.terms, self.position, LOW, HIGH, 1, 0.05)
        self.assertAlmostEqual(odd[1], 0.05)
        right = nudge_position(self.terms, self.position, LOW, HIGH, 6, 0.05)
        self.assertAlmostEqual(right[7], 0.35)
        self.assertEqual(right[6], 1.0)
        saturated = [1.98, *self.position[1:]]
        clamped = nudge_position(self.terms, saturated, LOW, HIGH, 0, 0.05)
        self.assertEqual(clamped[0], 2.0)
        outside = clamp_position([3.0, *self.position[1:]], list(zip(LOW, HIGH, strict=True)))
        self.assertEqual(outside[0], 2.0)
        self.assertEqual(outside[6], 0.2)
        with self.assertRaises(ValueError):
            nudge_position(self.terms, self.position, LOW, HIGH, -1, 0.05)
        with self.assertRaises(ValueError):
            hold_position(self.terms, self.position, gripper_open=1.5)

    def test_schema_documents_the_dual_arm_command(self):
        bounds = list(zip(LOW, HIGH, strict=True))
        schema = tool_schema(bounds, self.terms)
        self.assertEqual(schema[0]["name"], "joint_position")
        self.assertEqual(schema[0]["parameters"]["properties"]["position"]["minItems"], 14)
        description = schema[0]["description"]
        self.assertIn("radians", description)
        self.assertIn("0.8", description)
        self.assertIn("left_arm_joint_state", description)
        self.assertIn("right_ee_joint_state", description)
        self.assertEqual(CAMERAS, ("cam_head", "cam_left_wrist", "cam_right_wrist"))

    def test_zero_process_score_is_not_success(self):
        reward, success, terminated, truncated = episode_outcome(False, True, 0.0, 0, 800)
        self.assertEqual((reward, success, terminated, truncated), (0.0, False, False, False))
        partial = episode_outcome(True, False, 15.0, 4, 800)
        self.assertEqual(partial, (0.15, False, False, True))
        solved = episode_outcome(True, True, 0.0, 3, 800)
        self.assertEqual(solved, (1.0, True, True, False))
        limited = episode_outcome(False, True, 0.0, 800, 800)
        self.assertEqual(limited, (0.0, False, False, True))
        with self.assertRaises(ValueError):
            episode_outcome(True, False, float("nan"), 1, 800)

    def test_instruction_uses_the_task_text(self):
        self.assertEqual(instruction_text(["Stack the three bowls together."]), "Stack the three bowls together.")
        self.assertEqual(instruction_text(None), "Stack the three bowls together.")
        self.assertEqual(instruction_text("  Place the bowls.  "), "Place the bowls.")


def self_position():
    return [0.1] * 6 + [0.2] + [0.3] * 6 + [0.4]


class RoboDojoImportTests(unittest.TestCase):
    def test_adapter_import_does_not_launch_isaac(self):
        forbidden = {"isaacsim", "isaaclab", "torch", "omni", "carb", "curobo"}
        before = set(sys.modules)
        import benches.robodojo.adapter as adapter

        imported = set(sys.modules) - before
        self.assertFalse(any(name.split(".", 1)[0] in forbidden for name in imported))
        with self.assertRaisesRegex(RuntimeError, "HARNESS_ROBODOJO_RUNTIME"):
            adapter.create_environment()
        previous = os.environ.get("HARNESS_ROBODOJO_RUNTIME")
        os.environ["HARNESS_ROBODOJO_RUNTIME"] = "1"
        try:
            with self.assertRaisesRegex(RuntimeError, "RoboDojo checkout not found"):
                adapter.create_environment(checkout=str(ROOT / "third_party" / "missing-robodojo"))
        finally:
            if previous is None:
                os.environ.pop("HARNESS_ROBODOJO_RUNTIME", None)
            else:
                os.environ["HARNESS_ROBODOJO_RUNTIME"] = previous
        imported = set(sys.modules) - before
        self.assertFalse(any(name.split(".", 1)[0] in forbidden for name in imported))


class PolicyObservationTests(unittest.TestCase):
    def test_older_checkout_reads_without_the_extra_render(self):
        seen = {}

        class Older:
            def get_obs(self, env_idx_list):
                seen["idx"] = env_idx_list
                return [{"color": 1}]

        self.assertEqual(policy_observation(Older()), [{"color": 1}])
        self.assertEqual(seen["idx"], [0])

    def test_newer_checkout_renders_before_the_read(self):
        order = []

        class Newer:
            def render_for_capture(self):
                order.append("render")

            def get_obs(self, env_idx_list):
                order.append("read")
                return [{"color": 2}]

        policy_observation(Newer(), env_idx_list=(0,))
        self.assertEqual(order, ["render", "read"])
