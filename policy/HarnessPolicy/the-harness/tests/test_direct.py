from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from harness_policy.runtime.direct import ActionError, action_program
from harness_policy.runtime.protocol import DIRECT_KINDS, ProtocolError, parse_reply


class DirectActionTests(unittest.TestCase):
    def test_actions_compile(self):
        for text in ('{"type": "move", "arm": "left_arm", "position": [0, -0.1, 0.9]}',
                     '{"type": "move", "arm": "right_arm", "position": [0.1, 0, 0.85],'
                     ' "orientation": [0.707, 0, 0, 0.707], "gripper": 0}',
                     '{"type": "gripper", "arm": "right_arm", "value": 0.5}',
                     '{"type": "observe"}', '{"type": "wait", "seconds": 2}'):
            compile(action_program(text), "<action>", "exec")
        self.assertIn("close_gripper('right_arm')", action_program(
            '{"type": "gripper", "arm": "right_arm", "value": 0}'))

    def test_invalid_actions(self):
        for text in ("not json", "[]", '{"type": "fly"}', '{"type": "move", "arm": "left_arm"}',
                     '{"type": "move", "arm": "third_arm", "position": [0, 0, 1]}',
                     '{"type": "gripper", "arm": "left_arm", "value": 2}',
                     '{"type": "wait", "seconds": 60}'):
            with self.assertRaises(ActionError):
                action_program(text)

    def test_policy_action_only_with_learned_skills(self):
        text = '{"type": "policy", "name": "g05", "instruction": "stack the bowls", "arms": ["left_arm"]}'
        with self.assertRaises(ActionError):
            action_program(text)
        program = action_program(text, allow_policy=True)
        compile(program, "<action>", "exec")
        self.assertIn("run_policy(\"g05\", \"stack the bowls\", arms=['left_arm'], max_seconds=8.0, from_anywhere=True)", program)
        with self.assertRaises(ActionError):
            action_program('{"type": "policy", "name": "g05", "instruction": ""}', allow_policy=True)
        # Without an instruction the policy gets the task's own.
        program = action_program('{"type": "policy", "name": "g05"}', allow_policy=True)
        compile(program, "<action>", "exec")
        self.assertIn('run_policy("g05", None,', program)
        # Without a name, the agent's first policy (R1's prompt names none).
        program = action_program('{"type": "policy", "seconds": 5}', allow_policy=True)
        compile(program, "<action>", "exec")
        self.assertIn("run_policy(list_policies()[0]['name'], None,", program)
        with self.assertRaises(ActionError):
            action_program('{"type": "policy", "name": ""}', allow_policy=True)

    def test_skills_only_with_the_skill_vocabulary(self):
        texts = ('{"type": "locate", "object": "red cup"}',
                 '{"type": "pick", "object": "red cup"}',
                 '{"type": "pick", "object": "red cup", "arm": "right_arm"}',
                 '{"type": "place", "arm": "left_arm", "target": "blue plate"}',
                 '{"type": "place", "arm": "left_arm", "position": [0, -0.1, 0.78], "height": 0.05}')
        for text in texts:
            with self.assertRaises(ActionError):
                action_program(text)
            compile(action_program(text, allow_skills=True), "<action>", "exec")
        for text in ('{"type": "pick", "object": ""}', '{"type": "pick", "object": "cup", "arm": "third_arm"}',
                     '{"type": "place", "arm": "left_arm"}',
                     '{"type": "place", "arm": "left_arm", "target": "plate", "position": [0, 0, 1]}',
                     '{"type": "place", "arm": "left_arm", "target": "plate", "clearance": 1}'):
            with self.assertRaises(ActionError):
                action_program(text, allow_skills=True)

    def test_pick_and_place_call_the_session_primitives(self):
        calls = []
        state = {"joint_position": [0.0] * 14, "left_ee_pose": [0, -0.2, 0.95, 1, 0, 0, 0],
                 "right_ee_pose": [0.3, -0.2, 0.95, 1, 0, 0, 0]}
        namespace = {"pick": lambda *args, **kwargs: calls.append(("pick", args, kwargs)) or {"ok": True},
                     "place": lambda *args, **kwargs: calls.append(("place", args, kwargs)) or {"ok": True},
                     "get_state": lambda: state, "fingertip_of": lambda pose: pose[:3]}
        exec(action_program('{"type": "pick", "object": "red cup"}', allow_skills=True), dict(namespace))
        exec(action_program('{"type": "place", "arm": "left_arm", "position": [0.1, 0.0, 0.78]}',
                            allow_skills=True), dict(namespace))
        exec(action_program('{"type": "place", "arm": "left_arm", "target": "plate", "clearance": 0.02}',
                            allow_skills=True), dict(namespace))
        self.assertEqual(calls, [("pick", ("red cup",), {"arm": None}),
                                 ("place", ([0.1, 0.0, 0.78], "left_arm"), {"clearance": 0.01}),
                                 ("place", ("plate", "left_arm"), {"clearance": 0.02})])

    def test_direct_protocol(self):
        reply = parse_reply('go <action>{"type": "observe"}</action>', DIRECT_KINDS)
        self.assertEqual((reply.kind, reply.body), ("action", '{"type": "observe"}'))
        with self.assertRaises(ProtocolError):
            parse_reply("<python>x = 1</python>", DIRECT_KINDS)


if __name__ == "__main__":
    unittest.main()
