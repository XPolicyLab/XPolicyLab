from pathlib import Path
import sys
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from harness_policy.runtime.kernel import Kernel
from harness_policy.runtime.protocol import ProtocolError, parse_reply
from harness_policy.runtime.reactive import ReactiveFault, ReactiveLayer, call_reactive
from harness_policy.runtime.shield import PASS, REJECTED, Shield, ShieldProfile


class ProtocolTests(unittest.TestCase):
    def test_one_block_with_thought(self):
        reply = parse_reply("Grasp the cup first.\n<python>\nx = 1\nprint(x)\n</python>")
        self.assertEqual(reply.kind, "python")
        self.assertEqual(reply.thought, "Grasp the cup first.")
        self.assertEqual(reply.body, "x = 1\nprint(x)")

    def test_fenced_python_is_unwrapped(self):
        reply = parse_reply("<python>\n```python\nif True:\n    y = 2\n```\n</python>")
        self.assertEqual(reply.body, "if True:\n    y = 2")

    def test_finish_and_verify(self):
        self.assertEqual(parse_reply("done <finish> cup is on plate </finish>").body, "cup is on plate")
        self.assertEqual(parse_reply("<verify>the drawer is closed</verify>").kind, "verify")

    def test_leaked_tool_call_markup_is_dropped(self):
        bar = "\uff5c"
        text = (f"<python>\n# read the policies\nprint(list_policies())\n</{bar}DSML{bar} parameter>\n"
                f"</{bar}DSML{bar} invoke>\n</{bar}DSML{bar} calls>")
        reply = parse_reply(text)
        self.assertEqual((reply.kind, reply.body), ("python", "# read the policies\nprint(list_policies())"))
        closed = parse_reply(f"<python>\nx = 1\n</{bar}DSML{bar} invoke>\n</python>")
        self.assertEqual(closed.body, "x = 1")

    def test_zero_or_two_blocks_are_rejected(self):
        with self.assertRaises(ProtocolError):
            parse_reply("I will think more.")
        with self.assertRaises(ProtocolError):
            parse_reply("<python>a=1</python><finish>x</finish>")
        with self.assertRaises(ProtocolError):
            parse_reply("<python>   </python>")


def _profile(**overrides):
    data = {"groups": {"left": {"joints": [0, 1], "gripper": 2}},
            "max_joint_jump_rad": 0.1, "workspace": {"x": [-1, 1], "y": [-1, 1], "z": [0, 1]},
            "max_cartesian_jump_m": 0.25}
    data.update(overrides)
    return ShieldProfile.from_dict(data, [-1, -1, 0], [1, 1, 1])


def _fk(command):
    return {"left": [command[0], command[1], 0.5]}


class ShieldTests(unittest.TestCase):
    def test_small_command_passes(self):
        result = Shield(_profile()).check([0.05, 0.0, 1.0], [0, 0, 0], _fk)
        self.assertEqual(result.verdict, PASS)
        self.assertEqual(result.command, [0.05, 0.0, 1.0])

    def test_joint_jump_is_rejected_not_gripper(self):
        result = Shield(_profile(max_cartesian_jump_m=None)).check([0.2, -0.2, 1.0], [0, 0, 0])
        self.assertEqual((result.verdict, result.command), (REJECTED, None))
        self.assertEqual(result.reasons[0]["check"], "joint_jump")
        result = Shield(_profile(max_cartesian_jump_m=None)).check([0.05, -0.05, 1.0], [0, 0, 0])
        self.assertEqual((result.verdict, result.command), (PASS, [0.05, -0.05, 1.0]))

    def test_joint_limit_is_rejected_not_clamped(self):
        result = Shield(_profile()).check([0, 0, 3.0], [0, 0, 0])
        self.assertEqual((result.verdict, result.command), (REJECTED, None))
        self.assertEqual(result.reasons[0]["check"], "joint_limit")

    def test_large_cartesian_jump_is_rejected(self):
        profile = _profile(max_joint_jump_rad=1.0)
        result = Shield(profile).check([0.3, 0.0, 0.0], [0, 0, 0], _fk)
        self.assertEqual(result.verdict, REJECTED)
        self.assertIsNone(result.command)
        self.assertEqual(result.reasons[-1]["check"], "cartesian_jump")

    def test_workspace_and_nonfinite_are_rejected(self):
        profile = _profile(workspace={"x": [-0.5, 0.5], "y": [-1, 1], "z": [0, 1]}, max_joint_jump_rad=1.0,
                           max_cartesian_jump_m=None)
        self.assertEqual(Shield(profile).check([0.6, 0, 0], [0, 0, 0], _fk).reasons[-1]["check"], "workspace")
        self.assertEqual(Shield(_profile()).check([float("nan"), 0, 0], [0, 0, 0]).verdict, REJECTED)
        self.assertEqual(Shield(_profile()).check([0, 0], [0, 0, 0]).verdict, REJECTED)


class ReactiveTests(unittest.TestCase):
    def test_when_and_watch_fire_in_order(self):
        layer = ReactiveLayer(budget_seconds=0.01)
        layer.when(lambda obs: obs["width"] < 0.002, "EMPTY_GRASP", name="empty")
        layer.watch("force.left", high=20.0, event="OVERLOAD")
        self.assertIsNone(layer.evaluate({"width": 0.01, "force": {"left": 3.0}}))
        self.assertEqual(layer.evaluate({"width": 0.0, "force": {"left": 30.0}}).action, "EMPTY_GRASP")
        layer.clear("empty")
        guard = layer.evaluate({"width": 0.0, "force": {"left": 30.0}})
        self.assertEqual((guard.action, guard.last_value), ("OVERLOAD", 30.0))

    def test_io_and_overrun_are_faults(self):
        with self.assertRaises(ReactiveFault):
            call_reactive("io", lambda: open(__file__).read(), budget_seconds=0.05)
        with self.assertRaises(ReactiveFault):
            call_reactive("slow", lambda: sum(range(10 ** 8)), budget_seconds=0.001)
        with self.assertRaises(ReactiveFault):
            call_reactive("raises", lambda: 1 / 0, budget_seconds=0.01)
        # Outside reactive code, file access works again.
        with open(__file__) as handle:
            self.assertTrue(handle.read())

    def test_missing_signal_and_bad_event_name(self):
        layer = ReactiveLayer(budget_seconds=0.01)
        layer.watch("torque", high=1.0)
        with self.assertRaises(ReactiveFault):
            layer.evaluate({})
        with self.assertRaises(ValueError):
            layer.when(lambda obs: True, "not an event")


class KernelTests(unittest.TestCase):
    def setUp(self):
        self.kernel = Kernel("harness_policy.runtime.kernel:PlainSession", {})

    def tearDown(self):
        self.kernel.close()

    def test_state_persists_and_output_is_captured(self):
        first = self.kernel.run("import os\nx = 41\nprint('hello')\nos.write(1, b'stray\\n')", timeout=30)
        self.assertEqual((first["status"], first["stdout"]), ("ok", "hello\n"))
        second = self.kernel.run("print(x + 1)", timeout=30)
        self.assertEqual(second["stdout"], "42\n")

    def test_errors_show_program_frames(self):
        result = self.kernel.run("def f():\n    return 1 / 0\nf()", timeout=30)
        self.assertEqual(result["status"], "error")
        self.assertIn('File "<cell-1>", line 3', result["error"])
        self.assertIn("ZeroDivisionError", result["error"])
        self.assertNotIn("kernel.py", result["error"])

    def test_escalation_is_reported(self):
        code = ("from harness_policy.runtime.events import Escalation, Event, ExecutionTrace\n"
                "raise Escalation(ExecutionTrace('move_eef', {}, 0, 5, Event('SHIELD_REJECTED', 'shield')))")
        result = self.kernel.run(code, timeout=30)
        self.assertEqual(result["status"], "escalated")
        self.assertEqual(result["escalation"]["event"]["kind"], "SHIELD_REJECTED")

    def test_timeout_kills_the_kernel(self):
        started = time.perf_counter()
        with self.assertRaises(Exception):
            self.kernel.run("while True: pass", timeout=1)
        self.assertLess(time.perf_counter() - started, 10)
        self.assertIsNotNone(self.kernel.process.poll())


if __name__ == "__main__":
    unittest.main()
