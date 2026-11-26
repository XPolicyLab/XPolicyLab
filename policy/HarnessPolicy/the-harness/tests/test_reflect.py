from pathlib import Path
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from harness_policy.runtime.driver import TaskConfig, attach_library
from harness_policy.runtime.reflect import parse_candidate, summarize_episode


class ParseCandidateTests(unittest.TestCase):
    def test_note_skill_reactive_and_none(self):
        self.assertEqual(parse_candidate('<note name="reach_first">Check reach before moving.</note>'),
                         ("notes", "reach_first.md", "Check reach before moving.\n"))
        skill = parse_candidate('<skill name="lift">\n```python\ndef lift(arm):\n    """Lift."""\n'
                                '    return arm\n```\n</skill>')
        self.assertEqual(skill[:2], ("skills", "lift.py"))
        self.assertTrue(skill[2].startswith("def lift(arm):"))
        self.assertEqual(parse_candidate('<reactive name="empty">def empty(obs):\n    """Empty grasp."""\n'
                                         '    return obs["left_gripper"] < 0.2</reactive>')[0], "reactive")
        self.assertIsNone(parse_candidate("<none/>"))

    def test_rejects_bad_code_and_names(self):
        self.assertIsNone(parse_candidate('<skill name="f">def f(:\n</skill>'))
        self.assertIsNone(parse_candidate('<skill name="f">def f():\n    return 1\n</skill>'))  # no docstring
        self.assertIsNone(parse_candidate('<skill name="Bad-Name">def f():\n    """x"""\n</skill>'))
        self.assertIsNone(parse_candidate('<note name="long">' + "x" * 700 + "</note>"))

    def test_a_memory_entry_is_text(self):
        entry = parse_candidate('<memory name="held_by_check">Situation: pick said held, verify said NO. '
                                'What worked: trust the pick check.</memory>')
        self.assertEqual(entry[:2], ("memory", "held_by_check.md"))
        self.assertIsNotNone(parse_candidate('<memory name="long_one">' + "x" * 1000 + "</memory>"))
        self.assertIsNone(parse_candidate('<memory name="too_long">' + "x" * 1300 + "</memory>"))


class SummaryAndAttachTests(unittest.TestCase):
    def test_summary_and_prompt(self):
        with tempfile.TemporaryDirectory() as root:
            run = Path(root) / "run"
            run.mkdir()
            events = [
                {"event": "reset", "observation": {"instruction": "Stack the bowls."}},
                {"event": "model_reply", "call": 1, "text": "<python>move_eef(...)</python>"},
                {"event": "cell", "result": {"status": "escalated", "stdout": "", "escalation": {
                    "event": {"kind": "SKILL_ERROR", "source": "move_eef", "detail": {"reason": "no IK"}}}}},
            ]
            (run / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
            (run / "summary.json").write_text(json.dumps({"success": False, "reward": 0.2, "status": "finished"}))
            text = summarize_episode(run)
            self.assertIn("Instruction: Stack the bowls.", text)
            self.assertIn("SKILL_ERROR from move_eef", text)
            self.assertIn("score=0.2", text)

            library = Path(root) / "lib"
            (library / "notes").mkdir(parents=True)
            (library / "notes" / "a.md").write_text("Lesson A.")
            config = TaskConfig(Path("."), {"endpoint": "x"}, {"model": "m"},
                                       {"layers": {"learned": False, "reactive": True, "handlers": True}},
                                       "BASE\n", "stack_bowls", [0], 3, 60.0)
            first = attach_library(config, library)
            self.assertTrue(first.system_prompt.startswith("BASE\n"))
            self.assertIn("Lesson A.", first.system_prompt)
            (library / "notes" / "b.md").write_text("Lesson B.")
            second = attach_library(first, library)
            self.assertEqual(second.system_prompt.count("Lesson A."), 1)  # built on the base prompt
            self.assertNotEqual(first.runtime["library"]["manifest"]["version"],
                                second.runtime["library"]["manifest"]["version"])
            self.assertNotIn("library", config.runtime)


if __name__ == "__main__":
    unittest.main()
