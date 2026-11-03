from pathlib import Path
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from harness_policy.runtime.chat import ScriptedChat
from harness_policy.runtime.driver import TaskConfig, run_program_episode
from harness_policy.runtime.library import Library

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_program_driver import FakeEnvironment  # noqa: E402


def make_library(root: Path) -> Path:
    (root / "notes").mkdir(parents=True)
    (root / "skills").mkdir()
    (root / "reactive").mkdir()
    (root / "notes" / "01_reach.md").write_text("Bowls near y = +0.1 need the FORWARD orientation.\n")
    (root / "skills" / "lift.py").write_text(
        'def lift(amount=0.05):\n    """Raise a value by amount."""\n    return 1 + amount\n')
    (root / "reactive" / "guards.py").write_text(
        'def empty_grasp(obs):\n    """True when the left gripper closed on nothing."""\n    return obs["left_gripper"] < 0.2\n')
    return root


class LibraryTests(unittest.TestCase):
    def test_version_changes_with_content_and_layers_filter_entries(self):
        with tempfile.TemporaryDirectory() as root:
            path = make_library(Path(root) / "lib")
            full = Library(path)
            version = full.version()
            self.assertEqual(version, Library(path).version())
            self.assertIn("lift(amount)  Raise a value by amount.", full.api_summary(reactive_enabled=True))
            self.assertIn("empty_grasp(obs)", full.api_summary(reactive_enabled=True))
            self.assertNotIn("empty_grasp", full.api_summary(reactive_enabled=False))
            self.assertIn("FORWARD orientation", full.notes())
            notes_only = Library(path, layers=("notes",))
            self.assertEqual(notes_only.kernel_code(True), "")
            self.assertNotEqual(notes_only.version(), version)
            (path / "skills" / "lift.py").write_text("def lift():\n    return 2\n")
            self.assertNotEqual(Library(path).version(), version)
            with self.assertRaises(ValueError):
                Library(path, layers=("memory",))

    def test_driver_loads_library_and_rejects_changes(self):
        with tempfile.TemporaryDirectory() as root:
            path = make_library(Path(root) / "lib")
            library = Library(path)
            runtime = {"bootstrap": "harness_policy.runtime.kernel:PlainSession", "cameras": ["cam_head"],
                       "layers": {"learned": False, "reactive": True, "handlers": True},
                       "library": {"path": str(path), "manifest": library.manifest()}}
            config = TaskConfig(
                path=Path("."), environment={"endpoint": "http://127.0.0.1:1"}, main={"model": "scripted"},
                runtime=runtime, system_prompt="system", task="stack_bowls", seeds=[0], max_model_calls=3,
                max_wall_seconds=60)
            replies = ["<python>\nprint(lift(0.5), empty_grasp({'left_gripper': 0.1}))\n</python>", "<finish>x</finish>"]
            summary = run_program_episode(config, 0, Path(root) / "run", chat=ScriptedChat(replies),
                                          verifier=ScriptedChat([]), environment=FakeEnvironment())
            self.assertEqual(summary["status"], "finished")
            events = [json.loads(line) for line in (Path(root) / "run" / "events.jsonl").read_text().splitlines()]
            cell = next(e for e in events if e["event"] == "cell")
            self.assertEqual(cell["result"]["stdout"], "1.5 True\n")
            (path / "notes" / "02_new.md").write_text("changed")
            summary = run_program_episode(config, 0, Path(root) / "run2", chat=ScriptedChat(replies),
                                          verifier=ScriptedChat([]), environment=FakeEnvironment())
            self.assertEqual(summary["status"], "error")
            self.assertIn("changed after the configuration was loaded", summary["error"])


if __name__ == "__main__":
    unittest.main()
