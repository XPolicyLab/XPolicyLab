"""Playground freeze/export and the outer Test loader, with no model or simulator."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

from export_workspace import export_workspace
from orchestrator.config import load_task
from orchestrator.files import hashes, write_json
from orchestrator import session_runner


class ExportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run = self.root / "202610140000_playground"
        self.output = self.root / "checkpoint"
        self.cfg = load_task("RoboDojo", env_task="stack_bowls")
        self.policy = Path(__file__).resolve().parents[2]
        self.enterContext(patch.dict(os.environ, EMBODIEDEVO_RUN_DIR=""))

    def episode(self, cfg, episode, experiment, *, phase, **kwargs):
        self.assertEqual(phase, "playground")
        workspace = session_runner.create_workspace(cfg, episode, experiment / "inputs", phase=phase)
        (workspace / "skills/helper.py").write_text("def learned():\n    return 7\n")
        (workspace / "lessons/contact.md").write_text("Check contact before lifting.\n")
        (workspace / "submission/solution.py").write_text("playground_only = 42\n")
        (workspace / "observations/private.txt").write_text("old frame")
        (workspace / "scratch/private.txt").write_text("old plan")
        write_json(episode / "timing.json", {"simulator_ready_at": 1, "agent_started_at": 2})
        result = {"valid": True, "task_completed": False}
        write_json(episode / "result.json", result)
        return result

    def explore(self, *, recipe=True):
        with patch.dict(os.environ, EMBODIEDEVO_USE_RECIPE=str(int(recipe))), \
                patch.object(session_runner, "run_episode", side_effect=self.episode):
            self.assertEqual(session_runner.explore(self.cfg, self.run), 0)

    def test_fresh_test_loads_only_verified_learned_knowledge(self):
        from XPolicyLab.policy.EmbodiedEvo.runtime.workspace import prepare_run, create_workspace, tree_hashes

        self.explore()
        before = hashes(self.run)
        self.assertEqual(export_workspace(self.run, self.output), self.output)
        self.assertEqual(hashes(self.run), before)
        manifest = json.loads((self.output / "experience-manifest.json").read_text())
        expected = manifest["tasks"]["stack_bowls"]["files"]
        self.assertEqual(set(expected), {
            "skills/helper.py", "lessons/contact.md", "skills/example.md", "lessons/example.md",
        })
        self.assertNotIn("playground_only", (self.output / "submission/solution.py").read_text())
        cfg = yaml.safe_load((self.policy / "deploy.yml").read_text())
        cfg.update(ckpt_name=str(self.output), diagnostic=True, task_name="stack_bowls",
                   run_dir=str(self.root / "test"))
        resolved, frozen = prepare_run(cfg)
        self.assertEqual(frozen, expected)
        workspace = create_workspace(Path(resolved["run_dir"]) / "episode",
                                     Path(resolved["run_dir"]) / "inputs",
                                     Path(resolved["run_dir"]) / "frozen_harness")
        self.assertEqual(tree_hashes(workspace, components=("skills", "lessons")), expected)
        self.assertEqual(list((workspace / "observations").iterdir()), [])
        self.assertEqual(list((workspace / "scratch").iterdir()), [])
        self.assertIn("TASK RECIPE BEGIN", (workspace / "instruction.md").read_text())
        self.assertFalse((workspace / "playground.md").exists())

    def test_recipe_free_playground_stays_recipe_free_in_test(self):
        from XPolicyLab.policy.EmbodiedEvo.runtime.workspace import task_inputs

        self.explore(recipe=False)
        export_workspace(self.run, self.output)
        self.assertNotIn(b"TASK RECIPE BEGIN", task_inputs(self.output, "stack_bowls")["test.md"])

    def test_refuses_changed_frozen_files_and_existing_exports(self):
        self.explore()
        export_workspace(self.run, self.output)
        with self.assertRaises(FileExistsError):
            export_workspace(self.run, self.output)
        (self.run / "frozen_harness/skills/helper.py").write_text("changed = True\n")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            export_workspace(self.run, self.root / "tampered")
        self.assertFalse((self.root / "tampered").exists())

    def test_refuses_incomplete_or_linked_knowledge(self):
        self.explore()
        protocol = json.loads((self.run / "protocol.json").read_text())
        protocol["playground_valid"] = False
        write_json(self.run / "protocol.json", protocol)
        with self.assertRaisesRegex(ValueError, "completed, valid, frozen"):
            export_workspace(self.run, self.output)
        protocol["playground_valid"] = True
        write_json(self.run / "protocol.json", protocol)
        (self.run / "frozen_harness/skills/link").symlink_to(self.root)
        with self.assertRaisesRegex(ValueError, "symlinks"):
            export_workspace(self.run, self.output)
        self.assertFalse(self.output.exists())

    def test_launcher_preserves_caller_relative_paths(self):
        self.explore()
        result = subprocess.run([
            "bash", str(self.policy / "playground.sh"), "export", "--run-dir", self.run.name,
            "--output=relative-export",
        ], cwd=self.root, env={**os.environ, "EMBODIEDEVO_PLAYGROUND_PYTHON": sys.executable},
            capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.root / "relative-export/skills/stack_bowls/helper.py").is_file())


if __name__ == "__main__":
    unittest.main()
