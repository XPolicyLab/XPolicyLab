"""Task loading, frozen experience and episode isolation; no model API required."""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from XPolicyLab.policy.EmbodiedRSI.runtime.workspace import (
    POLICY_DIR, create_workspace, prepare_run, verify_tree,
)
from XPolicyLab.policy.EmbodiedRSI.deploy import PolicyFailure, _call
from XPolicyLab.policy.EmbodiedRSI.runtime.config import config_from_dict
from XPolicyLab.policy.EmbodiedRSI.runtime.agent.session import agent_command


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.checkpoint = self.root / "checkpoint"
        shutil.copytree(POLICY_DIR / "Workspace", self.checkpoint)
        self.config = yaml.safe_load((POLICY_DIR / "deploy.yml").read_text())
        self.config.update(checkpoint_path=str(self.checkpoint), run_dir=str(self.root / "run"),
                           task_name="stack_bowls_random", diagnostic=True)
        self.enterContext(patch.dict(os.environ, EMBODIEDRSI_RUN_DIR=""))

    def test_test_runs_without_global_release_or_learned_experience(self):
        self.config["diagnostic"] = False
        with patch.dict(os.environ, EMBODIEDRSI_CODEX_HOME=str(self.root / "provider")):
            cfg, frozen = prepare_run(self.config)
        self.assertEqual(frozen, {})
        self.assertEqual(cfg["task_config"]["simulator"]["env_task"], "stack_bowls")
        self.assertEqual(cfg["task_config"]["test"]["budget"], 100)
        manifest = json.loads((Path(cfg["run_dir"]) / "workspace-manifest.json").read_text())
        self.assertEqual(manifest["experience"], {})
        self.assertIn("test.md", manifest["inputs"])
        with self.assertRaises(FileExistsError):
            prepare_run({**self.config, "diagnostic": True})

    def test_only_selected_task_is_snapshotted_and_tampering_is_detected(self):
        skill = self.checkpoint / "skills/stack_bowls/reach.py"
        skill.parent.mkdir(parents=True)
        skill.write_text("def reach(): return 1\n")
        unrelated = self.checkpoint / "skills/another_task"
        unrelated.mkdir()
        (unrelated / "link").symlink_to(self.root)
        cfg, frozen = prepare_run(self.config)
        harness = Path(cfg["run_dir"]) / "frozen_harness"
        self.assertEqual(set(frozen), {"skills/reach.py"})
        skill.write_text("changed after snapshot\n")
        verify_tree(harness, frozen)
        (harness / "skills/reach.py").write_text("changed snapshot\n")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            verify_tree(harness, frozen)

    def test_invalid_requests_fail_before_creating_run(self):
        for change in ({"phase": "playground"}, {"task_name": "unknown"},
                       {"task_name": "../stack_bowls"}, {"eval_batch": "true"},
                       {"execution_budget": 0}, {"ckpt_name": "diagnostic", "diagnostic": False}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                prepare_run({**self.config, **change})
            self.assertFalse((self.root / "run").exists())

    def test_checkpoint_symlinks_are_rejected(self):
        path = self.checkpoint / "skills/stack_bowls"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(self.root)
        with self.assertRaisesRegex(ValueError, "linked asset"):
            prepare_run(self.config)

    def test_fresh_workspaces_and_container_readonly_mounts(self):
        spec, _ = prepare_run(self.config)
        cfg = config_from_dict(spec["task_config"])
        run = Path(spec["run_dir"])
        first, second = run / "episodes/episode_0000", run / "episodes/episode_0001"
        for episode in (first, second):
            create_workspace(episode, run / "inputs", run / "frozen_harness")
        (first / "workspace/submission/solution.py").write_text("marker = 'previous episode'\n")
        self.assertNotIn("previous episode", (second / "workspace/submission/solution.py").read_text())
        self.assertFalse((second / "workspace/ledger.jsonl").exists())
        self.assertFalse((second / "workspace/playground.md").exists())
        provider = self.root / "provider"
        provider.mkdir()
        (provider / "config.toml").write_text('model_provider="remote"\n[model_providers.remote]\nname="remote"\nbase_url="https://example.invalid/v1"\nwire_api="responses"\nenv_key="EMBODIEDRSI_TEST_KEY"\n')
        socket_dir = self.root / "socket"
        socket_dir.mkdir()
        module = "XPolicyLab.policy.EmbodiedRSI.runtime.agent.launch"
        with patch.dict(os.environ, EMBODIEDRSI_CODEX_HOME=str(provider), EMBODIEDRSI_TEST_KEY="test-secret"), \
             patch(module + ".require_image"), patch(module + ".ensure_network"):
            _, argv = agent_command(cfg, second, socket_dir, "test")
        self.assertNotIn("test-secret", " ".join(argv))
        self.assertIn("EMBODIEDRSI_TEST_KEY", argv)
        for component in ("skills", "lessons", "instruction.md", "primitives", "scripts", "observations"):
            self.assertTrue(any(f"dst=/workspace/{component},readonly" in arg for arg in argv))

    def test_policy_errors_do_not_become_native_scene_retries(self):
        class BrokenClient:
            def call(self, **kwargs):
                raise RuntimeError("provider unavailable")
        with self.assertRaises(PolicyFailure):
            _call(BrokenClient(), "get_action")
        self.assertFalse(issubclass(PolicyFailure, Exception))


if __name__ == "__main__":
    unittest.main()
