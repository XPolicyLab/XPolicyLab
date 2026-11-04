"""Real-file input integrity fixtures and owned-process cleanup exit propagation."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from XPolicyLab.policy.EmbodiedEvoRepro.frozen_inputs import (
    NATIVE_BUDGET_WORDING, WORDING_PATHS, validate_source_inputs, verify_pinned_inputs,
)
from XPolicyLab.policy.EmbodiedEvoRepro.model import Model

ADAPTER = Path(__file__).resolve().parents[1]
MODULE = "XPolicyLab.policy.EmbodiedEvoRepro.model"
CFG = {"bench_name": "RoboDojo", "task_name": "stack_bowls", "env_cfg_type": "arx_x5",
       "action_type": "ee", "seed": 1, "protocol": "ws", "num_envs": 1, "eval_batch": False}


def file_hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file() and p != root / "freeze.json"}


def frozen_source(root):
    source = root / "source"
    files = {
        "inputs/test.md": "There is no action-count budget.\nKeep all task goals unchanged.\n",
        "inputs/primitives/primitives.md": "Native actions have no step-count limit;\nstep(action)\n",
        "inputs/scripts/env.py": "def get_observation():\n    return 'fixture'\n",
        "inputs/submission/solution.py": "# Frozen initial submission\n",
        "frozen_harness/skills/example.md": "Frozen fixture skill\n",
    }
    for name, content in files.items():
        p = source / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    state = {"inputs": file_hashes(source / "inputs"),
             "definition": {"config": {"agent": {"isolation": "docker"}}}}
    (source / "protocol.json").write_text(json.dumps(state))
    provision = root / "vendor/robodojo/.embodiedevo-provisioned.json"
    provision.parent.mkdir(parents=True)
    provision.write_text("{}")
    return source, state


def pinned_copy(source, run):
    """Fixture with the pinned upstream's documented two-file transformations."""
    run.mkdir()
    shutil.copytree(source / "inputs", run / "inputs")
    shutil.copytree(source / "frozen_harness", run / "frozen_harness")
    changes = []
    for name in WORDING_PATHS:
        before = (source / "inputs" / name).read_text()
        after = before
        for old, new in NATIVE_BUDGET_WORDING:
            after = after.replace(old, new)
        (run / "inputs" / name).write_text(after)
        changes.append({"path": name, "before": before, "after": after})
    (run / "protocol-description-changes.json").write_text(json.dumps(changes))


class FrozenInputTests(unittest.TestCase):
    def test_missing_or_empty_protocol_input_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, _ = frozen_source(Path(temporary))
            for state in ({}, {"inputs": {}}, {"inputs": []}):
                with self.subTest(state=state), self.assertRaisesRegex(ValueError, "nonempty"):
                    validate_source_inputs(source, state, file_hashes)

    def test_modified_prompt_rpc_script_or_submission_never_reaches_pin(self):
        for relative in ("test.md", "scripts/env.py", "submission/solution.py"):
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source, _ = frozen_source(root)
                (source / "inputs" / relative).write_text("changed after freeze\n")
                run = root / "202610270000_new"
                pin = Mock(side_effect=AssertionError("pin must not be reached"))
                modules = {
                    "integrations.robodojo_policy.run": SimpleNamespace(pin=pin),
                    "integrations.robodojo_policy.policy.EmbodiedEvo.model": SimpleNamespace(Model=Mock()),
                    "orchestrator.files": SimpleNamespace(hashes=file_hashes),
                }
                with patch.dict(os.environ, {
                    "EVAL_ENV_TYPE": "sim", "EMBODIEDEVO_INTERFACE_PROBE": "0",
                    "EMBODIEDEVO_SOURCE_RUN": str(source), "EMBODIEDEVO_RUN_DIR": str(run),
                    "EMBODIEDEVO_EPISODES": "1", "EMBODIEDEVO_AGENT_RECOVERY": "",
                }), patch.dict(sys.modules, modules), \
                        patch(MODULE + ".verified_upstream", return_value=root), \
                        patch(MODULE + ".get_robot_action_dim_info", return_value={"arm_dim": [6, 6], "ee_dim": [1, 1]}):
                    with self.assertRaisesRegex(ValueError, "source inputs differ"):
                        Model(CFG)
                pin.assert_not_called()
                self.assertFalse(run.exists())

    def test_output_must_not_be_within_frozen_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, _ = frozen_source(root)
            before = file_hashes(source)
            for run in (source / "inputs/202610270000_new", source / "frozen_harness/202610270000_new"):
                with self.subTest(run=run), patch.dict(os.environ, {
                    "EVAL_ENV_TYPE": "sim", "EMBODIEDEVO_INTERFACE_PROBE": "0",
                    "EMBODIEDEVO_SOURCE_RUN": str(source), "EMBODIEDEVO_RUN_DIR": str(run),
                }), patch(MODULE + ".verified_upstream", return_value=root), \
                        patch(MODULE + ".get_robot_action_dim_info", return_value={"arm_dim": [6, 6], "ee_dim": [1, 1]}):
                    with self.assertRaisesRegex(ValueError, "outside the frozen source"):
                        Model(CFG)
                self.assertFalse(run.exists())
                self.assertEqual(before, file_hashes(source))

    def test_only_recorded_native_budget_wording_is_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, state = frozen_source(root)
            run = root / "run"
            pinned_copy(source, run)
            self.assertEqual(verify_pinned_inputs(source, run, state, file_hashes), file_hashes(run / "inputs"))

    def test_modified_copied_input_is_rejected_even_with_forged_receipt(self):
        for relative in ("test.md", "scripts/env.py", "submission/solution.py"):
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source, state = frozen_source(root)
                run = root / "run"
                pinned_copy(source, run)
                (run / "inputs" / relative).write_text("tampered copied input\n")
                receipt = run / "protocol-description-changes.json"
                changes = json.loads(receipt.read_text())
                if relative == "test.md":
                    changes[0]["after"] = "tampered copied input\n"
                    receipt.write_text(json.dumps(changes))
                with self.assertRaisesRegex(ValueError, "beyond the allowed"):
                    verify_pinned_inputs(source, run, state, file_hashes)

    def test_change_receipt_must_match_independent_verification(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, state = frozen_source(root)
            run = root / "run"
            pinned_copy(source, run)
            (run / "protocol-description-changes.json").write_text("[]")
            with self.assertRaisesRegex(ValueError, "receipt"):
                verify_pinned_inputs(source, run, state, file_hashes)


class CleanupExitTests(unittest.TestCase):
    def run_fixture(self, server_status, client_status=0, early_exit=False):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            policy = root / "policy/EmbodiedEvoRepro"
            utils = root / "utils"
            policy.mkdir(parents=True)
            utils.mkdir()
            for name in ("eval.sh", "shell_common.sh"):
                shutil.copy2(ADAPTER / name, policy / name)
            (utils / "get_free_port.sh").write_text("echo 12345\n")
            (utils / "wait_for_policy_server.sh").write_text(
                'for i in {1..100}; do [[ -f "$FIXTURE_READY" ]] && exit 0; sleep .01; done\nexit 1\n')
            (policy / "setup_eval_policy_server.sh").write_text(
                "trap 'exit \"${FIXTURE_SERVER_STATUS}\"' TERM\n"
                'touch "$FIXTURE_READY"\n'
                '[[ "$FIXTURE_EARLY_EXIT" == 1 ]] && exit "$FIXTURE_SERVER_STATUS"\n'
                'while :; do sleep .02; done\n')
            (policy / "setup_eval_env_client.sh").write_text(
                'sleep .05\nexit "$FIXTURE_CLIENT_STATUS"\n')
            environment = {**os.environ, "EVAL_ENV_TYPE": "debug", "EMBODIEDEVO_INTERFACE_PROBE": "1",
                           "FIXTURE_READY": str(root / "ready"), "FIXTURE_SERVER_STATUS": str(server_status),
                           "FIXTURE_CLIENT_STATUS": str(client_status), "FIXTURE_EARLY_EXIT": str(int(early_exit))}
            return subprocess.run(
                ["bash", str(policy / "eval.sh"), "RoboDojo", "stack_bowls", "probe", "arx_x5", "ee", "0", "0", "0", "fake", "fake"],
                env=environment, capture_output=True, text=True, timeout=5,
            )

    def test_successful_client_preserves_shutdown_failure(self):
        result = self.run_fixture(server_status=7)
        self.assertEqual(result.returncode, 7, result.stdout + result.stderr)
        self.assertIn("cleanup/exit failed", result.stderr)

    def test_already_exited_server_is_waited_and_failure_is_preserved(self):
        result = self.run_fixture(server_status=9, early_exit=True)
        self.assertEqual(result.returncode, 9, result.stdout + result.stderr)

    def test_client_failure_takes_precedence_over_cleanup_failure(self):
        result = self.run_fixture(server_status=7, client_status=5)
        self.assertEqual(result.returncode, 5, result.stdout + result.stderr)

    def test_clean_client_and_server_exit_zero(self):
        result = self.run_fixture(server_status=0)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
