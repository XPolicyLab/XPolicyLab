"""Actual SDK Bash oracle and CPU-only transport/process fixtures."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from rsi.official_client_lifecycle import OfficialClientRestarts, official_restart_contract, wait_client
from rsi.rule_configuration import settings
from tests.contracts import test_official_resume as resume_fixture

ROOT = Path(__file__).resolve().parents[2]


class OfficialClientLifecycle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sdk = Path(settings()['official_root'])
        cls.script = (cls.sdk / 'scripts/eval_policy.sh').read_bytes()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)

    def controller(self, environment=None):
        return OfficialClientRestarts(self.sdk, self.output, environment or {}, 'same-run')

    def fixture(self):
        case = resume_fixture.OfficialResume(methodName='test_normal_launcher_keeps_original_source_and_new_result_namespace')
        case.setUp()
        self.addCleanup(case.doCleanups)
        return case

    def test_script_is_the_separately_pinned_actual_official_source(self):
        lock = json.loads((ROOT / 'configs/official-runtime-inputs.json').read_text())
        self.assertEqual(hashlib.sha256(self.script).hexdigest(), lock['official_files']['scripts/eval_policy.sh'])
        self.assertEqual(official_restart_contract(self.sdk)['official_commit'], lock['official_commit'])

    def test_retry_sequences_match_the_actual_bash_loop(self):
        # Execute the *official* loop, replacing only its Python and sleep
        # commands with inert functions. No simulator or task code is loaded.
        loop = 'MAX_BASH_RETRIES=' + self.script.decode().split('MAX_BASH_RETRIES=', 1)[1]
        for codes, limit in [([99, 0], 10), ([134, 139, 0], 10),
                             ([99] * 9 + [0], 10), ([99] * 10 + [0], 10),
                             ([99] * 3 + [0], 3), ([1], 10), ([137], 10)]:
            with self.subTest(codes=codes, limit=limit):
                preamble = '''set -eu
task_name=x; env_cfg_type=x; num_envs=1; KIT_ARGS=x; device_id=0
policy_name=x; port=1; protocol=ws; policy_server_url=x
additional_info=x; seed=0; host=x; extra_args=(); call_i=0
python() { printf 'call:%s\\n' "$call_i"; local c=${codes[$call_i]}; call_i=$((call_i+1)); return "$c"; }
sleep() { printf 'delay:%s\\n' "$1"; }
'''
                script = preamble + 'codes=(' + ' '.join(map(str, codes)) + ')\n' + loop
                bash = subprocess.run(['bash', '-c', script], text=True, capture_output=True,
                    env=dict(os.environ, ROBODOJO_MAX_BASH_RETRIES=str(limit), ROBODOJO_RUN_ID='same-run'))
                controller = self.controller({'ROBODOJO_MAX_BASH_RETRIES': str(limit)})
                calls = sleeps = 0
                for code in codes:
                    calls += 1
                    if not controller.should_restart(code):
                        terminal = code
                        break
                    sleeps += 1
                self.assertEqual(bash.returncode, terminal)
                self.assertEqual(bash.stdout.count('call:'), calls)
                self.assertEqual(bash.stdout.count('delay:5'), sleeps)

    def test_posix_signal_codes_match_the_shell_codes(self):
        for native, shell in [(-6, 134), (-11, 139)]:
            controller = self.controller()
            self.assertTrue(controller.should_restart(native))
            self.assertEqual(controller.events[0]['shell_returncode'], shell)

    def test_other_failures_never_request_a_retry(self):
        for code in [1, 2, 124, 137, -9]:
            self.assertFalse(self.controller().should_restart(code))

    def test_same_server_lease_command_and_run_id_survive_a_restart(self):
        case = self.fixture()
        with patch('tools.official_launch.time.sleep'):
            calls, output = case._exercise_launcher(False, (99, 0), self.script)
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[1][0], calls[2][0])
        for key in ['cwd', 'env', 'pass_fds', 'start_new_session']:
            self.assertEqual(calls[1][1][key], calls[2][1][key])
        proof = json.loads((output / 'official-client-restarts.json').read_text())
        self.assertEqual(proof['status'], 'finished')
        self.assertFalse(proof['policy_server_restarted'])

    def test_sdk_resume_keeps_the_original_scored_prefix_across_restart(self):
        case = self.fixture()
        with patch('tools.official_launch.time.sleep'):
            calls, output = case._exercise_launcher(True, (99, 0), self.script)
        self.assertEqual(len(calls), 3)
        final = json.loads(case.result_path.read_text())
        for key, value in case.before['details'].items():
            self.assertEqual(final['details'][key], value)
        self.assertEqual(json.loads((output / 'sdk-continuation-input.json').read_text())['official_prefix'], case.before)

    def test_policy_errors_and_timeouts_propagate_without_retries(self):
        for failure in [1, subprocess.TimeoutExpired(['official-client'], 3600)]:
            case = self.fixture()
            expected = subprocess.CalledProcessError if failure == 1 else subprocess.TimeoutExpired
            with self.assertRaises(expected):
                case._exercise_launcher(False, (failure,), self.script, wall_timeout=.01)
            self.assertFalse((case.root / 'new-attempt/official-client-restarts.json').exists())

    def test_attempt_cap_is_durable_before_the_final_error(self):
        controller = self.controller({'ROBODOJO_MAX_BASH_RETRIES': '3'})
        self.assertEqual([controller.should_restart(99) for _ in range(3)], [True, True, False])
        proof = json.loads(controller.path.read_text())
        self.assertEqual(proof['status'], 'attempt_cap_reached')
        self.assertEqual(len(proof['events']), 3)

    def test_changed_official_script_is_rejected(self):
        sdk = self.output / 'sdk'
        (sdk / 'scripts').mkdir(parents=True)
        (sdk / 'scripts/eval_policy.sh').write_bytes(self.script + b'\n')
        with self.assertRaisesRegex(ValueError, 'restart script changed'):
            official_restart_contract(sdk)

    def fatal_fixture(self):
        case = self.fixture()
        expected = {key: case.manifest[key] for key in
                    ('run_id', 'task_name', 'policy_name', 'config_name', 'eval_seed', 'additional_info')}
        controller = OfficialClientRestarts(self.sdk, self.output, {}, case.run_id,
                                            case.result_path, expected=expected, expected_count=3)
        log = self.output / 'client.log'
        log.write_text('[FATAL] PhysX kernel failure detected: native CUDA error; persisted manifest. '
                       'In-process restart attempt 1/3.\nSimulation App Shutting Down\n')
        return case, controller, log

    def test_native_fatal_close_zero_uses_only_matching_sdk_checkpoint(self):
        case, controller, log = self.fatal_fixture()
        self.assertTrue(controller.should_restart(0, client_log=log))
        event = json.loads(controller.path.read_text())['events'][0]
        self.assertTrue(event['fatal_during_kit_close'])
        self.assertEqual(event['returncode'], 0)
        self.assertEqual(event['shell_returncode'], 99)
        self.assertEqual(event['official_prefix'], case.before)

    def test_zero_with_policy_error_or_arbitrary_log_is_never_retried(self):
        _, controller, log = self.fatal_fixture()
        log.write_text('RuntimeError: policy failed\nSimulation App Shutting Down\n')
        self.assertFalse(controller.should_restart(0, client_log=log))
        self.assertFalse(controller.path.exists())

    def test_sdk_self_restart_then_full_completion_is_not_replayed(self):
        case, controller, log = self.fatal_fixture()
        final = resume_fixture.result(dict(case.before['details'],
            **{'2': dict(layout_id=2, success=True, score=1.)}))
        case.result_path.write_text(json.dumps(final))
        case.manifest_path.unlink()  # The official SDK removes it on completion.
        self.assertFalse(controller.should_restart(0, client_log=log))
        self.assertFalse(controller.path.exists())

    def test_zero_fatal_without_checkpoint_or_with_wrong_seed_is_rejected(self):
        case, controller, log = self.fatal_fixture()
        data = dict(case.manifest, eval_seed=1)
        case.manifest_path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'checkpoint identity'):
            controller.should_restart(0, client_log=log)
        case.manifest_path.unlink()
        with self.assertRaisesRegex(ValueError, 'ordinary evidence file'):
            controller.should_restart(0, client_log=log)

    def test_actual_sdk_fatal_function_survives_a_close_that_exits_zero(self):
        _, controller, log = self.fatal_fixture()
        source = (self.sdk / 'src/eval_client/main.py').read_text()
        function = next(node for node in ast.parse(source).body
                        if isinstance(node, ast.FunctionDef) and node.name == '_restart_or_exit')
        # Use the pinned SDK function unchanged. Model only Kit's process-exit
        # behavior in close(); no simulator, task or reward code is imported.
        script = ('import os, sys\nMAX_INPROC_RESTARTS=3\n' + ast.unparse(function) +
                  '\nclass Env:\n def persist_resume_manifest(self, **kwargs): pass\n' +
                  'class Kit:\n def close(self): os._exit(0)\n' +
                  '_restart_or_exit(Env(), Kit(), "test fatal")\n')
        with log.open('wb') as output:
            child = subprocess.run([sys.executable, '-u', '-c', script], stdout=output,
                                   stderr=subprocess.STDOUT, check=True)
        self.assertEqual(child.returncode, 0)
        self.assertTrue(controller.should_restart(0, client_log=log))

    def test_renderer_bootstrap_failure_stops_before_layout_resets(self):
        log = self.output / 'failed-renderer.log'
        script = ('import time; print("Failed to create any GPU devices, including an attempt with compatibility mode.", flush=True); time.sleep(20)')
        with log.open('wb') as stream:
            child = subprocess.Popen([sys.executable, '-u', '-c', script], stdout=stream)
            try:
                with self.assertRaisesRegex(RuntimeError, 'renderer bootstrap failed'):
                    wait_client(child, log, timeout=5)
            finally:
                child.terminate(); child.wait(timeout=5)

    def test_zero_exit_with_incomplete_results_is_not_normal_completion(self):
        case = self.fixture()
        case.route['official_eval_nums'] = 4
        with self.assertRaisesRegex(RuntimeError, 'complete official cohort'):
            case._exercise_launcher(False)

    def test_normal_completion_writes_no_recovery_receipt(self):
        controller = self.controller()
        self.assertFalse(controller.should_restart(0))
        self.assertFalse(controller.path.exists())

    def test_scored_cases_require_a_matching_native_checkpoint_and_remain_unchanged(self):
        case = self.fixture()
        controller = OfficialClientRestarts(self.sdk, self.output, {}, case.run_id, case.result_path)
        manifest = case.manifest_path.read_bytes()
        case.manifest_path.unlink()
        with self.assertRaises(ValueError):
            controller.should_restart(99)
        case.manifest_path.write_bytes(manifest)
        self.assertTrue(controller.should_restart(99))
        changed = dict(case.before['details'])
        changed['1'] = dict(layout_id=1, success=True, score=1.)
        case.result_path.write_text(json.dumps(resume_fixture.result(changed)))
        with self.assertRaisesRegex(ValueError, 'previously scored'):
            controller.should_restart(0)
