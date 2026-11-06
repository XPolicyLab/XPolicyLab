"""SDK-native continuation contracts; process fixtures never initialize GPUs."""
import ast
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rsi.official_resume import inspect_resume, verify_extension

ROOT = Path(__file__).resolve().parents[2]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def result(details):
    return dict(details=details, eval_time=len(details),
                success_rate=sum(d['success'] for d in details.values()) / len(details),
                score=sum(d['score'] for d in details.values()) / len(details) * 100)


class OfficialResume(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.previous = self.root / 'previous'
        self.identity = dict(source_versioning='git', source_commit='a' * 40, source_dirty=False)
        self.args = SimpleNamespace(bench='RoboDojo', task='insert_key', seed='0',
                                    action='joint', checkpoint='rule', robot='arx_x5')
        self.deploy = dict(policy_name='Riemann-RSI-Preview', task_name='insert_key', seed=0,
                           action_type='joint', env_cfg_type='arx_x5', bench_name='RoboDojo',
                           ckpt_name='rule', output_dir=str(self.previous / 'policy'),
                           unchanged_resource_binding='same-original-asset')
        self.route = dict(backend='rule', official_eval_nums=3)
        self.run_id = 'same-run'
        self.parent = self.previous / 'official-client/eval_result/RoboDojo/insert_key/Riemann-RSI-Preview/arx_x5/0_ckpt_name=rule,action_type=joint'
        self.result_path = self.parent / self.run_id / '_result.json'
        self.manifest_path = self.parent / ('_resume_' + self.run_id + '.json')
        self.before = result({'0': dict(layout_id=0, success=True, score=1.),
                              '1': dict(layout_id=1, success=False, score=.4)})
        self.manifest = dict(run_id=self.run_id, task_name='insert_key', policy_name='Riemann-RSI-Preview',
                             config_name='arx_x5', eval_seed=0,
                             additional_info='ckpt_name=rule,action_type=joint',
                             save_dir=str(self.result_path.parent.relative_to(self.previous / 'official-client')),
                             success_nums=1, fail_nums=1, total_score=1.4,
                             completed_layout_ids=[0, 1], abandoned_layout_ids=[],
                             restart_count=0, details=self.before['details'])
        for name in ('source-identity.json', 'source-identity-final.json'):
            write(self.previous / name, self.identity)
        write(self.previous / 'lease.json', dict(node=socket.gethostname(), owner_pid=123))
        write(self.previous / 'deploy.json', self.deploy)
        write(self.result_path, self.before)
        write(self.manifest_path, self.manifest)

    def inspect(self):
        with patch('rsi.official_resume.os.kill', side_effect=ProcessLookupError):
            return inspect_resume(self.previous, self.identity, self.args, self.route, self.run_id)

    def test_checkpoint_keeps_original_failure_success_and_public_config(self):
        proof = self.inspect()
        self.assertEqual(proof['official_prefix'], self.before)
        self.assertEqual(proof['deploy'], self.deploy)
        self.assertNotIn('completed_layout_ids', proof['deploy'])
        self.assertEqual(proof['manifest_sha256'], hashlib.sha256(self.manifest_path.read_bytes()).hexdigest())

    def test_live_or_wrong_host_owner_cannot_be_resumed(self):
        with patch('rsi.official_resume.os.kill', return_value=None):
            with self.assertRaisesRegex(ValueError, 'still live'):
                inspect_resume(self.previous, self.identity, self.args, self.route, self.run_id)
        write(self.previous / 'lease.json', dict(node='different-host', owner_pid=123))
        with self.assertRaisesRegex(ValueError, 'actual compute node'):
            self.inspect()

    def test_source_and_public_deployment_cannot_change(self):
        write(self.previous / 'source-identity-final.json', dict(self.identity, source_commit='b' * 40))
        with self.assertRaisesRegex(ValueError, 'same clean'):
            self.inspect()
        write(self.previous / 'source-identity-final.json', self.identity)
        write(self.previous / 'deploy.json', dict(self.deploy, seed=1))
        with self.assertRaisesRegex(ValueError, 'public deployment'):
            self.inspect()

    def test_missing_or_different_sdk_checkpoint_is_rejected(self):
        write(self.manifest_path, dict(self.manifest, eval_seed=1))
        with self.assertRaisesRegex(ValueError, 'checkpoint identity'):
            self.inspect()
        self.manifest_path.unlink()
        with self.assertRaisesRegex(ValueError, 'ordinary evidence file'):
            self.inspect()

    def test_stale_or_repeated_completion_evidence_is_rejected(self):
        write(self.manifest_path, dict(self.manifest, completed_layout_ids=[0, 0]))
        with self.assertRaisesRegex(ValueError, 'checkpoint and scored'):
            self.inspect()
        write(self.manifest_path, self.manifest)
        write(self.result_path, result({'0': self.before['details']['0'],
                                       '1': dict(layout_id=0, success=False, score=.4)}))
        with self.assertRaisesRegex(ValueError, 'repeated completed'):
            self.inspect()

    def test_complete_cohort_cannot_be_run_again(self):
        self.route['official_eval_nums'] = 2
        with self.assertRaisesRegex(ValueError, 'incomplete official cohort'):
            self.inspect()

    def test_extension_preserves_failed_and_successful_original_scores(self):
        after = result(dict(self.before['details'], **{'2': dict(layout_id=2, success=True, score=1.)}))
        self.assertEqual(verify_extension(self.before, after), 3)
        after['details']['1'] = dict(layout_id=1, success=True, score=1.)
        with self.assertRaises(ValueError):
            verify_extension(self.before, result(after['details']))

    def test_actual_sdk_loader_reads_the_alias_without_rewriting_manifest(self):
        declaration = json.loads((ROOT / 'configs/official-runtime-inputs.json').read_text())
        from rsi.rule_configuration import settings
        path = Path(settings()['official_root']) / 'src/eval_client/main.py'
        source = path.read_bytes()
        self.assertEqual(hashlib.sha256(source).hexdigest(), declaration['official_files']['src/eval_client/main.py'])
        tree = ast.parse(source)
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name in ('_resume_manifest_path', '_load_resume_manifest')]
        self.assertEqual(len(functions), 2)
        namespace = dict(os=os, json=json, BENCHMARK='RoboDojo')
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
        client = self.root / 'new-client'
        client.mkdir()
        (client / 'eval_result').symlink_to(self.previous / 'official-client/eval_result', target_is_directory=True)
        previous_cwd = Path.cwd()
        try:
            os.chdir(client)
            config = dict(task_name='insert_key', policy_name='Riemann-RSI-Preview', config_name='arx_x5',
                          seed=0, additional_info='ckpt_name=rule,action_type=joint')
            self.assertEqual(namespace['_load_resume_manifest'](config, self.run_id), self.manifest)
        finally:
            os.chdir(previous_cwd)

    def _exercise_launcher(self, resume, client_codes=(0,), restart_script=None, wall_timeout=3600):
        from tools import official_launch as launch
        source = self.root / 'unchanged-policy-source'
        source.mkdir()
        sdk = self.root / 'sdk'
        sdk.mkdir()
        if restart_script is not None:
            (sdk / 'scripts').mkdir()
            (sdk / 'scripts/eval_policy.sh').write_bytes(restart_script)
        output = self.root / 'new-attempt'
        calls = []
        codes = iter(client_codes)

        @contextmanager
        def lease(*args, **kwargs):
            yield [{'index': 0, 'uuid': 'GPU-original'}]

        def popen(command, **kwargs):
            calls.append((command, kwargs))
            if Path(command[2]).name == 'serve_official_policy.py':
                kwargs['stdout'].write(json.dumps({'event': 'official_rule_policy_ready', 'mode': 'insert_key'}) + '\n')
                kwargs['stdout'].flush()
                return SimpleNamespace(pid=99999998, poll=lambda: None)
            code = next(codes)
            def wait(timeout=None):
                if isinstance(code, BaseException):
                    raise code
                if resume:
                    self.assertEqual((Path(kwargs['cwd']) / 'eval_result').resolve(),
                                     (self.previous / 'official-client/eval_result').resolve())
                    if code == 0:
                        write(self.result_path, result(dict(self.before['details'], **{'2': dict(layout_id=2, success=True, score=1.)})))
                else:
                    self.assertFalse((Path(kwargs['cwd']) / 'eval_result').exists())
                    if code == 0:
                        target = (Path(kwargs['cwd']) / self.result_path.relative_to(self.previous / 'official-client'))
                        write(target, result(dict(self.before['details'], **{'2': dict(layout_id=2, success=True, score=1.)})))
                return code
            return SimpleNamespace(pid=99999999, poll=lambda: None, wait=wait, args=command)

        argv = ['RoboDojo', 'insert_key', 'rule', 'arx_x5', 'joint', '0', '0', '0', sys.executable, '/eval/python']
        with patch('rsi.policy_catalog.resolve_task', return_value=('insert_key', self.route)), \
             patch('rsi.policy_catalog.runtime_settings', return_value=dict(python='/eval/python', official_root=str(sdk))), \
             patch('rsi.deployment.interpreter_python', return_value='/eval/python') as normalized_eval, \
             patch('rsi.official_source.read', return_value=dict(route=self.route,
                    settings=dict(python='/eval/python', official_root=str(sdk)))), \
             patch('rsi.skills.git_versioning.runtime_source_identity', return_value=self.identity), \
             patch('rsi.graphics_runtime.configure', return_value=dict(renderer_index=0)), \
             patch('rsi.gpu_resources.gpu_leases', lease), patch('rsi.gpu_resources.stop_owned'), \
             patch('rsi.gpu_resources.remember_owned_group'), \
             patch('rsi.official_resume.os.kill', side_effect=ProcessLookupError), \
             patch('rsi.deployment.shutil.which', return_value='/ffmpeg'), \
             patch.object(launch.subprocess, 'Popen', side_effect=popen), \
             patch.dict(os.environ, {'EVAL_NUM': '1', 'EVAL_ENV_TYPE': 'sim'}):
            launch.evaluate(argv, output=output, run_id=self.run_id, timeout=wall_timeout,
                            policy_source=source if resume else None,
                            resume_from=self.previous if resume else None)
            normalized_eval.assert_called_once_with('/eval/python')
        expected_source = source if resume else launch.ROOT
        self.assertEqual(Path(calls[0][0][2]), expected_source / 'tools/serve_official_policy.py')
        self.assertEqual(Path(calls[1][0][2]), launch.ROOT / 'tools/run_official_eval.py')
        for command, options in calls:
            self.assertNotIn('EVAL_NUM', options['env'])
            self.assertEqual(options['env']['ROBODOJO_RUN_ID'], self.run_id)
            self.assertEqual(options['env']['PATH'].split(os.pathsep)[0], '/eval')
        deploy = json.loads((output / 'deploy.json').read_text())
        if resume:
            self.assertEqual(deploy, dict(self.deploy, output_dir=str(output / 'policy')))
            self.assertEqual(json.loads((output / 'sdk-continuation-input.json').read_text())['official_prefix'], self.before)
        else:
            self.assertEqual(deploy['task_name'], 'insert_key')
            self.assertEqual(deploy['seed'], 0)
            self.assertFalse((output / 'sdk-continuation-input.json').exists())
        return calls, output

    def test_launcher_uses_same_actor_source_fresh_diagnostics_and_sdk_namespace(self):
        self._exercise_launcher(True)

    def test_normal_launcher_keeps_original_source_and_new_result_namespace(self):
        self._exercise_launcher(False)
