"""Split entries resolve physical devices before CUDA or Kit imports."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from rsi.gpu_resources import cuda_visibility, configure_split_devices

ROOT = Path(__file__).resolve().parents[2]


class SplitDeploymentTests(unittest.TestCase):
    def test_explicit_conda_names_take_precedence_over_evaluator_variable(self):
        with tempfile.TemporaryDirectory() as folder:
            conda = Path(folder) / 'conda'
            conda.write_text('#!' + sys.executable + '\nimport sys,json;print(json.dumps(sys.argv[1:]))\n')
            conda.chmod(0o700)
            environment = dict(os.environ, PATH=folder + os.pathsep + os.environ.get('PATH', ''),
                               RIEMANN_EVAL_PYTHON='/must-not-select/evaluator/bin/python')
            common = ['RoboDojo', 'fold_clothes', 'rule', 'arx_x5', 'joint', '0', '4']
            cases = [
                ('setup_eval_policy_server.sh', [*common, 'explicit-policy', '45551'], 'explicit-policy'),
                ('setup_eval_env_client.sh', [*common, 'explicit-evaluator', 'probe', '45551'], 'explicit-evaluator'),
                ('eval.sh', [*common, '5', 'explicit-policy', 'explicit-evaluator'], 'explicit-policy'),
            ]
            for entry, arguments, expected in cases:
                with self.subTest(entry=entry):
                    output = subprocess.check_output(['bash', str(ROOT / entry), *arguments],
                                                     cwd=folder, env=environment, text=True)
                    self.assertEqual(json.loads(output)[:5],
                                     ['run', '--no-capture-output', '-n', expected, 'python'])

    def test_numeric_devices_map_to_uuids_in_requested_order(self):
        inventory = [{'index': 5, 'uuid': 'GPU-five'}, {'index': 2, 'uuid': 'GPU-two'}]
        self.assertEqual(cuda_visibility([2, 5], inventory), 'GPU-two,GPU-five')
        with patch('subprocess.check_output', return_value='5, GPU-five\n2, GPU-two\n') as query:
            self.assertEqual(cuda_visibility('2,5', query=query), 'GPU-two,GPU-five')
            self.assertIn('--query-gpu=index,uuid', query.call_args.args[0])

    def test_ambiguous_missing_or_duplicate_devices_fail_before_startup(self):
        for selected in ('', '0,0', '-1', 'GPU-saved', '0, 1'):
            with self.subTest(selected=selected), self.assertRaises(ValueError):
                cuda_visibility(selected, [])
        for inventory in ([{'index': 0, 'uuid': 'GPU-a'}, {'index': 0, 'uuid': 'GPU-b'}],
                          [{'index': 0, 'uuid': 'GPU-a'}, {'index': 1, 'uuid': 'GPU-a'}],
                          [{'index': 0, 'uuid': 'invalid'}], []):
            with self.subTest(inventory=inventory), self.assertRaises(ValueError):
                cuda_visibility([0], inventory)

    def test_split_renderer_uses_selected_uuid_and_local_native_libraries(self):
        environment = {}
        with patch('rsi.gpu_resources.cuda_visibility', return_value='GPU-four'), \
                patch('rsi.deployment.apply_native_libraries') as native, \
                patch('rsi.graphics_runtime.configure') as renderer:
            self.assertEqual(configure_split_devices('4', renderer=True,
                                                    environment=environment), 'GPU-four')
            self.assertEqual(environment['CUDA_VISIBLE_DEVICES'], 'GPU-four')
            self.assertEqual(environment['RIEMANN_PHYSICAL_ENV_GPU'], '4')
            native.assert_called_once_with(environment)
            renderer.assert_called_once_with(environment, 'GPU-four')

    def test_policy_two_device_mapping_does_not_probe_renderer(self):
        with patch('rsi.gpu_resources.cuda_visibility', return_value='GPU-five,GPU-six'), \
                patch('rsi.deployment.apply_native_libraries'), \
                patch('rsi.graphics_runtime.configure') as renderer:
            environment = {}
            configure_split_devices('5,6', environment=environment)
            self.assertEqual(environment['CUDA_VISIBLE_DEVICES'], 'GPU-five,GPU-six')
            renderer.assert_not_called()
            with self.assertRaisesRegex(ValueError, 'One simulator GPU'):
                configure_split_devices('5,6', renderer=True, environment={})

    def test_shell_entries_forward_selection_to_python_bootstrap(self):
        with tempfile.TemporaryDirectory() as folder:
            python = Path(folder) / 'python'
            python.write_text('#!' + sys.executable + '\nimport sys,json;print(json.dumps(sys.argv[1:]))\n')
            python.chmod(0o700)
            common = ['RoboDojo', 'fold_clothes', 'rule', 'arx_x5', 'joint', '0', '4', str(python)]
            for name, suffix in (('setup_eval_policy_server.sh', ['45551']),
                                 ('setup_eval_env_client.sh', ['probe', '45551'])):
                with self.subTest(entry=name):
                    output = subprocess.check_output(['bash', str(ROOT / name), *common, *suffix],
                                                     cwd=folder, text=True)
                    arguments = json.loads(output)
                    self.assertEqual(arguments[arguments.index('--physical-gpus') + 1], '4')
                    if name == 'setup_eval_env_client.sh':
                        self.assertNotIn('--bench_name', arguments)

    def test_split_client_uses_evaluator_ffmpeg_and_checks_it_before_kit(self):
        from rsi.deployment import evaluator_path
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); binary = root / 'bin'; binary.mkdir()
            environment = {'PATH': '/unrelated/system/bin'}
            with self.assertRaisesRegex(RuntimeError, 'ffmpeg'):
                evaluator_path(environment, binary / 'python')
            executable = binary / 'ffmpeg'; executable.write_text('#!/bin/sh\nexit 0\n'); executable.chmod(0o700)
            self.assertTrue(evaluator_path(environment, binary / 'python').startswith(str(binary)))
            self.assertEqual(evaluator_path(environment, binary / 'python').split(':').count(str(binary)), 1)


if __name__ == '__main__':
    unittest.main()
