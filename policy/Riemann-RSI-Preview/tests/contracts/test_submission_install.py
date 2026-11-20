"""Portable policy discovery, local-only locations and isolated install plans."""
from contextlib import redirect_stdout, redirect_stderr
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


class SubmissionInstallTests(unittest.TestCase):
    def test_official_namespace_imports_without_cwd_or_pythonpath(self):
        code = r'''import importlib, importlib.util, pathlib, sys, types
root=pathlib.Path(sys.argv[1]); namespace=pathlib.Path(sys.argv[2]); policy=namespace/'policy'; policy.mkdir(); (policy/'Riemann-RSI-Preview').symlink_to(root, target_is_directory=True)
sdk=types.ModuleType('XPolicyLab'); sdk.__path__=[]
policies=types.ModuleType('XPolicyLab.policy'); policies.__path__=[str(policy)]
template=types.ModuleType('XPolicyLab.model_template'); template.ModelTemplate=object
utils=types.ModuleType('XPolicyLab.utils'); utils.__path__=[]
process=types.ModuleType('XPolicyLab.utils.process_data'); process.get_robot_action_dim_info=lambda *a: {}
demo=types.ModuleType('XPolicyLab.policy.demo_policy'); demo.__path__=[]
loop=types.ModuleType('XPolicyLab.policy.demo_policy.deploy'); loop.eval_one_episode=lambda *a: None
for module in (sdk,policies,template,utils,process,demo,loop): sys.modules[module.__name__]=module
sys.modules['numpy']=types.ModuleType('numpy')
hooks=list(sys.meta_path)
entry=importlib.import_module('XPolicyLab.policy.Riemann-RSI-Preview.model')
deploy=importlib.import_module('XPolicyLab.policy.Riemann-RSI-Preview.deploy')
from rsi.official_adapter import Model
assert entry.Model is Model
assert deploy.eval_one_episode is loop.eval_one_episode
assert sys.meta_path==hooks
assert 'torch' not in sys.modules and 'isaacsim' not in sys.modules
print('official namespace imports passed')
'''
        with tempfile.TemporaryDirectory() as folder:
            result=subprocess.run([sys.executable,'-I','-S','-B','-c',code,str(ROOT),folder],
                                  cwd=folder,capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('official namespace imports passed',result.stdout)

    def test_direct_model_file_import_also_bootstraps_its_own_root(self):
        code = r'''import importlib.util,pathlib,sys,types
root=pathlib.Path(sys.argv[1]); sdk=types.ModuleType('XPolicyLab'); sdk.__path__=[]
template=types.ModuleType('XPolicyLab.model_template'); template.ModelTemplate=object
utils=types.ModuleType('XPolicyLab.utils'); utils.__path__=[]
process=types.ModuleType('XPolicyLab.utils.process_data'); process.get_robot_action_dim_info=lambda *a: {}
for module in (sdk,template,utils,process):sys.modules[module.__name__]=module
sys.modules['numpy']=types.ModuleType('numpy')
spec=importlib.util.spec_from_file_location('_submission_entry',root/'model.py'); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
assert module.Model.__module__=='rsi.official_adapter'
'''
        with tempfile.TemporaryDirectory() as folder:
            result=subprocess.run([sys.executable,'-I','-S','-B','-c',code,str(ROOT)],cwd=folder,capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_local_binding_has_locations_only_and_environment_has_explicit_precedence(self):
        from rsi.deployment import locations, model_options
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve()
            binding=dict(official_runtime=dict(official_root=str(root/'sdk'),python=str(root/'eval/bin/python')),
                         policy_python=str(root/'model/bin/python'),wam_checkpoint=str(root/'weights'),encoder_path=str(root/'encoder'))
            (root/'deploy.local.json').write_text(json.dumps(binding))
            actual=locations(root,environment={'RIEMANN_OFFICIAL_ROOT':str(root/'other-sdk')})
            self.assertEqual(actual['official_root'],str(root/'other-sdk'))
            self.assertEqual(actual['eval_prefix'],str(root/'eval'))
            self.assertEqual(model_options(root,environment={}),{'wam_checkpoint':str(root/'weights'),'encoder_path':str(root/'encoder')})
            binding['official_eval']=False
            (root/'deploy.local.json').write_text(json.dumps(binding))
            with self.assertRaisesRegex(ValueError,'Unknown local deployment'):locations(root,environment={})

    def test_resource_relocation_covers_runtime_environment_and_nested_asset_roots(self):
        from rsi.resource_locations import relocate_manifest, relocate_path, reference_for
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve()
            env=dict(RIEMANN_OFFICIAL_ROOT=str(root/'sdk'),RIEMANN_EVAL_PYTHON=str(root/'eval/bin/python'),
                     RIEMANN_RESOURCE_ROOTS=json.dumps({'official_assets':str(root/'assets'),'models':str(root/'models')}))
            original=dict(official_root='resource://official_runtime',python_prefix='resource://eval_environment',
                          origin='resource://eval_environment/lib/python3.11/site-packages/numpy/__init__.py',
                          asset='resource://official_assets/Robots/x5/X5A.urdf',model='resource://models/sam3-robodojo-v21')
            actual=relocate_manifest(original,environment=env)
            self.assertEqual(actual['official_root'],str(root/'sdk'))
            self.assertEqual(actual['python_prefix'],str(root/'eval'))
            self.assertEqual(actual['asset'],str(root/'assets/Robots/x5/X5A.urdf'))
            self.assertEqual(actual,relocate_manifest(actual,environment=env))
            self.assertEqual(reference_for(actual['model'],environment=env),'resource://models/sam3-robodojo-v21')
            with self.assertRaises(ValueError):relocate_path('resource://models/../secret',environment=env)

    def test_overlapping_model_destination_is_idempotent(self):
        from rsi.resource_locations import relocate_path
        env={'RIEMANN_RESOURCE_ROOTS':json.dumps({'models':'/opt/models/relocated'}),
             'RIEMANN_RESOURCE_ALIASES':json.dumps({'models':['/opt/models']})}
        first=relocate_path('/opt/models/sam3',environment=env)
        self.assertEqual(first,'/opt/models/relocated/sam3')
        self.assertEqual(relocate_path(first,environment=env),first)

    def test_environment_install_targets_only_a_new_venv(self):
        import venv
        from tools.configure_official_policy import environment_commands, python_identity
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'new-env';commands=environment_commands(target,Path(sys.executable))
            self.assertEqual(commands[0],[sys.executable,'-m','venv',str(target)])
            self.assertEqual(commands[1][0],str(target/'bin/python'))
            self.assertIn('--require-virtualenv',commands[1]);self.assertNotIn('--upgrade',commands[1])
            venv.EnvBuilder(with_pip=False).create(target)
            self.assertEqual(Path(python_identity(target/'bin/python')['prefix']).resolve(), target.resolve())
            with self.assertRaisesRegex(ValueError,'must not already exist'):environment_commands(target,Path(sys.executable))

    def test_install_plan_does_not_touch_env_or_binding(self):
        from tools.configure_official_policy import main
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'model';output=io.StringIO()
            with redirect_stdout(output), patch('tools.configure_official_policy.subprocess.run',side_effect=AssertionError('Plan cannot execute')):
                self.assertEqual(main(['--plan','--create-policy-env',str(target),'--eval-python',sys.executable,
                                      '--resource-root','models='+str(Path(folder)/'models')]),0)
            proof=json.loads(output.getvalue())
            self.assertFalse(proof['files_changed']);self.assertEqual(len(proof['install_commands']),2)
            self.assertFalse(target.exists())

    def test_install_plan_rejects_unknown_resource_roots(self):
        from tools.configure_official_policy import main
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            main(['--plan','--resource-root','unknown=/tmp/model'])
        self.assertEqual(error.exception.code,2)

    def test_standard_conda_evaluator_argument_resolves_to_its_declared_python(self):
        from rsi.deployment import interpreter_python
        with patch('rsi.deployment.subprocess.check_output',return_value='/opt/official/bin/python\n') as query:
            self.assertEqual(interpreter_python('official-eval'),'/opt/official/bin/python')
        self.assertEqual(query.call_args.args[0][:5],['conda','run','--no-capture-output','-n','official-eval'])

    def test_scheduler_executable_has_an_explicit_portable_location(self):
        from rsi.deployment import slurm_srun
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); binary=root/'srun'; binary.write_text('#!/bin/sh\n'); binary.chmod(0o700)
            with patch.dict(os.environ, {'RIEMANN_SLURM_BIN':folder}):
                self.assertEqual(slurm_srun(root),str(binary))
            with patch.dict(os.environ, {}, clear=True), patch('rsi.deployment.shutil.which',return_value=None):
                with self.assertRaisesRegex(ValueError,'Slurm srun is missing'): slurm_srun(root)

    def test_native_library_binding_is_child_local_and_keeps_existing_paths(self):
        from rsi.deployment import apply_native_libraries
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); native=root/'native'; native.mkdir()
            (root/'deploy.local.json').write_text(json.dumps(dict(native_library_dirs=[str(native)])))
            environment={'LD_LIBRARY_PATH':'/opt/existing'}
            self.assertEqual(apply_native_libraries(environment,root),[str(native)])
            self.assertEqual(environment['LD_LIBRARY_PATH'],str(native)+os.pathsep+'/opt/existing')
            apply_native_libraries(environment,root)
            self.assertEqual(environment['LD_LIBRARY_PATH'],str(native)+os.pathsep+'/opt/existing')
            with self.assertRaisesRegex(ValueError,'Missing declared native'):
                apply_native_libraries({'RIEMANN_NATIVE_LIBRARY_DIRS':str(root/'absent')},root)

    def test_help_works_from_a_different_directory_without_sdk(self):
        with tempfile.TemporaryDirectory() as folder:
            result=subprocess.run([sys.executable,'-I','-S','-B',str(ROOT/'tools/configure_official_policy.py'),'--help'],
                                  cwd=folder,capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr);self.assertIn('--create-policy-env',result.stdout)
