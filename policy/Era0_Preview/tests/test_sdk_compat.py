"""CPU coverage of SDK source selection, fail-closed validation and output paths."""
import ast
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

POLICY=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(POLICY.parents[2]))
from XPolicyLab.policy.Era0_Preview import sdk_compat


class SDKCompatTests(unittest.TestCase):
    def make_official(self,root):
        for source in (POLICY/'sdk_sources/base').rglob('*.py'):
            target=root/source.relative_to(POLICY/'sdk_sources/base')
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source.read_bytes())

    def test_clean_official_sdk_accepts_both_profiles_without_edits(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.make_official(root)
            before={str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*.py')}
            for profile in ['local','official']:
                spec=sdk_compat.validate(root,profile)
                self.assertEqual(spec['ik_guard_enabled'],profile=='local')
                self.assertEqual(spec['unsupported_sdk_files'],[])
                self.assertIn(b'ROBODOJO_EVAL_OUTPUT_DIR',sdk_compat.source(root,'src/eval_client/eval_env.py',profile))
            self.assertEqual(before,{str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*.py')})
            self.assertFalse((root/'src/eval_client/ik_guard.py').exists())

    def test_unknown_sdk_fails_before_loading_simulator(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.make_official(root)
            (root/'src/eval_client/eval_env.py').write_text('# unexpected version')
            with self.assertRaisesRegex(RuntimeError,'review before running'):
                sdk_compat.install(root,'local')

    def test_runtime_overlay_preserves_host_and_uses_selected_source(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);policy=root/'policy';policy.mkdir();sources=policy/'sdk_sources';sources.mkdir()
            name='era0_test_sdk/probe.py';host=root/name;host.parent.mkdir();host.write_text('value="host"\n')
            (host.parent/'__init__.py').write_text('')
            local=sources/'probe.py';local.write_text('value="guarded"\n')
            sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
            (sources/'manifest.json').write_text(json.dumps(dict(host_files={name:[sha(host)]},
                profiles={'local':{name:dict(file='sdk_sources/probe.py',sha256=sha(local))}})))
            sys.path.insert(0,str(root));finder=None
            try:
                with patch.object(sdk_compat,'POLICY',policy),patch.object(sdk_compat,'SOURCES',sources):
                    sdk_compat.install(root,'local');finder=sys.meta_path[0]
                    mod=importlib.import_module('era0_test_sdk.probe')
                    self.assertEqual(mod.value,'guarded');self.assertEqual(mod.__file__,str(host))
                    self.assertEqual(host.read_text(),'value="host"\n')
            finally:
                if finder in sys.meta_path:sys.meta_path.remove(finder)
                sys.path.remove(str(root))
                sys.modules.pop('era0_test_sdk.probe',None);sys.modules.pop('era0_test_sdk',None)

    def test_resume_paths_match_in_both_profiles(self):
        # Execute the exact main.py and EvalEnv path functions without importing Isaac.
        for profile in ['local','official']:
            main=sdk_compat.source(POLICY,'src/eval_client/main.py',profile).decode()
            tree=ast.parse(main);func=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_resume_manifest_path')
            env=sdk_compat.source(POLICY,'src/eval_client/eval_env.py',profile).decode()
            tree2=ast.parse(env);method=next(n for n in ast.walk(tree2) if isinstance(n,ast.FunctionDef) and n.name=='resume_manifest_path')
            ns=dict(os=os,BENCHMARK='RoboDojo')
            exec(compile(ast.Module(body=[func],type_ignores=[]),'<resume-main>','exec'),ns)
            ns2=dict(os=os,BENCHMARK='RoboDojo')
            exec(compile(ast.Module(body=[method],type_ignores=[]),'<resume-env>','exec'),ns2)
            cfg=dict(task_name='stack_bowls',policy_name='Era0_Preview',config_name='arx_x5',seed=2,additional_info='private/path')
            from types import SimpleNamespace
            obj=SimpleNamespace(task_name=cfg['task_name'],policy_name=cfg['policy_name'],config_name=cfg['config_name'],eval_seed=2,run_id='run')
            with patch.dict(os.environ,{'ROBODOJO_EVAL_OUTPUT_DIR':'/portable/output'}):
                expected='/portable/output/RoboDojo/stack_bowls/Era0_Preview/arx_x5/2/_resume_run.json'
                self.assertEqual(ns['_resume_manifest_path'](cfg,'run'),expected)
                self.assertEqual(ns2['resume_manifest_path'](obj),expected)

    def test_wrapper_reinstalls_selected_overlay_after_exec_restart(self):
        # Real Python execv, with tiny SDK stand-ins instead of Isaac Sim.
        for profile in ['local', 'official']:
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as d:
                root=Path(d);policy=root/'policy';policy.mkdir()
                for name in ['simulator_entry.py','sdk_compat.py','obs_lag.py']:
                    shutil.copy2(POLICY/name,policy/name)
                sources=policy/'sdk_sources';sources.mkdir()
                sdk=root/'SDK'
                for name in ['src','src/eval_client']:
                    path=sdk/name;path.mkdir(parents=True,exist_ok=True)
                    (path/'__init__.py').touch()
                host_files={};profiles={p:{} for p in ['local','official']}
                code='''import json, os, sys
from src.eval_client import eval_env
with open(os.environ['TEST_TRACE'], 'a') as stream:
    stream.write(json.dumps([eval_env.value, __file__, sys.argv[0]])+'\\n')
if os.environ.get('TEST_RESTARTED') != '1':
    os.environ['TEST_RESTARTED']='1'
    os.execv(sys.executable, [sys.executable, '-P', sys.argv[0]])
'''
                for name in ['src/eval_client/main.py','src/eval_client/eval_env.py']:
                    host=sdk/name;host.write_text('# host remains untouched\n')
                    host_files[name]=[hashlib.sha256(host.read_bytes()).hexdigest()]
                    for p in profiles:
                        snapshot=sources/(p+'_'+Path(name).name)
                        snapshot.write_text(code if name.endswith('/main.py') else f'value={p!r}\n')
                        profiles[p][name]=dict(file=str(snapshot.relative_to(policy)),
                                              sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest())
                (sources/'manifest.json').write_text(json.dumps(dict(host_files=host_files,profiles=profiles)))
                trace=root/'trace.jsonl'
                env=dict(os.environ,ROBODOJO_ROOT=str(sdk),ERA0_EXECUTION_PROFILE=profile,
                         ROBODOJO_EVAL_LEGACY_RENDER='0',PYTHONPATH=str(sdk),TEST_TRACE=str(trace))
                env.pop('TEST_RESTARTED',None)
                subprocess.run([sys.executable,'-P',str(policy/'simulator_entry.py')],env=env,
                               capture_output=True,text=True,check=True,timeout=20)
                rows=[json.loads(line) for line in trace.read_text().splitlines()]
                self.assertEqual(rows,[[profile,str(sdk/'src/eval_client/main.py'),
                                       str(policy/'simulator_entry.py')]]*2)
                self.assertEqual((sdk/'src/eval_client/eval_env.py').read_text(),'# host remains untouched\n')

if __name__=='__main__':unittest.main()
