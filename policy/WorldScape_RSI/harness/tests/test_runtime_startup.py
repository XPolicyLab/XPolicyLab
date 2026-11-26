"""Exercise production imports and preserve nested checkpoint identities."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from test_harness import ROOT, framework
from r2si_under_test import wam, release


class RuntimeStartupTests(unittest.TestCase):
    def test_cuda_runtime_has_its_native_sources(self):
        root=ROOT/'vendor/runtime_support/curobo'
        sources=json.loads((root/'cuda-sources.json').read_text())
        self.assertIn('curobo/_src/curobolib/kernels/kinematics/kinematics_forward_kernel.cuh',sources)
        for name,expected in sources.items():
            self.assertEqual(release.digest(root/name),expected,name)

    def test_real_backend_constructs_without_evaluation_driver(self):
        self.assertTrue(framework, 'Install the XPolicyLab framework for this test')
        code = '''
import os,sys,tempfile
from pathlib import Path
sys.path[:0]=PATHS
from XPolicyLab.policy.WorldScape_RSI.harness.backends import AgentBackend
with tempfile.TemporaryDirectory() as temp:
 root=Path(temp)
 files={
  'Assets/Robots/x5/X5A.urdf':'<robot>'+''.join('<joint name="joint%d"><limit lower="-2" upper="2"/></joint>'%i for i in range(1,7))+'</robot>',
  'env_cfg/arx_x5.yml':'observation: {collect_freq: 25}',
  'env_cfg/sim/sim_config.yml':'dt: 0.004',
  'env/robot_manager/control_manager.py':'gripper_eps=0.01',
  'env/robot_manager/robot_config/x5.py':'velocity_limit_sim = 5.0',
 }
 for name,text in files.items():
  p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
 os.environ.update(ROBODOJO_ROOT=str(root),TEST_API_BASE='https://example.invalid',TEST_API_KEY='fixture-only')
 spec={'model':'fixture','effort':'high','api_base_env':'TEST_API_BASE','api_key_env':'TEST_API_KEY','wam_skills':{}}
 backend=AgentBackend(spec,{'seed':0,'env_cfg_type':'arx_x5'})
 try:
  assert len(backend.spec.labels)==14
  backend.reset()
  assert not backend.wam.models
  assert not any(name.endswith('.deploy') and 'L3_Inspect' in name for name in sys.modules)
 finally:backend.close()
'''.replace('PATHS',repr([str(Path(framework).parent), framework, *sys.path]))
        result=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True,timeout=60)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_same_basename_in_subdirectory_is_verified_independently(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'tokenizer').mkdir()
            (root/'config.json').write_text('model config')
            (root/'tokenizer/config.json').write_text('tokenizer config')
            manifest={name:release.digest(root/name) for name in ['config.json','tokenizer/config.json']}
            wam._verify_files(root,manifest,'test')
            for name in manifest:
                original=(root/name).read_bytes()
                (root/name).write_text('changed')
                with self.assertRaisesRegex(RuntimeError,'hash mismatch'):
                    wam._verify_files(root,manifest,'test')
                (root/name).write_bytes(original)

    def test_frozen_checkpoints_keep_model_and_tokenizer_configs(self):
        specs=json.loads((ROOT/'configs/wam_skills.json').read_text())
        for name,spec in specs.items():
            files=spec['checkpoint_files']
            with self.subTest(skill=name):
                self.assertGreaterEqual(len(files),17)
                self.assertIn('checkpoint_manifest.json',files)
                self.assertIn('transform_bundle.json',files)
                self.assertNotEqual(files['config.json'],files['tokenizer/config.json'])
                self.assertIn('tokenizer/tokenizer.json',files)
                self.assertNotIn('tokenizer.json',files)
