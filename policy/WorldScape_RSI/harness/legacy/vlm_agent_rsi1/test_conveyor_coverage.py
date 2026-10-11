import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import conveyor_coverage as m
from conveyor_coverage import choose,TASK
def c(seed,layout):return dict(task=TASK,seed=seed,layout_id=layout)
class CoverageTests(unittest.TestCase):
 def test_skip_success(self):self.assertEqual(choose([c(0,0),c(0,1)],[c(0,0)],[]),c(0,1))
 def test_no_wrap(self):self.assertIsNone(choose([c(0,0)],[c(0,0)],[dict(c(0,0),outcome='failure')]))
 def test_fresh_before_retry(self):self.assertEqual(choose([c(0,0),c(0,1)],[],[dict(c(0,0),outcome='failure')]),c(0,1))
 def test_retry_failed_only(self):self.assertEqual(choose([c(0,0)],[],[dict(c(0,0),outcome='failure')]),c(0,0))
 def test_unknown_seed(self):self.assertEqual(choose([c(0,0),c(1,0),c(0,1)],[c(None,0)],[]),c(0,1))
 def test_unknown_identity(self):self.assertIsNone(choose([c(0,0)],[c(None,None)],[]))
 def test_version_not_new(self):self.assertIsNone(choose([dict(c(0,0),version='new')],[dict(c(0,0),version='old')],[]))
 def test_pending_delivery_not_retry(self):self.assertIsNone(choose([c(0,0)],[],[dict(c(0,0),outcome='failure'),dict(c(0,0),outcome='success_pending_delivery')]))
 def test_pause_blocks_preflight(self):
  with tempfile.TemporaryDirectory() as t,patch.object(m,'CONTROL',Path(t)):
   with self.assertRaisesRegex(RuntimeError,'Paused'):m.preflight(0)
 def test_actual_reset_mismatch(self):
  with tempfile.TemporaryDirectory() as t,patch.object(m,'CONTROL',Path(t)):
   m.save(Path(t)/'policy.json',{'paused':False});m.save(Path(t)/'selection.json',c(0,1))
   with self.assertRaisesRegex(RuntimeError,'Actual reset'):m.check_episode(SimpleNamespace(eval_seed=0,current_env_seed_map={0:0}))
 def test_native_queue_original_index(self):
  with tempfile.TemporaryDirectory() as t,patch.object(m,'CONTROL',Path(t)),patch.object(m,'rebuild',return_value={'successes':[]}):
   p=Path(t)/'layout.json';p.write_text('{}');s=dict(c(0,7),file_sha256=m.hashlib.sha256(p.read_bytes()).hexdigest())
   m.save(Path(t)/'selection.json',s);m.save(Path(t)/'policy.json',{'paused':False})
   manager=SimpleNamespace(seed_info={7:{'scene_layout':str(p)}})
   m.apply_queue(SimpleNamespace(eval_seed=0,seed_manager=manager))
   self.assertEqual(manager.seed_list,[7]);self.assertEqual(manager.ed_idx,1)
if __name__=='__main__':unittest.main(verbosity=2)
