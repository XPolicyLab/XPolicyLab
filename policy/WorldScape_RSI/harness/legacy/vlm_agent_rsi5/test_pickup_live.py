import unittest
from types import SimpleNamespace as NS
from pathlib import Path
import tempfile
from pickup_live import LivePickupFeedback

class Tests(unittest.TestCase):
    def make(self):
        rm=NS(check_list=[[[('is_lift',{'label':'target','z_threshold':0.1})]]],score_meta=[{'mode':None}],final_score_meta=[{'mode':None}],final_check_list=[[]],trigger_check_list=[[]],query_list=[[]],trigger_query_list=[[]])
        e=NS(task_name='general_pickup',step_lim=200,take_action_cnt=[0],reward_manager=rm,end_flag=[False],success=[True])
        d=tempfile.TemporaryDirectory();self.addCleanup(d.cleanup)
        f=LivePickupFeedback(e,str(Path(d.name)/'feedback.jsonl'));f.sample('baseline','test');f.configure({})
        return e,f
    def test_stage_timestamp_and_budget(self):
        e,f=self.make();e.take_action_cnt[0]=80;f.sample('primitive','test')
        f.configure({'stage':'grasp','approach_clear':True,'observation_step':79})
        self.assertEqual(f.preflight([{}],0,'gripper')['decision'],'needs_observation')
        f.configure({'stage':'grasp','approach_clear':True,'observation_step':80})
        self.assertEqual(f.stage,'grasp')
        self.assertEqual(f.preflight([{}]*10,53,'gripper')['decision'],'allow')
        e.take_action_cnt[0]=110;f.sample('primitive','test')
        self.assertEqual(f.preflight([{}]*10,53,'gripper')['decision'],'replan')
    def test_verified_failed_retention_stops_before_action(self):
        e,f=self.make();f.configure({'stage':'grasp','approach_clear':True,'observation_step':0})
        f.configure({'stage':'lift','object_retained':False,'observation_step':0})
        self.assertTrue(f.stop_requested)
        self.assertNotEqual(f.preflight([{}],0,'wait')['decision'],'allow')
        self.assertEqual(e.take_action_cnt[0],0)
    def test_native_success_partial(self):
        e,f=self.make();e.take_action_cnt[0]=100;e.end_flag[0]=True
        f.command_done({'kind':'move','ok':False})
        self.assertFalse(f.stop_requested)
        self.assertEqual(f.latest['supervisor']['acceptance']['decision'],'stop_success')
    def test_rejection_no_terminal(self):
        e,f=self.make();f.command_done({'kind':'move','ok':False,'budget_rejected':{'decision':'replan'}})
        self.assertFalse(f.stop_requested);self.assertEqual(e.take_action_cnt[0],0)
if __name__=='__main__':unittest.main(verbosity=2)
