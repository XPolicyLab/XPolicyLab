import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from pickup_feedback import PickupFeedback, FeedbackResults

def environment():
    rm = NS(check_list=[[[('is_lift', {'label': 'target', 'z_threshold': 0.1})]]],
            score_meta=[{'mode':None}], final_score_meta=[{'mode':None}],
            final_check_list=[[]], trigger_check_list=[[]], query_list=[[]], trigger_query_list=[[]])
    return NS(task_name='general_pickup', step_lim=200, take_action_cnt=[0],
              end_flag=[False], success=[True], reward_manager=rm)

class FeedbackTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.env = environment()
        self.f = PickupFeedback(self.env, Path(self.tmp.name)/'feedback.jsonl')

    def test_baseline_and_read_only(self):
        before = copy.deepcopy(self.env)
        for _ in range(3):
            row = self.f.sample('baseline', 'test')
            self.assertEqual((row['current_score'], row['delta_score'], row['native_success']), (0,0,False))
        self.assertEqual(before, self.env)

    def test_success_and_same_step_history(self):
        self.f.sample('baseline','test')
        self.env.take_action_cnt[0] = 166
        self.env.end_flag[0] = True
        self.env.reward_manager.check_list[0].clear()
        row = self.f.sample('lift','test')
        self.assertEqual((row['current_score'],row['delta_score'],row['remaining_steps']), (1,1,34))
        self.assertEqual(row['stage_conditions']['lift_10cm'], {'current':None,'historically_achieved':True})
        self.assertEqual(self.f.sample('read','test')['delta_score'],0)

    def test_timeout_is_not_success(self):
        self.f.sample('baseline','test')
        self.env.take_action_cnt[0] = 200
        self.env.end_flag[0] = True
        self.env.success[0] = False
        self.assertEqual(self.f.sample('end','test')['current_score'],0)

    def test_unregistered_and_late_attach_unknown(self):
        for change in ('empty','late','different_task'):
            e = environment()
            if change == 'empty': e.reward_manager.check_list[0].clear()
            if change == 'late': e.take_action_cnt[0] = 1
            if change == 'different_task': e.task_name = 'other'
            f = PickupFeedback(e,Path(self.tmp.name)/change)
            self.assertIsNone(f.sample('baseline','test')['current_score'])

    def test_primitive_sidecar_and_no_result_pollution(self):
        self.f.sample('baseline','test')
        results = FeedbackResults(self.f)
        result = {'command':'close', 'note':'closure completed'}
        results.append(result)
        self.assertEqual(result, {'command':'close', 'note':'closure completed'})
        rows = [json.loads(s) for s in self.f.path.read_text().splitlines()]
        self.assertEqual(rows[-1]['action_id'],'primitive-1')
        self.assertTrue(rows[-1]['uses_privileged_reward_feedback'])

if __name__ == '__main__': unittest.main(verbosity=2)
