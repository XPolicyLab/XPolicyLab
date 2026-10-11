import unittest
from rsi_supervisor_v1 import Evidence,Snapshot
from pickup_supervisor_adapter import PickupSupervisorAdapter,snapshot

def row(step,success=False):
    return dict(native_step=step,status='available',action_id='baseline' if step==0 else 'primitive',
        initial_baseline=0,current_score=float(success),native_success=success,
        stage_conditions={'lift_10cm':{'historically_achieved':success}})

class Tests(unittest.TestCase):
    def test_unknown_event_not_read_id(self):
        s=snapshot(row(89),terminal=False)
        self.assertIsNone(s.score_event);self.assertIsNone(s.fresh_score())
        a=PickupSupervisorAdapter();a.begin('lift',s)
        result=a.after(s,kind='lift',action_ok=True,checkpoint_step=89,phase_complete=True)
        self.assertEqual(result['acceptance']['decision'],'needs_observation')

    def test_zero_step_failure_bounded(self):
        s=snapshot(row(0),terminal=False); a=PickupSupervisorAdapter();a.begin('approach',s)
        first=a.after(s,kind='IK',action_ok=False,checkpoint_step=0,phase_complete=False,recovery_fits=True)
        second=a.after(s,kind='IK',action_ok=False,checkpoint_step=0,phase_complete=False,recovery_fits=True)
        self.assertEqual(first['acceptance']['decision'],'recover')
        self.assertEqual(second['acceptance']['decision'],'stop_failure')

    def test_terminal_overrides_partial_motion(self):
        a=PickupSupervisorAdapter();a.begin('lift',snapshot(row(89),terminal=False))
        r=a.after(snapshot(row(100,True),terminal=True),kind='lift',action_ok=False,
                  checkpoint_step=100,phase_complete=True)
        self.assertEqual(r['acceptance']['decision'],'stop_success')

    def test_history_not_retention(self):
        a=PickupSupervisorAdapter();s=snapshot(row(120),terminal=False)
        a.begin('grasp',s)
        self.assertEqual(a.after(s,kind='close',action_ok=True,checkpoint_step=120,
                         phase_complete=True)['acceptance']['reason'],'postcondition_unknown')

    def test_budget_includes_sync_settle_reserve(self):
        a=PickupSupervisorAdapter();s=Snapshot(step=150)
        self.assertEqual(a.preflight(s)['decision'],'needs_estimate')
        self.assertEqual(a.preflight(s,planned_actions=[{}]*10,max_settle_steps=10,finish_reserve=40)['decision'],'replan')
        self.assertEqual(a.preflight(s,planned_actions=[{}]*10,max_settle_steps=10,finish_reserve=40,is_finishing=True)['decision'],'allow')
        self.assertEqual(a.preflight(Snapshot(step=200),planned_actions=[],max_settle_steps=0)['decision'],'stop')

    def test_stale_visual_not_passed(self):
        a=PickupSupervisorAdapter();s=snapshot(row(89),terminal=False,current={'object_retained':Evidence(True,77,'old image')})
        a.begin('grasp',s)
        self.assertEqual(a.after(s,kind='lift',action_ok=True,checkpoint_step=89,phase_complete=True)['acceptance']['decision'],'needs_observation')

if __name__=='__main__': unittest.main(verbosity=2)
