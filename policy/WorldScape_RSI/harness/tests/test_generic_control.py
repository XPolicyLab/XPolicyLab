import copy,json,types,unittest
from unittest.mock import patch
import numpy as np
from test_harness import ROOT,DIMS,obs,hold,backends,model,FakeRelease,framework
from r2si_under_test.control_guard import ControlGuard
from r2si_under_test.wam import WamSkills
from test_wam_skills import FakeWsp,MEM,NOMEM

def assessment(mode='flash',progress='initial',**kw):
    return dict(mode=mode,progress=progress,evidence='The current RGB supports a clear local approach',expected_change='The visible approach gap should decrease',instruction_sufficient=True,requires_precise_control=False,requires_agent_memory=False,history_available=False,**kw)
class GenericControlTests(unittest.TestCase):
    def setUp(self):self.g=ControlGuard();self.g.observe(obs(),0)
    def test_missing_assessment_cannot_delegate(self):
        with self.assertRaises(ValueError):self.g.consume(NOMEM)
    def test_matching_assessment_is_single_use(self):
        self.g.assess(assessment());self.g.consume(NOMEM)
        with self.assertRaises(ValueError):self.g.consume(NOMEM)
    def test_precise_control_and_agent_memory_reject(self):
        for key in ('requires_precise_control','requires_agent_memory'):
            g=ControlGuard();g.observe(obs(),0);a=assessment();a[key]=True
            with self.assertRaises(ValueError):g.assess(a)
            with self.assertRaises(ValueError):g.consume(NOMEM)
    def test_wrong_direction_blocks_both_variants(self):
        self.assertTrue(self.g.review(dict(decision='continue',progress='wrong_direction',evidence='The latest frame shows movement away from the intended contact')))
        for key in (MEM,NOMEM):
            with self.assertRaises(ValueError):self.g.consume(key)
    def test_two_distinct_stalled_frames_block_switching(self):
        a=dict(decision='continue',progress='stalled',evidence='The same object and gripper remain unchanged in the fresh frame')
        self.assertFalse(self.g.review(a));self.assertFalse(self.g.review(a))
        self.g.observe(obs(),1,hold(),True);self.assertTrue(self.g.review(a))
        self.g.observe(obs(),2,hold(),False)
        self.assertTrue(self.g.native_required)
        with self.assertRaises(ValueError):self.g.assess(assessment('memory','advancing'))
    def test_actual_native_motion_and_fresh_assessment_required(self):
        self.g.review(dict(decision='takeover',progress='uncertain',evidence='Contact cannot be confirmed from the obstructed current view'))
        self.g.observe(obs(),1,hold(),False);self.assertTrue(self.g.native_required)
        moved=obs();moved['state']['left_arm_joint_state'][0]=.01
        self.g.observe(moved,2,hold(.01),False);self.assertFalse(self.g.native_required)
        with self.assertRaises(ValueError):self.g.consume(NOMEM)
        self.g.assess(assessment());self.g.consume(NOMEM)
    def test_initial_cannot_erase_wam_execution(self):
        self.g.observe(obs(),1,hold(),True)
        with self.assertRaises(ValueError):self.g.assess(assessment())
    def test_cold_memory_cannot_claim_missing_history(self):
        a=assessment('memory');a['history_available']=True;self.g.assess(a)
        with self.assertRaises(ValueError):self.g.consume(MEM,None)
    def test_both_arms_pose_and_quaternion_sign(self):
        o=obs();o['state']['right_ee_pose'][0]=.1;self.g.observe(o,1)
        self.assertFalse(self.g.home_status()['verified'])
        o=obs();o['state']['left_ee_pose'][3:]*=-1;self.g.observe(o,2)
        self.assertTrue(self.g.home_status()['verified'])
    def test_homing_blocks_policy_even_if_other_flags_good(self):
        self.g.assess(assessment('return_home'));self.g.observe(obs(),1)
        with self.assertRaises(ValueError):self.g.assess(assessment())
    def test_partial_discard_requires_ack_and_rebases(self):
        f=FakeWsp();w=WamSkills({MEM:{}},{},factory=lambda *_:f,dims=DIMS);w.observe(obs(),None);a=w.plan(MEM);w.before(a[0])
        with self.assertRaises(RuntimeError):w.interrupt_pending()
        w.observe(obs(),a[0]);w.interrupt_pending()
        self.assertEqual(f.events[-2:], [('discard',{'execution_started':True}),('obs',)])
        self.assertFalse(any(e[0]=='commit' for e in f.events));self.assertIsNone(w.pending)
        with self.assertRaises(RuntimeError):w.interrupt_pending()
    def test_model_boundary_discards_suffix_before_second_action(self):
        from collections import deque
        f=FakeWsp();w=WamSkills({MEM:{}},{},factory=lambda *_:f,dims=DIMS);w.observe(obs(),None);actions=w.plan(MEM);w.before(actions[0]);w.observe(obs(),actions[0])
        b=object.__new__(backends.AgentBackend);b.wam=w;b.control=self.g;b.current=types.SimpleNamespace(state=obs()['state']);b.reasons=[];b._audit=lambda:None
        b.policy=types.SimpleNamespace(review_wam_execution=lambda *a:dict(decision='takeover',progress='wrong_direction',evidence='The actual frame shows the object slipping toward the wrong side'))
        completed=[];b.completed=lambda *a:completed.append(a);b.plan=lambda:[hold(.7)];b.before=lambda *a:None
        m=object.__new__(model.Model);m._lock=__import__('threading').RLock();m._check=lambda:None;m._observation=obs();m._pending=None;m._queue=deque([actions[1]]);m._repeat_remaining=0;m._executed=1;m._planned=2;m._backend=b;m.dims=DIMS;m._failed=False
        got=m.get_action()[0]
        np.testing.assert_allclose(got['left_arm_joint_state'],.7)
        self.assertEqual(completed,[(1,2,True)]);self.assertIn(('discard',{'execution_started':True}),f.events)
        self.assertFalse(any(e[0]=='commit' for e in f.events))
    def test_pure_generic_catalog_and_preload_never_silently_drops(self):
        backends.register_vendor()
        from XPolicyLab.policy.RoboRSI_RoboProbe.library import SkillLibrary
        lib=SkillLibrary(ROOT/'vendor/policy/RoboRSI_RoboProbe/skills');self.assertTrue(all(e['scope']=='generic' for e in lib.entries.values()))
        self.assertIn('return-home',lib.entries)
        with self.assertRaises(ValueError):lib.initial(1)
    def test_return_home_uses_saved_pose_not_agent_coordinates(self):
        backends.register_vendor()
        from XPolicyLab.policy.RoboRSI_RoboProbe import tools
        actor=types.SimpleNamespace(library=None,control_guard=self.g,native_move=lambda t,o:t)
        current=types.SimpleNamespace(state=obs()['state']);current.state['left_ee_pose'][0]=.2
        self.g.observe({'state':current.state},1)
        got=tools.execute(actor,'return_home',{'arm':'left'},current)
        from XPolicyLab.policy.RoboDojo_Agent_L3_Inspect_EEF import pose
        expected={ 'left_'+k:v for k,v in pose.pose_to_values(self.g.initial['left_ee_pose']).items()}
        self.assertEqual(got,expected)
        with self.assertRaises(ValueError):tools.execute(actor,'return_home',{'arm':'left','position':[9,9,9]},current)

class InstructionAndReviewTests(unittest.TestCase):
    def test_hold_with_tracking_drift_does_not_unlock_delegation(self):
        g=ControlGuard();g.observe(obs(),0);g.review(dict(decision='takeover',progress='wrong_direction',evidence='Visible contact is in the wrong direction'))
        drift=obs();drift['state']['left_arm_joint_state'][0]=.01
        g.observe(drift,1,hold(),False)
        self.assertTrue(g.native_required)
    def test_wam_receives_environment_instruction_unchanged(self):
        received=[]
        f=FakeWsp();f.update_obs=lambda o:received.append(copy.deepcopy(o))
        w=WamSkills({NOMEM:{}},{},factory=lambda *_:f,dims=DIMS)
        o=obs();o['instruction']='Follow the ORIGINAL visible relation; preserve orientation.'
        w.observe(o,None);w.plan(NOMEM)
        self.assertEqual(received[-1]['instruction'],o['instruction'])
        self.assertNotIn('agent_plan',received[-1])
    def test_policy_adapter_uses_only_explicit_checkpoint_format(self):
        import ast
        from typing import Mapping,Any
        p=ROOT.parent.parent/'WorldScape_RSI/xpolicylab_adapter/robodojo_model.py'
        tree=ast.parse(p.read_text());names={'_instruction','_format_policy_instruction'}
        ns={'Mapping':Mapping,'Any':Any};exec(compile(ast.Module(body=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in names],type_ignores=[]),str(p),'exec'),ns)
        goal='Maintain the ORIGINAL Orientation.'
        self.assertEqual(ns['_instruction']({'instruction':goal,'agent_plan':'invented subgoal'},'fallback'),goal)
        self.assertEqual(ns['_format_policy_instruction'](goal,mode='interactive'),goal)
        self.assertEqual(ns['_format_policy_instruction'](goal,mode='interactive',t5_prompt_template='Do: {instruction}'),'Do: '+goal.lower())
    def test_malformed_review_withholds_queued_suffix(self):
        from test_shared_capabilities import SharedAgentTests
        fixture=SharedAgentTests();fixture.setUp();actor=fixture.actor()
        observation=fixture.observation('private_fixture');actor.prepare(observation)
        # Existing fixture replies with call_tool rather than the required review.
        result=actor.review_wam_execution(observation,{})
        self.assertEqual(result['decision'],'takeover');self.assertEqual(result['progress'],'uncertain')

class HomeGripperTests(unittest.TestCase):
    def test_initial_gripper_restored_only_after_pose_return(self):
        backends.register_vendor()
        from XPolicyLab.policy.RoboRSI_RoboProbe import tools
        g=ControlGuard();g.observe(obs(),0)
        o=obs();o['state']['left_ee_joint_state'][0]=1.;g.observe(o,1)
        self.assertFalse(g.home_status()['verified'])
        actor=types.SimpleNamespace(library=None,control_guard=g,native_move=lambda t,o:t)
        target=tools.execute(actor,'return_home',{'arm':'left'},types.SimpleNamespace(state=o['state']))
        self.assertEqual(target,{'left_gripper':0.})
        g.observe(obs(),2);self.assertTrue(g.home_status()['verified'])

class CleanupFailureTests(unittest.TestCase):
    def test_diagnostic_failure_still_retires_both_upstreams(self):
        b=object.__new__(backends.AgentBackend);seen=[]
        def bad():raise OSError('synthetic quota failure')
        b._audit=bad;b.wam=types.SimpleNamespace(close=lambda:seen.append('wam'));b.client=types.SimpleNamespace(close=lambda:seen.append('agent'))
        with self.assertRaises(OSError):b.close()
        self.assertEqual(seen,['wam','agent'])
