"""Toolbox adapter for the frozen pure supervisor; no simulator/reward calls."""
import importlib.util
import sys
from pathlib import Path

_name='rsi3_frozen_supervisor_v1'
if _name not in sys.modules:
    _spec=importlib.util.spec_from_file_location(_name,Path(__file__).with_name('rsi_supervisor_v1.py'))
    _mod=importlib.util.module_from_spec(_spec);sys.modules[_name]=_mod;_spec.loader.exec_module(_mod)
core=sys.modules[_name]

def snapshot(data, *, terminal=False, current=None):
    step=int(data['native_step'])
    fresh=data.get('status')=='fresh' and data.get('source_update_step')==step
    return core.Snapshot(step, data.get('current_score') if fresh else None,
        data.get('source_update_step'),data.get('source',''),'0..100',terminal,
        data.get('native_success') if terminal else None,
        data.get('score_event') if fresh else None,
        core.Evidence(data.get('score_evaluated') if fresh else None,step,'audited existing native get_reward/get_score callback'),
        current or {}, {})

def stage_assessment(flow,state,data):
    step=int(state.get('steps_used',0));index=len(flow.get('completed_visual_checks') or [])
    opened=all(isinstance(state.get(a,{}).get('opening'),(float,int)) and state[a]['opening']>.8 for a in ('left','right'))
    current={'jaws_open':core.Evidence(opened,step,'current measured gripper positions')}
    snap=snapshot(data,current=current)
    if snap.step!=step:return {'decision':'needs_observation','reason':'state_feedback_step_mismatch'}
    sup=core.Supervisor('store_tools_in_toolbox',900)
    sup.begin_stage(core.StageContract('released_'+str(flow.get('tool')),expected_score=25*(index+1),score_scale='0..100',required_current=('jaws_open',)),core.Snapshot(0))
    return sup.assess(snap,action_ok=True,checkpoint_step=step,phase_complete=True)

def primitive_feedback(sup,data,event):
    step=int(event['native_step']);ok=event.get('action_ok') is True and not event.get('retention_warning')
    snap=snapshot(data,terminal=event.get('native_terminal') is True,
        current={'execution_accepted':core.Evidence(ok,step,'actual controller tracking and retention-warning checks; not proof of object retention')})
    late=step<sup.last_step
    if late:sup=core.Supervisor('store_tools_in_toolbox',900)
    feedback=sup.feedback(snap)
    feedback['late_flush']=late
    sup.begin_stage(core.StageContract('primitive_result',required_current=('execution_accepted',),require_trigger=False,max_recoveries=1),core.Snapshot(step))
    verdict=sup.assess(snap,action_ok=ok,checkpoint_step=step,phase_complete=True,recovery_fits=900-step>100)
    return {'feedback':feedback,'assessment':verdict}
