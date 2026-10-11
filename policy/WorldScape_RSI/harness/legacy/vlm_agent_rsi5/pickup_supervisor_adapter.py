"""RSI-5 offline integration facade; no simulator/evaluator calls or auto-actions.

Caller supplies planner-expanded control actions and maximum settling bounds.
Not yet activated in deploy.py: no new GPU episode is authorized.
"""
from dataclasses import asdict
import hashlib
from pathlib import Path
from rsi_supervisor_v1 import Evidence, Snapshot, StageContract, Supervisor

RELEASE_SHA256='1c016b898de743e884ce8fe26aa28365c3a1db1c01998282b2f619411f094bb0'
assert hashlib.sha256(Path(__file__).with_name('rsi_supervisor_v1.py').read_bytes()).hexdigest()==RELEASE_SHA256

STAGES={
    'approach': StageContract('approach', required_current=('approach_clear',), require_trigger=False),
    # Close alone does not complete grasp: short initial lift must retain object.
    'grasp': StageContract('grasp', required_current=('object_retained',), require_trigger=False),
    'lift': StageContract('lift', expected_score=1.0, score_scale='binary_0_1', require_trigger=True),
}

def snapshot(row, *, terminal, current=None):
    step=row['native_step']
    known=row['status']=='available'
    initial=known and row['action_id']=='baseline' and step==0 and row['initial_baseline']==0
    # No evaluation-event stamp exists in old diagnostics. Never use read IDs.
    # Nonterminal binary zero is completion state, not a proven fresh reward.
    verified=known and (terminal or initial)
    return Snapshot(step=step, score=row['current_score'] if known else None,
        score_step=step if verified else None,
        score_source=('native terminal binary result' if terminal else 'registered initial binary baseline') if verified else '',
        score_scale='binary_0_1',terminal=terminal,
        native_success=row['native_success'] if terminal and known else None,
        score_event=None,trigger=Evidence(),current=current or {},
        historical={'lift_10cm':Evidence(row['stage_conditions']['lift_10cm']['historically_achieved'],
                                       step,'native check_list historical latch')})

class PickupSupervisorAdapter:
    def __init__(self):
        self.supervisor=Supervisor('general_pickup',200)
        self.last_step=0
        self.initial_baseline=0.0

    def begin(self,stage,state):
        self.supervisor.begin_stage(STAGES[stage],state)

    def preflight(self,state,*,planned_actions=None,max_settle_steps=None,
                  finish_reserve=60,is_finishing=False,kind=None):
        # The caller must expand sync/settle into native control actions, not
        # count strings or add both-arm motions that execute simultaneously.
        bounded=(planned_actions is not None and max_settle_steps is not None)
        if bounded and (not isinstance(max_settle_steps,int) or max_settle_steps<0):
            raise ValueError('Maximum settling steps must be a nonnegative integer')
        return self.supervisor.budget(state,
            planned_steps=len(planned_actions)+max_settle_steps if bounded else None,
            includes_settle=bounded,finish_reserve=finish_reserve,
            is_finishing=is_finishing,kind=kind)

    def after(self,state,*,kind,action_ok,checkpoint_step,phase_complete,recovery_fits=False):
        self.supervisor.record_cost(kind,self.last_step,state.step)
        self.last_step=state.step
        feedback=self.supervisor.feedback(state)
        acceptance=self.supervisor.assess(state,action_ok=action_ok,checkpoint_step=checkpoint_step,
            phase_complete=phase_complete,recovery_fits=recovery_fits)
        return {'uses_privileged_reward_feedback':True,'initial_baseline':self.initial_baseline,
                'feedback':feedback,'acceptance':acceptance,
                'prompt_text':'Privileged collection diagnostics: '+str(feedback)+'; stage acceptance: '+str(acceptance)}
