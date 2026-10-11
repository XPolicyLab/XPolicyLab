"""Private RSI4 primitive boundary diagnostics; no reward predicate calls."""
import json, os, math
from pathlib import Path
from .rsi_supervisor_v1 import Supervisor, Snapshot, StageContract, Evidence

class Live:
    def __init__(self, env):
        assert env.step_lim == 200
        self.s = Supervisor('align_blocks', 200)
        self.path = Path(os.environ['RSI4_RUN'])/'primitive_feedback.jsonl'
        self.index = 0
        self.previous_kind = None
    def snap(self, env, current=None):
        terminal = bool(env.end_flag[0])
        return Snapshot(step=int(env.take_action_cnt[0]), terminal=terminal,
            native_success=bool(env.success[0]) if terminal else None,
            score_scale='binary_episode_0_1', current=current or {})
    def begin(self, env, cmd, settle):
        self.index += 1
        self.before = int(env.take_action_cnt[0]); self.cmd = cmd
        self.clearance = self.previous_kind == "push" and cmd.kind == "sync" and all(v["kind"] == "move3d" and v["value"][2] > 0 and v["value"][1] <= 0 and abs(v["value"][0]) < .01 for v in cmd.value)
        self.checked = False; self.budget_result = None
        self.gripper_bound = math.ceil(settle.max_s/settle.dt)
        self.finishing = cmd.kind in ('home','homeplan') or (cmd.kind=='sync' and
            all(v['kind'] in ('home','homeplan') for v in cmd.value))
        self.s.begin_stage(StageContract(name=cmd.kind,require_trigger=False,
            required_current=('controller_postcondition',),max_recoveries=1),self.snap(env))
        self.s.feedback(self.snap(env))
    def preflight(self, env, full_steps):
        if self.checked: return self.budget_result['decision']=='allow'
        self.budget_result=self.s.budget(self.snap(env),planned_steps=full_steps,
            includes_settle=True,finish_reserve=25 if self.clearance else 40,is_finishing=self.finishing)
        self.checked=True
        return self.budget_result['decision']=='allow'
    def end(self, env, result):
        step=int(env.take_action_cnt[0]); ok=result.get('ok')
        if self.budget_result and self.budget_result['decision']!='allow':
            ok=False; result.update(ok=False,note='RSI4 budget preflight rejected; no action: '+str(self.budget_result))
        snap=self.snap(env,{'controller_postcondition':Evidence(ok,step,'executor measured robot outcome'),
                           'object_stage_acceptance':Evidence(None,step,'requires current RGB assessment')})
        self.s.record_cost(self.cmd.kind,self.before,step)
        feedback=self.s.feedback(snap)
        # Object acceptance is checked on the next RGB observation, not inferred
        # from robot convergence. One primitive per model decision enforces this.
        decision=self.s.assess(snap,action_ok=ok,checkpoint_step=step,
            phase_complete=True,recovery_fits=step<160)
        record=dict(action_id=self.index,command=self.cmd.text(),before_step=self.before,
            executed_steps=step-self.before,feedback=feedback,budget=self.budget_result,
            controller_acceptance=decision,object_acceptance='unknown_until_RGB',
            uses_privileged_reward_feedback=True)
        with self.path.open('a') as f:f.write(json.dumps(record,default=str)+'\n')
        result['rsi4_feedback']=record
        env._rsi4_latest=record
        self.previous_kind = "push" if self.cmd.kind == "sync" and all(v["kind"] == "push3d" for v in self.cmd.value) else self.cmd.kind

def start(env):
    if not hasattr(env,'_rsi4_live'):env._rsi4_live=Live(env)
    return env._rsi4_live
