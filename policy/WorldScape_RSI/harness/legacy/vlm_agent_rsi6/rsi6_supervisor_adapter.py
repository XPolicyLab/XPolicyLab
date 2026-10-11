"""Task-local integration of frozen pure supervisor; no reward calls."""
import json
from pathlib import Path
from .rsi_supervisor_v1 import Supervisor, Snapshot, Evidence, StageContract

REQUIRED = {'approach': ('controller_reached',), 'grasp': ('controller_reached',),
            'lift': ('retained',), 'transport': ('retained',),
            'place': ('aligned_supported',), 'release': ('released_on_pad',),
            'return': ('controller_reached',), 'recover': ('controller_reached',)}

class EquationSupervisor:
    def __init__(self, run):
        self.core=Supervisor('solve_equation',300)
        self.path=Path(run)/'supervisor_feedback.jsonl'
        self.pending=None
        self.phase='approach'

    def snap(self, env, current=None):
        step=int(env.take_action_cnt[0])
        # Native take_action and the executor's existing stop check have returned.
        terminal=bool(env.end_flag[0])
        return Snapshot(step=step,score=float(terminal and env.success[0]),score_step=step,
            score_source='native end_flag AND success; audited binary mapping',score_scale='binary_0_1',
            terminal=terminal,native_success=bool(env.success[0]) if terminal else None,
            score_event=None,trigger=Evidence(),current=current or {})

    def log(self, value):
        with self.path.open('a') as out: out.write(json.dumps(value)+'\n')

    def before(self, env, plan):
        step=int(env.take_action_cnt[0])
        result=None
        if self.pending and self.pending["action_ok"] is not True:
            result=self.pending["assessment"]
            if result["decision"] == "recover":
                self.pending=None
            else:
                return result
        if self.pending:
            acceptance=plan.get('acceptance') or {}
            evidence={}
            if acceptance.get('step') == step:
                evidence={k:Evidence(v,step,'model RGB assessment') for k,v in acceptance.get('checks',{}).items()
                          if isinstance(v,bool)}
            if self.pending['action_ok'] is True:
                evidence['controller_reached']=Evidence(True,step,'audited executor motion/gripper result')
            result=self.core.assess(self.snap(env,evidence),action_ok=self.pending['action_ok'],
                checkpoint_step=self.pending['step'],phase_complete=True,recovery_fits=300-step>90)
            self.log({'kind':'phase_acceptance','native_step':step,'result':result,
                      'evidence':{k:vars(v) for k,v in evidence.items()}})
            if result['decision'] not in ('advance_stage','recover','continue_stage'):
                return result
            self.pending=None
        phase=plan.get('phase','approach')
        if phase not in REQUIRED: raise ValueError('Unknown equation phase')
        self.phase=phase
        return result

    def begin(self, env):
        self.core.begin_stage(StageContract(self.phase,required_current=REQUIRED[self.phase],
            require_trigger=False,max_recoveries=1),self.snap(env))

    def budget(self, env, kind, count):
        result=self.core.budget(self.snap(env),planned_steps=count,includes_settle=count is not None,
            finish_reserve=45,is_finishing=(self.phase=='return' and kind=='homeplan'),kind=kind)
        self.log({'kind':'budget','phase':self.phase,'native_step':int(env.take_action_cnt[0]),
                  'full_planned_steps':count,'result':result})
        return result

    def after(self, env, result):
        before=result['steps_before']; after=int(env.take_action_cnt[0])
        self.core.record_cost(result['kind'],before,after)
        evidence={'controller_reached':Evidence(result.get('ok'),after,'audited executor result')}
        snap=self.snap(env,evidence)
        feedback=self.core.feedback(snap)
        decision=self.core.assess(snap,action_ok=result.get('ok'),checkpoint_step=after,
            phase_complete=True,recovery_fits=300-after>90)
        self.pending={'action_ok':result.get('ok'),'step':after,'assessment':decision}
        self.log({'kind':'primitive_end','action':result['command'],'feedback':feedback,'assessment':decision})
        return {'feedback':feedback,'assessment':decision}
