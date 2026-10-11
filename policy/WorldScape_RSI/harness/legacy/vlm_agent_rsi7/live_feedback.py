"""RSI7 observation-only instrumentation of native scoring and step budgeting."""
import json,os
from pathlib import Path

def install(env):
    if hasattr(env,'_rsi7_event'):
        return
    env._rsi7_event=None
    env._rsi7_event_n=0
    env._rsi7_previous=None
    rm=env.reward_manager
    original=rm._evaluate_score_entries
    def observed(*args,**kwargs):
        # Called ONLY by native rm.step. Do not evaluate any predicate twice.
        idx=args[0]
        process=args[1] is rm.score_list
        pending=process and rm.score_completed_count[idx]<len(rm.score_list[idx])
        result=original(*args,**kwargs)
        if pending:
            env._rsi7_event_n+=1
            env._rsi7_event=(int(env.take_action_cnt[0]),env._rsi7_event_n)
        return result
    rm._evaluate_score_entries=observed

def snapshot(env):
    step=int(env.take_action_cnt[0]);rm=env.reward_manager
    gated=rm._gated_score_lst
    score=float(gated[0]) if gated is not None else None
    ev=env._rsi7_event
    n=int(rm.score_completed_count[0])
    return dict(native_step=step,remaining_steps=max(0,400-step),native_step_limit=400,
                current_score=score,score_scale='raw_0_100',score_source='native_gated_cache',
                score_event=f'native-eval-{ev[1]}' if ev else None,
                score_event_step=ev[0] if ev else None,
                scoring_trigger=True if ev and ev[0]==step else None,
                historical={'middle_transition':n>=1,'top_transition':n>=2},
                current={'object_retained':None,'tower_stable':None},
                terminal=bool(env.end_flag[0]),
                native_success=bool(env.end_flag[0] and env.success[0]),
                uses_privileged_reward_feedback=True)

class Results(list):
    def __init__(self,env):
        super().__init__();self.env=env
    def append(self,result):
        row=snapshot(self.env)
        prev=self.env._rsi7_previous
        row['delta_score']=None if prev is None or row['current_score'] is None else row['current_score']-prev
        if row['current_score'] is not None:self.env._rsi7_previous=row['current_score']
        row['action_id']=result.get('command')
        row['action_ok']=result.get('ok')
        result['rsi7_feedback']=row
        with (Path(os.environ['RSI7_RUN'])/'feedback.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        super().append(result)

def check_budget(env,cmd,n,config,motion_hold):
    step=int(env.take_action_cnt[0])
    assert env.step_lim==400
    reserve=0 if cmd.kind in ('home','homeplan') else 20
    settle=int(round(float((config.get('gripper_settle') or {}).get('max_s',1))*25))
    gripper_only = cmd.kind == "gripper" or (cmd.kind == "sync" and bool(cmd.value) and all(v["kind"] == "gripper" for v in cmd.value))
    bound = max(n,settle) if gripper_only else (max(n+motion_hold,settle) if cmd.kind == "sync" else n+motion_hold)
    allowed=bound+reserve<=400-step
    with (Path(os.environ['RSI7_RUN'])/'budget.jsonl').open('a') as f:
        f.write(json.dumps(dict(step=step,command=cmd.text(),bound=bound,finish_reserve=reserve,allow=allowed))+'\n')
    return allowed
