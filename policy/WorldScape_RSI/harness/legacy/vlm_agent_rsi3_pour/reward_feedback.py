"""RSI2 pour adapter: cached native score plus separately authorized read-only fluid predicate diagnostics.
Diagnostic channel and JSONL sidecar, never ordinary SFT observation fields.
"""
import json,math,os,time
from .live_fluid_diagnostic import evaluate as live_fluid
from pathlib import Path

def enabled():
    return os.environ.get('ROBODAWN_REWARD_FEEDBACK','0')=='1' and os.environ.get('ROBODAWN_RUN_MODE','benchmark') in ('collection','diagnostic')

def scalar(x):
    try:
        v=float(x)
        return v if math.isfinite(v) else None
    except (ValueError,TypeError,OverflowError):return None

def cached(env):
    step=int(env.take_action_cnt[0]);limit=int(env.native_step_limit if hasattr(env,'native_step_limit') else env.step_lim)
    rm=env.reward_manager;cache=getattr(rm,'_gated_score_lst',None)
    task=getattr(env,'task_name',None)
    score=None;status='unknown:no_gated_cache'
    if cache is not None:
        try:score=scalar(cache[0]);status='available' if score is not None else 'unknown:invalid_cached_score'
        except (IndexError,TypeError):status='unknown:invalid_cached_score'
    success=None
    try:success=bool(env.end_flag[0]) and bool(env.success[0])
    except (AttributeError,IndexError,TypeError):pass
    count=None
    if task=='pour_by_language':
        try:
            meta=rm.score_meta[0]
            if meta['mode']=='by_count' and list(meta['gradient'])==[20,50,100]:
                value=rm.score_completed_count[0]
                if isinstance(value,int) and 0<=value<=3:count=value
        except (AttributeError,KeyError,IndexError,TypeError):pass
    trigger=None
    if task=='pour_by_language':
        try:
            meta=rm.score_trigger_meta[0]
            if meta['trigger_mode']=='rising_edge' and isinstance(meta['pre_status'],bool):trigger=meta['pre_status']
        except (AttributeError,KeyError,IndexError,TypeError):pass
    return {'trigger_status':{'native_home_or_cached':trigger,'source':'score_trigger_meta[0].pre_status; native OR(left_home,right_home), NOT both-home; no independent timestamp'},'current_score':score,'native_success':success,'native_step':step,'remaining_steps':max(0,limit-step),
      'native_step_limit':limit,'status':status,'score_source':'RewardManager._gated_score_lst[0]',
      'score_scale':{'raw':[0,100] if task=='pour_by_language' else None,'display':'identity' if task=='pour_by_language' else None,'native_result_detail_scale':[0,1] if task=='pour_by_language' else None},
      'stage_conditions':{'current':None,'historical':None,'historical_completed_count':count,'source':'score_completed_count (cached aggregate only)','unknown_reason':'No side-effect-free cached per-condition truth/identity contract verified'},
      'freshness':{'basis':'synchronous native take_action returns after reward_manager.step and is_episode_end/get_reward','cache_timestamp_available':False,'sampled_at_native_step':step}}

class Feedback:
    def __init__(self,sidecar):
        self.path=Path(sidecar);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.trigger_previous=None;self.trigger_false_step=None;self.trigger_rise_step=None;self.previous=None;self.baseline=None;self.events=[];self.latest=None;self.ordinal=0
    def sample(self,env,action_id,phase,reason=None,action_status=None):
        row=cached(env);row['live_fluid_diagnostic']=live_fluid(env);score=row['current_score'];self.ordinal+=1
        if score is not None and self.baseline is None:
            self.baseline={'score':score,'native_step':row['native_step'],'action_id':action_id}
            delta=0.0
        elif score is not None and self.previous is not None:delta=score-self.previous
        else:delta=None
        row.update(schema='rsi.reward_feedback.v1',uses_privileged_reward_feedback=True,task=getattr(env,'task_name',None),
                   action_id=action_id,event_index=self.ordinal,phase=phase,decision_reason=reason,action_status=action_status,
                   delta_score=delta,delta_reference='previous available adjacent boundary; first valid sample establishes baseline',initial_baseline=self.baseline,
                   observed_unix_s=time.time())
        contract=(reason or {}).get('plan',{}).get('stage_contract',{}) if isinstance(reason,dict) else {}
        armed=contract.get('armed',False);expected=contract.get('expected_min_score');trigger=row['trigger_status']['native_home_or_cached']
        step=row['native_step']
        if trigger is False and step>0:self.trigger_false_step=step
        if trigger is True and self.trigger_previous is False and self.trigger_false_step is not None:self.trigger_rise_step=step
        self.trigger_previous=trigger
        row['trigger_status'].update(observed_false_step=self.trigger_false_step,observed_rise_step=self.trigger_rise_step)
        row['official_refresh_observation']='score_changed' if delta not in (None,0.0) else ('trigger_edge_observed' if self.trigger_rise_step==step and step>0 else 'no_new_edge_observed_at_action_boundaries')
        acceptance='not_armed';early=None
        if armed:
            if action_status and action_status.get('ok') is False and action_status.get('steps_cost')==0:
                acceptance='execution_failed:no_motion'
            elif score is None or trigger is None:acceptance='unknown'
            elif not trigger:acceptance='trigger_pending'
            elif expected is None:acceptance='unknown'
            elif score>=expected:acceptance='passed'
            elif self.trigger_rise_step is None or self.trigger_rise_step<(contract.get('trigger_after_step') or 0):
                acceptance='unknown:no_post_pour_edge'
            else:acceptance='failed';early='stage_score_not_reached_after_observed_native_edge'
        row.update(stage_contract=contract or None,stage_acceptance=acceptance,early_stop_reason=early,
          zero_step_failure=bool(action_status and action_status.get('ok') is False and action_status.get('steps_cost')==0))
        self.previous=score;self.latest=row;self.events.append(row)
        with self.path.open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        return row
    def observation(self):
        result={'uses_privileged_reward_feedback':True,'latest':self.latest,'primitive_events':list(self.events),'not_sft_observation':True}
        self.events.clear();return result
