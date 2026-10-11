"""Local rule-based judgement/execution verifier; no network, keys or learned weights."""
import time
class Jev:
    def __init__(self, model='local-jev-v1', timeout=None, transport=None, audit=lambda e:None):
        if transport is not None: raise ValueError('Network transports disabled')
        self.audit=audit
    def choose(self,purpose,state,choices):
        start=time.monotonic();o=state.get('observation',{});f=o.get('facts',{});m=o.get('motion',{});s=state.get('stage',{})
        def holds(p):return bool(p) and all(f.get(k) is True for k in p)
        def norm(v):return ' '.join(str(v).lower().split())
        if 'keep_executor' in choices:
            ready=[k for k,p in choices.items() if isinstance(p,dict) and holds(p.get('preconditions',[]))]
            d=ready[0] if len(ready)==1 else 'keep_executor'
            reason='unique_eligible_code_skill_or_keep_executor'
        elif 'new_plan' in choices:
            match=[k for k,p in choices.items() if isinstance(p,dict) and norm(p.get('goal'))==norm(state.get('instruction'))]
            ready=[k for k in match if not choices[k].get('preconditions') or holds(choices[k]['preconditions'])]
            d=ready[0] if len(ready)==1 else ('observe' if match and not f else 'new_plan')
            reason='exact_goal_reuse_or_request_planning'
        elif 'continue' not in choices:
            ready=[k for k,p in choices.items() if isinstance(p,dict) and p.get('validated') and holds(p.get('preconditions',[]))]
            d=ready[0] if len(ready)==1 else 'escalate';reason='verified_recovery_matching'
        elif f.get('stop_recommended') is True:d='stop';reason='explicit_observation_based_stop'
        elif o.get('remaining',0)<=0:d='stop';reason='budget'
        elif m.get('hard_error'):d='recover';reason='execution_error'
        elif holds(s.get('completion',[])):d='complete';reason='current_completion_evidence'
        elif f.get('progressing') is True:d='continue';reason='current_visual_progress'
        elif f:d='escalate';reason='inspection_did_not_establish_progress'
        else:
            recent=m.get('recent_execution',[]);tail=recent[-3:]
            stalled=len(tail)==3 and all(r.get('motion',{}).get('joint_delta_l2',1)<1e-4 and bool(r.get('motion',{}).get('image_change_mean_abs')) and max(r['motion']['image_change_mean_abs'].values())<0.25 for r in tail)
            d='observe' if stalled or len(recent)<2 else 'continue'
            reason='stagnation_or_initial_check' if d=='observe' else 'bounded_execution_until_semantic_refresh'
        if d not in choices:raise ValueError('Unsupported local decision')
        a={'choice':d,'confidence':1.0,'probabilities':{k:float(k==d) for k in choices},'reason':reason,'confidence_semantics':'deterministic_rule_not_physical_success_probability'}
        self.audit({'kind':'jev','purpose':purpose,'input':state,'answer':a,'model':'local-jev-v1','usage':{'input_tokens':0,'output_tokens':0},'elapsed_s':time.monotonic()-start})
        return a
