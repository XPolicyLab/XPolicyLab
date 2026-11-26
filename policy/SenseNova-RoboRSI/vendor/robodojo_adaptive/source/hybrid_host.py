"""Enforce GPT review before native EEF execution, with configured waypoint spans."""
from copy import deepcopy
import time
from policylab.api import InvalidAction
from policylab.common import event
from policylab import runner
from adaptive_steps import configured_adaptive_steps

ORIGINAL=runner.Experiment.request

def request(self, req):
    op=req.get('op') if isinstance(req,dict) else None
    if op not in ('vla_propose','vla_review'):
        if op=='act' and not getattr(self,'hybrid_executing',False):
            raise InvalidAction('Use vla_propose and vla_review before every EEF execution')
        result=ORIGINAL(self,req)
        if op=='observe' and getattr(self,'vla_pending',None) is not None:
            result=deepcopy(result)
            result['pending_vla_proposal']=deepcopy(self.vla_pending)
        return result
    with self.lock:
        self.request_sequence+=1
        event(self.run/'requests.jsonl',sequence=self.request_sequence,request=req)
        if time.monotonic()>=self.deadline or self.segment_done or self.fault:
            raise InvalidAction('Segment ended or environment unavailable')
        if self.phase!='evaluation' or self.active is None or self.latest['done']:
            raise InvalidAction('VLA is available only during active evaluation')
        if self.review_status(self.latest).get('review_due'):
            raise InvalidAction('Call review(analysis) at the checkpoint first')
        pending=getattr(self,'vla_pending',None)
        limits=configured_adaptive_steps(self.cfg.get('options',{}))
        if op=='vla_propose':
            if set(req)!={'op','subgoal'}: raise InvalidAction('Use vla_propose(subgoal)')
            if pending is not None: raise InvalidAction('Review the pending proposal first')
            try:
                proposal=self.backend.request(op='vla_propose',subgoal=req['subgoal'])
            except InvalidAction: raise
            except Exception:
                import traceback
                self.fault=traceback.format_exc()
                (self.run/'adapter_error.txt').write_text(self.fault)
                self.finalize('adapter_error')
                raise
            if proposal['control_step']!=self.latest['observation']['control_step'] or len(proposal['targets'])!=self.cfg['options']['vla_horizon']:
                raise RuntimeError('Invalid proposal provenance/span')
            proposal.update(limits)
            self.vla_pending=proposal
            event(self.active['record_dir']/'vla_review.jsonl',event='proposal',**proposal)
            return dict(status='awaiting_review',**deepcopy(proposal))
        if set(req)-{'approved_steps'}!={'op','proposal_id','decision','reason','assessment','targets'}:
            raise InvalidAction('Use vla_review(proposal_id, decision, reason, assessment, targets=None)')
        if pending is None or req['proposal_id']!=pending['proposal_id']:
            raise InvalidAction('Current proposal_id required')
        if pending['control_step']!=self.latest['observation']['control_step']:
            raise InvalidAction('Stale proposal cannot execute')
        decision=req['decision']; supplied=req['targets']
        span=req.get('approved_steps',limits['approved_steps_default'])
        if type(span) is not int or not limits['approved_steps_min']<=span<=limits['approved_steps_max']:
            raise InvalidAction(
                f"approved_steps must be an integer from {limits['approved_steps_min']} to {limits['approved_steps_max']}; "
                f"default is {limits['approved_steps_default']}")
        if not isinstance(req['reason'],str) or not req['reason'].strip(): raise InvalidAction('Evidence-based review reason required')
        assessment=req['assessment']
        if not isinstance(assessment,dict) or set(assessment)!={'current_subgoal'} or not isinstance(assessment['current_subgoal'],str) or not assessment['current_subgoal'].strip():
            raise InvalidAction('Provide assessment.current_subgoal')
        if decision=='accept':
            if supplied is not None: raise InvalidAction('accept must omit replacement targets')
            if any(pending['out_of_bounds'][:span]): raise InvalidAction('Cannot accept out-of-bounds proposal; modify or reject')
            targets=deepcopy(pending['targets'][:span])
        elif decision in ('modify','reject'):
            if decision=='reject' and supplied is None:
                event(self.active['record_dir']/'vla_review.jsonl',event='review',proposal_id=pending['proposal_id'],decision=decision,approved_steps=span,reason=req['reason'],executed_steps=0)
                self.vla_pending=None
                return dict(status='rejected',robot_moved=False,**self.public)
            if not isinstance(supplied,list) or len(supplied)!=span: raise InvalidAction(f"Supply exactly {span} full dual-arm EEF targets")
            targets=deepcopy(supplied)
            if decision=='modify' and targets==pending['targets'][:span]: raise InvalidAction('modify requires an actual change')
        else: raise InvalidAction('decision must be accept, modify or reject')
        # Validate both target structures before any movement; pose bounds are rechecked from measured state at each native step.
        import math
        for dual in targets:
            if not isinstance(dual,dict) or set(dual)!={'left','right'}: raise InvalidAction('Explicit left/right targets required')
            for value in dual.values():
                if not isinstance(value,dict) or set(value)!={'position','quaternion_wxyz','gripper_closed'}: raise InvalidAction('Invalid EEF target')
                for name,count in [('position',3),('quaternion_wxyz',4)]:
                    seq=value[name]
                    if not isinstance(seq,list) or len(seq)!=count or any(type(x) not in (int,float) or not math.isfinite(x) for x in seq): raise InvalidAction('Finite position and quaternion required')
                if abs(sum(x*x for x in value['quaternion_wxyz'])**.5-1)>1e-4 or type(value['gripper_closed']) is not bool:
                    raise InvalidAction('Unit quaternion and boolean gripper required')
        event(self.active['record_dir']/'vla_review.jsonl',event='review',proposal_id=pending['proposal_id'],decision=decision,reason=req['reason'],assessment=assessment,approved_steps=span,selected_targets=targets)
        before=self.latest['observation']['control_step']
        self.hybrid_executing=True
        try:
            for target in targets:
                ORIGINAL(self,dict(op='act',repeat=1,action=dict(mode='eef',target=target,reason=req['reason'],assessment=assessment)))
                if self.latest['done'] or self.public['observation'].get('review_due'): break
        finally:
            moved=self.latest['observation']['control_step']-before
            self.hybrid_executing=False
            # A zero-movement validation rejection keeps the proposal repairable.
            if moved: self.vla_pending=None
            event(self.active['record_dir']/'vla_review.jsonl',event='execution',proposal_id=pending['proposal_id'],approved_steps=span,executed_steps=moved,remaining_queue_cancelled=moved<len(targets))
        self.vla_pending=None
        return dict(status='executed',decision=decision,executed_steps=moved,**self.public)

def install():
    runner.Experiment.request=request
