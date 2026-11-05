"""ANS13: ANS12 plus possession carry-over when the wrist view is unresolved at pre_place.

eval15 seed0 layout 4: over the pale leftmost pad the wrist finger detector
returned fingers_not_resolved, the local status fell to unknown although the
digit had been confirmed held after lift and the model still saw it held, and
the bounded observation ended in a put-back onto the staging spot. A hold
confirmed at post_lift now carries over to pre_place while no observation shows
an open, empty gripper; a confident model 'dropped/empty' still overrides it.
"""
import numpy as np
from .ans12_policy import Policy as ANS12
from .ans9_policy import DecisiveSupervisor


class CarryOverSupervisor(DecisiveSupervisor):
    def review(self,stage,step,payload,frames,allowed):
        evidence=payload.get('local_evidence') or {}
        if stage=='pre_place' and evidence.get('carried_over'):
            answer=super(DecisiveSupervisor,self).review(stage,step,payload,frames,allowed)
            if answer.get('possession') in ('empty','dropped') and answer.get('confidence',0)>=.8:
                return dict(answer,carried_over_vetoed=True)
            if answer['action']!='continue' or answer['possession'] not in ('held_clear','held_occluded') or answer['placement'] in ('misaligned','wrong_digit','unknown'):
                answer=dict(answer,action='continue',possession='held_clear' if answer['possession'] not in ('held_clear','held_occluded') else answer['possession'],
                    placement='aligned',advisory_only=dict(action=answer['action'],possession=answer['possession'],placement=answer['placement']))
            return answer
        return super().review(stage,step,payload,frames,allowed)


class Policy(ANS12):
    def __init__(self,task,calibration,supervisor=None):
        super().__init__(task,calibration,supervisor)
        self.supervisor=CarryOverSupervisor(self.supervisor.inner)
        self.carry_overs=[]

    def _evidence(self,obs,context,observations):
        result=super()._evidence(obs,context,observations)
        stage=self.pending.get('stage') if isinstance(self.pending,dict) else None
        if (stage=='pre_place' and result['local_status']=='unknown'
                and self.route_states.get(str(context['digit']))=='held_confirmed'
                and not any(o.get('available') and not o.get('contact') for o in result['wrist_observations'])):
            result['local_status']='held';result['carried_over']=True
            if self.grasp_checks:self.grasp_checks[-1].update(status='held',carried_over=True)
            self.carry_overs.append(dict(step=self.step,label=context['label'],reasons=[o.get('reason') for o in result['wrist_observations']]))
        return result

    def diagnostics(self):
        result=super().diagnostics()
        result.update(candidate='ANS13',possession_carry_overs=self.carry_overs)
        return result
