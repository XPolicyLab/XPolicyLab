"""ANS14: ANS13 plus centred-blob grasp evidence, lenient repair thresholds and a one-shot
identity re-check with both arms lifted when the model disputes the initial reading.

ANS12 dev5 layout 0: the 2 pinched at its bottom bar was visibly held (model
0.9+) but the wrist heuristic's contact test wants the blob to extend below the
fingertips, so the local status stayed unknown and the digit was put back.
eval15 seed1 layout 10: 8 and 5, each partly behind a parked gripper, were
swapped by the detector; the initial model review said so from the wrist views.
Repairs for 9-29 deg rotations or 12-15 mm offsets cost ~130 steps and made the
7 worse while the official tolerance is 45 deg / 20 mm.
"""
import numpy as np
from .ans13_policy import Policy as ANS13, CarryOverSupervisor
from .ans8_policy import DegradingSupervisor
from .ans10_policy import FREE_STEP
from .ans11_policy import VIEW_CLEAR_LIFT, VIEW_CLEAR_STEPS
from .contracts import ARMS
from .geometry import bounded_pose

REPAIR_ANGLE=.50          # rad; official check is 45 deg
REPAIR_CENTRE=.015        # m; official check is 20 mm
BLOB_AREA=.03
BLOB_CENTRE_TOLERANCE=.12


class IdentityRecheckSupervisor(CarryOverSupervisor):
    def __init__(self,inner,policy):
        super().__init__(inner);self.policy=policy
    def review(self,stage,step,payload,frames,allowed):
        if stage=='initial' and not self.policy.identity_recheck_done:
            answer=DegradingSupervisor.review(self,stage,step,payload,frames,allowed)   # raw model opinion once (degradation only)
            if answer['action']!='continue' or answer['placement']=='wrong_digit':
                answer=dict(answer,action='observe',identity_dispute=True)
            return answer
        return super().review(stage,step,payload,frames,allowed)


class Policy(ANS13):
    def __init__(self,task,calibration,supervisor=None):
        super().__init__(task,calibration,supervisor)
        self.supervisor=IdentityRecheckSupervisor(self.supervisor.inner,self)
        self.identity_recheck_done=False;self.identity_rechecks=[];self.lift_both=None

    # (1) centred stable blob between the fingers counts as coarse contact evidence
    def _evidence(self,obs,context,observations):
        result=super()._evidence(obs,context,observations)
        if result['local_status']=='unknown' and not result['coarse_stable_contact'] and result['lift_metres']>=.035 and len(result['wrist_observations'])==2:
            a,b=result['wrist_observations']
            objects=[(o.get('object') or {}) for o in (a,b)]
            if all(o.get('available') for o in (a,b)) and all(x.get('area',0)>BLOB_AREA and abs(x.get('centre',[0,0])[0]-.5)<BLOB_CENTRE_TOLERANCE for x in objects):
                if np.linalg.norm(np.asarray(objects[0]['centre'])-objects[1]['centre'])<.035 and .65<objects[0]['area']/objects[1]['area']<1.55:
                    result['coarse_stable_contact']=True;result['coarse_source']='centred_blob'
                    if self.grasp_checks:self.grasp_checks[-1].update(coarse_stable_contact=True,coarse_source='centred_blob')
        return result

    # (2) repair only beyond what the official tolerance would already forgive generously
    def _repair_plan(self,obs,context,found):
        if found is None:return None
        error=float(np.linalg.norm(np.asarray(found['xy'])-context['target']));angle=abs(float(found['yaw']))
        if error<REPAIR_CENTRE and angle<REPAIR_ANGLE:return None
        return super()._repair_plan(obs,context,found)

    # (3) disputed identities: lift both arms once, re-detect, then proceed on local evidence
    def _defer_or_stop(self,obs,pending,reason):
        if pending.get('stage')=='initial' and not self.identity_recheck_done:
            self.identity_recheck_done=True
            self.lift_both=dict(goals={s:np.r_[np.asarray(self.home[s],float)[:3]+[0.,0.,VIEW_CLEAR_LIFT],np.asarray(self.home[s],float)[3:]] for s in ARMS},age=0)
            self.identity_rechecks.append(dict(step=self.step,cause=reason,detected=[(d['digit'],round(d.get('confidence',0),3)) for d in self.digits]))
            self._event('identity_recheck_lift_both',None)
            return self._stationary(obs['state'])
        if pending.get('stage')=='initial':
            # Second opinion already taken: the local reading stands.
            self.identity_rechecks[-1].update(after=[(d['digit'],round(d.get('confidence',0),3)) for d in self.digits],still_disputed=True)
            self._event('identity_recheck_exhausted_proceed',None);self.pending=None
            return self._stationary(obs['state'])
        return super()._defer_or_stop(obs,pending,reason)

    def _act(self,obs):
        if self.lift_both is not None:
            lb=self.lift_both;state=obs['state'];lb['age']+=1;action=self._stationary(state);remaining=0
            for side,goal in lb['goals'].items():
                current=np.asarray(state[side+'_ee_pose'],float)
                if np.linalg.norm(current[:3]-goal[:3])>=.009:
                    remaining+=1;action[side+'_ee_pose']=bounded_pose(current,goal,FREE_STEP,self.calibration['max_rotation_step'])
            if remaining and lb['age']<=VIEW_CLEAR_STEPS:
                self.step+=1;self.reason='identity_recheck_lift';return action
            self.lift_both=None;self.identity_rechecks[-1].update(lift_steps=lb['age'],arrived=not remaining)
            home=self.home;result=super()._act(obs);self.home=home;return result
        return super()._act(obs)

    def diagnostics(self):
        result=super().diagnostics()
        result.update(candidate='ANS14',identity_rechecks=self.identity_rechecks,repair_thresholds=dict(angle_rad=REPAIR_ANGLE,centre_m=REPAIR_CENTRE),centred_blob_area=BLOB_AREA)
        return result
