"""ANS9: ANS8 plus a digit-2 antipodal grasp, wide dropped-digit relocation, and tilted reach for far pads.

Digit 2 is pinched across its height (top arc / bottom bar) instead of across
the slanted middle stroke, which let the fingers slide into the bottom notch.
A pad just beyond the vertical reach limit is served with the gripper pitched
forward so the EE stays inside the observed workspace. RGB/proprioception only.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from .ans8_policy import Policy as ANS8, DegradingSupervisor, REACH_BASE, REACH_LIMIT, route_digit
from .ans1_policy import Policy as ANS1
from .geometry import tcp_to_ee, quat_matrix

Y_PINCH_DIGITS={2}
TILT=np.deg2rad(15.)
TILT_SHIFT=.145*np.sin(TILT)   # EE moves this much toward the robot for the same TCP


class DecisiveSupervisor(DegradingSupervisor):
    """Local evidence decides. Confirmed-empty selects the checked retry/skip; confirmed-held proceeds.

    ANS8 dev5 layout 10: the model vetoed three geometrically correct pre_place
    descents as misaligned/wrong_digit and the digits were put back. The model
    may now only veto a confirmed-empty retry with a confident held, and its
    placement opinion is recorded, not executed, while the wrist confirms the hold.
    """
    def review(self,stage,step,payload,frames,allowed):
        answer=super().review(stage,step,payload,frames,allowed)
        local=(payload.get('local_evidence') or {}).get('local_status')
        if stage=='initial' and (answer['action']!='continue' or answer['placement']=='wrong_digit'):
            answer=dict(answer,action='continue',placement='not_applicable',advisory_only=dict(action=answer['action'],placement=answer['placement']))
        elif stage in ('post_lift','pre_place') and local=='held':
            if answer['action']!='continue' or answer['possession'] not in ('held_clear','held_occluded') or answer['placement'] in ('misaligned','wrong_digit','unknown'):
                answer=dict(answer,action='continue',possession='held_clear' if answer['possession'] not in ('held_clear','held_occluded') else answer['possession'],
                    placement='aligned',advisory_only=dict(action=answer['action'],possession=answer['possession'],placement=answer['placement']))
        elif local=='empty' and not (answer.get('possession')=='held_clear' and answer.get('confidence',0)>=.9):
            if 'retry' in allowed and answer['action']!='retry':
                answer=dict(answer,action='retry',possession='empty',decisive='local_empty_retry')
            elif 'skip_empty' in allowed and answer['action'] not in ('retry','skip_empty'):
                answer=dict(answer,action='skip_empty',possession='empty',decisive='local_empty_skip')
        return answer


class Policy(ANS8):
    def __init__(self,task,calibration,supervisor=None):
        super().__init__(task,calibration,supervisor)
        self.supervisor=DecisiveSupervisor(self.supervisor.inner)
        self.tilted_goals=set();self.relocations=[];self.grasp_variants=[]

    # ---- routing: pitched gripper extends the TCP reach straight ahead ----
    def options(self,xy,goal,yaw):
        choices=super().options(xy,goal,yaw)
        if choices:return choices
        for cost,side,height in ANS1.options(self,xy,goal,yaw):
            cross=(side=='left' and goal[0]>.03) or (side=='right' and goal[0]<-.03)
            shifted=np.asarray(goal,float)-[0.,TILT_SHIFT]
            if cross or self.reach(side,xy)>REACH_LIMIT or self.reach(side,shifted)>REACH_LIMIT:continue
            self.tilted_goals.add(tuple(np.round(np.asarray(goal,float),3).tolist()))
            self.routing.append(dict(side=side,source=np.round(xy,3).tolist(),goal=np.round(goal,3).tolist(),
                reach=round(self.reach(side,goal),3),accepted='tilted_place',ee_reach=round(self.reach(side,shifted),3)))
            choices.append((cost+.08,side,height))
        return sorted(choices)

    def _tilted(self,pose,angle):
        offset=np.asarray(self.calibration['tool_offset_in_ee'])
        tcp=np.asarray(pose[:3])+quat_matrix(pose[3:])@offset
        r=Rotation.from_euler('z',angle).as_matrix()@Rotation.from_euler('y',np.pi/2-TILT).as_matrix()
        return tcp_to_ee(tcp,r,offset)

    def transfer(self,side,xy,goal,yaw,height,label):
        digit=int(label.split('_')[0]);tilted=tuple(np.round(np.asarray(goal,float),3).tolist()) in self.tilted_goals
        y_pinch=digit in Y_PINCH_DIGITS and not tilted
        super().transfer(side,xy,goal,yaw-np.pi/2 if y_pinch else yaw,height,label)
        own=[w for w in self.plan if w['name'].split('_',1)[1]==label]
        goal_side=[w for w in own if w['name'].split('_')[0] in ('carry','lower','place','softopen','release','clearance','retreat')]
        if y_pinch:
            # Fingers close along the digit's own y axis; the digit still lands at yaw 0.
            q=self.pose([0.,0.],0.,0.)[3:]
            for w in goal_side:w['pose']=np.r_[w['pose'][:3],q]
            self.grasp_variants.append(dict(label=label,grasp='y_pinch',grasp_angle=float(yaw),place_angle=0.))
        if tilted:
            for w in goal_side:w['pose']=self._tilted(w['pose'],np.pi/2)
            self.grasp_variants.append(dict(label=label,grasp='x_pinch',place='tilted',tilt_deg=float(np.degrees(TILT))))

    # ---- dropped digits: relocate anywhere plausible, then let local evidence drive the retry ----
    def _table_digit(self,obs,context,features):
        found=super()._table_digit(obs,context,features)
        if found is not None or not (features.get('available') and not features.get('contact')):return found
        side=context['side'];ee=np.asarray(obs['state'][side+'_ee_pose'])
        tcp=ee[:3]+quat_matrix(ee[3:])@np.asarray(self.calibration['tool_offset_in_ee'])
        rgb=obs['vision']['cam_head']['color'];best=None
        for centre in (tcp[:2],context['target'],context['source']):
            candidate=self.source_checker.locate(rgb,context['digit'],np.asarray(centre,float),context['family'],height=.014,max_distance=.16)
            if candidate is not None and candidate['confidence']>=.65 and (best is None or candidate['confidence']>best['confidence']):best=candidate
        if best is not None:
            self.relocations.append(dict(step=self.step,label=context['label'],xy=np.round(best['xy'],3).tolist(),yaw=round(float(best['yaw']),3),confidence=round(best['confidence'],3)))
        return best

    def _known_empty(self,digit):
        for check in reversed(self.grasp_checks):
            if check.get('digit')==digit:return check.get('status')=='empty'
        return False

    def _put_back(self,w,cause):
        digit=route_digit(w['name'])
        if digit is not None and self.home is not None and self._known_empty(digit):
            # Nothing is held: do not spend a put-back excursion; clear the route and park open.
            self._drop_route(digit);side=w['side']
            self.plan.insert(self.index,dict(side=side,pose=self.home[side].copy(),gripper=1.,name='empty_home_'+str(digit),dwell=3))
            self.age=self.dwell=0;self.route_states[str(digit)]='dropped_'+cause
            self._event('dropped_no_put_back',dict(label=w['name'].split('_',1)[1]))
            return True
        return super()._put_back(w,cause)

    def diagnostics(self):
        result=super().diagnostics()
        result.update(candidate='ANS9',relocations=self.relocations,grasp_variants=self.grasp_variants,
            tilted_goals=sorted(self.tilted_goals),y_pinch_digits=sorted(Y_PINCH_DIGITS),tilt_deg=float(np.degrees(TILT)))
        return result
