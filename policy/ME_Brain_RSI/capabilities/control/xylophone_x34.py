"""X34: deterministic best-effort controller; the Astra advisor is never used.

Holding still after a rejected local audit scores zero. X34 therefore keeps
X33's geometry but replaces most hold states with bounded geometric fallbacks:
a nominal grasp anchor when the RGB binding is not consistent, an endpoint and
clearance checked vertical stroke route when sampled-IK audits reject every
candidate, best-effort advancement on motion stalls, and genuine arrival is
never turned into a deadline. Hit phases keep their original acceptance.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from .xylophone_x33 import XylophonePolicy as X33
from .xylophone_policy import (XylophonePolicy as Legacy,restrict_observation,validate_observation,
    validate_action,ARMS,tool_translation_step,high_arrival)
from .xylophone_feedback import ExecutionWindow,allow_more_actions
from .xylophone_astra33 import Advisor
from .xylophone_anchor import fit_moving_anchor
from .geometry import pose_matrix,quat_matrix,matrix_quat,bounded_anchor_pose
from .policy import Waypoint
from .xylophone_extended_routes import transfer_clearance,tool_clearance
from .xylophone_lift import lift_head_height
from .xylophone_routes import JOINT_MARGIN

HIT_SLOW_DISTANCE=.025
HIT_SLOW_STEP=.004
DESCEND_STEP=.005
FAST_STEP=.020
FAST_REMAINING=.05
KEY_HIGH_DWELL=2
SOFT_ARRIVAL_AGE=30
SOFT_ARRIVAL_POSITION=.008
SOFT_ARRIVAL_ANGLE=.06
MAX_UNREACHABLE_ADVANCES=3
FALLBACK_ENDPOINT_ERROR=.002
FALLBACK_CLEARANCE=.005
ARM_BASES={'left':np.array([-.3,-.45]),'right':np.array([.3,-.45])}
FAST_PHASES=('xy_approach','xy_transfer','xy_hexashell','xy_key_0_hover')


def remaining_path(current,goal,anchor):
    c,g=pose_matrix(current),pose_matrix(goal);anchor=np.asarray(anchor,float)
    a0=c[:3,3]+c[:3,:3]@anchor;a1=g[:3,3]+g[:3,:3]@anchor
    angle=Rotation.from_matrix(c[:3,:3].T@g[:3,:3]).magnitude()
    return float(np.linalg.norm(a1-a0)+np.linalg.norm(anchor)*angle)


class XylophonePolicy(X33):
    MOVABLE_PREFIXES=('xy_approach','xy_probe_','xy_transfer','xy_hexashell','xy_key_')
    FALLBACK_HIT_Z=.803

    def reset(self):
        super().reset()
        self.advisor=Advisor(enabled=False)
        self.unreachable_advances=0
        self.program_notes.update(candidate='X34_deterministic_best_effort',
            gpt_api_enabled=False,api_events=self.advisor.events)

    def _choose_side(self,grasp_xy,keys):
        # The farthest key, not the sign of the handle x, decides the arm.
        cost={}
        for side,base in ARM_BASES.items():
            cost[side]=max(float(np.max(np.linalg.norm(np.asarray(keys)[:,:2]-base,axis=1))),
                           float(np.linalg.norm(np.asarray(grasp_xy)[:2]-base)))
        self.program_notes['x34_side_cost']=cost
        return min(cost,key=cost.get)

    def _set_plan(self,waypoints,next_stage):
        for w in waypoints:
            if w.name.startswith('xy_key_') and w.name.endswith(('_hover','_lift')) and w.dwell>KEY_HIGH_DWELL:
                w.dwell=KEY_HIGH_DWELL
        super()._set_plan(waypoints,next_stage)

    def _translation_bound(self,w,current,anchor):
        if w.name=='xy_descend':return DESCEND_STEP
        move=getattr(w,'translation_limit',.012)
        if w.name.startswith(FAST_PHASES) and not hasattr(w,'translation_limit'):
            if remaining_path(current,w.pose,anchor)>FAST_REMAINING:move=FAST_STEP
        if w.name.endswith('_hit'):move=tool_translation_step(current,w.pose,anchor,move,HIT_SLOW_DISTANCE,HIT_SLOW_STEP)
        return move

    def _bind(self):
        visible=[s for s in self.samples if s[1] is not None]
        point=fit_moving_anchor(self.perception.camera,[e for e,p in visible],[p for e,p in visible],stable=True)
        measured={k:v for k,v in point.items() if k!='ee_anchor'}
        self.program_notes['x34_all_point_binding']=measured
        inside=point['state']=='consistent' and np.linalg.norm(point['ee_anchor']-self.nominal_anchor)<=.035
        if inside:
            self.anchor=np.asarray(point['ee_anchor'],float).copy()
            self.binding={**measured,'source':'measured_rgb_point_fit'}
            self.reason='xy_x34_bound_measured_pending_keyboard'
        else:
            # Public mallet geometry plus the executed grasp gives a usable
            # prior; refusing to strike scores nothing.
            self.anchor=np.asarray(self.nominal_anchor,float).copy()
            self.binding={'state':'nominal_fallback','measured_state':point['state'],
                          'measured_reason':point.get('reason'),'source':'grasp_geometry_prior'}
            self.reason='xy_x34_bound_nominal_pending_keyboard'
        self.validation_observation_pose=np.asarray(self.state_now[self.side+'_ee_pose']).copy()
        self.stage='keyboard'

    def _endpoint_checked_strokes(self,side,anchor,prefix,key_indices=None):
        arm=self.arms[side];seed=np.asarray(self.state_now[side+'_arm_joint_state'],float)
        current=pose_matrix(self.state_now[side+'_ee_pose']);head=current[:3,3]+current[:3,:3]@anchor
        targets=[('xy_transfer_functional_raise',np.r_[head[:2],max(lift_head_height(),head[2])])]
        indices=range(len(self.keys)) if key_indices is None else list(key_indices)
        for i in indices:
            xy=self.keys[i]
            for phase,z in (('hover',lift_head_height()),('hit',self.FALLBACK_HIT_Z),('lift',lift_head_height())):
                targets.append((f'xy_key_{i}_{phase}',np.r_[xy,z]))
        route=[];evidence=[]
        for name,target in targets:
            old=arm.forward(seed)[:3,:3]
            low=np.maximum(arm.lower+JOINT_MARGIN,seed-.9);high=np.minimum(arm.upper-JOINT_MARGIN,seed+.9)
            def residual(q):
                f=arm.forward(q);p=f[:3,3]+f[:3,:3]@anchor
                clearance=tool_clearance(f,anchor,self.calibration['tool_offset_in_ee'],
                    self.calibration['contact_tip_offset_in_ee'],self.calibration['table_z'])
                return np.r_[300*(p-target),.002*Rotation.from_matrix(old.T@f[:3,:3]).as_rotvec(),
                             .0005*(q-seed),300*max(0.,.008-clearance)]
            fit=least_squares(residual,np.clip(seed,low+1e-8,high-1e-8),bounds=(low,high),
                max_nfev=120,ftol=1e-10,xtol=1e-10,gtol=1e-10)
            frame=arm.forward(fit.x);error=float(np.linalg.norm(frame[:3,3]+frame[:3,:3]@anchor-target))
            clearance=tool_clearance(frame,anchor,self.calibration['tool_offset_in_ee'],
                self.calibration['contact_tip_offset_in_ee'],self.calibration['table_z'])
            evidence.append(dict(phase=name,head_error_m=error,nominal_clearance_m=clearance))
            if error>FALLBACK_ENDPOINT_ERROR or clearance<FALLBACK_CLEARANCE:
                self.program_notes['x34_fallback_route']=dict(accepted=False,reason='endpoint',endpoints=evidence);return None
            route.append(Waypoint(side,np.r_[frame[:3,3],matrix_quat(frame[:3,:3])],0.,name,
                2 if name.endswith('_hit') else KEY_HIGH_DWELL,anchor));seed=fit.x
        clear,minimum=transfer_clearance(self.state_now[side+'_ee_pose'],list(prefix)+route,anchor,
            self.calibration['tool_offset_in_ee'],self.calibration['contact_tip_offset_in_ee'],self.calibration['table_z'])
        self.program_notes['x34_fallback_route']=dict(accepted=bool(clear),reason='endpoint_and_clearance_only' if clear else 'transfer_clearance',
            minimum_clearance=minimum,endpoints=evidence,sampled_ik_audit=False)
        return route if clear else None

    def _choose_strokes(self,side,rotation,anchor,prefix=(),reserve=60):
        route,choice=super()._choose_strokes(side,rotation,anchor,prefix,reserve)
        if route is not None:return route,choice
        route=self._endpoint_checked_strokes(side,anchor,prefix)
        if route is None:return None,None
        return route,dict(degrees=0.,heading_degrees=0.,orientation='x34_vertical_endpoint_fallback')

    def _put_back(self,reason):
        if reason in ('xy_keyboard_changed_or_unstable','xy_x33_keyboard_shift_requires_new_plan'):
            # Keep striking the initially measured template instead of quitting.
            self.program_notes['x34_keyboard_refresh_override']=reason
            self.reason='xy_x34_keyboard_refresh_rejected_keep_initial'
            if reason=='xy_x33_keyboard_shift_requires_new_plan' and self.recovery_keyboard_plan is not None:
                route,stage=self.recovery_keyboard_plan;self._set_plan(route,stage)
            return
        super()._put_back(reason)

    def _advance(self,w,current,how):
        self.feedback_events.append(dict(step=self.step,phase=w.name,action=how,age=self.age))
        if how=='unreachable_advance':
            self.unreachable_advances+=1
            if self.unreachable_advances>MAX_UNREACHABLE_ADVANCES:
                self.finished=True;self.reason='xy_x34_repeated_unreachable_hold';return
        elif how in ('reached','soft_arrival'):self.unreachable_advances=0
        self.last_reached_pose=np.asarray(current).copy();self.index+=1;self.age=self.dwell=0
        self.execution=ExecutionWindow();self.hit_stall.clear_window()
        if self.index==len(self.plan):self.stage=self.next_stage

    def _recover_motion(self,w,current,action,evidence):
        action[w.side+'_ee_pose']=np.asarray(current).copy()
        event=dict(step=self.step,phase=w.name,evidence=evidence,trials=[],task_success=False)
        self.motion_recoveries.append(event)
        if w.name.startswith('xy_probe_'):
            event['action']='skip_remaining_probes'
            self.plan=[];self.index=0;self.age=self.dwell=0;self.execution=ExecutionWindow()
            self.stage='bind';self.reason='xy_x34_probe_stall_bind_with_existing_views';return
        anchor=self.anchor if self.anchor is not None else self.nominal_anchor
        if self.motion_retry_count<1 and anchor is not None:
            self.motion_retry_count+=1
            for dz,dx in ((.02,0.),(.03,-.02),(.03,.02)):
                raised=np.asarray(current).copy();raised[2]+=dz
                lateral=raised.copy();lateral[0]+=dx
                extra=[Waypoint(w.side,raised,w.gripper,'xy_motion_raise',3,anchor)]
                if dx:extra.append(Waypoint(w.side,lateral,w.gripper,'xy_motion_lateral',3,anchor))
                for p in extra:p.translation_limit=.004
                route=extra+list(self.plan[self.index:])
                accepted=self._audit(w.side,route,anchor,reserve=60,local_only=True)
                event['trials'].append(dict(dz=dz,dx=dx,accepted=accepted,audit=dict(self.program_notes.get('path_audit',{}))))
                if accepted:
                    self._set_plan(route,self.next_stage);self.reason='xy_x34_actual_state_replan'
                    event['action']='audited_replan';self.execution=ExecutionWindow();return
        event['action']='unreachable_advance'
        self._advance(w,current,'unreachable_advance')
        if self.finished:event['action']='hold_repeated_unreachable'

    def _recover_stalled_hit(self,w,current,anchor,action,evidence):
        self.binding_validated=False
        Legacy._recover_stalled_hit(self,w,current,anchor,action,evidence)
        if self.finished and self.reason in ('xy_hit_recovery_limit_hold','xy_hit_retry_path_rejected_hold','xy_hit_escape_path_rejected_hold'):
            # Retries are exhausted or unauditable: leave this key through its
            # own lift target rather than freezing for the rest of the episode.
            self.finished=False;self.recovery_events[-1]['x34_action']='advance_to_lift_without_claiming_hit'
            self._advance(w,current,'hit_abandoned')
            action[w.side+'_ee_pose']=np.asarray(current).copy()
        self.execution=ExecutionWindow()

    def act(self,raw):
        obs=restrict_observation(raw);validate_observation(obs)
        self.state_now=obs['state'];self.step+=1
        self.current_vision=obs['vision'];rgb=obs['vision']['cam_head']['color']
        if self.home is None:self.home={s:np.asarray(self.state_now[s+'_ee_pose']).copy() for s in ARMS}
        action={s+'_'+k:np.asarray(self.state_now.get(s+'_'+k,[1.])).copy() for s in ARMS for k in ('ee_pose','ee_joint_state')}
        for side,target in self.gripper_targets.items():action[side+'_ee_joint_state']=np.array([target])
        if self.step>500:self.finished=True;self.reason='xy_x34_original_action_budget_hold'
        if not self.finished and self.index>=len(self.plan):
            self.plan=[];self.index=0
            if self.stage=='observe':self._observe(rgb)
            elif self.stage=='bind':self._bind()
            elif self.stage=='observer_probe':self._observer_probe()
            elif self.stage=='observer_return':self._observer_return()
            elif self.stage=='keyboard':
                if self.keyboard_refreshed or self._refresh_keyboard(rgb):self._start_strokes()
            elif self.stage=='verify_binding':self._finish_anchor_validation()
            elif self.stage=='x33_recovery_keyboard':
                if self._refresh_keyboard(rgb) and not self.finished:
                    route,stage=self.recovery_keyboard_plan
                    if self.program_notes['keyboard_refresh'].get('max_shift_m',0.)>.003:
                        self._put_back('xy_x33_keyboard_shift_requires_new_plan')
                    else:self._set_plan(route,stage)
            else:self.finished=True;self.reason='xy_program_complete_not_success'
        if not self.finished and self.index<len(self.plan):
            w=self.plan[self.index];current=np.asarray(self.state_now[w.side+'_ee_pose'])
            anchor=self.anchor if self.anchor is not None else self.nominal_anchor
            if w.side!=self.side and w.anchor_offset is not None:anchor=np.asarray(w.anchor_offset)
            if anchor is None:anchor=np.asarray(self.calibration['tool_offset_in_ee'])
            move=self._translation_bound(w,current,anchor)
            action[w.side+'_ee_pose']=bounded_anchor_pose(current,w.pose,anchor,move,.10)
            action[w.side+'_ee_joint_state']=np.array([w.gripper]);self.gripper_targets[w.side]=float(w.gripper)
            error=float(np.linalg.norm(current[:3]-w.pose[:3]))
            angle=float(Rotation.from_matrix(quat_matrix(current[3:]).T@quat_matrix(w.pose[3:])).magnitude())
            reached=error<.003 and angle<.035;self.age+=1
            if w.name.startswith('xy_validate_view_'):self._record_validation_view(rgb,current,force=reached)
            if w.name.startswith('xy_probe_'):self._record_probe(rgb,current,force=reached)
            if w.name.startswith('xy_key_') and w.name.endswith(('_hover','_lift')):
                arrival=high_arrival(current,w.pose,anchor)
                self.program_notes['high_arrival']=dict(step=self.step,phase=w.name,**arrival)
                reached=reached and arrival['accepted']
            self.motion=dict(side=w.side,current_ee=current.tolist(),goal_ee=w.pose.tolist(),position_error=error,orientation_error=angle)
            trend=self.execution.observe(w.name,error,angle)
            self.program_notes['execution_trend']=dict(step=self.step,phase=w.name,age=self.age,**trend)
            stall=self.hit_stall.observe(w.name,current,w.pose,anchor)
            if not reached and w.name.endswith('_hit') and (stall['qualified'] or trend['stalled']):
                self._recover_stalled_hit(w,current,anchor,action,dict(contact_guard=stall,error_trend=trend))
                validate_action(action);return action
            movable=w.name.startswith(self.MOVABLE_PREFIXES)
            if not reached and movable and self.age>=18 and trend['stalled']:
                if self.age>=SOFT_ARRIVAL_AGE and error<SOFT_ARRIVAL_POSITION and angle<SOFT_ARRIVAL_ANGLE:
                    # Close, stationary and not a hit: the next target is reachable
                    # from here; waiting for 3 mm would only burn the budget.
                    self._advance(w,current,'soft_arrival')
                else:self._recover_motion(w,current,action,trend)
                validate_action(action);return action
            self.dwell=self.dwell+1 if reached else 0
            if self.dwell>=w.dwell:
                if w.name=='xy_close' or w.name.startswith('xy_probe_'):self._record_probe(rgb,current,force=True)
                if w.name.startswith('xy_validate_view_'):self._record_validation_view(rgb,current,force=True)
                self._advance(w,current,'reached')
            elif reached:pass  # settling toward the dwell count is arrival, never a deadline
            elif not allow_more_actions(self.age,self.step,trend,self._reserve(),w.name):
                self.feedback_events.append(dict(step=self.step,phase=w.name,action='bounded_deadline',age=self.age,trend=trend))
                if w.name in ('xy_descend','xy_close') or movable:
                    self._advance(w,current,'unreachable_advance')
                else:self.finished=True;self.reason='xy_x34_progress_or_budget_deadline_hold'
                action[w.side+'_ee_pose']=current.copy()
        validate_action(action);return action
