"""X33: actual execution feedback, quality-first binding, temporal Astra.

X32 and shared controllers remain immutable. No scene IDs/evaluator inputs.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from .xylophone_x32 import XylophonePolicy as X32,cone_from_candidates
from .xylophone_policy import (XylophonePolicy as Legacy,head_pixel,center_strokes,
    restrict_observation,validate_observation,validate_action,ARMS,
    tool_translation_step,high_arrival)
from .xylophone_hexashell import camera_views,XylophonePolicy as HexBase
from .xylophone_feedback import qualified_cones,ExecutionWindow,allow_more_actions
from .xylophone_astra33 import Advisor
from .xylophone_anchor import fit_moving_anchor
from .geometry import pose_matrix,quat_matrix,matrix_quat,bounded_anchor_pose
from .policy import Waypoint
from .xylophone_extended_routes import transfer_clearance,tool_clearance
from .xylophone_lift import HIGH_DWELL,lift_head_height
from .xylophone_routes import JOINT_MARGIN


class XylophonePolicy(X32):
    def reset(self):
        super().reset()
        self.advisor=Advisor();self.execution=ExecutionWindow()
        self.before=None;self.latest_frame=None;self.roi_step=-100
        self.rois={};self.roi_events=[];self.feedback_events=[]
        self.recovery_keyboard_plan=None
        self.diagnostic_holds=set();self.pending_observation=None
        self.program_notes.update(candidate='X33_actual_feedback_quality_first',
            gpt_api_enabled=self.advisor.enabled,api_events=self.advisor.events,
            execution_feedback=self.feedback_events,roi_events=self.roi_events)

    def _advise(self,event,options):
        if event in self._advised:return self.advice.get(event,{})
        self._advised.add(event)
        answer=self.advisor.ask(event,self.current_vision,options,dict(
            step=self.step,phase=self.plan[self.index].name if self.index<len(self.plan) else self.stage,
            remaining_steps=500-self.step,reason=self.reason,
            motion=getattr(self,'motion',{}),trend=self.program_notes.get('execution_trend',{}),
            binding_state=self.binding.get('state','unknown'),
            cone_quality=self.program_notes.get('x33_cone_binding',{}),
            rejection=self.program_notes.get('path_audit',{})),previous=self.before)
        self.advice[event]=answer;self.program_notes['astra_advice']=self.advice
        self.rois=answer.get('rois',{});self.roi_step=self.step
        return answer

    def _adoption(self,event,adopted,reason):
        for row in reversed(self.advisor.events):
            if row.get('event')==event and row.get('state')=='valid_proposal':
                row.update(adopted=bool(adopted),adoption_reason=reason,decision_step=self.step)
                return

    def _record_probe(self,rgb,current,force=False):
        # Keep original point/multiview evidence, but do not run X32/HexaShell
        # hard-scatter veto or collect duplicate unqualified cone histories.
        Legacy._record_probe(self,rgb,current,force)
        if self.nominal_anchor is None or (not force and self.step-self.x32_last<4):return
        self.x32_last=self.step;ee=pose_matrix(current)
        predicted=ee[:3,3]+ee[:3,:3]@self.nominal_anchor
        fresh=self.step-self.roi_step<=4
        for camera,(k,frame,image) in camera_views(self.perception.camera,self.state_now,self.current_vision).items():
            roi=self.rois.get(camera) if fresh else None
            fit,diagnostic=cone_from_candidates(image,k,frame,predicted,roi)
            # A failed semantic ROI must not hide otherwise valid geometry.
            if roi is not None:
                self.roi_events.append(dict(step=self.step,camera=camera,roi_used=True,
                    qualified=fit is not None,proposal_step=self.roi_step))
                if fit is None:fit,diagnostic=cone_from_candidates(image,k,frame,predicted)
            local=np.linalg.inv(ee)@frame
            if fit is not None and not any(r['camera']==camera and np.linalg.norm(r['local']-local)<.008 for r in self.x32_rows):
                if len(self.x32_rows)<36:self.x32_rows.append(dict(camera=camera,ee=ee,local=local,fit=fit))
            self.x32_observations.append(dict(step=self.step,camera=camera,**diagnostic))
        self.program_notes['x33_cone_observations']=self.x32_observations

    def _bind(self):
        cone=qualified_cones(self.x32_rows,self.nominal_anchor) if self.nominal_anchor is not None else {}
        self.program_notes['x33_cone_binding']={k:v for k,v in cone.items() if k!='ee_anchor'}
        visible=[s for s in self.samples if s[1] is not None]
        point=fit_moving_anchor(self.perception.camera,[e for e,p in visible],[p for e,p in visible],stable=True)
        self.program_notes['x33_all_point_binding']={k:v for k,v in point.items() if k!='ee_anchor'}
        contradiction=cone.get('state')=='inconsistent' or point['state']=='inconsistent'
        if contradiction:
            advice=self._advise('geometry_conflict',['keep_geometric_history','external_wrist_view'])
            # Extra observation can clarify a conflict, never delete it or
            # declare a successful grasp. All earlier evidence is retained.
            if advice.get('choice')=='external_wrist_view' and not self.observer_attempted:
                if self._begin_observer():
                    self._adoption('geometry_conflict',True,'audited_external_view');return
            self._adoption('geometry_conflict',False,'contradiction_not_overridden')
            self.binding=self.program_notes['x33_cone_binding'] if cone.get('state')=='inconsistent' else self.program_notes['x33_all_point_binding']
            self._put_back('xy_x33_qualified_binding_conflict');return
        if cone.get('state')=='consistent' and point['state']!='consistent':
            ee=pose_matrix(self.state_now[self.side+'_ee_pose'])
            centre=ee[:3,3]+ee[:3,:3]@cone['ee_anchor']
            if centre[2]<.8206:self._put_back('xy_x33_no_lift_evidence');return
            self.anchor=cone['ee_anchor'];self.binding=self.program_notes['x33_cone_binding']
            self.validation_observation_pose=np.asarray(self.state_now[self.side+'_ee_pose']).copy()
            self.stage='keyboard';self.reason='xy_x33_bound_pending_keyboard';return
        if point['state']=='unknown':
            advice=self._advise('head_confirmation',['keep_geometric_history','external_wrist_view'])
            # Apply fresh ROIs to real pixels now, not just to a trace field.
            if self.rois:self._record_probe(self.current_vision['cam_head']['color'],
                np.asarray(self.state_now[self.side+'_ee_pose']),force=True)
            if advice.get('choice')=='external_wrist_view' and not self.observer_attempted:
                if self._begin_observer():
                    self._adoption('head_confirmation',True,'audited_external_view');return
            self._adoption('head_confirmation',advice.get('choice')=='keep_geometric_history',
                           'retain_all_history_local_confirmation_required')
        # Bypass only the unconditioned cone veto. All existing stationary,
        # all-point, leave-one-out, nominal geometry and unknown guards remain.
        Legacy._bind(self)
        if self.anchor is not None and self.binding.get('state')=='consistent':
            self.validation_observation_pose=np.asarray(self.state_now[self.side+'_ee_pose']).copy()

    def _functional_strokes(self,side,anchor,prefix,reserve):
        arm=self.arms[side];seed=np.asarray(self.state_now[side+'_arm_joint_state'],float)
        current=pose_matrix(self.state_now[side+'_ee_pose']);head=current[:3,3]+current[:3,:3]@anchor
        targets=[('xy_transfer_functional_raise',np.r_[head[:2],max(lift_head_height(),head[2])])]
        for i,xy in enumerate(self.keys):
            for phase,z in (('hover',lift_head_height()),('hit',.803),('lift',lift_head_height())):
                targets.append((f'xy_key_{i}_{phase}',np.r_[xy,z]))
        vertical=[];evidence=[]
        for name,target in targets:
            old=arm.forward(seed)[:3,:3]
            low=np.maximum(arm.lower+JOINT_MARGIN,seed-.9);high=np.minimum(arm.upper-JOINT_MARGIN,seed+.9)
            def residual(q):
                f=arm.forward(q);p=f[:3,3]+f[:3,:3]@anchor
                clearance=tool_clearance(f,anchor,self.calibration['tool_offset_in_ee'],
                    self.calibration['contact_tip_offset_in_ee'],self.calibration['table_z'])
                # Solve the functional target more accurately, not a relaxed
                # endpoint acceptance gate. Spherical-head roll is unobserved.
                return np.r_[300*(p-target),.002*Rotation.from_matrix(old.T@f[:3,:3]).as_rotvec(),
                             .0005*(q-seed),300*max(0.,.008-clearance)]
            fit=least_squares(residual,np.clip(seed,low+1e-8,high-1e-8),bounds=(low,high),
                max_nfev=120,ftol=1e-10,xtol=1e-10,gtol=1e-10)
            frame=arm.forward(fit.x);error=float(np.linalg.norm(frame[:3,3]+frame[:3,:3]@anchor-target))
            clearance=tool_clearance(frame,anchor,self.calibration['tool_offset_in_ee'],
                self.calibration['contact_tip_offset_in_ee'],self.calibration['table_z'])
            evidence.append(dict(phase=name,head_error_m=error,nominal_clearance_m=clearance))
            if error>.0005 or clearance<.005:
                self.program_notes['x33_vertical_functional_path']=dict(accepted=False,endpoints=evidence,reason='endpoint');return None
            vertical.append(Waypoint(side,np.r_[frame[:3,3],matrix_quat(frame[:3,:3])],0.,name,
                2 if name.endswith('_hit') else HIGH_DWELL,anchor));seed=fit.x
        clear,minimum=transfer_clearance(self.state_now[side+'_ee_pose'],list(prefix)+vertical,anchor,
            self.calibration['tool_offset_in_ee'],self.calibration['contact_tip_offset_in_ee'],self.calibration['table_z'])
        accepted=clear and self._audit(side,list(prefix)+vertical,anchor,reserve=reserve)
        self.program_notes['x33_vertical_functional_path']=dict(accepted=accepted,minimum_clearance=minimum,endpoints=evidence)
        return vertical if accepted else None

    def _choose_strokes(self,side,rotation,anchor,prefix=(),reserve=60):
        # Imported template's safe TCP transfer and vertical key sequence first.
        route,choice=HexBase._choose_strokes(self,side,rotation,anchor,prefix,reserve)
        if route is not None:return route,choice
        advice=self._advise('strike_path',['current_orientation','inboard_then_turn'])
        route=self._functional_strokes(side,anchor,prefix,reserve)
        if route is not None:
            self._adoption('strike_path',False,'local_vertical_functional_route_selected')
            return route,dict(degrees=0.,heading_degrees=0.,orientation='vertical_functional_ik')
        current=pose_matrix(self.state_now[side+'_ee_pose']);tcp_offset=np.asarray(self.calibration['tool_offset_in_ee'])
        trials=[]
        for pitch in (0.,-10.,10.):
            r=rotation@Rotation.from_euler('y',pitch,degrees=True).as_matrix()
            # High lift covers the rotated tool's sweep. Pivot about TCP,
            # rather than forcing the uncertain remote mallet head stationary.
            raised=current.copy();raised[2,3]=max(current[2,3],.90+np.linalg.norm(anchor-tcp_offset))
            tcp=raised[:3,3]+raised[:3,:3]@tcp_offset
            near=raised.copy()
            if advice.get('choice')=='inboard_then_turn':near[:2,3]=.7*raised[:2,3]+.3*self.keys[0]
            tcp=near[:3,3]+near[:3,:3]@tcp_offset
            frames=[('xy_transfer_raise',raised),('xy_transfer_near',near)]
            rv=Rotation.from_matrix(r@near[:3,:3].T).as_rotvec();n=max(1,int(np.ceil(np.linalg.norm(rv)/.05)))
            for i in range(1,n+1):
                frame=near.copy();frame[:3,:3]=Rotation.from_rotvec(rv*i/n).as_matrix()@near[:3,:3]
                frame[:3,3]=tcp-frame[:3,:3]@tcp_offset;frames.append((f'xy_transfer_turn_{i}',frame))
            frames+=center_strokes(self.keys,r,anchor)
            points=[Waypoint(side,np.r_[f[:3,3],matrix_quat(f[:3,:3])],0.,name,
                1 if name.startswith('xy_transfer_turn') else 2 if name.endswith('_hit') else HIGH_DWELL,anchor) for name,f in frames]
            for w in points:
                if w.name.startswith('xy_transfer'):w.translation_limit=.006
            full=list(prefix)+points
            clear,minimum=transfer_clearance(self.state_now[side+'_ee_pose'],full,anchor,
                self.calibration['tool_offset_in_ee'],self.calibration['contact_tip_offset_in_ee'],self.calibration['table_z'])
            accepted=clear and self._audit(side,full,anchor,reserve=reserve)
            trials.append(dict(pitch=pitch,accepted=accepted,minimum_clearance=minimum))
            if accepted:
                self.program_notes['x33_tcp_transfer_trials']=trials
                self._adoption('strike_path',bool(advice),'audited_tcp_route_choice')
                return points,dict(degrees=pitch,heading_degrees=0.)
        self.program_notes['x33_tcp_transfer_trials']=trials
        self._adoption('strike_path',False,'no_locally_accepted_route');return None,None

    def _recover_motion(self,w,current,action,evidence):
        event_name='execution_stall'
        advice=self._advise(event_name,['local_replan','hold_and_observe'])
        if advice.get('choice')=='hold_and_observe' and event_name not in self.diagnostic_holds:
            self.diagnostic_holds.add(event_name)
            self.pending_observation=dict(remaining=3,kind='motion',phase=w.name,evidence=evidence)
            self._adoption(event_name,True,'three_fresh_observations_then_guarded_replan')
            self.reason='xy_x33_bounded_diagnostic_observation'
            action[w.side+'_ee_pose']=current.copy();return
        self._adoption(event_name,bool(advice),'candidate_replan_requires_local_audit')
        action[w.side+'_ee_pose']=current.copy()
        if self.motion_retry_count>=1:
            self.finished=True;self.reason='xy_x33_recovery_exhausted_hold';return
        self.motion_retry_count+=1
        anchor=self.anchor if self.anchor is not None else self.nominal_anchor
        event=dict(step=self.step,phase=w.name,evidence=evidence,trials=[],task_success=False)
        self.motion_recoveries.append(event)
        # Replan from the observed state, not an ideal previously reached pose.
        # Probe path is replaceable; grasp/strike ordering is not.
        for dz,dx in ((.02,0.),(.03,-.02),(.03,.02)):
            raised=current.copy();raised[2]+=dz
            lateral=raised.copy();lateral[0]+=dx
            route=[Waypoint(w.side,raised,w.gripper,'xy_probe_recover_raise' if w.name.startswith('xy_probe') else 'xy_motion_raise',3,anchor)]
            if dx:route.append(Waypoint(w.side,lateral,w.gripper,'xy_probe_recover_lateral' if w.name.startswith('xy_probe') else 'xy_motion_lateral',3,anchor))
            probe=w.name.startswith('xy_probe')
            if not probe:route+=list(self.plan[self.index:])
            for p in route:
                if p.name.startswith(('xy_probe_recover','xy_motion')):p.translation_limit=.004
            audit=route+self._nominal_return() if probe or w.name.startswith('xy_approach') else route
            accepted=anchor is not None and self._audit(w.side,audit,anchor,reserve=100,local_only=True)
            event['trials'].append(dict(dz=dz,dx=dx,accepted=accepted,audit=dict(self.program_notes.get('path_audit',{}))))
            if accepted:
                self._set_plan(route,'bind' if probe else self.next_stage)
                self.reason='xy_x33_actual_state_replan';event['action']='audited_replan'
                self.execution=ExecutionWindow();return
        self.finished=True;self.reason='xy_x33_no_safe_recovery_hold';event['action']='hold_audits_rejected'
        self._adoption(event_name,False,'all_local_recovery_audits_rejected')

    def _recover_stalled_hit(self,w,current,anchor,action,evidence):
        self._advise('strike_stall',['reobserve_then_retry','hold_and_observe'])
        if self.advice.get('strike_stall',{}).get('choice')=='hold_and_observe' and 'strike_stall' not in self.diagnostic_holds:
            self.diagnostic_holds.add('strike_stall')
            self.pending_observation=dict(remaining=3,kind='hit',phase=w.name,evidence=evidence)
            self.reason='xy_x33_bounded_diagnostic_observation'
            action[w.side+'_ee_pose']=current.copy()
            self._adoption('strike_stall',True,'three_fresh_observations_then_guarded_revalidation');return
        # Never skip this key or call an error plateau a successful contact.
        self._recover_hit_with_validation(w,current,action,evidence)
        self._adoption('strike_stall',not self.finished,'audited_retreat_new_views' if not self.finished else 'revalidation_audit_rejected')
        self.execution=ExecutionWindow()

    def _finish_anchor_validation(self):
        super()._finish_anchor_validation()
        if not self.finished and self.binding_validated:
            self.recovery_keyboard_plan=(list(self.plan),self.next_stage)
            self.plan=[];self.index=0;self.stage='x33_recovery_keyboard'
            self.keyboard_refreshed=False;self.keyboard_images=[];self.keyboard_refresh_frames=[]
            self.keyboard_reference_ee=None

    def _reserve(self):
        # Conservative floor, not a promise of completing the remaining path.
        return max(60,sum(w.dwell+2 for w in self.plan[self.index+1:]))

    def act(self,raw):
        obs=restrict_observation(raw);validate_observation(obs)
        self.state_now=obs['state'];self.step+=1
        self.current_vision=obs['vision'];rgb=obs['vision']['cam_head']['color']
        if self.latest_frame is None or self.step-self.latest_frame['step']>=12:
            self.before=self.latest_frame
            self.latest_frame=dict(step=self.step,vision={c:{'color':np.asarray(v['color']).copy()} for c,v in obs['vision'].items()})
        if self.home is None:self.home={s:np.asarray(self.state_now[s+'_ee_pose']).copy() for s in ARMS}
        action={s+'_'+k:np.asarray(self.state_now.get(s+'_'+k,[1.])).copy() for s in ARMS for k in ('ee_pose','ee_joint_state')}
        for side,target in self.gripper_targets.items():action[side+'_ee_joint_state']=np.array([target])
        if self.step>500:self.finished=True;self.reason='xy_x33_original_action_budget_hold'
        if not self.finished and self.pending_observation is not None:
            pending=self.pending_observation;w=self.plan[self.index]
            current=np.asarray(self.state_now[w.side+'_ee_pose'])
            if w.name!=pending['phase']:
                self.finished=True;self.reason='xy_x33_observation_phase_changed_hold'
            else:
                pending['remaining']-=1
                if w.name.startswith('xy_probe_'):self._record_probe(rgb,current,force=True)
                if pending['remaining']==0:
                    self.pending_observation=None
                    if pending['kind']=='hit':self._recover_stalled_hit(w,current,self.anchor,action,pending['evidence'])
                    else:self._recover_motion(w,current,action,pending['evidence'])
            validate_action(action);return action
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
                    # Template may have changed: do not silently reuse old hit
                    # targets. Only a negligible rigid shift permits this retry.
                    if self.program_notes['keyboard_refresh'].get('max_shift_m',0.)>.003:
                        self._put_back('xy_x33_keyboard_shift_requires_new_plan')
                    else:self._set_plan(route,stage)
            else:self.finished=True;self.reason='xy_program_complete_not_success'
        if not self.finished and self.index<len(self.plan):
            w=self.plan[self.index];current=np.asarray(self.state_now[w.side+'_ee_pose'])
            anchor=self.anchor if self.anchor is not None else self.nominal_anchor
            if w.side!=self.side and w.anchor_offset is not None:anchor=np.asarray(w.anchor_offset)
            if anchor is None:anchor=np.asarray(self.calibration['tool_offset_in_ee'])
            move=.003 if w.name=='xy_descend' else getattr(w,'translation_limit',.012)
            if w.name.endswith('_hit'):move=tool_translation_step(current,w.pose,anchor,move,.038)
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
            movable=w.name.startswith(('xy_approach','xy_probe_','xy_transfer','xy_hexashell','xy_key_'))
            if not reached and movable and self.age>=18 and trend['stalled']:
                self._recover_motion(w,current,action,trend);validate_action(action);return action
            self.dwell=self.dwell+1 if reached else 0
            if self.dwell>=w.dwell:
                if w.name=='xy_close' or w.name.startswith('xy_probe_'):self._record_probe(rgb,current,force=True)
                if w.name.startswith('xy_validate_view_'):self._record_validation_view(rgb,current,force=True)
                self.feedback_events.append(dict(step=self.step,phase=w.name,action='reached',age=self.age))
                self.last_reached_pose=current.copy();self.index+=1;self.age=self.dwell=0
                self.execution=ExecutionWindow()
                if self.index==len(self.plan):self.stage=self.next_stage
            elif not allow_more_actions(self.age,self.step,trend,self._reserve(),w.name):
                self.finished=True;self.reason='xy_x33_progress_or_budget_deadline_hold'
                action[w.side+'_ee_pose']=current.copy()
                self.feedback_events.append(dict(step=self.step,phase=w.name,action='bounded_deadline',age=self.age,trend=trend))
        validate_action(action);return action
