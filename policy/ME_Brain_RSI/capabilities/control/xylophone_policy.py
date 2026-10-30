"""RGB-only xylophone controller: observe, grasp, bind head, ordered strokes.

Independent functional-point anchoring; no hidden poses/depth/reward. Public
sphere symmetry leaves roll unobserved. Both public beat points lie within
10.05 mm of the head, so center targets use the middle of the vertical band.
Nominal sampled IK is not a world collision guarantee.
"""
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from .policy import Policy, Waypoint
from .contracts import ARMS, restrict_observation, validate_observation, validate_action
from .geometry import pose_matrix, matrix_quat, quat_matrix, bounded_anchor_pose
from .kinematics import SerialArm
from .xylophone_measurement import round_heads
from .xylophone_rgb import fit_painted_keys
from .xylophone_template import fit_fixed_keyboard
from .xylophone_mallet_rgb import detect_planar_mallet
from .xylophone_joint_rgb import detect_joint_mallet
from .xylophone_anchor import fit_moving_anchor, ee_translation_for_anchor
from .xylophone_multiview import measure_wrist, confirm_multiview
from .xylophone_observer import (wrist_camera,observer_pixel,observer_candidates,
                                 confirm_observer,MOUNT)
from .xylophone_information import information_probe_candidates
from .xylophone_precision import precision_candidates
from .xylophone_extended_routes import extended_candidates, transfer_clearance
from .xylophone_validation import validation_frames,validate_anchor,distinct_view
from .xylophone_tilt import bounded_tilt_candidates
from .xylophone_routes import (heading_candidates,reverse_tilt_candidates,route_rank,PROBE_PATTERNS,
                               MAX_JOINT_STEP,JOINT_MARGIN,MotionProgress)
from .xylophone_path import audit_tool_path, tool_translation_step
from .xylophone_lift import lift_head_height, high_arrival, HIGH_DWELL
from .xylophone_recovery import (HitStall, RETRY_LIFT, MAX_RETRIES_PER_KEY,
                                 MAX_RETRIES_TOTAL)


def head_pixel(rgb, expected, max_distance=18.):
    """Associate a measured color core near the robot-predicted projection.

    The low-saturation reflection can connect the head to surrounding red
    regions after lifting. If the original measurement is missing, try fixed
    saturation windows with the SAME circle/ellipse and threshold agreement
    tests. Every local window must agree; never resolve ambiguity by choosing
    whichever point best fits the later attachment model. This observes only
    a pixel; fit_moving_anchor still decides whether rigid binding is supported.
    """
    choices=[h['center'] for h in round_heads(rgb)
             if np.linalg.norm(h['center']-expected)<=max_distance]
    if choices:
        return choices[0] if len(choices)==1 else None
    cores=[]
    for saturation in (140,160,180,200):
        cores.extend(h['center'] for h in round_heads(rgb,thresholds=(
            (saturation-10,35),(saturation,40),(saturation+10,45)))
            if np.linalg.norm(h['center']-expected)<=max_distance)
    if not cores:return None
    points=np.asarray(cores)
    if np.max(np.linalg.norm(points[:,None]-points[None,:],axis=2))>1.:
        return None
    return points.mean(0)


def rectified_xy(perception, pixels, image_shape):
    x0,x1,y0,y1=perception.bounds
    return np.asarray(pixels)/np.array([image_shape[1]-1,image_shape[0]-1])*[x1-x0,y1-y0]+[x0,y0]


HANDLE_GRASP_DISTANCE_M = .05


def handle_grasp_point(head_xy, axis):
    """Grasp the proximal exposed shaft, using only RGB head and shaft axis.

    X17 videos show the 90 mm target closing around the central support and
    lifting without the mallet. The fixed 50 mm target is outside that region
    and outside the public 13 mm head radius. This is a generic shaft fraction,
    not a layout lookup or a claim of collision-free grasping. Keep the .791 m
    public-static height prior unchanged to isolate the longitudinal offset;
    measured binding and the complete local path audits remain mandatory.
    """
    return np.r_[np.asarray(head_xy)+HANDLE_GRASP_DISTANCE_M*np.asarray(axis), .791]



def center_strokes(keys, rotation, anchor):
    """Keep X06 hits; lift above the earliest possible acceptance reference."""
    frames=[]
    for key,xy in enumerate(keys):
        for phase,z in (('hover',lift_head_height()),('hit',.803),('lift',lift_head_height())):
            target=np.r_[xy,z]
            ee=np.eye(4);ee[:3,:3]=rotation
            ee[:3,3]=ee_translation_for_anchor(target,rotation,anchor)
            frames.append((f'xy_key_{key}_{phase}',ee))
    return frames


class XylophonePolicy(Policy):
    HEAD_SEARCH_PX=18.

    def reset(self):
        super().reset()
        self.stage='observe'
        self.arms={s:SerialArm.public_x5(s) for s in ARMS}
        self.side=None; self.anchor=None; self.nominal_anchor=None
        self.samples=[]; self.probe_evidence=[]; self.view_attempted=False; self.state_now={}
        self.keys=None; self.grasp_pose=None; self.binding={}
        self.measurement_frames=[];self.view_count=0;self.extra_probe_used=False
        self.motion_progress=MotionProgress();self.motion_recoveries=[]
        self.motion_retry_count=0;self.last_reached_pose=None
        self.hit_stall=HitStall();self.hit_retries={};self.recovery_events=[]
        self.validation_context=None;self.binding_validated=False
        self.validation_observation_pose=None
        self.gripper_targets={}
        self.keyboard_refresh_frames=[];self.keyboard_refreshed=False
        self.current_vision={};self.multiview_samples=[];self.multiview_evidence=[]
        self.observer_attempted=False;self.observer_samples=[];self.observer_evidence=[]
        self.observer_home=None;self.observer_started=None
        self.keyboard_images=[];self.keyboard_reference_ee=None
        self.program_notes={'controller':'functional_head_v4','roll_observed':False,
                            'held_head_measurement':'X20_local_saturation_consensus',
                            'lift_head_height_m':lift_head_height(),
                            'lift_reference':'public_acceptance_upper_bound_plus_25mm_and_8mm_margin'}
        self.program_notes['candidate']='X30_independent_wrist_deferred_keyboard_template'

    def _set_plan(self, waypoints, next_stage):
        self.plan=waypoints; self.index=self.age=self.dwell=0
        self.motion_progress=MotionProgress()
        if waypoints and self.state_now:
            self.last_reached_pose=np.asarray(self.state_now[waypoints[0].side+'_ee_pose']).copy()
        self.next_stage=next_stage

    def _audit(self, side, waypoints, anchor, reserve=60, local_only=False,
               position_tolerance=.002,orientation_tolerance=.035):
        frames=[pose_matrix(w.pose) for w in waypoints]
        result=audit_tool_path(self.arms[side],pose_matrix(self.state_now[f'{side}_ee_pose']),
            frames,anchor,self.state_now[f'{side}_arm_joint_state'],
            remaining_steps=max(1,500-self.step),reserve_steps=reserve,
            dwell=3,translation=.012,rotation=.10,
            position_tolerance=position_tolerance,orientation_tolerance=orientation_tolerance,
            target_translations=[.003 if w.name=='xy_descend' else getattr(w,'translation_limit',.012) for w in waypoints],
            target_slow_distances=[.038 if w.name.endswith('_hit') else 0. for w in waypoints],
            target_dwells=[w.dwell for w in waypoints],
            ik_restarts_on_failure=0,max_joint_step=MAX_JOINT_STEP,
            joint_limit_margin=JOINT_MARGIN,local_seed_retries=True)
        self.program_notes['path_audit']={k:v for k,v in result.items() if k!='samples'}
        return result['accepted']

    def _observe(self, rgb):
        self.attempts+=1
        key_image=self.perception.rectified(rgb,.78685)
        key_image[self.perception.valid==0]=0
        keys=fit_fixed_keyboard(key_image,legacy_fit=fit_painted_keys)
        self.program_notes['keyboard_pose_method']=None if keys is None else keys['pose_method']
        if keys is not None and 'dense_score' in keys:
            self.program_notes['keyboard_template_evidence']={k:keys[k] for k in
                ('dense_score','paint_support','angle_deg','ambiguity_margin')}
        original_bounds=self.perception.bounds
        # Retain visible handle pixels outside the former narrow table crop.
        self.perception.bounds=[original_bounds[0],original_bounds[1],-.50,.25]
        mallet_image=self.perception.rectified(rgb,.7956)
        mallet_image[self.perception.valid==0]=0
        mallet=detect_planar_mallet(mallet_image)
        if mallet is not None:
            mallet['head_xy']=rectified_xy(self.perception,mallet['head_px'],mallet_image.shape)
            self.program_notes['mallet_detector']='legacy_preserved'
        else:
            # Original RGB joint head/shaft evidence, only on legacy unknown.
            # The resting plane, grasp point and all downstream actions stay
            # unchanged. This does not provide measured depth or attachment.
            joint,diagnostic=detect_joint_mallet(rgb,self.perception.camera,diagnostics=True)
            self.program_notes['mallet_detector']='original_rgb_joint_fallback'
            self.program_notes['mallet_detection_reason']=diagnostic['reason']
            self.program_notes['mallet_joint_candidates']=len(diagnostic['candidates'])
            if joint is not None:
                mallet=dict(joint,axis_px=joint['axis_xy'])
                self.program_notes['mallet_joint_evidence']=joint['evidence']
        self.perception.bounds=original_bounds
        self.program_notes['keys_visible']=keys is not None
        self.program_notes['mallet_visible']=mallet is not None
        frame=None
        if keys is not None and mallet is not None:
            frame={'keys':rectified_xy(self.perception,keys['key_pixels'],key_image.shape),
                   'head':mallet['head_xy'],'axis':mallet['axis_px']}
        self.measurement_frames.append(frame)
        if len(self.measurement_frames)<3:
            self.reason='xy_collect_stationary_rgb';return
        valid=[f for f in self.measurement_frames if f is not None]
        self.measurement_frames=[]
        stable=len(valid)>=2
        if stable:
            key_array=np.array([f['keys'] for f in valid])
            head_array=np.array([f['head'] for f in valid])
            axes=np.array([f['axis'] for f in valid])
            key_spread=float(np.max(np.linalg.norm(key_array-key_array.mean(0),axis=2)))
            head_spread=float(np.max(np.linalg.norm(head_array-head_array.mean(0),axis=1)))
            axis_agreement=float(np.min(axes@axes.T))
            stable=key_spread<=.003 and head_spread<=.002 and axis_agreement>=np.cos(np.deg2rad(2))
            self.program_notes['measurement_stability']={
                'valid_frames':len(valid),'key_spread_m':key_spread,
                'head_spread_m':head_spread,'axis_agreement':axis_agreement,
                'accepted':bool(stable),'observed_depth':False}
        if not stable:
            # Unknown is not absent. Clear each potential occluding arm once.
            if not self.view_attempted and self.view_count<2:
                side=('left','right')[self.view_count];self.view_count+=1
                pose=np.asarray(self.state_now[f'{side}_ee_pose']).copy()
                pose[0]+=-.10 if side=='left' else .10;pose[2]+=.06
                view=[Waypoint(side,pose,1.,'xy_clear_view',3)]
                if self._audit(side,view,np.asarray(self.calibration['tool_offset_in_ee'])):
                    self._set_plan(view,'observe');return
            self.view_attempted=True
            self.finished=True;self.reason='xy_rgb_unknown_after_bounded_views';return
        self.keys=key_array.mean(0)
        head_xy=head_array.mean(0)
        axis=axes.mean(0);axis/=np.linalg.norm(axis)
        grasp_point=handle_grasp_point(head_xy,axis)
        grasp_xy=grasp_point[:2]
        self.program_notes['handle_grasp_distance_m']=HANDLE_GRASP_DISTANCE_M
        self.last_detections={'head_xy':head_xy.tolist(),'shaft_axis':axis.tolist(),
                              'keys_xy':self.keys.tolist()}
        side=self._choose_side(grasp_xy,self.keys)
        self.side=side
        candidates=[];trials=[]
        for yaw_index,yaw in enumerate((np.arctan2(axis[1],axis[0]),np.arctan2(axis[1],axis[0])+np.pi)):
            grab=self.pose(side,grasp_xy,float(grasp_point[2]-self.calibration['table_z']),yaw)
            frame=pose_matrix(grab)
            anchor=frame[:3,:3].T@(np.r_[head_xy,.7956]-frame[:3,3])
            approach=grab.copy();approach[2]+=.09
            lifted=grab.copy();lifted[2]+=.065
            for probe_index,offsets in enumerate(PROBE_PATTERNS):
                probes=[]
                for j,delta in enumerate(offsets):
                    target=lifted.copy();target[:3]+=delta
                    probes.append(Waypoint(side,target,0.,f'xy_probe_{j+1}',3))
                grasp=[Waypoint(side,approach,1.,'xy_approach',2),
                       Waypoint(side,grab,1.,'xy_descend',3),
                       Waypoint(side,grab,0.,'xy_close',10),
                       Waypoint(side,lifted,0.,'xy_probe_0',3),*probes]
                # Before calibration the head offset is only a support prior.
                # Require a connected grasp/probe AND return corridor, but do
                # not pretend nominal offset predicts the measured eight-key
                # route. Re-audit all 24 strokes after binding, before any hit.
                back=[Waypoint(side,lifted,0.,'xy_return_above',2),
                      Waypoint(side,grab,0.,'xy_return_lower',3),
                      Waypoint(side,grab,1.,'xy_release',8),
                      Waypoint(side,approach,1.,'xy_retreat',2)]
                if not self._audit(side,grasp+back,anchor,reserve=70,local_only=True):
                    trials.append({'yaw_variant':yaw_index,'probe_variant':probe_index,
                                   'prefix_and_return':dict(self.program_notes['path_audit'])})
                    continue
                audit=dict(self.program_notes['path_audit'])
                trials.append({'yaw_variant':yaw_index,'probe_variant':probe_index,
                               'grasp_return_safe_nominal':True,'audit':audit,
                               'eight_key_path':'pending_measured_binding'})
                candidates.append((self._rank_grasp(audit,yaw_index*2+probe_index),grab,anchor,
                                   grasp,audit,yaw_index,probe_index))
        self.program_notes['grasp_route_trials']=trials
        if candidates:
            _,grab,anchor,grasp,audit,yaw_index,probe_index=min(candidates,key=lambda x:x[0])
            self.grasp_pose=grab;self.nominal_anchor=anchor
            self.program_notes.update(path_audit=audit,grasp_yaw_variant=yaw_index,
                                      probe_variant=probe_index,
                                      nominal_stroke_orientation={'status':'pending_measured_binding'})
            self.samples=[];self._set_plan(grasp,'bind')
            self.reason='xy_grasp_and_probe';return
        self.finished=True;self.reason='xy_no_continuous_grasp_return_path'

    def _choose_side(self,grasp_xy,keys):
        return 'left' if grasp_xy[0]<0 else 'right'

    def _rank_grasp(self,audit,alternative_index):
        return route_rank(audit,alternative_index)

    def _audit_precise(self,side,points,anchor,reserve):
        # Planning preference only. Execution/arrival/score thresholds unchanged.
        return self._audit(side,points,anchor,reserve=reserve,local_only=True,
                           position_tolerance=.0005,orientation_tolerance=.01)

    def _prefer_precise_strokes(self,side,rotation,anchor,strokes,choice,prefix,reserve):
        original_audit=dict(self.program_notes.get('path_audit',{}))
        trials=[]
        if self._audit_precise(side,list(prefix)+strokes,anchor,reserve):
            self.program_notes['precision_route']={'selected':'original_precise',
                'path_audit':dict(self.program_notes.get('path_audit',{})),
                'position_tolerance_m':.0005,'orientation_tolerance_rad':.01}
            return strokes,choice
        trials.append({'source':'original','choice':dict(choice),
                       'path_audit':dict(self.program_notes.get('path_audit',{}))})
        for candidate in precision_candidates(rotation,anchor,
                self.calibration['tool_offset_in_ee'],self.calibration['contact_tip_offset_in_ee'],
                self.calibration['table_z']):
            alternative=[Waypoint(side,np.r_[t[:3,3],matrix_quat(t[:3,:3])],0.,name,
                          2 if name.endswith('_hit') else HIGH_DWELL,anchor)
                         for name,t in center_strokes(self.keys,candidate['rotation'],anchor)]
            accepted=self._audit_precise(side,list(prefix)+alternative,anchor,reserve)
            audit=dict(self.program_notes.get('path_audit',{}))
            failed=audit.get('target_index',-1)-len(prefix)
            if not accepted and audit.get('reason')=='joint_discontinuity' and 0<=failed<len(alternative):
                alternative[failed].translation_limit=.006
                accepted=self._audit_precise(side,list(prefix)+alternative,anchor,reserve)
                audit=dict(self.program_notes.get('path_audit',{}))
            entry={k:v for k,v in candidate.items() if k!='rotation'}
            trials.append(dict(source='bounded_ten_degree',choice=entry,path_audit=audit))
            if accepted:
                self.program_notes['precision_route']={'selected':'bounded_alternative',
                    'trials':trials,'position_tolerance_m':.0005,'orientation_tolerance_rad':.01}
                return alternative,entry
        # A preference must not turn an otherwise accepted route into refusal.
        self.program_notes['path_audit']=original_audit
        self.program_notes['precision_route']={'selected':'original_normal_fallback',
            'trials':trials,'position_tolerance_m':.0005,'orientation_tolerance_rad':.01}
        return strokes,choice

    def _choose_strokes(self,side,rotation,anchor,prefix=(),reserve=60):
        choices=[{'rotation':rotation,'degrees':0.}]
        choices+=bounded_tilt_candidates(rotation,anchor,
            self.calibration['tool_offset_in_ee'],self.calibration['contact_tip_offset_in_ee'],
            self.calibration['table_z'])
        choices+=heading_candidates(rotation,anchor,self.calibration['tool_offset_in_ee'],
                                    self.calibration['contact_tip_offset_in_ee'],self.calibration['table_z'])
        choices+=reverse_tilt_candidates(rotation,anchor,self.calibration['tool_offset_in_ee'],
                                         self.calibration['contact_tip_offset_in_ee'],
                                         self.calibration['table_z'])
        trials=[]
        for candidate in choices:
            strokes=[Waypoint(side,np.r_[t[:3,3],matrix_quat(t[:3,:3])],0.,name,
                              2 if name.endswith('_hit') else HIGH_DWELL,anchor)
                     for name,t in center_strokes(self.keys,candidate['rotation'],anchor)]
            accepted=self._audit(side,list(prefix)+strokes,anchor,reserve=reserve,local_only=True)
            audit=dict(self.program_notes.get('path_audit',{}))
            # Finite subdivision, not a looser accuracy bound. Execute the same
            # smaller step that passed the second full-route audit.
            failed=audit.get('target_index',-1)-len(prefix)
            if not accepted and audit.get('reason')=='joint_discontinuity' and 0<=failed<len(strokes):
                strokes[failed].translation_limit=.006
                accepted=self._audit(side,list(prefix)+strokes,anchor,reserve=reserve,local_only=True)
                audit=dict(self.program_notes.get('path_audit',{}))
            entry={k:v for k,v in candidate.items() if k!='rotation'}
            entry['path_audit']=audit;trials.append(entry)
            if accepted:
                self.program_notes['orientation_trials']=trials
                choice={k:v for k,v in entry.items() if k!='path_audit'}
                return self._prefer_precise_strokes(side,rotation,anchor,strokes,choice,prefix,reserve)
        self.program_notes['orientation_trials']=trials
        # X21: original passing routes remain byte-for-byte unchanged. Only
        # after all original choices fail, try this finite orientation lattice.
        # No global IK restarts, tolerance relaxation, changed hit/lift height,
        # key order, grasp point or recovery allowance.
        extended=[]
        if side+'_ee_pose' not in self.state_now or side+'_arm_joint_state' not in self.state_now:
            self.program_notes['extended_route_trials']=[]
            self.program_notes['extended_route_selected']=False
            return None,None
        for candidate in extended_candidates(rotation,anchor,
                self.calibration['tool_offset_in_ee'],self.calibration['contact_tip_offset_in_ee'],
                self.calibration['table_z']):
            strokes=[Waypoint(side,np.r_[t[:3,3],matrix_quat(t[:3,:3])],0.,name,
                              2 if name.endswith('_hit') else HIGH_DWELL,anchor)
                     for name,t in center_strokes(self.keys,candidate['rotation'],anchor)]
            full=list(prefix)+strokes
            clear,minimum=transfer_clearance(self.state_now[side+'_ee_pose'],full,anchor,
                self.calibration['tool_offset_in_ee'],self.calibration['contact_tip_offset_in_ee'],
                self.calibration['table_z'])
            entry={k:v for k,v in candidate.items() if k!='rotation'}
            entry['sampled_nominal_clearance_m']=minimum
            if not clear:
                entry['path_audit']={'accepted':False,'reason':'nominal_transfer_clearance'}
                extended.append(entry);continue
            accepted=self._audit_precise(side,full,anchor,reserve)
            entry['path_audit']=dict(self.program_notes['path_audit']);extended.append(entry)
            if accepted:
                self.program_notes['extended_route_trials']=extended
                self.program_notes['extended_route_selected']=True
                return strokes,{k:v for k,v in entry.items() if k!='path_audit'}
        self.program_notes['extended_route_trials']=extended
        self.program_notes['extended_route_selected']=False
        return None,None

    def _put_back(self, reason):
        side=self.side
        above=self.grasp_pose.copy();above[2]+=.065
        back=[Waypoint(side,above,0.,'xy_return_above',2),
              Waypoint(side,self.grasp_pose.copy(),0.,'xy_return_lower',3),
              Waypoint(side,self.grasp_pose.copy(),1.,'xy_release',8),
              Waypoint(side,above,1.,'xy_retreat',2)]
        self.reason=reason
        if self._audit(side,back,self.anchor if self.anchor is not None else self.nominal_anchor,reserve=5):
            self._set_plan(back,'done')
        else:
            self.finished=True;self.reason=reason+'_hold_no_safe_return'

    def _record_probe(self, rgb, current, force=False):
        """Observe during existing closed-gripper motion, never add a move.

        Missing images remain in the audit log but are not point constraints.
        Do not repeatedly count a stationary image as new geometric evidence.
        """
        self._record_multiview_probe(rgb,current,force)
        self._record_observer(current)
        if len(self.samples)>=32:return
        ee=pose_matrix(current)
        spacing=.003 if force else .015
        if self.samples:
            distances=[np.linalg.norm(ee[:3,3]-s[0][:3,3]) for s in self.samples]
            nearest=int(np.argmin(distances))
            if min(distances)<spacing and not force:return
            if min(distances)<spacing:
                # Dwell images improve a measurement, not geometric baseline.
                old,pixel_old=self.samples[nearest]
                angle=Rotation.from_matrix(old[:3,:3].T@ee[:3,:3]).magnitude()
                if min(distances)>.00025 or angle>.002:return
                evidence=self.probe_evidence[nearest]
                if self.step in evidence.get('fusion_steps',[evidence['step']]):return
                predicted=self.perception.camera.project([ee[:3,:3]@self.nominal_anchor+ee[:3,3]])[0]
                pixel=head_pixel(rgb,predicted,self.HEAD_SEARCH_PX)
                if pixel is None or pixel_old is None:return
                pixel=pixel+self.perception.camera.project([old[:3,:3]@self.nominal_anchor+old[:3,3]])[0]-predicted
                group=evidence.setdefault('stationary_pixels',[pixel_old.tolist()])
                if len(group)>=3:return
                group.append(pixel.tolist())
                evidence.setdefault('fusion_steps',[evidence['step']]).append(self.step)
                if np.max(np.linalg.norm(np.array(group)-np.mean(group,axis=0),axis=1))>1.:
                    evidence['stationary_inconsistent']=True
                merged=np.mean(group,axis=0)
                self.samples[nearest]=(old,merged)
                evidence['observed_pixel']=merged.tolist()
                return
        predicted=self.perception.camera.project(
            [ee[:3,:3]@self.nominal_anchor+ee[:3,3]])[0]
        pixel=head_pixel(rgb,predicted,self.HEAD_SEARCH_PX)
        self.samples.append((ee,pixel))
        self.probe_evidence.append({'step':self.step,'actual_ee':np.asarray(current).tolist(),
            'predicted_pixel':predicted.tolist(),
            'observed_pixel':None if pixel is None else pixel.tolist()})

    def _record_multiview_probe(self,rgb,current,force=False):
        if self.side is None or self.nominal_anchor is None or len(self.multiview_samples)>=48:return
        wrist=self.current_vision.get('cam_'+self.side+'_wrist',{}).get('color')
        if wrist is None:return
        ee=pose_matrix(current)
        if self.multiview_samples:
            if self.multiview_samples[-1]['step']==self.step:return
            distance=min(np.linalg.norm(ee[:3,3]-s['ee'][:3,3]) for s in self.multiview_samples)
            if distance<.004:return
        measurement=measure_wrist(wrist,self.nominal_anchor)
        self.multiview_samples.append(dict(step=self.step,ee=ee,measurement=measurement))
        summary=None if measurement is None else {k:v.tolist() if isinstance(v,np.ndarray) else v
            for k,v in measurement.items() if k!='boundary'}
        self.multiview_evidence.append(dict(step=self.step,camera='cam_'+self.side+'_wrist',
            actual_ee=np.asarray(current).tolist(),measurement=summary))

    def _record_observer(self,current):
        if not self.observer_attempted or self.observer_home is None or len(self.observer_samples)>=40:return
        other='right' if self.side=='left' else 'left'
        ee=pose_matrix(current);camera=wrist_camera(pose_matrix(self.state_now[other+'_ee_pose']))
        if self.observer_samples:
            last=self.observer_samples[-1]
            if self.step==last['step']:return
            if np.linalg.norm(ee[:3,3]-last['ee'][:3,3])<.004:return
        nominal=ee[:3,:3]@self.nominal_anchor+ee[:3,3]
        pixel=observer_pixel(self.current_vision.get('cam_'+other+'_wrist',{}).get('color'),camera,nominal)
        self.observer_samples.append(dict(step=self.step,ee=ee,camera=camera,pixel=pixel))
        self.observer_evidence.append(dict(step=self.step,camera='cam_'+other+'_wrist',
            actual_held_ee=np.asarray(current).tolist(),actual_observer_ee=np.asarray(self.state_now[other+'_ee_pose']).tolist(),
            pixel=None if pixel is None else pixel.tolist()))

    def _begin_observer(self):
        self.observer_attempted=True
        if self.step>210:return False
        other='right' if self.side=='left' else 'left'
        if other+'_ee_pose' not in self.state_now or other+'_arm_joint_state' not in self.state_now:return False
        home=np.asarray(self.state_now[other+'_ee_pose']).copy()
        held=pose_matrix(self.state_now[self.side+'_ee_pose'])
        trials=[];self.program_notes['observer_path_trials']=trials
        camera_anchor=MOUNT[:3,3].copy()
        for proposal in observer_candidates(pose_matrix(home),held,self.nominal_anchor):
            frame=proposal['target'];goal=np.r_[frame[:3,3],matrix_quat(frame[:3,:3])]
            view=Waypoint(other,goal,1.,'xy_observer_aim',3,camera_anchor)
            back=Waypoint(other,home.copy(),1.,'xy_observer_return',2,camera_anchor)
            accepted=self._audit(other,[view,back],camera_anchor,reserve=230,local_only=True)
            audit=dict(self.program_notes['path_audit'])
            trials.append(dict({k:v for k,v in proposal.items() if k!='target'},path_audit=audit))
            if accepted:
                self.observer_home=home;self.observer_started=self.step
                self._set_plan([view],'observer_probe');self.reason='xy_aim_independent_wrist';return True
        return False

    def _observer_probe(self):
        current=np.asarray(self.state_now[self.side+'_ee_pose']).copy()
        # Move the held tool across actual independent-camera views, not just
        # repeated stationary pixels. A closed grip remains latched throughout.
        for direction in ([.04,0,.01],[-.04,0,.01],[0,-.04,.01]):
            target=current.copy();target[:3]+=direction
            points=[Waypoint(self.side,target,0.,'xy_probe_observer_out',3,self.nominal_anchor),
                    Waypoint(self.side,current.copy(),0.,'xy_probe_observer_back',3,self.nominal_anchor)]
            if self._audit(self.side,points,self.nominal_anchor,reserve=210,local_only=True):
                self._set_plan(points,'observer_return');self.reason='xy_independent_wrist_motion';return
        self._observer_return()

    def _observer_return(self):
        other='right' if self.side=='left' else 'left';camera_anchor=MOUNT[:3,3].copy()
        back=[Waypoint(other,self.observer_home.copy(),1.,'xy_observer_return',2,camera_anchor)]
        if self._audit(other,back,camera_anchor,reserve=180,local_only=True):
            self._set_plan(back,'bind');self.reason='xy_restore_observer_arm'
        else:self.finished=True;self.reason='xy_observer_return_path_rejected_hold'

    def _bind(self):
        # Keep every visible observation (including inconsistent ones); never
        # select the best-fitting subset. >=3 and all original fit gates stay.
        if any(e.get('stationary_inconsistent') for e in self.probe_evidence):
            self._put_back('xy_attachment_inconsistent_stationary_rgb');return
        visible=[s for s in self.samples if s[1] is not None]
        fit=fit_moving_anchor(self.perception.camera,
            [s[0] for s in visible],[s[1] for s in visible],stable=True)
        # Preserve every previously valid head-only binding exactly. Only a
        # genuinely unknown fit may use new measured wrist scale/rigidity.
        if fit['state']=='unknown' and self.multiview_samples:
            fused=confirm_multiview(self.perception.camera,self.multiview_samples,self.samples,self.nominal_anchor)
            self.program_notes['multiview_confirmation']={k:v for k,v in fused.items() if k!='ee_anchor'}
            if fused['state'] in ('consistent','inconsistent'):fit=fused
        if fit['state']=='unknown' and self.observer_samples:
            external=confirm_observer(self.perception.camera,self.observer_samples,self.samples,self.nominal_anchor)
            self.program_notes['observer_confirmation']={k:v for k,v in external.items() if k!='ee_anchor'}
            if external['state'] in ('consistent','inconsistent'):fit=external
        self.binding={k:v for k,v in fit.items() if k!='ee_anchor'}
        if fit['state']=='unknown' and self.side is not None and not self.observer_attempted:
            if self._begin_observer():return
        if fit['state']=='unknown' and not self.extra_probe_used and self.side is not None:
            self.extra_probe_used=True
            current=pose_matrix(self.state_now[f'{self.side}_ee_pose'])
            trials=[]
            self.program_notes['information_probe_trials']=trials
            proposals=information_probe_candidates(self.perception.camera,current,
                                                    self.nominal_anchor,self.samples)
            for candidate in proposals:
                target=candidate['target']
                extra=np.r_[target[:3,3],matrix_quat(target[:3,:3])]
                probe=[Waypoint(self.side,extra,0.,'xy_probe_extra',3,self.nominal_anchor)]
                passed=self._audit(self.side,probe,self.nominal_anchor,local_only=True,reserve=220)
                trials.append({**{k:v for k,v in candidate.items() if k!='target'},
                               'path_audit':dict(self.program_notes.get('path_audit',{}))})
                if passed:
                    self._set_plan(probe,'bind');self.reason='xy_add_information_probe';return
        if fit['state']!='consistent':
            self._put_back('xy_attachment_'+fit['state']);return
        anchor=fit['ee_anchor']
        if np.linalg.norm(anchor-self.nominal_anchor)>.035:
            self._put_back('xy_binding_outside_grasp_geometry');return
        self.anchor=anchor
        if not self.keyboard_refreshed:
            self.stage='keyboard';self.reason='xy_bound_pending_keyboard_review';return
        self._start_strokes()

    def _start_strokes(self):
        anchor=self.anchor
        rotation=quat_matrix(self.grasp_pose[3:])
        waypoints,orientation=self._choose_strokes(self.side,rotation,anchor)
        trials=self.program_notes.get('orientation_trials',[])
        self.program_notes['held_tilt_trials']=trials[1:]
        if waypoints is None:
            self._put_back('xy_bound_path_rejected');return
        self.program_notes['held_tilt_degrees']=orientation['degrees']
        self.program_notes['held_heading_degrees']=orientation.get('heading_degrees',0.)
        # X17 returns to the X14 motion/binding baseline. The X16 validation
        # helpers stay testable but are not entered by this controller.
        self._set_plan(waypoints,'done')
        self.reason='xy_ordered_head_strokes'
        self.program_notes['independent_validation_enabled']=False

    def _refresh_keyboard(self, rgb):
        """One post-grasp rigid-pose check, not eight independent targets.

        Keep the previously measured template if the arm occludes the board.
        Accept only a small, stable rigid update; a large/unstable displacement
        requests putback rather than silently striking the old position.
        """
        held=np.asarray(self.state_now[self.side+'_ee_pose']) if self.side is not None else None
        if held is not None:
            if self.keyboard_reference_ee is None:self.keyboard_reference_ee=held.copy()
            if np.linalg.norm(held[:3]-self.keyboard_reference_ee[:3])>.003:
                self.keyboard_images=[];self.keyboard_refresh_frames=[];self.keyboard_reference_ee=held.copy()
        image=self.perception.rectified(rgb,.78685)
        image[self.perception.valid==0]=0
        self.keyboard_images.append(image.copy())
        fit=fit_fixed_keyboard(image)
        self.keyboard_refresh_frames.append(None if fit is None else
            rectified_xy(self.perception,fit['key_pixels'],image.shape))
        if len(self.keyboard_refresh_frames)<3:
            self.reason='xy_postgrasp_keyboard_observe';return False
        visible=[f for f in self.keyboard_refresh_frames if f is not None]
        if len(visible)<2:
            if len(self.keyboard_images)<9:
                self.reason='xy_postgrasp_keyboard_observe';return False
            self.keyboard_refreshed=True
            self.program_notes['keyboard_refresh']={'state':'occluded_keep_initial','valid_frames':len(visible)}
            return True
        values=np.array(visible);mean=values.mean(0)
        spread=float(np.max(np.linalg.norm(values-mean,axis=2)))
        shift=float(np.max(np.linalg.norm(mean-self.keys,axis=1)))
        accepted=spread<=.003 and shift<=.020
        if not accepted and len(self.keyboard_images)<9:
            self.reason='xy_keyboard_collect_stationary_consensus';return False
        if not accepted:
            # Three disjoint temporal means use all nine stationary images.
            # Refit one rigid template per block; never choose a favorable
            # frame or move eight targets independently to pass the check.
            groups=[]
            for block in np.array_split(self.keyboard_images,3):
                averaged=np.rint(np.mean(block,axis=0)).astype(np.uint8)
                model=fit_fixed_keyboard(averaged)
                groups.append(None if model is None else rectified_xy(self.perception,model['key_pixels'],averaged.shape))
            if all(g is not None for g in groups):
                values=np.array(groups);mean=values.mean(0)
                spread=float(np.max(np.linalg.norm(values-mean,axis=2)))
                shift=float(np.max(np.linalg.norm(mean-self.keys,axis=1)))
                accepted=spread<=.003 and shift<=.020
            self.program_notes['keyboard_temporal_consensus']={'images':len(self.keyboard_images),
                'valid_block_fits':sum(g is not None for g in groups),'accepted':bool(accepted)}
        self.keyboard_refreshed=True
        self.program_notes['keyboard_refresh']={'state':'accepted' if accepted else 'rejected',
            'valid_frames':len(visible),'spread_m':spread,'max_shift_m':shift}
        if not accepted:
            self._put_back('xy_keyboard_changed_or_unstable');return False
        self.keys=mean
        return True

    def _begin_anchor_validation(self, remaining, recovery_name=None):
        """Audit a bounded independent view sequence AND every remaining key."""
        current=pose_matrix(self.state_now[self.side+'_ee_pose'])
        start=current
        pre=[]
        if recovery_name is not None:
            if self.validation_observation_pose is None:
                return False
            # First release downward pressure through this key's original high
            # target. Then return to a previously ACTUALLY visible high view.
            hit=pose_matrix(remaining[0].pose)
            head=hit[:3,:3]@self.anchor+hit[:3,3]
            head[2]=lift_head_height()
            high=hit.copy();high[:3,3]=head-hit[:3,:3]@self.anchor
            pre=[Waypoint(self.side,np.r_[high[:3,3],matrix_quat(high[:3,:3])],0.,
                          'xy_validate_retreat',HIGH_DWELL,self.anchor),
                 Waypoint(self.side,self.validation_observation_pose.copy(),0.,
                          'xy_validate_return_view',3,self.anchor)]
            start=pose_matrix(self.validation_observation_pose)
        rotation=quat_matrix(remaining[0].pose[3:])
        trials=[]
        for frames in validation_frames(start,rotation,self.anchor,self.perception.camera):
            prefix=pre+[Waypoint(self.side,np.r_[f[:3,3],matrix_quat(f[:3,:3])],0.,
                                 f'xy_validate_view_{i}',3,self.anchor)
                        for i,f in enumerate(frames)]
            suffix=list(remaining)
            if recovery_name is not None:
                high=pose_matrix(remaining[0].pose)
                head=high[:3,:3]@self.anchor+high[:3,3];head[2]=lift_head_height()
                high[:3,3]=head-high[:3,:3]@self.anchor
                suffix=[Waypoint(self.side,np.r_[high[:3,3],matrix_quat(high[:3,:3])],0.,
                                 recovery_name[:-3]+'hover',HIGH_DWELL,self.anchor)]+suffix
            accepted=self._audit(self.side,prefix+suffix,self.anchor,reserve=60,local_only=True)
            trials.append(dict(self.program_notes.get('path_audit',{})))
            if accepted:
                self.validation_context={
                    'old_samples':[(e.copy(),None if p is None else p.copy()) for e,p in self.samples],
                    'preliminary':self.anchor.copy(),'new_samples':[],'evidence':[],
                    'remaining':suffix,'recovery_name':recovery_name}
                self.program_notes['independent_binding_validation']={
                    'phase':'collecting','started_step':self.step,
                    'recovery_key':recovery_name,'path_trials':trials,
                    'absolute_accuracy_verified':False,'new_evidence':self.validation_context['evidence']}
                self._set_plan(prefix,'verify_binding')
                self.reason='xy_independent_binding_validation'
                return True
        self.program_notes['independent_binding_validation']={
            'phase':'path_rejected','path_trials':trials,'recovery_key':recovery_name}
        return False

    def _record_validation_view(self,rgb,current,force=False):
        context=self.validation_context
        if context is None:return
        old=context['old_samples'];new=context['new_samples']
        if len(old)+len(new)>=32:return
        ee=pose_matrix(current)
        # Rotated actual poses are independent views even at the same EE xyz.
        # Stationary repetitions never inflate the observation count.
        if any(not distinct_view(ee,e,translation=.003 if force else .008,
                                 rotation=.035) for e,_ in old+new):return
        predicted=self.perception.camera.project(
            [ee[:3,:3]@context['preliminary']+ee[:3,3]])[0]
        pixel=head_pixel(rgb,predicted,self.HEAD_SEARCH_PX)
        new.append((ee,None if pixel is None else pixel.copy()))
        context['evidence'].append({'step':self.step,'actual_ee':np.asarray(current).tolist(),
            'predicted_pixel':predicted.tolist(),
            'observed_pixel':None if pixel is None else pixel.tolist(),
            'held_out_from_preliminary_fit':True})

    def _finish_anchor_validation(self):
        context=self.validation_context
        if context is None:
            self.finished=True;self.reason='xy_missing_validation_context_hold';return
        verdict=validate_anchor(self.perception.camera,context['old_samples'],
                                context['new_samples'],context['preliminary'],self.nominal_anchor)
        self.program_notes['independent_binding_validation'].update(
            phase='completed',finished_step=self.step,
            verdict={k:v for k,v in verdict.items() if k!='ee_anchor'})
        # Keep all observations, including missing and conflicting ones, in the
        # audit. Do not repair the fit by selecting only a favorable subset.
        self.samples=context['old_samples']+context['new_samples']
        self.probe_evidence.extend(context['evidence'])
        self.validation_context=None
        if verdict['state']!='consistent':
            self.binding_validated=False
            self.binding={**self.binding,'state':verdict['state'],
                          'reason':verdict['reason'],'independent_views_checked':True}
            self._put_back('xy_independent_binding_'+verdict['state']);return
        previous=context['preliminary']
        self.anchor=verdict['ee_anchor']
        self.binding={**verdict['all_point_fit'],'independent_views_checked':True}
        self.binding_validated=True
        visible=[e for e,p in context['new_samples'] if p is not None]
        self.validation_observation_pose=np.r_[visible[-1][:3,3],matrix_quat(visible[-1][:3,:3])]
        remaining=context['remaining']
        # Preserve functional targets, orientation choices and per-target
        # timing. Only the newly measured EE-to-head binding may change xyz.
        for w in remaining:
            frame=pose_matrix(w.pose)
            target=frame[:3,:3]@previous+frame[:3,3]
            w.pose=np.r_[target-frame[:3,:3]@self.anchor,matrix_quat(frame[:3,:3])]
            w.anchor_offset=self.anchor
        if not self._audit(self.side,remaining,self.anchor,reserve=60,local_only=True):
            self._put_back('xy_validated_remaining_path_rejected');return
        if context['recovery_name'] is not None:
            name=context['recovery_name']
            self.hit_retries[name]=self.hit_retries.get(name,0)+1
            self.hit_stall.clear_window()
            self.reason='xy_same_key_retry_after_new_evidence'
        else:
            self.reason='xy_ordered_head_strokes_after_validation'
        self._set_plan(remaining,'done')

    def _recover_hit_with_validation(self,w,current,action,evidence):
        """No blind same-target retry: new real views and all-point fit first."""
        tries=self.hit_retries.get(w.name,0)
        event={'step':self.step,'phase':w.name,'prior_retries':tries,
               'remaining_steps':500-self.step,'evidence':evidence,'task_success':False}
        action[w.side+'_ee_pose']=np.asarray(current).copy()
        action[w.side+'_ee_joint_state']=np.array([0.])
        if tries>=MAX_RETRIES_PER_KEY or sum(self.hit_retries.values())>=MAX_RETRIES_TOTAL:
            self.finished=True;self.reason='xy_hit_recovery_limit_hold'
            event['action']='hold_retry_limit'
        elif self._begin_anchor_validation(list(self.plan[self.index:]),recovery_name=w.name):
            event['action']='audited_retreat_and_new_view_validation'
        else:
            self.finished=True;self.reason='xy_revalidation_path_rejected_hold'
            event['action']='hold_no_audited_revalidation'
        self.recovery_events.append(event)
        self.program_notes['hit_recovery_events']=self.recovery_events


    def _nominal_return(self):
        if self.grasp_pose is None:return []
        above=self.grasp_pose.copy();above[2]+=.065
        return [Waypoint(self.side,above,0.,'xy_return_above',2),
                Waypoint(self.side,self.grasp_pose.copy(),0.,'xy_return_lower',3),
                Waypoint(self.side,self.grasp_pose.copy(),1.,'xy_release',8),
                Waypoint(self.side,above,1.,'xy_retreat',2)]

    def _recover_motion(self,w,current,action,evidence):
        action[w.side+'_ee_pose']=np.asarray(current).copy()
        action[w.side+'_ee_joint_state']=np.array([w.gripper])
        event={'step':self.step,'phase':w.name,'evidence':evidence,
               'task_success':False,'remaining_steps':500-self.step}
        self.motion_recoveries.append(event)
        if self.motion_retry_count>=1 or self.last_reached_pose is None:
            self.finished=True;self.reason='xy_motion_recovery_exhausted_hold'
            event['action']='hold';return
        self.motion_retry_count+=1
        anchor=self.anchor if self.anchor is not None else self.nominal_anchor
        if anchor is None:
            self.finished=True;self.reason='xy_motion_recovery_no_anchor_hold';return
        previous=self.last_reached_pose.copy()
        back=Waypoint(w.side,previous,w.gripper,'xy_motion_backtrack',2,anchor)
        if w.name=='xy_approach':
            via=previous.copy();via[2]+=.03
            route=[back,Waypoint(w.side,via,w.gripper,'xy_motion_via',2,anchor)]+list(self.plan[self.index:])
            audit_route=route+self._nominal_return()
        else:
            # Probe geometry is replaceable. Keep ALL previous observations.
            # Never skip a key/strike or reinterpret progress as task success.
            raised=previous.copy();raised[2]+=.025
            lateral=raised.copy();lateral[0]-=.04
            route=[back,Waypoint(w.side,raised,0.,'xy_probe_recover_0',3,anchor),
                   Waypoint(w.side,lateral,0.,'xy_probe_recover_1',3,anchor)]
            audit_route=route+self._nominal_return()
        for point in route:
            if point.name.startswith(('xy_motion_','xy_probe_recover_')):
                point.translation_limit=.006
        if self._audit(w.side,audit_route,anchor,reserve=70,local_only=True):
            self._set_plan(route,'bind');self.reason='xy_bounded_motion_recovery'
            event['action']='audited_backtrack_retry'
            action[w.side+'_ee_pose']=np.asarray(current).copy()
            action[w.side+'_ee_joint_state']=np.array([w.gripper])
        else:
            event['audit']=dict(self.program_notes.get('path_audit',{}))
            event['action']='hold_no_continuous_return'
            self.finished=True;self.reason='xy_motion_recovery_path_rejected_hold'


    def _guarded_stall_exit(self, w, current, anchor, action, event):
        """End pressure only via this key's original lift and audited suffix."""
        remaining=self.plan[self.index+1:]
        expected=w.name[:-3]+'lift'
        if not remaining or remaining[0].name!=expected:
            event['exit_audit']={'accepted':False,'reason':'missing_same_key_lift'}
            accepted=False
        else:
            accepted=self._audit(w.side,remaining,anchor,local_only=True)
            event['exit_audit']=dict(self.program_notes.get('path_audit',{}))
        if accepted:
            self.index+=1;self.age=self.dwell=0;self.hit_stall.clear_window()
            self.reason='xy_contact_limited_stroke_unverified'
            event['action']='stop_downward_pressure_then_guarded_lift'
            action[f'{w.side}_ee_pose']=bounded_anchor_pose(
                current,remaining[0].pose,anchor,.012,.10)
        else:
            self.finished=True;self.reason='xy_hit_escape_path_rejected_hold'
            event['action']='hold_escape_audit_failed'

    def _recover_stalled_hit(self, w, current, anchor, action, evidence):
        """One same-key retry, then bounded release of downward pressure.

        Neither plateau nor retry termination claims a successful hit. Native
        evaluation stays independent. Every escape checks remaining local IK
        and budget; do not skip a key solely because a timer expired.
        """
        if self.binding_validated:
            self._recover_hit_with_validation(w,current,action,evidence);return
        tries=self.hit_retries.get(w.name,0)
        event={'step':self.step,'phase':w.name,'prior_retries':tries,
               'evidence':evidence,'task_success':False,
               'remaining_steps':max(0,500-self.step),'reserve_steps':60}
        if tries<MAX_RETRIES_PER_KEY:
            if sum(self.hit_retries.values())>=MAX_RETRIES_TOTAL:
                self.finished=True;self.reason='xy_hit_recovery_limit_hold'
                event['action']='hold_retry_limit'
            else:
                retreat=current.copy();retreat[2]+=RETRY_LIFT
                escape=Waypoint(w.side,retreat,0.,'xy_recover_'+w.name[:-4],3,anchor)
                remaining=[escape]+self.plan[self.index:]
                if self._audit(w.side,remaining,anchor,local_only=True):
                    self.hit_retries[w.name]=tries+1
                    self.plan=self.plan[:self.index]+remaining
                    self.age=self.dwell=0;self.hit_stall.clear_window()
                    self.reason='xy_same_key_retry'
                    event['action']='vertical_retreat_then_same_key'
                    action[f'{w.side}_ee_pose']=bounded_anchor_pose(
                        current,retreat,anchor,.012,.10)
                else:
                    retry_audit=dict(self.program_notes.get('path_audit',{}))
                    event['retry_audit']=retry_audit
                    if retry_audit.get('reason')=='budget':
                        event['exit_trigger']='retry_budget'
                        self._guarded_stall_exit(w,current,anchor,action,event)
                    else:
                        self.finished=True;self.reason='xy_hit_retry_path_rejected_hold'
                        event['action']='hold_retry_audit_failed'
        else:
            event['exit_trigger']='same_key_retry_exhausted'
            self._guarded_stall_exit(w,current,anchor,action,event)
        if self.finished:
            action[f'{w.side}_ee_pose']=current.copy()
        self.recovery_events.append(event)
        self.program_notes['hit_recovery_events']=self.recovery_events

    def act(self, raw):
        obs=restrict_observation(raw);validate_observation(obs)
        self.state_now=obs['state'];self.step+=1
        if self.home is None:self.home={s:np.asarray(self.state_now[f'{s}_ee_pose']).copy() for s in ARMS}
        action={f'{s}_{k}':np.asarray(self.state_now.get(f'{s}_{k}',[1.])).copy()
                for s in ARMS for k in ('ee_pose','ee_joint_state')}
        for side,target in self.gripper_targets.items():
            action[f'{side}_ee_joint_state']=np.array([target])
        rgb=obs['vision']['cam_head']['color']
        self.current_vision=obs['vision']
        if not self.finished and self.index>=len(self.plan):
            self.plan=[];self.index=0
            if self.stage=='observe':self._observe(rgb)
            elif self.stage=='bind':self._bind()
            elif self.stage=='observer_probe':self._observer_probe()
            elif self.stage=='observer_return':self._observer_return()
            elif self.stage=='keyboard':
                if self.keyboard_refreshed or self._refresh_keyboard(rgb):self._start_strokes()
            elif self.stage=='verify_binding':self._finish_anchor_validation()
            else:self.finished=True;self.reason='xy_program_complete_not_success'
        if not self.finished and self.index<len(self.plan):
            w=self.plan[self.index];current=np.asarray(self.state_now[f'{w.side}_ee_pose'])
            anchor=self.anchor if self.anchor is not None else self.nominal_anchor
            if w.side!=self.side and w.anchor_offset is not None:anchor=np.asarray(w.anchor_offset)
            if anchor is None:anchor=np.asarray(self.calibration['tool_offset_in_ee'])
            move=.003 if w.name=='xy_descend' else getattr(w,'translation_limit',.012)
            if w.name.endswith('_hit'):
                # 25 mm final descent plus public 13 mm head radius stays slow.
                move=tool_translation_step(current,w.pose,anchor,move,.038)
            action[f'{w.side}_ee_pose']=bounded_anchor_pose(current,w.pose,anchor,move,.10)
            action[f'{w.side}_ee_joint_state']=np.array([w.gripper])
            self.gripper_targets[w.side]=float(w.gripper)
            error=np.linalg.norm(current[:3]-w.pose[:3])
            angle=Rotation.from_matrix(quat_matrix(current[3:]).T@quat_matrix(w.pose[3:])).magnitude()
            if w.name.startswith('xy_validate_view_'):
                self._record_validation_view(rgb,current,force=error<.003 and angle<.035)
            if w.name.startswith('xy_probe_'):
                self._record_probe(rgb,current,force=error<.003 and angle<.035)
            self.age+=1
            reached=error<.003 and angle<.035
            if w.name.startswith('xy_key_') and w.name.endswith(('_hover','_lift')):
                arrival=high_arrival(current,w.pose,anchor)
                self.program_notes['high_arrival']={'step':self.step,'phase':w.name,**arrival}
                reached=reached and arrival['accepted']
            self.motion={'side':w.side,'current_ee':current.tolist(),'goal_ee':w.pose.tolist(),
                         'position_error':float(error),'orientation_error':float(angle)}
            stall=self.hit_stall.observe(w.name,current,w.pose,anchor)
            if not reached and stall['qualified']:
                self._recover_stalled_hit(w,current,anchor,action,stall)
                validate_action(action)
                return action
            progress=self.motion_progress.observe(w.name,current,w.pose)
            if not reached and self.age>=18 and progress['qualified']:
                self._recover_motion(w,current,action,progress)
                validate_action(action);return action
            self.dwell=self.dwell+1 if reached else 0
            if self.dwell>=w.dwell:
                if w.name=='xy_close' or w.name.startswith('xy_probe_'):
                    self._record_probe(rgb,current,force=True)
                if w.name.startswith('xy_validate_view_'):
                    self._record_validation_view(rgb,current,force=True)
                self.last_reached_pose=current.copy()
                self.index+=1;self.age=self.dwell=0
                if self.index==len(self.plan):self.stage=self.next_stage
            elif self.age>85:
                self.finished=True;self.reason='xy_motion_timeout_hold'
        validate_action(action)
        return action

    def diagnostics(self):
        result=super().diagnostics()
        result['xylophone']={'stage':self.stage,'binding':self.binding,
             'probe_count':len(self.samples),'visible_probe_count':sum(p is not None for _,p in self.samples),
             'anchor':None if self.anchor is None else self.anchor.tolist(),
             'probe_evidence':self.probe_evidence,
             'multiview_evidence':self.multiview_evidence,
             'observer_evidence':self.observer_evidence,
             'motion_recoveries':self.motion_recoveries}
        return result
