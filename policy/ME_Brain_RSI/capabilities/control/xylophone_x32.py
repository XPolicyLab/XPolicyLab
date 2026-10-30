"""X32: connected paths, qualified RGB features, optional bounded Astra advice.

Same algorithm in both experiment arms; only advisor.enabled differs. No
native IDs, hidden object poses, score, result files or simulation access.
"""
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from .geometry import pose_matrix, matrix_quat, quat_matrix
from .policy import Waypoint
from .xylophone_hexashell import XylophonePolicy as HexBase, camera_views, hx, confirm_cones
from .xylophone_policy import XylophonePolicy as Legacy, center_strokes, handle_grasp_point
from .xylophone_connected import ConnectedArm
from .xylophone_path import audit_tool_path
from .xylophone_routes import MAX_JOINT_STEP, JOINT_MARGIN
from .xylophone_lift import HIGH_DWELL,lift_head_height
from .xylophone_extended_routes import transfer_clearance
from .xylophone_extended_routes import tool_clearance


def cone_from_candidates(rgb,k,frame,predicted,roi=None):
    """Use several measured interior seeds, not an arbitrary largest blob.

    GPT ROI may associate a component but never provides the measured centre.
    Original upstream silhouette validation remains byte-identical. Detailed
    reasons retain clipping/merged-mask failure instead of one catch-all.
    """
    seed=hx.project((k,frame,rgb.shape[:2]),predicted)
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    mask=np.uint8(((hsv[:,:,0]<10)|(hsv[:,:,0]>155))&(hsv[:,:,1]>65)&(hsv[:,:,2]>40))
    n,labels,stats,_=cv2.connectedComponentsWithStats(mask)
    fits=[];reasons=[];bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    for i in range(1,n):
        if stats[i,cv2.CC_STAT_AREA]<20:continue
        ys,xs=np.where(labels==i)
        if np.min(np.linalg.norm(np.c_[xs,ys]-seed,axis=1))>50:continue
        if roi is not None:
            x0,y0,x1,y1=roi
            if not np.any((xs>=x0)&(xs<=x1)&(ys>=y0)&(ys<=y1)):continue
        # The old distance-transform maximum could be on a connected red key
        # instead of the small head. Choose observed saturated pixels nearest
        # the projected head, plus the component's interior maximum.
        dist=cv2.distanceTransform(np.uint8(labels==i),cv2.DIST_L2,3)
        near=np.argsort((xs-seed[0])**2+(ys-seed[1])**2)
        interior=[j for j in near if dist[ys[j],xs[j]]>=1.5][:1]
        maxy,maxx=np.unravel_index(dist.argmax(),dist.shape)
        seeds=[[float(xs[j]),float(ys[j])] for j in interior]+[[float(maxx),float(maxy)]]
        for point in seeds:
            try:
                fit=hx.silhouette(bgr,k,point)
                if min(fit['hue'],180-fit['hue'])>=25:continue
                if np.linalg.norm(fit['pixel']-seed)>50:continue
                fit['intrinsics']=k
                if not any(np.linalg.norm(fit['pixel']-f['pixel'])<.5 for f in fits):fits.append(fit)
            except (ValueError,np.linalg.LinAlgError) as exc:reasons.append(str(exc))
    if len(fits)!=1:return None,dict(reason='no_unique_valid_cone',candidates=len(fits),failures=sorted(set(reasons)))
    return fits[0],dict(reason='qualified_upstream_silhouette',pixel=fits[0]['pixel'].tolist())


class XylophonePolicy(HexBase):
    def reset(self):
        super().reset()
        self.arms={s:ConnectedArm.public_x5(s) for s in ('left','right')}
        self.x32_rows=[];self.x32_observations=[];self.x32_last=-100
        self.x32_grasp_recovery=False;self.x32_view_used=False
        from .xylophone_astra import Advisor
        self.advisor=Advisor()
        self.advice={};self._advised=set()
        self.program_notes.update(candidate='X32_connected_HexaShell_Astra',
            gpt_api_enabled=self.advisor.enabled, api_events=self.advisor.events)

    def _audit(self,side,waypoints,anchor,reserve=60,local_only=False,
               position_tolerance=.002,orientation_tolerance=.035):
        result=audit_tool_path(self.arms[side],pose_matrix(self.state_now[side+'_ee_pose']),
            [pose_matrix(w.pose) for w in waypoints],anchor,self.state_now[side+'_arm_joint_state'],
            remaining_steps=max(1,500-self.step),reserve_steps=reserve,dwell=3,
            translation=.012,rotation=.10,position_tolerance=position_tolerance,
            orientation_tolerance=orientation_tolerance,
            target_translations=[.003 if w.name=='xy_descend' else getattr(w,'translation_limit',.012) for w in waypoints],
            target_slow_distances=[.038 if w.name.endswith('_hit') else 0. for w in waypoints],
            target_dwells=[w.dwell for w in waypoints],max_joint_step=MAX_JOINT_STEP,
            joint_limit_margin=JOINT_MARGIN,local_seed_retries=False)
        self.program_notes['path_audit']={k:v for k,v in result.items() if k!='samples'}
        return result['accepted']

    def _advise(self,event,options):
        if event in self._advised:return self.advice.get(event,{})
        self._advised.add(event)
        answer=self.advisor.ask(event,self.current_vision,options,
            dict(remaining_steps=500-self.step,reason=self.reason,
                 binding_state=self.binding.get('state','unknown'),
                 rejection=self.program_notes.get('path_audit',{})))
        self.advice[event]=answer
        self.program_notes['astra_advice']=self.advice
        return answer

    def _observe(self,rgb):
        super()._observe(rgb)
        if self.reason!='xy_no_continuous_grasp_return_path' or self.x32_grasp_recovery:return
        self.x32_grasp_recovery=True
        advice=self._advise('pregrasp_path',['compact_probe','raised_approach'])
        head=np.asarray(self.last_detections['head_xy']);axis=np.asarray(self.last_detections['shaft_axis'])
        point=handle_grasp_point(head,axis);side=self.side;trials=[]
        # Local pickup + lift + safe return is checked before committing.
        # Optional wide observations are planned after grasp from actual state.
        for yaw_extra in (0.,180.,-15.,15.,165.,195.):
            yaw=np.arctan2(axis[1],axis[0])+np.deg2rad(yaw_extra)
            grab=self.pose(side,point[:2],point[2]-self.calibration['table_z'],yaw)
            frame=pose_matrix(grab);anchor=frame[:3,:3].T@(np.r_[head,.7956]-frame[:3,3])
            above=grab.copy();above[2]+=.09
            lift=grab.copy();lift[2]+=.065
            grasp=[Waypoint(side,above,1.,'xy_approach',2),Waypoint(side,grab,1.,'xy_descend',3),
                   Waypoint(side,grab,0.,'xy_close',10),Waypoint(side,lift,0.,'xy_probe_0',3)]
            if advice.get('choice')=='raised_approach':
                raised=np.asarray(self.state_now[side+'_ee_pose']).copy();raised[2]+=.04
                grasp.insert(0,Waypoint(side,raised,1.,'xy_approach_raise',2))
            back=[Waypoint(side,lift,0.,'xy_return_above',2),Waypoint(side,grab,0.,'xy_return_lower',3),
                  Waypoint(side,grab,1.,'xy_release',8),Waypoint(side,above,1.,'xy_retreat',2)]
            for w in grasp+back:w.translation_limit=.006
            accepted=self._audit(side,grasp+back,anchor,reserve=220)
            trials.append(dict(yaw_extra=yaw_extra,accepted=accepted,audit=dict(self.program_notes['path_audit'])))
            if accepted:
                self.finished=False;self.grasp_pose=grab;self.nominal_anchor=anchor
                self.samples=[];self._set_plan(grasp,'bind');self.reason='xy_grasp_and_probe'
                self.program_notes['x32_compact_grasp_trials']=trials
                return
        self.program_notes['x32_compact_grasp_trials']=trials

    def _record_probe(self,rgb,current,force=False):
        super()._record_probe(rgb,current,force)
        if self.nominal_anchor is None or self.step-self.x32_last<4:return
        self.x32_last=self.step;ee=pose_matrix(current)
        predicted=ee[:3,3]+ee[:3,:3]@self.nominal_anchor
        rois=self.advice.get('head_confirmation',{}).get('rois',{})
        for camera,(k,frame,image) in camera_views(self.perception.camera,self.state_now,self.current_vision).items():
            fit,diagnostic=cone_from_candidates(image,k,frame,predicted,rois.get(camera))
            local=np.linalg.inv(ee)@frame
            if fit is not None and not any(r['camera']==camera and np.linalg.norm(r['local']-local)<.008 for r in self.x32_rows):
                if len(self.x32_rows)<36:self.x32_rows.append(dict(camera=camera,ee=ee,local=local,fit=fit))
            self.x32_observations.append(dict(step=self.step,camera=camera,**diagnostic))
        self.program_notes['x32_cone_observations']=self.x32_observations

    def _bind(self):
        fit=confirm_cones(self.x32_rows,self.nominal_anchor) if self.nominal_anchor is not None else {}
        self.program_notes['x32_cone_binding']={k:v for k,v in fit.items() if k!='ee_anchor'}
        # Preserve every original positive confirmation. New geometric
        # contradictions remain blocking; API cannot turn unknown into true.
        if fit.get('state')=='consistent':
            self.hx_rows=self.x32_rows
        elif fit.get('state')=='inconsistent':
            self.binding=self.program_notes['x32_cone_binding'];self._put_back('xy_x32_inconsistent');return
        if fit.get('state')=='unknown' and not self.x32_view_used:
            self.x32_view_used=True
            advice=self._advise('head_confirmation',['keep_geometric_history','head_lateral_view','external_wrist_view'])
            if advice.get('choice')=='external_wrist_view' and not self.observer_attempted:
                if self._begin_observer():return
            if advice.get('choice')=='head_lateral_view' and self.step<180:
                current=np.asarray(self.state_now[self.side+'_ee_pose']).copy()
                target=current.copy();target[0]+=.035 if current[0]<0 else -.035
                route=[Waypoint(self.side,target,0.,'xy_probe_x32_lateral',3,self.nominal_anchor)]
                if self._audit(self.side,route,self.nominal_anchor,reserve=250):
                    self._set_plan(route,'bind');self.reason='xy_x32_information_view';return
        super()._bind()

    def _choose_strokes(self,side,rotation,anchor,prefix=(),reserve=60):
        # First keep a passing imported/legacy route. This prevents the new
        # fallback from silently replacing previously demonstrated successes.
        route,choice=super()._choose_strokes(side,rotation,anchor,prefix,reserve)
        if route is not None:return route,choice
        advice=self._advise('strike_path',['current_orientation','inboard_then_turn'])
        functional=self._functional_strokes(side,anchor,prefix,reserve)
        if functional is not None:return functional,dict(degrees=0.,heading_degrees=0.,orientation='per_key_functional_ik')
        current=pose_matrix(self.state_now[side+'_ee_pose']);trials=[]
        rotations=[(current[:3,:3],0.,0.)]
        for pitch in (0.,-10.,10.,-20.,20.):
            for yaw in (0.,-20.,20.,-40.,40.):
                rotations.append((Rotation.from_euler('z',yaw,degrees=True).as_matrix()@rotation@
                    Rotation.from_euler('y',pitch,degrees=True).as_matrix(),pitch,yaw))
        for r,pitch,yaw in rotations:
            # Transfer at the current orientation first, near the first key;
            # only then turn. Avoid a remote-feature pivot at a reach limit.
            head=current[:3,3]+current[:3,:3]@anchor
            high=max(head[2],.90)
            via=current.copy();via[:3,3]=np.r_[head[:2],high]-via[:3,:3]@anchor
            near=via.copy();near[:3,3]=np.r_[self.keys[0],high]-near[:3,:3]@anchor
            if advice.get('choice')=='inboard_then_turn':
                near[:3,3]=np.r_[.6*self.keys[0]+.4*head[:2],high]-near[:3,:3]@anchor
            frames=[('xy_transfer_raise',via),('xy_transfer_near',near)]
            # Small orientation waypoints ensure execution follows the same
            # finite angle subdivision as the nominal audit.
            rv=Rotation.from_matrix(r@near[:3,:3].T).as_rotvec()
            for i in range(1,max(1,int(np.ceil(np.linalg.norm(rv)/.05)))+1):
                n=max(1,int(np.ceil(np.linalg.norm(rv)/.05)))
                t=near.copy();t[:3,:3]=Rotation.from_rotvec(rv*i/n).as_matrix()@near[:3,:3]
                # Pivot around the observed head above the board, not a low contact.
                t[:3,3]=near[:3,3]+near[:3,:3]@anchor-t[:3,:3]@anchor
                frames.append(('xy_transfer_turn_'+str(i),t))
            frames+=center_strokes(self.keys,r,anchor)
            points=[Waypoint(side,np.r_[t[:3,3],matrix_quat(t[:3,:3])],0.,name,
                    1 if name.startswith('xy_transfer_turn') else 2 if name.endswith('_hit') else HIGH_DWELL,anchor)
                    for name,t in frames]
            for w in points:
                if w.name.startswith('xy_transfer'):w.translation_limit=.006
            full=list(prefix)+points
            clear,minimum=transfer_clearance(self.state_now[side+'_ee_pose'],full,anchor,
                self.calibration['tool_offset_in_ee'],self.calibration['contact_tip_offset_in_ee'],self.calibration['table_z'])
            accepted=clear and self._audit(side,full,anchor,reserve=reserve)
            trials.append(dict(pitch=pitch,yaw=yaw,clearance=minimum,accepted=accepted,
                audit=dict(self.program_notes['path_audit']) if clear else {'reason':'clearance'}))
            if accepted:
                self.program_notes['x32_transfer_trials']=trials
                return points,dict(degrees=pitch,heading_degrees=yaw)
        self.program_notes['x32_transfer_trials']=trials
        return None,None

    def _functional_strokes(self,side,anchor,prefix,reserve):
        """Spherical head position is the task constraint, not fixed wrist roll.

        Choose realizable per-key EE orientations via public forward kinematics,
        then audit the EXACT Cartesian interpolation used by the controller.
        A soft orientation preference is NOT a looser pose-audit tolerance.
        """
        arm=self.arms[side];seed=np.asarray(self.state_now[side+'_arm_joint_state'],float)
        actual=pose_matrix(self.state_now[side+'_ee_pose'])
        head=actual[:3,3]+actual[:3,:3]@anchor
        targets=[('xy_transfer_functional_raise',np.r_[head[:2],max(lift_head_height(),head[2])])]
        for i,xy in enumerate(self.keys):
            for phase,z in (('hover',lift_head_height()),('hit',.803),('lift',lift_head_height())):
                # After each fully confirmed lift, travel diagonally to the
                # next key's low target, as in HexaShell's compact chain. All
                # eight hit/lift endpoints remain; no key or lift is skipped.
                if i>0 and phase=='hover':continue
                targets.append((f'xy_key_{i}_{phase}',np.r_[xy,z]))
        points=[];evidence=[]
        for name,target in targets:
            old_rotation=arm.forward(seed)[:3,:3]
            lower=np.maximum(arm.lower+JOINT_MARGIN,seed-.9)
            upper=np.minimum(arm.upper-JOINT_MARGIN,seed+.9)
            def residual(q):
                f=arm.forward(q);p=f[:3,3]+f[:3,:3]@anchor
                clear=tool_clearance(f,anchor,self.calibration['tool_offset_in_ee'],
                                    self.calibration['contact_tip_offset_in_ee'],self.calibration['table_z'])
                # Keep head accuracy primary. Orientation is selected freely
                # but stays close to the previous feasible wrist orientation.
                return np.r_[30*(p-target),.015*Rotation.from_matrix(old_rotation.T@f[:3,:3]).as_rotvec(),
                             .0005*(q-seed),30*max(0.,.008-clear)]
            fit=least_squares(residual,np.clip(seed,lower+1e-8,upper-1e-8),bounds=(lower,upper),
                              max_nfev=120,ftol=1e-10,xtol=1e-10,gtol=1e-10)
            frame=arm.forward(fit.x)
            error=float(np.linalg.norm(frame[:3,3]+frame[:3,:3]@anchor-target))
            clearance=tool_clearance(frame,anchor,self.calibration['tool_offset_in_ee'],
                                    self.calibration['contact_tip_offset_in_ee'],self.calibration['table_z'])
            evidence.append(dict(phase=name,head_error_m=error,nominal_clearance_m=clearance))
            if error>.0005 or clearance<.005:
                self.program_notes['x32_functional_path']=dict(accepted=False,reason='endpoint',endpoints=evidence)
                return None
            points.append(Waypoint(side,np.r_[frame[:3,3],matrix_quat(frame[:3,:3])],0.,name,
                                  2 if name.endswith('_hit') else HIGH_DWELL,anchor))
            seed=fit.x
        full=list(prefix)+points
        clear,minimum=transfer_clearance(self.state_now[side+'_ee_pose'],full,anchor,
            self.calibration['tool_offset_in_ee'],self.calibration['contact_tip_offset_in_ee'],self.calibration['table_z'])
        accepted=clear and self._audit(side,full,anchor,reserve=reserve)
        self.program_notes['x32_functional_path']=dict(accepted=accepted,endpoints=evidence,
            nominal_clearance_m=minimum,audit=dict(self.program_notes['path_audit']) if clear else {'reason':'clearance'})
        return points if accepted else None
