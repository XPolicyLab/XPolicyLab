"""X35: X34 plus grasp planning that does not quit on one arm or on 2 mm IK noise.

Sampled-IK audits use 4 mm / 0.06 rad tolerances (the public arm model is only
approximate; execution is bounded step by step anyway). Grasp candidates are
ranked by nominal joint travel because the simulator arm is joint-velocity
limited. When every legacy route is rejected, X35 plans its own compact grasp
(ranked, executed at normal speed, with audited lateral probes so the head can
still be measured); when the preferred arm has no route at all, the other arm
is planned from the same stationary detections. The nominal anchor fallback
includes the observed head sag, and a hit that plateaus on contact lifts
immediately instead of retrying at the same place.
"""
import numpy as np
from .xylophone_x34 import XylophonePolicy as X34
from .xylophone_policy import handle_grasp_point
from .xylophone_feedback import ExecutionWindow
from .geometry import pose_matrix
from .policy import Waypoint
from .xylophone_path import audit_tool_path
from .xylophone_routes import MAX_JOINT_STEP,JOINT_MARGIN

AUDIT_POSITION_TOLERANCE=.004
AUDIT_ORIENTATION_TOLERANCE=.06
JOINT_RATE_PER_STEP=.02
NO_GRASP_REASONS=('xy_no_continuous_grasp_return_path',)
COMPACT_AUDIT_STEP=.006
COMPACT_YAW_EXTRAS=(0.,180.,-15.,15.,165.,195.)
EXTRA_PROBE_PATTERNS=(((.05,0.,0.),(0.,.05,0.)),((-.05,0.,0.),(0.,-.05,0.)),
                      ((.05,0.,0.),(0.,-.05,0.)),((-.05,0.,0.),(0.,.05,0.)))
# Four consistent RGB bindings (layouts 13/28/45) all placed the held head
# 1.1-1.7 cm below the planar grasp prior: the mallet sags toward its head.
NOMINAL_SAG=.013


class XylophonePolicy(X34):
    def reset(self):
        super().reset()
        self.forced_side=None;self.grasp_sides_tried=[]
        self.program_notes.update(candidate='X35_two_arm_compact_grasp')

    def _audit(self,side,waypoints,anchor,reserve=60,local_only=False,
               position_tolerance=AUDIT_POSITION_TOLERANCE,orientation_tolerance=AUDIT_ORIENTATION_TOLERANCE):
        result=audit_tool_path(self.arms[side],pose_matrix(self.state_now[side+'_ee_pose']),
            [pose_matrix(w.pose) for w in waypoints],anchor,self.state_now[side+'_arm_joint_state'],
            remaining_steps=max(1,500-self.step),reserve_steps=reserve,dwell=3,
            translation=.012,rotation=.10,position_tolerance=position_tolerance,
            orientation_tolerance=orientation_tolerance,
            target_translations=[.003 if w.name=='xy_descend' else getattr(w,'translation_limit',.012) for w in waypoints],
            target_slow_distances=[.038 if w.name.endswith('_hit') else 0. for w in waypoints],
            target_dwells=[w.dwell for w in waypoints],max_joint_step=MAX_JOINT_STEP,
            joint_limit_margin=JOINT_MARGIN,local_seed_retries=False)
        summary={k:v for k,v in result.items() if k!='samples'}
        samples=result.get('samples') or []
        if samples:
            joints=np.array([np.asarray(self.state_now[side+'_arm_joint_state'],float)]+[s['joints'] for s in samples])
            summary['joint_travel_rad']=float(np.sum(np.max(np.abs(np.diff(joints,axis=0)),axis=1)))
        self.program_notes['path_audit']=summary
        return result['accepted']

    def _rank_grasp(self,audit,alternative_index):
        margin=audit.get('min_joint_margin_rad') or 0.
        travel=audit.get('joint_travel_rad')
        cost=audit['nominal_steps'] if travel is None else max(audit['nominal_steps'],travel/JOINT_RATE_PER_STEP)
        return (0 if margin>=.30 else 1,cost,alternative_index)

    def _choose_side(self,grasp_xy,keys):
        side=self.forced_side if self.forced_side is not None else super()._choose_side(grasp_xy,keys)
        if side not in self.grasp_sides_tried:self.grasp_sides_tried.append(side)
        return side

    def _rank_compact(self,audit,yaw_extra,alternative_index):
        return self._rank_grasp(audit,alternative_index)

    def _grasp_planned(self):
        return not (self.finished and self.reason in NO_GRASP_REASONS)

    def _compact_grasp(self):
        d=getattr(self,'last_detections',None)
        if d is None or self.side is None:return False
        head=np.asarray(d['head_xy'],float);axis=np.asarray(d['shaft_axis'],float)
        point=handle_grasp_point(head,axis);side=self.side;trials=[];candidates=[]
        for yaw_extra in COMPACT_YAW_EXTRAS:
            yaw=np.arctan2(axis[1],axis[0])+np.deg2rad(yaw_extra)
            grab=self.pose(side,point[:2],point[2]-self.calibration['table_z'],yaw)
            frame=pose_matrix(grab);anchor=frame[:3,:3].T@(np.r_[head,.7956]-frame[:3,3])
            above=grab.copy();above[2]+=.09
            lift=grab.copy();lift[2]+=.065
            grasp=[Waypoint(side,above,1.,'xy_approach',2),Waypoint(side,grab,1.,'xy_descend',3),
                   Waypoint(side,grab,0.,'xy_close',10),Waypoint(side,lift,0.,'xy_probe_0',3)]
            back=[Waypoint(side,lift,0.,'xy_return_above',2),Waypoint(side,grab,0.,'xy_return_lower',3),
                  Waypoint(side,grab,1.,'xy_release',8),Waypoint(side,above,1.,'xy_retreat',2)]
            # Fine audit steps keep the seeded IK continuous; execution uses the
            # normal X34 bounds, so the limit is removed again below.
            for w in grasp+back:w.translation_limit=COMPACT_AUDIT_STEP
            accepted=self._audit(side,grasp+back,anchor,reserve=220)
            audit=dict(self.program_notes.get('path_audit',{}))
            trials.append(dict(yaw_extra=yaw_extra,accepted=accepted,audit=audit))
            if accepted:candidates.append((self._rank_compact(audit,yaw_extra,len(candidates)),yaw_extra,grab,anchor,grasp))
        self.program_notes['x35_compact_grasp_trials']=trials
        if not candidates:return False
        _,yaw_extra,grab,anchor,grasp=min(candidates,key=lambda c:c[0])
        for w in grasp:del w.translation_limit
        probe_trials=[]
        for offsets in EXTRA_PROBE_PATTERNS:
            probes=[]
            for j,delta in enumerate(offsets):
                target=grasp[-1].pose.copy();target[:3]+=delta
                probes.append(Waypoint(side,target,0.,f'xy_probe_{j+1}',3))
            accepted=self._audit(side,grasp+probes,anchor,reserve=150)
            probe_trials.append(dict(offsets=offsets,accepted=accepted,audit=dict(self.program_notes.get('path_audit',{}))))
            if accepted:grasp=grasp+probes;break
        self.program_notes['x35_compact_grasp']=dict(yaw_extra=yaw_extra,side=side,probes=len(grasp)-4,probe_trials=probe_trials)
        self.finished=False;self.grasp_pose=grab;self.nominal_anchor=anchor
        self.samples=[];self._set_plan(grasp,'bind');self.reason='xy_grasp_and_probe'
        return True

    def _observe(self,rgb):
        # X35 plans its own compact fallback; the X32 one is never entered.
        self.x32_grasp_recovery=True
        super()._observe(rgb)
        if self._grasp_planned() or self._compact_grasp():return
        other=[s for s in ('left','right') if s not in self.grasp_sides_tried]
        d=getattr(self,'last_detections',None)
        if not other or d is None:return
        frame={'keys':np.asarray(d['keys_xy'],float),'head':np.asarray(d['head_xy'],float),
               'axis':np.asarray(d['shaft_axis'],float)}
        # Same stationary measurement, other arm.
        self.forced_side=other[0];self.finished=False
        self.measurement_frames=[frame,frame]
        self.program_notes['x35_other_arm_retry']=dict(step=self.step,first=self.grasp_sides_tried[0],second=other[0])
        super()._observe(rgb)
        self.forced_side=None
        if not self._grasp_planned():self._compact_grasp()

    def _bind(self):
        super()._bind()
        if self.binding.get('state')=='nominal_fallback' and self.anchor is not None:
            self.anchor=np.asarray(self.anchor,float).copy();self.anchor[0]+=NOMINAL_SAG
            self.binding={**self.binding,'sag_compensation_m':NOMINAL_SAG}

    def _recover_stalled_hit(self,w,current,anchor,action,evidence):
        guard=evidence.get('contact_guard',{}) if isinstance(evidence,dict) else {}
        if guard.get('qualified'):
            # The head already rests at hit depth; repeating the same descent
            # cannot change the official check. Lift and continue the sequence.
            self.recovery_events.append(dict(step=self.step,phase=w.name,action='contact_plateau_lift',
                                             evidence=evidence,task_success=False))
            self.program_notes['hit_recovery_events']=self.recovery_events
            action[w.side+'_ee_pose']=np.asarray(current).copy()
            self._advance(w,current,'contact_plateau_lift');self.execution=ExecutionWindow();return
        super()._recover_stalled_hit(w,current,anchor,action,evidence)
