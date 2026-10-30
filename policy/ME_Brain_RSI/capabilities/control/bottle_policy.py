"""Budgeted RGB bottle disposal with support-based arm transfer.

Only RGB/proprioception/public geometry. No claim that reaching a waypoint or
seeing a raised shape is official task success. Both upright and lying support
transfers remain development candidates, not collision-free guarantees.
"""
import numpy as np
from .bottle_lying import plan_lying_pick,after_lying_lift
from .policy import Policy,Waypoint
from .bottle_geometry import fit_bottles
from .contracts import restrict_observation
from .kinematics import SerialArm,pose_matrix
from .bottle_attachment import capture_upper_patch,check_upper_following
from .bottle_transport import radial_transport
from .bottle_grasp import upright_descent_clearance
from .bottle_handoff import staging_site,staging_candidates
from .bottle_execution import BottleExecution
from .geometry import quat_matrix
from .bottle_revision17 import remember_held
from .bottle_revision18 import attachment_decision,upright_references,robot_lift_ok


def exposed_drop_xy(radius):
    """Clear the public table's x=-.70 edge and remain inside the bin.

    Bucket centre x=-.63 is UNDER the tabletop. Upright bottle radius plus
    15 mm clearance determines a point in the exposed left strip instead.
    Reject a body too wide for the public .23 half-width; don't clip it in.
    """
    if not np.isfinite(radius) or not .015<radius<.055:return None
    x=-.70-radius-.015
    if x-radius<=-.63-.23+.01:return None
    return np.array([x,-.1])


class BottlePolicy(BottleExecution,Policy):
    def reset(self):
        super().reset()
        self.pending_bottle=None
        self.source_attempts=[]
        self.bin_available=False
        self.robot_state={}
        self.budget_return=False
        self.drop_programs=0
        self.handoff_programs=0
        self.open_timeout_recoveries=0

    def _recover_open_timeout(self,waypoint):
        """Withdraw an unclosed approach, not an unconfirmed carried object.

        Cool down the failed source, retain arrival/grasp thresholds, and
        require a new RGB observation after a same-arm vertical retreat/home.
        This is not collision sensing or proof that another bottle is reachable.
        """
        if (waypoint.name not in {'bottle_approach','bottle_insert_fingers'}
                or waypoint.gripper!=1. or self.pending_bottle is None
                or self.open_timeout_recoveries>=4 or self.remaining_steps()<120):return False
        side=waypoint.side
        pose=np.asarray(self.robot_state.get(f'{side}_ee_pose',[]),float)
        opening=np.asarray(self.robot_state.get(f'{side}_ee_joint_state',[]),float).reshape(-1)
        if (pose.shape!=(7,) or len(opening)!=1 or not np.isfinite(np.r_[pose,opening]).all()
                or opening[0]<.9 or np.linalg.norm(pose[3:])<.5):return False
        pose=pose.copy();pose[3:]/=np.linalg.norm(pose[3:])
        tcp=pose[:3]+quat_matrix(pose[3:])@np.asarray(self.calibration['tool_offset_in_ee'])
        clearance=self.perception.z+2*self.pending_bottle['height']+.045
        rise=max(.06,float(clearance-tcp[2]))
        if rise>.25:return False
        raised=pose.copy();raised[2]+=rise
        retreat=Waypoint(side,raised,1.,'bottle_open_timeout_retreat',3)
        ok,reports=self._qualify([retreat])
        if not ok:return False
        self.plan=[retreat];self.return_home(sides=[side])
        self.index=self.age=self.dwell=0
        self.finish_source('failed')
        self.pending_bottle=None;self.finished=False;self.reobserve_after_plan=True
        self.open_timeout_recoveries+=1
        self.program_notes['open_timeout_recovery']=dict(side=side,rise=rise,ik=reports,
            attempts=self.open_timeout_recoveries,failed_source_cooldown_and_variant=True)
        self.reason='bottle_open_timeout_withdraw_reobserve'
        return True

    def _transport(self,source,drop,height,side='left'):
        from .bottle_revision23 import compress_transport
        offset=np.asarray(self.calibration['tool_offset_in_ee'])
        stages=[Waypoint(side,self.pose(side,xy,height,yaw,0.),0.,
                         'bottle_carry',1,offset.copy())
                for xy,yaw in radial_transport(source,drop,side=side)]
        return compress_transport(stages)

    def _qualify(self,stages):
        from .bottle_revision23 import qualification_stages
        stages=qualification_stages(stages)
        side=stages[0].side
        arm=SerialArm.public_x5(side)
        q=np.asarray(self.robot_state.get(f'{side}_arm_joint_state',np.zeros(6)))
        reports=[]
        for stage in stages:
            if stage.side!=side:return False,reports
            result=arm.inverse(pose_matrix(stage.pose),initial=q,restarts=1)
            reports.append({k:result[k] for k in ('position_error_m','orientation_error_rad')})
            q=np.asarray(result['joints'])
            if result['position_error_m']>.004 or result['orientation_error_rad']>.06 or abs(q[-1])>3.0:
                return False,reports
        return True,reports

    def make_plan(self,rgb):
        if self.budget_return:
            self.finished=True;self.reason=self.stop_reason or 'bottle_final_home_reserve';return False
        if self.pending_bottle is not None:
            return self._after_lift(rgb)
        self.held_bottle=None  # previous plan completed; never clear on a carry timeout
        detections,_=self.perception.detect(rgb,self.task)
        objects=self.perception.select(self.task,detections)
        self.current_bottle_objects=objects
        self.bin_available |= 'target' in objects
        self.last_detections={k:dict(xy=v.xy.tolist(),height=v.grasp_height,semantic=v.semantic) for k,v in objects.items()}
        if not self.bin_available:
            self.reason='bin_target_unavailable';return False
        # Qualified direct-left disposal can include right-side upright bodies.
        # Otherwise keep right support placement plus fresh RGB before regrasp.
        candidates=self.ranked_sources(objects)
        if not candidates:
            return self.observe_or_finish('no_candidate_or_budget')
        for track_id,source in candidates:
            if source.grasp_height<=.075:
                if plan_lying_pick(self,source,rgb):
                    self.begin_source(track_id)
                    return True
                continue
            side=self.source_sides.get(id(source),'left' if source.xy[0]<.10 else 'right')
            radius=float(max(source.size)/2)
            if side=='right':
                drop=next((xy for xy in staging_candidates(source,objects)
                           if self.receiver_feasible(xy,float(source.grasp_height),radius)),None)
            else:drop=exposed_drop_xy(radius)
            if drop is None:continue
            base=np.array([-.3 if side=='left' else .3,-.45])
            direction=source.xy-base;direction/=np.linalg.norm(direction)
            yaw=float(np.arctan2(direction[1],direction[0]));h=float(source.grasp_height)
            variant=self.bottle_tracks[track_id]['variant']
            yaw+=(0.,np.pi/6,-np.pi/6)[min(variant,2)]
            behind=source.xy-.09*direction
            descent=upright_descent_clearance(h,float(max(source.size)/2))
            if descent is None:continue
            self.plan=[]
            # Clear the whole upright body before descending with open fingers;
            # avoid the low folded-arm backoff posture next to the robot base.
            self.add(side,source.xy,descent['hover_height'],1.,'bottle_approach',angle=yaw,pitch=0.)
            self.add(side,source.xy,h,1.,'bottle_insert_fingers',angle=yaw,pitch=0.)
            self.add(side,source.xy,h,0.,'bottle_close',10,angle=yaw,pitch=0.)
            self.add(side,source.xy,h+.08,0.,'bottle_lift',5,angle=yaw,pitch=0.)
            try:
                transport=self._transport(source.xy,drop,h+.08,side)
            except ValueError:
                self.plan=[];continue
            if side=='right':
                direction_to_drop=drop-base
                drop_yaw=float(np.arctan2(direction_to_drop[1],direction_to_drop[0]))
                transport.append(Waypoint(side,self.pose(side,drop,h,drop_yaw,0.),0.,'bottle_stage_lower',5))
            ok,reports=self._qualify(self.plan+transport)
            self.program_notes={'bottle_path_ik':reports,'collision_checked':False,
                                'acting_arm':side,'staging_screen':'detected RGB bottles only',
                                'source_arm_selection':('qualified_direct_left' if side=='left' and source.xy[0]>=.10 else 'original_side_route'),
                                'other_gripper_spacing_only':True,
                                'open_descent_geometry':descent,
                                'transport_kind':'translation_with_public_base_radial_heading',
                                'bin_target_source':'public task fixed geometry',
                                'bin_rgb_verified':False,
                                'unsupported':'no complete world collision model',
                                'drop_programs_not_successes':self.drop_programs}
            if not ok:
                self.plan=[]
                self.bottle_tracks[track_id]['variant']=min(2,variant+1)
                continue
            self.pending_bottle=dict(xy=source.xy.copy(),height=h,radius=radius,yaw=yaw,behind=behind,drop=drop,side=side)
            self.pending_bottle['upper_patch']=capture_upper_patch(
                rgb,self.perception.camera,source.xy,h,float(max(source.size)/2),
                self.perception.z,yaw)
            self.pending_bottle['attachment_refs']=upright_references(
                rgb,self.perception.camera,self.pending_bottle['upper_patch'])
            self.begin_source(track_id)
            self.reobserve_after_plan=True;self.reason='bottle_picking';return True
        return self.observe_or_finish('path_unqualified')

    def _after_lift(self,rgb):
        pending=self.pending_bottle
        if pending.get('mode')=='lying':
            return after_lying_lift(self,rgb)
        # Preserve the pre-existing raised-shape confirmation path. The
        # extra table-colour proposals are only for objects on the table.
        fits=fit_bottles(self.perception.camera,self.perception.z+.08+pending.get('proof_extra',0.),rgb,color_proposals=False)
        matching=[f for f in fits if np.linalg.norm(f['xy']-pending['xy'])<.035
                  and abs(f['grasp_height']-pending['height'])<.02]
        h=pending['height'];yaw=pending['yaw'];xy=pending['xy']
        side=pending.get('side','left')
        following=check_upper_following(pending.get('upper_patch'),rgb,self.perception.camera,
                                        self.robot_state,self.calibration['tool_offset_in_ee'],side=side)
        self.program_notes['upper_rgb_following']=following
        strong=bool(matching) or following['confirmed']
        confirmed=attachment_decision(self,pending,rgb,pending.get('attachment_refs',[]),
            strong,robot_lift_ok(self,pending),
            following.get('moved_correlation',-1.),following.get('stationary_correlation',-1.))
        if confirmed is None:return True
        self.program_notes['raised_shape_rgb_confirmed']=bool(matching)
        self.program_notes['grasp_rgb_confirmed']=confirmed
        if confirmed:
            # Move the grasp anchor while following a radial wrist heading;
            # do not first reverse the arm orientation at the source.
            self.plan=self._transport(xy,pending['drop'],h+.08+pending.get('proof_extra',0.),side)
            end=pending['drop']-np.array([-.3 if side=='left' else .3,-.45])
            drop_yaw=float(np.arctan2(end[1],end[0]))
            if side=='right':
                self.add(side,pending['drop'],h,0.,'bottle_stage_lower',5,angle=drop_yaw,pitch=0.)
                self.add(side,pending['drop'],h,1.,'bottle_stage_open',12,angle=drop_yaw,pitch=0.)
                self.add(side,pending['drop'],2*h+.045,1.,'bottle_stage_retreat',angle=drop_yaw,pitch=0.)
                self.handoff_programs+=1
                self.staged_hint=pending['drop'].copy()
                self.reason='bottle_support_transfer_reobserve_required'
            else:
                self.add(side,pending['drop'],h+.08+pending.get('proof_extra',0.),1.,'bottle_release',12,angle=drop_yaw,pitch=0.)
                self.add(side,pending['drop']+[.08,0],h+.10,1.,'bottle_retract',angle=drop_yaw,pitch=0.)
                self.drop_programs+=1
                self.reason='bottle_disposal_not_success_claim'
        else:
            # Unknown attachment: put back at the original visible source,
            # open, retreat, and exclude this attempt from an endless loop.
            self.program_notes['raised_shape_rgb_confirmed']=False
            self.add(side,xy,h,0.,'bottle_recovery_lower',5,angle=yaw,pitch=0.)
            self.add(side,xy,h,1.,'bottle_recovery_open',10,angle=yaw,pitch=0.)
            self.add(side,pending['behind'],h+.08,1.,'bottle_recovery_retreat',angle=yaw,pitch=0.)
            self.reason='bottle_lift_not_confirmed'
        self.finish_source('staged' if confirmed and side=='right' else 'released_unverified' if confirmed else 'failed')
        # Donor must vacate the shared workspace; direct left disposal only
        # retracts and observes, instead of homing both arms after every bottle.
        if side=='right':self.return_home(sides=['right'])
        if confirmed:remember_held(self,pending)
        self.pending_bottle=None;self.reobserve_after_plan=True
        return True
