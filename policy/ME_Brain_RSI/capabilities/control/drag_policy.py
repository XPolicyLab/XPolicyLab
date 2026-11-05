"""Experimental RGB-only T stem clamp/drag; original pushing remains separate."""
import numpy as np
from .policy import Policy,Waypoint
from .planar_drag import tabletop_drag
from .geometry import quat_matrix
from .contracts import restrict_observation,validate_observation


class TabletopDragPolicy(Policy):
    def reset(self):
        super().reset()
        self.drag_used_sides=set()
        self.drag_robot_poses={}

    def act(self,raw):
        obs=restrict_observation(raw)
        validate_observation(obs)
        self.drag_robot_poses={s:np.asarray(obs['state'][s+'_ee_pose'],float).copy()
                               for s in ('left','right')}
        return super().act(raw)

    def __init__(self,task,calibration):
        if task.removesuffix('_random')!='push_T':
            raise ValueError('Drag candidate is only for push_T')
        super().__init__(task,calibration)
        self.tabletop_drag_enabled=True

    def make_drag_plan(self,source,target,side):
        distance=float(np.linalg.norm(target.xy-source.xy))
        angle=float(np.arctan2(np.sin(target.angle-source.angle),np.cos(target.angle-source.angle)))
        self.program_notes={'controller':'tabletop_drag_candidate','t_distance':distance,'t_angle_error':angle}
        if distance<.005 and abs(angle)<.09:
            self.return_home()
            return True
        # Do not drag a far-side target using a fully extended opposite arm.
        # Release at the shared table centre and reacquire from fresh RGB.
        if abs(source.xy[0])<.02:
            side='left' if target.xy[0]<0 else 'right'
        stage_xy=np.asarray(target.xy,float).copy()
        stage_yaw=target.angle
        if source.xy[0]*target.xy[0]<0 and abs(source.xy[0])>=.02:
            stage_xy=np.array([0.,float(target.xy[1])])
            stage_yaw=source.angle
        travel=stage_xy-source.xy
        # Stay inside the 25 cm RGB association gate with a 2 cm margin,
        # while avoiding an extra grasp cycle for a roughly 43 cm transfer.
        if np.linalg.norm(travel)>.23:
            stage_xy=source.xy+travel*.23/np.linalg.norm(travel)
            stage_yaw=source.angle
        self.program_notes.update(stage_xy=stage_xy.tolist(),stage_yaw=float(stage_yaw),
                                  active_side=side,relay_reobserve=True)
        # Public footprint stem spans local x=-20..40 mm, width 20 mm.
        stem=source.xy+.010*np.array([np.cos(source.angle),np.sin(source.angle)])
        # Parallel jaws admit a pi-equivalent heading. Choose the direction
        # facing out from the public robot base to avoid a wrist half-turn.
        # This is a geometric heuristic, not an IK/collision certificate.
        base=np.array([-.3 if side=='left' else .3,-.45])
        radial=stem-base
        grasp_yaw=source.angle
        if np.dot(radial,[np.cos(grasp_yaw),np.sin(grasp_yaw)])<0:
            grasp_yaw+=np.pi
        if side in self.drag_used_sides and side in self.drag_robot_poses:
            # The same arm may finish the prior rotation on the opposite
            # equivalent jaw heading. Do not flip it back by pi on regrasp.
            current=quat_matrix(self.drag_robot_poses[side][3:])
            candidates=[source.angle,source.angle+np.pi]
            grasp_yaw=max(candidates,key=lambda yaw:np.trace(
                current.T@quat_matrix(self.pose(side,stem,.13,yaw)[3:])))
        self.program_notes['grasp_yaw']=float(grasp_yaw)
        self.drag_used_sides.add(side)
        # Keep distal finger tips at the 15 mm block's mid-height. The TCP
        # is above the tips by the public tool-vs-contact offset difference.
        height=.0075+float(self.calibration['contact_tip_offset_in_ee'][0]
                          -self.calibration['tool_offset_in_ee'][0])
        grasp=self.pose(side,stem,height,grasp_yaw)
        z=self.calibration['table_z']+.0075
        anchor=quat_matrix(grasp[3:]).T@(np.r_[source.xy,z]-grasp[:3])
        self.add(side,stem,.13,1.,'drag_approach',3,angle=grasp_yaw)
        self.plan.append(Waypoint(side,grasp,1.,'drag_contact',3))
        self.plan.append(Waypoint(side,grasp.copy(),0.,'drag_close',12))
        path=tabletop_drag(source.xy,source.angle,stage_xy,stage_yaw,z,grasp,
                          angle_step=.15,distance_step=.02)
        for phase,pose in path:
            self.plan.append(Waypoint(side,pose,0.,'drag_'+phase,1,anchor.copy()))
        last=path[-1][1]
        self.plan.append(Waypoint(side,last.copy(),1.,'drag_release',15))
        raised=last.copy();raised[2]+=.12
        self.plan.append(Waypoint(side,raised,1.,'drag_lift',3))
        self.reobserve_after_plan=True
        self.view_cleared=False
        self.reason='executing_tabletop_drag_not_success_claim'
        return True
