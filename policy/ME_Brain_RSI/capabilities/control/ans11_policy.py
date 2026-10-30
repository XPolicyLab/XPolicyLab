"""ANS11: ANS10 with antipodal grasp offsets for 2/7 and a view-clearing re-detection.

The ANS9 across-height pinch of the 2 spun the released digit by 90 deg (dev5
layout 20). Digits 2 and 7 are now pinched along the standard closing axis but
at the top arc / top bar, where the silhouette is a 34-36 mm solid band with
opposed side faces, instead of the slanted 14-20 mm middle stroke. A digit read
with low confidence right in front of a parked gripper (layout 10: the 0 behind
the right fingers was read as 6) lifts that arm 10 cm before planning.
"""
import numpy as np
from . import ans9_policy as _ans9
from .ans10_policy import Policy as ANS10, FREE_STEP
from .contracts import ARMS
from .geometry import bounded_pose, quat_matrix

_ans9.Y_PINCH_DIGITS.discard(2)
GRASP_OFFSETS={2:(0.,.013),7:(0.,.016)}   # object frame metres, +y toward the digit's top
VIEW_CLEAR_CONFIDENCE=.75
VIEW_CLEAR_RADIUS=.25
VIEW_CLEAR_LIFT=.10
VIEW_CLEAR_STEPS=40


class Policy(ANS10):
    def __init__(self,task,calibration,supervisor=None):
        super().__init__(task,calibration,supervisor)
        self.view_clear=None;self.view_clear_checked=False;self.view_clear_log=[]

    def transfer(self,side,xy,goal,yaw,height,label):
        digit=int(label.split('_')[0]);offset=GRASP_OFFSETS.get(digit)
        tilted=tuple(np.round(np.asarray(goal,float),3).tolist()) in self.tilted_goals
        if offset is None or tilted:return super().transfer(side,xy,goal,yaw,height,label)
        offset=np.asarray(offset,float);c,s=np.cos(yaw),np.sin(yaw)
        source=np.asarray(xy,float)+np.array([[c,-s],[s,c]])@offset;target=np.asarray(goal,float)+offset
        super().transfer(side,source,target,yaw,height,label)
        # Records keep the digit's own position; only the TCP waypoints carry the offset.
        self.transfers[-1].update(source=np.asarray(xy,float).tolist(),target=np.asarray(goal,float).tolist(),grasp_offset=offset.tolist())
        self.grasp_variants.append(dict(label=label,grasp='x_pinch_offset',offset=offset.tolist()))

    def _home_tcp(self,side):
        pose=np.asarray(self.home[side],float)
        return pose[:3]+quat_matrix(pose[3:])@np.asarray(self.calibration['tool_offset_in_ee'])

    def _plan_view_clear(self,obs):
        digits=self.perception.digits_from_rgb(obs['vision']['cam_head']['color'])
        weak=[d for d in digits if d.get('confidence',1.)<VIEW_CLEAR_CONFIDENCE]
        best=None
        for d in weak:
            for side in ARMS:
                distance=float(np.linalg.norm(np.asarray(d['xy'],float)-self._home_tcp(side)[:2]))
                if distance<=VIEW_CLEAR_RADIUS and (best is None or distance<best[0]):best=(distance,side,d)
        self.view_clear_log.append(dict(step=self.step,detected=len(digits),weak=[(d['digit'],round(d['confidence'],3)) for d in weak],
            lifted_side=best and best[1],occluder_distance=best and round(best[0],3)))
        if best is None:return None
        goal=np.asarray(self.home[best[1]],float).copy();goal[2]+=VIEW_CLEAR_LIFT
        return dict(side=best[1],goal=goal,age=0)

    def _act(self,obs):
        if self.planned:return super()._act(obs)
        state=obs['state']
        if self.home is None:self.home={s:np.asarray(state[s+'_ee_pose'],float).copy() for s in ARMS}
        if not self.view_clear_checked:
            self.view_clear_checked=True;self.view_clear=self._plan_view_clear(obs)
        if self.view_clear is not None:
            v=self.view_clear;key=v['side']+'_ee_pose';current=np.asarray(state[key],float);v['age']+=1
            arrived=bool(np.linalg.norm(current[:3]-v['goal'][:3])<.009)
            if arrived or v['age']>VIEW_CLEAR_STEPS:
                self.view_clear=None;self.view_clear_log[-1].update(lift_steps=v['age'],arrived=arrived)
            else:
                self.step+=1;self.reason='view_clear_lift'
                action=self._stationary(state);action[key]=bounded_pose(current,v['goal'],FREE_STEP,self.calibration['max_rotation_step'])
                return action
        home=self.home
        result=super()._act(obs)
        self.home=home   # the parents re-read the (possibly lifted) pose as home; keep the true origin
        return result

    def diagnostics(self):
        result=super().diagnostics()
        result.update(candidate='ANS11',grasp_offsets={str(k):list(v) for k,v in GRASP_OFFSETS.items()},view_clear=self.view_clear_log,
            view_clear_confidence=VIEW_CLEAR_CONFIDENCE,y_pinch_digits=sorted(_ans9.Y_PINCH_DIGITS))
        return result
