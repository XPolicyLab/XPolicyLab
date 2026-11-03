"""Episode-local empty-wrist cup observation before the original pour program."""
import json
from pathlib import Path
import numpy as np
from physicalrsi_observed_geometry import Camera,GeometryUnavailable
from physicalrsi_observed_geometry.calibrated_depth_scene import calibrated_depth_scene
from physicalrsi_observed_geometry.support_projection import support_point
from physicalrsi_observed_geometry.wrist_inspection import qualify_view
from .controller import pose_error,bounded_pose
from .perception import observed_liquid_cup_opening
from .sam.sam3_http import segment_many_http

KEYS=('left_ee_pose','right_ee_pose','left_ee_joint_state','right_ee_joint_state')

class CupInspection:
    def __init__(self,out):
        self.out=Path(out);self.out.mkdir(parents=True,exist_ok=True)
        self.phase='initialize';self.calls=0;self.station_calls=0;self.settle=0
        self.opening=None
    def record(self,**row):
        with (self.out/'trace.jsonl').open('a') as f:f.write(json.dumps({'phase':self.phase,'calls':self.calls,**row},default=lambda x:x.tolist())+'\n')
    def initialize(self,obs):
        from rdj_insert_policy.ik import X5IK
        self.scene=calibrated_depth_scene(obs)
        groups=segment_many_http(self.scene.camera.image,[('cup',.5)],timeout_s=600)
        masks=[self.scene.mask(mask) for group in groups for mask in group.masks if np.asarray(mask).sum()>80]
        if not masks:raise GeometryUnavailable('No observed cup for inspection')
        self.mask=max(masks,key=lambda m:int(m.sum()))
        if any((m&self.mask).sum()/max(1,m.sum())<.8 for m in masks):
            raise GeometryUnavailable('Multiple cup targets are ambiguous')
        y,x=np.nonzero(self.mask);plane=self.scene.depth_calibration_report['observed_plane']
        proposal=support_point([x.mean(),y.mean()],0,camera=self.scene.camera,support_plane=plane)
        self.solver=X5IK()
        self.plan=qualify_view(obs,'right',proposal,self.scene,self.solver)
        # qualify_view samples the complete fixed-position orientation path.
        # Execute that same path with bounded_pose rather than stopping at every
        # numerical qualification station (which consumes task action budget).
        self.poses=[self.plan['poses'][-1]];self.joints=[self.plan['joints'][-1]];self.index=0;self.phase='view'
        self.record(event='qualified',plan=self.plan,proposal=proposal)
    def measure(self,obs):
        head=Camera.from_observation(obs)
        if not np.allclose(head.k,self.scene.camera.k) or not np.allclose(head.transform,self.scene.camera.transform):
            raise GeometryUnavailable('Head camera moved during cup inspection')
        self.opening=observed_liquid_cup_opening(obs,self.mask,self.scene.depth_calibration_report['observed_plane'],segment_many_http)
        self.record(event='opening',opening=self.opening)
        self.poses=[self.plan['original_pose']]
        self.joints=[self.plan['original_joints']]
        self.index=0;self.station_calls=0;self.phase='return'
    def step(self,obs):
        if self.phase=='complete':return None
        action={k:np.asarray(obs['state'][k]).reshape(-1).tolist() for k in KEYS}
        if self.phase=='failed':return [action]
        self.calls+=1
        try:
            if self.calls>200:raise GeometryUnavailable('Cup inspection motion budget exceeded')
            if self.phase=='initialize':self.initialize(obs)
            if self.phase=='settle':
                self.settle+=1
                if self.settle>=3:self.measure(obs)
                return [action]
            arm=self.plan['arm'];target=self.poses[self.index]
            position,angle=pose_error(obs['state'][arm+'_ee_pose'],target)
            if position<.005 and angle<.04:
                if np.max(abs(np.asarray(obs['state'][arm+'_arm_joint_state'])-self.joints[self.index]))>.1:
                    raise GeometryUnavailable('Observed inspection joint branch differs from qualified path')
                self.index+=1;self.station_calls=0
                if self.index==len(self.poses):
                    if self.phase=='view':self.phase='settle';self.settle=0
                    else:self.phase='complete';self.record(event='complete');return None
                    return [action]
                target=self.poses[self.index]
            self.station_calls+=1
            if self.station_calls>20:raise GeometryUnavailable('Cup inspection waypoint stalled')
            action[arm+'_ee_pose']=bounded_pose(obs['state'][arm+'_ee_pose'],target)
            return [action]
        except (GeometryUnavailable,ValueError) as e:
            self.phase='failed';self.record(event='rejected',reason=str(e));return [action]
