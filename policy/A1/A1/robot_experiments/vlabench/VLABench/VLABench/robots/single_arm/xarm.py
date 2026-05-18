from scipy.spatial.transform import Rotation as R
from dm_control.utils.inverse_kinematics import qpos_from_site_pose
from VLABench.robots.single_arm.base import SingleArm
from VLABench.utils.register import register

@register.add_robot("xarm")
class XArm(SingleArm):
    def __init__(self, **kwargs):
        super().__init__(name=kwargs.get("name", "xarm"), **kwargs)
    
    @property
    def link_base(self):
        return self._mjcf_model.find("body", "link_base")
    
    @property
    def end_effector_site(self):
        return self.mjcf_model.find("site", "link_tcp")
    
    @property
    def gripper_geoms(self):
        left_finger_geoms = self.mjcf_model.find("body", "left_finger").find_all("geom")
        right_finger_geoms = self.mjcf_model.find("body", "right_finger").find_all("geom")
        return left_finger_geoms + right_finger_geoms
          
    def get_qpos_from_ee_pos(self, physics, pos, quat=None, inplace=False, **kwargs):
        ik_result = qpos_from_site_pose(physics,
                                        site_name=self.end_effector_site.full_identifier,
                                        target_pos=pos,
                                        target_quat=quat,
                                        inplace=inplace,
                                        **kwargs)
        target_qpos = ik_result.qpos
        return target_qpos
    
    def initialize_episode(self, physics, random_state):
        super().initialize_episode(physics, random_state)
        for i, qpos in enumerate(self.default_qpos):
            if i < 7: physics.bind(self.mjcf_model.find("joint", f"act{i + 1}")).qpos = qpos
            else: physics.bind(self.mjcf_model.find("joint", f"gripper")).qpos = qpos
        physics.data.ctrl = self.default_qpos