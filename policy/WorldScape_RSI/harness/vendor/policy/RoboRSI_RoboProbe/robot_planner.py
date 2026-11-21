"""Robot-only CuRobo adapter: no task/scene collision geometry is supplied.

The native robot configuration conversion below is kept identical to the native
runtime's robot-only prefix. Scene obstacles must not be inferred from asset
constants. Installed CuRobo supports scene_model=None independently of its
self_collision_check=True setting inherited from the native planner.
"""
from copy import deepcopy
from XPolicyLab.policy.WorldScape_RSI.harness.vendor.runtime_support.native_runtime.env.planner_manager.curobo_planner import CuroboPlanner

class PublicCuroboPlanner(CuroboPlanner):
    def _build_robot_and_scene_cfg(self, yml_data, table_height):
        if table_height is not None:
            raise ValueError("Public planner requires no scene geometry")
        kinematics = deepcopy(yml_data.get("robot_cfg", {}).get("kinematics", {}))
        ee_link = kinematics.get("ee_link", None)
        tool_frames = kinematics.get("tool_frames", None)
        if tool_frames is None:
            if ee_link is None:
                raise ValueError(f"robot_cfg.kinematics in {self.yml_path} requires ee_link or tool_frames.")
            tool_frames = [ee_link]
        kinematics["tool_frames"] = list(tool_frames)
        kinematics["format_version"] = 2.0

        cspace = deepcopy(kinematics.get("cspace", {}))
        if "default_joint_position" not in cspace and "retract_config" in cspace:
            cspace["default_joint_position"] = cspace["retract_config"]
        cspace.pop("retract_config", None)
        cspace_allowed = {
            "joint_names",
            "default_joint_position",
            "cspace_distance_weight",
            "null_space_weight",
            "null_space_maximum_distance",
            "max_acceleration",
            "max_jerk",
            "velocity_scale",
            "acceleration_scale",
            "jerk_scale",
            "position_limit_clip",
        }
        cspace = {k: v for k, v in cspace.items() if k in cspace_allowed}
        if "joint_names" not in cspace:
            cspace["joint_names"] = list(self.active_joints_name)
        if "default_joint_position" not in cspace:
            cspace["default_joint_position"] = [0.0 for _ in cspace["joint_names"]]
        kinematics["cspace"] = cspace

        kinematics_allowed = {
            "format_version",
            "base_link",
            "tool_frames",
            "collision_link_names",
            "collision_spheres",
            "collision_sphere_buffer",
            "extra_collision_spheres",
            "self_collision_ignore",
            "self_collision_buffer",
            "asset_root_path",
            "mesh_link_names",
            "lock_joints",
            "extra_links",
            "cspace",
            "urdf_path",
            "use_global_cumul",
            "grasp_contact_link_names",
        }
        kinematics = {k: v for k, v in kinematics.items() if k in kinematics_allowed}
        robot_cfg_v2 = {
            "robot_cfg": {"kinematics": kinematics},
            "load_dynamics": bool(yml_data.get("load_dynamics", False)),
        }
        # Relocate only robot asset paths; robot geometry/settings are unchanged.
        import os
        from pathlib import Path
        robot_root = Path(os.environ['ROBODOJO_ROOT']) / 'Assets/Robots/x5'
        local = robot_cfg_v2['robot_cfg']['kinematics']
        local['asset_root_path'] = str(robot_root.resolve())
        local['urdf_path'] = str(robot_root / Path(local['urdf_path']).name)
        return robot_cfg_v2, None

    def _record_public_config(self, kind, config):
        if config.scene_collision_cfg is not None:
            raise RuntimeError("Scene collision configuration is forbidden")
        if kind.startswith("motion"):
            solvers = (config.ik_solver_config, config.trajopt_solver_config)
        else:
            solvers = (config,)
        if any(s.scene_collision_cfg is not None or not s.self_collision_check for s in solvers):
            raise RuntimeError("Empty world and robot self-collision must remain independent")
        if not hasattr(self, "public_contract"):
            self.public_contract = []
        self.public_contract.append({
            "kind": kind, "scene_collision_cfg_is_none": True,
            "solver_scene_collision_cfg_is_none": True,
            "self_collision_check": True,
        })

    def _build_motion_planner_cfg(self, max_batch_size):
        config = super()._build_motion_planner_cfg(max_batch_size)
        self._record_public_config("motion:" + str(max_batch_size), config)
        return config

    def _build_ik_cfg(self):
        config = super()._build_ik_cfg()
        self._record_public_config("ik", config)
        return config
