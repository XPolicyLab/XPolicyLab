from copy import deepcopy
from pathlib import Path
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import DeviceCfg, GoalToolPose, JointState, Pose, ToolPoseCriteria
import curobo.runtime as curobo_runtime
import numpy as np
from scipy.spatial.transform import Rotation
import torch
import transforms3d as t3d
import yaml
curobo_runtime.cuda_graph_reset = True

def owned_configuration(filename):
    filename = Path(filename).resolve(strict=True)
    with filename.open() as stream:
        configuration = yaml.safe_load(stream)
    kinematics = configuration['robot_cfg']['kinematics']
    if kinematics['asset_root_path'] != 'owned_robot_model':
        raise ValueError('Only the frozen owned-robot asset root is permitted')
    urdf = filename.parent / kinematics['urdf_path']
    if urdf.is_symlink() or not urdf.is_file() or urdf.resolve().parent != filename.parent:
        raise ValueError('The robot description must be a contained frozen asset')
    kinematics['asset_root_path'] = str(filename.parent)
    kinematics['urdf_path'] = str(urdf)
    return configuration

from skills.shared.geometry.rigid_pose_transforms.implementation import (
    pose_to_matrix,
    matrix_to_pose,
    calculate_target_pose,
)



class FrozenSingleArmPlanner:

    def __init__(self, robot_origin_pose, active_joints_name, all_joints, dt=None, yml_path=None, table_height=None):
        if yml_path is not None:
            self.yml_path = yml_path
        else:
            raise ValueError('[Planner.py]: CuroboPlanner yml_path is None!')
        self.dt = dt
        self.robot_origin_pose = robot_origin_pose
        self.active_joints_name = active_joints_name
        self.all_joints = all_joints
        yml_data = owned_configuration(self.yml_path)
        self.frame_bias = yml_data['planner']['frame_bias']
        self.robot_cfg, self.scene_model = self._build_robot_and_scene_cfg(yml_data, 0.0 if table_height is None else table_height)
        if table_height is None:
            self.scene_model = {}
        cspace_cfg = self.robot_cfg['robot_cfg']['kinematics'].get('cspace', {})
        self.cspace_joint_names = list(cspace_cfg.get('joint_names', list(self.active_joints_name)))
        retract_config = cspace_cfg.get('default_joint_position', None)
        if retract_config is None:
            self.cspace_retract_config = np.zeros(len(self.cspace_joint_names), dtype=np.float32)
        else:
            self.cspace_retract_config = np.asarray(retract_config, dtype=np.float32).reshape(-1)
        if self.cspace_retract_config.shape[0] != len(self.cspace_joint_names):
            raise ValueError(f'curobo cspace config mismatch: {self.cspace_retract_config.shape[0]} retract values for {len(self.cspace_joint_names)} joints in {self.yml_path}')
        self._active_to_cspace_idx = []
        for joint_name in self.active_joints_name:
            if joint_name not in self.cspace_joint_names:
                raise ValueError(f'Active joint {joint_name} not found in curobo cspace joints {self.cspace_joint_names}')
            self._active_to_cspace_idx.append(self.cspace_joint_names.index(joint_name))
        self.device_cfg = DeviceCfg()
        self.use_cuda_graph = True
        self.use_graph_planner = False
        self.motion_planner = MotionPlanner(self._build_motion_planner_cfg(1))
        self.motion_planner.warmup(enable_graph=self.use_graph_planner, num_warmup_iterations=5)
        planner_joint_names = list(self.motion_planner.joint_names)
        if planner_joint_names != self.cspace_joint_names:
            new_retract = np.zeros(len(planner_joint_names), dtype=np.float32)
            for joint_index, name in enumerate(planner_joint_names):
                if name in self.cspace_joint_names:
                    new_retract[joint_index] = self.cspace_retract_config[self.cspace_joint_names.index(name)]
            self.cspace_joint_names = planner_joint_names
            self.cspace_retract_config = new_retract
            self._active_to_cspace_idx = [planner_joint_names.index(joint_name) for joint_name in self.active_joints_name]
        self.tool_frames = list(self.motion_planner.tool_frames)
        if not self.tool_frames:
            raise ValueError('No tool_frames found in curobo robot configuration.')
        self.ee_link = self.tool_frames[0]
        self.ik_solver = InverseKinematics(self._build_ik_cfg())
        self._default_tool_pose_criteria = {frame_name: ToolPoseCriteria(device_cfg=self.device_cfg) for frame_name in self.tool_frames}

    def _build_cspace_joint_values(self, active_joint_values):
        active_joint_values = self._extract_active_joint_values(active_joint_values)
        full_joint_values = self.cspace_retract_config.copy()
        for src_idx, dst_idx in enumerate(self._active_to_cspace_idx):
            full_joint_values[dst_idx] = active_joint_values[src_idx]
        return full_joint_values

    def _extract_active_joint_values(self, joint_values):
        joint_array = np.asarray(joint_values, dtype=np.float32).reshape(-1)
        if joint_array.shape[0] == len(self.active_joints_name):
            return joint_array
        if joint_array.shape[0] == len(self.all_joints):
            indices = [self.all_joints.index(joint_name) for joint_name in self.active_joints_name]
            return joint_array[indices]
        raise ValueError(f'Joint state size {joint_array.shape[0]} does not match active joints {len(self.active_joints_name)} or all joints {len(self.all_joints)} for planner.')

    def _build_joint_state(self, joint_pos):
        cspace_pos = self._build_cspace_joint_values(joint_pos)
        device = self.device_cfg.device
        joint_pos_tensor = torch.as_tensor(cspace_pos, dtype=torch.float32, device=device).reshape(1, -1)
        return JointState.from_position(joint_pos_tensor, joint_names=self.cspace_joint_names)

    def _build_goal_pose(self, target_pose_p, target_pose_q):
        device = self.device_cfg.device
        pos = torch.as_tensor(target_pose_p, dtype=torch.float32, device=device).reshape(1, 3)
        quat = torch.as_tensor(target_pose_q, dtype=torch.float32, device=device).reshape(1, 4)
        return GoalToolPose.from_poses({self.ee_link: Pose(position=pos, quaternion=quat)}, ordered_tool_frames=self.tool_frames, num_goalset=1)

    def plan_path(self, curr_joint_pos, target_ee_pose, real_robot_pose, constraint_pose=None):
        target_pose = deepcopy(target_ee_pose)
        target_pose = calculate_target_pose(real_robot_pose, self.robot_origin_pose, target_pose)
        world_base_pose = np.array(self.robot_origin_pose, dtype=np.float32)
        world_target_pose = np.array(target_pose, dtype=np.float32)
        target_pose_p, target_pose_q = self._trans_from_world_to_base(world_base_pose, world_target_pose)
        target_pose_p[0] += self.frame_bias[0]
        target_pose_p[1] += self.frame_bias[1]
        target_pose_p[2] += self.frame_bias[2]
        goal_tool_poses = self._build_goal_pose(target_pose_p, target_pose_q)
        start_joint_states = self._build_joint_state(curr_joint_pos)
        max_attempts = 5
        enable_graph_attempt = 1 if self.use_graph_planner else max_attempts
        result = self._run_with_constraint(self.motion_planner, constraint_pose, lambda: self.motion_planner.plan_pose(goal_tool_poses=goal_tool_poses, current_state=start_joint_states, max_attempts=max_attempts, enable_graph_attempt=enable_graph_attempt))
        res_result = dict()
        if result is None or not self._is_success(result.success):
            res_result['status'] = 'Fail'
            return res_result
        else:
            interpolated = result.get_interpolated_plan()
            jnames = getattr(interpolated, 'joint_names', None)
            res_result['status'] = 'Success'
            res_result['position'] = self._extract_trajectory(interpolated.position, batched=False, traj_joint_names=jnames)
            res_result['velocity'] = self._extract_trajectory(interpolated.velocity, batched=False, traj_joint_names=jnames)
            if res_result['velocity'] is None and res_result['position'] is not None:
                res_result['velocity'] = np.zeros_like(res_result['position'])
            return res_result

    def plan_joint(self, start_joint_pos, goal_joint_pos):
        start_joint_state = self._build_joint_state(start_joint_pos)
        goal_joint_state = self._build_joint_state(goal_joint_pos)
        max_attempts = 5
        result = self.motion_planner.plan_cspace(current_state=start_joint_state, goal_state=goal_joint_state, max_attempts=max_attempts, enable_graph_attempt=0)
        res_result = dict()
        if result is None or not self._is_success(result.success):
            res_result['status'] = 'Fail'
            return res_result
        interpolated = result.get_interpolated_plan()
        jnames = getattr(interpolated, 'joint_names', None)
        res_result['status'] = 'Success'
        res_result['position'] = self._extract_trajectory(interpolated.position, batched=False, traj_joint_names=jnames)
        res_result['velocity'] = self._extract_trajectory(interpolated.velocity, batched=False, traj_joint_names=jnames)
        res_result['acceleration'] = self._extract_trajectory(interpolated.acceleration, batched=False, traj_joint_names=jnames)
        if res_result['velocity'] is None and res_result['position'] is not None:
            res_result['velocity'] = np.zeros_like(res_result['position'])
        if res_result['acceleration'] is None and res_result['position'] is not None:
            res_result['acceleration'] = np.zeros_like(res_result['position'])
        return res_result

    def solve_ik_to_joint(self, curr_joint_pos, target_ee_pose, real_robot_pose, num_seeds: int=32):
        target_pose = deepcopy(target_ee_pose)
        target_pose = calculate_target_pose(real_robot_pose, self.robot_origin_pose, target_pose)
        world_base_pose = np.array(self.robot_origin_pose, dtype=np.float32)
        world_target_pose = np.array(target_pose, dtype=np.float32)
        target_pose_p, target_pose_q = self._trans_from_world_to_base(world_base_pose, world_target_pose)
        target_pose_p[0] += self.frame_bias[0]
        target_pose_p[1] += self.frame_bias[1]
        target_pose_p[2] += self.frame_bias[2]
        goal_tool_poses = self._build_goal_pose(target_pose_p, target_pose_q)
        current_state = self._build_joint_state(curr_joint_pos)
        ik_result = self.ik_solver.solve_pose(goal_tool_poses=goal_tool_poses, current_state=current_state, return_seeds=1)
        if ik_result.success is None or not self._is_success(ik_result.success):
            return {'status': 'Fail', 'joint_names': self.active_joints_name}
        joint_solution = np.asarray(ik_result.js_solution.position.detach().cpu())
        while joint_solution.ndim > 1:
            joint_solution = joint_solution[0]
        dof = joint_solution.shape[0]
        n_active = len(self.active_joints_name)
        n_cspace = len(self.cspace_joint_names)
        js_names = list(getattr(ik_result.js_solution, 'joint_names', None) or [])
        if len(js_names) == dof:
            try:
                indices = [js_names.index(joint_name) for joint_name in self.active_joints_name]
            except ValueError as exc:
                raise ValueError(f'IK solution joint_names {js_names} do not contain all active joints {self.active_joints_name}: {exc}') from None
            joint_solution = joint_solution[indices]
        elif dof == n_cspace:
            joint_solution = joint_solution[self._active_to_cspace_idx]
        elif dof != n_active:
            raise ValueError(f'Unexpected IK solution dimension from curobo: got {dof}, expected either {n_active} active joints or {n_cspace} cspace joints (and no usable joint_names on js_solution).')
        return {'status': 'Success', 'joint_names': self.active_joints_name, 'joint_value': joint_solution}

    def _trans_from_world_to_base(self, base_pose, target_pose):
        base_p, base_q = (base_pose[0:3], base_pose[3:])
        target_p, target_q = (target_pose[0:3], target_pose[3:])
        rel_p = target_p - base_p
        wRb = t3d.quaternions.quat2mat(base_q)
        wRt = t3d.quaternions.quat2mat(target_q)
        result_p = wRb.T @ rel_p
        result_q = t3d.quaternions.mat2quat(wRb.T @ wRt)
        return (result_p, result_q)

    def _build_robot_and_scene_cfg(self, yml_data, table_height):
        kinematics = deepcopy(yml_data.get('robot_cfg', {}).get('kinematics', {}))
        ee_link = kinematics.get('ee_link', None)
        tool_frames = kinematics.get('tool_frames', None)
        if tool_frames is None:
            if ee_link is None:
                raise ValueError(f'robot_cfg.kinematics in {self.yml_path} requires ee_link or tool_frames.')
            tool_frames = [ee_link]
        kinematics['tool_frames'] = list(tool_frames)
        kinematics['format_version'] = 2.0
        cspace = deepcopy(kinematics.get('cspace', {}))
        if 'default_joint_position' not in cspace and 'retract_config' in cspace:
            cspace['default_joint_position'] = cspace['retract_config']
        cspace.pop('retract_config', None)
        cspace_allowed = {'joint_names', 'default_joint_position', 'cspace_distance_weight', 'null_space_weight', 'null_space_maximum_distance', 'max_acceleration', 'max_jerk', 'velocity_scale', 'acceleration_scale', 'jerk_scale', 'position_limit_clip'}
        cspace = {key_name: value for key_name, value in cspace.items() if key_name in cspace_allowed}
        if 'joint_names' not in cspace:
            cspace['joint_names'] = list(self.active_joints_name)
        if 'default_joint_position' not in cspace:
            cspace['default_joint_position'] = [0.0 for _ in cspace['joint_names']]
        kinematics['cspace'] = cspace
        kinematics_allowed = {'format_version', 'base_link', 'tool_frames', 'collision_link_names', 'collision_spheres', 'collision_sphere_buffer', 'extra_collision_spheres', 'self_collision_ignore', 'self_collision_buffer', 'asset_root_path', 'mesh_link_names', 'lock_joints', 'extra_links', 'cspace', 'urdf_path', 'use_global_cumul', 'grasp_contact_link_names'}
        kinematics = {key_name: value for key_name, value in kinematics.items() if key_name in kinematics_allowed}
        robot_cfg_v2 = {'robot_cfg': {'kinematics': kinematics}, 'load_dynamics': bool(yml_data.get('load_dynamics', False))}
        scene_model = {'cuboid': {'table': {'dims': [3.0, 3.0, 0.05], 'pose': [0.0, 0.0, float(table_height) - 0.025, 1.0, 0.0, 0.0, 0.0]}}}
        return (robot_cfg_v2, scene_model)

    def _build_motion_planner_cfg(self, max_batch_size):
        cfg = MotionPlannerCfg.create(robot=self.robot_cfg, scene_model=self.scene_model, device_cfg=self.device_cfg, num_ik_seeds=32, num_trajopt_seeds=4, self_collision_check=True, use_cuda_graph=self.use_cuda_graph, max_batch_size=max_batch_size, multi_env=False, max_goalset=1)
        if self.dt is not None:
            cfg.trajopt_solver_config.interpolation_dt = float(self.dt)
        if not self.use_graph_planner:
            cfg.graph_planner_config = None
        return cfg

    def _build_ik_cfg(self):
        return InverseKinematicsCfg.create(robot=self.robot_cfg, scene_model=self.scene_model, device_cfg=self.device_cfg, num_seeds=32, seed_solver_num_seeds=32, position_tolerance=0.001, orientation_tolerance=0.02, self_collision_check=True, use_cuda_graph=self.use_cuda_graph, max_batch_size=1, multi_env=False, max_goalset=1)

    def _run_with_constraint(self, planner, constraint_pose, plan_fn):
        if constraint_pose is None:
            return plan_fn()
        hold_vec = np.asarray(constraint_pose, dtype=np.float32).reshape(-1)
        if hold_vec.shape[0] != 6:
            raise ValueError(f'constraint_pose should be 6D, got shape {hold_vec.shape}')
        planner_frames = list(planner.tool_frames)
        criteria = {frame_name: ToolPoseCriteria(device_cfg=self.device_cfg) for frame_name in planner_frames}
        criteria[self.ee_link] = ToolPoseCriteria(terminal_pose_axes_weight_factor=hold_vec.tolist(), non_terminal_pose_axes_weight_factor=hold_vec.tolist(), device_cfg=self.device_cfg)
        default_criteria = {frame_name: ToolPoseCriteria(device_cfg=self.device_cfg) for frame_name in planner_frames}
        planner.update_tool_pose_criteria(criteria)
        try:
            return plan_fn()
        finally:
            planner.update_tool_pose_criteria(default_criteria)

    def _is_success(self, success):
        if success is None:
            return False
        if isinstance(success, torch.Tensor):
            return bool(success.any().item())
        return bool(np.asarray(success).any())

    def _extract_trajectory(self, tensor, batched, traj_joint_names=None):
        if tensor is None:
            return None
        arr = np.asarray(tensor.detach().cpu())
        target_ndim = 3 if batched else 2
        while arr.ndim > target_ndim:
            arr = arr[:, 0] if batched else arr[0]
        if not batched and arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if batched and arr.ndim == 2:
            arr = arr.reshape(1, arr.shape[0], arr.shape[1])
        dof = arr.shape[-1]
        n_active = len(self.active_joints_name)
        n_cspace = len(self.cspace_joint_names)
        if dof == n_cspace:
            return arr[..., self._active_to_cspace_idx]
        if dof == n_active:
            return arr
        if traj_joint_names is not None and len(traj_joint_names) == dof:
            traj_names = list(traj_joint_names)
            try:
                indices = [traj_names.index(joint_name) for joint_name in self.active_joints_name]
                return arr[..., indices]
            except ValueError:
                raise ValueError(f'Trajectory joint_names {traj_names} do not contain all active joints {self.active_joints_name}') from None
