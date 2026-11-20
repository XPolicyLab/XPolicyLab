from pathlib import Path
import json
import numpy as np
from scipy.spatial.transform import Rotation
from skills.shared.motion.measured_path_sampling.implementation import run as sample_path
from skills.shared.motion.urdf_pose_kinematics_observed_start.implementation import OwnURDFKinematics, pose_matrix
OPERATIONS = {'sampled_contact': 'contact', 'sampled_carry': 'carry', 'sampled_free': 'free', 'sampled_home': 'home', 'sampled_correction': 'contact_correction', 'sampled_retreat': 'free', 'sampled_view': 'view'}
CONSTRAINED = {'sampled_contact', 'sampled_carry', 'sampled_correction', 'sampled_retreat'}

class CheckedSampledMotion:

    def __init__(self, urdf):
        self.kinematics = OwnURDFKinematics(Path(urdf))
        self.checkers = {}

    def reset(self):
        self.checkers.clear()

    def validate(self, side, planner, positions):
        import torch
        from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
        if side not in self.checkers:
            configuration = RobotCollisionCheckerCfg.load_from_config(robot_config=planner.robot_cfg, device_cfg=planner.device_cfg, scene_collision_checker=planner.motion_planner.scene_collision_checker, collision_activation_distance=0.0, self_collision_activation_distance=0.0)
            self.checkers[side] = RobotCollisionChecker(configuration)
        full = np.stack([planner._build_cspace_joint_values(joints) for joints in positions])
        checked = torch.as_tensor(full, dtype=torch.float32, device=planner.device_cfg.device).unsqueeze(0)
        return bool(self.checkers[side].validate(checked).all().item())

    def plan(self, planner, payload):
        profile = OPERATIONS[payload['operation']]
        side = payload['side']
        current = np.asarray(payload['observation']['state'][side + '_arm_joint_state'], dtype=float)
        observed = payload['observation']['state'][side + '_ee_pose']
        base = self.kinematics.observed_base(current, observed)
        dense = None
        failure = {'status': 'Fail', 'positions': [], 'dense_waypoints': 0, 'official_waypoints': 0}
        if payload['operation'] == 'sampled_home':
            result = planner.plan_joint(current, np.asarray(payload['joint_target'], dtype=float).tolist())
            if result.get('status') == 'Success':
                dense = np.asarray(result['position'], dtype=float)
        else:
            solution = self.kinematics.solve(base, payload['pose'], current)
            if solution is None:
                return failure
            if payload['operation'] in CONSTRAINED:
                dense = self.kinematics.cartesian(base, current, payload['pose'], lambda positions: self.validate(side, planner, positions))
            if dense is None:
                result = planner.plan_joint(current, solution['joints'])
                if result.get('status') == 'Success':
                    dense = np.asarray(result['position'], dtype=float)
                else:
                    from skills.shared.motion.urdf_pose_kinematics_observed_start.implementation import matrix_pose
                    result = planner.plan_path(current, np.asarray(payload['pose']).tolist(), matrix_pose(base).tolist())
                    if result.get('status') == 'Success':
                        dense = np.asarray(result['position'], dtype=float)
        if dense is None:
            return failure
        sampled = sample_path(dense, current, profile, payload['remaining_actions'])
        if sampled['status'] != 'Success':
            return {'status': 'Budget', 'positions': [], 'dense_waypoints': len(dense), 'official_waypoints': 0}
        endpoint = base @ self.kinematics.forward(sampled['positions'][-1])
        requested = pose_matrix(payload['pose'])
        print('[RIEMANN SAMPLED OWN ENDPOINT] ' + json.dumps({'side': side, 'operation': payload['operation'], 'commanded_joints': sampled['positions'][-1].tolist(), 'requested_own_pose': np.asarray(payload['pose']).tolist(), 'own_fk_position_error_m': float(np.linalg.norm(endpoint[:3, 3] - requested[:3, 3])), 'own_fk_rotation_error_radians': float(Rotation.from_matrix(requested[:3, :3].T @ endpoint[:3, :3]).magnitude()), 'public_robot_inputs_only': True, 'trajectory_changed': False, 'is_contact_or_object_state_sensor': False}), flush=True)
        return {'status': 'Success', 'positions': sampled['positions'], 'dense_waypoints': len(dense), 'official_waypoints': len(sampled['positions'])}
