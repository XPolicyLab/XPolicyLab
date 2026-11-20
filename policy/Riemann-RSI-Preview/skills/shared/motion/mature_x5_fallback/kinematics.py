import math

import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


EEF_FROM_TCP = np.eye(4, dtype=np.float64)
EEF_FROM_TCP[:3, :3] = np.asarray([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
EEF_FROM_TCP[:3, 3] = [0.145, 0.0, 0.0]


class X5MeasuredFallbackKinematics:

    def __init__(self, urdf):
        self.model = pin.buildModelFromUrdf(str(urdf))
        self.data = self.model.createData()
        self.indices = [self.model.joints[self.model.getJointId(f'joint{index}')].idx_q for index in range(1, 9)]
        self.link6 = self.model.getFrameId('link6')
        self.lower = self.model.lowerPositionLimit[self.indices[:6]].copy()
        self.upper = self.model.upperPositionLimit[self.indices[:6]].copy()

    def _configuration(self, arm: np.ndarray) -> np.ndarray:
        value = np.zeros(self.model.nq)
        value[self.indices] = np.r_[np.asarray(arm, dtype=np.float64), 0.044, 0.044]
        return value

    def eef(self, arm: np.ndarray) -> np.ndarray:
        pin.forwardKinematics(self.model, self.data, self._configuration(arm))
        pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.link6].homogeneous.copy()

    def tcp(self, arm: np.ndarray) -> np.ndarray:
        return self.eef(arm) @ EEF_FROM_TCP

    def solve_tcp(self, target: np.ndarray, seed: np.ndarray, *, position_tolerance: float=0.003, rotation_tolerance: float=0.035, branch_jump_tolerance: float=0.3, rotation_weight: float=0.2, accept_best: bool=False, retry_relaxed: bool=True) -> np.ndarray:
        seed = np.asarray(seed, dtype=np.float64)
        initial = np.zeros_like(seed) if np.linalg.norm(seed) < 1e-07 else seed.copy()

        def residual(arm_joint_values):
            actual = self.tcp(arm_joint_values)
            rotation = Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).as_rotvec()
            return np.r_[actual[:3, 3] - target[:3, 3], float(rotation_weight) * rotation]
        fit = least_squares(residual, np.clip(initial, self.lower + 1e-07, self.upper - 1e-07), bounds=(self.lower, self.upper), max_nfev=90, ftol=None, xtol=1e-09, gtol=1e-08)
        solution = fit.x
        actual = self.tcp(solution)
        position_error = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
        rotation_error = float(np.linalg.norm(Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).as_rotvec()))
        if position_error > float(position_tolerance) or rotation_error > float(rotation_tolerance):
            if not accept_best:
                if retry_relaxed:
                    fit = least_squares(residual, np.clip(initial, self.lower + 1e-07, self.upper - 1e-07), bounds=(self.lower, self.upper), max_nfev=140, ftol=None, xtol=1e-09, gtol=1e-08)
                    solution = fit.x
                    actual = self.tcp(solution)
                    position_error = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
                    rotation_error = float(np.linalg.norm(Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).as_rotvec()))
                else:
                    raise RuntimeError(f'X5 measured fallback IK rejected: position={position_error:.5f} rotation={rotation_error:.5f}')
        branch_jump = float(np.max(np.abs(solution - seed)))
        if branch_jump > float(branch_jump_tolerance):
            raise RuntimeError(f'X5 measured fallback IK branch jump exceeds {float(branch_jump_tolerance):.3f} rad: {branch_jump:.5f}')
        return solution.astype(np.float32)
