"""X32 scoped continuous IK. No other task's arm/solver is changed.

Pose residuals and physical joint margin are unchanged. A bounded local box
prevents a numerical branch jump before accepting a solution. Failure remains
unknown reachability, not a physical impossibility or collision certificate.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .kinematics import SerialArm
from .xylophone_routes import MAX_JOINT_STEP, JOINT_MARGIN


class ConnectedArm(SerialArm):
    def inverse(self, target, initial=None, restarts=0, collision_config=None):
        if initial is None:
            return super().inverse(target, initial, restarts, collision_config)
        seed = np.asarray(initial, float)
        original = super().inverse(target, seed, 0, collision_config)
        def admissible(f):
            q = np.asarray(f['joints'])
            return (f['position_error_m'] <= .0005 and f['orientation_error_rad'] <= .01
                    and np.max(np.abs(q-seed)) <= MAX_JOINT_STEP
                    and np.min(np.minimum(q-self.lower,self.upper-q)) >= JOINT_MARGIN)
        if admissible(original):
            return original
        lower = np.maximum(self.lower+JOINT_MARGIN+1e-7, seed-MAX_JOINT_STEP+1e-7)
        upper = np.minimum(self.upper-JOINT_MARGIN-1e-7, seed+MAX_JOINT_STEP-1e-7)
        if np.any(lower >= upper):
            return original
        def residual(q):
            frame = self.forward(q)
            return np.r_[10*(frame[:3,3]-target[:3,3]),
                Rotation.from_matrix(target[:3,:3].T@frame[:3,:3]).as_rotvec()]
        starts = [seed, seed+np.array([0,.04,-.04,.04,0,0]),
                  seed-np.array([0,.04,-.04,.04,0,0])]
        fits = []
        for start in starts:
            fit = least_squares(residual, np.clip(start,lower+1e-8,upper-1e-8),
                bounds=(lower,upper), max_nfev=90, ftol=1e-10, xtol=1e-10, gtol=1e-10)
            frame = self.forward(fit.x)
            result = dict(joints=fit.x.tolist(), position_error_m=float(np.linalg.norm(frame[:3,3]-target[:3,3])),
                orientation_error_rad=float(Rotation.from_matrix(target[:3,:3].T@frame[:3,:3]).magnitude()),
                collision_checked=False, attempted_starts=len(fits)+2, ik_branch_selection='X32_connected_box')
            fits.append(result)
            if admissible(result):
                return result
        # Always return the best CONNECTED result; a tiny residual on a remote
        # branch must not suppress a physically continuous approximate result.
        return min(fits,key=lambda f:10*f['position_error_m']+f['orientation_error_rad'])
