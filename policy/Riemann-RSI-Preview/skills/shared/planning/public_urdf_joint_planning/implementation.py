"""Numerical own-URDF planning; public frames, no simulator or measured jaw API.
Selected GO-3 PR214 mathematics retained; source identity in SKILL.md.
"""
import math
import numpy as np
from skills.shared.motion.urdf_pose_kinematics.implementation import OwnURDFKinematics
TCP_X = 0.13
MAX_LINE_JUMP = 0.25
STEP_M = .012
STEP_RAD = .06
_SEEDS = (np.array([0.0, 1.6, 1.0, -0.9, 0.0, 1.5]), np.array([0.0, 2.1, 1.6, -1.1, 0.0, 1.5]), np.array([0.0, 1.3, 1.3, -1.5, 0.0, 1.5]))

class PlanError(Exception):
    pass

def rot_y(a):
    (c, s) = (np.cos(a), np.sin(a))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def axis_angle(axis, angle):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)

def make_T(R=None, p=None):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if p is not None:
        T[:3, 3] = p
    return T

def rot_err(R_target, R_cur):
    Re = R_target @ R_cur.T
    c = np.clip((np.trace(Re) - 1) / 2, -1.0, 1.0)
    angle = np.arccos(c)
    if angle < 1e-09:
        return np.zeros(3)
    if np.pi - angle < 1e-06:
        (w, v) = np.linalg.eigh((Re + Re.T) / 2)
        axis = v[:, np.argmax(w)]
        return axis * angle
    axis = np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]]) / (2 * np.sin(angle))
    return axis * angle

def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi

def plan_joint(q_start, q_goal, max_dq=0.06):
    (q0, q1) = (np.asarray(q_start, float), np.asarray(q_goal, float))
    n = int(max(1, np.ceil(np.max(np.abs(q1 - q0)) / max_dq)))
    return [q0 + (q1 - q0) * (i / n) for i in range(1, n + 1)]

def approach_pose(p, azimuth, pitch):
    a = np.array([np.cos(pitch) * np.cos(azimuth), np.cos(pitch) * np.sin(azimuth), -np.sin(pitch)])
    j = np.array([-np.sin(azimuth), np.cos(azimuth), 0.0])
    return make_T(np.stack([a, j, np.cross(a, j)], axis=1), np.asarray(p, float))

class JointPlanner:

    def __init__(self, urdf, state):
        self.model = OwnURDFKinematics(urdf, preserve_declared_limits=True)
        self.origins = [j[0] for j in self.model.joints]
        self.axes = [j[1] for j in self.model.joints]
        self.limits = np.c_[self.model.lower_limits, self.model.upper_limits]
        self.bases = {a: self.model.observed_base(state[a + '_arm_joint_state'], state[a + '_ee_pose']) for a in ('left', 'right')}
        self.reach = sum((np.linalg.norm(j[0][:3, 3]) for j in self.model.joints)) + TCP_X

    def fk_chain(self, arm, q):
        T = self.bases[arm].copy()
        frames = []
        for i in range(6):
            T = T @ self.origins[i] @ make_T(axis_angle(self.axes[i], q[i]))
            frames.append(T)
        return frames

    def fk(self, arm, q):
        return self.fk_chain(arm, q)[-1]

    def tcp_from_link6(self, T6, offset=TCP_X):
        return T6 @ make_T(None, np.array([offset, 0.0, 0.0]))

    def link6_from_tcp(self, Ttcp, offset=TCP_X):
        return Ttcp @ make_T(None, np.array([-offset, 0.0, 0.0]))

    def jacobian(self, arm, q):
        frames = self.fk_chain(arm, q)
        p_end = frames[-1][:3, 3]
        J = np.zeros((6, 6))
        for (i, T) in enumerate(frames):
            z = T[:3, :3] @ self.axes[i]
            J[:3, i] = np.cross(z, p_end - T[:3, 3])
            J[3:, i] = z
        return (J, frames[-1])

    def ik(self, arm, T_target, q_seed, iters=150, tol_p=0.0005, tol_r=0.005, w_rot=0.35, limits=None):
        if limits is None:
            limits = self.limits
        q = np.array(q_seed, float)
        lam = 0.01
        best = (q.copy(), 1000000000.0, 1000000000.0)
        for _ in range(iters):
            (J, T) = self.jacobian(arm, q)
            ep = T_target[:3, 3] - T[:3, 3]
            er = rot_err(T_target[:3, :3], T[:3, :3])
            (np_, nr) = (np.linalg.norm(ep), np.linalg.norm(er))
            if np_ + w_rot * nr < best[1] + w_rot * best[2]:
                best = (q.copy(), np_, nr)
            if np_ < tol_p and nr < tol_r:
                break
            e = np.concatenate([ep, w_rot * er])
            Jw = J.copy()
            Jw[3:] *= w_rot
            dq = Jw.T @ np.linalg.solve(Jw @ Jw.T + lam * lam * np.eye(6), e)
            n = np.linalg.norm(dq)
            if n > 0.3:
                dq *= 0.3 / n
            q = np.clip(q + dq, limits[:, 0], limits[:, 1])
        return best

    def ik_multi(self, arm, T_target, q_seed, extra_seeds=(), **kw):
        sols = []
        for s in (q_seed, *extra_seeds):
            (q, ep, er) = self.ik(arm, T_target, s, **kw)
            sols.append((ep < 0.002 and er < 0.02, np.linalg.norm(wrap(q - q_seed)), q, ep, er))
        good = [s for s in sols if s[0]]
        pick = min(good, key=lambda s: s[1]) if good else min(sols, key=lambda s: s[3] + 0.3 * s[4])
        return (pick[2], pick[3], pick[4])

    def tcp(self, arm, q, offset=TCP_X):
        return self.tcp_from_link6(self.fk(arm, q), offset)

    def plan_line(self, arm, q_start, T_goal, step_m=STEP_M, step_rad=STEP_RAD, offset=TCP_X):
        T0 = self.tcp(arm, q_start, offset)
        (p0, p1) = (T0[:3, 3], T_goal[:3, 3])
        rv = rot_err(T_goal[:3, :3], T0[:3, :3])
        n = int(max(1, np.ceil(np.linalg.norm(p1 - p0) / step_m), np.ceil(np.linalg.norm(rv) / step_rad)))
        q = np.array(q_start, float)
        out = []
        for i in range(1, n + 1):
            t = i / n
            R = axis_angle(rv, np.linalg.norm(rv) * t) @ T0[:3, :3] if np.linalg.norm(rv) > 1e-09 else T0[:3, :3]
            T6 = self.link6_from_tcp(make_T(R, p0 + t * (p1 - p0)), offset)
            (qn, ep, er) = self.ik(arm, T6, q)
            if ep > 0.002 or er > 0.03:
                (qn, ep, er) = self.ik_multi(arm, T6, q, extra_seeds=_SEEDS)
                if ep > 0.002 or er > 0.03:
                    raise PlanError(f'{arm} line IK failed at {i}/{n}: ep={ep:.4f} er={er:.3f}')
            qn = q + wrap(qn - q)
            jump = float(np.max(np.abs(qn - q)))
            if jump > MAX_LINE_JUMP:
                raise PlanError(f'{arm} line IK jumped {jump:.2f} rad at {i}/{n}')
            q = qn
            out.append(q.copy())
        return out

    def solve(self, arm, T, q_ref):
        seeds = (np.asarray(q_ref, float),) + _SEEDS + (np.zeros(6),)
        best = None
        for s in seeds:
            (q, ep, er) = self.ik(arm, self.link6_from_tcp(T), s)
            if ep < 0.002 and er < 0.02:
                d = np.max(np.abs(wrap(q - q_ref)))
                if best is None or d < best[0]:
                    best = (d, q)
        if best is None:
            raise PlanError(f'{arm} pose unreachable at {np.round(T[:3, 3], 3)}')
        return q_ref + wrap(best[1] - q_ref)

    def link_frames(self, arm, q, grip=0.044):
        frames = self.fk_chain(arm, q)
        T6 = frames[-1]
        g = max(0.0, min(GRIP_MAX, grip))
        out = {'base_link': self.bases[arm]}
        out.update({f'link{i + 1}': T for (i, T) in enumerate(frames)})
        out['link7'] = T6 @ make_T(None, np.array([0.08657, 0.024896 + g, -0.0002436]))
        out['link8'] = T6 @ make_T(None, np.array([0.08657, -0.0249 - g, -0.00024366]))
        return out

    def link_box_corners(self, arm, q, grip=0.044, pad=0.006):
        out = {}
        for (name, T) in self.link_frames(arm, q, grip).items():
            (lo, hi) = (np.array(LINK_BOXES[name][0]) - pad, np.array(LINK_BOXES[name][1]) + pad)
            c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
            out[name] = c @ T[:3, :3].T + T[:3, 3]
        return out
GRIP_MAX = 0.044
LINK_BOXES = {'base_link': ((-0.03, -0.03, 0.0), (0.03, 0.03, 0.062)), 'link1': ((-0.022, -0.039, 0.0), (0.037, 0.039, 0.057)), 'link2': ((-0.292, -0.03, -0.028), (0.028, 0.036, 0.028)), 'link3': ((-0.018, -0.038, -0.085), (0.274, 0.038, 0.018)), 'link4': ((-0.017, -0.031, -0.086), (0.099, 0.031, 0.018)), 'link5': ((-0.031, -0.031, 0.002), (0.032, 0.031, 0.115)), 'link6': ((0.0, -0.087, -0.03), (0.084, 0.087, 0.03)), 'link7': ((-0.015, -0.024, -0.03), (0.071, 0.013, 0.031)), 'link8': ((-0.015, -0.013, -0.03), (0.071, 0.024, 0.031))}
