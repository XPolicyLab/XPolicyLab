
from __future__ import annotations

import numpy as np

from ...lib.kinematics import chain as kin
from ...lib.geometry.transforms import make_T
from ...lib.geometry.transforms import TABLE_Z
from ...lib.geometry.camera import HeadCamera
from . import pb_kin as K

BOTTLE_H = 0.2184

_PROFILE = ((0.0, 0.0371), (0.018, 0.0371), (0.101, 0.0406), (0.102, 0.0406), (0.136, 0.0148), (BOTTLE_H, 0.0148))

def bottle_radius(s):
    hs = [p[0] for p in _PROFILE]
    rs = [p[1] for p in _PROFILE]
    return float(np.interp(s, hs, rs))

GRASP_GZ = 0.040

CUPS = {
    "mug": dict(rim=0.0762, opening=(-0.01431, 0.0), r_keep=0.05177, aim_back=0.020),
    "goblet": dict(rim=0.1003, opening=(0.0, 0.0), r_keep=0.04127, aim_back=0.020),
    "mug_fit": dict(rim=0.0823, opening=(0.0, 0.0), r_keep=0.05921, aim_back=0.010),
}

PHI1 = np.deg2rad(95.0)
PHI_P = np.deg2rad(150.0)

MOUTH_ABOVE_RIM = 0.035
KEEP_CLEAR = 0.012
KEEP_MARGIN = 0.012
STAGE_C_N = 7
AZ_STEP_DEG = 15

PRE_D = 0.085
LIFT_DZ = 0.20
GRASP_DAZ_DEG = (0, 15, -15, 30, -30, 45, -45, 60, -60, 75, -75, 90, -90)

def grip_R(a):
    a3 = np.array([a[0], a[1], 0.0])
    j = np.array([-a[1], a[0], 0.0])
    return np.stack([a3, j, np.cross(a3, j)], axis=1)

def rx(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])

def roll_pose(a, sgn, phi, tcp):
    return make_T(grip_R(a) @ rx(sgn * phi), np.asarray(tcp, float))

def bottle_axis(a, sgn, phi):
    return (grip_R(a) @ rx(sgn * phi))[:, 2]

def cup_kind(cup):
    if cup.get("model") is not None:
        return "mug_fit"
    return "goblet" if cup["height"] >= 0.1002 else "mug"

def opening_xy(cup_xy, kind):
    return np.asarray(cup_xy, float) + np.array(CUPS[kind]["opening"])

_RING = np.linspace(0.0, 2 * np.pi, 12, endpoint=False)
_SAMPLES = np.array([0.0, 0.03, 0.06, 0.09, 0.105, 0.115, 0.126, 0.15, 0.175, 0.20, BOTTLE_H])

def bottle_points(tcp, axis, gz=GRASP_GZ):
    axis = axis / np.linalg.norm(axis)
    ref = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(axis, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    out = []
    for s in _SAMPLES:
        c = tcp + (s - gz) * axis
        r = bottle_radius(s)
        out.append(c + r * (np.cos(_RING)[:, None] * e1 + np.sin(_RING)[:, None] * e2))
    return np.vstack(out)

def _over_cup(P, cup_xy, kind):
    k = CUPS[kind]
    d = np.hypot(P[:, 0] - cup_xy[0], P[:, 1] - cup_xy[1])
    return (d < k["r_keep"] + KEEP_MARGIN) & (P[:, 2] < TABLE_Z + k["rim"] + KEEP_CLEAR)

def bottle_clear(tcp, axis, cup_xy, kind):
    return not np.any(_over_cup(bottle_points(tcp, axis), cup_xy, kind))

def arm_clear(arm, q, cup_xy, kind):
    for name, c in kin.link_box_corners(arm, q, pad=0.0).items():
        if name in ("base_link", "link1"):
            continue
        if np.any(_over_cup(c, cup_xy, kind)):
            return False
    return True

def _box_samples(c):
    lo_hi = c.reshape(2, 2, 2, 3)
    pts = [c]
    for t in (0.5,):
        pts.append(((lo_hi[0] + lo_hi[1]) * t).reshape(-1, 3))
        pts.append(((lo_hi[:, 0] + lo_hi[:, 1]) * t).reshape(-1, 3))
        pts.append(((lo_hi[:, :, 0] + lo_hi[:, :, 1]) * t).reshape(-1, 3))
    return np.vstack(pts)

BOTTLE_KEEP_R = 0.050
BOTTLE_TOP_Z = TABLE_Z + BOTTLE_H + 0.012

def swept_clear(arm, qs, cup_xy, kind, bottle_xy=None, skip_fingers=False):
    k = CUPS[kind]
    for q in qs[::2] + qs[-1:]:
        for name, c in kin.link_box_corners(arm, q, pad=0.0).items():
            if name in ("base_link", "link1") or (skip_fingers and name in ("link7", "link8")):
                continue
            P = _box_samples(c)
            if np.any(_over_cup(P, cup_xy, kind)):
                return False
            if bottle_xy is not None:
                d = np.hypot(P[:, 0] - bottle_xy[0], P[:, 1] - bottle_xy[1])
                if np.any((d < BOTTLE_KEEP_R) & (P[:, 2] < BOTTLE_TOP_Z)):
                    return False
            if np.any(P[:, 2] < TABLE_Z + 0.004):
                return False
    return True

def _ik(arm, T, q_ref, margin=0.03):
    return K.solve_tcp(arm, T, q_ref, margin=margin)

def camera_azimuth(xy):
    return float(np.arctan2(xy[1] - HeadCamera.POS[1], xy[0] - HeadCamera.POS[0]))

def plan_grasp(arm, bottle_xy, cup_xy, kind, q_ref):
    az0 = camera_azimuth(bottle_xy)
    for daz in GRASP_DAZ_DEG:
        az = az0 + np.deg2rad(daz)
        a = np.array([np.cos(az), np.sin(az)])
        T = make_T(grip_R(a), np.array([bottle_xy[0], bottle_xy[1], TABLE_Z + GRASP_GZ]))
        pre = T.copy()
        pre[:3, 3] -= PRE_D * np.array([a[0], a[1], 0.0])
        lift = T.copy()
        lift[2, 3] += LIFT_DZ
        q_pre = _ik(arm, pre, q_ref)
        if q_pre is None:
            continue
        q_g = _ik(arm, T, q_pre)
        if q_g is None or np.max(np.abs(q_g - q_pre)) > 0.8:
            continue
        q_l = _ik(arm, lift, q_g)
        if q_l is None or np.max(np.abs(q_l - q_g)) > 0.8:
            continue
        if not (arm_clear(arm, q_pre, cup_xy, kind) and arm_clear(arm, q_g, cup_xy, kind)):
            continue
        return dict(a=a, daz=daz, pre=pre, grasp=T, lift=lift, q_pre=q_pre, q_grasp=q_g, q_lift=q_l)
    return None

def plan_pour(arm, cup_xy, kind, q_ref, phi1=PHI1, phip=PHI_P, aim_back=0.0):
    k = CUPS[kind]
    d = BOTTLE_H - GRASP_GZ
    zm = TABLE_Z + k["rim"] + MOUTH_ABOVE_RIM
    target = opening_xy(cup_xy, kind)
    best = None
    for az in np.deg2rad(np.arange(0, 360, AZ_STEP_DEG)):
        a = np.array([np.cos(az), np.sin(az)])
        for sgn in (1.0, -1.0):
            ax1 = bottle_axis(a, sgn, phi1)
            w = ax1[:2] / max(1e-9, float(np.linalg.norm(ax1[:2])))
            mouth = np.array([target[0] - aim_back * w[0], target[1] - aim_back * w[1], zm])
            P = mouth - d * ax1
            if P[2] - GRASP_GZ < TABLE_Z + 0.015:
                continue
            q_up = _ik(arm, roll_pose(a, sgn, 0.0, P), q_ref)
            if q_up is None:
                continue
            q6 = q_up[5] + sgn * phip
            if not -3.05 <= q6 <= 3.05:
                continue

            ok = arm_clear(arm, q_up, cup_xy, kind)
            for phi in np.linspace(0.0, phi1, 8):
                if not ok:
                    break
                ok = bottle_clear(P, bottle_axis(a, sgn, phi), cup_xy, kind)
            if not ok:
                continue
            qb = q_up.copy()
            qb[5] = q_up[5] + sgn * phi1
            if not arm_clear(arm, qb, cup_xy, kind):
                continue

            qs, Ts = [], []
            qc = qb
            for phi in np.linspace(phi1, phip, STAGE_C_N)[1:]:
                ax = bottle_axis(a, sgn, phi)
                tcp = mouth - d * ax
                T = roll_pose(a, sgn, phi, tcp)
                qn = _ik(arm, T, qc)
                if qn is None or np.max(np.abs(qn - qc)) > 0.6:
                    ok = False
                    break
                if not (arm_clear(arm, qn, cup_xy, kind) and bottle_clear(tcp, ax, cup_xy, kind)):
                    ok = False
                    break
                qs.append(qn)
                Ts.append(T)
                qc = qn
            if not ok:
                continue
            cost = float(np.max(np.abs(q_up - np.asarray(q_ref))))
            if best is None or cost < best["cost"]:
                best = dict(a=a, sgn=sgn, P=P, mouth=mouth, w=w, q_up=q_up, q_b=qb,
                            stage_c_q=qs, stage_c_T=Ts, cost=cost, az_deg=float(np.rad2deg(az)))
    return best
