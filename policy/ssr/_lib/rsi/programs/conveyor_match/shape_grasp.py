from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ...planning import kinematics as K
from ...skills import grasp as G

PAD_X = 0.128, 0.158
PAD_HALF_Z = 0.008
GAP_OPEN = 0.088
WIDTH_MARGIN = 0.004
OPENINGS = 0.035, 0.05, 0.065, GAP_OPEN
OPEN_MARGIN = 0.012
MIN_WIDTH = 0.003
COLLIDE_MARGIN = 0.003
SUPPORT_CLEAR = 0.004


def grip_for_gap(gap: float) -> float:
    return float(np.clip((gap / 2.0 - 0.0004 + 0.01) / 0.054, 0.0, 1.0))


def _finger_q(grip: float) -> float:
    return float(np.clip(grip, 0.0, 1.0)) * 0.054 - 0.01


def gripper_boxes(grip: float = 1.0) -> np.ndarray:
    q = _finger_q(grip)
    b = [
        (0.0, 0.084, -0.031, 0.031, -0.03, 0.03),
        (0.009, 0.078, -0.087, -0.027, -0.012, 0.028),
        (0.009, 0.078, 0.027, 0.087, -0.012, 0.028),
        (0.047, 0.088, -0.045, 0.045, 0.0, 0.072),
    ]
    for s in (1.0, -1.0):
        y0, y1 = q + 0.007, q + 0.038
        b.append((0.072, 0.123, min(s * y0, s * y1), max(s * y0, s * y1), -0.031, 0.03))
        y0, y1 = q + 0.0004, q + 0.013
        b.append((0.103, 0.158, min(s * y0, s * y1), max(s * y0, s * y1), -0.018, 0.012))
    return np.asarray(b, dtype=float)


def _box_corners(boxes: np.ndarray) -> np.ndarray:
    c = []
    for x0, x1, y0, y1, z0, z1 in boxes:
        for x in (x0, x1):
            for y in (y0, y1):
                for z in (z0, z1):
                    c.append((x, y, z))
    return np.asarray(c)


def n_inside(p_l6: np.ndarray, boxes: np.ndarray, margin: float = COLLIDE_MARGIN) -> int:
    x, y, z = p_l6[:, 0, None], p_l6[:, 1, None], p_l6[:, 2, None]
    ins = (
        (x >= boxes[:, 0] - margin)
        & (x <= boxes[:, 1] + margin)
        & (y >= boxes[:, 2] - margin)
        & (y <= boxes[:, 3] + margin)
        & (z >= boxes[:, 4] - margin)
        & (z <= boxes[:, 5] + margin)
    )
    return int(ins.any(axis=1).sum())


MAX_SLANT = 0.45


def _antipodal(pad_pts: np.ndarray, x_ref: float, tz: float) -> bool:
    px, py, pz = pad_pts[:, 0], pad_pts[:, 1], pad_pts[:, 2]
    for coord, lo, hi in ((pz, tz - PAD_HALF_Z, tz + PAD_HALF_Z), (px, x_ref + PAD_X[0], x_ref + PAD_X[1])):
        edges = np.linspace(lo, hi, 4)
        span = (hi - lo) * 2.0 / 3.0
        for side in (1.0, -1.0):
            ext = []
            for a, b in zip(edges[:-1], edges[1:]):
                m = (coord >= a) & (coord <= b)
                if m.sum() >= 2:
                    ext.append(float((side * py[m]).max()))
            if len(ext) >= 2 and max(ext) - min(ext) > MAX_SLANT * span:
                return False
    return True


@dataclass
class ShapeGrasp:
    rot: np.ndarray
    tcp: np.ndarray
    heading: float
    tilt: float
    width: float
    approach: str
    score: float
    gap: float = GAP_OPEN
    robust: float = 0.0
    roll: float = 0.0


BLADE_X = 0.103, 0.158
BLADE_Z = -0.018, 0.012
BLADE_TOL = 0.003
ROBUST_SHIFT = 0.008
ROBUST_W = 0.2
NORMAL_MAX = math.radians(40.0)
NORMAL_FREE = math.radians(15.0)


def surface_normals(pts: np.ndarray, k: int = 12) -> np.ndarray:
    from scipy.spatial import cKDTree

    _, idx = cKDTree(pts).query(pts, k=min(k, len(pts)))
    nb = pts[idx] - pts[idx].mean(axis=1, keepdims=True)
    _, v = np.linalg.eigh(np.einsum("nki,nkj->nij", nb, nb))
    return v[:, :, 0]


def _contact_angle(p_l6: np.ndarray, n_l6: np.ndarray, gap: float) -> float:
    w = np.nonzero(_pad_window(p_l6) & (np.abs(p_l6[:, 1]) < gap / 2.0))[0]
    if len(w) < 4:
        return math.pi / 2
    order = np.argsort(p_l6[w, 1])
    k = max(2, min(6, len(w) // 4))
    ang = 0.0
    for sel in (w[order[:k]], w[order[-k:]]):
        ang = max(ang, math.acos(min(1.0, float(np.abs(n_l6[sel, 1]).mean()))))
    return ang


def _between_blades(p_l6: np.ndarray, gap: float) -> np.ndarray:
    return (
        (p_l6[:, 0] >= BLADE_X[0])
        & (p_l6[:, 0] <= BLADE_X[1])
        & (p_l6[:, 2] >= BLADE_Z[0])
        & (p_l6[:, 2] <= BLADE_Z[1])
        & (np.abs(p_l6[:, 1]) < gap / 2.0)
    )


def _pad_window(p_l6: np.ndarray) -> np.ndarray:
    return (p_l6[:, 0] >= PAD_X[0]) & (p_l6[:, 0] <= PAD_X[1]) & (np.abs(p_l6[:, 2]) <= PAD_HALF_Z)


def _first_contact_on_pads(p_l6: np.ndarray, gap: float, width: float) -> bool:
    b = _between_blades(p_l6, gap)
    if b.sum() < 5:
        return False
    yb = p_l6[b, 1]
    return float(yb.max() - yb.min()) <= width + 2 * BLADE_TOL


def _robustness(p_l6: np.ndarray, rot: np.ndarray, boxes: np.ndarray, gap: float, shift: float) -> float:
    ok = 0
    for d in ((shift, 0.0, 0.0), (-shift, 0.0, 0.0), (0.0, shift, 0.0), (0.0, -shift, 0.0)):
        p = p_l6 + rot.T @ np.asarray(d)
        if n_inside(p, boxes):
            continue
        w = _pad_window(p) & (np.abs(p[:, 1]) < gap / 2.0)
        if w.sum() < 5:
            continue
        width = float(p[w, 1].max() - p[w, 1].min())
        if width < MIN_WIDTH or not _first_contact_on_pads(p, gap, width):
            continue
        ok += 1
    return ok / 4.0


SWING_FREE = math.radians(20.0)
PARTIAL_MARGIN = 0.008


def _evaluate(
    pg,
    ng,
    rot,
    tx,
    ty,
    tz,
    width,
    gap,
    sel,
    boxes,
    corners,
    heading,
    tdeg,
    sgn,
    centre,
    antipodal_check,
    tcp_z_min,
    blade_check,
    approach_len,
    robust_shift,
    swing_w,
    roll=0.0,
):
    if antipodal_check and (not _antipodal(pg[sel], tx - K.TCP_OFFSET, tz)):
        return None
    tcp_g = np.array([tx, ty, tz])
    tcp = rot @ tcp_g
    if tcp_z_min is not None and tcp[2] < tcp_z_min:
        return None
    l6 = tcp - rot[:, 0] * K.TCP_OFFSET
    cz = (corners @ rot.T + l6)[:, 2]
    if cz.min() < SUPPORT_CLEAR:
        return None
    p_l6 = pg - (tcp_g - np.array([K.TCP_OFFSET, 0.0, 0.0]))
    if n_inside(p_l6, boxes):
        return None
    if blade_check and (not _first_contact_on_pads(p_l6, gap, width)):
        return None
    n_ang = 0.0
    if ng is not None:
        n_ang = _contact_angle(p_l6, ng, gap)
        if n_ang > NORMAL_MAX:
            return None
    appr = None
    for kind, d in (("up", rot.T @ np.array([0.0, 0.0, 1.0])), ("axis", np.array([-1.0, 0, 0]))):
        ok = True
        for s in np.linspace(0.015, approach_len, 5):
            if n_inside(p_l6 - d * s, boxes):
                ok = False
                break
        if ok:
            appr = kind
            break
    if appr is None:
        return None
    robust = _robustness(p_l6, rot, boxes, gap, robust_shift) if robust_shift else 0.0
    pad_c = tcp
    lever = float(np.linalg.norm(pad_c[:2] - centre[:2])) + 0.5 * abs(pad_c[2] - centre[2])
    clearance = gap - 2 * WIDTH_MARGIN - width
    score = (
        -math.radians(tdeg) * 0.02
        - 2.0 * lever
        + 0.002 * min(int(sel.sum()), 40)
        + 2.0 * min(0.02, clearance)
        + ROBUST_W * robust
        - 0.17 * max(0.0, n_ang - NORMAL_FREE)
    )
    if swing_w:
        score -= swing_w * max(0.0, swing_angle(rot, pad_c, centre) - SWING_FREE)
    return ShapeGrasp(rot, tcp, heading, sgn * math.radians(tdeg), width, appr, score, gap, robust, roll)


def swing_angle(rot: np.ndarray, pad: np.ndarray, com: np.ndarray) -> float:
    c = np.asarray(rot, dtype=float)[:, 1]
    c = c / max(1e-09, float(np.linalg.norm(c)))
    g = np.array([0.0, 0.0, -1.0])
    g = g - c * float(np.dot(g, c))
    if float(np.linalg.norm(g)) < 1e-06:
        return 0.0
    g /= float(np.linalg.norm(g))
    d = np.asarray(com, dtype=float) - np.asarray(pad, dtype=float)
    d = d - c * float(np.dot(d, c))
    down = float(np.dot(d, g))
    side = float(np.linalg.norm(d - g * down))
    return math.atan2(side, down)


def hang_drop(rot: np.ndarray, pad: np.ndarray, com: np.ndarray) -> float:
    c = np.asarray(rot, dtype=float)[:, 1]
    c = c / max(1e-09, float(np.linalg.norm(c)))
    g = np.array([0.0, 0.0, -1.0])
    g = g - c * float(np.dot(g, c))
    if float(np.linalg.norm(g)) < 1e-06:
        return 0.0
    g /= float(np.linalg.norm(g))
    d = np.asarray(com, dtype=float) - np.asarray(pad, dtype=float)
    d = d - c * float(np.dot(d, c))
    return float(np.linalg.norm(d) - np.dot(d, g))


def search(
    points: np.ndarray,
    to_base,
    *,
    headings: int = 12,
    tilts_deg=(0, 15, 30, 45, 60, 70, 75),
    step: float = 0.007,
    max_keep: int = 12,
    per_tilt: int = 2,
    approach_len: float = 0.07,
    symmetric: bool = False,
    blade_check: bool = True,
    robust_shift: float = ROBUST_SHIFT,
    centroid_lever: bool = True,
    normal_check: bool = True,
    antipodal_check: bool = True,
    swing_w: float = 0.0,
    tcp_z_min: float | None = None,
    partial: bool = False,
    rolls_deg=(0,),
    exclude=None,
    diverse_dz: float | None = None,
    heading_offset: float = 0.0,
) -> list[ShapeGrasp]:
    pts = np.asarray(points, dtype=float)
    if len(pts) > 1500:
        pts = pts[:: int(math.ceil(len(pts) / 1500))]
    to_base = np.asarray(to_base, dtype=float)[:2]
    to_base = to_base / max(1e-09, float(np.linalg.norm(to_base)))
    if centroid_lever:
        centre = pts.mean(axis=0)
    else:
        centre = np.array([0.0, 0.0, float(pts[:, 2].mean())])
    nrm = surface_normals(pts) if normal_check else None
    box_sets = {gap: gripper_boxes(grip_for_gap(gap)) for gap in OPENINGS}
    corners = _box_corners(box_sets[GAP_OPEN])
    out = []
    if symmetric:
        h0 = math.atan2(to_base[1], to_base[0]) + math.pi / 2
        heading_list = [h0 + k * math.pi / 2 for k in range(4)]
    else:
        heading_list = [heading_offset + math.pi * hi / headings for hi in range(2 * headings)]
    orientations = []
    for heading in heading_list:
        for tdeg in tilts_deg:
            for sgn in (1.0,) if tdeg == 0 else (1.0, -1.0):
                rot = G.grasp_rotation(heading, sgn * math.radians(tdeg))
                if tdeg and float(np.dot(-rot[:2, 0], to_base)) < 0.5:
                    continue
                orientations.append((heading, tdeg, sgn, 0.0, rot))
        for rdeg in rolls_deg:
            if not rdeg:
                continue
            for rs in (1.0, -1.0):
                r0 = G.grasp_rotation(heading, 0.0)
                rot = K.axis_angle(r0[:, 2], rs * math.radians(rdeg)) @ r0
                lean = -rot[:2, 0]
                if float(np.dot(lean, to_base)) < 0.7 * float(np.linalg.norm(lean)):
                    continue
                orientations.append((heading, 0, 1.0, rs * math.radians(rdeg), rot))
    for heading, tdeg, sgn, roll, rot in orientations:
        pg = pts @ rot
        ng = nrm @ rot if nrm is not None else None
        xs = np.arange(pg[:, 0].min() - 0.01, pg[:, 0].max() + 0.0101, step)
        zs = np.arange(pg[:, 2].min(), pg[:, 2].max() + 0.0001, step)
        for tx in xs:
            in_x = (pg[:, 0] >= tx + PAD_X[0] - K.TCP_OFFSET) & (pg[:, 0] <= tx + PAD_X[1] - K.TCP_OFFSET)
            if in_x.sum() < 5:
                continue
            for tz in zs:
                sel = in_x & (np.abs(pg[:, 2] - tz) <= PAD_HALF_Z)
                if sel.sum() < 5:
                    continue
                ys = pg[sel, 1]
                width = float(ys.max() - ys.min())
                pinches = []
                if MIN_WIDTH <= width <= GAP_OPEN - 2 * WIDTH_MARGIN:
                    gap = next((o for o in OPENINGS if o >= width + 2 * OPEN_MARGIN), GAP_OPEN)
                    pinches.append((float(ys.max() + ys.min()) / 2.0, width, gap, sel))
                elif partial and width > GAP_OPEN - 2 * WIDTH_MARGIN:
                    idx = np.nonzero(sel)[0]
                    seen = set()
                    for gap in OPENINGS[:-1]:
                        for ty0 in np.arange(ys.min(), ys.max() + 0.0001, 0.004):
                            m = np.abs(ys - ty0) < gap / 2.0
                            if m.sum() < 5:
                                continue
                            w_p = float(ys[m].max() - ys[m].min())
                            if w_p < MIN_WIDTH or w_p > gap - 2 * PARTIAL_MARGIN:
                                continue
                            ty_p = float(ys[m].max() + ys[m].min()) / 2.0
                            m = np.abs(ys - ty_p) < gap / 2.0
                            w_p = float(ys[m].max() - ys[m].min())
                            key = gap, round(ty_p / 0.004)
                            if w_p > gap - 2 * PARTIAL_MARGIN or key in seen:
                                continue
                            seen.add(key)
                            sub = np.zeros_like(sel)
                            sub[idx[m]] = True
                            pinches.append((ty_p, w_p, gap, sub))
                for ty, width, gap, sel in pinches:
                    g = _evaluate(
                        pg,
                        ng,
                        rot,
                        tx,
                        ty,
                        tz,
                        width,
                        gap,
                        sel,
                        box_sets[gap],
                        corners,
                        heading,
                        tdeg,
                        sgn,
                        centre,
                        antipodal_check,
                        tcp_z_min,
                        blade_check,
                        approach_len,
                        robust_shift,
                        swing_w,
                        roll,
                    )
                    if g is not None:
                        out.append(g)
    if exclude is not None:
        out = [g for g in out if not exclude(g)]
    out.sort(key=lambda g: -g.score)
    kept, per = [], {}
    for g in out:
        t = round(abs(g.tilt), 2), round(g.roll, 2)
        if per.get(t, 0) >= per_tilt:
            continue
        if all(
            abs(math.remainder(g.heading - k.heading, 2 * math.pi)) > 0.2
            or abs(g.tilt - k.tilt) > 0.2
            or abs(g.roll - k.roll) > 0.2
            or (np.linalg.norm(g.tcp - k.tcp) > 0.02)
            or (diverse_dz is not None and abs(float(g.tcp[2] - k.tcp[2])) >= diverse_dz)
            for k in kept
        ):
            kept.append(g)
            per[t] = per.get(t, 0) + 1
        if len(kept) >= max_keep:
            break
    kept.sort(key=lambda g: -g.score)
    return kept
