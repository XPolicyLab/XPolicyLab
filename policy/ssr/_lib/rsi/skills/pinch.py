from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..planning import kinematics as K

GAP_MAX = 0.088
FINGER_HALF_W = 0.011
TIP_BELOW = K.FINGERTIP_OFFSET - K.TCP_OFFSET
PAD_LEN = 0.03
PALM_ABOVE = 0.042
TIP_CLEAR = 0.004
TIP_CLEAR_THIN = 0.0005
THIN_TOP = 0.02
SIDE_MARGIN = 0.006
WIDER_TOL = 0.004
MIN_WIDTH = 0.004
YAW_ERR = math.radians(2.0)
CONTACT_CONE = math.radians(40.0)
CONTACT_PATCH = 0.003
CONE_PENALTY = 100.0
SLOPE_BAND = 0.008
HOUSING_X = 0.084
HOUSING_HALF_Y = 0.087
HOUSING_HALF_Z = 0.03
HOUSING_CLEAR = 0.008


@dataclass
class Pinch:
    heading: float
    tcp: np.ndarray
    width: float
    gap: float
    cost: float
    n_pad: int
    approach: np.ndarray = None
    closing: np.ndarray = None
    slope: float = 0.0
    part: bool = False
    nz: float = 0.0
    turns: bool = False
    contact_angle: float | None = None

    def __post_init__(self):
        if self.approach is None:
            self.approach = np.array([0.0, 0.0, -1.0])
        if self.closing is None:
            self.closing = np.array([math.cos(self.heading), math.sin(self.heading), 0.0])

    @property
    def tilt(self) -> float:
        return math.acos(float(np.clip(-self.approach[2], -1.0, 1.0)))


def _frame(th: float, tilt: float, lean):
    ax = np.array([math.cos(th), math.sin(th), 0.0])
    pe = np.array([-math.sin(th), math.cos(th), 0.0])
    if not tilt:
        return _DOWN.copy(), ax, pe
    ln = np.array([lean[0], lean[1], 0.0], dtype=float)
    ln /= max(np.linalg.norm(ln), 1e-09)
    n = np.cross(_DOWN, ln)
    Rt = K.axis_angle(n / np.linalg.norm(n), tilt)
    return Rt @ _DOWN, Rt @ ax, Rt @ pe


_DOWN = np.array([0.0, 0.0, -1.0])
PART_GAP = 0.016


def _column_parts(col: np.ndarray, s: np.ndarray):
    yield (col, None)
    idx = np.nonzero(col)[0]
    order = idx[np.argsort(s[idx])]
    ss = s[order]
    cuts = np.nonzero(np.diff(ss) >= PART_GAP)[0]
    if not len(cuts):
        return
    for i0, i1 in zip(np.concatenate([[0], cuts + 1]), np.concatenate([cuts, [len(ss) - 1]])):
        if i1 - i0 + 1 < 10:
            continue
        m = np.zeros_like(col)
        m[order[i0 : i1 + 1]] = True
        yield (m, (float(ss[i0]), float(ss[i1])))


def _squeeze_turns(pts, pad, ax, b, up, c3, t0, h_tip, hw, part, w_pad, turn, eps) -> bool:
    cx, cy = (float(v) for v in pts[pad, :2].mean(axis=0))
    for sg in (-1.0, 1.0):
        cs, sn = math.cos(sg * turn), math.sin(sg * turn)
        dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
        r = np.stack([cx + cs * dx - sn * dy, cy + sn * dx + cs * dy, pts[:, 2]], axis=1) - c3
        s2, t2, h2 = r @ ax, r @ b, r @ up
        m = (np.abs(t2 - t0) <= hw + 0.002) & (h2 >= h_tip) & (h2 <= h_tip + PAD_LEN)
        if part is not None:
            m &= (s2 >= part[0] - SIDE_MARGIN) & (s2 <= part[1] + SIDE_MARGIN)
        if m.sum() >= 3 and float(s2[m].max() - s2[m].min()) < w_pad - eps:
            return True
    return False


def candidates(
    pts: np.ndarray,
    z0: float = K.TABLE_Z,
    *,
    headings: int = 12,
    step: float = 0.008,
    max_cost: float | None = None,
    tilt: float = 0.0,
    lean=None,
    normals: np.ndarray | None = None,
    cone: float = CONTACT_CONE,
    margin: float | None = None,
    squeeze_turn: float | None = None,
    squeeze_eps: float = 0.002,
    squeeze_drop: bool = True,
    through=None,
) -> list[Pinch]:
    side_m = SIDE_MARGIN if margin is None else float(margin)
    c3 = np.array([*pts[:, :2].mean(axis=0), 0.0])
    _turns = [(math.cos(d), math.sin(d)) for d in (-YAW_ERR, YAW_ERR)]
    out: list[Pinch] = []
    rel3 = pts - c3
    turned = []
    for cd, sd in _turns:
        sw = rel3[:, 0] * cd - rel3[:, 1] * sd
        tw = rel3[:, 0] * sd + rel3[:, 1] * cd
        turned.append(np.stack([sw, tw, rel3[:, 2]], axis=1))
    hw = FINGER_HALF_W
    for k in range(headings):
        th = math.pi * k / headings
        a, ax, b = _frame(th, tilt, lean)
        up = -a
        s = rel3 @ ax
        t = rel3 @ b
        h = rel3 @ up
        axz, bz, uz = float(ax[2]), float(b[2]), float(up[2])
        tw2 = [(r2 @ ax, r2 @ b, r2 @ up) for r2 in turned]

        def h_for(s0: float, gap: float, t0: float, clr: float) -> float:
            off = s0 * axz - abs(axz) * 0.5 * gap + t0 * bz - abs(bz) * hw
            return (z0 + clr - off) / uz + TIP_BELOW

        t_lo, t_hi = t.min(), t.max()
        n_t = max(1, int(math.ceil((t_hi - t_lo) / step)))
        t_grid = np.linspace(t_lo, t_hi, n_t + 1)
        if through is not None:
            t_x = [float((np.asarray(q, dtype=float) - c3) @ b) for q in through]
            t_grid = np.concatenate([t_grid, [v for v in t_x if t_lo <= v <= t_hi]])
        for t0 in t_grid:
            col_all = np.abs(t - t0) <= hw + 0.002
            if col_all.sum() < 10:
                continue
            for col, part in _column_parts(col_all, s):
                in_part = None if part is None else (s >= part[0] - SIDE_MARGIN) & (s <= part[1] + SIDE_MARGIN)
                col_top = float(h[col].max())
                clr = TIP_CLEAR_THIN if float(pts[col, 2].max()) - z0 < THIN_TOP else TIP_CLEAR
                sc = s[col]
                s0 = 0.5 * (float(sc.max()) + float(sc.min()))
                gap = min(GAP_MAX, float(sc.max() - sc.min()) + 2 * side_m)
                h_tcp = max(h_for(s0, gap, t0, clr), col_top - PALM_ABOVE + 0.004)
                ok = False
                for _ in range(3):
                    h_tip = h_tcp - TIP_BELOW
                    blade = col & (h >= h_tip) & (h <= h_tcp + PALM_ABOVE)
                    pad = blade & (h <= h_tip + PAD_LEN)
                    if pad.sum() < 6:
                        break
                    sp, sb = s[pad], s[blade]
                    w_pad = float(sp.max() - sp.min())
                    w_blade = float(sb.max() - sb.min())
                    if w_pad < MIN_WIDTH or w_blade > w_pad + 2 * WIDER_TOL:
                        break
                    w_rob = w_blade
                    for s2, t2, h2 in tw2:
                        col2 = (np.abs(t2 - t0) <= hw + 0.002) & (h2 >= h_tip) & (h2 <= h_tcp + PALM_ABOVE)
                        if part is not None:
                            col2 &= (s2 >= part[0] - SIDE_MARGIN) & (s2 <= part[1] + SIDE_MARGIN)
                        if col2.any():
                            w_rob = max(w_rob, float(s2[col2].max() - s2[col2].min()))
                    gap = w_rob + 2 * side_m
                    if gap > GAP_MAX:
                        break
                    s0 = 0.5 * (float(sp.max()) + float(sp.min()))
                    need = h_for(s0, gap, t0, clr)
                    if need <= h_tcp + 0.0001:
                        ok = True
                        break
                    h_tcp = need
                if not ok:
                    continue
                palm = (np.abs(t - t0) <= 0.03) & (np.abs(s - s0) <= 0.045) & (h > h_tcp + PALM_ABOVE)
                if palm.any():
                    continue
                fing = col_all & (h >= h_tip) & (np.abs(s - s0) >= 0.5 * gap) & (np.abs(s - s0) <= 0.5 * gap + 0.01)
                if fing.any():
                    continue
                band = (h >= h_tip) & (h <= h_tip + PAD_LEN)
                if in_part is not None:
                    band &= in_part
                slope = 0.0
                for tb, sg in ((t0 - hw, -1.0), (t0 + hw, 1.0)):
                    d = (t - tb) * sg
                    inside = band & (d <= 0) & (d >= -SLOPE_BAND)
                    outside = band & (d > 0) & (d <= SLOPE_BAND)
                    if inside.sum() < 3 or outside.sum() < 3:
                        continue
                    grow = min(float((side * s[outside]).max() - (side * s[inside]).max()) for side in (1.0, -1.0))
                    slope = max(slope, grow / SLOPE_BAND)
                o6 = c3 + ax * s0 + b * t0 + up * h_tcp - a * K.TCP_OFFSET
                zc = o6[2] + min(0.0, HOUSING_X * a[2]) - HOUSING_HALF_Y * abs(ax[2]) - HOUSING_HALF_Z * abs(b[2])
                if zc < z0 + HOUSING_CLEAR:
                    continue
                antipodal = True
                nzs = []
                c_ang = None
                if normals is not None:
                    cos_min = math.cos(cone)
                    c_ang = 0.0
                    for side in (sp.max(), sp.min()):
                        patch = pad & (np.abs(s - side) <= CONTACT_PATCH)
                        nm = normals[patch].mean(axis=0)
                        nn = float(np.linalg.norm(nm))
                        if nn < 1e-06:
                            antipodal = False
                            c_ang = math.pi / 2
                            break
                        cs = abs(float(nm @ ax)) / nn
                        c_ang = max(c_ang, math.acos(min(1.0, cs)))
                        if cs < cos_min:
                            antipodal = False
                            break
                        nzs.append(float(nm[2]) / nn)
                p = c3 + ax * s0 + b * t0 + up * h_tcp
                dxy = math.hypot(*p[:2] - c3[:2])
                cost = dxy / 0.02 + 2.0 * max(0.0, gap - 0.07) / 0.018 - 0.002 * float(pad.sum())
                if not antipodal:
                    cost += CONE_PENALTY
                turns = bool(squeeze_turn) and _squeeze_turns(
                    pts, pad, ax, b, up, c3, t0, h_tip, hw, part, w_pad, squeeze_turn, squeeze_eps
                )
                if turns and squeeze_drop:
                    continue
                out.append(
                    Pinch(
                        th,
                        p,
                        w_pad,
                        gap,
                        cost,
                        int(pad.sum()),
                        a.copy(),
                        ax.copy(),
                        slope,
                        part is not None,
                        nz=float(np.mean(nzs)) if len(nzs) == 2 else 0.0,
                        turns=turns,
                        contact_angle=c_ang,
                    )
                )
    out.sort(key=lambda p: p.cost)
    if max_cost is not None:
        out = [p for p in out if p.cost <= max_cost]
    return out


def rotation(heading: float, arm: str, approach=None, closing=None) -> np.ndarray:
    want = -1.0 if arm == "left" else 1.0
    best = None
    if approach is None or closing is None or approach[2] < -0.9999:
        for h in (heading, heading + math.pi):
            R = K.top_down(h)
            score = R[0, 2] * want
            if best is None or score > best[0]:
                best = score, R
        return best[1]
    x = np.asarray(approach, dtype=float)
    x = x / np.linalg.norm(x)
    for sgn in (1.0, -1.0):
        y = sgn * np.asarray(closing, dtype=float)
        y = y - y @ x * x
        y /= np.linalg.norm(y)
        R = np.stack([x, y, np.cross(x, y)], axis=1)
        score = R[0, 2] * want
        if best is None or score > best[0]:
            best = score, R
    return best[1]


def grip_for_gap(gap: float) -> float:
    return float(np.clip((gap / 2.0 - 0.0004 + 0.01) / 0.054, 0.0, 1.0))
