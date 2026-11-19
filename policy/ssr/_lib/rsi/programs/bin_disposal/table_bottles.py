from __future__ import annotations

from ... import ASSETS

import json
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from ...planning import kinematics as K

DATA = ASSETS / "bottle_profiles.json"
N_SEG = 10
N_ANG = 16
REGION = (-0.405, 0.58), (-0.37, 0.45)
MIN_AREA = 150


@dataclass
class BottleModel:
    name: str
    height: float
    rest_r: float
    z: np.ndarray
    r: np.ndarray

    @property
    def com(self) -> float:
        w = self.r**2
        return float(np.trapz(self.z * w, self.z) / np.trapz(w, self.z))


@dataclass
class Bottle:
    xy: np.ndarray
    standing: bool
    yaw: float
    model: str
    height: float
    radius: float
    iou: float
    extra: dict = field(default_factory=dict)


def _load_models() -> list[BottleModel]:
    raw = json.loads(DATA.read_text())
    out = []
    for name, m in sorted(raw.items()):
        prof = np.asarray(m["profile"], dtype=float)
        h = float(m["height"])
        rest = 0.5 * float(min(m["extents"][0], m["extents"][1]))
        scale = rest / max(1e-06, float(prof[:, 1].max()))
        zs = np.linspace(0.0, h, N_SEG + 1)
        rs = []
        for z in zs:
            sel = np.abs(prof[:, 0] - z) <= h / N_SEG * 0.75
            rs.append(prof[sel, 1].max() if sel.any() else np.interp(z, prof[:, 0], prof[:, 1]))
        out.append(BottleModel(name, h, rest, zs, np.asarray(rs) * scale))
    return out


MODELS = _load_models()
BY_NAME = {m.name: m for m in MODELS}


def _circle(n=N_ANG) -> np.ndarray:
    a = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    return np.stack([np.cos(a), np.sin(a)], axis=1)


_CIRCLE = _circle()


def model_points(m: BottleModel, standing: bool, x: float, y: float, yaw: float = 0.0) -> np.ndarray:
    c = _CIRCLE
    if standing:
        pts = np.empty((len(m.z), len(c), 3))
        pts[:, :, 0] = x + m.r[:, None] * c[None, :, 0]
        pts[:, :, 1] = y + m.r[:, None] * c[None, :, 1]
        pts[:, :, 2] = K.TABLE_Z + m.z[:, None]
        return pts
    d = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    n1 = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    n2 = np.array([0.0, 0.0, 1.0])
    mid = np.array([x, y, K.TABLE_Z + m.rest_r])
    s = m.z - m.height / 2
    centres = mid[None, :] + s[:, None] * d[None, :]
    return centres[:, None, :] + m.r[:, None, None] * (
        c[None, :, 0, None] * n1[None, None, :] + c[None, :, 1, None] * n2[None, None, :]
    )


def render(cam, pts: np.ndarray, box) -> np.ndarray:
    u0, v0, u1, v1 = box
    m = np.zeros((v1 - v0, u1 - u0), np.uint8)
    n_s, n_a, _ = pts.shape
    uv, d = cam.project(pts.reshape(-1, 3))
    if (d <= 0.01).any():
        return m
    uv = (uv - [u0, v0]).reshape(n_s, n_a, 2)
    for i in range(n_s - 1):
        seg = np.concatenate([uv[i], uv[i + 1]])
        hull = cv2.convexHull(np.round(seg * 4).astype(np.int32))
        cv2.fillConvexPoly(m, hull, 1, lineType=cv2.LINE_8, shift=2)
    return m


def _iou(a: np.ndarray, b: np.ndarray, bad: np.ndarray | None = None) -> float:
    if bad is not None:
        keep = bad == 0
        a, b = a & keep, b & keep
    inter = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    return inter / union if union else 0.0


def table_polygon_mask(cam, shape=(480, 640), region=REGION) -> np.ndarray:
    (x0, x1), (y0, y1) = region
    corners = np.array([[x0, y0, K.TABLE_Z], [x1, y0, K.TABLE_Z], [x1, y1, K.TABLE_Z], [x0, y1, K.TABLE_Z]])
    uv, _ = cam.project(corners)
    m = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(m, np.round(uv).astype(np.int32), 1)
    return m


def _is_table(
    lab: np.ndarray, st: dict, k_l: float = 3.5, k_h: float = 4.5, min_dh: float = 10.0, min_c: float = 6.0
) -> np.ndarray:
    L = lab[..., 0]
    a, b = lab[..., 1] - 128.0, lab[..., 2] - 128.0
    c = np.hypot(a, b)
    dh = (np.degrees(np.arctan2(b, a)) - st["h"][0] + 180.0) % 360.0 - 180.0
    return (
        (np.abs(L - st["L"][0]) <= k_l * st["L"][1])
        & (np.abs(dh) <= max(k_h * st["h"][1], min_dh))
        & (c >= max(min_c, st["C"][0] - k_l * st["C"][1]))
    )


def table_colour(
    lab: np.ndarray, valid: np.ndarray, ref: dict | None = None, min_ref_px: int = 500, **range_kw
) -> dict:
    if ref is not None:
        valid = valid & _is_table(lab, ref, **range_kw)
        if int(np.count_nonzero(valid)) < min_ref_px:
            return {k: tuple(v) for k, v in ref.items()}
    px = lab[valid].reshape(-1, 3)
    if len(px) > 80000:
        px = px[:: len(px) // 80000 + 1]
    L = px[:, 0]
    a, b = px[:, 1] - 128.0, px[:, 2] - 128.0
    c = np.hypot(a, b)
    h = np.degrees(np.arctan2(b, a))
    sel = c > 8.0
    hm = float(np.degrees(np.arctan2(np.median(np.sin(np.radians(h[sel]))), np.median(np.cos(np.radians(h[sel]))))))
    dh = (h[sel] - hm + 180.0) % 360.0 - 180.0
    hm += float(np.median(dh))
    dh = (h[sel] - hm + 180.0) % 360.0 - 180.0

    def med_sd(v):
        m = float(np.median(v))
        return m, float(1.4826 * np.median(np.abs(v - m))) + 0.001

    Lm, Ls = med_sd(L)
    cm, cs = med_sd(c)
    return {"L": (Lm, Ls), "C": (cm, cs), "h": (hm, float(1.4826 * np.median(np.abs(dh))) + 0.001)}


def foreground(
    img: np.ndarray,
    cam,
    bad: np.ndarray | None = None,
    region=REGION,
    k_l: float = 3.5,
    k_h: float = 4.5,
    min_dh: float = 10.0,
    min_c: float = 6.0,
    valid=None,
    table_ref: dict | None = None,
) -> tuple[np.ndarray, dict]:
    lab = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2LAB).astype(np.float32)
    if valid is None:
        valid = table_polygon_mask(cam, img.shape[:2], region) > 0
    valid = valid.copy()
    if bad is not None:
        valid &= bad == 0
    rk = dict(k_l=k_l, k_h=k_h, min_dh=min_dh, min_c=min_c)
    st = table_colour(lab, valid) if table_ref is None else table_colour(lab, valid, ref=table_ref, **rk)
    table = _is_table(lab, st, **rk)
    fg = valid & ~table
    fg = fg.astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return fg, st


def _fit_region(cam, comp: np.ndarray, box, bad=None, models=None, n_refine: int = 4) -> Bottle | None:
    models = MODELS if models is None else models
    u0, v0, _, _ = box
    vs, us = np.nonzero(comp)
    if len(us) == 0:
        return None
    low = vs >= np.percentile(vs, 97)
    p_low = cam.on_plane_z(np.array([[us[low].mean() + u0, vs[low].max() + v0]]), K.TABLE_Z)[0]
    P = cam.on_plane_z(np.stack([us + u0, vs + v0], axis=1).astype(float), K.TABLE_Z + 0.03)[:, :2]
    P = P[np.isfinite(P).all(axis=1)]
    c = P.mean(axis=0)
    _, v = np.linalg.eigh(np.cov((P - c).T))
    yaw0 = math.atan2(v[1, 1], v[0, 1])

    def score(m, standing, x, y, yaw):
        return _iou(render(cam, model_points(m, standing, x, y, yaw), box), comp, bad)

    tops = []
    for m in models:
        away = p_low[:2] - cam.pos[:2]
        away = away / max(1e-09, np.linalg.norm(away))
        xs0 = p_low[:2] + away * m.r[0]
        cands = [
            (score(m, True, xs0[0] + dx, xs0[1] + dy, 0.0), True, xs0[0] + dx, xs0[1] + dy, 0.0)
            for dx in np.arange(-0.02, 0.0201, 0.01)
            for dy in np.arange(-0.02, 0.0201, 0.01)
        ]
        tops.append((max(cands, key=lambda t: t[0]), m))
        cands = []
        for yaw in (yaw0, yaw0 + math.pi):
            for dyaw in np.radians([-8, 0, 8]):
                for dx in np.arange(-0.015, 0.0151, 0.015):
                    for dy in np.arange(-0.015, 0.0151, 0.015):
                        cands.append(
                            (score(m, False, c[0] + dx, c[1] + dy, yaw + dyaw), False, c[0] + dx, c[1] + dy, yaw + dyaw)
                        )
        tops.append((max(cands, key=lambda t: t[0]), m))
    tops.sort(key=lambda t: -t[0][0])
    sel = tops[:n_refine]
    for kind in (True, False):
        if not any(t[0][1] == kind for t in sel):
            best_k = next((t for t in tops if t[0][1] == kind), None)
            if best_k is not None:
                sel.append(best_k)
    fits = []
    for (iou, standing, x, y, yaw), m in sel:
        iou, x, y, yaw = _refine(score, m, standing, iou, x, y, yaw)
        fits.append((iou, standing, x, y, yaw, m))
    fits.sort(key=lambda t: -t[0])
    iou, standing, x, y, yaw, m = fits[0]
    alt = next((f for f in fits if f[1] != standing), None)
    extra = {"rest_r": m.rest_r}
    if alt is not None:
        extra["alt"] = {"standing": alt[1], "xy": [alt[2], alt[3]], "yaw": alt[4], "model": alt[5].name, "iou": alt[0]}
    return Bottle(
        np.array([x, y]),
        standing,
        0.0 if standing else math.remainder(yaw, 2 * math.pi),
        m.name,
        m.height,
        float(m.r.max()),
        float(iou),
        extra=extra,
    )


def _refine(score, m, standing, iou, x, y, yaw):
    for step, astep in ((0.005, math.radians(4)), (0.002, math.radians(2)), (0.001, math.radians(1))):
        improved = True
        while improved:
            improved = False
            for dx, dy, da in ((step, 0, 0), (-step, 0, 0), (0, step, 0), (0, -step, 0), (0, 0, astep), (0, 0, -astep)):
                if standing and da:
                    continue
                s = score(m, standing, x + dx, y + dy, yaw + da)
                if s > iou + 0.0001:
                    iou, x, y, yaw = s, x + dx, y + dy, yaw + da
                    improved = True
    return iou, x, y, yaw


def detect(
    img: np.ndarray,
    cam,
    bad: np.ndarray | None = None,
    region=REGION,
    min_area: int = MIN_AREA,
    max_regions: int = 8,
    pad: int = 12,
) -> tuple[list[Bottle], np.ndarray]:
    fg, _ = foreground(img, cam, bad, region)
    invalid = table_polygon_mask(cam, img.shape[:2], region) == 0
    if bad is not None:
        invalid |= bad > 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    order = sorted(range(1, n), key=lambda i: -stats[i, cv2.CC_STAT_AREA])
    out = []
    H, W = fg.shape
    for i in order[:max_regions]:
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            break
        x, y, w, h = stats[i, :4]
        box = max(0, x - pad), max(0, y - pad), min(W, x + w + pad), min(H, y + h + pad)
        comp = (lab[box[1] : box[3], box[0] : box[2]] == i).astype(np.uint8)
        badc = invalid[box[1] : box[3], box[0] : box[2]].astype(np.uint8)
        b = _fit_region(cam, comp, box, badc)
        if b is not None:
            b.extra.update(area=int(stats[i, cv2.CC_STAT_AREA]), box=[int(v) for v in box])
            out.append(b)
    return out, fg


def detect_instances(
    img: np.ndarray, cam, masks, bad: np.ndarray | None = None, min_area: int = MIN_AREA, pad: int = 12
) -> list[Bottle]:
    H, W = img.shape[:2]
    masks = [np.asarray(m, bool) for m in masks]
    union = np.zeros((H, W), np.int32)
    for m in masks:
        union += m
    out = []
    for m in masks:
        vs, us = np.nonzero(m)
        if len(us) < min_area:
            continue
        box = (
            max(0, int(us.min()) - pad),
            max(0, int(vs.min()) - pad),
            min(W, int(us.max()) + 1 + pad),
            min(H, int(vs.max()) + 1 + pad),
        )
        sl = slice(box[1], box[3]), slice(box[0], box[2])
        comp = m[sl].astype(np.uint8)
        badc = union[sl] - m[sl] > 0
        if bad is not None:
            badc |= bad[sl] > 0
        b = _fit_region(cam, comp, box, badc.astype(np.uint8))
        if b is not None:
            b.extra.update(area=int(len(us)), box=[int(v) for v in box])
            out.append(b)
    return out


def _footprint_dist(p, o: "Bottle") -> float:
    if o.standing:
        return float(np.linalg.norm(p - o.xy)) - o.radius
    a = np.array([math.cos(o.yaw), math.sin(o.yaw)])
    s = float(np.clip((p - o.xy) @ a, -o.height / 2, o.height / 2))
    return float(np.linalg.norm(p - (o.xy + a * s))) - o.radius


def detect_with_instances(
    img: np.ndarray,
    cam,
    instances,
    bad: np.ndarray | None = None,
    region=REGION,
    min_score: float = 0.15,
    min_area: int = MIN_AREA,
    max_robot: float = 0.4,
    min_support: float = 0.3,
    min_fit_iou: float = 0.6,
    max_regions: int = 8,
) -> tuple[list[Bottle], np.ndarray]:
    fg, _ = foreground(img, cam, bad, region)
    H, W = fg.shape
    table = table_polygon_mask(cam, (H, W), region) > 0
    robot = np.zeros((H, W), bool) if bad is None else bad > 0
    fgb = fg > 0
    kept = []
    for inst in sorted(instances, key=lambda i: -i.score):
        if inst.score < min_score:
            continue
        m = inst.mask
        n = int(m.sum())
        if n < min_area or n > 0.08 * H * W:
            continue
        n_tab = int((m & table & ~robot).sum())
        if (m & robot).sum() > max_robot * n or (m & table).sum() < 0.3 * n or (m & fgb).sum() < min_support * n_tab:
            continue
        if any((m & k).sum() > 0.5 * min(n, int(k.sum())) for k in kept):
            continue
        kept.append(m)
    out = [b for b in detect_instances(img, cam, kept, bad, min_area) if b.iou >= min_fit_iou]
    for b in out:
        b.extra["source"] = "instance"
    covered = np.zeros((H, W), np.uint8)
    for m in kept:
        covered[m] = 1
    covered = cv2.dilate(covered, np.ones((7, 7), np.uint8)) > 0
    rest = (fgb & ~covered).astype(np.uint8)
    rest = cv2.morphologyEx(rest, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    invalid = ~table | robot | covered
    n, lab, stats, _ = cv2.connectedComponentsWithStats(rest, connectivity=8)
    order = sorted(range(1, n), key=lambda i: -stats[i, cv2.CC_STAT_AREA])
    for i in order[:max_regions]:
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            break
        x, y, w, h = stats[i, :4]
        box = max(0, x - 12), max(0, y - 12), min(W, x + w + 12), min(H, y + h + 12)
        comp = (lab[box[1] : box[3], box[0] : box[2]] == i).astype(np.uint8)
        badc = invalid[box[1] : box[3], box[0] : box[2]].astype(np.uint8)
        b = _fit_region(cam, comp, box, badc)
        if b is None or any(_footprint_dist(b.xy, o) < b.radius + 0.02 for o in out):
            continue
        b.extra.update(area=int(stats[i, cv2.CC_STAT_AREA]), box=[int(v) for v in box], source="colour")
        out.append(b)
    for m in kept:
        fgb = fgb | m
    return out, fgb.astype(np.uint8)


TABLE_RECT = (-0.41, 0.59), (-0.6, 0.5)


def table_behind(cam, shape=(480, 640), rect=TABLE_RECT, margin: float = 0.02) -> np.ndarray:
    h, w = shape
    vs, us = np.mgrid[0:h, 0:w]
    P = cam.on_plane_z(np.stack([us.ravel(), vs.ravel()], axis=1).astype(float), K.TABLE_Z)
    (x0, x1), (y0, y1) = rect
    ok = (
        np.isfinite(P[:, 0])
        & (P[:, 0] > x0 + margin)
        & (P[:, 0] < x1 - margin)
        & (P[:, 1] > y0 + margin)
        & (P[:, 1] < y1 - margin)
    )
    return ok.reshape(h, w)


def hypothesis_mask(cam, model: str, standing: bool, x: float, y: float, yaw: float, shape=(480, 640)) -> np.ndarray:
    pts = model_points(BY_NAME[model], standing, x, y, yaw)
    _, d = cam.project(pts.reshape(-1, 3))
    if (d <= 0.02).any():
        return np.zeros(shape, bool)
    return render(cam, pts, (0, 0, shape[1], shape[0])) > 0


def side_evidence(
    img: np.ndarray,
    cam,
    b: Bottle,
    others=(),
    bad=None,
    min_px: int = 40,
    contest_margin_px: int = 0,
    table_ref: dict | None = None,
) -> dict | None:
    alt = b.extra.get("alt")
    if alt is None:
        return None
    hs = [
        (b.model, b.standing, b.xy[0], b.xy[1], b.yaw),
        (alt["model"], alt["standing"], alt["xy"][0], alt["xy"][1], alt["yaw"]),
    ]
    shape = img.shape[:2]
    m0 = hypothesis_mask(cam, *hs[0], shape=shape)
    m1 = hypothesis_mask(cam, *hs[1], shape=shape)
    valid = table_behind(cam, shape)
    if bad is not None:
        valid &= bad == 0
    fg, _ = foreground(img, cam, valid=valid, table_ref=table_ref)
    contested = np.zeros(shape, bool)
    for o in others:
        if o is b:
            continue
        contested |= hypothesis_mask(cam, o.model, o.standing, o.xy[0], o.xy[1], o.yaw, shape)
        oa = o.extra.get("alt")
        if oa is not None:
            contested |= hypothesis_mask(cam, oa["model"], oa["standing"], oa["xy"][0], oa["xy"][1], oa["yaw"], shape)
    if contest_margin_px > 0 and contested.any():
        import cv2

        k = 2 * int(contest_margin_px) + 1
        contested = cv2.dilate(contested.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    keep = valid & ~contested
    only0 = m0 & ~m1 & keep
    only1 = m1 & ~m0 & keep
    res = {}
    for m, h in ((only0, hs[0]), (only1, hs[1])):
        key = "stand" if h[1] else "lie"
        n = int(m.sum())
        res["n_" + key] = n
        res[key] = float(fg[m].mean()) if n >= min_px else None
    return res


def decide_standing(b: Bottle, evidences, margin: float = 0.25) -> tuple[bool, float]:
    scores = []
    for e in evidences:
        if not e:
            continue
        fs, fl = e.get("stand"), e.get("lie")
        if fs is not None and fl is not None:
            scores.append(fs - fl)
        elif fs is not None:
            scores.append(2.0 * (fs - 0.5))
        elif fl is not None:
            scores.append(2.0 * (0.5 - fl))
    if not scores:
        return b.standing, 0.0
    sc = float(np.mean(scores))
    if sc > margin:
        return True, sc
    if sc < -margin:
        return False, sc
    return b.standing, sc


def swap_to_alt(b: Bottle) -> Bottle:
    a = b.extra["alt"]
    m = BY_NAME[a["model"]]
    extra = {
        "rest_r": m.rest_r,
        "alt": {"standing": b.standing, "xy": list(b.xy), "yaw": b.yaw, "model": b.model, "iou": b.iou},
    }
    for k in ("area", "box"):
        if k in b.extra:
            extra[k] = b.extra[k]
    return Bottle(
        np.asarray(a["xy"], float),
        bool(a["standing"]),
        0.0 if a["standing"] else float(a["yaw"]),
        m.name,
        m.height,
        float(m.r.max()),
        float(a["iou"]),
        extra=extra,
    )


def _close_view(
    img: np.ndarray, cam, pts0: np.ndarray, bad, others, max_shift: float, excl_px: int, table_ref: dict | None = None
):
    h, w = img.shape[:2]
    _, dep = cam.project(pts0.reshape(-1, 3))
    if (dep <= 0.02).any():
        return None
    m0 = render(cam, pts0, (0, 0, w, h))
    if int(m0.sum()) < 200:
        return None
    grow = int(cam.f * max_shift / max(0.05, float(np.median(dep)))) + 6
    win = cv2.dilate(m0, np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)) > 0
    vs, us = np.nonzero(win)
    box = int(us.min()), int(vs.min()), int(us.max()) + 1, int(vs.max()) + 1
    sl = slice(box[1], box[3]), slice(box[0], box[2])
    valid = table_behind(cam, (h, w))
    if bad is not None:
        valid &= bad == 0
    excl = ~valid
    for o in others:
        om = hypothesis_mask(cam, o.model, o.standing, float(o.xy[0]), float(o.xy[1]), o.yaw, (h, w))
        if om.any():
            excl |= cv2.dilate(om.astype(np.uint8), np.ones((2 * excl_px + 1, 2 * excl_px + 1), np.uint8)) > 0
    if int((valid & win).sum()) < 2000:
        return None
    fg, _ = foreground(img, cam, valid=valid, table_ref=table_ref)
    keep = win[sl] & ~excl[sl]
    F = (fg[sl] > 0) & keep
    nf = int(F.sum())
    if nf < 200:
        return None

    def score(pts) -> float:
        mk = (render(cam, pts, box) > 0) & keep
        inter = int(np.count_nonzero(mk & F))
        union = int(np.count_nonzero(mk | F))
        return inter / union if union else 0.0

    return score, nf


def refine_lying(
    img: np.ndarray,
    cam,
    b: Bottle,
    bad: np.ndarray | None = None,
    others=(),
    max_shift: float = 0.03,
    step: float = 0.01,
    dyaws=(-0.1, 0.0, 0.1),
    excl_px: int = 8,
    table_ref: dict | None = None,
) -> dict | None:
    m = BY_NAME[b.model]
    d = np.array([math.cos(b.yaw), math.sin(b.yaw)])
    nrm = np.array([-math.sin(b.yaw), math.cos(b.yaw)])

    def pts_at(lat, ax, dyaw):
        p = np.asarray(b.xy, float) + lat * nrm + ax * d
        return model_points(m, False, float(p[0]), float(p[1]), b.yaw + dyaw)

    cv = _close_view(img, cam, pts_at(0.0, 0.0, 0.0), bad, others, max_shift, excl_px, table_ref=table_ref)
    if cv is None:
        return None
    sc, nf = cv

    def score(lat, ax, dyaw):
        return sc(pts_at(lat, ax, dyaw))

    iou0 = score(0.0, 0.0, 0.0)
    grid = np.arange(-max_shift, max_shift + 1e-09, step)
    best = iou0, 0.0, 0.0, 0.0
    for dyaw in dyaws:
        for lat in grid:
            for ax in grid:
                s = score(lat, ax, dyaw)
                if s > best[0]:
                    best = s, float(lat), float(ax), float(dyaw)
    dy_step = (dyaws[1] - dyaws[0]) / 2 if len(dyaws) > 1 else 0.0
    _, l0, a0, y0 = best
    for dyaw in (y0 - dy_step, y0, y0 + dy_step):
        for lat in (l0 - step / 2, l0, l0 + step / 2):
            for ax in (a0 - step / 2, a0, a0 + step / 2):
                if abs(lat) > max_shift + 1e-09 or abs(ax) > max_shift + 1e-09:
                    continue
                s = score(lat, ax, dyaw)
                if s > best[0]:
                    best = s, float(lat), float(ax), float(dyaw)
    return {"lat": best[1], "ax": best[2], "dyaw": best[3], "iou0": iou0, "iou": best[0], "n": nf}


def refine_standing(
    img: np.ndarray,
    cam,
    b: Bottle,
    bad: np.ndarray | None = None,
    others=(),
    max_shift: float = 0.03,
    step: float = 0.01,
    excl_px: int = 8,
    table_ref: dict | None = None,
) -> dict | None:
    m = BY_NAME[b.model]
    x0, y0 = float(b.xy[0]), float(b.xy[1])

    def pts_at(dx, dy):
        return model_points(m, True, x0 + dx, y0 + dy)

    cv = _close_view(img, cam, pts_at(0.0, 0.0), bad, others, max_shift, excl_px, table_ref=table_ref)
    if cv is None:
        return None
    sc, nf = cv
    iou0 = sc(pts_at(0.0, 0.0))
    grid = np.arange(-max_shift, max_shift + 1e-09, step)
    best = iou0, 0.0, 0.0
    for dx in grid:
        for dy in grid:
            s = sc(pts_at(dx, dy))
            if s > best[0]:
                best = s, float(dx), float(dy)
    _, bx, by = best
    for dx in (bx - step / 2, bx, bx + step / 2):
        for dy in (by - step / 2, by, by + step / 2):
            if abs(dx) > max_shift + 1e-09 or abs(dy) > max_shift + 1e-09:
                continue
            s = sc(pts_at(dx, dy))
            if s > best[0]:
                best = s, float(dx), float(dy)
    return {"dx": best[1], "dy": best[2], "iou0": iou0, "iou": best[0], "n": nf}


def refine_standing_wide(
    img: np.ndarray,
    cam,
    b: Bottle,
    bad: np.ndarray | None = None,
    others=(),
    max_shift: float = 0.03,
    step: float = 0.01,
    excl_px: int = 8,
    rounds: int = 2,
    models=None,
    good_iou: float = 0.85,
    table_ref: dict | None = None,
) -> dict | None:
    r = refine_standing(
        img, cam, b, bad=bad, others=others, max_shift=max_shift, step=step, excl_px=excl_px, table_ref=table_ref
    )
    if r is None:
        return None
    r = dict(r, model=b.model, rounds=1)
    edge = max_shift - 1e-09
    last = r["dx"], r["dy"]
    for k in range(1, rounds):
        if r["iou"] >= good_iou or (abs(last[0]) < edge and abs(last[1]) < edge):
            break
        best = None
        for name in models or (b.model,):
            m = BY_NAME[name]
            pb = Bottle(
                np.asarray(b.xy, float) + np.array([r["dx"], r["dy"]]),
                True,
                0.0,
                name,
                m.height,
                float(m.r.max()),
                b.iou,
                extra={"rest_r": m.rest_r},
            )
            q = refine_standing(
                img,
                cam,
                pb,
                bad=bad,
                others=others,
                max_shift=max_shift,
                step=step,
                excl_px=excl_px,
                table_ref=table_ref,
            )
            if q is not None and (best is None or q["iou"] > best[0]["iou"]):
                best = q, name
        r["rounds"] = k + 1
        if best is None or best[0]["iou"] <= r["iou"]:
            break
        q, name = best
        last = q["dx"], q["dy"]
        r.update(dx=r["dx"] + q["dx"], dy=r["dy"] + q["dy"], iou=q["iou"], model=name)
    return r


def refine_lying_wide(
    img: np.ndarray,
    cam,
    b: Bottle,
    bad: np.ndarray | None = None,
    others=(),
    max_shift: float = 0.03,
    step: float = 0.01,
    dyaws=(-0.1, 0.0, 0.1),
    excl_px: int = 8,
    rounds: int = 2,
    table_ref: dict | None = None,
) -> dict | None:
    r = refine_lying(
        img,
        cam,
        b,
        bad=bad,
        others=others,
        max_shift=max_shift,
        step=step,
        dyaws=dyaws,
        excl_px=excl_px,
        table_ref=table_ref,
    )
    if r is None:
        return None
    r = dict(r, rounds=1)
    edge = max_shift - 1e-09
    last_lat = r["lat"]
    for k in range(1, rounds):
        if abs(last_lat) < edge:
            break
        d = np.array([math.cos(b.yaw), math.sin(b.yaw)])
        nrm = np.array([-math.sin(b.yaw), math.cos(b.yaw)])
        yaw = b.yaw + r["dyaw"]
        xy = np.asarray(b.xy, float) + r["lat"] * nrm + r["ax"] * d
        pb = Bottle(xy, False, yaw, b.model, b.height, b.radius, b.iou, extra=dict(b.extra))
        q = refine_lying(
            img,
            cam,
            pb,
            bad=bad,
            others=others,
            max_shift=max_shift,
            step=step,
            dyaws=dyaws,
            excl_px=excl_px,
            table_ref=table_ref,
        )
        r["rounds"] = k + 1
        if q is None or q["iou"] <= r["iou"]:
            break
        d2 = np.array([math.cos(yaw), math.sin(yaw)])
        n2 = np.array([-math.sin(yaw), math.cos(yaw)])
        tot = xy + q["lat"] * n2 + q["ax"] * d2 - np.asarray(b.xy, float)
        last_lat = q["lat"]
        r.update(lat=float(tot @ nrm), ax=float(tot @ d), dyaw=r["dyaw"] + q["dyaw"], iou=q["iou"])
    return r
