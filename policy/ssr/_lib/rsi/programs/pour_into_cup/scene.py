from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from ...perception.cameras import Camera

TABLE_Z = 0.765
BOTTLE_PROFILE = (
    (0.0, 0.0382),
    (0.03, 0.0382),
    (0.07, 0.0395),
    (0.1, 0.0415),
    (0.105, 0.04),
    (0.111, 0.036),
    (0.117, 0.032),
    (0.122, 0.0253),
    (0.127, 0.016),
    (0.218, 0.0155),
)
BOTTLE_H = 0.218
NECK_R_IN = 0.0121


@dataclass(frozen=True)
class CupModel:
    name: str
    profile: tuple
    r_in: float
    floor: float
    handle: tuple | None

    @property
    def height(self) -> float:
        return float(self.profile[-1][0])

    @property
    def r_out(self) -> float:
        return float(max(r for _, r in self.profile))


CUP_MODELS = (
    CupModel("mug15", ((0.0, 0.0324), (0.0762, 0.0323)), 0.0289, 0.0095, (0.0323, 0.0587, 0.005, 0.0115, 0.0652)),
    CupModel(
        "mug16",
        ((0.0, 0.0347), (0.0152, 0.0346), (0.0381, 0.034), (0.0762, 0.0329)),
        0.0301,
        0.0095,
        (0.034, 0.0682, 0.0067, 0.0048, 0.0743),
    ),
    CupModel(
        "goblet6",
        (
            (0.0, 0.025),
            (0.005, 0.03),
            (0.01, 0.034),
            (0.02, 0.038),
            (0.03, 0.0397),
            (0.04, 0.0401),
            (0.06, 0.0391),
            (0.08, 0.0376),
            (0.1003, 0.0355),
        ),
        0.0328,
        0.009,
        None,
    ),
    CupModel("mug17", ((0.0, 0.0373), (0.0823, 0.0372)), 0.0334, 0.01, (0.0372, 0.0678, 0.0071, 0.0102, 0.065)),
    CupModel(
        "mug18",
        ((0.0, 0.04), (0.0073, 0.045), (0.0146, 0.0485), (0.025, 0.0495), (0.0732, 0.0494)),
        0.0459,
        0.009,
        (0.0494, 0.0677, 0.0072, 0.0162, 0.0649),
    ),
    CupModel(
        "mug19",
        ((0.0, 0.028), (0.0078, 0.032), (0.0155, 0.0345), (0.025, 0.0357), (0.035, 0.0361), (0.0777, 0.0357)),
        0.0328,
        0.008,
        None,
    ),
)
SS = 4
N_RING = 40


def _rings(profile):
    a = np.linspace(0.0, 2 * math.pi, N_RING, endpoint=False)
    zs = np.array([p[0] for p in profile])
    rs = np.array([p[1] for p in profile])
    return zs, rs, np.cos(a), np.sin(a)


def silhouette_polys(cam: Camera, x: float, y: float, profile, z0: float = TABLE_Z, handle=None) -> list[np.ndarray]:
    import cv2

    zs, rs, ca, sa = _rings(profile)
    pts = np.concatenate([np.stack([x + r * ca, y + r * sa, np.full_like(ca, z0 + z)], axis=1) for z, r in zip(zs, rs)])
    uv, d = cam.project(pts)
    if np.any(d <= 0.01):
        return []
    uv = uv.reshape(len(zs), N_RING, 2).astype(np.float32)
    polys = [cv2.convexHull(np.vstack([uv[i], uv[i + 1]])).reshape(-1, 2) for i in range(len(zs) - 1)]
    if handle is not None:
        hx0, hx1, hy, hz0, hz1 = handle
        c = np.array([[x + a, y + b, z0 + c] for a in (hx0, hx1) for b in (-hy, hy) for c in (hz0, hz1)])
        huv, hd = cam.project(c)
        if np.all(hd > 0.01):
            polys.append(cv2.convexHull(huv.astype(np.float32)).reshape(-1, 2))
    return polys


def render(cam: Camera, x: float, y: float, profile, shape, z0: float = TABLE_Z, handle=None) -> np.ndarray:
    import cv2

    m = np.zeros(shape, np.uint8)
    for p in silhouette_polys(cam, x, y, profile, z0, handle):
        cv2.fillConvexPoly(m, np.round((p - 0.5) * 4).astype(np.int32), 1, shift=2)
    return m.astype(bool)


def render_posed(cam: Camera, T: np.ndarray, profile, shape) -> np.ndarray:
    import cv2

    zs, rs, ca, sa = _rings(profile)
    loc = np.concatenate([np.stack([r * ca, r * sa, np.full_like(ca, z)], axis=1) for z, r in zip(zs, rs)])
    uv, d = cam.project(loc @ T[:3, :3].T + T[:3, 3])
    m = np.zeros(shape, np.uint8)
    if np.any(d <= 0.01):
        return m.astype(bool)
    uv = uv.reshape(len(zs), N_RING, 2).astype(np.float32)
    for i in range(len(zs) - 1):
        p = cv2.convexHull(np.vstack([uv[i], uv[i + 1]])).reshape(-1, 2)
        cv2.fillConvexPoly(m, np.round((p - 0.5) * 4).astype(np.int32), 1, shift=2)
    return m.astype(bool)


CHANGE_T = 35
SHADOW_RATIO = 0.25, 0.92
SHADOW_SPREAD = 0.12


def changed_mask(img: np.ndarray, img0: np.ndarray) -> np.ndarray:
    import cv2

    a = img.astype(np.float32)
    b = img0.astype(np.float32)
    ch = np.abs(a - b).max(axis=2) > CHANGE_T
    r = (a + 4.0) / (b + 4.0)
    shadow = (
        (r.max(axis=2) < SHADOW_RATIO[1])
        & (r.min(axis=2) > SHADOW_RATIO[0])
        & (r.max(axis=2) - r.min(axis=2) < SHADOW_SPREAD)
    )
    m = (ch & ~shadow).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return m.astype(bool)


def _rot(axis: int, a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    if axis == 0:
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def held_pose(T_tb0: np.ndarray, contact, theta: float, psi: float, dx: float, dz: float) -> np.ndarray:
    c = np.asarray(contact, float)
    R = _rot(0, psi) @ _rot(1, theta)
    T = np.eye(4)
    T[:3, :3] = R @ T_tb0[:3, :3]
    T[:3, 3] = R @ (T_tb0[:3, 3] - c) + c + np.array([dx, 0.0, dz])
    return T


HELD_RANGES = ("theta", math.radians(22.0), math.radians(2.0)), ("dx", 0.006, 0.003), ("dz", 0.006, 0.003)


def fit_held_bottle(
    cam: Camera, observed: np.ndarray, bad: np.ndarray, T_tcp: np.ndarray, T_tb0: np.ndarray, contact, pad: int = 50
):
    shape = observed.shape
    m0 = render_posed(cam, T_tcp @ T_tb0, BOTTLE_PROFILE, shape)
    if m0.sum() < 100:
        return None
    vs, us = np.nonzero(m0)
    v0, v1 = max(0, vs.min() - pad), min(shape[0], vs.max() + pad + 1)
    u0, u1 = max(0, us.min() - pad), min(shape[1], us.max() + pad + 1)
    obs = observed[v0:v1, u0:u1]
    ok = ~bad[v0:v1, u0:u1]
    obs_ok = obs & ok

    def iou(p):
        T = T_tcp @ held_pose(T_tb0, contact, p["theta"], p["psi"], p["dx"], p["dz"])
        sil = render_posed(cam, T, BOTTLE_PROFILE, shape)[v0:v1, u0:u1] & ok
        inter = float((sil & obs_ok).sum())
        return inter / max(1.0, float((sil | obs_ok).sum()))

    p = dict(theta=0.0, psi=0.0, dx=0.0, dz=0.0)
    best = iou0 = iou(p)
    for scale in (1.0, 0.5, 0.25):
        for _ in range(2):
            for name, rng, step in HELD_RANGES:
                st = step * scale
                lo, hi = (-rng, rng) if scale == 1.0 else (p[name] - 3 * st, p[name] + 3 * st)
                for v in np.arange(lo, hi + 1e-09, st):
                    q = dict(p, **{name: float(np.clip(v, -rng, rng))})
                    f = iou(q)
                    if f > best + 1e-09:
                        best, p = f, q
    return dict(iou=best, iou0=iou0, T_tb=held_pose(T_tb0, contact, p["theta"], p["psi"], p["dx"], p["dz"]), **p)


def mirrored(profile, handle=None):
    mp = tuple((-z, r) for z, r in profile)
    mh = None if handle is None else (handle[0], handle[1], handle[2], -handle[4], -handle[3])
    return mp, mh


def fit_profile(
    cam: Camera,
    region: np.ndarray,
    profile,
    *,
    init,
    bad: np.ndarray | None = None,
    span: float = 0.02,
    coarse: float = 0.003,
    z0: float = TABLE_Z,
    pad: int = 8,
    handle=None,
    mirror: bool = False,
):
    import cv2

    rows, cols = np.nonzero(region)
    if len(rows) < 10:
        return None
    polys = []
    for dx in (-span, span):
        for dy in (-span, span):
            polys += silhouette_polys(cam, init[0] + dx, init[1] + dy, profile, z0, handle)
    allp = np.vstack([np.column_stack([cols, rows])] + [p for p in polys if len(p)])
    c0, r0 = (np.floor(allp.min(0)) - pad).astype(int)
    c1, r1 = (np.ceil(allp.max(0)) + pad).astype(int)
    h, w = region.shape
    c0, r0 = max(0, c0), max(0, r0)
    c1, r1 = min(w, c1), min(h, r1)
    win = region[r0:r1, c0:c1].astype(np.uint8)
    m_ss = np.kron(win, np.ones((SS, SS), np.uint8))
    keep = None
    if bad is not None:
        keep = np.kron((~bad[r0:r1, c0:c1]).astype(np.uint8), np.ones((SS, SS), np.uint8))
        m_ss &= keep
    am = float(np.count_nonzero(m_ss))
    buf = np.zeros(m_ss.shape, np.uint8)
    mbuf = np.zeros(m_ss.shape, np.uint8) if mirror else None
    mprofile, mhandle = mirrored(profile, handle)
    shift = np.array([c0, r0], dtype=np.float64)

    def iou(x, y) -> float:
        ps = silhouette_polys(cam, x, y, profile, z0, handle)
        if not ps:
            return -1.0
        buf[:] = 0
        for poly in ps:
            p = (poly - shift) * SS - 0.5
            cv2.fillConvexPoly(buf, np.round(p * 4).astype(np.int32), 1, shift=2)
        if keep is not None:
            buf[:] &= keep
        inter = float(np.count_nonzero(buf & m_ss))
        ign = 0.0
        if mirror:
            mbuf[:] = 0
            for poly in silhouette_polys(cam, x, y, mprofile, z0, mhandle):
                p = (poly - shift) * SS - 0.5
                cv2.fillConvexPoly(mbuf, np.round(p * 4).astype(np.int32), 1, shift=2)
            mbuf[buf > 0] = 0
            ign = float(np.count_nonzero(mbuf & m_ss))
        union = am - ign + float(np.count_nonzero(buf)) - inter
        return inter / union if union > 0 else 0.0

    best = -1.0, float(init[0]), float(init[1])
    g = np.arange(-span, span + 1e-09, coarse)
    for dx in g:
        for dy in g:
            v = iou(init[0] + dx, init[1] + dy)
            if v > best[0]:
                best = v, init[0] + dx, init[1] + dy
    for sp, st in ((coarse, 0.001), (0.001, 0.0005)):
        _, bx, by = best
        offs = np.arange(-sp, sp + 1e-09, st)
        for dx in offs:
            for dy in offs:
                v = iou(bx + dx, by + dy)
                if v > best[0]:
                    best = v, bx + dx, by + dy
    return best


BOTTLE_REGION = (0.05, 0.3), (-0.2, 0.05)
CUP_REGION = (-0.05, 0.05), (-0.1, 0.0)
REGION_MARGIN = 0.03
BOTTLE_LABELS = "bottle", "vase"
CUP_LABELS = "cup", "wine glass", "vase", "bowl"
MIN_SCORE = 0.2
ANY_IOU = 0.5
MERGED_IOU = 0.4
MERGED_COVER = 0.9
LOW_SCORE = 0.05
OCCLUDED_SPAN = 0.04
ZOOM = 2
MIRROR_COVER = 0.2
MIRROR_SPAN = 0.03
MIRROR_PRETEST = 0.1
BOTTLE_CUP_TOL = 0.015
WEAK_MIN_IOU = 0.8
BOTTLE_CUP_MIN = BOTTLE_PROFILE[0][1] + min(m.profile[0][1] for m in CUP_MODELS) - BOTTLE_CUP_TOL


def on_cup(x: float, y: float, cup) -> bool:
    return cup is not None and math.hypot(x - cup.x, y - cup.y) < BOTTLE_CUP_MIN


@dataclass
class Fit:
    kind: str
    x: float
    y: float
    iou: float
    score: float
    model: CupModel | None = None
    mask: np.ndarray | None = field(default=None, repr=False, compare=False)
    weak: bool = False

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y])

    def as_dict(self) -> dict:
        return dict(
            kind=self.kind, x=round(self.x, 4), y=round(self.y, 4), iou=round(self.iou, 3), score=round(self.score, 3)
        )


def _in_region(xy, region, margin: float = REGION_MARGIN, mirror: bool = False) -> bool:
    (x0, x1), (y0, y1) = region
    x = abs(xy[0]) if mirror else xy[0]
    return x0 - margin <= x <= x1 + margin and y0 - margin <= xy[1] <= y1 + margin


def _centre_guess(cam: Camera, mask: np.ndarray, z_mid: float, radius: float, bottom_only: bool = False):
    vs, us = np.nonzero(mask)
    vb = vs.max()
    ub = float(us[vs >= vb - 2].mean())
    p0 = cam.on_plane_z(np.array([[ub + 0.5, vb + 0.5]]), TABLE_Z)[0]
    d = p0[:2] - np.asarray(cam.pos[:2], float)
    p0 = p0[:2] + radius * d / max(1e-06, float(np.linalg.norm(d)))
    p1 = cam.on_plane_z(np.array([[float(us.mean()) + 0.5, float(vs.mean()) + 0.5]]), z_mid)[0][:2]
    if not (np.all(np.isfinite(p0)) and np.all(np.isfinite(p1))):
        return None
    return p0 if bottom_only else 0.5 * (p0 + p1)


def _bottle_any_label(cam: Camera, instances, cup, cup_inst, bad, visible: bool = False, span: float = 0.02):
    best = None
    for inst in instances:
        m = inst.mask
        if visible and bad is not None:
            m = m & ~bad
        if m.sum() < 150 or (cup is not None and inst is cup_inst):
            continue
        f = None
        for bottom_only in (
            False,
            True,
        ):
            g = _centre_guess(cam, m, TABLE_Z + 0.09, 0.04, bottom_only=bottom_only)
            if g is None or not _in_region(g, BOTTLE_REGION, mirror=True):
                continue
            fg = fit_profile(cam, m, BOTTLE_PROFILE, init=g, bad=bad, span=span)
            if fg is not None and (f is None or fg[0] > f[0]):
                f = fg
        if f is None or f[0] <= MERGED_IOU or on_cup(float(f[1]), float(f[2]), cup):
            continue
        sil = render(cam, float(f[1]), float(f[2]), BOTTLE_PROFILE, m.shape)
        if visible and bad is not None:
            sil &= ~bad
        cover = float((sil & m).sum()) / max(1.0, float(sil.sum()))
        if (f[0] > ANY_IOU or cover > MERGED_COVER) and (best is None or f[0] > best.iou):
            best = Fit("bottle", float(f[1]), float(f[2]), float(f[0]), inst.score, mask=m)
    return best


def mirror_cover(cam: Camera, region: np.ndarray, x: float, y: float, profile, handle=None, bad=None) -> float:
    sil = render(cam, x, y, profile, region.shape, handle=handle)
    mp, mh = mirrored(profile, handle)
    e = render(cam, x, y, mp, region.shape, handle=mh) & ~sil
    if bad is not None:
        e &= ~bad
    n = float(e.sum())
    return float((e & region).sum()) / n if n > 0 else 0.0


def mirror_refit(cam: Camera, fit: Fit, bad) -> Fit:
    m = fit.mask
    if m is None:
        return fit
    prof0, h0 = (BOTTLE_PROFILE, None) if fit.model is None else (fit.model.profile, fit.model.handle)
    cover0 = mirror_cover(cam, m, fit.x, fit.y, prof0, h0, bad)
    if fit.model is None and cover0 < MIRROR_PRETEST:
        return fit
    models = [None] if fit.model is None else list(CUP_MODELS)
    best = None
    for model in models:
        prof, handle = (BOTTLE_PROFILE, None) if model is None else (model.profile, model.handle)
        f = fit_profile(
            cam, m, prof, init=(fit.x, fit.y), bad=bad, span=MIRROR_SPAN, coarse=0.005, handle=handle, mirror=True
        )
        if f is not None and (best is None or f[0] > best[0][0]):
            best = f, model, prof, handle
    if best is None:
        return fit
    (iou, x, y), model, prof, handle = best
    if max(abs(x - fit.x), abs(y - fit.y)) > MIRROR_SPAN - 0.003:
        f = fit_profile(cam, m, prof, init=(x, y), bad=bad, span=MIRROR_SPAN, coarse=0.005, handle=handle, mirror=True)
        if f is not None and f[0] > iou:
            iou, x, y = f
    cover = mirror_cover(cam, m, x, y, prof, handle, bad)
    take = cover >= MIRROR_COVER and iou > fit.iou
    if not take:
        return fit
    return Fit(
        fit.kind if model is None else model.name,
        float(x),
        float(y),
        float(iou),
        fit.score,
        model,
        mask=m,
        weak=fit.weak,
    )


def zoom_crops(cam: Camera, shape) -> list[tuple[int, int]]:
    H, W = shape[:2]
    h, w = H // ZOOM, W // ZOOM
    (ax0, ax1), (y0, y1) = BOTTLE_REGION
    m = REGION_MARGIN
    out = []
    for s in (-1.0, 1.0):
        pts = np.array(
            [[s * x, y, TABLE_Z + z] for x in (ax0 - m, ax1 + m) for y in (y0 - m, y1 + m) for z in (0.0, 0.2)]
        )
        uv = cam.project(pts)[0]
        cu = 0.5 * (float(uv[:, 0].min()) + float(uv[:, 0].max()))
        cv_ = 0.5 * (float(uv[:, 1].min()) + float(uv[:, 1].max()))
        out.append((int(np.clip(round(cu - w / 2), 0, W - w)), int(np.clip(round(cv_ - h / 2), 0, H - h))))
    return out


def zoom_instances(img: np.ndarray, cam: Camera, cup=None) -> list:
    import cv2
    from ...perception import detector as D

    H, W = img.shape[:2]
    h, w = H // ZOOM, W // ZOOM
    out = []
    for c0, r0 in zoom_crops(cam, img.shape):
        up = cv2.resize(np.ascontiguousarray(img[r0 : r0 + h, c0 : c0 + w]), (W, H), interpolation=cv2.INTER_LINEAR)
        for i in D.detect(up, labels=None, min_score=LOW_SCORE):
            prob = np.zeros((H, W), np.float32)
            prob[r0 : r0 + h, c0 : c0 + w] = cv2.resize(i.prob.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
            box = np.asarray(i.box, float) / ZOOM + np.array([c0, r0, c0, r0], float)
            inst = D.Instance(i.label, float(i.score), box, prob)
            if cup is not None and getattr(cup, "mask", None) is not None:
                m, cm = inst.mask, cup.mask
                if float((m & cm).sum()) > 0.5 * float((m | cm).sum()):
                    continue
            out.append(inst)
    return sorted(out, key=lambda i: -i.score)


def detect(img: np.ndarray, cam: Camera, bad: np.ndarray | None = None, instances=None):
    from ...perception import detector as D

    if instances is None:
        instances = D.detect(img, labels=None, min_score=LOW_SCORE)
    low = instances
    instances = [i for i in low if i.score >= MIN_SCORE]
    names = {v: k for k, v in D.COCO.items()}
    bottle, cup = None, None
    seen = []
    cup_inst = None
    for inst in instances:
        name = names.get(inst.label, "?")
        m = inst.mask
        if m.sum() < 150:
            continue
        if bottle is None and name in BOTTLE_LABELS:
            g = _centre_guess(cam, m, TABLE_Z + 0.09, 0.04)
            if g is not None and _in_region(g, BOTTLE_REGION, mirror=True):
                f = fit_profile(cam, m, BOTTLE_PROFILE, init=g, bad=bad, span=0.02)
                if f is not None and f[0] > 0.5:
                    bottle = Fit("bottle", float(f[1]), float(f[2]), float(f[0]), inst.score, mask=m)
                    seen.append(id(inst))
                    continue
        if cup is None and name in CUP_LABELS and (id(inst) not in seen):
            g = _centre_guess(cam, m, TABLE_Z + 0.04, 0.035)
            if g is not None and _in_region(g, CUP_REGION):
                best = None
                for model in CUP_MODELS:
                    f = fit_profile(cam, m, model.profile, init=g, bad=bad, span=0.02, handle=model.handle)
                    if f is not None and (best is None or f[0] > best[0][0]):
                        best = f, model
                if best is not None and best[0][0] > 0.5:
                    f, model = best
                    cup = Fit(model.name, float(f[1]), float(f[2]), float(f[0]), inst.score, model, mask=m)
                    cup_inst = inst
        if bottle is not None and cup is not None:
            break
    if bottle is not None and on_cup(bottle.x, bottle.y, cup):
        bottle = None
    if bottle is None:
        bottle = _bottle_any_label(cam, instances, cup, cup_inst, bad)
    if bottle is None and bad is not None:
        bottle = _bottle_any_label(cam, low, cup, cup_inst, bad, visible=True, span=OCCLUDED_SPAN)
        bottle = None if bottle is None else replace(bottle, weak=True)
        if bottle is not None and bottle.iou < WEAK_MIN_IOU:
            bottle = None
    if bottle is None and bad is not None and (img is not None):
        bottle = _bottle_any_label(cam, zoom_instances(img, cam, cup), cup, None, bad, visible=True, span=OCCLUDED_SPAN)
        bottle = None if bottle is None else replace(bottle, weak=True)
        if bottle is not None and bottle.iou < WEAK_MIN_IOU:
            bottle = None
    if bottle is not None:
        bottle = mirror_refit(cam, bottle, bad)
    if cup is not None:
        cup = mirror_refit(cam, cup, bad)
    return bottle, cup
