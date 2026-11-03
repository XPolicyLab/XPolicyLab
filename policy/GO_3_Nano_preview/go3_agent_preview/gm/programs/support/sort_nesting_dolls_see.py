
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from ...lib.kinematics import chain as kin
from ...lib.geometry.transforms import TABLE_Z
from ...lib.geometry.camera import HeadCamera as C
from ...lib.perception.colour import _inside

DATA_DIR = Path(__file__).resolve().parent / ".." / "data"
MODELS = json.loads((DATA_DIR / "sort_nesting_dolls_by_size_models.json").read_text())

TABLE_BOX = (-0.65, 0.65, -0.45, 0.15)

CHROMA_T = 10.0
LOCAL_KERN = 21
LIGHT_T = 20.0
MIN_BLOB = 60
HOLE_MAX = 3000
LARGE_KERN = 31
LARGE_SCALE = 4
LARGE_CHROMA_T = 20.0

MERGE_PX = 6

NPTS_FIT = 48

_fam_hulls = []
for _f in MODELS["families"]:
    _H = np.asarray(_f["hull"], float)
    _h = np.asarray(_f["heights"], float)
    _fam_hulls.append((_H, _h, float(_f["rep_height"])))

def _rot(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

def size_of(idx):
    return float(_fam_hulls[idx // 5][1][idx % 5])

_DEC = {}

def _dec_points(idx, n):
    key = (idx, n)
    if key not in _DEC:
        H, hs, hrep = _fam_hulls[idx // 5]
        _DEC[key] = decimate(H * (hs[idx % 5] / hrep), n)
    return _DEC[key]

def model_points(idx, x, y, yaw, n):
    return _dec_points(idx, n) @ _rot(yaw) + np.array([x, y, TABLE_Z])

def table_mask(shape=(C.H, C.W)):
    x0, x1, y0, y1 = TABLE_BOX
    pts = [C.project([x, y, TABLE_Z]) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
    m = np.zeros(shape, np.uint8)
    if any(p is None for p in pts):
        m[:] = 1
    else:
        cv2.fillConvexPoly(m, np.round(np.array(pts, np.float32)).astype(np.int32), 1)
    return m > 0

STAND_BOX = (185, 378, 465, 480)

def arm_mask(qs, grips=None, shape=(C.H, C.W), dilate=6):
    m = np.zeros(shape, np.uint8)
    for arm in ("left", "right"):
        q = qs[arm]
        g = kin.GRIP_MAX if grips is None else float(np.clip(grips[arm], 0.0, kin.GRIP_MAX))
        for _, corners in kin.link_box_corners(arm, q, g, pad=0.004).items():
            uv = [C.project(p) for p in corners]
            uv = [p for p in uv if p is not None]
            if len(uv) >= 3:
                cv2.fillConvexPoly(m, np.round(np.array(uv, np.float32)).astype(np.int32), 1)
    u0, v0, u1, v1 = STAND_BOX
    m[v0:v1, u0:u1] = 1
    if dilate:
        m = cv2.dilate(m, np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8))
    return m > 0

def foreground(rgb, bg, valid):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    d = lab - bg
    chroma = np.sqrt(d[..., 1] ** 2 + d[..., 2] ** 2)


    fg = ((chroma > CHROMA_T) | (np.abs(d[..., 0]) > LIGHT_T)) & valid
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
    d = lab - large_background(rgb)
    big = (np.sqrt(d[..., 1] ** 2 + d[..., 2] ** 2) > LARGE_CHROMA_T) & valid
    big = cv2.morphologyEx(big.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    hulls = np.zeros(fg.shape, np.uint8)
    n, lab_, st, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), connectivity=8)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= MIN_BLOB // 6:
            vs, us = np.nonzero(lab_ == i)
            cv2.fillConvexPoly(hulls, cv2.convexHull(np.stack([us, vs], 1).astype(np.int32)), 1)
    return fill_holes(fg | (big & (hulls > 0))) & valid

def large_background(rgb, kern=LARGE_KERN, scale=LARGE_SCALE):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    h, w = lab.shape[:2]
    small = cv2.resize(lab, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
    return cv2.resize(cv2.medianBlur(small, kern), (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)

def fill_holes(fg, max_hole=HOLE_MAX):
    n, lab, st, _ = cv2.connectedComponentsWithStats((~fg).astype(np.uint8), connectivity=4)
    h, w = fg.shape
    keep = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] <= max_hole and st[i, 0] > 0 and st[i, 1] > 0
            and st[i, 0] + st[i, 2] < w and st[i, 1] + st[i, 3] < h]
    return fg | np.isin(lab, keep) if keep else fg

def local_background(rgb, kern=LOCAL_KERN):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    return cv2.medianBlur(lab, kern).astype(np.float32)

def blobs(fg, min_area=MIN_BLOB, merge_px=MERGE_PX):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), connectivity=8)
    parts = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= max(12, min_area // 6)]
    parent = {i: i for i in parts}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def gap(a, b):
        ax, ay, aw, ah = stats[a, 0], stats[a, 1], stats[a, 2], stats[a, 3]
        bx, by, bw, bh = stats[b, 0], stats[b, 1], stats[b, 2], stats[b, 3]
        return max(bx - (ax + aw), ax - (bx + bw), by - (ay + ah), ay - (by + bh), 0)

    for i, a in enumerate(parts):
        for b in parts[i + 1:]:
            if gap(a, b) <= merge_px:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra
    groups = {}
    for i in parts:
        groups.setdefault(find(i), []).append(i)
    out = []
    for members in groups.values():
        pix = np.concatenate([np.argwhere(lab == m) for m in members])
        pix = np.stack([pix[:, 1], pix[:, 0]], axis=1)
        if len(pix) < min_area:
            continue
        out.append({"pix": pix, "raw": int(len(pix)), "fill": 1.0, "area": int(len(pix))})
    out.sort(key=lambda b: -b["area"])
    return out

def project(vu_pts):
    c = (vu_pts - C.POS) @ C.R
    z = -c[:, 2]
    ok = z > 1e-6
    v = np.full(len(vu_pts), np.nan)
    u = np.full(len(vu_pts), np.nan)
    v[ok] = C.CY - C.FX * c[ok, 1] / z[ok]
    u[ok] = C.CX + C.FX * c[ok, 0] / z[ok]
    return np.stack([v, u], axis=1)

def decimate(P, n):
    if len(P) <= n:
        return P
    keep = [int(np.argmax(P[:, 2]))]
    d = np.linalg.norm(P - P[keep[0]], axis=1)
    while len(keep) < n:
        k = int(np.argmax(d))
        keep.append(k)
        d = np.minimum(d, np.linalg.norm(P - P[k], axis=1))
    return P[keep]

def lab_gradient(rgb):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    gx = np.stack([cv2.Sobel(lab[..., k], cv2.CV_32F, 1, 0, ksize=3) for k in range(3)], -1)
    gy = np.stack([cv2.Sobel(lab[..., k], cv2.CV_32F, 0, 1, ksize=3) for k in range(3)], -1)
    return gx, gy, np.sqrt((gx ** 2 + gy ** 2).sum(-1))

def edge_score(grad, idx, x, y, yaw, norm):
    gx, gy, _ = grad
    vu = project(model_points(idx, x, y, yaw, 200))
    pts = vu[np.isfinite(vu).all(1)][:, ::-1].astype(np.float32)
    if len(pts) < 3:
        return 0.0
    h = cv2.convexHull(pts)[:, 0, :]
    c = h.mean(0)
    P, N = [], []
    for i in range(len(h)):
        a, b = h[i], h[(i + 1) % len(h)]
        L = float(np.linalg.norm(b - a))
        k = max(1, int(L / 1.5))
        d = (b - a) / max(L, 1e-6)
        n = np.array([d[1], -d[0]])
        if np.dot(n, (a + b) / 2 - c) < 0:
            n = -n
        P.append(a + (b - a) * np.linspace(0, 1, k, endpoint=False)[:, None])
        N.append(np.repeat(n[None], k, 0))
    P, N = np.concatenate(P), np.concatenate(N)
    u = np.clip(np.round(P[:, 0]).astype(int), 0, gx.shape[1] - 1)
    v = np.clip(np.round(P[:, 1]).astype(int), 0, gx.shape[0] - 1)
    g = np.sqrt(((gx[v, u] * N[:, :1] + gy[v, u] * N[:, 1:]) ** 2).sum(-1))
    return float(g.mean() / norm)

def edge_fit(grad, idx, x0, y0, yaw0=0.0, levels=3, span=0.03):
    _, _, mag = grad
    uv = C.project(np.array([x0, y0, TABLE_Z + 0.04]))
    if uv is None:
        return (0.0, x0, y0, yaw0)
    u, v = int(uv[0]), int(uv[1])
    norm = float(mag[max(v - 60, 0):v + 60, max(u - 60, 0):u + 60].mean()) + 1e-6
    best = (-1.0, x0, y0, yaw0)
    step = span
    for level in range(levels):
        bx, by, byaw = (x0, y0, yaw0) if level == 0 else best[1:]
        yaws = np.linspace(-YAW_SPAN, YAW_SPAN, 5) + yaw0 if level == 0 else (byaw - 4 * step, byaw, byaw + 4 * step)
        for dx in np.linspace(-step, step, 5):
            for dy in np.linspace(-step, step, 5):
                for yaw in yaws:
                    sc = edge_score(grad, idx, bx + dx, by + dy, yaw, norm)
                    if sc > best[0]:
                        best = (sc, bx + dx, by + dy, yaw)
        step /= 3
    return best

def render(vu, shape, off):
    dv, du = off
    pts = np.stack([vu[:, 1] - du, vu[:, 0] - dv], axis=1)
    pts = pts[np.isfinite(pts).all(axis=1)]
    m = np.zeros(shape, np.uint8)
    if len(pts) < 3:
        return m > 0
    hull = cv2.convexHull(pts.astype(np.float32))
    cv2.fillConvexPoly(m, np.round(hull).astype(np.int32), 1)
    return m > 0

class Blob:

    def __init__(self, pix, pad=12, fill=1.0):
        u, v = pix[:, 0], pix[:, 1]
        self.u0 = max(0, int(u.min()) - pad)
        self.v0 = max(0, int(v.min()) - pad)
        self.u1 = int(u.max()) + pad + 1
        self.v1 = int(v.max()) + pad + 1
        self.shape = (self.v1 - self.v0, self.u1 - self.u0)
        self.mask = np.zeros(self.shape, bool)
        self.mask[v - self.v0, u - self.u0] = True
        self.off = (self.v0, self.u0)
        self.height_px = int(v.max() - v.min())
        self.area = int(len(pix))
        self.fill = float(fill)
        c = C.to_plane(u.mean(), v.mean(), TABLE_Z)
        self.xy = np.array([c[0], c[1]])
        self.mask_pts = self.mask.sum()

    def iou(self, idx, x, y, yaw, care=None):
        sil = self.render_model(idx, x, y, yaw)
        if care is None:
            inter = np.count_nonzero(sil & self.mask)
            union = np.count_nonzero(sil | self.mask)
        else:
            inter = np.count_nonzero(sil & self.mask & care)
            union = np.count_nonzero((sil | self.mask) & care)
        return inter / union if union else 0.0

    def render_model(self, idx, x, y, yaw, n=NPTS_FIT):
        return render(project(model_points(idx, x, y, yaw, n)), self.shape, self.off)

YAW_SPAN = 0.30

def _yaw_offsets(step, nyaw):
    if nyaw <= 1:
        return (0.0,)
    if nyaw == 3:
        return (-step * 3, 0.0, step * 3)
    return (-step * 3, -step * 1.5, 0.0, step * 1.5, step * 3)

def fit(blob, idx, x0, y0, yaw0=0.0, levels=3, step0=0.030, nyaw=5, care=None):
    best = (-1.0, x0, y0, yaw0)
    span = np.array([step0, step0, YAW_SPAN])
    for level in range(levels):
        step = span[0]
        cand = []
        for dx in (-step, -step / 2, 0.0, step / 2, step):
            for dy in (-step, -step / 2, 0.0, step / 2, step):
                for dyaw in _yaw_offsets(step, nyaw):
                    cand.append((best[1] + dx if level else x0 + dx,
                                 best[2] + dy if level else y0 + dy,
                                 best[3] + dyaw if level else yaw0 + dyaw))
        for x, y, yaw in cand:
            v = blob.iou(idx, x, y, yaw, care)
            if v > best[0]:
                best = (v, x, y, yaw)
        span[0] = max(step / 4, 4e-4)
        span[1] = max(step / 4, 4e-4)
        span[2] = max(step * 0.75, 0.004)
    return best

PLAUSIBLE_FIT = (-0.50, 0.50, -0.30, 0.00)
PLAUSIBLE_BLOB = (-0.55, 0.55, -0.34, 0.05)
ASSIGN_FLOOR = 0.30
EXPLAIN_T = 0.50
EDGE_LEAD = 0.95

EDGE_WEIGHT = {3: 0.75, 4: 0.75}

EDGE_PROPOSE = {3: True, 4: True}
RED_SPLIT = 6.5


PAIR_GAIN = 0.05
PAIR_REFINE = 3

def _two_seeds(b):
    vs, us = np.nonzero(b.mask)
    if len(us) < 2 * MIN_BLOB:
        return None
    P = np.stack([us + b.u0, vs + b.v0], 1).astype(float)
    c = P.mean(0)
    w, V = np.linalg.eigh(np.cov((P - c).T))
    t = (P - c) @ V[:, -1]
    order = np.sort(t)
    best = None
    for k in range(len(order) // 10, len(order) - len(order) // 10, max(1, len(order) // 200)):
        a, z = order[:k], order[k:]
        v = len(a) * a.var() + len(z) * z.var()
        if best is None or v < best[0]:
            best = (v, 0.5 * (order[k - 1] + order[k]))
    if best is None:
        return None
    seeds = []
    for part in (P[t <= best[1]], P[t > best[1]]):
        if len(part) < MIN_BLOB // 2:
            return None
        q = C.to_plane(float(part[:, 0].mean()), float(part[:, 1].mean()), TABLE_Z + 0.03)
        seeds.append((float(q[0]), float(q[1])))
    return seeds

def _fit_two(b, ia, ib, pa, pb, rounds=2, levels=1, step0=0.03, nyaw=1):
    k = np.ones((5, 5), np.uint8)
    for _ in range(rounds):
        care = ~cv2.dilate(b.render_model(ib, *pb).astype(np.uint8), k).astype(bool)
        _, xa, ya, wa = fit(b, ia, pa[0], pa[1], pa[2], levels=levels, step0=step0, nyaw=nyaw, care=care)
        pa = (xa, ya, wa)
        care = ~cv2.dilate(b.render_model(ia, *pa).astype(np.uint8), k).astype(bool)
        _, xb, yb, wb = fit(b, ib, pb[0], pb[1], pb[2], levels=levels, step0=step0, nyaw=nyaw, care=care)
        pb = (xb, yb, wb)
    both = b.render_model(ia, *pa) | b.render_model(ib, *pb)
    inter = np.count_nonzero(both & b.mask)
    union = np.count_nonzero(both | b.mask)
    return (inter / union if union else 0.0), pa, pb

def _radius(idx):
    P = _dec_points(idx, 48)
    return float(np.max(np.hypot(P[:, 0], P[:, 1])))

def _pairs_for(b, fam, single_best):
    seeds = _two_seeds(b)
    if seeds is None:
        return []
    cheap = []
    for ra in range(5):
        for rb in range(5):
            if ra == rb:
                continue
            ia, ib = fam * 5 + ra, fam * 5 + rb
            v, pa, pb = _fit_two(b, ia, ib, (seeds[0][0], seeds[0][1], 0.0), (seeds[1][0], seeds[1][1], 0.0))
            cheap.append((v, ra, rb, pa, pb))
    cheap.sort(key=lambda t: -t[0])
    out = []
    for v, ra, rb, pa, pb in cheap[:PAIR_REFINE]:
        ia, ib = fam * 5 + ra, fam * 5 + rb
        v, pa, pb = _fit_two(b, ia, ib, pa, pb, rounds=2, levels=2, step0=0.012, nyaw=3)
        sep = float(np.hypot(pa[0] - pb[0], pa[1] - pb[1]))
        if v < single_best + PAIR_GAIN or sep < 0.8 * (_radius(ia) + _radius(ib)):
            continue
        if not (_inside(pa[:2], PLAUSIBLE_FIT) and _inside(pb[:2], PLAUSIBLE_FIT)):
            continue
        out.append((v, ra, rb, pa, pb))
    return out

def assign_with_pairs(M, pairs, floor=ASSIGN_FLOOR):
    dp = {0: (0, 0.0, ())}
    for i in range(M.shape[0]):
        u = float(M[i].max()) if M[i].max() >= EXPLAIN_T else 0.0
        nxt = {k: (c, t - u, ch) for k, (c, t, ch) in dp.items()}
        for mask, (c, t, ch) in dp.items():
            for r in range(M.shape[1]):
                if (mask >> r) & 1 or M[i, r] < floor:
                    continue
                cand = (c + 1, t + float(M[i, r]), ch + (("one", i, r),))
                cur = nxt.get(mask | (1 << r))
                if cur is None or cand[:2] > cur[:2]:
                    nxt[mask | (1 << r)] = cand
            for j, (v, ra, rb, pa, pb) in enumerate(pairs.get(i, [])):
                bit = (1 << ra) | (1 << rb)
                if mask & bit or v < floor:
                    continue
                cand = (c + 2, t + 2 * float(v), ch + (("two", i, j),))
                cur = nxt.get(mask | bit)
                if cur is None or cand[:2] > cur[:2]:
                    nxt[mask | bit] = cand
        dp = nxt
    return max(dp.values(), key=lambda v: v[:2])

EDGE_GRID_X = np.arange(-0.46, 0.4601, 0.02)
EDGE_GRID_Y = np.arange(-0.26, -0.0399, 0.02)
EDGE_LOCS = 8
EDGE_LOC_SEP = 0.07
EDGE_WIN = 0.06

def edge_locations(grad, fams, k=EDGE_LOCS, min_sep=EDGE_LOC_SEP):
    _, _, mag = grad
    nm = cv2.blur(mag, (121, 121)) + 1e-6
    maps = []
    for idx in [f * 5 + r for f in fams for r in range(5)]:
        g = np.full((len(EDGE_GRID_Y), len(EDGE_GRID_X)), -1.0)
        for j, y in enumerate(EDGE_GRID_Y):
            for i, x in enumerate(EDGE_GRID_X):
                uv = C.project(np.array([x, y, TABLE_Z + 0.04]))
                if uv is None:
                    continue
                n = float(nm[min(max(int(uv[1]), 0), nm.shape[0] - 1), min(max(int(uv[0]), 0), nm.shape[1] - 1)])
                g[j, i] = max(edge_score(grad, idx, x, y, yaw, n) for yaw in (0.0, 0.6, 1.2, 1.8, 2.4))
        maps.append((g - g.mean()) / (g.std() + 1e-9))
        yield
    top = np.max(maps, 0)
    peaks = []
    for j in range(len(EDGE_GRID_Y)):
        for i in range(len(EDGE_GRID_X)):
            if top[j, i] >= top[max(j - 1, 0):j + 2, max(i - 1, 0):i + 2].max():
                peaks.append((top[j, i], EDGE_GRID_X[i], EDGE_GRID_Y[j]))
    peaks.sort(reverse=True)
    out = []
    for v, x, y in peaks:
        if all(np.hypot(x - a, y - b) >= min_sep for a, b in out):
            out.append((x, y))
        if len(out) == k:
            break
    return out

def window_blob(fg, xy):
    pts = [C.project([xy[0] + dx, xy[1] + dy, TABLE_Z + dz]) for dx in (-EDGE_WIN, EDGE_WIN)
           for dy in (-EDGE_WIN, EDGE_WIN) for dz in (0.0, 0.14)]
    pts = np.array([p for p in pts if p is not None])
    if len(pts) < 2:
        return None
    u0, v0 = np.floor(pts.min(0)).astype(int)
    u1, v1 = np.ceil(pts.max(0)).astype(int)
    u0, v0 = max(u0, 0), max(v0, 0)
    sub = fg[v0:v1, u0:u1]
    vs, us = np.nonzero(sub)
    if len(us) < MIN_BLOB:
        return None
    return Blob(np.stack([us + u0, vs + v0], 1))

def _coarse(b, models):
    return {idx: fit(b, idx, b.xy[0], b.xy[1], levels=1, step0=0.045) for idx in models}

def _family_steps(bl_, coarse, fam, grad, mix, every):
    M = np.full((len(bl_), 5), -1.0)
    pose = {}
    for i, b in enumerate(bl_):
        _, x0, y0, yaw0 = max((coarse[i][fam * 5 + r] for r in range(5)), key=lambda t: t[0])
        for r in range(5):
            iou, x, y, yaw = fit(b, fam * 5 + r, x0, y0, yaw0, levels=2, step0=0.022)
            pose[(i, r)] = (x, y, yaw)
            if _inside((x, y), PLAUSIBLE_FIT):
                M[i, r] = iou
        if every:
            yield
        w = EDGE_WEIGHT.get(fam, 0.0) if mix and grad is not None else 0.0
        if w > 0 and (M[i] > 0).any():
            E = np.array([edge_fit(grad, fam * 5 + r, *pose[(i, r)])[0] for r in range(5)])
            ok = M[i] > 0
            z = lambda a: (a - a[ok].mean()) / (a[ok].std() + 1e-9)

            mixed = w * z(E) + (1.0 - w) * z(M[i])
            M[i] = np.where(ok, np.clip(M[i][ok].mean() + mixed * (M[i][ok].std() + 1e-9), 0.0, 1.0), M[i])
            yield
    if not every:
        yield
    pairs = {}
    for i, b in enumerate(bl_):
        found = _pairs_for(b, fam, float(max(M[i].max(), 0.0)))
        if found:
            pairs[i] = found
        if every:
            yield
    if not every:
        yield
    cnt, tot, choice = assign_with_pairs(M, pairs)
    return (fam, cnt, tot, choice, M, pose, pairs)

def explain_steps(fg, want=5, debug=None, families=None, grad=None, a_rel=None):
    bl_ = [Blob(b["pix"], fill=b.get("fill", 1.0)) for b in blobs(fg)]
    bl_ = [b for b in bl_ if _inside(b.xy, PLAUSIBLE_BLOB)]
    fams = tuple(range(5)) if families is None else tuple(families)
    if not bl_:
        return []
    models = [f * 5 + r for f in fams for r in range(5)]
    coarse = []
    for b in bl_:
        coarse.append(_coarse(b, models))
        yield
    tried = []
    for fam in fams:
        tried.append((yield from _family_steps(bl_, coarse, fam, grad, False, False)))
    top = max(tried, key=lambda t: t[1:3])
    score = {t[0]: round(float(t[2]), 3) for t in tried}
    lead = tuple(t[0] for t in tried if EDGE_PROPOSE.get(t[0], False) and t[1] >= top[1] and t[2] >= EDGE_LEAD * top[2])
    if grad is not None and lead:
        sub = tuple(f for f in fams if EDGE_PROPOSE.get(f, False))
        locs = yield from edge_locations(grad, sub)
        for xy in locs:
            if all(np.linalg.norm(b.xy - np.asarray(xy)) > EDGE_LOC_SEP / 2 for b in bl_):
                wb = window_blob(fg, xy)
                if wb is not None and _inside(wb.xy, PLAUSIBLE_BLOB):
                    bl_.append(wb)
                    coarse.append(_coarse(wb, [f * 5 + r for f in sub for r in range(5)]))
                    yield
        tried = []
        for fam in sub:
            tried.append((yield from _family_steps(bl_, coarse, fam, grad, True, True)))
    best = max(tried, key=lambda t: t[1:3])
    best = (best[1], best[2], best[0]) + best[3:]


    if a_rel is not None and best[2] in (3, 4) and {3, 4} <= {t[0] for t in tried}:
        chosen = [bl_[i] for kind, i, _ in best[3]]
        def under(b):
            sub = a_rel[b.v0:b.v1, b.u0:b.u1]
            return sub[b.mask[:sub.shape[0], :sub.shape[1]]]
        vals = np.concatenate([under(b) for b in chosen]) if chosen else np.zeros(0)
        if len(vals):
            want_fam = 3 if np.percentile(vals, 75) > RED_SPLIT else 4
            alt = [t for t in tried if t[0] == want_fam]
            if alt and alt[0][1] >= best[0] - 1:
                fam, cnt, tot, choice, M, pose, pairs = alt[0]
                best = (cnt, tot, fam, choice, M, pose, pairs)
    cnt, tot, fam, choice, M, pose, pairs = best
    out = []
    for kind, i, rj in choice:
        if kind == "one":
            r = rj
            x, y, yaw = pose[(i, r)]
            out.append({"idx": fam * 5 + r, "rank": r, "iou": float(M[i, r]),
                        "xy": np.array([x, y]), "yaw": float(yaw), "h": size_of(fam * 5 + r),
                        "h_px": bl_[i].height_px, "source": "joint", "fam": fam})
        else:
            v, ra, rb, pa, pb = pairs[i][rj]
            for r, (x, y, yaw) in ((ra, pa), (rb, pb)):
                out.append({"idx": fam * 5 + r, "rank": r, "iou": float(v),
                            "xy": np.array([x, y]), "yaw": float(yaw), "h": size_of(fam * 5 + r),
                            "h_px": bl_[i].height_px, "source": "pair", "fam": fam})
    out.sort(key=lambda d: -d["iou"])
    if debug is not None:
        debug.append({"n": len(out), "blobs": len(bl_), "family": fam, "lead": list(lead),
                      "family_score": {int(f): v for f, v in score.items()},
                      "dolls": [{"idx": d["idx"], "rank": d["rank"], "iou": round(d["iou"], 3),
                                 "xy": [round(float(d["xy"][0]), 4), round(float(d["xy"][1]), 4)]}
                                for d in out]})
    return out[:want]

def measure_steps(rgb, qs, grips=None, want=5, debug=None, table=None, blocked=None, families=None):
    valid = (table_mask() if table is None else table) & ~(
        arm_mask(qs, grips) if blocked is None else blocked)
    fg = foreground(rgb, local_background(rgb), valid)
    yield
    fams = tuple(range(5)) if families is None else tuple(families)
    grad = lab_gradient(rgb) if any(EDGE_WEIGHT.get(f, 0.0) > 0 for f in fams) else None
    a_rel = None
    if {3, 4} <= set(fams):
        a_rel = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)[..., 1] - local_background(rgb)[..., 1]
    return (yield from explain_steps(fg, want=want, debug=debug, families=families, grad=grad, a_rel=a_rel))

def one_per_rank(dolls, want=5, partial=False):
    by_rank = {}
    for d in dolls:
        r = d["rank"] % 5
        if r not in by_rank or d["iou"] > by_rank[r]["iou"]:
            by_rank[r] = d
    if len(by_rank) < want and not partial:
        return [], None
    fams = {}
    for r, d in by_rank.items():
        fams.setdefault(d["idx"] // 5, []).append(d)
    if not fams:
        return [], None
    fam, group = max(fams.items(), key=lambda kv: len(kv[1]))
    if len(group) < want and not partial:
        return [], None
    return sorted(group, key=lambda d: -d["iou"]), fam
