from __future__ import annotations

from ... import ASSETS

import math
from dataclasses import dataclass, field

import numpy as np

from ...planning import kinematics as K
from ...perception import prism_fit
from ...perception.cameras import Camera

TOOLS = "hammer", "wrench", "pliers", "tape_measure"
SEG_NAMES = {
    "hammer": ("hammer",),
    "wrench": ("wrench",),
    "pliers": ("pliers", "plier", "piler"),
    "tape_measure": ("tape_measure",),
    "toolbox": ("toolbox",),
}


@dataclass(frozen=True)
class ToolPrior:
    size: tuple
    z_lo: float
    grasp: tuple
    width: float


TOOL_PRIOR = {
    "hammer": ToolPrior((0.0764, 0.2282, 0.0257), -0.0117, (-0.0012, 0.0043), 0.015),
    "wrench": ToolPrior((0.0413, 0.1669, 0.0173), -0.0069, (-0.0028, 0.0183), 0.02),
    "pliers": ToolPrior((0.0609, 0.1547, 0.0246), -0.0116, (-0.0003, -0.0065), 0.054),
    "tape_measure": ToolPrior((0.0612, 0.0618, 0.0315), -0.0158, (0.0035, 0.0014), 0.054),
}
TOOLBOX_REST_YAW = -math.pi / 2
TOOLBOX_ORIGIN_DZ = 0.0251
TOOLBOX_TOP_DZ = 0.05
TOOLBOX_MODEL = prism_fit.PrismModel(boxes=((-0.1389, 0.1385, -0.1372, 0.1372),), height=0.05, origin_z=0.0251)
SLOTS_ASSET = {
    "hammer": (-0.01195, -0.06284, -0.00441),
    "pliers": (0.01936, 0.08417, -0.0061),
    "tape_measure": (-0.0947, 0.07506, -0.00659),
    "wrench": (0.011, -0.00093, -0.00737),
}


@dataclass
class Tool:
    name: str
    x: float
    y: float
    z: float
    yaw: float
    score: float
    n_px: int
    extra: dict = field(default_factory=dict)

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y])


@dataclass
class Toolbox:
    x: float
    y: float
    yaw: float
    iou: float
    score: float
    erosion: float = 0.0

    def slot(self, tool: str) -> np.ndarray:
        ax, ay, az = SLOTS_ASSET[tool]
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return np.array([self.x + c * ax - s * ay, self.y + s * ax + c * ay, K.TABLE_Z + TOOLBOX_ORIGIN_DZ + az])


@dataclass
class Scene:
    toolbox: Toolbox | None
    tools: dict
    instances: list = field(default_factory=list)


def _plane_points(cam: Camera, mask: np.ndarray, z: float) -> np.ndarray:
    v, u = np.nonzero(mask)
    uv = np.stack([u + 0.5, v + 0.5], axis=1).astype(float)
    p = cam.on_plane_z(uv, z)
    return p[np.all(np.isfinite(p), axis=1)]


def tool_pose(cam: Camera, mask: np.ndarray, name: str, z0: float = K.TABLE_Z, q: float = 0.5) -> Tool | None:
    pr = TOOL_PRIOR[name]
    zm = z0 + pr.size[2] / 2
    p = _plane_points(cam, mask, zm)
    if len(p) < 30:
        return None
    lo = np.percentile(p[:, :2], q, axis=0)
    hi = np.percentile(p[:, :2], 100 - q, axis=0)
    c = (lo + hi) / 2
    d = p[:, :2] - p[:, :2].mean(0)
    w, vec = np.linalg.eigh(d.T @ d / len(d))
    ax = vec[:, 1]
    yaw = math.remainder(math.atan2(ax[1], ax[0]) - math.pi / 2, math.pi)
    ext = hi - lo
    return Tool(
        name,
        float(c[0]),
        float(c[1]),
        z0 - pr.z_lo,
        float(yaw),
        0.0,
        int(mask.sum()),
        extra={"ext": [float(ext[0]), float(ext[1])], "elong": float(math.sqrt(w[1] / max(w[0], 1e-12)))},
    )


_MODELS = None


def models() -> dict:
    global _MODELS
    if _MODELS is None:
        d = np.load(ASSETS / "store_tools_models.npz")
        _MODELS = {k: d[k].astype(float) for k in d.files}
    return _MODELS


def _splat(uv: np.ndarray, r0: int, c0: int, h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), bool)
    c = np.floor(uv[:, 0]).astype(int) - c0
    r = np.floor(uv[:, 1]).astype(int) - r0
    ok = (c >= 0) & (c < w) & (r >= 0) & (r < h)
    m[r[ok], c[ok]] = True
    return m


FIT_SS = 4
FIT_CLOSE = 5


def fit_tool(
    cam: Camera,
    mask: np.ndarray,
    name: str,
    init_xy,
    z0: float = K.TABLE_Z,
    span: float = 0.008,
    erosion: float | None = None,
):
    if erosion is not None:
        return _fit_tool_ss(cam, mask, name, init_xy, z0, span, max(0.0, float(erosion)))
    import cv2

    P = models()[name]
    P = P - np.array([0.0, 0.0, P[:, 2].min()])
    v, u = np.nonzero(mask)
    r0, r1 = max(0, v.min() - 12), min(mask.shape[0], v.max() + 12 + 1)
    c0, c1 = max(0, u.min() - 12), min(mask.shape[1], u.max() + 12 + 1)
    M = mask[r0:r1, c0:c1]
    h, w = M.shape
    k = np.ones((3, 3), np.uint8)
    base = P + np.array([init_xy[0], init_xy[1], z0])
    uv0, _ = cam.project(base)
    uvx, _ = cam.project(base + np.array([0.001, 0.0, 0.0]))
    uvy, _ = cam.project(base + np.array([0.0, 0.001, 0.0]))
    Jx, Jy = (uvx - uv0) / 0.001, (uvy - uv0) / 0.001

    def iou(dx, dy):
        R = _splat(uv0 + Jx * dx + Jy * dy, r0, c0, h, w)
        R = cv2.morphologyEx(R.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)
        inter = np.logical_and(R, M).sum()
        return inter / max(1, np.logical_or(R, M).sum())

    best = iou(0.0, 0.0), 0.0, 0.0
    for step, n in ((0.002, int(round(span / 0.002))), (0.0007, 2), (0.00025, 2)):
        cx, cy = best[1], best[2]
        for dx in cx + np.arange(-n, n + 1) * step:
            for dy in cy + np.arange(-n, n + 1) * step:
                s = iou(dx, dy)
                if s > best[0]:
                    best = s, dx, dy
    return float(init_xy[0] + best[1]), float(init_xy[1] + best[2]), float(best[0])


def _fit_tool_ss(cam: Camera, mask: np.ndarray, name: str, init_xy, z0: float, span: float, erosion: float):
    import cv2

    ss = FIT_SS
    P = models()[name]
    P = P - np.array([0.0, 0.0, P[:, 2].min()])
    v, u = np.nonzero(mask)
    r0, r1 = max(0, v.min() - 12), min(mask.shape[0], v.max() + 12 + 1)
    c0, c1 = max(0, u.min() - 12), min(mask.shape[1], u.max() + 12 + 1)
    M = np.kron(mask[r0:r1, c0:c1].astype(np.uint8), np.ones((ss, ss), np.uint8)).astype(bool)
    h, w = M.shape
    kc = np.ones((FIT_CLOSE, FIT_CLOSE), np.uint8)
    n_er = int(round(erosion * ss))
    ke = np.ones((2 * n_er + 1, 2 * n_er + 1), np.uint8) if n_er > 0 else None
    base = P + np.array([init_xy[0], init_xy[1], z0])
    uv0, _ = cam.project(base)
    uvx, _ = cam.project(base + np.array([0.001, 0.0, 0.0]))
    uvy, _ = cam.project(base + np.array([0.0, 0.001, 0.0]))
    Jx, Jy = (uvx - uv0) / 0.001, (uvy - uv0) / 0.001
    off = np.array([c0, r0], dtype=float)

    def iou(dx, dy):
        uv = (uv0 + Jx * dx + Jy * dy - off) * ss
        R = _splat(uv, 0, 0, h, w).astype(np.uint8)
        R = cv2.morphologyEx(R, cv2.MORPH_CLOSE, kc)
        if ke is not None:
            R = cv2.erode(R, ke)
        R = R.astype(bool)
        return np.logical_and(R, M).sum() / max(1, np.logical_or(R, M).sum())

    best = iou(0.0, 0.0), 0.0, 0.0
    for step, n in ((0.002, int(round(span / 0.002))), (0.0007, 2), (0.00025, 2)):
        cx, cy = best[1], best[2]
        for dx in cx + np.arange(-n, n + 1) * step:
            for dy in cy + np.arange(-n, n + 1) * step:
                s = iou(dx, dy)
                if s > best[0]:
                    best = s, dx, dy
    return float(init_xy[0] + best[1]), float(init_xy[1] + best[2]), float(best[0])


def toolbox_pose(cam: Camera, mask: np.ndarray, z0: float = K.TABLE_Z) -> Toolbox | None:
    p = _plane_points(cam, mask, z0 + TOOLBOX_TOP_DZ / 2)
    if len(p) < 200:
        return None
    lo = np.percentile(p[:, :2], 0.3, axis=0)
    hi = np.percentile(p[:, :2], 99.7, axis=0)
    c = (lo + hi) / 2
    fit = prism_fit.fit(
        cam, mask, TOOLBOX_MODEL, z0, init=c, yaw_init=0.0, yaw_span=math.radians(12), coarse_yaw=math.radians(3)
    )
    if fit is None:
        return Toolbox(float(c[0]), float(c[1]), TOOLBOX_REST_YAW, 0.0, 0.0)
    dev = math.remainder(fit.yaw, math.pi / 2)
    return Toolbox(fit.x, fit.y, TOOLBOX_REST_YAW + dev, fit.iou, 0.0)


TOOLBOX_BODY = prism_fit.PrismModel(boxes=((-0.1389, 0.1385, -0.1372, 0.1372),), height=0.042, origin_z=0.0251)
DEBIAS_MAX_SHIFT = 0.008


def _box_render(cam: Camera, x: float, y: float, dyaw: float, shape, off, z0: float, ss: int) -> np.ndarray:
    return prism_fit.render(cam, TOOLBOX_BODY, x, y, dyaw, shape, z0, off=off, ss=ss).astype(bool)


def _edges(R: np.ndarray, cols: np.ndarray, rows: np.ndarray, ss: int) -> np.ndarray:
    t, b, l, r = [], [], [], []
    for c in cols:
        on = np.nonzero(R[:, c * ss : (c + 1) * ss].any(1))[0]
        if len(on):
            t.append(on[0] / ss)
            b.append((on[-1] + 1) / ss)
    for rr in rows:
        on = np.nonzero(R[rr * ss : (rr + 1) * ss].any(0))[0]
        if len(on):
            l.append(on[0] / ss)
            r.append((on[-1] + 1) / ss)
    return np.array([np.median(t), np.median(b), np.median(l), np.median(r)]) if t and l else np.full(4, np.nan)


def toolbox_debias(cam: Camera, mask: np.ndarray, box: Toolbox, z0: float = K.TABLE_Z, iters: int = 3) -> Toolbox:
    v, u = np.nonzero(mask)
    r0, r1 = max(0, v.min() - 10), min(mask.shape[0], v.max() + 10 + 1)
    c0, c1 = max(0, u.min() - 10), min(mask.shape[1], u.max() + 10 + 1)
    M = mask[r0:r1, c0:c1].astype(bool)
    shape = M.shape
    cu, rv = np.nonzero(M.any(0))[0], np.nonzero(M.any(1))[0]
    cols = np.arange(cu[0] + (cu[-1] - cu[0]) // 10, cu[-1] - (cu[-1] - cu[0]) // 10 + 1)
    rows = np.arange(rv[0] + (rv[-1] - rv[0]) // 10, rv[-1] - (rv[-1] - rv[0]) // 10 + 1)
    em = _edges(np.kron(M, np.ones((4, 4), bool)), cols, rows, 4)
    sign = np.array([1.0, -1.0, 1.0, -1.0])
    dyaw = math.remainder(box.yaw - TOOLBOX_REST_YAW, 2 * math.pi)
    x, y, e = box.x, box.y, 0.0
    for _ in range(iters):
        e0 = _edges(_box_render(cam, x, y, dyaw, shape, (r0, c0), z0, 4), cols, rows, 4)
        ex = _edges(_box_render(cam, x + 0.001, y, dyaw, shape, (r0, c0), z0, 4), cols, rows, 4)
        ey = _edges(_box_render(cam, x, y + 0.001, dyaw, shape, (r0, c0), z0, 4), cols, rows, 4)
        A = np.stack([(ex - e0) / 0.001, (ey - e0) / 0.001, sign], axis=1)
        if not (np.all(np.isfinite(A)) and np.all(np.isfinite(em - e0))):
            return box
        sol, *_ = np.linalg.lstsq(A, em - e0, rcond=None)
        x, y, e = x + float(sol[0]), y + float(sol[1]), float(sol[2])
        if abs(sol[0]) + abs(sol[1]) < 0.0001:
            break
    if math.hypot(x - box.x, y - box.y) > DEBIAS_MAX_SHIFT:
        return box
    return Toolbox(float(x), float(y), box.yaw, box.iou, box.score, erosion=float(e))


def perceive(
    img: np.ndarray, cam: Camera, instances=None, z0: float = K.TABLE_Z, fit: bool = False, debias: bool = False
) -> Scene:
    if instances is None:
        from ...perception import seg

        instances = seg.segment(img, camera="head")
    box = None
    cand = [i for i in instances if i.category in SEG_NAMES["toolbox"]]
    if cand:
        best = max(cand, key=lambda i: i.score * math.sqrt(i.area))
        box = toolbox_pose(cam, best.mask, z0)
        if box is not None:
            box.score = float(best.score)
            if debias:
                box = toolbox_debias(cam, best.mask, box, z0)
    tools = {}
    for name in TOOLS:
        cand = [i for i in instances if i.category in SEG_NAMES[name]]
        best = None
        for inst in sorted(cand, key=lambda i: -i.score):
            t = tool_pose(cam, inst.mask, name, z0)
            if t is None:
                continue
            if box is not None and abs(t.x - box.x) < 0.12 and (abs(t.y - box.y) < 0.12):
                continue
            t.score = float(inst.score)
            if fit:
                t.extra["extremes"] = [t.x, t.y]
                er = box.erosion if debias and box is not None else None
                t.x, t.y, t.extra["iou"] = fit_tool(cam, inst.mask, name, (t.x, t.y), z0, erosion=er)
            best = t
            break
        if best is not None:
            tools[name] = best
    return Scene(box, tools, list(instances))
