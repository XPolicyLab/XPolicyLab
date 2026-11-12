from __future__ import annotations

from ... import ASSETS

import json
import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from ...planning import kinematics as K
from ...perception.cameras import Camera, H, W, head

_DATA = ASSETS / "store_assets.npz"
SCALE = 0.5
TABLE_Z = K.TABLE_Z
STAND_HALF_H = 0.1375
RACK_Z0 = 0.025
LSTAND_HALF_H = 0.0391
LSTAND_TOP = TABLE_Z + LSTAND_HALF_H + 0.0388
LAPTOP_Z = LSTAND_TOP + 0.0035
HINGE_Y, HINGE_Z = 0.12356, 0.00021
SCREEN_REST = -110.0


@dataclass
class Fit:
    category: str
    x: float
    y: float
    yaw: float
    z: float
    iou: float
    index: int | None = None
    extra: dict = field(default_factory=dict)

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y])

    def T(self) -> np.ndarray:
        M = np.eye(4)
        M[:3, :3] = K.rot_z(self.yaw) @ _rest_rot(self.category, self.index)
        M[:3, 3] = self.x, self.y, self.z
        return M


@dataclass
class Scene:
    laptop: Fit | None
    rack: Fit | None
    stand: Fit | None
    headset: Fit | None
    laptop_stand: Fit | None
    masks: dict = field(default_factory=dict)


@lru_cache(maxsize=1)
def _priors():
    d = np.load(_DATA)
    meta = json.loads(bytes(d["meta"]).decode())
    return {k: d[k] for k in d.files if k != "meta"}, meta


def _quat_mat(q) -> np.ndarray:
    return K.quat_to_mat(np.asarray(q, dtype=float))


def _rest_rot(category: str, index: int | None) -> np.ndarray:
    if category == "headset":
        _, meta = _priors()
        return _quat_mat(meta[f"headset/{index}"]["places"]["side"])
    return np.eye(3)


def screen_points(angle_deg: float = SCREEN_REST) -> np.ndarray:
    pts, meta = _priors()
    T = np.asarray(meta["laptop"]["hinge"]["T_body_screen_rest"], dtype=float)
    p = pts["laptop/screen"] @ T[:3, :3].T + T[:3, 3]
    a = math.radians(angle_deg - SCREEN_REST)
    c, s = math.cos(a), math.sin(a)
    o = np.array([0.0, HINGE_Y, HINGE_Z])
    q = p - o
    R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return q @ R.T + o


def laptop_points(angle_deg: float = SCREEN_REST) -> np.ndarray:
    pts, _ = _priors()
    return np.concatenate([pts["laptop/body"], screen_points(angle_deg)])


def object_points(category: str, index: int | None = None) -> np.ndarray:
    pts, _ = _priors()
    if category == "laptop":
        return laptop_points()
    key = f"{category}/{(index if index is not None else 0)}"
    return pts[f"{key}/points"] @ _rest_rot(category, index).T


def asset_points(category: str, index: int = 0) -> np.ndarray:
    pts, _ = _priors()
    return pts[f"{category}/{index}/points"]


def headset_indices() -> list[int]:
    _, meta = _priors()
    return sorted(int(k.split("/")[1]) for k in meta if k.startswith("headset/"))


def headset_z(index: int) -> float:
    p = object_points("headset", index)
    return TABLE_Z - float(p[:, 2].min())


def render(cam: Camera, pts: np.ndarray, scale: float = SCALE, centred: bool = False) -> np.ndarray:
    import cv2

    h, w = int(round(H * scale)), int(round(W * scale))
    uv, d = cam.project(pts)
    ok = d > 0.001
    c = 0.5 if centred else 0.0
    u = np.floor(uv[ok, 0] * scale - c).astype(int)
    v = np.floor(uv[ok, 1] * scale - c).astype(int)
    m = np.zeros((h, w), np.uint8)
    inb = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    m[v[inb], u[inb]] = 1
    m = cv2.morphologyEx(cv2.dilate(m, np.ones((2, 2), np.uint8)), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return m.astype(bool)


def _down(mask: np.ndarray, scale: float = SCALE) -> np.ndarray:
    import cv2

    h, w = int(round(H * scale)), int(round(W * scale))
    return cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_AREA) > 0


def _iou(sil: np.ndarray, mask: np.ndarray, valid: np.ndarray) -> float:
    a = sil & valid
    inter = np.count_nonzero(a & mask)
    union = np.count_nonzero(a | mask)
    return inter / union if union else 0.0


def fit_pose(
    cam: Camera,
    mask: np.ndarray,
    pts_obj: np.ndarray,
    z: float,
    xy0,
    yaw_range,
    *,
    ignore: np.ndarray | None = None,
    span: float = 0.04,
    centred: bool = False,
) -> tuple[float, float, float, float]:
    m = _down(mask)
    valid = ~_down(ignore) if ignore is not None else np.ones_like(m)
    m &= valid
    rng = np.random.default_rng(0)
    order = rng.permutation(len(pts_obj))
    best = -1.0, float(xy0[0]), float(xy0[1]), float(np.mean(yaw_range))
    stages = [
        (span, 0.008, (yaw_range[0], yaw_range[1]), math.radians(3.0), 700),
        (0.01, 0.002, None, math.radians(1.0), 1500),
        (0.0025, 0.0005, None, math.radians(0.25), 4000),
    ]
    for half, step, yr, ystep, n in stages:
        p = pts_obj[order[:n]]
        cx, cy, cyaw = best[1], best[2], best[3]
        if yr is None:
            yr = cyaw - 3 * ystep, cyaw + 3 * ystep
        yaws = np.arange(yr[0], yr[1] + 1e-09, ystep)
        offs = np.arange(-half, half + 1e-09, step)
        for yaw in yaws:
            q = p @ K.rot_z(yaw).T
            for dx in offs:
                for dy in offs:
                    sil = render(cam, q + (cx + dx, cy + dy, z), centred=centred)
                    s = _iou(sil, m, valid)
                    if s > best[0]:
                        best = s, cx + dx, cy + dy, float(yaw)
    return best


def _centroid_on_plane(cam: Camera, mask: np.ndarray, z: float) -> np.ndarray:
    v, u = np.nonzero(mask)
    p = cam.on_plane_z(np.array([[u.mean(), v.mean()]]), z)[0]
    return p[:2]


def _best(instances, category):
    c = [i for i in instances if i.category == category]
    return max(c, key=lambda i: i.score * np.count_nonzero(i.mask)) if c else None


def fit_laptop(
    rgb: np.ndarray,
    *,
    xy0,
    yaw0: float,
    z: float = LAPTOP_Z,
    angle_deg: float = SCREEN_REST,
    dyaw: float = math.radians(20.0),
    ignore: np.ndarray | None = None,
    instances=None,
    cam: Camera | None = None,
    span: float = 0.04,
) -> Fit | None:
    cam = cam or head()
    if instances is None:
        from ...perception import seg

        instances = seg.segment(rgb, camera="head")
    i = _best(instances, "laptop")
    if i is None:
        return None
    s, x, y, a = fit_pose(
        cam,
        i.mask,
        laptop_points(angle_deg),
        z,
        np.asarray(xy0, float),
        (yaw0 - dyaw, yaw0 + dyaw),
        ignore=ignore,
        span=span,
    )
    return Fit("laptop", x, y, a, z, s)


def perceive(
    rgb: np.ndarray,
    *,
    ignore: np.ndarray | None = None,
    instances=None,
    cam: Camera | None = None,
    headset_all: bool = True,
    only=None,
    headset_yaw=(math.radians(-35.0), math.radians(35.0)),
    headset_index: int | None = None,
) -> Scene:
    cam = cam or head()
    if instances is None:
        from ...perception import seg

        instances = seg.segment(rgb, camera="head")
    masks = {
        c: _best(instances, c)
        for c in ("laptop", "vertical_laptop_storage_rack", "headset_stand", "headset", "laptop_stand")
    }
    out = {}
    want = set(only) if only is not None else {"laptop_stand", "laptop", "rack", "stand", "headset"}
    yr = math.radians(-14.0), math.radians(4.0)
    i = masks["laptop_stand"]
    if i is not None and "laptop_stand" in want:
        z = TABLE_Z + LSTAND_HALF_H
        s, x, y, a = fit_pose(
            cam, i.mask, object_points("laptop_stand", 1), z, _centroid_on_plane(cam, i.mask, z), yr, ignore=ignore
        )
        out["laptop_stand"] = Fit("laptop_stand", x, y, a, z, s, 1)
    i = masks["laptop"]
    if i is not None and "laptop" in want:
        p0 = _centroid_on_plane(cam, i.mask, LAPTOP_Z + 0.05) - (0.0, 0.06)
        s, x, y, a = fit_pose(cam, i.mask, laptop_points(), LAPTOP_Z, p0, yr, ignore=ignore)
        out["laptop"] = Fit("laptop", x, y, a, LAPTOP_Z, s)
    i = masks["vertical_laptop_storage_rack"]
    if i is not None and "rack" in want:
        z = TABLE_Z + RACK_Z0
        s, x, y, a = fit_pose(
            cam,
            i.mask,
            object_points("vertical_laptop_storage_rack", 0),
            z,
            _centroid_on_plane(cam, i.mask, z + 0.01),
            (math.radians(70.0), math.radians(110.0)),
            ignore=ignore,
        )
        out["rack"] = Fit("vertical_laptop_storage_rack", x, y, a, z, s, 0)
    i = masks["headset_stand"]
    if i is not None and "stand" in want:
        z = TABLE_Z + STAND_HALF_H
        s, x, y, a = fit_pose(
            cam,
            i.mask,
            object_points("headset_stand", 0),
            z,
            _centroid_on_plane(cam, i.mask, z),
            (math.radians(-5.0), math.radians(35.0)),
            ignore=ignore,
            span=0.05,
        )
        out["stand"] = Fit("headset_stand", x, y, a, z, s, 0)
    i = masks["headset"]
    if i is not None and "headset" in want:
        best = None
        cands = []
        idxs = [headset_index] if headset_index is not None else headset_indices() if headset_all else [6]
        for idx in idxs:
            z = headset_z(idx)
            s, x, y, a = fit_pose(
                cam,
                i.mask,
                object_points("headset", idx),
                z,
                _centroid_on_plane(cam, i.mask, z),
                tuple(headset_yaw),
                ignore=ignore,
            )
            cands.append((float(s), int(idx)))
            if best is None or s > best.iou:
                best = Fit("headset", x, y, a, z, s, idx)
        best.extra["cands"] = sorted(cands, reverse=True)
        out["headset"] = best
    return Scene(
        laptop=out.get("laptop"),
        rack=out.get("rack"),
        stand=out.get("stand"),
        headset=out.get("headset"),
        laptop_stand=out.get("laptop_stand"),
        masks={k: v.mask if v is not None else None for k, v in masks.items()},
    )


def _rotvec_mat(r_deg) -> np.ndarray:
    r = np.radians(np.asarray(r_deg, dtype=float))
    a = float(np.linalg.norm(r))
    if a < 1e-12:
        return np.eye(3)
    k = r / a
    Kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(a) * Kx + (1.0 - math.cos(a)) * Kx @ Kx


def fit_in_hand(
    rgb: np.ndarray,
    cam: Camera,
    T_tcp: np.ndarray,
    G0: np.ndarray,
    pts: np.ndarray,
    *,
    ignore: np.ndarray | None = None,
    T_stand: np.ndarray | None = None,
    instances=None,
    mask: np.ndarray | None = None,
    scale: float = SCALE,
    lam: float = 0.02,
    rot_max: float = 35.0,
    lin_max: float = 0.04,
    near_px: int = 61,
) -> tuple[np.ndarray, float, float]:
    import cv2

    G0 = np.asarray(G0, dtype=float)
    T_tcp = np.asarray(T_tcp, dtype=float)
    M0 = T_tcp @ G0
    near = (
        cv2.dilate(
            render(cam, pts @ M0[:3, :3].T + M0[:3, 3], 1.0).astype(np.uint8), np.ones((near_px, near_px), np.uint8)
        )
        > 0
    )
    if mask is None:
        if instances is None:
            from ...perception import seg

            instances = seg.segment(rgb, camera="head", conf=0.05)
        mask = np.zeros((H, W), bool)
        for i in instances:
            if i.category in ("headset", "headset_stand") and np.count_nonzero(i.mask & near) > 200:
                mask |= i.mask
    bad = np.zeros((H, W), bool) if ignore is None else np.asarray(ignore, bool).copy()
    if T_stand is not None:
        Ts = np.asarray(T_stand, dtype=float)
        st = render(cam, object_points("headset_stand", 0) @ Ts[:3, :3].T + Ts[:3, 3], 1.0)
        bad |= cv2.dilate(st.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    m = _down(mask, scale)
    valid = ~_down(bad, scale)
    mv = m & valid

    def corr(x) -> np.ndarray:
        C = np.eye(4)
        C[:3, :3] = _rotvec_mat(x[:3])
        C[:3, 3] = 0.001 * np.asarray(x[3:], dtype=float)
        return C

    def iou(x) -> float:
        M = T_tcp @ corr(x) @ G0
        s = render(cam, pts @ M[:3, :3].T + M[:3, 3], scale) & valid
        u = np.count_nonzero(s | mv)
        return np.count_nonzero(s & mv) / u if u else 0.0

    def obj(x) -> float:
        if np.any(np.abs(x[:3]) > rot_max) or np.any(np.abs(x[3:]) > 1000.0 * lin_max):
            return -1.0
        return iou(x) - lam * (float(np.sum((x[:3] / 20.0) ** 2)) + float(np.sum((x[3:] / 20.0) ** 2)))

    if not mv.any():
        return G0.copy(), 0.0, 0.0
    best = None
    for x0 in [np.zeros(6)] + [np.eye(6)[k] * v for k in range(3) for v in (-20.0, 20.0)]:
        x, f = x0.copy(), obj(x0)
        for stp in (8.0, 4.0, 2.0, 1.0):
            improved = True
            while improved:
                improved = False
                for k in range(6):
                    for sg in (-1.0, 1.0):
                        y = x.copy()
                        y[k] += sg * stp
                        fy = obj(y)
                        if fy > f + 1e-06:
                            x, f, improved = y, fy, True
        if best is None or f > best[0]:
            best = f, x
    xb = best[1]
    return corr(xb) @ G0, iou(xb), iou(np.zeros(6))
