from __future__ import annotations

from ... import ASSETS

import math
from dataclasses import dataclass, field

import numpy as np

from ...planning import kinematics as K
from ...perception.cameras import Camera

DATA = ASSETS / "sweep_shapes.npz"
DUSTPAN, BROOM = "broom_shovel", "broom"
ORIGIN_DZ = {DUSTPAN: 0.0239, BROOM: 0.0136}
OBJECT_HALF_H = {"small_cube": 0.0096, "brick": 0.015, "earbuds": 0.012, "paper_ball": 0.019}
DEFAULT_HALF_H = 0.015
SUPPORT_C = np.array([0.0, -0.053])
SUPPORT_R = 0.081
HANDLE_Z = 0.0116, 0.0237


@dataclass
class Fit:
    name: str
    x: float
    y: float
    yaw: float
    iou: float
    extra: dict = field(default_factory=dict)

    @property
    def R(self) -> np.ndarray:
        return K.rot_z(self.yaw) @ rest_rotation(self.name)

    @property
    def origin(self) -> np.ndarray:
        return np.array([self.x, self.y, K.TABLE_Z + ORIGIN_DZ[self.name]])

    def to_world(self, p_local) -> np.ndarray:
        return np.atleast_2d(np.asarray(p_local, dtype=float)) @ self.R.T + self.origin


_SHAPES: dict | None = None


def shapes() -> dict:
    global _SHAPES
    if _SHAPES is None:
        d = np.load(DATA)
        _SHAPES = {k: np.asarray(d[k], dtype=float) for k in d.files}
    return _SHAPES


def rest_rotation(name: str) -> np.ndarray:
    if name == DUSTPAN:
        return np.eye(3)
    return K.quat_to_mat(shapes()[f"{name}/rest_quat"])


def world_points(name: str, x: float, y: float, yaw: float) -> np.ndarray:
    R = K.rot_z(yaw) @ rest_rotation(name)
    return shapes()[f"{name}/points"] @ R.T + np.array([x, y, K.TABLE_Z + ORIGIN_DZ[name]])


_FOOTPRINT: dict = {}


def footprint_points(name: str, x: float, y: float, yaw: float, cell: float = 0.01) -> np.ndarray:
    key = name, cell
    if key not in _FOOTPRINT:
        p = (shapes()[f"{name}/points"] @ rest_rotation(name).T)[:, :2]
        k = np.floor(p / cell).astype(np.int64)
        _, idx = np.unique(k, axis=0, return_index=True)
        _FOOTPRINT[key] = p[np.sort(idx)]
    c, s = math.cos(yaw), math.sin(yaw)
    return _FOOTPRINT[key] @ np.array([[c, s], [-s, c]]) + np.array([x, y])


def render(cam: Camera, pts: np.ndarray):
    from ...perception.pickup_find import silhouette

    return silhouette(cam, pts)


def _iou(mask: np.ndarray, patch, off) -> float:
    if patch is None:
        return 0.0
    u0, v0 = off
    h, w = patch.shape
    H, W = mask.shape
    a0, b0 = max(0, v0), max(0, u0)
    a1, b1 = min(H, v0 + h), min(W, u0 + w)
    if a1 <= a0 or b1 <= b0:
        return 0.0
    sub = mask[a0:a1, b0:b1]
    p = patch[a0 - v0 : a1 - v0, b0 - u0 : b1 - u0] > 0
    inter = float(np.count_nonzero(sub & p))
    union = float(mask.sum() + np.count_nonzero(patch) - inter)
    return inter / max(1.0, union)


def pose_iou(cam: Camera, mask: np.ndarray, name: str, x: float, y: float, yaw: float) -> float:
    return _iou(mask, *render(cam, world_points(name, x, y, yaw)))


def mask_xy(cam: Camera, mask: np.ndarray, z: float) -> np.ndarray:
    vv, uu = np.nonzero(mask)
    xyz = cam.on_plane_z(np.c_[uu, vv].astype(float), z)
    return np.nanmean(xyz[:, :2], axis=0)


def fit(
    cam: Camera,
    mask: np.ndarray,
    name: str,
    *,
    yaw0: float | None = None,
    yaw_span: float = math.pi,
    coarse_deg: float = 6.0,
) -> Fit | None:
    return run_to_end(fit_iter(cam, mask, name, yaw0=yaw0, yaw_span=yaw_span, coarse_deg=coarse_deg))


def run_to_end(gen):
    while True:
        try:
            next(gen)
        except StopIteration as e:
            return e.value


def fit_iter(
    cam: Camera,
    mask: np.ndarray,
    name: str,
    *,
    yaw0: float | None = None,
    yaw_span: float = math.pi,
    coarse_deg: float = 6.0,
    chunk: int = 20,
    refine_chunk: int = 4,
):
    import cv2

    if mask.sum() < 30:
        return None
    z0 = K.TABLE_Z + ORIGIN_DZ[name]
    c = mask_xy(cam, mask, z0)
    vv, uu = np.nonzero(mask)
    r0, r1 = max(0, vv.min() - 60), min(mask.shape[0], vv.max() + 60 + 1)
    c0, c1 = max(0, uu.min() - 60), min(mask.shape[1], uu.max() + 60 + 1)
    win = mask[r0:r1, c0:c1].astype(np.float32)
    m_area = float(mask.sum())
    if yaw0 is None:
        yaws = np.radians(np.arange(0.0, 360.0, coarse_deg))
    else:
        yaws = yaw0 + np.radians(np.arange(-math.degrees(yaw_span), math.degrees(yaw_span) + 1e-06, coarse_deg))
    best = None
    for i, yaw in enumerate(yaws):
        if i and i % chunk == 0:
            yield
        patch, (pu, pv) = render(cam, world_points(name, c[0], c[1], yaw))
        if patch is None or patch.shape[0] >= win.shape[0] or patch.shape[1] >= win.shape[1]:
            continue
        res = cv2.matchTemplate(win, patch.astype(np.float32), cv2.TM_CCORR)
        _, inter, _, loc = cv2.minMaxLoc(res)
        score = inter / max(1.0, m_area + float(patch.sum()) - inter)
        if best is None or score > best[0]:
            du, dv = c0 + loc[0] - pu, r0 + loc[1] - pv
            best = score, yaw, du, dv
    if best is None:
        return None
    _, yaw, du, dv = best
    uv0, _ = cam.project(np.array([[c[0], c[1], z0]]))
    x, y = cam.on_plane_z(uv0 + np.array([[du, dv]]), z0)[0, :2]
    yield
    cur = _iou(mask, *render(cam, world_points(name, x, y, yaw))), x, y, yaw
    rounds = 0
    for step, dyaw in ((0.004, math.radians(2.0)), (0.0015, math.radians(0.7))):
        improved = True
        while improved:
            rounds += 1
            if rounds % refine_chunk == 0:
                yield
            improved = False
            for ddx, ddy, dd in ((step, 0, 0), (-step, 0, 0), (0, step, 0), (0, -step, 0), (0, 0, dyaw), (0, 0, -dyaw)):
                cand = cur[1] + ddx, cur[2] + ddy, cur[3] + dd
                s = _iou(mask, *render(cam, world_points(name, *cand)))
                if s > cur[0] + 0.0001:
                    cur = (s, *cand)
                    improved = True
    return Fit(name, float(cur[1]), float(cur[2]), float(math.atan2(math.sin(cur[3]), math.cos(cur[3]))), float(cur[0]))


def support_center(dustpan: Fit) -> np.ndarray:
    return dustpan.to_world([SUPPORT_C[0], SUPPORT_C[1], 0.0])[0, :2]
