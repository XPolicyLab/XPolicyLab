from __future__ import annotations

from ... import ASSETS

import json
import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from ...planning import kinematics as K
from ...perception.cameras import Camera, H, W

_DATA = ASSETS / "object_library.npz"


@dataclass(frozen=True)
class Shape:
    category: str
    place: str
    points: np.ndarray
    extent: np.ndarray
    rot: np.ndarray | None = field(default=None, compare=False)

    @property
    def key(self) -> str:
        return f"{self.category}/{self.place}"

    def posed(self, x: float, y: float, yaw: float, z0: float, n: int | None = None) -> np.ndarray:
        p = self.points if n is None or n >= len(self.points) else self.points[:: max(1, len(self.points) // n)]
        c, s = math.cos(yaw), math.sin(yaw)
        out = np.empty_like(p)
        out[:, 0] = c * p[:, 0] - s * p[:, 1] + x
        out[:, 1] = s * p[:, 0] + c * p[:, 1] + y
        out[:, 2] = p[:, 2] + z0
        return out


@lru_cache(maxsize=1)
def _raw():
    d = np.load(_DATA)
    meta = json.loads(bytes(d["meta"]).decode())
    pts = {k.split("/")[0]: np.asarray(d[k], dtype=float) for k in d.files if k.endswith("/points")}
    return meta, pts


def categories() -> list[str]:
    return sorted(_raw()[1])


@lru_cache(maxsize=None)
def shapes(category: str) -> tuple[Shape, ...]:
    meta, pts = _raw()
    out = []
    for place, q in meta[category]["places"].items():
        rots = [(place, K.quat_to_mat(q))]
        rots += [(f"{place}_rest{k}", r) for k, r in enumerate(settled(pts[category], K.quat_to_mat(q)))]
        for name, rot in rots:
            p = pts[category] @ rot.T
            lo, hi = p.min(0), p.max(0)
            c = (lo + hi) / 2
            p = p - [c[0], c[1], lo[2]]
            p = p[np.random.default_rng(0).permutation(len(p))]
            out.append(Shape(category, name, p, hi - lo, rot))
    return tuple(out)


SETTLE_TOL = 0.0008


def _tilt(a: float, b: float) -> np.ndarray:
    axis = np.array([math.cos(b), math.sin(b), 0.0])
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(a) * k + (1 - math.cos(a)) * (k @ k)


def settled(points: np.ndarray, rot0: np.ndarray, max_tip: float = math.radians(75.0)) -> list[np.ndarray]:
    import heapq

    p = np.asarray(points, dtype=float)[::4]
    com = p.mean(0)
    d0 = rot0.T @ np.array([0.0, 0.0, -1.0])
    t1 = np.cross(d0, [1.0, 0.0, 0.0] if abs(d0[0]) < 0.9 else [0.0, 1.0, 0.0])
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(d0, t1)
    na, nb = int(round(max_tip / math.radians(2.5))), 72
    a = np.arange(na + 1) * math.radians(2.5)
    b = np.arange(nb) * (2 * math.pi / nb)
    A, B = np.meshgrid(a, b, indexing="ij")
    D = np.cos(A)[..., None] * d0 + np.sin(A)[..., None] * (np.cos(B)[..., None] * t1 + np.sin(B)[..., None] * t2)
    Dv = D.reshape(-1, 3)
    h = ((p @ Dv.T).max(axis=0) - com @ Dv.T).reshape(na + 1, nb)
    h[0, :] = h[0, 0]

    def nbrs(i, j):
        if i == 0:
            return [(1, k) for k in range(nb)]
        out = [(i, (j + 1) % nb), (i, (j - 1) % nb), (i - 1, j if i > 1 else 0)]
        if i < na:
            out.append((i + 1, j))
        return out

    best = np.full(h.shape, np.inf)
    best[0, 0] = h[0, 0]
    heap = [(h[0, 0], 0, 0)]
    while heap:
        m, i, j = heapq.heappop(heap)
        if m > best[i, j]:
            continue
        for ii, jj in nbrs(i, j):
            mm = max(m, h[ii, jj])
            if mm < best[ii, jj]:
                best[ii, jj] = mm
                heapq.heappush(heap, (mm, ii, jj))
    h0 = float(h[0, 0])
    reach = best <= h0 + 0.0005
    mins = []
    for i in range(1, na):
        for j in range(nb):
            if reach[i, j] and h[i, j] < h0 - SETTLE_TOL and all(h[i, j] <= h[ii, jj] for ii, jj in nbrs(i, j)):
                mins.append((float(h[i, j]), i, j))
    out: list[np.ndarray] = []
    for _, i, j in sorted(mins):
        d = D[i, j] / np.linalg.norm(D[i, j])
        if any(float(np.dot(d, r.T @ np.array([0.0, 0.0, -1.0]))) > math.cos(math.radians(20.0)) for r in out):
            continue
        w = rot0 @ d
        axis = np.cross(w, [0.0, 0.0, -1.0])
        s, c = float(np.linalg.norm(axis)), float(np.dot(w, [0.0, 0.0, -1.0]))
        ang = math.atan2(s, c)
        out.append(_tilt(ang, math.atan2(axis[1], axis[0])) @ rot0 if s > 1e-09 else rot0.copy())
    return out


def all_shapes() -> list[Shape]:
    return [s for c in categories() for s in shapes(c)]


MASS = {
    "bag": 0.5,
    "bottle": 0.19,
    "camera": 0.5,
    "game_machine": 0.05,
    "garage": 0.25,
    "headset": 0.32,
    "ice_cream": 0.25,
    "phone": 0.5,
    "plate": 0.5,
    "puppet": 0.5,
}
MASS_DEFAULT = 0.5


def mass(category: str) -> float:
    return float(MASS.get(category, MASS_DEFAULT))


def render(cam: Camera, pts: np.ndarray, scale: float = 0.5, close: int = 3) -> np.ndarray:
    import cv2

    h, w = int(round(H * scale)), int(round(W * scale))
    uv, d = cam.project(pts)
    ok = d > 0.001
    u = np.floor(uv[ok, 0] * scale).astype(int)
    v = np.floor(uv[ok, 1] * scale).astype(int)
    m = np.zeros((h + 2, w + 2), np.uint8)
    inb = (u >= -1) & (u <= w) & (v >= -1) & (v <= h)
    m[v[inb] + 1, u[inb] + 1] = 1
    m = m[1:-1, 1:-1]
    if close:
        k = np.ones((close, close), np.uint8)
        m = cv2.morphologyEx(cv2.dilate(m, np.ones((2, 2), np.uint8)), cv2.MORPH_CLOSE, k)
    return m.astype(bool)
