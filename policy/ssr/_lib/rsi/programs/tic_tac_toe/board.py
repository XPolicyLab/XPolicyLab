from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ...planning.kinematics import TABLE_Z
from ...perception import discs as D
from ...perception import prism_fit as PF
from ...perception.cameras import Camera

BOARD_SIDE = 0.2234
BOARD_H = 0.016
CELL = 0.074
PIECE_R = 0.025
PIECE_H = 0.01
BOARD_MODEL = PF.PrismModel(boxes=((-BOARD_SIDE / 2, BOARD_SIDE / 2, -BOARD_SIDE / 2, BOARD_SIDE / 2),), height=BOARD_H)
BOARD_TOP = TABLE_Z + BOARD_H
GOLD_H = 15, 40
GOLD_S = 60
GOLD_V = 40
DARK_V = 70
CELL_PROBE_R = 0.023
CHANGE_V = 15
CHANGE_S = 40
TAKEN_FRAC = 0.5
OPP_ZONE = (-0.3, 0.3), (0.32, 0.8), (0.765, 1.5)
ZONE_DIFF = 25


def hsv(img: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2HSV)


def gold_mask(img: np.ndarray, hsv_img: np.ndarray | None = None) -> np.ndarray:
    h = hsv(img) if hsv_img is None else hsv_img
    return (h[..., 0] >= GOLD_H[0]) & (h[..., 0] <= GOLD_H[1]) & (h[..., 1] >= GOLD_S) & (h[..., 2] >= GOLD_V)


def dark_mask(img: np.ndarray, hsv_img: np.ndarray | None = None) -> np.ndarray:
    h = hsv(img) if hsv_img is None else hsv_img
    return h[..., 2] < DARK_V


def plane_polygon_mask(cam: Camera, shape, pts_xy, z: float) -> np.ndarray:
    import cv2

    p = np.column_stack([np.asarray(pts_xy, dtype=float), np.full(len(pts_xy), z)])
    uv, d = cam.project(p)
    m = np.zeros(shape[:2], np.uint8)
    if np.any(d <= 0.01):
        return m.astype(bool)
    hull = cv2.convexHull(np.round(uv * 4).astype(np.int32))
    cv2.fillConvexPoly(m, hull, 1, shift=2)
    return m.astype(bool)


def table_roi(cam: Camera, shape, xr=(-0.45, 0.45), yr=(-0.42, 0.25), z: float = TABLE_Z) -> np.ndarray:
    pts = [(xr[0], yr[0]), (xr[1], yr[0]), (xr[1], yr[1]), (xr[0], yr[1])]
    return plane_polygon_mask(cam, shape, pts, z)


def circle_xy(x: float, y: float, r: float, n: int = 32) -> np.ndarray:
    a = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    return np.stack([x + r * np.cos(a), y + r * np.sin(a)], axis=1)


@dataclass
class Board:
    x: float
    y: float
    yaw: float
    iou: float

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y])

    def cells(self) -> np.ndarray:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        out = []
        for r in range(3):
            for k in range(3):
                lx, ly = (k - 1) * CELL, (r - 1) * CELL
                out.append((self.x + c * lx - s * ly, self.y + s * lx + c * ly))
        return np.array(out)

    def contains(self, xy, margin: float = 0.0) -> bool:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        d = np.asarray(xy, dtype=float)[:2] - self.xy
        lx, ly = c * d[0] + s * d[1], -s * d[0] + c * d[1]
        h = BOARD_SIDE / 2 + margin
        return abs(lx) <= h and abs(ly) <= h

    def as_dict(self) -> dict:
        return {
            "x": round(self.x, 4),
            "y": round(self.y, 4),
            "yaw_deg": round(math.degrees(self.yaw), 2),
            "iou": round(self.iou, 3),
        }


@dataclass
class Piece:
    x: float
    y: float
    iou: float
    area: int

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y])

    def as_dict(self) -> dict:
        return {"x": round(self.x, 4), "y": round(self.y, 4), "iou": round(self.iou, 3), "area": self.area}


def _fill(region: np.ndarray) -> np.ndarray:
    import cv2

    rows, cols = np.nonzero(region)
    out = np.zeros(region.shape, np.uint8)
    if len(rows) < 3:
        return out.astype(bool)
    hull = cv2.convexHull(np.column_stack([cols, rows]).astype(np.int32))
    cv2.fillConvexPoly(out, hull, 1)
    return out.astype(bool)


def detect_board(img: np.ndarray, cam: Camera, robot: np.ndarray | None = None) -> Board | None:
    import cv2

    hv = hsv(img)
    m = dark_mask(img, hv) & table_roi(cam, img.shape)
    if robot is not None:
        m &= ~robot
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)
    comps = D.components(m, min_area=800)
    if not comps:
        return None
    region = _fill(comps[0][1])
    f = PF.fit(cam, region, BOARD_MODEL, z0=TABLE_Z, yaw_init=0.0, yaw_span=math.radians(45.0), ignore=robot)
    if f is None:
        return None
    return Board(f.x, f.y, float(math.remainder(f.yaw, math.pi / 2)), f.iou)


def board_silhouette(cam: Camera, board: Board, shape) -> np.ndarray:
    import cv2

    m = np.zeros(shape[:2], np.uint8)
    for poly in PF.silhouette_polys(cam, BOARD_MODEL, board.x, board.y, board.yaw, TABLE_Z):
        cv2.fillConvexPoly(m, np.round(poly * 4).astype(np.int32), 1, shift=2)
    return m.astype(bool)


def detect_pieces(
    img: np.ndarray,
    cam: Camera,
    board: Board | None,
    robot: np.ndarray | None = None,
    min_area: int = 40,
    min_iou: float = 0.5,
) -> list[Piece]:
    import cv2

    g = gold_mask(img) & table_roi(cam, img.shape)
    if board is not None:
        sil = board_silhouette(cam, board, img.shape)
        sil = cv2.dilate(sil.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        g &= ~sil
    if robot is not None:
        g &= ~robot
    g = cv2.morphologyEx(g.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool)
    out = []
    for area, region, _uv in D.components(g, min_area=min_area):
        reg = _fill(region)
        best = D.fit_cylinder(cam, reg, PIECE_R, TABLE_Z, TABLE_Z + PIECE_H, bad=robot)
        if best is None or best[0] < min_iou:
            continue
        sil = D.silhouette(cam, best[1], best[2], PIECE_R, TABLE_Z, TABLE_Z + PIECE_H)
        exp = float(abs(cv2.contourArea(sil.astype(np.float32)))) if len(sil) >= 3 else 0.0
        ratio = float(reg.sum()) / max(1.0, exp)
        if not 0.6 <= ratio <= 1.5:
            continue
        out.append(Piece(float(best[1]), float(best[2]), float(best[0]), int(area)))
    return out


def opponent_zone(cam: Camera, shape=(480, 640)) -> np.ndarray:
    import cv2

    (x0, x1), (y0, y1), (z0, z1) = OPP_ZONE
    pts = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], dtype=float)
    uv, d = cam.project(pts)
    m = np.zeros(shape[:2], np.uint8)
    ok = d > 0.01
    if ok.sum() >= 3:
        cv2.fillConvexPoly(m, cv2.convexHull(np.round(uv[ok]).astype(np.int32)), 1)
    return m.astype(bool)


def grey(img: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2GRAY).astype(np.int16)


def zone_change(g: np.ndarray, ref: np.ndarray, zone: np.ndarray) -> float:
    d = np.abs(g - ref)[zone]
    return float((d > ZONE_DIFF).mean()) if d.size else 0.0


def changed_mask(img: np.ndarray, ref: np.ndarray) -> np.ndarray:
    h = hsv(img).astype(np.int16)
    h0 = hsv(ref).astype(np.int16)
    return (np.abs(h[..., 2] - h0[..., 2]) > CHANGE_V) | (h[..., 1] - h0[..., 1] > CHANGE_S)


def cell_fill(
    img: np.ndarray, cam: Camera, board: Board, robot: np.ndarray | None = None, ref: np.ndarray | None = None
) -> list[float]:
    if ref is None:
        ref = np.zeros_like(img)
        ref[...] = 31, 31, 31
    g = changed_mask(img, ref)
    out = []
    for c in board.cells():
        m = plane_polygon_mask(cam, img.shape, circle_xy(c[0], c[1], CELL_PROBE_R), BOARD_TOP + PIECE_H / 2)
        n = int(m.sum())
        if robot is not None:
            vis = m & ~robot
            if vis.sum() < 0.5 * max(1, n):
                out.append(float("nan"))
                continue
            m = vis
        out.append(float((g & m).sum()) / max(1, int(m.sum())))
    return out


def taken_cells(fill: list[float], thr: float = TAKEN_FRAC) -> list[bool | None]:
    return [None if not np.isfinite(f) else bool(f >= thr) for f in fill]
