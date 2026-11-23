# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Modifications Copyright © 2026 Martingale. All Rights Reserved.
# This file has been modified by Martingale.


from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..robot_dimensions import OWN_FINGER_THICK_M

from . import robodojo_kinematics as kin

IMAGE_SHAPE = (480, 640)
# Camera intrinsics, head pose and wrist mount: same values as PyRUA-Lean (hosts/robodojo_sight.py), unchanged.
HEAD_FOCAL_PX = 288.1325
HEAD_POSITION = (0.0, -0.41, 1.308)
HEAD_PITCH_DOWN_DEG = 60.0
WRIST_FOCAL_PX = 397.0413
WRIST_MOUNT = (
    (0.0, 0.5, -0.8660254, 0.084842),
    (-1.0, 0.0, 0.0, 0.0),
    (0.0, 0.8660254, 0.5, 0.05094),
    (0.0, 0.0, 0.0, 1.0),
)
# Own-finger and gripper-body outline sizes: same values as PyRUA-Lean, unchanged.
OWN_FINGER_INNER_M = 0.0445
OWN_FINGER_X_M = (0.10, 0.165)
OWN_FINGER_Z_M = (-0.02, 0.011)
ROBOT_OUTLINE = 0.75
BODY_Z_M = 0.045
BODY_X_M = (0.0, 0.11)
BODY_RADIUS_M = 0.04
GRID_CELLS = 40000
CELL_MIN_M = 0.001
LEVEL_M = 0.002
SWEEP_EVERY = 2
SWEEP_MAX_M = 0.36
SECOND_VIEW_MIN = 0.55
SECOND_VIEW_WITHIN = 0.8
SECOND_VIEW_MAX_SHARE = 0.3
SECOND_VIEW_ON_ROBOT = 0.85
SEED_HEIGHTS_M = (0.004, 0.02, 0.05, 0.10, 0.18)
LONE_COVER = 0.985
LONE_LOST_PX = 6.0
LONE_LOST_MIN = 0.003
TALLER_GAIN = 0.02
HULL_BELOW = 0.9
CARVE_KEEPS = 0.85
KNOWN_BETTER = 0.03
KNOWN_FITS = 0.9
KNOWN_SLACK = (1.25, 0.01)
EDGE_LEVEL = 0.3
HANG_MARGIN = 0.03
PART_GAP_PX = 12
MIN_PIXELS = 12
TRACE: list[dict[str, Any]] | None = None
MIN_CELLS = 6


class View:
    """A pinhole camera at a known pose (Isaac's convention: it looks down its -z, +y is up)."""

    def __init__(
        self,
        intrinsic: Any,
        extrinsic: Any,
        shape: tuple[int, int] = IMAGE_SHAPE,
    ) -> None:
        k = np.asarray(intrinsic, dtype=np.float64)
        self.pose = np.asarray(extrinsic, dtype=np.float64)
        self.fx, self.fy, self.cx, self.cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
        self.shape = (int(shape[0]), int(shape[1]))
        self.pos = self.pose[:3, 3]
        self.rot = self.pose[:3, :3]

    def project(self, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(row, col, depth along the optical axis) of world points; depth <= 0 is behind."""

        local = (np.asarray(xyz, dtype=np.float64) - self.pos) @ self.rot
        depth = -local[..., 2]
        safe = np.where(np.abs(depth) > 1e-9, depth, 1e-9)
        return (
            -self.fy * local[..., 1] / safe + self.cy,
            self.fx * local[..., 0] / safe + self.cx,
            depth,
        )

    def rays(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """World direction of each pixel's ray, per unit of depth along the optical axis."""

        local = np.stack(
            [
                (np.asarray(cols, dtype=np.float64) - self.cx) / self.fx,
                -(np.asarray(rows, dtype=np.float64) - self.cy) / self.fy,
                -np.ones(np.shape(rows)),
            ],
            axis=-1,
        )
        return local @ self.rot.T

    def on_plane(
        self, rows: np.ndarray, cols: np.ndarray, z: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Where each pixel's ray meets the horizontal plane ``z`` (N x 2), and which rays do."""

        d = self.rays(rows, cols)
        down = d[..., 2]
        ok = down < -1e-6
        t = np.where(ok, (z - self.pos[2]) / np.where(ok, down, -1.0), 0.0)
        ok &= t > 0
        return self.pos[:2] + d[..., :2] * t[..., None], ok


def head_view() -> View:
    """RoboDojo's fixed head camera."""

    a = math.radians(HEAD_PITCH_DOWN_DEG)
    pose = np.eye(4)
    pose[:3, 1] = [0.0, math.sin(a), math.cos(a)]
    pose[:3, 2] = [0.0, -math.cos(a), math.sin(a)]
    pose[:3, 3] = HEAD_POSITION
    k = [[HEAD_FOCAL_PX, 0, IMAGE_SHAPE[1] / 2], [0, HEAD_FOCAL_PX, IMAGE_SHAPE[0] / 2], [0, 0, 1]]
    return View(k, pose)


def _quat_matrix(quat: Sequence[float]) -> np.ndarray:
    w, x, y, z = (float(v) for v in quat)
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def wrist_view(ee_pose: Sequence[float]) -> View:
    """An arm's wrist camera, from that arm's end-effector pose [x, y, z, qw, qx, qy, qz]."""

    pose = np.eye(4)
    pose[:3, :3] = _quat_matrix(ee_pose[3:7])
    pose[:3, 3] = [float(v) for v in ee_pose[:3]]
    k = [
        [WRIST_FOCAL_PX, 0, IMAGE_SHAPE[1] / 2],
        [0, WRIST_FOCAL_PX, IMAGE_SHAPE[0] / 2],
        [0, 0, 1],
    ]
    return View(k, pose @ np.asarray(WRIST_MOUNT))


def _fill_convex(hit: np.ndarray, rc: np.ndarray) -> None:
    """Mark the pixels inside the convex hull of image points (N x 2: row, col)."""

    pts = np.unique(np.round(rc, 3), axis=0)
    if len(pts) < 3:
        return
    pts = pts[np.lexsort((pts[:, 0], pts[:, 1]))]

    def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float((a[1] - o[1]) * (b[0] - o[0]) - (a[0] - o[0]) * (b[1] - o[1]))

    lower: list[np.ndarray] = []
    for q in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    upper: list[np.ndarray] = []
    for q in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    hull = np.array(lower[:-1] + upper[:-1])
    if len(hull) < 3:
        return
    h, w = hit.shape
    r0, r1 = max(0, int(hull[:, 0].min())), min(h, int(hull[:, 0].max()) + 2)
    c0, c1 = max(0, int(hull[:, 1].min())), min(w, int(hull[:, 1].max()) + 2)
    if r0 >= r1 or c0 >= c1:
        return
    rows, cols = np.mgrid[r0:r1, c0:c1]
    inside = np.ones(rows.shape, dtype=bool)
    for i in range(len(hull)):
        o, a_ = hull[i], hull[(i + 1) % len(hull)]
        inside &= (a_[1] - o[1]) * (rows - o[0]) - (a_[0] - o[0]) * (cols - o[1]) >= -1e-9
    hit[r0:r1, c0:c1] |= inside


def robot_pixels(
    view: View,
    state: dict[str, Any],
    own: str | None = None,
    pad_px: float = 1.0,
    scale: float = ROBOT_OUTLINE,
    between: bool = True,
) -> np.ndarray:
    """The pixels of a view that may show the robot itself, from the state alone.

    Each link, gripper rail and finger is a capsule (the ones ``locate`` drops depth points by),
    drawn as discs along its axis.  Generous on purpose: these pixels are *unknown*, not empty.
    ``own`` names the arm whose wrist camera this view is: that arm is behind the camera except
    for its two fingers, which the camera sees from between them, as two boxes.  Without
    ``between`` the space between a gripper's fingers is left out: what shows there is what the
    gripper is over, not the gripper.
    """

    from . import robodojo_tools as tools

    rows, cols = np.mgrid[0 : view.shape[0], 0 : view.shape[1]]
    hit = np.zeros(view.shape, dtype=bool)

    def capsule(a: np.ndarray, b: np.ndarray, radius: float) -> None:
        n = max(2, int(np.linalg.norm(b - a) / 0.01) + 1)
        r, c, d = view.project(a + (b - a) * np.linspace(0.0, 1.0, n)[:, None])
        for ri, ci, di in zip(r, c, d, strict=True):
            if di <= 0.03:
                continue
            rad = radius * view.fx / di + pad_px
            if ri < -rad or ci < -rad or ri > view.shape[0] + rad or ci > view.shape[1] + rad:
                continue
            lo_r, hi_r = max(0, int(ri - rad)), min(view.shape[0], int(ri + rad) + 2)
            lo_c, hi_c = max(0, int(ci - rad)), min(view.shape[1], int(ci + rad) + 2)
            if lo_r >= hi_r or lo_c >= hi_c:
                continue
            box = (slice(lo_r, hi_r), slice(lo_c, hi_c))
            hit[box] |= (rows[box] - ri) ** 2 + (cols[box] - ci) ** 2 <= rad**2

    for arm in kin.ARMS:
        try:
            q = [float(v) for v in state[f"{arm}_arm_joint_state"]]
        except (KeyError, TypeError, ValueError):
            continue
        if len(q) != 6:
            continue
        opening = min(max(tools._opening(state, arm), 0.0), 1.0)
        if arm == own:
            pose = state.get(f"{arm}_ee_pose")
            if pose is None:
                continue
            rot, at = _quat_matrix(pose[3:7]), np.asarray(pose[:3], dtype=np.float64)
            inner = OWN_FINGER_INNER_M * opening
            for sign in (1.0, -1.0):
                corners = np.array(
                    [
                        [x, sign * y, z]
                        for x in OWN_FINGER_X_M
                        for y in (inner, inner + OWN_FINGER_THICK_M)
                        for z in OWN_FINGER_Z_M
                    ]
                )
                r, c, d = view.project(at + corners @ rot.T)
                if np.all(d > 0.005):
                    _fill_convex(hit, np.c_[r, c])
            continue
        fr = kin.frames(arm, q)
        origin, forward, side = fr[-1][:3, 3], fr[-1][:3, 0], fr[-1][:3, 1]
        chain = [f[:3, 3] for f in fr]
        chain.append(origin + forward * (kin.TIP_OFFSET if between else tools.RAIL_AT_M))
        for a, b in zip(chain, chain[1:], strict=False):
            capsule(a, b, 0.045 * scale)
        up = fr[-1][:3, 2]
        capsule(
            origin + forward * BODY_X_M[0] + up * BODY_Z_M,
            origin + forward * BODY_X_M[1] + up * BODY_Z_M,
            BODY_RADIUS_M * scale,
        )
        rail = origin + forward * tools.RAIL_AT_M
        capsule(
            rail - side * tools.RAIL_HALF_M,
            rail + side * tools.RAIL_HALF_M,
            tools.RAIL_RADIUS_M * scale,
        )
        for sign in (1.0, -1.0):
            off = side * sign * (tools.FINGER_CLOSED_M + tools.FINGER_TRAVEL_M * opening)
            capsule(
                origin + forward * tools.FINGER_FROM_M + off,
                origin + forward * kin.TIP_OFFSET + off,
                tools.FINGER_RADIUS_M * scale,
            )
    return hit


@dataclass
class Sight:
    """What one camera shows of the object: its mask, and the pixels that cannot tell."""

    view: View
    mask: np.ndarray
    unknown: Any = None
    robot: Any = None

    def __post_init__(self) -> None:
        self.mask = np.asarray(self.mask, dtype=bool)
        self._object = self.mask.astype(np.float32)
        if self.unknown is None:
            self.unknown = np.zeros(self.mask.shape, dtype=bool)
        self.robot = np.asarray(self.unknown, dtype=bool)
        self.unknown = self.robot & ~self.mask
        self._maybe: Any = None

    def _sample(self, field: np.ndarray, xyz: np.ndarray, outside: float) -> np.ndarray:
        r, c, depth = self.view.project(xyz)
        h, w = field.shape
        ok = (depth > 1e-6) & (r >= 0) & (r <= h - 1) & (c >= 0) & (c <= w - 1)
        r0 = np.clip(np.floor(np.where(ok, r, 0.0)).astype(np.int64), 0, h - 2)
        c0 = np.clip(np.floor(np.where(ok, c, 0.0)).astype(np.int64), 0, w - 2)
        fr, fc = np.clip(r - r0, 0.0, 1.0), np.clip(c - c0, 0.0, 1.0)
        val = (
            field[r0, c0] * (1 - fr) * (1 - fc)
            + field[r0 + 1, c0] * fr * (1 - fc)
            + field[r0, c0 + 1] * (1 - fr) * fc
            + field[r0 + 1, c0 + 1] * fr * fc
        )
        return np.where(ok, val, outside)

    def shows(self, xyz: np.ndarray, level: float = 0.5) -> np.ndarray:
        """Points that project onto the object's own pixels.

        ``level`` is where the mask's edge is drawn between the last pixel inside and the first
        one outside: 0.5 half way, lower further out.
        """

        return self._sample(self._object, xyz, 0.0) >= level

    def allows(self, xyz: np.ndarray, level: float = 0.5) -> np.ndarray:
        """Points that may belong to the object: on its mask, on pixels that cannot tell, or
        off the image."""

        if self._maybe is None:
            self._maybe = (self.mask | self.unknown).astype(np.float32)
        return self._sample(self._maybe, xyz, 1.0) >= level

    def in_image(self, xyz: np.ndarray) -> np.ndarray:
        r, c, depth = self.view.project(xyz)
        h, w = self.mask.shape
        return (depth > 1e-6) & (r >= 0) & (r <= h - 1) & (c >= 0) & (c <= w - 1)


class _Grid:
    """Cells of the supporting plane around an object's shadow."""

    def __init__(self, lo: np.ndarray, hi: np.ndarray, base_z: float) -> None:
        area = float(np.prod(np.maximum(hi - lo, 1e-6)))
        self.cell = max(CELL_MIN_M, math.sqrt(area / GRID_CELLS))
        self.lo = lo
        self.nx = max(2, int(math.ceil((hi[0] - lo[0]) / self.cell)) + 1)
        self.ny = max(2, int(math.ceil((hi[1] - lo[1]) / self.cell)) + 1)
        gx, gy = np.meshgrid(np.arange(self.nx), np.arange(self.ny))
        self.xy = lo + np.stack([gx.ravel(), gy.ravel()], axis=1) * self.cell
        self.base_z = base_z

    def at(self, height: float, cells: np.ndarray | None = None) -> np.ndarray:
        xy = self.xy if cells is None else self.xy[cells]
        return np.c_[xy, np.full(len(xy), self.base_z + height)]

    def index(self, xy: np.ndarray) -> np.ndarray:
        """The cell each plane point falls into (-1: off the grid)."""

        ix = np.rint((xy[..., 0] - self.lo[0]) / self.cell).astype(np.int64)
        iy = np.rint((xy[..., 1] - self.lo[1]) / self.cell).astype(np.int64)
        ok = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        return np.where(ok, iy * self.nx + ix, -1)

    def holds(self, cells: np.ndarray, xy: np.ndarray) -> np.ndarray:
        """Which plane points fall into one of the given cells (a flat boolean per cell)."""

        ix = np.rint((xy[..., 0] - self.lo[0]) / self.cell).astype(np.int64)
        iy = np.rint((xy[..., 1] - self.lo[1]) / self.cell).astype(np.int64)
        ok = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        out = np.zeros(np.shape(ix), dtype=bool)
        out[ok] = cells[iy[ok] * self.nx + ix[ok]]
        return out


class _Solid:
    """What a reading says the object is: over each cell, a column from ``low`` to ``high``
    (metres above the plane; NaN where there is none)."""

    def __init__(self, grid: _Grid, low: np.ndarray, high: np.ndarray) -> None:
        self.grid, self.low, self.high = grid, low, high
        self.cells = np.isfinite(high)

    def spans(self, xy: np.ndarray, z: np.ndarray, slack: float = 0.0) -> np.ndarray:
        """Which points (plane position and height above the plane) lie in the solid."""

        index = self.grid.index(xy)
        on = index >= 0
        low = np.where(on, self.low[np.where(on, index, 0)], np.nan)
        high = np.where(on, self.high[np.where(on, index, 0)], np.nan)
        with np.errstate(invalid="ignore"):
            return on & (z >= low - slack) & (z <= high + slack)


def _march(view: View, rows: np.ndarray, cols: np.ndarray, solid: _Solid) -> np.ndarray:
    """The depth along each pixel's ray at which it first meets the solid (NaN: never)."""

    grid = solid.grid
    d = view.rays(rows, cols)
    pos = view.pos
    lo_t = np.zeros(len(rows))
    hi_t = np.full(len(rows), 50.0)
    inside = solid.cells
    floor = float(np.nanmin(solid.low[inside])) if inside.any() else 0.0
    roof = float(np.nanmax(solid.high[inside])) if inside.any() else 0.0
    box_lo = np.array([grid.lo[0], grid.lo[1], grid.base_z + floor - 1e-4])
    box_hi = np.array(
        [
            grid.lo[0] + grid.nx * grid.cell,
            grid.lo[1] + grid.ny * grid.cell,
            grid.base_z + roof + 1e-4,
        ]
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        for axis in range(3):
            t1 = (box_lo[axis] - pos[axis]) / d[:, axis]
            t2 = (box_hi[axis] - pos[axis]) / d[:, axis]
            flat = np.abs(d[:, axis]) < 1e-12
            within = (pos[axis] >= box_lo[axis]) & (pos[axis] <= box_hi[axis])
            near = np.where(flat, np.where(within, -np.inf, np.inf), np.minimum(t1, t2))
            far = np.where(flat, np.where(within, np.inf, -np.inf), np.maximum(t1, t2))
            lo_t, hi_t = np.maximum(lo_t, near), np.minimum(hi_t, far)
    alive = hi_t >= lo_t
    first = np.full(len(rows), np.nan)
    if not alive.any() or not inside.any():
        return first
    run = np.linalg.norm(d, axis=1) * np.where(alive, hi_t - lo_t, 0.0)
    steps = int(min(400, max(2, math.ceil(float(run.max()) / (grid.cell * 0.8)) + 1)))
    slack = grid.cell * 0.5
    fractions = np.linspace(0.0, 1.0, steps)
    for offset in range(0, steps, 32):
        pending = np.flatnonzero(alive & np.isnan(first))
        if not len(pending):
            break
        f = fractions[offset : offset + 32, None]
        t = lo_t[pending] + (hi_t[pending] - lo_t[pending]) * f
        at = pos + d[pending] * t[..., None]
        hit = solid.spans(at[..., :2], at[..., 2] - grid.base_z, slack)
        found = hit.any(axis=0)
        columns = np.flatnonzero(found)
        first[pending[columns]] = t[hit[:, columns].argmax(axis=0), columns]
    return first


def _match(sight: Sight, solid: _Solid, pixels: tuple[np.ndarray, np.ndarray]) -> float | None:
    """How well a solid agrees with the object's mask in another view (F1).

    ``None``: this view cannot say (the solid lies outside its image, or on unknown pixels).
    """

    cells = solid.cells
    if cells.sum() < MIN_CELLS:
        return 0.0
    grid = solid.grid
    low, high = solid.low[cells], solid.high[cells]
    inside = total = 0
    for f in (0.0, 0.5, 1.0):
        pts = np.c_[grid.xy[cells], grid.base_z + low + (high - low) * f]
        seen = sight.in_image(pts)
        known = seen & ~sight.unknown[_rc(sight, pts, seen)]
        total += int(known.sum())
        inside += int(sight.shows(pts[known]).sum())
    if total < MIN_CELLS:
        return None
    precision = inside / total
    if len(pixels[0]) == 0:
        return 0.0
    recall = float(np.isfinite(_march(sight.view, pixels[0], pixels[1], solid)).mean())
    if precision + recall <= 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _rc(sight: Sight, pts: np.ndarray, seen: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r, c, _ = sight.view.project(pts)
    h, w = sight.mask.shape
    ri = np.clip(np.rint(np.where(seen, r, 0.0)).astype(np.int64), 0, h - 1)
    ci = np.clip(np.rint(np.where(seen, c, 0.0)).astype(np.int64), 0, w - 1)
    return ri, ci


class _Levels:
    """Which points over the grid the head mask allows, level by level."""

    def __init__(self, head: Sight, grid: _Grid, top_height: float) -> None:
        self.grid = grid
        count = int(top_height / LEVEL_M) + 1
        rows = [head.shows(grid.at(0.0), EDGE_LEVEL)]
        for j in range(1, count):
            row = head.allows(grid.at(j * LEVEL_M), EDGE_LEVEL)
            if not row.any() and not rows[-1].any():
                break
            rows.append(row)
        self.inside = np.array(rows)
        self.count = len(rows)
        self.rise = np.logical_and.accumulate(self.inside, axis=0).sum(axis=0)
        self.drop = np.zeros(self.inside.shape, dtype=np.int32)
        run = np.zeros(self.inside.shape[1], dtype=np.int32)
        for j in range(self.count):
            run = np.where(self.inside[j], run + 1, 0)
            self.drop[j] = run

    def cone(self) -> _Solid:
        """Everything the head mask allows over the grid: per cell, from the lowest level that
        is inside to the highest."""

        any_in = self.inside.any(axis=0)
        first = self.inside.argmax(axis=0)
        last = self.count - 1 - self.inside[::-1].argmax(axis=0)
        return _Solid(
            self.grid,
            np.where(any_in, first * LEVEL_M, np.nan),
            np.where(any_in, last * LEVEL_M, np.nan),
        )

    def prism(self, level: int, rise: np.ndarray | None = None) -> _Solid:
        """What stands on the plane, no taller than ``level``: over each cell of the shadow a
        column as high as the mask allows it there.  For a box or a can that is the box or the
        can; a bottle keeps its shoulders and a mug its handle, each at its own height.
        ``rise`` replaces the head mask's word on how high each column may go (the least that
        every view allows)."""

        rise = self.rise if rise is None else rise
        on = rise > 0
        top = np.minimum(rise - 1, level) * LEVEL_M
        return _Solid(self.grid, np.where(on, 0.0, np.nan), np.where(on, top, np.nan))

    def block(self, level: int) -> _Solid:
        """The cells whose column reaches ``level``, all cut there: the upright prism proper."""

        on = self.rise > level
        return _Solid(
            self.grid,
            np.where(on, 0.0, np.nan),
            np.where(on, level * LEVEL_M, np.nan),
        )

    def rise_in(self, sight: Sight, level: int) -> np.ndarray:
        """How many levels from the plane up another view allows over each cell, counted up to
        ``level`` and only where the head mask has a column at all."""

        out = np.zeros(len(self.grid.xy), dtype=np.int64)
        alive = self.rise > 0
        for j in range(min(level, self.count - 1) + 1):
            if not alive.any():
                break
            ok = sight.allows(self.grid.at(j * LEVEL_M, alive), EDGE_LEVEL)
            index = np.flatnonzero(alive)
            out[index[ok]] = j + 1
            alive[index[~ok]] = False
        return out

    def hang(self, level: int) -> _Solid:
        on = self.inside[level]
        low = (level - self.drop[level] + 1) * LEVEL_M
        return _Solid(
            self.grid,
            np.where(on, low, np.nan),
            np.where(on, level * LEVEL_M, np.nan),
        )

    def solid(self, kind: str, level: int) -> _Solid:
        """The solid a reading is judged by in another view."""

        return self.block(level) if kind == "prism" else self.hang(level)


def _head_cover(head: Sight, solid: _Solid, shadow: np.ndarray) -> float:
    """The share of the head mask's shadow that a prism casts."""

    cells = solid.cells
    if cells.sum() < MIN_CELLS:
        return 0.0
    grid = solid.grid
    height = float(np.nanmax(solid.high[cells]))
    foot = head.view.pos[:2]
    above = head.view.pos[2] - grid.base_z
    scale = above / max(above - height, 1e-6)
    hit = np.zeros(len(shadow), dtype=bool)
    reach = float(np.linalg.norm(shadow - foot, axis=1).max()) * (1.0 - 1.0 / scale)
    for s in np.linspace(1.0 / scale, 1.0, max(2, int(reach / grid.cell) + 2)):
        hit |= grid.holds(cells, foot + (shadow - foot) * s)
    return float(hit.mean())


def largest_part(mask: np.ndarray, gap_px: int = PART_GAP_PX) -> np.ndarray:
    """A mask's largest connected part, with the parts that lie within ``gap_px`` of it.

    A text prompt that fits two objects comes back as one mask over both; a finger across an
    object cuts its mask in two.  The first is two objects, the second one.
    """

    mask = np.asarray(mask, dtype=bool)
    rows = np.nonzero(mask.any(axis=1))[0]
    runs: list[tuple[int, int, int]] = []
    for r in rows:
        line = np.flatnonzero(np.diff(np.r_[0, mask[r].astype(np.int8), 0]))
        runs.extend(
            (int(r), int(a), int(b) - 1) for a, b in zip(line[::2], line[1::2], strict=True)
        )
    if len(runs) <= 1:
        return mask
    parent = list(range(len(runs)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    by_row: dict[int, list[int]] = {}
    for i, (r, _, _) in enumerate(runs):
        by_row.setdefault(r, []).append(i)
    for i, (r, a, b) in enumerate(runs):
        for j in by_row.get(r - 1, ()):
            if runs[j][1] <= b + 1 and runs[j][2] >= a - 1:
                parent[find(i)] = find(j)
    roots = [find(i) for i in range(len(runs))]
    size: dict[int, int] = {}
    box: dict[int, list[int]] = {}
    for root, (r, a, b) in zip(roots, runs, strict=True):
        size[root] = size.get(root, 0) + b - a + 1
        bb = box.setdefault(root, [r, r, a, b])
        bb[0], bb[1], bb[2], bb[3] = min(bb[0], r), max(bb[1], r), min(bb[2], a), max(bb[3], b)
    if len(size) == 1:
        return mask
    main = max(size, key=size.get)  # type: ignore[arg-type]
    m = box[main]
    keep = {
        root
        for root, bb in box.items()
        if max(bb[0] - m[1], m[0] - bb[1], 0) <= gap_px
        and max(bb[2] - m[3], m[2] - bb[3], 0) <= gap_px
    }
    out = np.zeros_like(mask)
    for root, (r, a, b) in zip(roots, runs, strict=True):
        if root in keep:
            out[r, a : b + 1] = True
    return out


def _smear_height(head: Sight, shadow: np.ndarray, base_z: float) -> float | None:
    """An upright object's height from one mask alone.

    Seen from the point under the camera, the mask's shadow on the plane ends to either side in a
    stretch that runs straight away from that point: the shadow of an upright edge (or of the
    side of a can), whose near end is on the plane and whose far end is that edge's top.
    """

    rel = shadow - head.view.pos[:2]
    rho = np.hypot(rel[:, 0], rel[:, 1])
    theta = np.arctan2(rel[:, 0], rel[:, 1])
    if len(rho) < 30 or rho.min() <= 1e-6:
        return None
    width = 1.5 / head.view.fx
    scales = []
    for side in (np.percentile(theta, 1.5), np.percentile(theta, 98.5)):
        near = np.abs(theta - side) <= width
        if near.sum() >= 4:
            scales.append(np.percentile(rho[near], 95) / np.percentile(rho[near], 5))
    if not scales:
        return None
    above = float(head.view.pos[2] - base_z)
    return above * (1.0 - 1.0 / float(np.median(scales)))


def _seen(sights: Sequence[tuple[Sight, Any]], solid: _Solid) -> float:
    """The share of the other views' mask pixels whose ray meets the solid."""

    marks = [float(np.isfinite(_march(s.view, px[0], px[1], solid)).mean()) for s, px in sights]
    return float(np.mean(marks)) if marks else 0.0


def _taller(
    levels: _Levels, rise: np.ndarray, level: int, trusted: Sequence[tuple[Sight, Any]]
) -> int:
    """The level an object's top is at, given the level of its body.

    A prism is judged at one height, and for a bottle the best one is its shoulder: the neck
    above it is thin.  The columns every view allows go higher there.  They also go higher
    along the edge of a box that faces the cameras, where nothing shows whether the box ends:
    so the top is raised only as far as it lets the solid account for more of the other views'
    masks (a neck does, that edge does not).
    """

    last = int(rise.max()) - 1
    if last <= level:
        return level
    most = _seen(trusted, levels.prism(last, rise))
    if most - _seen(trusted, levels.prism(level, rise)) < TALLER_GAIN:
        return level
    low, high = level, last
    while high - low > 1:
        mid = (low + high) // 2
        if most - _seen(trusted, levels.prism(mid, rise)) < TALLER_GAIN / 2:
            high = mid
        else:
            low = mid
    return high


def seeds(
    head: Sight, base_z: float = kin.TABLE_Z, heights: Sequence[float] = SEED_HEIGHTS_M
) -> np.ndarray:
    """World points (N x 3) at which another camera may see the object of a head mask.

    The ray through the middle of the mask, taken at a few heights above the plane: one of them
    lies on or in the object, whatever its height.  Projected into another view they are where
    to point a segmenter that takes a pixel (:func:`measure` then tells which mask it gave is
    the object's).
    """

    mask = largest_part(head.mask)
    rows, cols = np.nonzero(mask)
    if len(rows) < MIN_PIXELS:
        return np.zeros((0, 3))
    mid_r, mid_c = (rows.min() + rows.max()) / 2, (cols.min() + cols.max()) / 2
    i = int(np.argmin((rows - mid_r) ** 2 + (cols - mid_c) ** 2))
    d = head.view.rays(rows[i : i + 1], cols[i : i + 1])[0]
    if d[2] >= -1e-6:
        return np.zeros((0, 3))
    out = []
    for h in heights:
        t = (base_z + h - head.view.pos[2]) / d[2]
        if t > 0:
            out.append(head.view.pos + d * t)
    return np.array(out)


def measure(
    head: Sight,
    others: Sequence[Sight | Sequence[Sight]] = (),
    base_z: float = kin.TABLE_Z,
    known: Sequence[Any] | None = None,
    capture: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Where an object's surface is, from its mask in one view and, if any, in others.

    ``head`` is the view the object is read from (a camera that looks down on it).  ``others``
    see the same object from elsewhere and settle its height; an entry may be a list of masks
    from one camera, of which the one that fits is taken.  ``known`` is the object's height,
    reading and match from an earlier measurement that two views agreed on (``(height_m, kind)``
    or ``(height_m, kind, match)``): it is used when no other view sees the object now, and
    when the views that do agree on it less well than those did; never when a prism that tall
    no longer fits the mask (the object has fallen over).  Returns ``points`` (N x 3), the world
    point each pixel of the head mask shows on the solid the reading describes (what a depth
    image would have given, for an object of that shape) with those pixels (``rows``,
    ``cols``), its ``height_m`` above ``base_z``, the reading taken (``kind``: ``"prism"`` or
    ``"hang"``), the number of ``views`` that agreed and their ``match``, whether the height is
    the ``remembered`` one, and whether a known height was ``outgrown`` (it no longer fits the
    mask); or ``None`` when the mask gives no footprint.
    """

    head = Sight(head.view, largest_part(head.mask))
    rows, cols = np.nonzero(head.mask)
    if len(rows) < MIN_PIXELS:
        return None
    shadow, ok = head.view.on_plane(rows, cols, base_z)
    shadow = shadow[ok]
    above = float(head.view.pos[2] - base_z)
    if len(shadow) < MIN_PIXELS or above <= 0.05:
        return None
    foot = head.view.pos[:2]
    top_height = min(SWEEP_MAX_M, above - 0.05)
    shrunk = foot + (shadow - foot) * (1.0 - top_height / above)
    lo = np.minimum(shadow.min(axis=0), shrunk.min(axis=0)) - 0.006
    hi = np.maximum(shadow.max(axis=0), shrunk.max(axis=0)) + 0.006
    if np.any(hi - lo > 1.6):
        return None
    grid = _Grid(lo, hi, base_z)
    levels = _Levels(head, grid, top_height)
    if levels.inside[0].sum() < MIN_CELLS:
        return None
    full_shadow = shadow
    if len(shadow) > 2500:
        shadow = shadow[:: len(shadow) // 2500 + 1]

    cone = levels.cone()
    seconds: list[list[tuple[Sight, tuple[np.ndarray, np.ndarray]]]] = []
    for entry in others:
        candidates = []
        for sight in entry if isinstance(entry, (list, tuple)) else [entry]:
            r2, c2 = np.nonzero(sight.mask)
            if len(r2) < MIN_PIXELS or len(r2) > SECOND_VIEW_MAX_SHARE * sight.mask.size:
                continue
            if sight.robot is not None and sight.robot[r2, c2].mean() > SECOND_VIEW_ON_ROBOT:
                continue
            if len(r2) > 1500:
                pick = slice(None, None, len(r2) // 1500 + 1)
                r2, c2 = r2[pick], c2[pick]
            within = float(np.isfinite(_march(sight.view, r2, c2, cone)).mean())
            if within >= SECOND_VIEW_WITHIN:
                candidates.append((sight, (r2, c2)))
        if candidates:
            seconds.append(candidates)

    kinds = ("prism", "hang")
    tried = list(range(0, levels.count, SWEEP_EVERY))
    solids: dict[tuple[str, int], _Solid] = {}

    def solid_of(kind: str, level: int) -> _Solid:
        if (kind, level) not in solids:
            solids[kind, level] = levels.solid(kind, level)
        return solids[kind, level]

    trusted = []
    coarse = tried[::2] if len(tried) > 12 else tried
    for candidates in seconds:
        ranked = []
        for sight, px in candidates:
            marks = [
                m
                for kind in kinds
                for j in coarse
                if (m := _match(sight, solid_of(kind, j), px)) is not None
            ]
            if marks:
                ranked.append((max(marks), len(ranked), sight, px))
        if TRACE is not None:
            TRACE.append({"ranked": [(round(m, 3), i, int(s.mask.sum())) for m, i, s, _ in ranked]})
        if ranked and max(ranked)[0] >= SECOND_VIEW_MIN:
            _, _, sight, px = max(ranked)
            trusted.append((sight, px))

    def score(kind: str, level: int) -> float | None:
        marks = [_match(s, solid_of(kind, level), px) for s, px in trusted]
        marks = [m for m in marks if m is not None]
        return float(np.mean(marks)) if marks else None

    best: dict[str, tuple[float, int]] = {}
    if trusted:
        for kind in kinds:
            for j in tried:
                mark = score(kind, j)
                if mark is not None and (kind not in best or mark > best[kind][0] + 1e-9):
                    best[kind] = (mark, j)
        for kind, (_, coarse) in list(best.items()):
            for j in range(
                max(0, coarse - SWEEP_EVERY + 1), min(levels.count, coarse + SWEEP_EVERY)
            ):
                mark = score(kind, j)
                if mark is not None and mark > best[kind][0] + 1e-9:
                    best[kind] = (mark, j)
    if TRACE is not None:
        TRACE.append(
            {"best": {k: (round(m, 3), j) for k, (m, j) in best.items()}, "levels": levels.count}
        )
        if trusted:
            TRACE.append(
                {
                    "sweep": {
                        kind: [(j, round(m, 3)) for j in tried if (m := score(kind, j)) is not None]
                        for kind in kinds
                    }
                }
            )
    recalled: tuple[str, int] | None = None
    if known is not None:
        k_kind = "hang" if known[1] == "hang" else "prism"
        k_level = int(min(levels.count - 1, max(0, round(float(known[0]) / LEVEL_M))))
        whole = _head_cover(head, levels.block(0), shadow)
        above = int(round((float(known[0]) * KNOWN_SLACK[0] + KNOWN_SLACK[1]) / LEVEL_M))
        if k_kind == "hang":
            fits = int(levels.inside[k_level].sum()) >= MIN_CELLS
        else:
            fits = _head_cover(head, levels.block(k_level), shadow) >= KNOWN_FITS * whole
            if fits and above < levels.count:
                fits = _head_cover(head, levels.block(above), shadow) < KNOWN_FITS * whole
        if fits:
            recalled = (k_kind, k_level)
    remembered = False
    if best:
        kind = "prism" if "prism" in best else "hang"
        if "hang" in best and (
            "prism" not in best or best["hang"][0] > best["prism"][0] + HANG_MARGIN
        ):
            kind = "hang"
        mark, level = best[kind]
        views = 1 + len(trusted)
        before = known[2] if known is not None and len(known) > 2 else None
        if recalled is not None and before is not None and mark < float(before) - KNOWN_BETTER:
            (kind, level), mark, views, trusted, remembered = recalled, None, 1, [], True
    elif recalled is not None:
        (kind, level), mark, views, remembered = recalled, None, 1, True
    else:
        kind, mark, views = "prism", None, 1
        level = 0
        limit = _smear_height(head, full_shadow, base_z)
        limit = top_height if limit is None else min(top_height, max(limit, 0.0) + 0.004)
        full = _head_cover(head, levels.block(0), shadow)
        lost = min(1.0 - LONE_COVER, max(LONE_LOST_MIN, LONE_LOST_PX / len(rows)))
        for j in range(1, min(levels.count, int(limit / LEVEL_M) + 1)):
            trial = levels.block(j)
            if trial.cells.sum() < MIN_CELLS:
                break
            if _head_cover(head, trial, shadow) < (1.0 - lost) * full:
                break
            level = j
    if kind == "prism":
        rise = levels.rise
        body = levels.block(level)
        if trusted and _seen(trusted, body) < HULL_BELOW:
            carved = rise
            for sight, _px in trusted:
                carved = np.minimum(carved, levels.rise_in(sight, levels.count - 1))
            if body.cells.any() and float((carved[body.cells] > 0).mean()) >= CARVE_KEEPS:
                rise = carved
                level = _taller(levels, rise, level, trusted)
        solid = levels.prism(level, rise)
    else:
        solid = solid_of(kind, level)
    if solid.cells.sum() < MIN_CELLS:
        return None
    depth = _march(head.view, rows, cols, solid)
    got = np.isfinite(depth)
    if got.sum() < MIN_PIXELS:
        return None
    points = head.view.pos + head.view.rays(rows[got], cols[got]) * depth[got, None]
    result = {
        "points": points,
        "rows": rows[got],
        "cols": cols[got],
        "height_m": float(level * LEVEL_M),
        "kind": kind,
        "views": views,
        "match": mark,
        "remembered": remembered,
        "outgrown": known is not None and recalled is None,
        "cell_m": grid.cell,
    }
    if capture is not None:
        capture.clear()
        capture.update(head=head, trusted=trusted, solid=solid, result=result)
    return result


__all__ = [
    "HEAD_FOCAL_PX",
    "HEAD_POSITION",
    "IMAGE_SHAPE",
    "WRIST_FOCAL_PX",
    "WRIST_MOUNT",
    "Sight",
    "View",
    "head_view",
    "largest_part",
    "measure",
    "robot_pixels",
    "seeds",
    "wrist_view",
]
