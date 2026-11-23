# Copyright 2022 Google LLC.
# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Adapted from Google Research, "Code as Policies" (code_as_policies, Apache-2.0).


from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

_EPS = 1e-12


def _array(value: object, name: str) -> np.ndarray:
    import numpy as np

    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite numbers") from exc
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite numbers")
    return result


def _point(value: object, name: str, dimensions: int | None = None) -> np.ndarray:
    result = _array(value, name)
    allowed = (2, 3) if dimensions is None else (dimensions,)
    if result.ndim != 1 or len(result) not in allowed:
        expected = "2 or 3" if dimensions is None else str(dimensions)
        raise ValueError(f"{name} must be a {expected}-coordinate point")
    return result


def _direction(start: np.ndarray, end: np.ndarray) -> tuple[np.ndarray, float]:
    import numpy as np

    if start.shape != end.shape:
        raise ValueError("start and end must have the same dimension")
    delta = end - start
    length = float(np.linalg.norm(delta))
    if not math.isfinite(length) or length <= _EPS:
        raise ValueError("start and end must define a nonzero finite segment")
    return delta / length, length


def _polygon(value: object, name: str) -> np.ndarray:
    import numpy as np

    points = _array(value, name)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError(f"{name} must contain at least three ordered 2D vertices")
    edges = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(edges, axis=1)
    if not np.isfinite(lengths).all() or np.any(lengths <= _EPS):
        raise ValueError(f"{name} must not repeat vertices or contain zero-length edges")
    normals = np.column_stack((-edges[:, 1], edges[:, 0])) / lengths[:, None]
    distances = np.sum((points[None, :, :] - points[:, None, :]) * normals[:, None, :], axis=2)
    keep = np.ones((len(points), len(points)), dtype=bool)
    indices = np.arange(len(points))
    keep[indices, indices] = False
    keep[indices, (indices + 1) % len(points)] = False
    sides = distances[keep]
    if not (np.all(sides > _EPS) or np.all(sides < -_EPS)):
        raise ValueError(f"{name} must be strictly convex, ordered, and have nonzero area")
    return points


def _edge_normals(points: np.ndarray) -> np.ndarray:
    import numpy as np

    edges = np.roll(points, -1, axis=0) - points
    return np.column_stack((-edges[:, 1], edges[:, 0])) / np.linalg.norm(edges, axis=1)[:, None]


def _points_to_edges_distance(points: np.ndarray, polygon: np.ndarray) -> float:
    import numpy as np

    edges = np.roll(polygon, -1, axis=0) - polygon
    offsets = points[:, None, :] - polygon[None, :, :]
    fractions = np.sum(offsets * edges[None, :, :], axis=2) / np.sum(edges * edges, axis=1)
    closest = polygon[None, :, :] + np.clip(fractions, 0.0, 1.0)[:, :, None] * edges
    return float(np.linalg.norm(points[:, None, :] - closest, axis=2).min())


class Geometry:
    """Prebuilt calculations on points supplied by the policy; no sensing or motion.

    Supply fresh measured points in ONE coordinate frame and unit system.
    For robot targets use world metres; image pixels must first be converted
    to a measured world plane.  2D operations require an explicitly chosen
    plane, such as world XY, and do not check height or 3D collisions.
    Numerical distances within 1e-12 input units are treated as coincident.
    Inputs are never modified. Invalid or ambiguous inputs raise ValueError.
    """

    def closest_point_on_segment(
        self,
        point: Sequence[float],
        start: Sequence[float],
        end: Sequence[float],
    ) -> tuple[float, ...]:
        """Project a 2D or 3D point onto the finite segment from start to end.

        All three points must have the same dimension, frame and units.
        Returns the closest point, clamped to the segment's endpoints.
        Subtract the supplied point from the result to obtain a correction.
        Coincident endpoints raise ValueError. This is geometry, not an
        executable or collision-checked robot motion.
        """
        import numpy as np

        p, a, b = _point(point, "point"), _point(start, "start"), _point(end, "end")
        if p.shape != a.shape:
            raise ValueError("point, start and end must have the same dimension")
        direction, length = _direction(a, b)
        closest = a + float(np.clip(np.dot(p - a, direction), 0.0, length)) * direction
        return tuple(float(v) for v in closest)

    def perpendicular_toward(
        self,
        start: Sequence[float],
        end: Sequence[float],
        point: Sequence[float],
    ) -> tuple[float, float]:
        """Return a 2D unit normal to the infinite line, toward point's side.

        All points are 2D in one measured plane. Multiply the result by an
        offset in that plane's units to choose an approach or standoff.
        The reference point selects the side; it is not a target. Unlike
        segment projection, this uses the line extended beyond its ends.
        Coincident endpoints or a reference point on the line raise
        ValueError because they do not define a unique side.
        """
        import numpy as np

        a = _point(start, "start", 2)
        b = _point(end, "end", 2)
        p = _point(point, "point", 2)
        direction, _ = _direction(a, b)
        normal = np.array([-direction[1], direction[0]])
        side = float(np.dot(p - a, normal))
        if abs(side) <= _EPS:
            raise ValueError("point must lie off the line to select a perpendicular direction")
        normal *= 1.0 if side > 0 else -1.0
        return float(normal[0]), float(normal[1])

    def rotate_points_2d(
        self,
        points: Sequence[Sequence[float]],
        angle_deg: float,
        center: Sequence[float] | None = None,
    ) -> tuple[tuple[float, float], ...]:
        """Rotate N 2D points counterclockwise by angle_deg in their plane.

        The default center is the arithmetic mean of the supplied points,
        not the polygon's area centroid or a robot's grasp point. Supply
        center explicitly to rotate about another measured pivot.
        In world XY positive rotation is counterclockwise viewed from +z;
        this angle is NOT a general RoboDojo gripper yaw/pitch/roll command.
        Returns transformed coordinates without moving anything.
        """
        import numpy as np

        pts = _array(points, "points")
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) == 0:
            raise ValueError("points must be a nonempty N by 2 array")
        try:
            angle = float(angle_deg)
        except (TypeError, ValueError) as exc:
            raise ValueError("angle_deg must be finite") from exc
        if not math.isfinite(angle):
            raise ValueError("angle_deg must be finite")
        pivot = pts.mean(axis=0) if center is None else _point(center, "center", 2)
        radians = math.radians(angle % 360.0)
        cosine, sine = math.cos(radians), math.sin(radians)
        rotation = np.array([[cosine, -sine], [sine, cosine]])
        result = (pts - pivot) @ rotation.T + pivot
        return tuple((float(row[0]), float(row[1])) for row in result)

    def convex_polygons_intersect(
        self,
        polygon_a: Sequence[Sequence[float]],
        polygon_b: Sequence[Sequence[float]],
        clearance: float = 0.0,
    ) -> bool:
        """Whether two convex 2D outlines overlap, touch, or are too close.

        Each polygon has at least three strictly convex vertices ordered
        clockwise or counterclockwise; do not repeat the first vertex.
        Concave, crossing, collinear or duplicate-vertex inputs raise
        ValueError. Supply measured outlines.
        True means overlap, boundary contact, or minimum Euclidean boundary
        distance <= clearance (nonnegative, in the input points' units).
        This screens planar footprints only: it does not account for height,
        uncertain perception, the full robot, or a swept motion path. False
        is therefore not proof that a robot move is collision-free.
        """
        import numpy as np

        a, b = _polygon(polygon_a, "polygon_a"), _polygon(polygon_b, "polygon_b")
        try:
            margin = float(clearance)
        except (TypeError, ValueError) as exc:
            raise ValueError("clearance must be finite and nonnegative") from exc
        if not math.isfinite(margin) or margin < 0:
            raise ValueError("clearance must be finite and nonnegative")
        axes = np.concatenate((_edge_normals(a), _edge_normals(b)))
        a_proj, b_proj = a @ axes.T, b @ axes.T
        gaps = np.maximum(a_proj.min(axis=0) - b_proj.max(axis=0),
                          b_proj.min(axis=0) - a_proj.max(axis=0))
        if np.all(gaps <= _EPS):
            return True
        distance = min(_points_to_edges_distance(a, b), _points_to_edges_distance(b, a))
        return distance <= margin + _EPS


__all__ = ["Geometry"]
