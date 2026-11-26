"""Metric evidence and bounded SE(3) constraints from public calibration only."""

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def vector(value, size=3):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"Expected {size} finite coordinates")
    return result


def unit(value):
    result = vector(value)
    norm = np.linalg.norm(result)
    if norm < 1e-5:
        raise ValueError("Degenerate axis or plane")
    return result / norm


def triangulate(first, pixel_a, second, pixel_b):
    a, b = vector(pixel_a, 2), vector(pixel_b, 2)
    ray_a = unit(first.optical[:3, :3] @ np.linalg.solve(first.k, np.r_[a, 1]))
    ray_b = unit(second.optical[:3, :3] @ np.linalg.solve(second.k, np.r_[b, 1]))
    angle = np.arccos(np.clip(np.dot(ray_a, ray_b), -1, 1))
    if (
        np.linalg.norm(first.optical[:3, 3] - second.optical[:3, 3]) < 0.01
        or not 0.02 < angle < np.pi - 0.02
    ):
        raise ValueError("Insufficient camera baseline or triangulation parallax")
    point = cv2.triangulatePoints(
        first.projection, second.projection, a[:, None], b[:, None]
    )[:, 0]
    if not np.isfinite(point).all() or abs(point[3]) < 1e-8:
        raise ValueError("Triangulation is at infinity")
    point = point[:3] / point[3]
    errors = [
        float(np.linalg.norm(camera.project(point) - pixel))
        for camera, pixel in ((first, a), (second, b))
    ]
    if max(errors) > 2:
        raise ValueError("Multi-view points disagree (reprojection exceeds 2 pixels)")
    return {
        "position": point.tolist(),
        "method": "rgb_triangulation",
        "reprojection_error_px": errors,
        "parallax_rad": float(angle),
        "assumptions": ["User-selected pixels identify the same physical feature"],
    }


def upright_feature(camera, base_pixels, top_pixel, table):
    """Estimate a vertical centerline from visible, symmetric table contacts."""
    if len(base_pixels) != 2:
        raise ValueError("Use two symmetric visible base contacts on the table")
    contacts = [camera.on_plane(vector(p, 2), table) for p in base_pixels]
    width = float(np.linalg.norm(contacts[0] - contacts[1]))
    if not 0.01 <= width <= 0.4:
        raise ValueError("Base contact separation must be 0.01..0.4 m")
    base = np.mean(contacts, axis=0)
    pixel = vector(top_pixel, 2)
    ray = unit(camera.optical[:3, :3] @ np.linalg.solve(camera.k, np.r_[pixel, 1]))
    # Intersect the viewing ray with the vertical line through the base midpoint.
    system = np.column_stack((ray, [0, 0, -1]))
    distance, height = np.linalg.lstsq(
        system, base - camera.optical[:3, 3], rcond=None
    )[0]
    point = base + [0, 0, height]
    error = float(np.linalg.norm(camera.project(point) - pixel))
    if distance <= 0 or not 0.015 <= height <= 0.6 or error > 3:
        raise ValueError("Selected top is inconsistent with the upright base axis")
    return {
        "position": point.tolist(),
        "method": "assumed_upright_axis",
        "base_center": base.tolist(),
        "height_above_table_m": float(height),
        "base_contact_separation_m": width,
        "reprojection_error_px": error,
        "assumptions": [
            "Base clicks are symmetric physical contact points on the public table",
            "The selected feature is on the upright object's vertical centerline",
            "Not valid after lifting, tilting or losing the base contacts",
        ],
        "sensor_ground_truth": False,
    }


def relative_pose(initial, translation, rotation_vector, pivot_local):
    translation, rotation_vector, pivot_local = map(
        vector, (translation, rotation_vector, pivot_local)
    )
    if np.linalg.norm(translation) > 0.15 + 1e-9:
        raise ValueError("Relative translation is bounded to 0.15 m")
    if np.linalg.norm(rotation_vector) > 0.6 + 1e-9:
        raise ValueError("Relative rotation is bounded to 0.6 radians")
    if np.linalg.norm(pivot_local) > 0.3 + 1e-9:
        raise ValueError("Local pivot must be within 0.3 m of the grasp center")
    result = initial.copy()
    result[:3, :3] = Rotation.from_rotvec(rotation_vector).as_matrix() @ initial[:3, :3]
    result[:3, 3] = (
        initial[:3, 3] + translation + (initial[:3, :3] - result[:3, :3]) @ pivot_local
    )
    return result


def solve_pose(initial, constraints):
    """Point coincidence, directed-axis alignment and point-on-plane constraints.

    Preserve the nearest initial pose in unconstrained degrees of freedom. This
    resolves geometry only; IK/path clearance and observation freshness are separate.
    """
    initial = np.asarray(initial, dtype=float)
    if initial.shape != (4, 4) or not np.isfinite(initial).all():
        raise ValueError("Expected a finite TCP transform")
    if not isinstance(constraints, list) or not 1 <= len(constraints) <= 8:
        raise ValueError("Use 1..8 geometric constraints")
    parsed = []
    for row in constraints:
        kind = row.get("kind")
        allowed = (
            {"kind", "local", "target"}
            if kind in {"point", "axis"}
            else {"kind", "local", "origin", "normal"}
        )
        if kind not in {"point", "axis", "plane"} or set(row) != allowed:
            raise ValueError("Invalid geometric constraint fields")
        parsed.append(
            {
                key: (
                    value
                    if key == "kind"
                    else (
                        unit(value)
                        if key == "normal" or kind == "axis"
                        else vector(value)
                    )
                )
                for key, value in row.items()
            }
        )

    def transform(x):
        return (
            Rotation.from_rotvec(x[3:]).as_matrix() @ initial[:3, :3],
            initial[:3, 3] + x[:3],
        )

    def errors(x):
        rotation, position = transform(x)
        rows = []
        for row in parsed:
            if row["kind"] == "point":
                rows.append(
                    (rotation @ row["local"] + position - row["target"]) / 0.003
                )
            elif row["kind"] == "axis":
                rows.append((rotation @ row["local"] - row["target"]) / 0.03)
            else:
                rows.append(
                    np.atleast_1d(
                        np.dot(
                            rotation @ row["local"] + position - row["origin"],
                            row["normal"],
                        )
                    )
                    / 0.003
                )
        return np.concatenate(rows)

    seed = np.zeros(6)
    axes = [row for row in parsed if row["kind"] == "axis"]
    if axes:
        aligned, _ = Rotation.align_vectors(
            np.asarray([row["target"] for row in axes]),
            np.asarray([initial[:3, :3] @ row["local"] for row in axes]),
        )
        seed[3:] = np.clip(aligned.as_rotvec(), -np.pi + 1e-6, np.pi - 1e-6)
    points = [row for row in parsed if row["kind"] == "point"]
    if points:
        rotation, _ = transform(seed)
        seed[:3] = np.clip(
            np.mean(
                [
                    row["target"] - rotation @ row["local"] - initial[:3, 3]
                    for row in points
                ],
                axis=0,
            ),
            -0.399,
            0.399,
        )
    result = least_squares(
        lambda x: np.r_[errors(x), x * 0.01],
        seed,
        bounds=([-0.4] * 3 + [-np.pi] * 3, [0.4] * 3 + [np.pi] * 3),
        max_nfev=200,
    )
    residual = errors(result.x)
    if not result.success or max(abs(residual)) > 1:
        raise ValueError(
            "Geometric constraints are inconsistent or outside bounded pose search"
        )
    rotation, position = transform(result.x)
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = rotation, position
    return pose, {
        "maximum_normalized_residual": float(max(abs(residual))),
        "position_tolerance_m": 0.003,
        "axis_component_tolerance": 0.03,
        "unconstrained_axes": "nearest initial pose; not a uniqueness guarantee",
        "scope": "geometric fit, not collision clearance or task success",
    }


def verify_pose(actual, target):
    position = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
    angle = float(Rotation.from_matrix(actual[:3, :3].T @ target[:3, :3]).magnitude())
    return {
        "status": "pass" if position <= 0.005 and angle <= 0.06 else "fail",
        "position_error_m": position,
        "orientation_error_rad": angle,
        "scope": "observed robot TCP pose only; not grasp, placement or official success",
    }
