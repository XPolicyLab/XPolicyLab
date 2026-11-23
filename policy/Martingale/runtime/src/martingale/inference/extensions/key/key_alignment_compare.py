# Copyright © 2026 Martingale. All Rights Reserved.
import math

import numpy as np


CAMERAS = {"cam_head", "cam_left_wrist", "cam_right_wrist"}


def _vector(value, size, name):
    a = np.asarray(value, dtype=float)
    if a.shape != (size,) or not np.isfinite(a).all():
        raise ValueError(f"{name} must contain {size} finite numbers")
    return a.copy()


def _unit(value, name):
    a = _vector(value, 3, name)
    if np.linalg.norm(a) < 1e-9:
        raise ValueError(f"{name} must be nonzero")
    return a / np.linalg.norm(a)


def _meta(meta):
    k = np.asarray(meta["intrinsic_matrix"], dtype=float)
    pose = np.asarray(meta["extrinsic_matrix"], dtype=float)
    if (k.shape != (3, 3) or pose.shape != (4, 4)
            or not np.isfinite(k).all() or not np.isfinite(pose).all()
            or k[0, 0] <= 0 or k[1, 1] <= 0
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5)
            or np.linalg.det(pose[:3, :3]) < .99):
        raise ValueError("Invalid camera calibration")
    return k, pose


def project(meta, point):
    """Camera-to-world convention: local -z forward, +y up."""
    k, pose = _meta(meta)
    x, y, z = pose[:3, :3].T @ (_vector(point, 3, "point") - pose[:3, 3])
    if z >= -1e-8:
        return None
    return np.array([k[0, 2] + k[0, 0]*x/-z, k[1, 2] - k[1, 1]*y/-z])


def _ray(meta, pixel):
    k, pose = _meta(meta)
    u, v = _vector(pixel, 2, "pixel")
    h, w = meta["shape"][:2]
    if not (0 <= u < w and 0 <= v < h):
        raise ValueError("Annotation is outside the current image")
    ray = pose[:3, :3] @ np.array([(u-k[0, 2])/k[0, 0], -(v-k[1, 2])/k[1, 1], -1.])
    return pose[:3, 3], ray / np.linalg.norm(ray)


def _basis(normal):
    along = np.eye(3)[np.argmin(abs(normal))]
    along -= normal * (normal @ along)
    along /= np.linalg.norm(along)
    return np.column_stack([along, np.cross(normal, along)])


def _triangulate(rays):
    matrices = [np.eye(3)-np.outer(d, d) for _, d in rays]
    system = sum(matrices)
    if np.linalg.cond(system) > 1e5:
        raise ValueError("Insufficient ray parallax")
    point = np.linalg.solve(system, sum(m @ o for m, (o, _) in zip(matrices, rays)))
    if any((point-o) @ d <= 0 for o, d in rays):
        raise ValueError("Reconstruction is behind a camera")
    residual = max(float(np.linalg.norm(m @ (point-o))) for m, (o, _) in zip(matrices, rays))
    angle = min(float(np.degrees(np.arccos(np.clip(abs(a[1] @ b[1]), -1, 1))))
                for i, a in enumerate(rays) for b in rays[i+1:])
    return point, residual, angle


def _axis_system(metas, pixels, entry, normal):
    basis = _basis(normal)
    rows, rhs, views = [], [], []
    for meta, pixel in zip(metas, pixels):
        origin, ray = _ray(meta, pixel)
        plane_normal = np.cross(ray, normal)
        if np.linalg.norm(plane_normal) < .05:
            raise ValueError("View is too close to the insertion-axis direction")
        plane_normal /= np.linalg.norm(plane_normal)
        rows.append(plane_normal @ basis)
        rhs.append(plane_normal @ (origin-entry))
        p0, p1 = project(meta, entry), project(meta, entry+.03*normal)
        if p0 is None or p1 is None or np.linalg.norm(p1-p0) < 2:
            raise ValueError("Projected insertion axis is unresolved")
        direction = (p1-p0) / np.linalg.norm(p1-p0)
        image_normal = np.array([-direction[1], direction[0]])
        signed_px = float(image_normal @ (pixel-p0))
        views.append({"axis_pixels": [p0.tolist(), p1.tolist()],
                      "signed_tip_axis_px": signed_px,
                      "tip_axis_distance_px": abs(signed_px),
                      "transverse_plane_error_mm": float(rhs[-1]*1000)})
    matrix = np.asarray(rows)
    singular = np.linalg.svd(matrix, compute_uv=False)
    rank = int(np.linalg.matrix_rank(matrix, tol=.05))
    condition = float(singular[0]/singular[-1]) if len(singular) > 1 else None
    if len(rows) > 1 and (rank < 2 or condition > 20):
        raise ValueError("Views do not resolve independent lateral directions")
    lateral = basis @ np.linalg.lstsq(matrix, np.asarray(rhs), rcond=None)[0]
    return lateral, rank, condition, views


def _quat_rotation(q):
    q = _vector(q, 4, "eef_quat_wxyz")
    if np.linalg.norm(q) < 1e-9:
        raise ValueError("Invalid hand quaternion")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def tip_fixed_rotation(tcp_xyz, eef_quat_wxyz, tip_xyz, shaft_axis_xyz,
                       normal_xyz, *, max_tilt_deg=2.):
    """Propose a capped world rotation about a MEASURED tip, preserving its xyz.

    Assumes a rigid retained grasp during the move. Return values are only a
    checked move_to proposal; inspect grip/clearance and fresh images afterward.
    """
    if not math.isfinite(max_tilt_deg) or not 0 < max_tilt_deg <= 3:
        raise ValueError("max_tilt_deg must be in (0, 3]")
    tcp, tip = _vector(tcp_xyz, 3, "tcp"), _vector(tip_xyz, 3, "tip")
    axis, normal = _unit(shaft_axis_xyz, "shaft axis"), _unit(normal_xyz, "normal")
    angle = float(np.arccos(np.clip(axis @ normal, -1, 1)))
    if axis @ normal <= 0:
        raise ValueError("Shaft station is not above the terminal tip")
    applied = min(angle, math.radians(max_tilt_deg))
    turn_axis = np.cross(axis, normal)
    if np.linalg.norm(turn_axis) < 1e-9:
        delta = np.eye(3)
    else:
        x, y, z = turn_axis / np.linalg.norm(turn_axis)
        skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        delta = np.eye(3)+math.sin(applied)*skew+(1-math.cos(applied))*(skew @ skew)
    rotation = delta @ _quat_rotation(eef_quat_wxyz)
    target = tip + delta @ (tcp-tip)
    cy = math.hypot(rotation[0, 0], rotation[1, 0])
    pitch = math.atan2(-rotation[2, 0], cy)
    if cy > 1e-8:
        heading, roll = math.atan2(rotation[1, 0], rotation[0, 0]), math.atan2(rotation[2, 1], rotation[2, 2])
    else:
        heading, roll = math.atan2(-rotation[0, 1], rotation[1, 1]), 0.
    return {"target_xyz": target.tolist(), "target_rotation_matrix": rotation.tolist(),
            "rotation_deg": math.degrees(applied), "tilt_before_deg": math.degrees(angle),
            "tcp_compensation_m": float(np.linalg.norm(target-tcp)),
            "pitch": math.degrees(pitch), "yaw": (math.degrees(heading)-90+180) % 360-180,
            "roll": math.degrees(roll), "assumes_retained_grasp": True}


def suggest(robot, opening, tip_pixels, *, arm, observed_step,
            shaft_pixels=None, max_translation_m=.001, max_tilt_deg=2.):
    """Return a bounded suggestion from freshly inspected RGB annotations.

    Any one or two existing cameras are supported, including head+holding wrist.
    `annotation_verified` is always False: the caller identifies physical points.
    `resolved_components=1` leaves the other lateral direction UNRESOLVED. A
    missing suggestion never forbids ordinary checked image-guided recovery.
    There is no motion, camera relocation, persistent state, or hidden-pose read.
    """
    if arm not in ("left", "right"):
        raise ValueError("arm must be left or right")
    if not math.isfinite(max_translation_m) or not 0 < max_translation_m <= .002:
        raise ValueError("max_translation_m must be in (0, .002]")
    state = robot.state()
    result = {"observed_step": observed_step, "annotation_verified": False,
              "geometry_resolved": False, "resolved_components": 0,
              "lateral_correction_xyz": None, "tip_xyz": None, "tip_height_m": None,
              "tilt_proposal": None, "views": {}, "reasons": [],
              "descent_certified": False,
              "note": "Geometric consistency does not verify landmark identity, opening calibration, grip or entry. Single-view centering leaves another direction unresolved."}
    if type(observed_step) is not int or observed_step != state.step:
        result["reasons"].append("stale_observation")
        return result
    names = sorted(tip_pixels)
    if len(names) not in (1, 2) or not set(names) <= CAMERAS:
        raise ValueError("Annotate the same tip in one or two existing cameras")
    if shaft_pixels is not None and set(shaft_pixels) != set(names):
        raise ValueError("Optional shaft pixels must match the tip cameras")
    entry, normal = _vector(opening["entry_xyz"], 3, "entry"), _unit(opening["normal_xyz"], "normal")
    metas = [robot.camera_meta(name, step=observed_step) for name in names]
    pixels = [_vector(tip_pixels[name], 2, "tip pixel") for name in names]
    try:
        lateral, rank, condition, views = _axis_system(metas, pixels, entry, normal)
        result["views"] = dict(zip(names, views))
        result["lateral_condition"] = condition
        if len(names) == 2:
            tip, residual, parallax = _triangulate([_ray(m, p) for m, p in zip(metas, pixels)])
            result.update(tip_ray_residual_m=residual, tip_ray_parallax_deg=parallax)
            if residual > .002 or parallax < 8:
                raise ValueError("Tip rays disagree (>2 mm) or have weak parallax (<8 degrees)")
            result["tip_xyz"] = tip.tolist()
            result["tip_height_m"] = float((tip-entry) @ normal)
        sensitivity = 0.
        for i in range(len(names)):
            for axis in range(2):
                changes = []
                for sign in (-1, 1):
                    changed = [p.copy() for p in pixels]
                    changed[i][axis] += sign
                    try:
                        estimate = _axis_system(metas, changed, entry, normal)[0]
                        changes.append(float(np.linalg.norm(estimate-lateral)))
                    except ValueError:
                        changes.append(math.inf)
                sensitivity += max(changes)
        correction = -lateral
        norm = float(np.linalg.norm(correction))
        if norm > max_translation_m:
            correction *= max_translation_m/norm
        result.update(geometry_resolved=rank == 2, resolved_components=rank,
                      lateral_error_m=norm, lateral_correction_xyz=correction.tolist(),
                      lateral_pixel_sensitivity_m=sensitivity,
                      error_exceeds_pixel_sensitivity=norm > sensitivity)
        if rank == 1:
            result["reasons"].append("Only one lateral component observed; tip height and other lateral component remain unresolved")
    except ValueError as exc:
        result["reasons"].append(str(exc))
        return result
    if shaft_pixels is not None and len(names) == 2 and result["tip_xyz"] is not None:
        try:
            shafts = [_vector(shaft_pixels[name], 2, "shaft pixel") for name in names]
            spans = [float(np.linalg.norm(s-p)) for s, p in zip(shafts, pixels)]
            result["shaft_span_pixels"] = dict(zip(names, spans))
            if min(spans) < 12:
                raise ValueError("Shaft too short in an image for tilt; tip centering remains usable")
            shaft, residual, parallax = _triangulate([_ray(m, p) for m, p in zip(metas, shafts)])
            tip = np.asarray(result["tip_xyz"])
            if residual > .002 or parallax < 8 or not .01 <= np.linalg.norm(shaft-tip) <= .075:
                raise ValueError("Shaft geometry is inconsistent; tip centering remains usable")
            hand = state.arm(arm)
            result["tilt_proposal"] = tip_fixed_rotation(hand.tcp_pos, hand.eef_quat, tip, shaft-tip,
                                                       normal, max_tilt_deg=max_tilt_deg)
        except ValueError as exc:
            result["reasons"].append(str(exc))
    return result
