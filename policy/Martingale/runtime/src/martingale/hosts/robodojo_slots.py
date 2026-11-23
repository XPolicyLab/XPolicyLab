# Copyright © 2026 Martingale. All Rights Reserved.
from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping

import numpy as np

from .robodojo_sight import View
from .robodojo_slot_fit import CONFIG, fit

CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")


def _validate_features(features: Mapping) -> tuple[dict, set[str]]:
    if not isinstance(features, Mapping):
        raise ValueError("features must be a mapping of visible rectangle pixels")
    rectangles = features.get("rectangles")
    if not isinstance(rectangles, Mapping):
        raise ValueError("features.rectangles is required")
    divider = features.get("divider_edges", {})
    if set(rectangles) == {"outer"}:
        if not divider:
            raise ValueError("An outer rim requires visible divider_edges to identify both slots")
    elif set(rectangles) != {"slot_left", "slot_right"}:
        raise ValueError("Supply both slot rectangles, or outer plus visible divider_edges")
    cameras: set[str] = set()
    for name, observations in rectangles.items():
        if not isinstance(observations, Mapping) or "cam_head" not in observations:
            raise ValueError(f"{name} needs a head-view quadrilateral")
        if len(observations) < 2:
            raise ValueError(f"{name} needs head plus at least one wrist view")
        for camera, corners in observations.items():
            if camera not in CAMERAS:
                raise ValueError(f"Unknown camera {camera!r}")
            pixels = np.asarray(corners, dtype=float)
            if (pixels.shape != (4, 2) or not np.isfinite(pixels).all()
                    or np.any(pixels < 0) or np.any(pixels[:, 0] >= 640)
                    or np.any(pixels[:, 1] >= 480)):
                raise ValueError("Each rectangle needs four visible original-image (column,row) points")
            cameras.add(camera)
    if "cam_head" not in cameras or not any(c != "cam_head" for c in cameras):
        raise ValueError("At least one supporting wrist camera is required")
    for camera, edges in divider.items():
        if camera not in cameras or set(edges) != {"left", "right"}:
            raise ValueError("Each divider view needs both long edges and a matching rectangle view")
        for points in edges.values():
            pixels = np.asarray(points, dtype=float)
            if (pixels.ndim != 2 or pixels.shape[1] != 2 or len(pixels) < 2
                    or not np.isfinite(pixels).all() or np.any(pixels < 0)
                    or np.any(pixels[:, 0] >= 640) or np.any(pixels[:, 1] >= 480)):
                raise ValueError("Divider edges need visible original-image (column,row) points")
    if "outer" in rectangles and set(divider) != cameras:
        raise ValueError("Observe both divider long edges in every fitted camera")
    return {"rectangles": dict(rectangles), "divider_edges": dict(divider),
            "coplanar": bool(features.get("coplanar", True))}, cameras


def slots_from_rectangles(rectangles: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Construct left/right polygons when one fitted outer rim and divider are given."""
    if "outer" not in rectangles:
        return {key: np.asarray(value, dtype=float) for key, value in rectangles.items()}
    outer = np.asarray(rectangles["outer"], dtype=float)
    divider = np.asarray(rectangles["divider"], dtype=float)
    u = (outer[2] + outer[3] - outer[0] - outer[1]) / 2
    u[2] = 0
    u /= np.linalg.norm(u)
    v = np.array([-u[1], u[0], 0.])
    if v @ (outer[1] - outer[0]) < 0:
        v = -v
    lo, hi = min(outer @ u), max(outer @ u)
    cuts = [min(outer @ v), min(divider @ v), max(divider @ v), max(outer @ v)]
    z = float(outer[:, 2].mean())
    return {
        name: np.array([a * u + b * v + [0., 0., z]
                        for a, b in ((lo, left), (lo, right), (hi, right), (hi, left))])
        for name, left, right in (("slot_left", cuts[0], cuts[1]),
                                  ("slot_right", cuts[2], cuts[3]))
    }


def fit_slots(robot, features: Mapping, *, step: int = -1) -> dict:
    """Fit slot rim geometry from corresponding RGB pixel annotations.

    `step` is forwarded identically to state, image and calibration reads.
    The result includes the resolved recorded step and RGB hashes for provenance.
    """
    annotations, cameras = _validate_features(features)
    state = robot.state(step)
    actual_step = int(state.step)
    metadata = {camera: robot.camera_meta(camera, step=actual_step) for camera in sorted(cameras)}
    images = {camera: np.asarray(robot.image(camera, step=actual_step)) for camera in sorted(cameras)}
    for camera, image in images.items():
        if image.shape != (480, 640, 3) or image.dtype != np.uint8:
            raise ValueError(f"{camera} step {actual_step} is not native uint8 RGB")
    views = {camera: View(metadata[camera]["intrinsic_matrix"],
                          metadata[camera]["extrinsic_matrix"],
                          shape=tuple(metadata[camera].get("shape", (480, 640, 3))[:2]))
             for camera in cameras}
    started = time.perf_counter()
    fitted = fit(annotations, views)
    polygons = slots_from_rectangles(fitted["corners"])
    reasons = []
    per_view = fitted["reprojection_per_view"]
    if not fitted["success"] or fitted["jacobian_rank"] < fitted["parameter_count"]:
        reasons.append("fit did not converge with fully constrained parameters")
    if fitted["reprojection_rms_px"] > 2 or max(per_view.values(), default=float("inf")) > 3:
        reasons.append("image disagreement; inspect correspondences or obtain a clearer view")
    center = np.mean(np.concatenate(list(polygons.values())), axis=0)
    directions = {camera: (view.pos - center) / np.linalg.norm(view.pos - center)
                  for camera, view in views.items()}
    parallax = max(float(np.rad2deg(np.arccos(np.clip(directions["cam_head"] @ directions[camera], -1, 1))))
                   for camera in cameras if camera != "cam_head")
    if parallax < 10:
        reasons.append("insufficient viewpoint separation")
    slots = {}
    for name, polygon in sorted(polygons.items(), key=lambda item: item[1][:, 0].mean()):
        along = (polygon[2] + polygon[3] - polygon[0] - polygon[1]) / 2
        across = (polygon[1] + polygon[2] - polygon[0] - polygon[3]) / 2
        length, width = float(np.linalg.norm(along)), float(np.linalg.norm(across))
        if not 0.003 < width < min(length, 0.09):
            reasons.append(f"{name} dimensions implausible")
        angle = float(np.rad2deg(np.arctan2(along[1], along[0])) % 180)
        slots[name] = {"center_xyz": polygon.mean(axis=0).tolist(),
                       "rim_z_m": float(polygon[:, 2].mean()),
                       "corners_xyz": polygon.tolist(), "length_m": length,
                       "width_m": width, "long_axis_deg": angle,
                       "cross_axis_deg": (angle + 90) % 180}
    return {
        "accepted": not reasons, "reasons": reasons, "step": actual_step,
        "cameras": sorted(cameras), "rgb_sha256": {
            camera: hashlib.sha256(images[camera].tobytes()).hexdigest() for camera in images},
        "slots": slots, "rim_geometry": fitted["corners"],
        "reprojection_rms_px": fitted["reprojection_rms_px"],
        "reprojection_per_view_px": per_view, "view_separation_deg": parallax,
        "elapsed_s": time.perf_counter() - started,
        "priors": {"coplanar": annotations["coplanar"], "fixed_height_baseline_m": CONFIG["fixed_height_baseline_m"],
                   "height_bounds_m": [CONFIG["minimum_height_m"], CONFIG["maximum_height_m"]],
                   "length_bounds_m": CONFIG["length_bounds_m"], "width_bounds_m": CONFIG["width_bounds_m"]},
        "note": "Rim geometry only; slot bottom, held bread pose, seating depth and insertion success are not measured.",
    }


fit_slots._readonly = True
