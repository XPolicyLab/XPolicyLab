"""Nominal public-finger projections on current RGB, never contact evidence."""

import hashlib
from pathlib import Path

import cv2
import numpy as np

from .finger_model import MODEL


def _check_files(parameters):
    files = {}
    for key in ("robot_urdf", "robot_config"):
        path = Path(parameters[key])
        if not path.is_file() or path.stat().st_size > 1_000_000:
            raise ValueError("Pinned public robot calibration is absent or oversized")
        if hashlib.sha256(path.read_bytes()).hexdigest() != MODEL[key + "_sha256"]:
            raise ValueError("Nominal finger preview needs its pinned public X5 calibration")
        files[key] = path
    sources = {}
    for finger in MODEL["fingers"].values():
        path = files["robot_urdf"].parent / finger["source"]
        if not path.is_file() or path.stat().st_size > 1_000_000:
            raise ValueError("Pinned public finger mesh is absent or oversized")
        if hashlib.sha256(path.read_bytes()).hexdigest() != finger["sha256"]:
            raise ValueError("Public finger mesh changed; preview needs revalidation")
        sources[finger["source"]] = finger["sha256"]
    return sources


def availability(parameters):
    try:
        _check_files(parameters.get("geometry", {}))
    except (OSError, ValueError, KeyError, TypeError) as error:
        return {"available": False, "reason": str(error)}
    return {
        "available": True,
        "reason": "Pinned public X5 finger-limit preview only; no actual jaw/contact measurement",
    }


def recognizes_calibration(geometry):
    hashes = getattr(geometry, "hashes", {})
    return all(hashes.get(key) == MODEL[key + "_sha256"] for key in ("robot_urdf", "robot_config"))


def validate_resources(geometry):
    hashes = getattr(geometry, "hashes", {})
    for key in ("robot_urdf", "robot_config"):
        if hashes.get(key) != MODEL[key + "_sha256"]:
            raise ValueError("Nominal finger preview needs its pinned public X5 calibration")
    if abs(float(geometry.tip_offset) - MODEL["tcp_offset_m"]) > 1e-9:
        raise ValueError("Nominal finger model and public TCP offset disagree")
    return _check_files(geometry.resources)


def _clip_convex_vertices(points, near=0.005):
    """Clip the nominal convex model for numerical display, not sensor depth.

    All true hull edges are among the vertex pairs. Extra intersections lie
    inside the clipped convex hull and do not enlarge its projected hull.
    """
    inside = points[:, 2] >= near
    if not inside.any():
        return None, int(len(points))
    if inside.all():
        return points, 0
    a, b = points[inside][:, None, :], points[~inside][None, :, :]
    fraction = (near - a[:, :, 2]) / (b[:, :, 2] - a[:, :, 2])
    intersections = a + fraction[:, :, None] * (b - a)
    return np.vstack([points[inside], intersections.reshape(-1, 3)]), int((~inside).sum())


def render(image, camera, tcp_pose):
    """Draw planned finger-limit poses in the ORIGINAL, CURRENT camera frame.

    A wrist camera will move during execution; this is not its future image.
    No observed gripper scalar is used to infer actual finger positions.
    """
    image = np.asarray(image)
    pose = np.asarray(tcp_pose, dtype=float)
    if (
        image.dtype != np.uint8
        or image.shape != (camera.height, camera.width, 3)
        or pose.shape != (4, 4)
        or not np.isfinite(pose).all()
        or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
        or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6, rtol=0)
        or abs(np.linalg.det(pose[:3, :3]) - 1) > 1e-6
    ):
        raise ValueError("Preview needs original RGB dimensions and a finite rigid TCP pose")
    wrist = pose.copy()
    wrist[:3, 3] -= wrist[:3, 0] * MODEL["tcp_offset_m"]
    models, polygons = [], []
    visible_models = 0
    for name, finger in MODEL["fingers"].items():
        for label, q in zip(("closed_limit", "open_limit"), finger["limits"], strict=True):
            local = (
                np.asarray(finger["vertices"], dtype=float)
                + np.asarray(finger["origin"], dtype=float)
                + float(q) * np.asarray(finger["axis"], dtype=float)
            )
            world = local @ wrist[:3, :3].T + wrist[:3, 3]
            optical = np.c_[world, np.ones(len(world))] @ camera.view.T
            if not np.isfinite(optical).all():
                return {
                    "status": "unknown", "reason": "Nominal optical coordinates are nonfinite",
                    "official_success": None,
                }, None
            visible, clipped = _clip_convex_vertices(optical[:, :3])
            if visible is None:
                return {
                    "status": "unknown",
                    "reason": "Nominal finger model lies behind the numerical preview clipping plane",
                    "official_success": None,
                }, None
            pixels = visible @ camera.k.T
            pixels = pixels[:, :2] / pixels[:, 2, None]
            if not np.isfinite(pixels).all() or np.abs(pixels).max() > 1e6:
                return {
                    "status": "unknown", "reason": "Nominal projection is unbounded",
                    "official_success": None,
                }, None
            low, high = pixels.min(axis=0), pixels.max(axis=0)
            intersects = bool(
                high[0] >= 0 and high[1] >= 0
                and low[0] < camera.width and low[1] < camera.height
            )
            visible_models += int(intersects)
            models.append({
                "finger": name, "joint": finger["joint"],
                "nominal_limit": label, "nominal_slide_m": float(q),
                "vertices_clipped_for_display": clipped,
                "unclipped_pixel_bbox": [*low.tolist(), *high.tolist()],
                "intersects_image": intersects,
                "partly_outside_image": bool(
                    low[0] < 0 or low[1] < 0
                    or high[0] >= camera.width or high[1] >= camera.height
                ),
            })
            if intersects:
                polygons.append((label, cv2.convexHull(np.round(pixels).astype(np.int32))))
    if not visible_models:
        return {
            "status": "unknown", "reason": "Nominal fingers are outside the selected current view",
            "models": models, "official_success": None,
        }, None
    preview = image.copy()
    for label, polygon in polygons:
        color = (255, 140, 0) if label == "closed_limit" else (0, 200, 255)
        cv2.polylines(preview, [polygon], True, color, 1, cv2.LINE_AA)
    cv2.putText(
        preview, "NOMINAL fingers: cyan=open limit; orange=closed limit",
        (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 0), 1, cv2.LINE_AA,
    )
    cv2.putText(
        preview, "CURRENT-camera projection, NOT actual grip/contact/future view",
        (6, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 0), 1, cv2.LINE_AA,
    )
    return {
        "status": "nominal_projection",
        "models": models,
        "image_dimensions": [camera.width, camera.height],
        "numerical_display_clip_m": 0.005,
        "actual_gripper_pose_measured": False,
        "object_geometry_measured": False,
        "contact_verified": False,
        "collision_clearance_verified": False,
        "official_success": None,
        "scope": (
            "Pinned public finger visual convex hulls at a planned TCP pose and "
            "URDF slide limits, projected into the selected CURRENT camera. "
            "Not actual finger positions, the complete hand/wrist, a future wrist "
            "image, object localization, occlusion/contact/clearance or retention "
            "proof. Limits assume empty space; real contact can stop fingers "
            "between them. Projection overlap can involve different depths. Inspect "
            "original RGB and retain every geometry/motion gate; never mark "
            "synthetic lines as observed material."
        ),
    }, preview
