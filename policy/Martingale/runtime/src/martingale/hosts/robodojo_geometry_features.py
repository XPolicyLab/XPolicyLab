# Copyright © 2026 Martingale. All Rights Reserved.
from __future__ import annotations

import numpy as np

CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
IMAGE_SHAPE = (480, 640)


def _views(primitives, obs, image_state, cameras):
    from . import robodojo_tools as tools
    views = {}
    for camera in cameras:
        cam = tools._camera(obs, camera)
        if cam is None or cam.get("color") is None:
            continue
        color = np.asarray(cam["color"])
        if color.shape != (*IMAGE_SHAPE, 3):
            continue
        view = tools._view(camera, image_state)
        if view is not None:
            views[camera] = view
    return views


def _image_step(primitives):
    status = primitives.env.get_status()
    return max(0, int(status.get("step", 0)) - int(status.get("pictures_behind") or 0))


def fit_pan(primitives, text_prompt="dustpan", yaw_hint=None):
    """Fit a table-supported pan from explicit RGB/SAM observations.

    Segment the named pan in each available head/wrist RGB view, mark robot
    pixels unknown in `Sight`, then invoke the pure table-supported silhouette
    fitter. Its donor prior fixes the table at `kin.TABLE_Z` and assumes a flat
    pan with a generic 3 mm floor shell; those are reported assumptions.
    """
    if not isinstance(text_prompt, str) or not text_prompt.strip():
        return {"kind": "pan", "accepted": False, "reason": "text_prompt must be nonempty"}
    from . import robodojo_sight as sight
    from . import robodojo_tools as tools
    from . import robodojo_sweep_pan as pan_fit

    obs = tools._obs(primitives)
    image_state = tools._shown(obs)
    source_step = _image_step(primitives)
    segmenter = getattr(primitives, "sam3_client", None)
    if segmenter is None:
        return {"kind": "pan", "accepted": False, "reason": "sam3_client not configured",
                "source_step": source_step}
    views = _views(primitives, obs, image_state, CAMERAS)
    observations = []
    for camera, view in views.items():
        color = np.asarray(tools._camera(obs, camera)["color"])
        response = segmenter.segment(color, text_prompt=text_prompt, min_score=.2)
        if not getattr(response, "found", False):
            continue
        mask = np.asarray(response.mask, dtype=bool)
        if mask.shape != IMAGE_SHAPE:
            return {"kind": "pan", "accepted": False,
                    "reason": f"{camera}: segmentation mask must be 480x640",
                    "source_step": source_step}
        robot = sight.robot_pixels(view, image_state, own=tools._WRISTS.get(camera))
        if np.asarray(robot).shape != IMAGE_SHAPE:
            return {"kind": "pan", "accepted": False,
                    "reason": f"{camera}: robot mask must be 480x640",
                    "source_step": source_step}
        observations.append(sight.Sight(view, mask, unknown=robot))
    if len(observations) < 2:
        return {"kind": "pan", "accepted": False,
                "reason": "need at least two synchronized RGB views",
                "source_step": source_step, "supporting_cameras": [
                    camera for camera in views]}
    result = pan_fit.fit(observations, yaw_hint=yaw_hint)
    result["source_step"] = source_step
    result["image_state_step"] = source_step
    result["supporting_cameras"] = [camera for camera, view in views.items()
                                    if any(ob.view is view for ob in observations)]
    result["pixel_convention"] = "original RGB; segmentation masks in row,column array order"
    result["table_prior"] = "kin.TABLE_Z; table-supported flat pan; generic 3 mm shell"
    return result


def fit_peg_landmarks(primitives, root_pixels, tip_pixels, radius_m=None, measured_step=None):
    """Fit peg axis from paired original-image `(row,column)` endpoint pixels.

    Pixel mappings are keyed by camera name and must contain root and tip points
    for the same visible endpoints. `measured_step` is the image step at which
    those annotations were made; stale annotations are rejected.
    """
    from . import robodojo_tools as tools
    from . import robodojo_peg_geometry as peg_fit

    if measured_step is None:
        return {"accepted": False, "reason": "measured_step is required",
                "pixel_convention": "(row,column)"}
    try:
        parsed_step = int(measured_step)
        if isinstance(measured_step, (bool, np.bool_)) or float(measured_step) != parsed_step:
            raise ValueError
        measured_step = parsed_step
    except (TypeError, ValueError, OverflowError):
        return {"accepted": False, "reason": "measured_step must be an integer"}
    obs = tools._obs(primitives)
    image_state = tools._shown(obs)
    source_step = _image_step(primitives)
    if measured_step != source_step:
        return {"accepted": False, "reason": "peg landmarks are stale for current displayed RGB",
                "measured_step": measured_step, "source_step": source_step,
                "pixel_convention": "(row,column)"}
    if not isinstance(root_pixels, dict) or not isinstance(tip_pixels, dict):
        return {"accepted": False, "reason": "root_pixels and tip_pixels must map camera names to pixels"}
    if set(root_pixels) != set(tip_pixels) or len(root_pixels) < 2:
        return {"accepted": False, "reason": "need matching root and tip pixels from at least two cameras"}
    cameras = list(root_pixels)
    if any(camera not in CAMERAS for camera in cameras):
        return {"accepted": False, "reason": "unknown camera in peg landmarks"}
    views = _views(primitives, obs, image_state, cameras)
    if set(views) != set(cameras):
        return {"accepted": False, "reason": "each landmark camera must have synchronized RGB and calibration",
                "available_cameras": sorted(views)}
    for camera in cameras:
        for name, mapping in (("root", root_pixels), ("tip", tip_pixels)):
            pixel = np.asarray(mapping[camera], dtype=float)
            if (pixel.shape != (2,) or not np.isfinite(pixel).all()
                    or pixel[0] < 0 or pixel[0] >= IMAGE_SHAPE[0]
                    or pixel[1] < 0 or pixel[1] >= IMAGE_SHAPE[1]):
                return {"accepted": False,
                        "reason": f"{camera} {name} pixel must be in original-image (row,column) bounds",
                        "pixel_convention": "(row,column)"}
    try:
        result = peg_fit.fit_peg_landmarks(
            [views[camera] for camera in cameras],
            [root_pixels[camera] for camera in cameras],
            [tip_pixels[camera] for camera in cameras], radius_m=radius_m)
    except (ValueError, np.linalg.LinAlgError) as exc:
        return {"accepted": False, "reason": str(exc), "source_step": source_step,
                "supporting_cameras": cameras, "pixel_convention": "(row,column)"}
    result["accepted"] = True
    result["source_step"] = source_step
    result["image_state_step"] = source_step
    result["supporting_cameras"] = cameras
    result["pixel_convention"] = "original RGB pixel coordinates (row,column)"
    return result


fit_pan._readonly = True
fit_peg_landmarks._readonly = True


def fit_peg(primitives, seeds, measured_step=None):
    """Fit one selected peg from explicit endpoint seeds and RGB.

    `seeds` maps camera names to `(root_rc, tip_rc)` in original-image
    `(row,column)` pixel coordinates. One local SAM point prompt near each
    seeded shaft chooses its RGB mask; calibrated views and robot occlusions
    are supplied to the mask-based pure fitter. No motion or clearance claim.
    """
    from . import robodojo_sight as sight
    from . import robodojo_tools as tools
    from . import robodojo_peg_geometry as peg_fit

    if measured_step is None:
        return {"valid": False, "reason": "measured_step is required",
                "pixel_convention": "(row,column)"}
    try:
        parsed_step = int(measured_step)
        if isinstance(measured_step, (bool, np.bool_)) or float(measured_step) != parsed_step:
            raise ValueError
        measured_step = parsed_step
    except (TypeError, ValueError, OverflowError):
        return {"valid": False, "reason": "measured_step must be an integer"}
    if not isinstance(seeds, dict) or len(seeds) < 2:
        return {"valid": False, "reason": "seeds must map at least two cameras to root/tip pixels"}
    cameras = list(seeds)
    if any(camera not in CAMERAS for camera in cameras):
        return {"valid": False, "reason": "unknown camera in peg seeds"}
    normalized = []
    for camera in cameras:
        try:
            pair = seeds[camera]
            if len(pair) != 2:
                raise ValueError
            root_rc, tip_rc = (np.asarray(p, dtype=float) for p in pair)
        except (TypeError, ValueError):
            return {"valid": False, "reason": f"{camera}: expected (root_rc, tip_rc) pair"}
        if any(p.shape != (2,) or not np.isfinite(p).all()
               or p[0] < 0 or p[0] >= IMAGE_SHAPE[0]
               or p[1] < 0 or p[1] >= IMAGE_SHAPE[1]
               for p in (root_rc, tip_rc)):
            return {"valid": False,
                    "reason": f"{camera}: seed points must be in original-image (row,column) bounds"}
        normalized.append((root_rc.tolist(), tip_rc.tolist()))

    obs = tools._obs(primitives)
    image_state = tools._shown(obs)
    source_step = _image_step(primitives)
    if measured_step != source_step:
        return {"valid": False, "reason": "peg seeds are stale for current displayed RGB",
                "measured_step": measured_step, "source_step": source_step,
                "pixel_convention": "(row,column)"}
    views = _views(primitives, obs, image_state, cameras)
    if set(views) != set(cameras):
        return {"valid": False, "reason": "each seed camera needs synchronized RGB and calibration",
                "available_cameras": sorted(views), "source_step": source_step}
    segmenter = getattr(primitives, "sam3_client", None)
    if segmenter is None:
        return {"valid": False, "reason": "sam3_client not configured", "source_step": source_step}

    masks, occluded = [], []
    for camera, (root_rc, tip_rc) in zip(cameras, normalized):
        rgb = np.asarray(tools._camera(obs, camera)["color"])
        point_rc = np.rint((np.asarray(root_rc) + np.asarray(tip_rc)) / 2).astype(int)
        response = segmenter.segment(rgb, point=point_rc.tolist(), min_score=.2)
        if not getattr(response, "found", False):
            return {"valid": False, "reason": f"{camera}: SAM found no seeded peg mask",
                    "source_step": source_step, "supporting_cameras": cameras}
        mask = np.asarray(response.mask, dtype=bool)
        robot = sight.robot_pixels(views[camera], image_state,
                                   own=tools._WRISTS.get(camera))
        if mask.shape != IMAGE_SHAPE or np.asarray(robot).shape != IMAGE_SHAPE:
            return {"valid": False, "reason": f"{camera}: mask shape must match original 480x640 RGB",
                    "source_step": source_step}
        masks.append(mask)
        occluded.append(np.asarray(robot, dtype=bool))
    try:
        result = peg_fit.fit_peg([views[c] for c in cameras], masks, normalized,
                                 occluded=occluded)
    except (ValueError, np.linalg.LinAlgError) as exc:
        return {"valid": False, "reason": str(exc), "source_step": source_step,
                "supporting_cameras": cameras, "pixel_convention": "(row,column)"}
    result["source_step"] = source_step
    result["image_state_step"] = source_step
    result["supporting_cameras"] = cameras
    result["pixel_convention"] = "original RGB pixel coordinates (row,column)"
    result["insertion_clearance"] = "not_evaluated"
    return result


fit_peg._readonly = True
