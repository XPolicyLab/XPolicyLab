# Copyright © 2026 Martingale. All Rights Reserved.
import itertools

import numpy as np

from . import robodojo_mug_shapes as shapes
from . import robodojo_sight as sight
from . import robodojo_tools as tools
from .sam3_local import _array

CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")


def _candidates(sam, image, prompt):
    """Get diverse SAM masks for one prompt; masks are kept within this call."""
    with sam._lock, sam._autocast():
        raw = sam._processor.set_text_prompt(prompt=prompt, state=sam._state_for(image))
        if raw.get("masks") is None:
            return []
        masks = (_array(raw["masks"]) > 0).reshape(-1, *image.shape[:2])
        scores = _array(raw["scores"]).reshape(-1)
    selected = []
    for index in np.argsort(-scores):
        mask = masks[index]
        if scores[index] < 0.2 or mask.sum() < 15:
            continue
        if any((mask & prior).sum() / max(1, (mask | prior).sum()) > 0.65
               for prior in selected):
            continue
        selected.append(mask)
    return selected


def _fit_instances(views, groups, handles, robot_masks):
    used = {camera: set() for camera in CAMERAS}
    models = []

    def center(camera, mask):
        rows, cols = np.nonzero(mask)
        points, valid = views[camera].on_plane(rows, cols, 0.81)
        if not np.any(valid):
            return np.full(2, np.nan)
        return np.median(points[valid], axis=0)

    def parts(camera, whole):
        selected = [handle for handle in handles[camera]
                    if (handle & whole).sum() / max(1, handle.sum()) > 0.35]
        loop = np.logical_or.reduce(selected) & whole if selected else np.zeros_like(whole)
        return whole & ~loop, loop

    for index, primary in enumerate(groups["cam_head"]):
        anchor = center("cam_head", primary)
        choices = []
        for camera in CAMERAS[1:]:
            ranked = sorted(
                (mask for mask in groups[camera] if id(mask) not in used[camera]),
                key=lambda mask: np.linalg.norm(center(camera, mask) - anchor),
            )
            ranked = [mask for mask in ranked[:2]
                      if np.linalg.norm(center(camera, mask) - anchor) < 0.14]
            if ranked:
                choices.append((camera, ranked))
        combinations = itertools.product(*(items for _, items in choices)) if choices else [()]
        best = None
        for combination in combinations:
            whole = {"cam_head": primary,
                     **{camera: mask for (camera, _), mask in zip(choices, combination)}}
            bodies, loops = [], []
            for camera, mask in whole.items():
                body_mask, loop_mask = parts(camera, mask)
                bodies.append(sight.Sight(views[camera], body_mask,
                                          robot_masks[camera] | loop_mask))
                loops.append(sight.Sight(views[camera], loop_mask,
                                         robot_masks[camera] | body_mask))
            body = shapes.fit_body(bodies)
            score = float(np.mean(body.get("per_view_iou", [0.0])))
            if best is None or score > best[0]:
                best = (score, body, loops, combination, whole)
        if best is None:
            continue
        _, body, loops, combination, whole = best
        handle = (shapes.fit_handle(np.asarray(body["parameters"], float), loops)
                  if body.get("valid") else {"valid": False, "reason": "Body fit lacks support"})
        if body.get("valid"):
            for (camera, _), mask in zip(choices, combination):
                used[camera].add(id(mask))
        rows, cols = np.nonzero(primary)
        models.append({
            "index": index,
            "head_bbox_row_col": [int(rows.min()), int(cols.min()),
                                   int(rows.max()), int(cols.max())],
            "valid": bool(body.get("valid") and handle.get("valid")),
            "body": body,
            "handle": handle,
            "views": list(whole),
        })
    return models


def fit_mugs(primitives):
    """Fit current table-supported mug bodies and handles from synchronized RGB.

    The bridge owner must call this in a no-motion observation block. It returns
    compact per-mug parameters/errors, the simulator step before/after fitting,
    and an explicit initial-upright-only limitation. No persistent IDs are made.
    """
    before = primitives.env.get_status().get("step")
    obs = tools._obs(primitives)
    shown = tools._shown(obs)
    views, groups, handles, robot_masks = {}, {}, {}, {}
    for camera in CAMERAS:
        frame = tools._camera(obs, camera)
        if frame is None or frame.get("color") is None:
            continue
        view = tools._view(camera, shown)
        if view is None:
            continue
        views[camera] = view
        groups[camera] = _candidates(primitives.sam3_client, frame["color"], "mug")
        handles[camera] = _candidates(primitives.sam3_client, frame["color"], "mug handle")
        robot_masks[camera] = sight.robot_pixels(view, shown, own=tools._WRISTS.get(camera))
    after = primitives.env.get_status().get("step")
    if before != after:
        return {
            "accepted": False, "mugs": [], "step_before": before, "step_after": after,
            "reason": "Simulator state changed during fitting; discard and reacquire a synchronized observation.",
        }
    models = (_fit_instances(views, groups, handles, robot_masks)
              if "cam_head" in views else [])
    valid_count = sum(bool(model["valid"]) for model in models)
    if valid_count:
        status = "fit_available"
        reasons = []
    elif models:
        status = "candidate_fits_rejected"
        reasons = ["No candidate has both an accepted body and handle fit; inspect per-part reasons and reobserve."]
    else:
        status = "no_mug_mask_in_head_view"
        reasons = ["No candidate mug mask was available in the current head view; obtain another view."]
    return {
        "accepted": bool(valid_count), "status": status, "reasons": reasons,
        "mug_candidate_count": len(models), "valid_mug_count": valid_count, "mugs": models,
        "step": after, "step_before": before, "step_after": after,
        "image_state_source": "same returned observation via robodojo_tools._shown",
        "detections_per_view": {camera: len(groups[camera]) for camera in groups},
        "assumptions": {"table_z_m": 0.765, "upright_table_supported": True,
                        "generic_wall_thickness_m": shapes.WALL},
        "note": "Local indices are not persistent identities. Initial upright, table-supported mugs only; refit after motion. Handle fit is approximate clearance geometry, not grasp certification.",
    }


fit_mugs._readonly = True
