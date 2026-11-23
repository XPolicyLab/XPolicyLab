# Copyright © 2026 Martingale. All Rights Reserved.
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from . import robodojo_basket_shape as shape


def _hand_pose(state, arm):
    if arm not in ("left", "right"):
        raise ValueError("arm must be left or right")
    p = np.asarray(state[f"{arm}_ee_pose"], float)
    if p.shape != (7,) or not np.isfinite(p).all():
        raise ValueError(f"{arm}_ee_pose must be finite xyz + wxyz quaternion")
    t = np.eye(4)
    t[:3, :3] = Rotation.from_quat(p[[4, 5, 6, 3]]).as_matrix()
    t[:3, 3] = p[:3]
    return t


def _transform(points, pose):
    return np.asarray(points) @ pose[:3, :3].T + pose[:3, 3]


def fit_basket_geometry(views, masks, robot_masks, *, table_z_m,
                        body_bounds, body_initial, handle_bounds, handle_initial,
                        source_step, image_state_step):
    """Fit body and arch from synchronized RGB-derived masks and calibrated views.

    Masks are keyed by camera and contain `whole` and `handle` bool arrays at
    original 640x480 resolution; robot_masks have the same shape. The caller
    supplies explicit, task-appropriate priors and image/action provenance.
    """
    if not isinstance(source_step, int) or not isinstance(image_state_step, int):
        raise ValueError("source_step and image_state_step must be integers")
    if source_step != image_state_step:
        return {"valid": False, "reason": "RGB/mask and calibrated pose steps differ",
                "source_step": source_step, "image_state_step": image_state_step}
    if not np.isfinite(table_z_m):
        raise ValueError("table_z_m must be finite")
    body, bi = shape.fit_part(views, masks, robot_masks, "body", body_bounds,
                              body_initial, float(table_z_m))
    if body is None or bi.get("mask_fit_score", 0) < .65:
        return {"valid": False, "reason": bi.get("reason", "body fit below 0.65 score"),
                "body_fit": bi, "source_step": source_step,
                "assumptions": {"table_z_m": float(table_z_m), "table_supported": True}}
    x, y, yaw, length, width, height, _taper = body
    hb = handle_bounds
    hi = handle_initial
    handle, hinfo = shape.fit_part(views, masks, robot_masks, "handle", hb, hi,
                                   float(table_z_m))
    if handle is None or hinfo.get("mask_fit_score", 0) < .65:
        return {"valid": False, "reason": hinfo.get("reason", "handle fit below 0.65 score"),
                "body_fit": bi, "handle_fit": hinfo, "source_step": source_step,
                "assumptions": {"table_z_m": float(table_z_m), "table_supported": True}}
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
    pose[:3, 3] = [x, y, table_z_m]
    local_body = (shape.box_vertices(body, table_z_m) - pose[:3, 3]) @ pose[:3, :3]
    local_handle = (shape.arch_points(handle) - pose[:3, 3]) @ pose[:3, :3]
    crown = (np.array([handle[0], handle[1], handle[4] + handle[5]]) - pose[:3, 3]) @ pose[:3, :3]
    return {"valid": True, "body_parameters": body.tolist(), "handle_parameters": handle.tolist(),
            "body_fit": bi, "handle_fit": hinfo, "pose_world": pose.tolist(),
            "body_local_xyz": local_body.tolist(), "handle_local_xyz": local_handle.tolist(),
            "crown_local_xyz": crown.tolist(),
            "handle_tangent_local": (pose[:3, :3].T @ np.array(
                [np.cos(handle[2]), np.sin(handle[2]), 0.])).tolist(),
            "handle_diameter_m": float(2 * handle[-1]), "source_step": source_step,
            "image_state_step": image_state_step,
            "camera_ids": sorted(set(bi["supporting_views"]) & set(hinfo["supporting_views"])),
            "assumptions": {"table_z_m": float(table_z_m), "table_supported": True,
                            "body_shape": "tapered rectangular basket prism",
                            "handle_shape": "half elliptical arch", "dimensions_from_rgb": True}}


class BasketTracker:
    """Small in-memory tracker updated only when caller supplies a fresh fit."""
    def __init__(self, fit, observation_step, image_state, held_by=None):
        if not fit.get("valid"):
            raise ValueError("tracker initialization requires a valid basket fit")
        if held_by not in (None, "", "left", "right"):
            raise ValueError("held_by must be left, right or None")
        if int(fit.get("source_step", -1)) != int(observation_step):
            raise ValueError("fit source_step must equal observation_step")
        self.pose = np.asarray(fit["pose_world"], float)
        self.body_local = np.asarray(fit["body_local_xyz"], float)
        self.handle_local = np.asarray(fit["handle_local_xyz"], float)
        self.crown_local = np.asarray(fit["crown_local_xyz"], float)
        self.tangent_local = np.asarray(fit["handle_tangent_local"], float)
        self.radius = float(fit["handle_diameter_m"]) / 2
        self.body_size = np.asarray(fit["body_parameters"][3:6], float)
        self.fit_score = min(float(fit["body_fit"]["mask_fit_score"]),
                             float(fit["handle_fit"]["mask_fit_score"]))
        self.observation_step = int(observation_step)
        self.seen_step = int(observation_step)
        self.image_state = image_state
        self.held_by = None
        self.attachment = None
        self.velocity = np.zeros(3)
        self.velocity_known = False
        self.support = len(fit.get("camera_ids", []))
        self.reason = "initialized from synchronized multi-view RGB fit"
        self.innovation = 0.0
        self.set_holder(held_by)

    def set_holder(self, arm):
        if arm == "":
            arm = None
        if arm not in (None, "left", "right"):
            raise ValueError("held_by must be left, right or None")
        if arm == self.held_by:
            return
        if arm:
            self.attachment = np.linalg.inv(_hand_pose(self.image_state, arm)) @ self.pose
        else:
            self.attachment = None
        self.held_by = arm
        self.velocity[:] = 0
        self.velocity_known = False

    def update(self, fit, observation_step, image_state):
        """Accept a fresh externally acquired/refitted observation; no I/O."""
        step = int(observation_step)
        if int(fit.get("source_step", -1)) != step:
            raise ValueError("fit source_step must equal observation_step")
        if step <= self.seen_step:
            return False
        self.seen_step = step
        if not fit.get("valid"):
            self.support = 0
            self.reason = fit.get("reason", "visual refit rejected")
            return False
        support = len(fit.get("camera_ids", []))
        candidate = np.asarray(fit["pose_world"], float)
        if candidate.shape != (4, 4) or not np.isfinite(candidate).all():
            self.reason = "invalid fitted pose"
            self.support = 0
            return False
        if support < 2:
            self.reason = "prediction only; fewer than two supporting cameras"
            self.support = support
            return False
        innovation = float(np.linalg.norm(candidate[:3, 3] - self.pose[:3, 3]))
        if innovation > .16:
            self.reason = "rejected visual fit with >160mm pose innovation"
            self.support = support
            return False
        dt = step - self.observation_step
        if not self.held_by:
            measured = (candidate[:3, 3] - self.pose[:3, 3]) / dt
            self.velocity = .65 * measured + .35 * self.velocity if self.velocity_known else measured
            self.velocity_known = True
        self.pose = candidate
        self.body_local = np.asarray(fit["body_local_xyz"], float)
        self.handle_local = np.asarray(fit["handle_local_xyz"], float)
        self.crown_local = np.asarray(fit["crown_local_xyz"], float)
        self.tangent_local = np.asarray(fit["handle_tangent_local"], float)
        self.radius = float(fit["handle_diameter_m"]) / 2
        self.body_size = np.asarray(fit["body_parameters"][3:6], float)
        self.fit_score = min(float(fit["body_fit"]["mask_fit_score"]),
                             float(fit["handle_fit"]["mask_fit_score"]))
        self.observation_step = step
        self.image_state = image_state
        if self.held_by:
            self.attachment = np.linalg.inv(_hand_pose(image_state, self.held_by)) @ candidate
        self.support = support
        self.innovation = innovation
        self.reason = "visually updated shared rigid pose"
        return True

    def measure(self, action_step, current_image_state, predict_actions=0):
        step = int(action_step)
        horizon = int(np.clip(predict_actions, 0, 8))
        if step < self.observation_step:
            raise ValueError("action_step predates observation")
        age = step - self.observation_step
        current = self._predict(step, current_image_state)
        future = self._predict(step + horizon, current_image_state)
        body_center = _transform(np.array([[0, 0, self.body_size[2] / 2]]), current)[0]
        handle = _transform(self.crown_local[None], current)[0]
        uncertainty = .005 + min(.03, self.innovation) + age * .002
        valid = age <= 10 and self.support >= 2
        return {"valid": valid, "status": "observed" if age == 0 else "predicted",
                "note": self.reason, "action_step": step, "observation_step": self.observation_step,
                "age_actions": age, "supporting_views": self.support,
                "fit_score": round(self.fit_score, 4), "held_by": self.held_by,
                "body_center_xyz": body_center.tolist(), "body_size_m": self.body_size.tolist(),
                "body_rotation_matrix": current[:3, :3].tolist(), "handle_xyz": handle.tolist(),
                "handle_tangent_xyz": (current[:3, :3] @ self.tangent_local).tolist(),
                "handle_diameter_m": 2 * self.radius,
                "handle_arch_xyz": _transform(self.handle_local[::4], current).tolist(),
                "velocity_m_per_action": self.velocity.tolist(),
                "velocity_known": self.velocity_known,
                "predicted_handle_xyz": _transform(self.crown_local[None], future)[0].tolist(),
                "predict_actions": horizon, "uncertainty_m": round(uncertainty, 4),
                "image_state_step": self.observation_step}

    def _predict(self, step, state):
        if self.held_by:
            return _hand_pose(state, self.held_by) @ self.attachment
        result = self.pose.copy()
        result[:3, 3] += self.velocity * np.clip(step - self.observation_step, 0, 20)
        return result


def fit_basket(primitives, *, table_z_m, body_bounds, body_initial,
               handle_bounds, handle_initial, min_score=.2):
    """Explicitly segment current RGB and fit basket geometry.

    This performs six local SAM segmentation requests (whole basket + handle in
    each available camera) and no robot motion. Callers own when refresh occurs.
    Bounds/initial values encode declared shape priors.
    """
    from . import robodojo_sight as sight
    from . import robodojo_tools as tools

    obs = tools._obs(primitives)
    image_state = tools._shown(obs)
    status = primitives.env.get_status()
    step = int(status.get("step", 0))
    source_step = max(0, step - int(status.get("pictures_behind") or 0))
    views, masks, robot_masks = {}, {}, {}
    segmenter = getattr(primitives, "sam3_client", None)
    if segmenter is None:
        return {"valid": False, "reason": "sam3_client not configured"}
    for camera in shape.CAMERAS:
        cam = tools._camera(obs, camera)
        if cam is None or cam.get("color") is None:
            continue
        view = tools._view(camera, image_state)
        if view is None:
            continue
        rgb = np.asarray(cam["color"])
        if rgb.shape != (480, 640, 3):
            return {"valid": False, "reason": f"{camera}: expected 480x640 RGB"}
        parts = {}
        for key, prompt in (("whole", "light blue basket"), ("handle", "basket handle")):
            response = segmenter.segment(rgb, text_prompt=prompt, min_score=float(min_score))
            mask = np.asarray(response.mask, bool) if response.found else np.zeros((480, 640), bool)
            if mask.shape != (480, 640):
                return {"valid": False, "reason": f"{camera}/{key}: segmenter mask shape mismatch"}
            parts[key] = mask
        views[camera] = view
        masks[camera] = parts
        robot_masks[camera] = sight.robot_pixels(view, image_state,
                                                   own=tools._WRISTS.get(camera))
    if len(views) < 2:
        return {"valid": False, "reason": "need at least two calibrated RGB cameras",
                "source_step": source_step}
    return fit_basket_geometry(views, masks, robot_masks, table_z_m=table_z_m,
                               body_bounds=body_bounds, body_initial=body_initial,
                               handle_bounds=handle_bounds, handle_initial=handle_initial,
                               source_step=source_step, image_state_step=source_step)


def track_basket(primitives, fit, *, held_by=None, reset=False,
                 predict_actions=0):
    """Explicitly update/read a host-side track; never called from after_action.

    `fit` must be the result of a caller-requested `fit_basket` invocation (or
    None to read/predict an existing track). State is in memory only.
    """
    from . import robodojo_tools as tools
    obs = tools._obs(primitives)
    status = primitives.env.get_status()
    step = int(status.get("step", 0))
    source_step = max(0, step - int(status.get("pictures_behind") or 0))
    image_state = tools._shown(obs)
    if fit is not None and int(fit.get("source_step", -1)) != source_step:
        return {"valid": False, "reason": "fit is stale relative to current image step",
                "fit_source_step": fit.get("source_step"), "current_image_step": source_step}
    tracker = None if reset else getattr(primitives, "_basket_tracker", None)
    if reset:
        primitives._basket_tracker = None
    if fit is not None and fit.get("valid"):
        if tracker is None:
            tracker = BasketTracker(fit, fit["source_step"], image_state, held_by)
            primitives._basket_tracker = tracker
        else:
            tracker.update(fit, fit["source_step"], image_state)
            if held_by is not None:
                tracker.set_holder(held_by)
    if tracker is None:
        return {"valid": False, "reason": "no basket track; call fit_basket then track_basket"}
    if held_by is not None and held_by != tracker.held_by:
        if fit is None:
            return {"valid": False, "reason": "refresh geometry when basket holder changes"}
        tracker.set_holder(held_by)
    return tracker.measure(step, image_state, predict_actions)


track_basket._readonly = True
