"""X31-H: scoped HexaShell geometry/path adaptation, not its unreleased agent.

Only RGB, public calibration and the robot's observed state enter this module.
The upstream source is unchanged in vendor/hexashell. We supply automatic
red-head pixel association, our existing pickup/keyboard template, link6/TCP
conversion, bounded Cartesian execution and nominal IK audits. Public radius
and keyboard plane remain explicit priors; this is not the upstream size-free
full system. No network/model calls or evaluator information.
"""
import itertools
import cv2
import numpy as np
from .vendor.hexashell import rgb_sequence as hx
from .xylophone_policy import XylophonePolicy as Base
from .xylophone_multiview import K, MOUNT, CV_TO_GL
from .geometry import pose_matrix, matrix_quat
from .policy import Waypoint
from .xylophone_lift import HIGH_DWELL
from .xylophone_extended_routes import transfer_clearance

UPSTREAM = "d1a2a24dc4c528a652b359fd689398530ef941d8"
RADIUS_PRIOR = .013


def camera_views(calibrated_head, state, vision):
    """World OpenCV frames from static public mount and observed link6."""
    flip = np.eye(4); flip[:3, :3] = CV_TO_GL
    frames = {"cam_head": (calibrated_head.k, calibrated_head.c2w @ flip)}
    for side in ("left", "right"):
        if side + "_ee_pose" in state:
            frames["cam_" + side + "_wrist"] = (
                K, pose_matrix(state[side + "_ee_pose"]) @ MOUNT @ flip)
    return {name: (k, frame, np.asarray(vision[name]["color"]))
            for name, (k, frame) in frames.items() if name in vision}


def associated_cone(rgb, k, frame, predicted):
    """Association uses a prior only to choose a component, never its centre."""
    view = (k, frame, rgb.shape[:2])
    seed = hx.project(view, predicted)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = np.uint8((((hsv[:, :, 0] < 10) | (hsv[:, :, 0] > 155))
                    & (hsv[:, :, 1] > 65) & (hsv[:, :, 2] > 40)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    candidates = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 20:
            continue
        component = np.uint8(labels == i)
        y, x = np.unravel_index(cv2.distanceTransform(component, cv2.DIST_L2, 3).argmax(),
                                component.shape)
        if np.linalg.norm(np.array([x, y]) - seed) > 50:
            continue
        try:
            fit = hx.silhouette(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), k, [x, y])
        except (ValueError, np.linalg.LinAlgError):
            continue
        if min(fit["hue"], 180 - fit["hue"]) < 25:
            fit["intrinsics"] = k
            candidates.append(fit)
    if len(candidates) != 1:
        raise ValueError("missing_or_ambiguous_unclipped_head_cone")
    return candidates[0]


def pair_offset(a, b):
    """Unchanged upstream stereo reconstruction in the observed link6 frame."""
    offset, radius, gap = hx.stereo_sphere(a["fit"], b["fit"], a["local"], b["local"])
    if abs(radius - RADIUS_PRIOR) > .003:
        raise ValueError("radius_inconsistent_with_public_asset_prior")
    return offset, radius, gap


def confirm_cones(rows, nominal):
    """All associated cones constrain the fit; never pick a favorable frame."""
    if len(rows) < 4:
        return dict(state="unknown", reason="hexashell_fewer_than_four_cones")
    movement = max(np.linalg.norm(r["ee"][:3, 3] - rows[0]["ee"][:3, 3]) for r in rows)
    if movement < .025:
        return dict(state="unknown", reason="hexashell_insufficient_actual_motion")
    pairs = []
    for a, b in itertools.combinations(rows, 2):
        try:
            p, radius, gap = pair_offset(a, b)
        except (ValueError, np.linalg.LinAlgError):
            continue
        pairs.append((p, radius, gap, a, b))
    if len(pairs) < 3:
        return dict(state="unknown", reason="hexashell_insufficient_stereo_pairs")
    mean = np.mean([p[0] for p in pairs], axis=0)
    spread = max(float(np.linalg.norm(p[0] - mean)) for p in pairs)
    if spread > .006 or np.linalg.norm(mean - nominal) > .035:
        return dict(state="inconsistent", reason="hexashell_inconsistent_offsets", spread_m=spread)
    errors = []
    for row in rows:
        view = (row["fit"]["intrinsics"], row["local"], (480, 640))
        try:
            errors.append(float(np.linalg.norm(hx.project(view, mean) - row["fit"]["pixel"])))
        except ValueError:
            return dict(state="inconsistent", reason="hexashell_offset_behind_camera")
    if max(errors) > 2.:
        return dict(state="inconsistent", reason="hexashell_all_view_disagreement",
                    maximum_reprojection_px=max(errors))
    # Report worst one-pixel perturbation for the widest-baseline pair. No
    # sqrt(frame-count) precision claim and no monocular radius range fallback.
    a, b = max(pairs, key=lambda p: np.linalg.norm(p[3]["local"][:3, 3]
                                                 - p[4]["local"][:3, 3]))[3:]
    base, _, _ = pair_offset(a, b)
    sensitivity = 0.
    for index in (0, 1):
        for axis in (0, 1):
            for sign in (-1, 1):
                changed = [dict(a, fit=dict(a["fit"])), dict(b, fit=dict(b["fit"]))]
                fit = changed[index]["fit"]
                pixel = fit["pixel"].copy(); pixel[axis] += sign
                direction = np.linalg.solve(fit["intrinsics"], [*pixel, 1.])
                fit["axis"] = direction / np.linalg.norm(direction)
                try:
                    offset, _, _ = pair_offset(*changed)
                except (ValueError, np.linalg.LinAlgError):
                    return dict(state="unknown", reason="hexashell_pixel_perturbation_rejected")
                sensitivity = max(sensitivity, float(np.linalg.norm(offset - base)))
    if sensitivity > .006:
        return dict(state="unknown", reason="hexashell_pixel_sensitivity", sensitivity_m=sensitivity)
    return dict(state="consistent", reason="hexashell_tangent_cone_rigid_hypothesis",
                ee_anchor=mean, views=len(rows), pairs=len(pairs), spread_m=spread,
                one_pixel_sensitivity_m=sensitivity, maximum_reprojection_px=max(errors),
                actual_motion_m=float(movement), radius_source="public_13mm_consistency_prior",
                rigid_attachment_assumed=True, not_a_grasp_certificate=True)


def tool_chain(pose, rotation, ee_anchor, tcp_offset, targets):
    """Keep upstream remote-feature construction; convert TCP back to link6."""
    tcp = pose.copy(); tcp[:3, 3] += pose[:3, :3] @ tcp_offset
    local = ee_anchor - tcp_offset
    frames = hx.make_chain(tcp, rotation, local, targets, .023, .04, 0, None)
    result = []
    for name, target in frames:
        ee = target.copy(); ee[:3, 3] -= ee[:3, :3] @ tcp_offset
        if ":" in name:
            phase, index = name.split(":")
            phase = {"above": "hover", "descend": "hit", "rise": "lift"}[phase]
            name = "xy_key_" + index + "_" + phase
        else:
            name = "xy_hexashell_" + name
        result.append((name, ee))
    return result


class XylophonePolicy(Base):
    def reset(self):
        super().reset()
        self.hx_rows = []; self.hx_log = []; self.hx_last_step = -100
        self.program_notes.update(candidate="X31H_HexaShell_geometry_series_adapter",
                                  hexashell_commit=UPSTREAM, gpt_api_enabled=False,
                                  upstream_full_agent_reproduced=False)

    def _record_probe(self, rgb, current, force=False):
        super()._record_probe(rgb, current, force)
        if self.nominal_anchor is None or self.step - self.hx_last_step < 4:
            return
        self.hx_last_step = self.step
        ee = pose_matrix(current)
        predicted = ee[:3, 3] + ee[:3, :3] @ self.nominal_anchor
        for camera, (k, frame, image) in camera_views(
                self.perception.camera, self.state_now, self.current_vision).items():
            try:
                fit = associated_cone(image, k, frame, predicted)
                local = np.linalg.inv(ee) @ frame
                # Identical mounted-camera poses add no triangulation baseline.
                if any(r["camera"] == camera and np.linalg.norm(r["local"] - local) < .008
                       for r in self.hx_rows):
                    continue
                if len(self.hx_rows) < 36:
                    self.hx_rows.append(dict(camera=camera, ee=ee, local=local, fit=fit))
                event = dict(step=self.step, camera=camera, state="measured",
                             pixel=fit["pixel"].tolist(), mode=fit["fit_mode"])
            except (ValueError, np.linalg.LinAlgError) as exc:
                event = dict(step=self.step, camera=camera, state="unknown", reason=str(exc))
            self.hx_log.append(event)
        self.program_notes["hexashell_observations"] = self.hx_log

    def _bind(self):
        fit = confirm_cones(self.hx_rows, self.nominal_anchor) if self.nominal_anchor is not None else {}
        self.program_notes["hexashell_binding"] = {k: v for k, v in fit.items() if k != "ee_anchor"}
        if fit.get("state") == "inconsistent":
            self.binding = self.program_notes["hexashell_binding"]
            self._put_back("xy_hexashell_attachment_inconsistent")
            return
        if fit.get("state") == "consistent":
            ee = pose_matrix(self.state_now[self.side + "_ee_pose"])
            centre = ee[:3, 3] + ee[:3, :3] @ fit["ee_anchor"]
            # Public resting plane is explicitly a prior, not measured depth.
            if centre[2] < .7956 + .025:
                self._put_back("xy_hexashell_no_lift_evidence"); return
            self.anchor = fit["ee_anchor"]
            self.binding = self.program_notes["hexashell_binding"]
            if not self.keyboard_refreshed:
                self.stage = "keyboard"; self.reason = "xy_hexashell_bound_pending_keyboard"
            else:
                self._start_strokes()
            return
        # Existing positive evidence remains usable; unknown is not false.
        super()._bind()

    def _choose_strokes(self, side, rotation, anchor, prefix=(), reserve=60):
        tcp_offset = np.asarray(self.calibration["tool_offset_in_ee"])
        pose = pose_matrix(self.state_now[side + "_ee_pose"])
        targets = np.c_[self.keys, np.full(len(self.keys), .78685)]
        attempts = []
        for r, angles in hx.orientation_candidates(rotation, anchor - tcp_offset):
            clearance = .023 - float((r @ (anchor - tcp_offset))[2])
            if clearance < .035:
                continue
            frames = tool_chain(pose, r, anchor, tcp_offset, targets)
            waypoints = [Waypoint(side, np.r_[t[:3, 3], matrix_quat(t[:3, :3])], 0.,
                                 name, 2 if name.endswith("_hit") else HIGH_DWELL, anchor)
                         for name, t in frames]
            full = list(prefix) + waypoints
            clear, minimum = transfer_clearance(self.state_now[side + "_ee_pose"], full, anchor,
                tcp_offset, self.calibration["contact_tip_offset_in_ee"], self.calibration["table_z"])
            accepted = clear and self._audit(side, full, anchor, reserve=reserve, local_only=True)
            attempts.append(dict(**angles, nominal_clearance_m=minimum, accepted=bool(accepted),
                                 audit=self.program_notes.get("path_audit") if clear else "clearance_rejected"))
            if accepted:
                self.program_notes["hexashell_series"] = dict(selected=True, attempts=attempts,
                    gap_m=.023, rise_m=.04, surface_z_prior=.78685,
                    execution="existing_bounded_cartesian_not_unreleased_curobo_harness")
                return waypoints, dict(degrees=angles["tilt_deg"], heading_degrees=angles["yaw_deg"])
        self.program_notes["hexashell_series"] = dict(selected=False, attempts=attempts,
                                                     fallback="existing_X30_audited_series")
        return super()._choose_strokes(side, rotation, anchor, prefix, reserve)
