"""Text-grounded RGB-D measurements in the world frame.

One mask from the segmenter plus one depth image from the same camera give an object's
top-face centre, height above its support and footprint (length, width, long axis). Learned
depth is offset by a few millimetres per view; in wrist views the offset is re-levelled on
the support seen in a ring of pixels just outside the mask. Measured with this pipeline on
RoboDojo (MoGe-2 depth): head camera about 8 mm XY / 6 mm height with an unreliable heading;
a wrist camera straight above the object about 0.4 mm XY / 2-3 mm height / 1 degree heading.
Every method raises PerceptionError with a specific reason instead of returning a guess.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from knowin_agent.perception import geometry as geo
from knowin_agent.perception.types import (
    AxisTipEstimate,
    DepthSource,
    Mask,
    ObjectEstimate,
    OpeningEstimate,
    PerceptionError,
    Segmenter,
    SupportPlaneEstimate,
)
from knowin_agent.runtime.backend import CameraFrame, Observation

ABOVE_SUPPORT_M = 0.003
TOP_PERCENTILE = 97.0
TOP_BAND_M = 0.008
FACE_BAND_M = 0.006
RIM_BAND_M = 0.004
MIN_POINTS = 20
MIN_MASK_PIXELS = 16
CORE_ERODE_PX = 2
EDGE_TOLERANCE_M, EDGE_TOLERANCE_REL = 0.004, 0.006
FOOTPRINT_TRIM_PCT = 0.2
TOP_DOWN_MAX_TILT_DEG = 20.0
SQUARE_TOLERANCE = 0.15
HINT_RADIUS_PX = 12
REPROJECT_MARGIN_PX = 4
RING_INNER_PX, RING_OUTER_PX = 3, 12
RING_MIN_POINTS = 50
RING_BAND_M = 0.03
RING_MAX_OFFSET_M = 0.025
RING_MAX_SPREAD_M = 0.008
WRIST_CAMERAS = ("left_wrist", "right_wrist")


def colour_word(rgb: np.ndarray, pixel: Sequence[float]) -> str:
    """The colour a person would name around the pixel (the most saturated material nearby)."""
    h, w = rgb.shape[:2]
    u = int(np.clip(round(float(pixel[0])), 0, w - 1))
    v = int(np.clip(round(float(pixel[1])), 0, h - 1))
    patch = np.asarray(rgb)[max(0, v - 15):v + 16, max(0, u - 15):u + 16, :3].reshape(-1, 3).astype(float)
    hi, lo = patch.max(axis=1), patch.min(axis=1)
    delta = hi - lo
    sat = np.where(hi > 0, 255.0 * delta / np.maximum(hi, 1e-9), 0.0)
    keep = sat >= np.percentile(sat, 70)
    s, val = float(sat[keep].mean()), float(hi[keep].mean())
    if s < 45:
        return "white" if val > 170 else "black" if val < 60 else "gray"
    r, g, b = patch[keep].T
    d = np.maximum(delta[keep], 1e-9)
    hue = np.where(hi[keep] == r, ((g - b) / d) % 6, np.where(hi[keep] == g, (b - r) / d + 2, (r - g) / d + 4)) * 60.0
    rad = np.radians(hue)
    mean = math.degrees(math.atan2(float(np.sin(rad).mean()), float(np.cos(rad).mean()))) % 360.0
    for limit, name in ((16, "red"), (40, "orange"), (68, "yellow"), (170, "green"), (200, "cyan"),
                        (260, "blue"), (310, "purple"), (344, "pink"), (360, "red")):
        if mean <= limit:
            return name
    return "red"


def pick_instance(masks: Sequence[Mask], pixel: Sequence[float]) -> Optional[int]:
    """Index of the mask covering the pixel (holes filled), else the nearest within HINT_RADIUS_PX."""
    u, v = int(round(float(pixel[0]))), int(round(float(pixel[1])))
    best, best_distance = None, None
    for index, m in enumerate(masks):
        filled = geo.fill_holes(m.mask)
        if 0 <= v < filled.shape[0] and 0 <= u < filled.shape[1] and filled[v, u]:
            return index
        rows, cols = np.nonzero(filled)
        if cols.size:
            distance = float(np.min(np.hypot(cols - u, rows - v)))
            if distance <= HINT_RADIUS_PX and (best_distance is None or distance < best_distance):
                best, best_distance = index, distance
    return best


def grasp_yaw_for(rect: geo.Rect) -> Optional[float]:
    """Yaw that closes the fingers across the short side; None for a near-square footprint."""
    if rect.length <= 0 or (rect.length - rect.width) < SQUARE_TOLERANCE * rect.length:
        return None
    return geo.normalize_axis_deg(rect.angle_deg + 90.0)


@dataclass
class _View:
    name: str
    frame: CameraFrame
    K: np.ndarray
    T: np.ndarray

    @property
    def tilt_deg(self) -> float:
        return geo.optical_axis_tilt_deg(self.T)

    @property
    def top_down(self) -> bool:
        return self.tilt_deg <= TOP_DOWN_MAX_TILT_DEG

    @property
    def shape(self) -> tuple[int, int]:
        return self.frame.rgb.shape[:2]


@dataclass
class _Prepared:
    """Depth (possibly re-levelled), support height and the object's points for one mask."""

    depth: np.ndarray
    support_z: float
    full: geo.Points  # all mask pixels with consistent depth
    core: geo.Points  # eroded mask, for heights


class Perceiver:
    """Text grounding plus RGB-D geometry over pluggable segmentation and depth.

    table_z is the robot profile's table height (RoboDojo's station table is at 0.765 m); it is
    the default support for objects and enables wrist-view depth re-levelling. Without it the
    support is measured on the ring of pixels around each object.
    """

    def __init__(self, segmenter: Segmenter, depth_source: DepthSource, *, table_z: Optional[float] = None,
                 min_mask_score: float = 0.5, relevel_cameras: Sequence[str] = WRIST_CAMERAS):
        self.segmenter = segmenter
        self.depth_source = depth_source
        self.table_z = None if table_z is None else float(table_z)
        self.min_mask_score = float(min_mask_score)
        self.relevel_cameras = tuple(relevel_cameras)

    @classmethod
    def from_env(cls, *, table_z: Optional[float] = None, **kwargs) -> "Perceiver":
        """Cached HTTP services at KNOWIN_AGENT_SAM3_URL / KNOWIN_AGENT_DEPTH_URL."""
        from knowin_agent.perception.services import default_services

        segmenter, depth = default_services()
        return cls(segmenter, depth, table_z=table_z, **kwargs)

    # ------------------------------------------------------------ public API

    def locate(self, obs: Observation, text: str, camera: str = "head", *, hint_px: Optional[Sequence[float]] = None,
               max_objects: int = 1, support_z: Optional[float] = None,
               relevel: Optional[bool] = None) -> list[ObjectEstimate]:
        """Objects matching `text`, best first: the mask under hint_px if given, then by segmenter score."""
        return self._locate(obs, text, camera, hint_px=hint_px, max_objects=max_objects, support_z=support_z,
                            relevel=relevel, detail=False)

    def inspect_shape(self, obs: Observation, text: str, camera: str = "head", *,
                      hint_px: Optional[Sequence[float]] = None, support_z: Optional[float] = None,
                      relevel: Optional[bool] = None) -> ObjectEstimate:
        """locate() for one object with footprint corners, roundness and a coarse shape class in quality."""
        return self._locate(obs, text, camera, hint_px=hint_px, max_objects=1, support_z=support_z,
                            relevel=relevel, detail=True)[0]

    def inspect_axis_tip(self, obs: Observation, text: str, camera: str = "head", *,
                         gripper_point: Optional[Sequence[float]] = None, hint_px: Optional[Sequence[float]] = None,
                         relevel: Optional[bool] = None) -> AxisTipEstimate:
        """Long axis of an elongated object; the tip is the end farther from gripper_point, else the lower end."""
        view = self._view(obs, camera)
        quality = self._quality()
        mask, phrase = self._candidates(view, text, hint_px, quality)[0]
        prep = self._prepare(view, mask.mask, None, relevel, quality, require_support=False)
        pts = prep.full.xyz
        if len(pts) < MIN_POINTS:
            raise PerceptionError(f"only {len(pts)} depth points on '{phrase}' in camera '{camera}'")
        centroid, direction, spread = geo.principal_axis(pts)
        if spread[0] < 4.0 * max(float(spread[1]), 1e-12):
            raise PerceptionError(f"'{phrase}' is not elongated (principal spreads {np.sqrt(spread[:2]).round(4)} m)")
        cylinder = geo.fit_cylinder_axis(pts, direction)
        if cylinder is not None and cylinder.radius < 0.5 * math.sqrt(spread[0]):
            point, direction, method = cylinder.point, cylinder.direction, "pca_cylinder_axis"
            quality.update(radius_m=cylinder.radius, cylinder_rms_m=cylinder.rms, cylinder_arc_deg=cylinder.arc_deg)
        else:
            point, method = centroid, "pca_surface_axis"
            quality["warnings"].append("cylinder_fit_failed: axis passes through visible surface points")
        t = (pts - point) @ direction
        lo, hi = geo.robust_percentile(t, [0.25, 99.75])
        ends = [point + lo * direction, point + hi * direction]
        if gripper_point is not None:
            g = np.asarray(gripper_point, dtype=float).reshape(3)
            tip_index = int(np.argmax([np.linalg.norm(e - g) for e in ends]))
            quality["tip_rule"] = "farther_from_gripper"
        else:
            tip_index = int(np.argmin([e[2] for e in ends]))
            quality["tip_rule"] = "lower_end"
        tip, base = ends[tip_index], ends[1 - tip_index]
        quality.update(points=len(pts), text_used=phrase, mask_pixels=mask.pixels,
                       residual_m=float(np.median(np.linalg.norm(np.cross(pts - point, direction), axis=1))))
        return AxisTipEstimate(label=text, camera=camera, obs_id=obs.obs_id,
                               axis_dir=geo.unit3(tip - base).tolist(), tip_xyz=tip.tolist(),
                               base_xyz=base.tolist(), length_m=float(hi - lo), method=method, quality=quality)

    def inspect_support_plane(self, obs: Observation, camera: str = "head", *,
                              around_xyz: Optional[Sequence[float]] = None, pixel: Optional[Sequence[float]] = None,
                              radius_m: float = 0.08, exclude_text: Optional[str] = None) -> SupportPlaneEstimate:
        """RANSAC plane on raw depth points in a disc around a world point (or a pixel's surface point)."""
        view = self._view(obs, camera)
        depth = self._depth(view)
        quality = self._quality()
        keep = geo.valid_depth(depth)
        if exclude_text:
            masks = self._masks(view, exclude_text)
            if masks:
                excluded = geo.dilate(np.any([m.mask for m in masks], axis=0), 3)
                keep &= ~excluded
                quality["excluded_pixels"] = int(excluded.sum())
            else:
                quality["warnings"].append(f"exclude_text_found_nothing: {exclude_text!r}")
        pts = geo.depth_to_world(depth, view.K, view.T, keep).xyz
        centre = None
        if around_xyz is not None:
            centre = np.asarray(around_xyz, dtype=float).reshape(3)
        elif pixel is not None:
            centre = self._surface_point(view, depth, pixel)
        if centre is not None:
            pts = pts[np.hypot(pts[:, 0] - centre[0], pts[:, 1] - centre[1]) <= radius_m]
        if len(pts) < 50:
            raise PerceptionError(f"only {len(pts)} depth points within {radius_m} m for a support plane")
        plane = geo.fit_plane(pts, threshold=0.003)
        if plane.normal[2] < 0.5:
            raise PerceptionError(f"dominant plane is tilted {plane.tilt_deg:.0f} deg; not a support surface")
        if plane.inlier_ratio < 0.3:
            quality["warnings"].append(f"low_inlier_ratio: {plane.inlier_ratio:.2f}")
        inliers = pts[plane.inliers]
        cx, cy = (centre[:2] if centre is not None else inliers[:, :2].mean(axis=0))
        z = plane.z_at(cx, cy)
        quality.update(points=len(pts), inliers=int(plane.inliers.sum()), rms_m=plane.rms, tilt_deg=plane.tilt_deg,
                       radius_m=radius_m if centre is not None else None)
        return SupportPlaneEstimate(camera=camera, obs_id=obs.obs_id, z=z, normal=plane.normal.tolist(),
                                    inlier_ratio=plane.inlier_ratio, center_xyz=[float(cx), float(cy), z],
                                    method="ransac_plane", quality=quality)

    def inspect_opening(self, obs: Observation, text: str, camera: str = "head", *,
                        hint_px: Optional[Sequence[float]] = None, support_z: Optional[float] = None,
                        relevel: Optional[bool] = None) -> OpeningEstimate:
        """Rim of a container or ring: circle through the points within a few mm of its top."""
        view = self._view(obs, camera)
        quality = self._quality()
        mask, phrase = self._candidates(view, text, hint_px, quality)[0]
        prep = self._prepare(view, mask.mask, support_z, relevel, quality)
        top, _ = self._top(view, prep, quality)
        rim = prep.full.select(prep.full.xyz[:, 2] >= top - RIM_BAND_M)
        if len(rim) < MIN_POINTS:
            raise PerceptionError(f"only {len(rim)} rim points near the top of '{phrase}'")
        xy = self._face_xy(view, rim, top)
        first = geo.fit_circle(xy)
        radial = np.linalg.norm(xy - first.center, axis=1)
        outer = geo.fit_circle(xy[radial >= np.percentile(radial, 80)])
        radial = np.linalg.norm(xy - outer.center, axis=1)
        inner_r, outer_r = (float(v) for v in np.percentile(radial, [1, 99]))
        if outer.arc_deg < 180:
            quality["warnings"].append(f"partial_rim_arc: {outer.arc_deg:.0f} deg")
        if inner_r > 0.3 * outer_r:
            radius, method = inner_r, "rim_circle_inner"
        else:
            radius, method = outer_r, "rim_circle_outer"
            quality["warnings"].append("no_hole_visible: radius is the outer rim")
        normal = np.array([0.0, 0.0, 1.0])
        if len(rim) >= 30:
            plane = geo.fit_plane(rim.xyz, threshold=0.002)
            quality["rim_tilt_deg"] = plane.tilt_deg
            if plane.tilt_deg > 5.0 and plane.inlier_ratio > 0.6:
                normal = plane.normal
                quality["warnings"].append(f"rim_tilted: {plane.tilt_deg:.1f} deg")
        quality.update(rim_points=len(rim), inner_radius_m=inner_r, outer_radius_m=outer_r,
                       circle_rms_m=outer.rms, text_used=phrase, mask_pixels=mask.pixels,
                       xy_source="top_plane_rays" if view.top_down else "depth_points")
        centre = [float(outer.center[0]), float(outer.center[1]), top]
        self._check_reprojection(view, mask.mask, centre)
        return OpeningEstimate(label=text, camera=camera, obs_id=obs.obs_id, center_xyz=centre, radius_m=radius,
                               rim_z=top, normal=normal.tolist(), method=method, quality=quality)

    def project(self, obs: Observation, camera: str, xyz: Sequence[float]) -> dict:
        view = self._view(obs, camera)
        uv, depth = geo.project_points(view.K, view.T, np.asarray(xyz, dtype=float).reshape(1, 3))
        h, w = view.shape
        return {"u": float(uv[0, 0]), "v": float(uv[0, 1]), "in_view": bool(geo.in_view(uv, depth, (w, h))[0]),
                "depth_m": float(depth[0]), "camera": camera, "obs_id": obs.obs_id}

    def pixel_to_world(self, obs: Observation, camera: str, u: float, v: float) -> list:
        view = self._view(obs, camera)
        return self._surface_point(view, self._depth(view), (u, v)).tolist()

    # ------------------------------------------------------------ pipeline

    def _locate(self, obs, text, camera, *, hint_px, max_objects, support_z, relevel, detail) -> list[ObjectEstimate]:
        if max_objects < 1:
            raise PerceptionError("max_objects must be at least 1")
        view = self._view(obs, camera)
        shared = self._quality()
        found = self._candidates(view, text, hint_px, shared)
        if hint_px is None and max_objects == 1 and len(found) > 1:
            shared["warnings"].append(f"ambiguous_instances: {len(found)} masks, highest score used; pass hint_px")
        candidates = found[:max_objects]
        results, failures = [], []
        for index, (mask, phrase) in enumerate(candidates):
            quality = {**shared, "warnings": list(shared["warnings"]), "text_used": phrase}
            try:
                results.append(self._measure(obs, view, text, mask, support_z, relevel, quality, detail))
            except PerceptionError as exc:
                if index == 0 and (hint_px is not None or max_objects == 1):
                    raise
                failures.append(f"mask at {np.round(geo.mask_centroid(mask.mask)).astype(int).tolist()}: {exc}")
        if not results:
            raise PerceptionError(f"no '{text}' could be measured in camera '{camera}': " + "; ".join(failures))
        if failures:
            results[0].quality["warnings"].extend(f"skipped {f}" for f in failures)
        if hint_px is None:
            results.sort(key=lambda r: r.score, reverse=True)
        return results

    def _measure(self, obs, view: _View, label: str, mask: Mask, support_z, relevel, quality: dict,
                 detail: bool) -> ObjectEstimate:
        prep = self._prepare(view, mask.mask, support_z, relevel, quality)
        top, above_core = self._top(view, prep, quality)
        support = prep.support_z
        above = prep.full.select(prep.full.xyz[:, 2] > support + ABOVE_SUPPORT_M)
        face = above.select(above.xyz[:, 2] >= top - FACE_BAND_M)
        if len(face) < 5:
            face = above
            quality["warnings"].append("top_face_sparse: centre from all points above support")
        face_xy = self._face_xy(view, face, top)
        face_rect = geo.min_area_rect(face_xy, FOOTPRINT_TRIM_PCT)
        if view.top_down:
            # Pixel centres fall up to half a pixel inside each edge of the face.
            pitch = float(np.median(prep.depth[face.rows, face.cols])) / view.K[0, 0]
            footprint = face_rect.grown(pitch)
            quality.update(footprint_source="top_plane_rays", pixel_pitch_m=pitch)
        else:
            footprint = geo.min_area_rect(above.xyz[:, :2], FOOTPRINT_TRIM_PCT)
            quality["footprint_source"] = "depth_points_above_support"
        centre = [float(face_rect.center[0]), float(face_rect.center[1]), top]
        self._check_reprojection(view, mask.mask, centre)
        grasp_yaw = grasp_yaw_for(footprint)
        if grasp_yaw is None:
            quality["warnings"].append("near_square_footprint: grasp yaw undefined")
        quality.update(points=len(above_core), face_points=len(face), footprint_points=len(above),
                       view_tilt_deg=view.tilt_deg, view="top_down" if view.top_down else "oblique")
        if detail:
            quality.update(self._shape_detail(footprint, face, above, face_xy))
        u, v = geo.mask_centroid(geo.fill_holes(mask.mask))
        return ObjectEstimate(
            label=label, camera=view.name, obs_id=obs.obs_id,
            center_xyz=[centre[0], centre[1], (top + support) / 2], top_z=top, height_m=top - support,
            length_m=footprint.length, width_m=footprint.width, long_axis_deg=footprint.angle_deg,
            grasp_yaw_deg=grasp_yaw, score=float(mask.score), pixel_centroid=[u, v], mask_pixels=mask.pixels,
            method="mask_depth_top_down" if view.top_down else "mask_depth_oblique", quality=quality)

    def _prepare(self, view: _View, mask: np.ndarray, support_z, relevel, quality: dict,
                 require_support: bool = True) -> _Prepared:
        depth = self._depth(view)
        if support_z is not None:
            support, source = float(support_z), "argument"
        elif self.table_z is not None:
            support, source = self.table_z, "table_z"
        else:
            support, source = None, "ring"
        offset = 0.0
        wanted = relevel if relevel is not None else view.name in self.relevel_cameras
        if support is None:
            support, reason = self._ring_support(view, depth, mask)
            if support is None and require_support:
                raise PerceptionError(f"support height unknown ({reason}); pass support_z or set Perceiver(table_z=...)")
            quality["relevel"] = "not_applicable: support measured on the ring"
        elif wanted:
            found, reason = self._ring_offset(view, depth, mask, support)
            if found is None:
                quality["relevel"] = f"skipped: {reason}"
                quality["warnings"].append(f"depth_not_relevelled: {reason}")
            else:
                offset = found
                depth = depth + offset
                quality["relevel"] = "applied"
        else:
            quality["relevel"] = "disabled"
        quality.update(support_z=support, support_source=source, depth_offset_m=offset)
        valid = geo.valid_depth(depth) & mask
        core_mask = geo.erode(mask, CORE_ERODE_PX) & valid
        if core_mask.sum() < MIN_POINTS:
            core_mask = valid
        full_mask = core_mask | (valid & self._within_core_depth(depth, mask, core_mask))
        quality["edge_pixels_dropped"] = int((valid & ~full_mask).sum())
        return _Prepared(depth, support if support is not None else float("nan"),
                         geo.depth_to_world(depth, view.K, view.T, full_mask),
                         geo.depth_to_world(depth, view.K, view.T, core_mask))

    def _top(self, view: _View, prep: _Prepared, quality: dict) -> tuple[float, geo.Points]:
        """Top height: a high percentile of the points above the support; from straight above, the
        median of the band just below it (the flat top face fills the view)."""
        above = prep.core.select(prep.core.xyz[:, 2] > prep.support_z + ABOVE_SUPPORT_M)
        if len(above) < MIN_POINTS:
            raise PerceptionError(f"only {len(above)} depth points above the support at z={prep.support_z:.4f}; "
                                  "the object is flat, occluded, or support_z is wrong")
        z = above.xyz[:, 2]
        top = float(np.percentile(z, TOP_PERCENTILE))
        if view.top_down:
            top = float(np.median(z[z >= top - TOP_BAND_M]))
            quality["top_source"] = "top_band_median"
        else:
            quality["top_source"] = f"p{TOP_PERCENTILE:g}"
        return top, above

    def _face_xy(self, view: _View, pts: geo.Points, top: float) -> np.ndarray:
        """XY of top-face points. From straight above, pixel rays meet the plane z=top, so the XY does not
        inherit depth noise; obliquely the depth points are used (rays would smear side faces outward)."""
        if not view.top_down:
            return pts.xyz[:, :2]
        origin, dirs = geo.pixel_rays(view.K, view.T, pts.cols, pts.rows)
        xy = geo.ray_plane_z(origin, dirs, top)[:, :2]
        return xy[np.isfinite(xy).all(axis=1)]

    def _ring_points(self, view: _View, depth: np.ndarray, mask: np.ndarray) -> geo.Points:
        ring = geo.ring_mask(mask, RING_INNER_PX, RING_OUTER_PX)
        return geo.depth_to_world(depth, view.K, view.T, ring)

    def _ring_offset(self, view: _View, depth: np.ndarray, mask: np.ndarray,
                     support: float) -> tuple[Optional[float], str]:
        """Optical-depth bias that puts the support seen around the mask at `support`.

        Each ring pixel's world Z changes by gain * bias, so the per-pixel bias is (support - z) / gain;
        the median is applied when the corrected ring is level and the bias is plausible.
        """
        ring = self._ring_points(view, depth, mask)
        gain = geo.depth_gain_z(view.K, view.T, ring.cols, ring.rows)
        near = (np.abs(ring.xyz[:, 2] - support) < RING_BAND_M) & (np.abs(gain) > 0.2)
        if near.sum() < RING_MIN_POINTS:
            return None, f"ring has {int(near.sum())} support pixels (< {RING_MIN_POINTS})"
        z, g = ring.xyz[near, 2], gain[near]
        bias = float(np.median((support - z) / g))
        spread = geo.iqr(z + bias * g)
        if spread > RING_MAX_SPREAD_M:
            return None, f"ring is uneven (IQR {spread * 1000:.1f} mm)"
        if abs(bias) > RING_MAX_OFFSET_M:
            return None, f"ring offset {bias * 1000:.1f} mm exceeds {RING_MAX_OFFSET_M * 1000:.0f} mm"
        return bias, "ok"

    def _ring_support(self, view: _View, depth: np.ndarray, mask: np.ndarray) -> tuple[Optional[float], str]:
        z = self._ring_points(view, depth, mask).xyz[:, 2]
        if len(z) < RING_MIN_POINTS:
            return None, f"ring has {len(z)} depth pixels"
        spread = geo.iqr(z)
        if spread > RING_MAX_SPREAD_M:
            return None, f"ring around the object is uneven (IQR {spread * 1000:.1f} mm)"
        return float(np.median(z)), "ok"

    def _shape_detail(self, footprint: geo.Rect, face: geo.Points, above: geo.Points, face_xy: np.ndarray) -> dict:
        hull = geo.convex_hull(face_xy)
        fill = geo.polygon_area(hull) / max(footprint.length * footprint.width, 1e-12)
        aspect = footprint.length / max(footprint.width, 1e-9)
        detail = {"footprint_corners": footprint.corners().tolist(), "aspect": aspect, "hull_fill": fill,
                  "top_flatness_m": float(np.std(face.xyz[:, 2])),
                  "height_percentiles": dict(zip(("p5", "p50", "p95"),
                                                 np.percentile(above.xyz[:, 2], [5, 50, 95]).tolist()))}
        shape = "elongated" if aspect >= 2.5 else "box"
        if len(hull) >= 8:
            circle = geo.fit_circle(hull)
            detail.update(circle_center=circle.center.tolist(), circle_radius_m=circle.radius,
                          circle_rms_m=circle.rms)
            if aspect < 1 + SQUARE_TOLERANCE and circle.rms < 0.05 * circle.radius and fill < 0.9:
                shape = "round"
        detail["shape"] = shape
        return detail

    # ------------------------------------------------------------ inputs

    def _view(self, obs: Observation, camera: str) -> _View:
        if camera not in obs.cameras:
            raise PerceptionError(f"camera {camera!r} not in observation ({sorted(obs.cameras)})")
        frame = obs.cameras[camera]
        if not frame.calibrated:
            raise PerceptionError(f"camera {camera!r} is uncalibrated (intrinsics/extrinsics missing); "
                                  "metric measurements need calibration")
        try:
            k, t = geo.check_calibration(frame.intrinsics, frame.extrinsics)
        except ValueError as exc:
            raise PerceptionError(f"camera {camera!r} calibration invalid: {exc}") from exc
        return _View(camera, frame, k, t)

    def _depth(self, view: _View) -> np.ndarray:
        depth = self.depth_source.depth(view.frame)
        if depth is None:
            raise PerceptionError(f"no depth available for camera {view.name!r}")
        depth = np.asarray(depth, dtype=float)
        if depth.shape != view.shape:
            raise PerceptionError(f"depth is {depth.shape}, image is {view.shape} for camera {view.name!r}")
        return depth

    def _within_core_depth(self, depth: np.ndarray, mask: np.ndarray, core: np.ndarray) -> np.ndarray:
        """Mask-edge pixels whose depth lies within the range of the eroded interior beside them.

        Depth blended across a silhouette (learned depth, stereo) puts edge pixels between the object
        and the background behind it, or toward a nearer occluder; those would distort the footprint
        and the top height, so they are dropped.
        """
        out = np.zeros(mask.shape, dtype=bool)
        rows, cols = np.nonzero(mask)
        if rows.size == 0:
            return out
        pad = CORE_ERODE_PX + 1
        r0, r1 = max(rows.min() - pad, 0), min(rows.max() + pad + 1, mask.shape[0])
        c0, c1 = max(cols.min() - pad, 0), min(cols.max() + pad + 1, mask.shape[1])
        crop = depth[r0:r1, c0:c1]
        interior = np.where(core[r0:r1, c0:c1], crop, np.nan)
        far, near = geo.local_max(interior, pad), -geo.local_max(-interior, pad)
        tolerance = np.maximum(EDGE_TOLERANCE_M, EDGE_TOLERANCE_REL * crop)
        with np.errstate(invalid="ignore"):
            out[r0:r1, c0:c1] = (crop <= far + tolerance) & (crop >= near - tolerance)
        return out

    def _masks(self, view: _View, text: str) -> list[Mask]:
        masks = self.segmenter.segment(view.frame.rgb, text, min_score=self.min_mask_score)
        out = []
        for m in masks:
            array = np.asarray(m.mask, dtype=bool)
            if array.shape != view.shape:
                raise PerceptionError(f"segmenter returned a {array.shape} mask for a {view.shape} image")
            if m.score >= self.min_mask_score and array.sum() >= MIN_MASK_PIXELS:
                out.append(Mask(float(m.score), array, tuple(m.box_xyxy)))
        return sorted(out, key=lambda m: m.score, reverse=True)

    def _candidates(self, view: _View, text: str, hint_px, quality: dict) -> list[tuple[Mask, str]]:
        """Masks for the text, best first. With a hint the mask covering it comes first; when no mask
        covers it, plainer phrasings are tried: '<colour at the hint> object', then 'object'."""
        if not isinstance(text, str) or not text.strip():
            raise PerceptionError("text must be a non-empty string")
        if hint_px is None:
            masks = self._masks(view, text)
            if not masks:
                raise PerceptionError(f"no mask for {text!r} in camera {view.name!r}")
            return [(m, text) for m in masks]
        h, w = view.shape
        u, v = float(hint_px[0]), float(hint_px[1])
        if not (np.isfinite([u, v]).all() and 0 <= u < w and 0 <= v < h):
            raise PerceptionError(f"hint pixel {list(hint_px)} outside the {w}x{h} image")
        phrases = list(dict.fromkeys([text, f"{colour_word(view.frame.rgb, (u, v))} object", "object"]))
        seen = []
        for index, phrase in enumerate(phrases):
            masks = self._masks(view, phrase)
            chosen = pick_instance(masks, (u, v)) if masks else None
            if chosen is None:
                seen.extend(f"{phrase!r}@{np.round(geo.mask_centroid(m.mask)).astype(int).tolist()}" for m in masks)
                continue
            if index:
                quality["warnings"].append(f"text_rephrased: {phrase!r}")
            ordered = [masks[chosen]] + [m for i, m in enumerate(masks) if i != chosen]
            return [(m, phrase) for m in ordered]
        raise PerceptionError(f"no mask covers hint pixel [{u:.0f}, {v:.0f}] for {phrases}"
                              + (f"; masks elsewhere: {', '.join(seen)}" if seen else ""))

    def _surface_point(self, view: _View, depth: np.ndarray, pixel: Sequence[float]) -> np.ndarray:
        h, w = view.shape
        u, v = int(round(float(pixel[0]))), int(round(float(pixel[1])))
        if not (0 <= u < w and 0 <= v < h):
            raise PerceptionError(f"pixel [{pixel[0]}, {pixel[1]}] outside the {w}x{h} image")
        d = depth[v, u]
        if not geo.valid_depth(np.array([d]))[0]:
            patch = depth[max(v - 1, 0):v + 2, max(u - 1, 0):u + 2]
            patch = patch[geo.valid_depth(patch)]
            if patch.size == 0:
                raise PerceptionError(f"no valid depth at pixel [{u}, {v}] in camera {view.name!r}")
            d = float(np.median(patch))
        ray = geo.camera_rays(view.K, [u], [v])[0]
        return geo.to_world(ray * d, view.T)[0]

    def _check_reprojection(self, view: _View, mask: np.ndarray, xyz: Sequence[float]) -> None:
        """The estimate must project back onto its own (hole-filled) mask, or it describes something else."""
        uv, depth = geo.project_points(view.K, view.T, np.asarray(xyz, dtype=float).reshape(1, 3))
        region = geo.dilate(geo.fill_holes(mask), REPROJECT_MARGIN_PX)
        h, w = mask.shape
        if not geo.in_view(uv, depth, (w, h))[0]:
            raise PerceptionError("estimate projects outside the image")
        u, v = int(round(uv[0, 0])), int(round(uv[0, 1]))
        if not region[v, u]:
            raise PerceptionError(f"estimate projects to pixel [{u}, {v}], outside its source mask")

    @staticmethod
    def _quality() -> dict:
        return {"warnings": []}
