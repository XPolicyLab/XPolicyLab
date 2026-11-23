from martingale import self_play_constants as sp
# Copyright © 2026 Martingale. All Rights Reserved.
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MaskView:
    episode: str
    step: int
    camera: str
    intrinsic: np.ndarray
    camera_to_world: np.ndarray
    mask: np.ndarray
    robot_mask: np.ndarray

    def __post_init__(self):
        if not self.episode or type(self.step) is not int or self.step < 0:
            raise ValueError("Explicit episode and nonnegative observation step required")
        if self.camera not in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
            raise ValueError("Unknown camera")
        for name, shape in (("intrinsic", (3, 3)), ("camera_to_world", (4, 4))):
            value = np.array(getattr(self, name), dtype=float, copy=True)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError("Invalid " + name)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        k, pose = self.intrinsic, self.camera_to_world
        if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1]):
            raise ValueError("Invalid pinhole intrinsics")
        rot = pose[:3, :3]
        if (not np.allclose(pose[3], [0, 0, 0, 1])
                or not np.allclose(rot.T @ rot, np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(rot), 1, atol=1e-5)):
            raise ValueError("Expected rigid camera-to-world transform")
        for name in ("mask", "robot_mask"):
            raw = np.asarray(getattr(self, name))
            if raw.dtype != np.bool_ or raw.shape != (480, 640):
                raise ValueError("Expected native-resolution boolean " + name)
            value = raw.copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)


def measure_static_pair(first, second, *, stationarity_basis):
    """Estimate table-supported geometry; no actions or success verdict.

    Inputs are two MaskViews with their ORIGINAL camera poses and
    robot occlusion masks. Do not mix an old image with a current robot mask.
    Camera coordinates follow Isaac's -Z optical axis, metres. The existing
    public vision library supplies its calibrated table plane and shape model.

    stationarity_basis must describe the caller's observation-based review of
    the intervening motion. None returns unresolved without performing a fit.
    A supplied explanation is an assumption, NOT verified object stationarity.
    A caller must discard the pair after contact, drift or uncertain tracking;
    never use this to carry geometry across resets or unreviewed manipulation.

    Returns status, source steps, assumptions and optional world-frame geometry.
    'estimated' is not permission to move or proof of seating/collision clearance.
    Unsupported/mismatched views return unresolved; malformed inputs raise.
    """
    if not isinstance(first, MaskView) or not isinstance(second, MaskView):
        raise TypeError("MaskView inputs required")
    if first.episode != second.episode or first.step >= second.step:
        raise ValueError("Need ordered observations within one episode")
    result = dict(status="unresolved", geometry=None,
                  source_steps=[first.step, second.step],
                  source_cameras=[first.camera, second.camera],
                  stationarity_verified=False, episode_actions_taken=0)
    if not isinstance(stationarity_basis, str) or not stationarity_basis.strip():
        return dict(result, reason="stationarity_not_reviewed")
    result["stationarity_assumption"] = stationarity_basis
    if (not first.mask.any() or not second.mask.any()
            or np.linalg.norm(first.camera_to_world[:3, 3]
                              - second.camera_to_world[:3, 3]) < .01):
        return dict(result, reason="empty_mask_or_insufficient_baseline")
    from martingale.hosts import robodojo_sight as sight
    from martingale.hosts.robodojo_tools import object_geometry
    def view(frame):
        return sight.Sight(sight.View(frame.intrinsic, frame.camera_to_world),
                           frame.mask, frame.robot_mask)
    fit = sight.measure(view(first), [[view(second)]])
    if fit is None or fit["views"] < 2 or fit.get("match") is None or fit["match"] < .9:
        return dict(result, reason="views_do_not_agree")
    geometry = object_geometry(np.asarray(fit["points"], dtype=float))
    if geometry is None:
        return dict(result, reason="geometry_unresolved")
    return dict(result, status="estimated", geometry=geometry,
                mask_agreement=float(fit["match"]),
                limitations="Static table-supported shape assumption; mask agreement is not accuracy or clearance")


def measure_planar_pair(first, second, *, plane_z, stationarity_basis,
                        max_center_disagreement_m=sp.PLANAR_CENTER_AGREEMENT_M):
    """Locate a flat marked region from two masks, without robot actions.

    plane_z is the caller's calibrated horizontal support plane in world metres. Masks must identify the same flat marking, not a
    raised object. Original per-view calibration and robot masks are required.
    Returns XY bounds/center from 1st/99th projected-pixel percentiles and both
    estimates. Agreement checks consistency, not metric accuracy or clearance.
    A wrong shared plane/calibration may bias both views. Region identity and
    stationarity remain caller assumptions. Unresolved results return to the agent.
    """
    if not isinstance(first, MaskView) or not isinstance(second, MaskView):
        raise TypeError('MaskView inputs required')
    if first.episode != second.episode or first.step >= second.step:
        raise ValueError('Need ordered observations within one episode')
    if not np.isfinite(plane_z) or not np.isfinite(max_center_disagreement_m) or max_center_disagreement_m <= 0:
        raise ValueError('Finite plane and positive disagreement tolerance required')
    result = dict(status='unresolved', source_steps=[first.step, second.step],
                  stationarity_verified=False, episode_actions_taken=0)
    if not isinstance(stationarity_basis, str) or not stationarity_basis.strip():
        return dict(result, reason='stationarity_not_reviewed')
    result['stationarity_assumption'] = stationarity_basis
    if np.linalg.norm(first.camera_to_world[:3, 3] - second.camera_to_world[:3, 3]) < .01:
        return dict(result, reason='insufficient_baseline')
    from martingale.hosts import robodojo_sight as sight
    estimates = []
    projected = []
    for frame in (first, second):
        rows, cols = np.where(frame.mask & ~frame.robot_mask)
        view = sight.View(frame.intrinsic, frame.camera_to_world)
        xy, valid = view.on_plane(rows, cols, float(plane_z))
        xy = xy[valid]
        if len(xy) < 20:
            return dict(result, reason='insufficient_visible_plane_pixels')
        projected.append(xy)
        lo, hi = np.quantile(xy, sp.PLANAR_EXTENT_QUANTILES, axis=0)
        if np.any(hi - lo < .005):
            return dict(result, reason='degenerate_region')
        estimates.append(dict(center_xy=((lo + hi) / 2).tolist(),
                              lower_xy=lo.tolist(), upper_xy=hi.tolist()))
    delta = float(np.linalg.norm(np.array(estimates[0]['center_xy']) - estimates[1]['center_xy']))
    result.update(per_view=estimates, center_disagreement_m=delta, plane_z=float(plane_z))
    if delta > max_center_disagreement_m:
        def border(mask):
            return bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
        lo = np.asarray(estimates[0]['lower_xy'])
        hi = np.asarray(estimates[0]['upper_xy'])
        def edges(points):
            d = np.column_stack([abs(points[:, 0]-lo[0]), abs(points[:, 0]-hi[0]),
                                 abs(points[:, 1]-lo[1]), abs(points[:, 1]-hi[1])])
            inside = ((points >= lo-.005) & (points <= hi+.005)).all(axis=1)
            spans = []
            for side in range(4):
                hit = points[(d[:, side] <= .005) & inside]
                axis = 1 if side < 2 else 0
                spans.append(float(np.ptp(hit[:, axis])/(hi[axis]-lo[axis])) if len(hit) >= 8 else 0.)
            return dict(fraction=float(((d.min(axis=1) <= .005) & inside).mean()), spans=spans)
        ref, partial = edges(projected[0]), edges(projected[1])
        result['outline_consistency'] = dict(reference=ref, secondary=partial)
        if (not border(first.mask) and border(second.mask)
                and ref['fraction'] >= .85 and min(ref['spans']) >= .7
                and partial['fraction'] >= .9
                and sum(x >= .4 for x in partial['spans']) >= 3):
            return dict(result, status='estimated', center_xy=estimates[0]['center_xy'],
                        estimate_source='complete_reference_with_three_cropped_edges',
                        limitations='Axis-aligned rectangular marking assumed; partial image validates visible edges only. Shared calibration/plane bias remains possible; no accuracy or clearance guarantee')
        return dict(result, reason='views_do_not_agree')
    return dict(result, status='estimated',
                center_xy=np.mean([e['center_xy'] for e in estimates], axis=0).tolist(),
                limitations='Flat-region identity, support plane and stationarity assumed; not an accuracy or collision-clearance guarantee')
