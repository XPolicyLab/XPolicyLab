"""Approximate an upright rotational silhouette using the current RGB view.

This is a view-derived radial model, not a measurement of hidden surfaces.
The lower silhouette midpoint is assumed to touch the observed support.
"""
import cv2
import numpy as np
from . import GeometryUnavailable
from .upright_contour import refine_mask, support_height


def validate_profile(profile):
    profile = np.asarray(profile, float)
    if (profile.ndim != 2 or profile.shape[1] != 2 or len(profile) < 3
            or not np.isfinite(profile).all() or np.any(profile[:, 1] <= 0)
            or np.any(np.diff(profile[:, 0]) <= 0) or abs(profile[0, 0]) > 1e-8):
        raise GeometryUnavailable('Observed increasing-height radial profile required')
    return profile


def profile_surface(profile):
    profile = validate_profile(profile)
    angle = np.linspace(0, 2*np.pi, 32, endpoint=False)
    return np.array([np.c_[r*np.cos(angle), r*np.sin(angle), np.full(32, z)] for z, r in profile])


def observed_neck_height(profile):
    profile = validate_profile(profile)
    height = profile[-1, 0]
    selected = (profile[:, 0] > .4*height) & (profile[:, 0] < .95*height) & (profile[:, 1] < .65*profile[:, 1].max())
    groups = np.split(np.flatnonzero(selected), np.flatnonzero(np.diff(np.flatnonzero(selected)) > 1)+1)
    groups = [group for group in groups if len(group) >= 2]
    if not groups:
        raise GeometryUnavailable('Observed contour has no resolved narrow upper grasp band')
    group = max(groups, key=lambda group: profile[group[-1], 0]-profile[group[0], 0])
    return float((profile[group[0], 0]+profile[group[-1], 0])/2)


def render_profile(profile, rotation, translation, camera, shape=(120, 160)):
    points = profile_surface(profile)
    xyz = points @ np.asarray(rotation).T + np.asarray(translation)
    pixels = camera.project(xyz.reshape(-1, 3))[0].reshape(*points.shape[:2], 2)
    pixels /= np.array([camera.width / shape[1], camera.height / shape[0]])
    pixels = np.rint(pixels).astype(np.int32)
    mask = np.zeros(shape, np.uint8)
    for a, b in zip(pixels[:-1], pixels[1:]):
        for i in range(32):
            j = (i+1) % 32
            cv2.fillConvexPoly(mask, np.array([a[i], a[j], b[j], b[i]]), 1)
    cv2.fillConvexPoly(mask, pixels[0], 1)
    cv2.fillConvexPoly(mask, pixels[-1], 1)
    return mask


def observe_profile(camera, mask, support_plane):
    mask = np.asarray(mask, bool)
    observed = refine_mask(mask, {}, camera=camera, plane=support_plane)
    if observed is None or not np.isfinite(observed['height_m']) or observed['height_m'] <= 0:
        raise GeometryUnavailable('Complete upright contour required for radial approximation')
    base = np.r_[observed['xy'], support_height(support_plane, observed['xy'])]
    origin = camera.transform[:3, 3]
    forward = base[:2] - origin[:2]
    norm = np.linalg.norm(forward)
    if norm < 1e-8:
        raise GeometryUnavailable('Overhead view cannot resolve radial profile')
    normal = np.r_[forward/norm, 0.]
    lateral = np.cross([0., 0., 1.], normal)
    contour = np.asarray(observed['contour'], float)
    rays = camera.unproject(contour, np.ones(len(contour))) - origin
    denominator = rays @ normal
    if np.any(np.abs(denominator) < 1e-8):
        raise GeometryUnavailable('Contour rays parallel to vertical profile plane')
    def candidate(offset, side=0.):
        center = base + normal * offset + lateral * side
        center[2] = support_height(support_plane, center[:2])
        distance = ((center-origin) @ normal) / denominator
        if np.any(distance <= 0):
            raise GeometryUnavailable('Profile plane behind camera')
        points = origin + rays * distance[:, None] - center
        height = float(points[:, 2].max())
        if not np.isfinite(height) or height <= 0:
            raise GeometryUnavailable('Unresolved observed profile height')
        levels = np.linspace(0, height, 17)
        widths = []
        for lower, upper in zip(levels[:-1], levels[1:]):
            selected = (points[:, 2] >= lower) & (points[:, 2] <= upper)
            cross = points[selected] @ lateral
            if len(cross) < 2 or cross.min() >= 0 or cross.max() <= 0:
                raise GeometryUnavailable('Both sides of each radial band must be visible')
            widths.append(float((cross.max()-cross.min()) / 2))
        middle = (levels[:-1]+levels[1:])/2
        profile = np.c_[levels, np.interp(levels, middle, widths)]
        rendered = render_profile(profile, np.eye(3), center, camera, mask.shape).astype(bool)
        iou = float(np.count_nonzero(rendered & mask)/max(np.count_nonzero(rendered | mask), 1))
        return iou, center, height, profile
    # The bottom silhouette belongs to the front rim, not the central axis.
    # Search the center displacement using only the observed apparent width.
    seed_distance = ((base-origin) @ normal) / denominator
    seed_points = origin + rays * seed_distance[:, None] - base
    radius = float(np.ptp(seed_points @ lateral) / 2)
    fits = []
    # Off-axis views need a lateral center search too: the image's lower
    # midpoint need not be the projection of a radial object's central axis.
    for offset in np.linspace(0., 2*radius, 25):
        for side in np.linspace(-radius, radius, 17):
            try:
                fits.append(candidate(float(offset), float(side)))
            except GeometryUnavailable:
                continue
    if not fits:
        raise GeometryUnavailable('No radial center explains both visible contour sides')
    best_before = max(fits, key=lambda item: item[0])
    # A rim's upper silhouette lies off the central vertical plane. Refine
    # the inferred axial extent against the complete projection instead of
    # treating the seed's top-ray intersection as the physical height.
    refined = []
    for _, center, candidate_height, candidate_profile in sorted(fits, key=lambda item: item[0], reverse=True)[:6]:
        for scale in np.linspace(.8, 1.15, 71):
            adjusted = candidate_profile * np.array([scale, 1.])
            rendered = render_profile(adjusted, np.eye(3), center, camera, mask.shape).astype(bool)
            overlap = float(np.count_nonzero(rendered & mask)/max(np.count_nonzero(rendered | mask), 1))
            refined.append((overlap, center, candidate_height*scale, adjusted))
    fits = refined
    iou, base, height, profile = max(fits, key=lambda item: item[0])
    if iou < .8:
        raise GeometryUnavailable('Observed radial approximation does not explain silhouette')
    comparable_heights = [item[2] for item in fits if item[0] >= iou-.01]
    if np.ptp(comparable_heights) > .1*height:
        raise GeometryUnavailable('Radial height ambiguous across comparable silhouette fits')
    return {'profile': profile.tolist(), 'base': base.tolist(), 'height': height,
            'source': 'current_RGB_contour_and_observed_support',
            'evidence': {**observed['height_evidence'], 'silhouette_iou': iou, 'competitive_height_spread_m': float(np.ptp(comparable_heights)),
                         'iou_before_axial_refinement': float(best_before[0]),
                         'rotational_symmetry_assumed': True,
                         'vertical_profile_plane_approximation': True,
                         'hidden_geometry_verified': False}}
