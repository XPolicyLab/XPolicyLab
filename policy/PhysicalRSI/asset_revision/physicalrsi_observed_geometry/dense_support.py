"""Validate sparse RGB plane proposals using held-out calibrated image evidence."""
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from . import GeometryUnavailable
from .multiview import projection
from .rgb_support import correspondences, support_correspondences, refine_plane


def _contrast(camera):
    gray = cv2.cvtColor(camera.image, cv2.COLOR_RGB2GRAY).astype(float) / 255
    delta = gray - gaussian_filter(gray, 3)
    return delta / np.sqrt(gaussian_filter(delta * delta, 3) + .0001)


def _evidence(cameras, masks, plane, validation):
    source = cameras[0]
    gray = _contrast(source)
    yy, xx = np.nonzero(validation)
    if len(xx) > 6000:
        selected = np.random.default_rng(273).choice(len(xx), 6000, replace=False)
        xx, yy = xx[selected], yy[selected]
    origin = source.transform[:3, 3]
    rays = source.unproject(np.c_[xx, yy], np.ones(len(xx))) - origin
    a, b, c = plane
    denominator = rays[:, 2] - a * rays[:, 0] - b * rays[:, 1]
    with np.errstate(divide='ignore', invalid='ignore'):
        distance = (c-origin[2]+a*origin[0]+b*origin[1]) / denominator
        world = origin + distance[:, None] * rays
    correlations, visible = [], np.zeros(len(xx), int)
    for camera, mask in zip(cameras[1:], masks[1:]):
        hom = np.c_[world, np.ones(len(world))] @ projection(camera).T
        with np.errstate(divide='ignore', invalid='ignore'):
            uv = hom[:, :2] / hom[:, 2:]
        valid = (np.isfinite(uv).all(1) & (distance > 0) & (hom[:, 2] > 0)
                 & (uv[:, 0] > 4) & (uv[:, 0] < camera.width-5)
                 & (uv[:, 1] > 4) & (uv[:, 1] < camera.height-5))
        index = np.flatnonzero(valid)
        index = index[mask[np.rint(uv[index, 1]).astype(int), np.rint(uv[index, 0]).astype(int)]]
        if len(index) < 200:
            continue
        reference = gray[yy[index], xx[index]]
        target = map_coordinates(_contrast(camera), [uv[index, 1], uv[index, 0]], order=1, mode='nearest')
        if min(np.std(reference), np.std(target)) < .05:
            continue
        correlation = float(np.corrcoef(reference, target)[0, 1])
        if np.isfinite(correlation) and correlation >= .5:
            correlations.append(correlation)
            visible[index] += 1
    if len(correlations) < 2 or (visible >= 2).sum() < 200:
        raise GeometryUnavailable('Plane lacks independent textured validation in two views')
    return float(min(correlations)), world[visible >= 2]


def photometric_initial_plane(cameras, masks, matcher=correspondences, *, proposal_error_px=1.):
    """Sparse points propose metric scale; separate image tiles verify the fit.

    No preset height is used. Two independent tile splits must recover the same
    surface. Untextured or ambiguous images remain unresolved.
    """
    if not np.isfinite(proposal_error_px) or proposal_error_px<1.:
        raise GeometryUnavailable('Proposal fallback tolerance must be finite and at least one pixel')
    points, _ = support_correspondences(cameras, masks, matcher)
    if not len(points) and proposal_error_px>1.:
        points, _ = support_correspondences(cameras, masks, matcher,
                                          max_reprojection_error_px=proposal_error_px)
    return photometric_plane_from_proposals(cameras,masks,points)


def photometric_plane_from_proposals(cameras, masks, points, *, seed_spacing=.02):
    """Validate initialization proposals against independent current RGB tiles.

    Proposal uncertainty never changes the two-view evidence, split agreement,
    or ambiguity gates below. Proposals alone are not usable metric geometry.
    """
    points=np.asarray(points,float)
    if not len(points):
        raise GeometryUnavailable('No calibrated metric plane proposal')
    if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():
        raise GeometryUnavailable('Finite three-dimensional initialization proposals required')
    if not np.isfinite(seed_spacing) or seed_spacing<=0:
        raise GeometryUnavailable('Positive finite proposal spacing required')
    yy, xx = np.indices(masks[0].shape)
    # Leave a guard band between tiles so local contrast windows do not share
    # pixels across the fitting/validation boundary.
    inner = (xx % 64 >= 13) & (xx % 64 < 51) & (yy % 64 >= 13) & (yy % 64 < 51)
    parity = (xx // 64 + yy // 64) % 2
    tiles = [np.asarray(masks[0], bool) & inner & (parity == i) for i in (0, 1)]
    seeds = []
    for point in points:
        if all(abs(point[2]-seed) >= seed_spacing for seed in seeds):
            seeds.append(float(point[2]))
    accepted = []
    for height in seeds:
        fits, evidence = [], []
        try:
            for split in (0, 1):
                fit = refine_plane(cameras, [tiles[split], *masks[1:]],
                                   np.array([0., 0., height]), minimum_reference_views=2)
                score, observed = _evidence(cameras, masks, fit, tiles[1-split])
                fits.append(fit)
                evidence.append((score, observed))
            xy = np.concatenate([entry[1][:, :2] for entry in evidence])
            design = np.c_[xy, np.ones(len(xy))]
            if np.max(np.abs(design @ (fits[0]-fits[1]))) > .003:
                continue
            accepted.append((min(entry[0] for entry in evidence), np.mean(fits, axis=0), xy))
        except GeometryUnavailable:
            continue
    if not accepted:
        raise GeometryUnavailable('No independently validated photometric support plane')
    accepted.sort(key=lambda item: item[0], reverse=True)
    score, plane, xy = accepted[0]
    for other_score, other_plane, _ in accepted[1:]:
        if np.max(np.abs(np.c_[xy, np.ones(len(xy))] @ (plane-other_plane))) > .005:
            raise GeometryUnavailable('Multiple photometric support planes remain plausible')
    # Extent comes from validated overlapping observations, not a supplied table.
    xyz = np.c_[xy, xy @ plane[:2] + plane[2]]
    pixels, _ = cameras[0].project(xyz)
    hull = cv2.convexHull(np.rint(pixels).astype('int32'))
    observed = np.zeros(masks[0].shape, np.uint8)
    cv2.fillConvexPoly(observed, hull, 1)
    return plane, observed.astype(bool) & masks[0]
