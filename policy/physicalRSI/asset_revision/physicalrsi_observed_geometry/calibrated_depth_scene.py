"""Calibrate learned depth against support geometry observed in current RGB."""
import time
import cv2
import numpy as np
from . import Scene, GeometryUnavailable
from .rgb_support import initial_plane, refine_plane
from .dense_support import photometric_initial_plane
from .observed_depth_alignment import ObservedPlaneDepthAlignment


def calibrated_depth_scene(observation, *, predictor=None, segment_fn=None, prior_plane=None):
    scene = Scene.from_observation(observation, predictor=predictor)
    if len(scene._other_cameras) < 2:
        raise GeometryUnavailable('Current calibrated reference views required for depth scale')
    # Color proposes visible support pixels only. All metric quantities below
    # come from camera baselines and observed correspondences, never color.
    hsv = cv2.cvtColor(scene.camera.image, cv2.COLOR_RGB2HSV)
    proposal = (hsv[:, :, 0] < 35) & (hsv[:, :, 1] > 35) & (hsv[:, :, 2] > 40)
    cameras = [scene.camera, *scene._other_cameras]
    masks = [proposal] + [np.ones(c.image.shape[:2], bool) for c in cameras[1:]]
    if segment_fn is not None:
        from .semantic_support import observe_semantic_plane
        kwargs = {} if prior_plane is None else {'prior_plane': prior_plane}
        plane, observed = observe_semantic_plane(observation, segment_fn, **kwargs)
    else:
        try:
            seed, observed = initial_plane(cameras, masks, unique_head_pixels=True, return_support_mask=True)
            plane = refine_plane(cameras, [observed, *masks[1:]], seed, minimum_reference_views=1)
        except GeometryUnavailable:
            plane, observed = photometric_initial_plane(cameras, masks)
    try:
        scene.depth, report = ObservedPlaneDepthAlignment(
            camera=scene.camera, plane=plane, target='axial'
        ).align(scene._raw_depth, observed, deadline=time.monotonic()+30)
    except ValueError as exc:
        raise GeometryUnavailable(str(exc)) from exc
    scene.depth_calibration_report = report
    # Retain the very same observed support pixels for downstream collision
    # patches; do not invent table extent from the fitted infinite plane.
    scene.observed_support_mask = np.asarray(observed,bool).copy()
    scene.observed_support_plane = np.asarray(plane,float).copy()
    return scene


def observed_world_height(prediction, observation, *, segment_fn=None, support_callback=None, prior_plane=None):
    """World-Z image from current calibrated RGB, with no absolute-height clamp."""
    kwargs = {} if segment_fn is None else {'segment_fn': segment_fn}
    if prior_plane is not None:
        kwargs['prior_plane'] = prior_plane
    scene = calibrated_depth_scene(observation, predictor=lambda image: prediction, **kwargs)
    yy, xx = np.indices(scene.depth.shape)
    points = scene.camera.unproject(np.column_stack((xx.ravel(), yy.ravel())), scene.depth.ravel())
    if support_callback is not None:
        support_callback(scene.camera, scene.observed_support_plane, scene.observed_support_mask)
    return points[:, 2].reshape(scene.depth.shape)


def observed_surface_height(values):
    """Summarize current metric surface samples without a task-height prior."""
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise GeometryUnavailable('Finite nonempty observed surface samples required')
    return float(np.median(values))


def observed_pixel_height(height_map, pixel):
    """Robust local height at a current image target, without a world-height prior."""
    heights = np.asarray(height_map, dtype=float)
    uv = np.asarray(pixel, dtype=float)
    if heights.ndim != 2 or uv.shape != (2,) or not np.isfinite(uv).all():
        raise GeometryUnavailable('Current height image and finite target pixel required')
    u, v = np.rint(uv).astype(int)
    h, w = heights.shape
    if not (0 <= u < w and 0 <= v < h):
        raise GeometryUnavailable('Target pixel lies outside current height image')
    return observed_surface_height(heights[max(0,v-3):min(h,v+4), max(0,u-3):min(w,u+4)])
