"""Validate an episode-measured stationary pose against visible RGB evidence."""
from itertools import product
import cv2
import numpy as np
from . import GeometryUnavailable


def verify_partial_box(camera, center, rotation, dimensions, observed_mask, occluder_mask):
    """Retain a pose only when enough of its predicted silhouette is observed.

    This is a consistency check, not a new depth measurement. Unexplained missing
    pixels (including robot occlusion) cause rejection. Dimensions and pose must
    come from observations in the current episode.
    """
    points=np.asarray(list(product([-.5,.5],repeat=3)))*np.asarray(dimensions)
    pixels,depth=camera.project(points@np.asarray(rotation).T+np.asarray(center))
    if np.any(depth<=0) or not np.isfinite(pixels).all():
        raise GeometryUnavailable('Box is not in front of the camera')
    predicted=np.zeros(camera.image.shape[:2],np.uint8)
    cv2.fillConvexPoly(predicted,cv2.convexHull(np.rint(pixels).astype(np.int32)),1)
    observed=np.asarray(observed_mask,bool);occluder=np.asarray(occluder_mask,bool)
    visible=predicted.astype(bool)&~occluder
    count=int(visible.sum())
    if count<20 or count<.25*int(predicted.sum()) or observed.sum()<20:
        raise GeometryUnavailable('Insufficient unoccluded box evidence')
    kernel=np.ones((5,5),np.uint8)
    near_observed=cv2.dilate(observed.astype(np.uint8),kernel).astype(bool)
    near_visible=cv2.dilate(visible.astype(np.uint8),kernel).astype(bool)
    recall=float(near_observed[visible].mean())
    precision=float(near_visible[observed].mean())
    if min(recall,precision)<.95:
        raise GeometryUnavailable('Visible RGB contradicts previous box pose')
    return {'visible_pixels':count,'recall':recall,'precision':precision,
            'source':'current_rgb_partial_silhouette_consistency'}
