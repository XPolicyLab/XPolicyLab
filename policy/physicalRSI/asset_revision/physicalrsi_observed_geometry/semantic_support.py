"""Current semantic support masks with independently validated metric scale."""
import numpy as np
from . import GeometryUnavailable
from .support_observation import observe_support_masks
from .rgb_support import correspondences, support_correspondences, initial_plane, refine_plane
from .dense_support import photometric_plane_from_proposals


def observe_semantic_plane(observation, segment_fn, *, prior_plane=None):
    try:
        kwargs = {} if prior_plane is None else {'prior_plane': prior_plane}
        return _observe_semantic_plane(observation, segment_fn, **kwargs)
    except GeometryUnavailable as error:
        from .geometry_evidence import capture_failure
        try:
            capture_failure(observation, error)
        except OSError as capture_error:
            import warnings
            warnings.warn(f'Could not save geometry failure evidence: {capture_error}')
        raise


def _observe_semantic_plane(observation, segment_fn, *, prior_plane=None):
    if prior_plane is not None:
        prior_plane = np.asarray(prior_plane, float)
        if prior_plane.shape != (3,) or not np.isfinite(prior_plane).all():
            raise GeometryUnavailable('Finite previously observed support seed required')
    cameras, masks = observe_support_masks(
        observation, segment_fn, occluders=('robot arm', 'basket'))
    cache = {}
    for camera, mask in zip(cameras[1:], masks[1:]):
        cache[id(camera)] = correspondences(cameras[0], camera, masks=(masks[0], mask))
    def matcher(head, reference):
        return cache[id(reference)]
    try:
        seed, visible = initial_plane(cameras, masks, matcher,
                                      unique_head_pixels=True, return_support_mask=True)
        plane = refine_plane(cameras, [visible, *masks[1:]], seed,
                             minimum_reference_views=1)
    except GeometryUnavailable:
        points, _ = support_correspondences(cameras, masks, matcher)
        if prior_plane is not None:
            # A measurement from earlier in this episode supplies an initial
            # hypothesis only. Current RGB masks and all validation gates below
            # are still required, even when today's sparse matching is empty.
            points = np.vstack([np.asarray(points).reshape(-1,3), [0.,0.,prior_plane[2]]])
        if not len(points):
            raise GeometryUnavailable('No triangulated semantic support proposal')
        # These offsets explore local optimizer initialization, never define
        # usable geometry. Every proposal must pass the unchanged held-out
        # two-view, split-agreement and competing-plane ambiguity gates.
        proposals = np.concatenate([
            points + np.array([0., 0., delta]) for delta in np.linspace(-.03, .03, 7)])
        plane, visible = photometric_plane_from_proposals(
            cameras, masks, proposals, seed_spacing=.005)
    return plane, visible
