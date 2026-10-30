"""Refine one extra demonstration silhouette after the formal color initializer."""
import itertools
from types import FunctionType

import numpy as np

from coap.tasks.memory import initial_object_groups
from coap.tasks.memory_initial import _camera, _contour_gap, _merge_head_parts


def refine_objects(robot, objects):
    """Extend the existing fragment criteria to the displaced six/five split."""
    sources, demonstrations, unused = initial_object_groups(objects)
    if (len(objects), len(sources), len(demonstrations)) != (11, 6, 5):
        return objects
    legal = []
    for first, second in itertools.combinations(demonstrations, 2):
        if _camera(first) != 'cam_head' or _camera(second) != 'cam_head':
            continue
        gap = _contour_gap(first, second)
        distance = float(np.linalg.norm(np.asarray(first['position'])[:2]
                                        - np.asarray(second['position'])[:2]))
        color = float(np.linalg.norm(np.asarray(first['mean_rgb'])
                                     - np.asarray(second['mean_rgb'])))
        if gap < 20 and distance < .10 and color < 65:
            legal.append((first, second, gap, distance, color))
    if len(legal) != 1:
        return objects
    first, second, gap, distance, color = legal[0]
    merged = _merge_head_parts(robot, first, second)
    result = [item for item in objects if item is not first and item is not second]+[merged]
    records = getattr(robot, 'sorting_initial_observations', [])
    if records:
        records[-1]['cross_boundary_demonstration_fragment'] = dict(
            before=11, after=10, first_bbox=tuple(first['bbox']), second_bbox=tuple(second['bbox']),
            merged_bbox=tuple(merged['bbox']), contour_gap_px=float(gap),
            position_distance_m=distance, mean_rgb_distance=color,
            source='normal_head_contours_with_unchanged_existing_fragment_criteria',
            additional_actions=0)
        records[-1]['retained_objects'] = len(result)
    return result


def finish_core(ctx, formal_core):
    """Preserve the existing initializer before a task-local post-refinement."""
    original = formal_core.__globals__['sorting_color_initial']

    def initialize(robot, objects, *args, **kwargs):
        return refine_objects(robot, original(robot, objects, *args, **kwargs))

    namespace = dict(formal_core.__globals__, sorting_color_initial=initialize)
    result = FunctionType(formal_core.__code__, namespace, formal_core.__name__,
                          formal_core.__defaults__, formal_core.__closure__)
    result.__kwdefaults__ = formal_core.__kwdefaults__
    return result
