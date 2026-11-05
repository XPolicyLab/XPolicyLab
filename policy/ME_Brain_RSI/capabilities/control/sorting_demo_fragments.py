"""Join one unambiguous extra demonstration contour without changing source objects."""
import ast
import hashlib
import inspect
import itertools
import linecache
from pathlib import Path
import textwrap

import numpy as np

from coap.tasks.memory import initial_object_groups
from coap.tasks.memory_initial import initial_objects as original_initial_objects
from coap.tasks.memory_initial import _camera, _contour_gap, _merge_head_parts


def initial_objects(ctx):
    """Repair only one extra head-owned demonstration after the retained initializer."""
    objects = original_initial_objects(ctx)
    sources, demonstrations, unused = initial_object_groups(objects)
    review = dict(method='one_extra_demonstration_unique_head_fragment_pair',
        before=len(objects), source_count=len(sources), demonstration_count=len(demonstrations),
        applied=False, execution_exit_added=False)
    legal = []
    if (len(objects), len(sources), len(demonstrations)) == (11, 5, 6):
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
        review['legal_pair_count'] = len(legal)
        if len(legal) == 1:
            first, second, gap, distance, color = legal[0]
            merged = _merge_head_parts(ctx, first, second)
            objects = [item for item in objects if item is not first and item is not second]+[merged]
            review.update(applied=True, first_bbox=tuple(first['bbox']),
                second_bbox=tuple(second['bbox']), merged_bbox=tuple(merged['bbox']),
                contour_gap_px=float(gap), position_distance_m=distance,
                mean_rgb_distance=color, after=len(objects),
                source='normal_head_contours_with_existing_numeric_fragment_criteria')
        else:
            review['reason'] = 'nonunique_or_absent_visual_fragment_association'
    else:
        review['reason'] = 'outside_one_extra_demonstration_scope'
    observations = getattr(ctx, 'sorting_initial_observations', [])
    if observations and isinstance(observations[-1], dict):
        observations[-1]['demonstration_fragment_merge'] = review
        if review['applied']:
            observations[-1]['retained_objects'] = len(objects)
    return objects


def prepare_core(replay):
    """Replace only the local initialization import in a private replay function."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(replay)))
    count = 0
    for node in ast.walk(tree):
        if (isinstance(node, ast.ImportFrom) and node.module == 'memory_initial'
                and any(alias.name == 'initial_objects' for alias in node.names)):
            node.module = 'coap.tasks.sorting_demo_fragments'
            node.level = 0
            count += 1
    if count != 1:
        raise ValueError('Expected one retained local initial_objects import')
    rendered = ast.unparse(ast.fix_missing_locations(tree))+'\n'
    digest = hashlib.sha256(rendered.encode()).hexdigest()[:16]
    filename = str(Path(__file__).with_name('_sorting_demo_fragments_'+digest+'.py'))
    linecache.cache[filename] = (len(rendered), None, rendered.splitlines(True), filename)
    namespace = dict(replay.__globals__)
    exec(compile(rendered, filename, 'exec'), namespace)
    result = namespace[replay.__name__]
    result.__defaults__ = replay.__defaults__
    result.__kwdefaults__ = replay.__kwdefaults__
    return result
