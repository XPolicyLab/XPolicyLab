"""Supplement a missing toy-dependent car ROI with the current raw head body."""
import ast
import copy
import hashlib
import inspect
import linecache
from pathlib import Path
import textwrap
from types import FunctionType

import cv2
import numpy as np

from coap.tasks.memory import compact_pale_car


FRAGMENT_AREA_RATIO = .45


def _head_region(obj):
    """Read the selected region in the stationary head camera."""
    views = obj.get('views', {})
    if 'cam_head' in views:
        return views['cam_head']
    return obj if obj.get('camera') == 'cam_head' else None


def current_car_body_fallback(ctx, obj):
    """Keep the acquired car semantics only for a nonfragment current body."""
    updates = getattr(ctx.memory, 'source_updates', [])
    sources = getattr(ctx, 'sorting_sources', {})
    reference = sources.get(updates[-1].get('demonstration_id')) if updates else None
    if reference is None:
        return None
    previous = getattr(ctx, '_sorting_current_car_raw', None)
    if previous is not None and previous.get('step') == ctx.steps:
        return None
    reference_head = (reference if reference.get('camera') == 'cam_head'
                      else reference.get('views', {}).get('cam_head'))
    selected_head = _head_region(obj)
    if reference_head is None or selected_head is None:
        return None
    initial_car = bool(compact_pale_car(reference))
    reference_bbox = tuple(map(int, reference_head['bbox']))
    x, y, width, height = map(int, selected_head['bbox'])
    selected_pixel = (float(x+width/2), float(y+height/2))
    anchor = np.asarray(obj['position'], float)
    matches = []
    for candidate in ctx.perception.detect(ctx.obs, 'object', 'cam_head', fuse=False):
        distance = float(np.linalg.norm(np.asarray(candidate['position'])[:2]-anchor[:2]))
        contour = np.asarray(candidate['contour'], np.float32).reshape(-1, 1, 2)
        if distance >= .08 or cv2.pointPolygonTest(contour, selected_pixel, False) < 0:
            continue
        matches.append((candidate, distance))
    if len(matches) != 1:
        return None
    current, distance = matches[0]
    bbox = tuple(map(int, current['bbox']))
    reference_area = float(reference_bbox[2]*reference_bbox[3])
    current_area = float(bbox[2]*bbox[3])
    ratio = reference_area/max(current_area, 1.)
    return dict(method='current_raw_body_supplements_missing_toy_roi',
        initial_compact_pale_car=initial_car, initial_reference_bbox=reference_bbox,
        selected_head_bbox=(x, y, width, height), selected_head_center=selected_pixel,
        current_complete_bbox=bbox, source_anchor_distance_m=distance,
        reference_area_pixel2=reference_area, current_area_pixel2=current_area,
        reference_to_current_area_ratio=ratio, fragment_area_ratio=FRAGMENT_AREA_RATIO,
        area_semantics='same_fixed_head_camera_bbox_area',
        initial_fragment=bool(ratio < FRAGMENT_AREA_RATIO),
        apply=bool(initial_car and ratio >= FRAGMENT_AREA_RATIO),
        current_raw_compact_pale_car=bool(compact_pale_car(current)),
        pinch_height_m=.021, requires_eye_pattern=False,
        source='current_normal_raw_head_selected_contour',
        original_color_shape_thresholds_unchanged=True)


def prepare_core(replay):
    """Insert one supplement after approach before the retained parent transforms."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(replay)))
    extension = ast.parse('''
current_car_body_evidence = current_car_body_fallback(ctx, obj)
if current_car_body_evidence is not None:
    memory.source_updates[-1]['current_car_body_fallback'] = current_car_body_evidence
    if current_car_body_evidence['apply']:
        solid_toy = False
        position[2] = ctx.table_z + .021
        approach = position + [0., 0., lift_height]
        memory.pickups[-1].update(pinch=position.copy(), current_car_body_fallback=True)
''').body
    inserted = 0

    class Insert(ast.NodeTransformer):
        def visit_For(self, node):
            nonlocal inserted
            self.generic_visit(node)
            output = []
            for statement in node.body:
                output.append(statement)
                if (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.YieldFrom)
                        and isinstance(statement.value.value, ast.Call)
                        and isinstance(statement.value.value.func, ast.Name)
                        and statement.value.value.func.id == 'move'
                        and any(keyword.arg == 'empty_approach' and isinstance(keyword.value, ast.Constant)
                                and keyword.value.value is True for keyword in statement.value.value.keywords)
                        and any(isinstance(arg, ast.Name) and arg.id == 'approach'
                                for arg in statement.value.value.args)):
                    output.extend(copy.deepcopy(extension))
                    inserted += 1
            node.body = output
            return node

    Insert().visit(tree)
    if inserted != 1:
        raise ValueError('Expected exactly one retained source empty approach')
    rendered = ast.unparse(ast.fix_missing_locations(tree))+'\n'
    digest = hashlib.sha256(rendered.encode()).hexdigest()[:16]
    filename = str(Path(__file__).with_name('_sorting_car_body_fallback_'+digest+'.py'))
    linecache.cache[filename] = (len(rendered), None, rendered.splitlines(True), filename)
    namespace = dict(replay.__globals__, current_car_body_fallback=current_car_body_fallback)
    exec(compile(rendered, filename, 'exec'), namespace)
    result = namespace[replay.__name__]
    result.__defaults__ = replay.__defaults__
    result.__kwdefaults__ = replay.__kwdefaults__
    return result


def finish_core(ctx, core):
    """Retain all selected parent bindings and expose the independent supplement."""
    namespace = dict(core.__globals__, current_car_body_fallback=current_car_body_fallback)
    result = FunctionType(core.__code__, namespace, core.__name__, core.__defaults__, core.__closure__)
    result.__kwdefaults__ = core.__kwdefaults__
    return result
