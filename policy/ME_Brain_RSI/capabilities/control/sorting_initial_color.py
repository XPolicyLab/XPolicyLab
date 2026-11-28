"""Separate comparable red and dark head bodies without splitting other families."""
import ast
import hashlib
import inspect
import linecache
from pathlib import Path
import textwrap
from types import FunctionType

import numpy as np

from coap.tasks.memory_initial import initial_objects as original_initial_objects


def _compatible_color(first, second):
    """Keep existing aliases except a comparable red-car and dark-body conflict."""
    families = {first.get('appearance_family'), second.get('appearance_family')}
    if families not in ({'toy_car', 'watch'}, {'toy_car', 'action_camera'}):
        return True
    first_area = float(first['bbox'][2]*first['bbox'][3])
    second_area = float(second['bbox'][2]*second['bbox'][3])
    comparable = min(first_area,second_area)/max(first_area,second_area,1.) >= .45
    distance = float(np.linalg.norm(np.asarray(first['mean_rgb'])-np.asarray(second['mean_rgb'])))
    return not (comparable and distance >= 65)


def _initializer():
    """Clone only the original head-parts merge eligibility expression."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(original_initial_objects)))
    changed = 0
    class ColorAssociation(ast.NodeTransformer):
        def visit_If(self, node):
            nonlocal changed
            self.generic_visit(node)
            if (isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Name)
                    and node.test.left.id == 'gap' and len(node.test.ops) == 1
                    and isinstance(node.test.ops[0], ast.LtE)
                    and isinstance(node.test.comparators[0], ast.Name)
                    and node.test.comparators[0].id == 'maximum_gap'):
                node.test = ast.BoolOp(op=ast.And(), values=[node.test,
                    ast.Call(func=ast.Name(id='_compatible_color', ctx=ast.Load()),
                        args=[ast.Name(id='first',ctx=ast.Load()),ast.Name(id='second',ctx=ast.Load())],
                        keywords=[])])
                changed += 1
            return node
    ColorAssociation().visit(tree)
    if changed != 1:
        raise ValueError('Expected one retained head fragment association expression')
    namespace = dict(original_initial_objects.__globals__, _compatible_color=_compatible_color)
    exec(compile(ast.fix_missing_locations(tree), __file__, 'exec'), namespace)
    return namespace[original_initial_objects.__name__]


_COLOR_INITIALIZER = _initializer()


def repair_initial(ctx, objects, refine_demonstrations=False):
    """Reprocess the same normal observation only after an original head-part merge."""
    observations = getattr(ctx, 'sorting_initial_observations', [])
    previous = observations[-1] if observations else {}
    if not previous.get('head_part_merges'):
        return objects
    result = _COLOR_INITIALIZER(ctx)
    observations[-1]['color_consistent_head_merge'] = dict(
        source='same_normal_observation_comparable_red_and_dark_body_conflict',
        original_head_part_merges=previous['head_part_merges'],
        original_retained_count=len(objects), revised_retained_count=len(result),
        repeated_detection_same_observation=True, additional_actions=0,
        rejected_family_pairs=[['toy_car','watch'],['toy_car','action_camera']],
        incompatible_rgb_distance_min=65, comparable_bbox_area_ratio_min=.45,
        execution_exit_added=False)
    if refine_demonstrations:
        from coap.tasks.sorting_demo_fragments import initial_objects as demo_initializer
        namespace = dict(demo_initializer.__globals__, original_initial_objects=lambda robot: result)
        private = FunctionType(demo_initializer.__code__, namespace, demo_initializer.__name__,
                               demo_initializer.__defaults__, demo_initializer.__closure__)
        private.__kwdefaults__ = demo_initializer.__kwdefaults__
        result = private(ctx)
    return result


def prepare_core(replay, refine_demonstrations=False):
    """Append local visual association refinement without replacing parent imports."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(replay)))
    count = 0
    class RefineInitialization(ast.NodeTransformer):
        def visit_Assign(self, node):
            nonlocal count
            self.generic_visit(node)
            if (any(isinstance(target, ast.Name) and target.id == 'objects' for target in node.targets)
                    and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == 'initial_objects'):
                addition = ast.Assign(targets=[ast.Name(id='objects',ctx=ast.Store())],
                    value=ast.Call(func=ast.Name(id='sorting_color_initial',ctx=ast.Load()),
                        args=[ast.Name(id='ctx',ctx=ast.Load()),ast.Name(id='objects',ctx=ast.Load())],
                        keywords=[ast.keyword(arg='refine_demonstrations',
                                              value=ast.Constant(value=bool(refine_demonstrations)))]))
                count += 1
                return [node,addition]
            return node
    RefineInitialization().visit(tree)
    if count != 2:
        raise ValueError('Expected the two retained initial and scout measurements')
    rendered = ast.unparse(ast.fix_missing_locations(tree))+'\n'
    digest = hashlib.sha256(rendered.encode()).hexdigest()[:16]
    filename = str(Path(__file__).with_name('_sorting_initial_color_'+digest+'.py'))
    linecache.cache[filename] = (len(rendered),None,rendered.splitlines(True),filename)
    namespace = dict(replay.__globals__,sorting_color_initial=repair_initial)
    exec(compile(rendered,filename,'exec'),namespace)
    result = namespace[replay.__name__]
    result.__defaults__ = replay.__defaults__
    result.__kwdefaults__ = replay.__kwdefaults__
    return result
