"""Rank existing direct-left headings by the first approach's existing IK fallback flag."""
import ast
import inspect
import textwrap
from types import FunctionType


def _clone(function, bindings):
    """Preserve every formal-parent binding except the explicitly private replacements."""
    namespace = dict(function.__globals__)
    namespace.update(bindings)
    result = FunctionType(function.__code__, namespace, function.__name__,
                          function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


def _first_approach_flag(candidate):
    """Read the unchanged planner report attached to the existing first approach segment."""
    segment = next((item for item in candidate['preflight_motion_segments']
                    if item['label'] == 'approach'), {})
    return bool(segment.get('adaptive_endpoint_fallback', True))


def _choose_existing_headings(accepted):
    """Keep all original candidates and use a lexicographic ordering without weights."""
    selected = min(accepted, key=lambda item: (_first_approach_flag(item),
                                              item['ideal_complete_route_frames']))
    selected['first_approach_endpoint_fallback'] = _first_approach_flag(selected)
    selected['heading_prefix_ranking'] = 'first_approach_fallback_then_original_total_frames'
    selected['heading_prefix_candidates'] = [dict(
        quaternion=item['quaternion'].tolist(),
        first_approach_endpoint_fallback=_first_approach_flag(item),
        ideal_complete_route_frames=int(item['ideal_complete_route_frames'])) for item in accepted]
    return selected


def _extend_direct(original):
    """Clone only the original segment recorder and direct heading's final selection."""
    pickup = original.__globals__['pickup']
    original_move = pickup.__globals__['move']

    def move(preview, segments, arm, point, quat, grip, duration, stage, label):
        original_move(preview, segments, arm, point, quat, grip, duration, stage, label)
        if label == 'approach':
            report = getattr(preview, 'motion_plan', {})
            segments[-1]['adaptive_endpoint_fallback'] = bool(
                report.get('adaptive_endpoint_fallback', True))
            segments[-1]['adaptive_endpoint_fallback_reported'] = 'adaptive_endpoint_fallback' in report

    private_pickup = _clone(pickup, {'move': move})
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))

    class SelectExistingHeadings(ast.NodeTransformer):
        """Replace the sole accepted-heading min expression, retaining its surrounding code."""
        def visit_Assign(self, node):
            node = self.generic_visit(node)
            if (any(isinstance(target, ast.Name) and target.id == 'selected' for target in node.targets)
                    and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == 'min' and node.value.args
                    and isinstance(node.value.args[0], ast.Name) and node.value.args[0].id == 'accepted'):
                node.value = ast.copy_location(ast.Call(func=ast.Name(id='_choose_existing_headings', ctx=ast.Load()),
                    args=[ast.Name(id='accepted', ctx=ast.Load())], keywords=[]), node.value)
            return node

    tree = ast.fix_missing_locations(SelectExistingHeadings().visit(tree))
    namespace = dict(original.__globals__)
    namespace.update(pickup=private_pickup, _choose_existing_headings=_choose_existing_headings)
    exec(compile(tree, __file__, 'exec'), namespace)
    return namespace[original.__name__]


def build_core(ctx, already_formal_transformed_replay):
    """Extend source and receiver direct rankings without applying any parent transform."""
    replay = already_formal_transformed_replay
    namespace = dict(replay.__globals__)
    source_select = namespace['select_route']
    source_direct = source_select.__globals__['direct_left_preflight']
    source_private_direct = _extend_direct(source_direct)
    cleared = source_select.__globals__['cleared_direct_left_preflight']
    private_cleared = _clone(cleared, {'direct_left_preflight': source_private_direct})
    private_select = _clone(source_select, dict(direct_left_preflight=source_private_direct,
        cleared_direct_left_preflight=private_cleared))
    receiver_direct = namespace['direct_left_preflight']
    receiver_private_direct = (source_private_direct if receiver_direct is source_direct
                               else _extend_direct(receiver_direct))
    return _clone(replay, dict(select_route=private_select,
                               direct_left_preflight=receiver_private_direct))
