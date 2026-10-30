"""Reuse retained heavy-object timing for a fixed paired car identity."""
import ast
import copy
import hashlib
import inspect
import linecache
from pathlib import Path
import textwrap


def prepare_core(replay):
    """Insert private timing overrides without changing geometry or parent AST targets."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(replay)))
    source_extension = ast.parse('''
if pairs[demo_id].get('sorting_paired_type') == 'car':
    ctx.sorting_heavy_grasp = True
    contact_duration = 12
    pickup_lift_duration = 18
    close_duration = 10
    close_settle_duration = 6
    memory.source_updates[-1]['paired_car_loaded_timing'] = dict(
        source='retained_heavy_vehicle_timing_for_fixed_paired_car',
        contact_frames=12, lift_frames=18, close_frames=10, settle_frames=6,
        right_carry_frames=28, delivery_frames=24, loaded_joint_velocity_rad_s=1.6,
        geometry_changed=False, holding_asserted=False)
''').body
    carry_extension = ast.parse('''
if pairs[demo_id].get('sorting_paired_type') == 'car':
    carry_duration = 28
''').body
    source_count = 0
    carry_count = 0

    class Insert(ast.NodeTransformer):
        def visit_For(self, node):
            nonlocal source_count
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
                    output.extend(copy.deepcopy(source_extension))
                    source_count += 1
            node.body = output
            return node

        def visit_Assign(self, node):
            nonlocal carry_count
            self.generic_visit(node)
            if (any(isinstance(target, ast.Name) and target.id == 'carry_duration'
                    for target in node.targets) and isinstance(node.value, ast.IfExp)
                    and isinstance(node.value.test, ast.Call)
                    and isinstance(node.value.test.func, ast.Name)
                    and node.value.test.func.id == 'compact_gold_vehicle'
                    and isinstance(node.value.body, ast.Constant) and node.value.body.value == 28
                    and isinstance(node.value.orelse, ast.Constant) and node.value.orelse.value == 16):
                carry_count += 1
                return [node, *copy.deepcopy(carry_extension)]
            return node

    Insert().visit(tree)
    if source_count != 1 or carry_count != 1:
        raise ValueError('Expected one retained source approach and right carry assignment')
    rendered = ast.unparse(ast.fix_missing_locations(tree))+'\n'
    digest = hashlib.sha256(rendered.encode()).hexdigest()[:16]
    filename = str(Path(__file__).with_name('_sorting_car_loaded_'+digest+'.py'))
    linecache.cache[filename] = (len(rendered), None, rendered.splitlines(True), filename)
    namespace = dict(replay.__globals__)
    exec(compile(rendered, filename, 'exec'), namespace)
    result = namespace[replay.__name__]
    result.__defaults__ = replay.__defaults__
    result.__kwdefaults__ = replay.__kwdefaults__
    return result
