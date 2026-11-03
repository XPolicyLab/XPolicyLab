"""Thread one observed support plane through kong perception and motion."""
import ast
import copy


def expr(text):
    return ast.parse(text, mode='eval').body


def has_table(node):
    return any(isinstance(n, ast.Name) and n.id == 'TABLE' for n in ast.walk(node))


def replace_table(node, value):
    class Replace(ast.NodeTransformer):
        def visit_Name(self, item):
            return copy.deepcopy(value) if item.id == 'TABLE' else item
    return Replace().visit(node)


def add_plane_argument(node):
    node.args.kwonlyargs.append(ast.arg(arg='support_plane'))
    node.args.kw_defaults.append(None)


def strip_table(tree):
    tree.body = [node for node in tree.body if not (isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'TABLE' for t in node.targets))]
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == 'vision':
            node.names = [name for name in node.names if name.name != 'TABLE']
    tree.body.insert(1, ast.parse('from physicalrsi_observed_geometry.support_projection import support_point, support_height').body[0])


def finish(tree):
    if has_table(tree):
        raise ValueError('Unconverted kong table height reference')
    return ast.unparse(ast.fix_missing_locations(tree)) + '\n'


def edit_kong_table_vision(text):
    tree = ast.parse(text)
    strip_table(tree)
    names = {'row_instances', 'flat_faces', 'declaration_stock'}
    class Bind(ast.NodeTransformer):
        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id == 'plane' and has_table(node.args[1]):
                node.func.id = 'support_point'
                node.args[1] = replace_table(node.args[1], ast.Constant(0.))
                node.keywords.append(ast.keyword(arg='support_plane', value=expr('support_plane')))
            return self.generic_visit(node)
        def visit_Assign(self, node):
            target = ast.unparse(node.targets[0])
            if target in {'bottom_a[2]', 'bottom_b[2]', 'center[2]'} and has_table(node.value):
                xy = target.split('[')[0] + '[:2]'
                node.value = replace_table(node.value, expr(f'support_height(support_plane, {xy})'))
            elif target == 'center_xyz' and has_table(node.value):
                node.value = replace_table(node.value, expr('support_height(support_plane, xy.mean(axis=0))'))
            return self.generic_visit(node)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            add_plane_argument(node)
            Bind().visit(node)
        if isinstance(node, ast.ClassDef) and node.name == 'Reader':
            for method in node.body:
                if not isinstance(method, ast.FunctionDef):
                    continue
                if method.name == '__init__':
                    method.body += ast.parse('self.support_provider = None').body
                if method.name == 'observe':
                    method.body[:0] = ast.parse('''
if self.support_provider is None:
    from physicalrsi_observed_geometry import GeometryUnavailable
    raise GeometryUnavailable('Observed kong support provider required')
support_plane = self.support_provider()
''').body
                    for call in ast.walk(method):
                        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id in names:
                            call.keywords.append(ast.keyword(arg='support_plane', value=expr('support_plane')))
    return finish(tree)


MOTION_FUNCTIONS = {'_front_push_plan', 'push_plan', 'handoff_plan', 'pickup_handoff_plan', 'upright_declaration_plan'}


def edit_kong_table_motion(text):
    tree = ast.parse(text)
    strip_table(tree)
    class Bind(ast.NodeTransformer):
        def visit_List(self, node):
            if len(node.elts) == 3 and has_table(node.elts[2]):
                xy = ast.List(elts=copy.deepcopy(node.elts[:2]), ctx=ast.Load())
                value = ast.Call(func=expr('support_height'), args=[expr('support_plane'), xy], keywords=[])
                node.elts[2] = replace_table(node.elts[2], value)
            return self.generic_visit(node)
        def visit_Assign(self, node):
            if ast.unparse(node.targets[0]) == 'zz' and has_table(node.value):
                node.value = replace_table(node.value, expr('support_height(support_plane, [x, yy])'))
            return self.generic_visit(node)
        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id in MOTION_FUNCTIONS:
                node.keywords.append(ast.keyword(arg='support_plane', value=expr('support_plane')))
            return self.generic_visit(node)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in MOTION_FUNCTIONS:
            add_plane_argument(node)
            Bind().visit(node)
            if node.name == 'upright_declaration_plan':
                for index, statement in enumerate(node.body):
                    if isinstance(statement, ast.AugAssign) and ast.unparse(statement.target) == 'start[0]':
                        node.body[index+1:index+1] = ast.parse('start[2] = support_height(support_plane, start[:2]) + THICK * .5 + .08').body
                        break
    return finish(tree)


def edit_kong_table_model(text):
    tree = ast.parse(text)
    strip_table(tree)
    class Bind(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id == 'TABLE':
                return expr('support_height(self.frames.get_support_plane(), xyz[:2])')
            return node
        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id in MOTION_FUNCTIONS | {'declaration_stock'}:
                node.keywords.append(ast.keyword(arg='support_plane', value=expr('self.frames.get_support_plane()')))
            return self.generic_visit(node)
    tree = Bind().visit(tree)
    episode = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'EpisodeModel')
    update = next(n for n in episode.body if isinstance(n, ast.FunctionDef) and n.name == 'update_obs')
    update.body += ast.parse('self.reader.support_provider = self.frames.get_support_plane').body
    motion = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MotionModel')
    action = next(n for n in motion.body if isinstance(n, ast.FunctionDef) and n.name == 'get_action')
    action.name = '_get_action_with_geometry'
    motion.body += ast.parse('''
def get_action(self):
    from physicalrsi_observed_geometry import GeometryUnavailable
    try:
        return self._get_action_with_geometry()
    except GeometryUnavailable as error:
        self.fail('observed_geometry_unavailable: ' + str(error))
        self.step += 1
        return [{key: value.copy() for key, value in self.hold.items()}]
''').body
    return finish(tree)
