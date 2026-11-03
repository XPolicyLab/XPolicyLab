"""Thread an episode-observed radial profile through bottle registration."""
import ast


def expr(code):
    return ast.parse(code, mode='eval').body


def keyword(function, name):
    function.args.kwonlyargs.append(ast.arg(arg=name)); function.args.kw_defaults.append(None)


def finish(tree):
    return ast.unparse(ast.fix_missing_locations(tree)) + '\n'


def edit_body_registration(text):
    tree = ast.parse(text)
    tree.body = [n for n in tree.body if not (
        isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {'F', 'PROFILE', 'POINTS'} for t in n.targets)
        or isinstance(n, ast.FunctionDef) and n.name in {'project', 'surface', 'render'})]
    tree.body[0] = ast.Expr(value=ast.Constant('RGB registration against the current episode observed radial approximation.'))
    tree.body[1:1] = ast.parse('''
from physicalrsi_observed_geometry.radial_profile import render_profile, validate_profile
from physicalrsi_observed_geometry import GeometryUnavailable
def render(rotation, translation, shape=(120, 160), *, profile, camera):
    validate_profile(profile)
    try:
        return render_profile(profile, rotation, translation, camera, shape)
    except GeometryUnavailable:
        # An optimizer proposal behind the current camera is inadmissible.
        # Empty projection has zero overlap and cannot qualify an attachment.
        return np.zeros(shape, np.uint8)
''').body
    for fn in tree.body:
        if not isinstance(fn, ast.FunctionDef) or fn.name not in {'fit', 'qualified_fit'}:
            continue
        keyword(fn, 'profile'); keyword(fn, 'camera')
        for node in ast.walk(fn):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in {'fit', 'render'}:
                    node.keywords += [ast.keyword(arg=k, value=expr(k)) for k in ['profile', 'camera']]
                if node.func.id == 'project':
                    node.func = expr('camera.project')
        class Projection(ast.NodeTransformer):
            def visit_Assign(self, node):
                self.generic_visit(node)
                if ast.unparse(node.targets[0]) == 'endpoints':
                    node.value = ast.Subscript(value=node.value, slice=ast.Constant(0), ctx=ast.Load())
                return node
            def visit_Constant(self, node):
                return expr('np.asarray(profile)[-1, 0]') if node.value == .2181 else node
        Projection().visit(fn)
    return finish(tree)


def edit_axis_registration(text):
    tree = ast.parse(text)
    for fn in tree.body:
        if not isinstance(fn, ast.FunctionDef) or fn.name not in {'estimate', 'qualified_axis_fit'}:
            continue
        keyword(fn, 'profile'); keyword(fn, 'camera')
        for node in ast.walk(fn):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {'estimate', 'render'}:
                node.keywords += [ast.keyword(arg=k, value=expr(k)) for k in ['profile', 'camera']]
    return finish(tree)


def edit_profile_model(text):
    tree = ast.parse(text)
    tree.body[1:1] = ast.parse('from physicalrsi_observed_geometry.radial_profile import observe_profile, observed_neck_height').body
    ground = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'ground')
    keyword(ground, 'support_plane')
    # The unused fixed-height workspace raster was never consulted by detection.
    ground.body = [n for n in ground.body if not (isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) in {'(h, s, v)', '(yy, xx)', 'xy', 'roi'})]
    for i, node in enumerate(ground.body):
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == 'bottle_xy':
            ground.body[i:i+1] = ast.parse("geometry = observe_profile(camera, masks[chosen['index']], support_plane)\ngeometry['grasp_height'] = observed_neck_height(geometry['profile'])\nbottle_xy = np.asarray(geometry['base'][:2])").body
            break
    debug = next(n for n in ground.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'debug')
    debug.value.keys.append(ast.Constant('bottle_geometry')); debug.value.values.append(expr('geometry'))
    model = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Model')
    reset = next(n for n in model.body if isinstance(n, ast.FunctionDef) and n.name == 'reset')
    reset.body[:0] = ast.parse('self.bottle_geometry = None').body
    build = next(n for n in model.body if isinstance(n, ast.FunctionDef) and n.name == 'build_pair')
    # The original grounding exception branch remains; GeometryUnavailable is a ValueError.
    index = next(i for i, n in enumerate(build.body) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'self.bxy')
    build.body[index:index] = ast.parse("self.bottle_geometry = debug['bottle_geometry']\nself.initial_bottle_height = self.bottle_geometry['height']").body
    class Bind(ast.NodeTransformer):
        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Load) and node.id == 'TABLE':
                return expr("self.bottle_geometry['base'][2]")
            return node
        def visit_Constant(self, node):
            return expr("self.bottle_geometry['height'] / 2") if isinstance(node.value, float) and node.value in {.10905, .11} else node
        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Name):
                if node.func.id == 'ground':
                    node.keywords.append(ast.keyword(arg='support_plane', value=expr('self.observed_support_plane')))
                if node.func.id == 'qualified_axis_fit':
                    node.keywords += [ast.keyword(arg='profile', value=expr("self.bottle_geometry['profile']")), ast.keyword(arg='camera', value=expr('self.head_camera'))]
                if node.func.id == 'program':
                    node.keywords.append(ast.keyword(arg='bottle_geometry', value=expr('self.bottle_geometry')))
                if node.func.id == 'upstream_spout':
                    node.keywords.append(ast.keyword(arg='spout_radius', value=expr("self.bottle_geometry['profile'][-1][1]")))
                if node.func.id in {'assess', 'observe'}:
                    node.keywords.append(ast.keyword(arg='bottle_height', value=expr("self.bottle_geometry['height']")))
            return node
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef) and cls.name in {'Model', 'StructuredModel'}:
            Bind().visit(cls)
    return finish(tree)


def edit_profile_motion(text):
    tree = ast.parse(text)
    tree.body[1:1] = ast.parse('from physicalrsi_observed_geometry.radial_profile import profile_surface, observed_neck_height').body
    program = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'program')
    keyword(program, 'bottle_geometry')
    replacements = {'height': "bottle_geometry['height']", 'grasp_height': "observed_neck_height(bottle_geometry['profile'])",
                    'verts': "profile_surface(bottle_geometry['profile']).reshape(-1, 3)"}
    program.body = [n for n in program.body if not (isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'neck_interval')]
    class Bind(ast.NodeTransformer):
        def visit_Assign(self, node):
            target = ast.unparse(node.targets[0])
            if target in replacements:
                node.value = expr(replacements[target]); return node
            if target in {'grasp', 'bottle[:3, 3]'}:
                class Base(ast.NodeTransformer):
                    def visit_Name(self, n):
                        return expr("bottle_geometry['base'][2]") if n.id == 'TABLE' else n
                Base().visit(node)
            return self.generic_visit(node)
        def visit_Constant(self, node):
            return expr("float(np.max(np.asarray(bottle_geometry['profile'])[:, 1]))") if node.value == .0412 else node
        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id == 'upstream_spout':
                node.keywords.append(ast.keyword(arg='spout_radius', value=expr("bottle_geometry['profile'][-1][1]")))
            return node
    Bind().visit(program)
    return finish(tree)


def edit_height_consumer(text, entry):
    tree = ast.parse(text)
    tree.body = [n for n in tree.body if not (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'HEIGHT' for t in n.targets))]
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == entry)
    keyword(fn, 'bottle_height')
    class Height(ast.NodeTransformer):
        def visit_Name(self, node):
            return expr('bottle_height') if node.id == 'HEIGHT' else node
        def visit_Constant(self, node):
            if node.value == .2181: return expr('bottle_height')
            if node.value in {.10905, .11}: return expr('bottle_height / 2')
            return node
    Height().visit(fn)
    return finish(tree)


def edit_spout(text):
    tree = ast.parse(text)
    tree.body = [n for n in tree.body if not (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SPOUT_RADIUS' for t in n.targets))]
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'upstream_spout')
    keyword(fn, 'spout_radius')
    class Radius(ast.NodeTransformer):
        def visit_Name(self, node):
            return expr('spout_radius') if node.id == 'SPOUT_RADIUS' else node
    Radius().visit(fn)
    return finish(tree).replace('spout_radius_public_m=', 'spout_radius_observed_m=')
