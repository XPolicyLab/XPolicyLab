"""Use acquired bowl geometry and observed robot bases in language pouring."""
import ast


def expr(text): return ast.parse(text, mode='eval').body
def finish(tree): return ast.unparse(ast.fix_missing_locations(tree))+'\n'
def keyword(fn, name):
    fn.args.kwonlyargs.append(ast.arg(arg=name)); fn.args.kw_defaults.append(None)


def strip_table(tree):
    tree.body = [n for n in tree.body if not (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'TABLE' for t in n.targets))]


class Bases(ast.NodeTransformer):
    def __init__(self, owner): self.owner = owner
    def visit_Call(self, node):
        self.generic_visit(node)
        if ast.unparse(node.func) != 'np.array' or len(node.args) != 1:
            return node
        value = node.args[0]
        if not isinstance(value, ast.List) or len(value.elts) != 2 or not isinstance(value.elts[0], ast.IfExp):
            return node
        choice = value.elts[0]
        try:
            matches = ast.literal_eval(choice.body) == -.3 and ast.literal_eval(choice.orelse) == .3 and ast.literal_eval(value.elts[1]) == -.45
        except (ValueError, TypeError):
            return node
        if matches:
            if self.owner == 'motion': return expr('np.asarray(robot_base)[:2]')
            return expr(self.owner+'.robot_base('+ast.unparse(choice.test.left)+')[:2]')
        return node


def edit_bowl_model(text):
    tree = ast.parse(text); strip_table(tree)
    tree.body[1:1] = ast.parse('from bowl_acquisition import BowlAcquisition').body
    ground = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'ground')
    keyword(ground, 'bowl_observations')
    for index, n in enumerate(ground.body):
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'bowl_xy':
            ground.body[index:index+1] = ast.parse("bowl_geometry = bowl_observations[cc]\nbowl_xy = np.asarray(bowl_geometry['rim_center'][:2])").body
            break
    debug = next(n for n in ground.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'debug')
    debug.value.keys.append(ast.Constant('bowl_geometry')); debug.value.values.append(expr('bowl_geometry'))
    model = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Model')
    reset = next(n for n in model.body if isinstance(n, ast.FunctionDef) and n.name == 'reset')
    reset.body[:0] = ast.parse('self.bowl_acquisition = None\nself.bowl_observations = None\nself.bowl_geometry = None').body
    build = next(n for n in model.body if isinstance(n, ast.FunctionDef) and n.name == 'build_pair')
    for index, n in enumerate(build.body):
        if isinstance(n, ast.Try) and any(isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == 'observe_support_geometry' for call in ast.walk(n)):
            build.body[index] = ast.If(test=expr('self.observed_support is None'), body=[n], orelse=[])
            break
    index = next(i for i, n in enumerate(build.body) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'self.bottle_geometry')
    build.body[index:index] = ast.parse("self.bowl_geometry = debug['bowl_geometry']").body
    for n in ast.walk(build):
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'bowl_radius':
            n.value = expr("self.bowl_geometry['radius']")
    class Bind(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Name):
                if node.func.id == 'ground':
                    node.keywords.append(ast.keyword(arg='bowl_observations', value=expr('self.bowl_observations')))
                if node.func.id == 'program':
                    node.keywords += [ast.keyword(arg='bowl_geometry', value=expr('self.bowl_geometry')),
                                      ast.keyword(arg='robot_base', value=expr('self.robot_base(arm)'))]
            return node
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef) or cls.name not in {'Model', 'StructuredModel'}: continue
        Bind().visit(cls); Bases('self').visit(cls)
        for method in cls.body:
            if isinstance(method, ast.FunctionDef) and method.name == 'get_action':
                method.body[:0] = ast.parse('bowl_action = self.acquire_bowls()\nif bowl_action is not None:\n    return bowl_action').body
    model.body += ast.parse('''
def robot_base(self, arm):
    from rdj_insert_policy.ik import observed_base
    return np.asarray(observed_base(get_ik(self).solver.chain, arm, self.obs['state'])[:3])

def acquire_bowls(self):
    if self.bowl_observations is not None:
        return None
    if self.bowl_acquisition is None:
        self.bowl_acquisition = BowlAcquisition()
    return self.bowl_acquisition.step(self)
''').body
    if any(isinstance(n, ast.Name) and n.id == 'TABLE' for n in ast.walk(tree)):
        raise ValueError('Unconverted language-pour table reference')
    return finish(tree)


def edit_bowl_motion(text):
    tree = ast.parse(text); strip_table(tree)
    program = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'program')
    keyword(program, 'bowl_geometry'); keyword(program, 'robot_base')
    Bases('motion').visit(program)
    class Rim(ast.NodeTransformer):
        def visit_BinOp(self, node):
            if isinstance(node.op, ast.Add) and isinstance(node.left, ast.Name) and node.left.id == 'TABLE' and isinstance(node.right, ast.Constant) and node.right.value == .04:
                return expr("bowl_geometry['rim_center'][2]")
            return self.generic_visit(node)
    Rim().visit(program)
    if any(isinstance(n, ast.Name) and n.id == 'TABLE' for n in ast.walk(tree)):
        raise ValueError('Unconverted language-pour motion support')
    return finish(tree)


def edit_bowl_observer(text):
    tree = ast.parse(text)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BowlObserver')
    Bases('model').visit(cls)
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    init.body[:0] = ast.parse("self.bowl_rotation = np.asarray(debug['bowl_geometry']['rotation'])").body
    for n in ast.walk(init):
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'self.center':
            n.value = expr("np.asarray(debug['bowl_geometry']['rim_center'])")
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'self.radius':
            n.value = expr("float(debug['bowl_geometry']['radius'])")
    observe = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'observe')
    for n in ast.walk(observe):
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'circle':
            n.value = expr('self.center + np.c_[np.cos(angles)*self.radius, np.sin(angles)*self.radius, np.zeros(40)] @ self.bowl_rotation.T')
    return finish(tree)
