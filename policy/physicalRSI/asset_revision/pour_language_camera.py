"""Bind head projection and callbacks to the current episode camera."""
import ast


def edit_language_head_camera(text):
    tree = ast.parse(text)
    tree.body = [n for n in tree.body if not (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'F' for t in n.targets))]
    tree.body[1:1] = ast.parse('from physicalrsi_observed_geometry import Camera, GeometryUnavailable').body
    replacements = ast.parse('''
def ray(u, v, z, *, camera):
    u, v, z = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float), np.asarray(z, float))
    pixels = np.stack([u, v, np.ones_like(u)], axis=-1)
    directions = (pixels @ np.linalg.inv(camera.k).T * [1., -1., -1.]) @ camera.transform[:3, :3].T
    origin = camera.transform[:3, 3]
    if not np.isfinite(pixels).all() or not np.isfinite(z).all() or np.any(np.abs(directions[..., 2]) < 1e-9):
        raise GeometryUnavailable('Finite ray and intersecting plane required')
    distance = (z - origin[2]) / directions[..., 2]
    if np.any(distance <= 0):
        raise GeometryUnavailable('Requested plane is behind the current camera')
    return (origin + directions * distance[..., None])[..., :2]

def project(xyz, *, camera):
    point = np.asarray(xyz, float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise GeometryUnavailable('Finite world point required')
    return camera.project(point)[0][0]
''').body
    for replacement in replacements:
        old = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == replacement.name)
        tree.body[tree.body.index(old)] = replacement
    ground = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'ground')
    ground.args.kwonlyargs.append(ast.arg(arg='camera')); ground.args.kw_defaults.append(None)
    for node in ast.walk(ground):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {'ray', 'project'}:
            node.keywords.append(ast.keyword(arg='camera', value=ast.Name(id='camera', ctx=ast.Load())))
    model = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Model')
    class Bind(ast.NodeTransformer):
        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Load) and node.id in {'ray', 'project'}:
                return ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr=node.id, ctx=ast.Load())
            return node
        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id == 'ground':
                node.keywords.append(ast.keyword(arg='camera', value=ast.parse('self.head_camera', mode='eval').body))
            return node
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef) and cls.name in {'Model', 'StructuredModel'}:
            Bind().visit(cls)
    reset = next(n for n in model.body if isinstance(n, ast.FunctionDef) and n.name == 'reset')
    reset.body[:0] = ast.parse('self.head_camera = None').body
    update = next(n for n in model.body if isinstance(n, ast.FunctionDef) and n.name == 'update_obs')
    update.body += ast.parse('self.head_camera = Camera.from_observation(obs)').body
    model.body += ast.parse('''
def ray(self, u, v, z):
    if self.head_camera is None:
        raise GeometryUnavailable('Bind current head camera before projection')
    return ray(u, v, z, camera=self.head_camera)

def project(self, xyz):
    if self.head_camera is None:
        raise GeometryUnavailable('Bind current head camera before projection')
    return project(xyz, camera=self.head_camera)
''').body
    return ast.unparse(ast.fix_missing_locations(tree)) + '\n'


def bind_observer_head_camera(text):
    tree = ast.parse(text)
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == 'model':
            node.names = [n for n in node.names if n.name != 'ray']
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'ray':
            node.func = ast.parse('model.ray', mode='eval').body
            count += 1
    if count != 2:
        raise ValueError(f'Expected two bowl-observer head rays, found {count}')
    return ast.unparse(ast.fix_missing_locations(tree)) + '\n'


def edit_observer_wrist_camera(text):
    tree = ast.parse(text)
    tree.body = [n for n in tree.body if not (
        isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {'FOCAL', 'WRIST_LOCAL'} for t in n.targets)
        or isinstance(n, ast.FunctionDef) and n.name == 'transform')]
    tree.body[1:1] = ast.parse('from physicalrsi_observed_geometry import Camera\nfrom physicalrsi_observed_geometry.wrist_camera import ee_relative_camera').body
    pixels = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'pixels')
    pixels.args.kwonlyargs.append(ast.arg(arg='intrinsic')); pixels.args.kw_defaults.append(None)
    pixels.body = ast.parse('''
q = (np.asarray(points) - camera[:3, 3]) @ camera[:3, :3]
depth = -q[:, 2]
if np.any(depth <= .005):
    return None
homogeneous = (q * [1., -1., -1.]) @ intrinsic.T
return homogeneous[:, :2] / depth[:, None]
''').body
    observer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BowlObserver')
    class Bind(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id == 'WRIST_LOCAL':
                return ast.Name(id='wrist_relative', ctx=ast.Load())
            if node.id == 'FOCAL':
                return ast.parse('K[1, 1]', mode='eval').body
            return node
        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id == 'pixels':
                node.keywords.append(ast.keyword(arg='intrinsic', value=ast.Name(id='K', ctx=ast.Load())))
            return node
    for method in observer.body:
        if not isinstance(method, ast.FunctionDef) or method.name not in {'__init__', 'observe'}:
            continue
        Bind().visit(method)
        calibration = ast.parse("wrist_relative, K = ee_relative_camera(model.obs, model.obs['state'][self.arm + '_ee_pose'])").body
        if method.name == '__init__':
            index = next(i for i, n in enumerate(method.body) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'self.arm')
            method.body[index+1:index+1] = calibration + ast.parse("wrist_camera = Camera.from_observation(model.obs, 'cam_' + self.arm + '_wrist')").body
            class Height(ast.NodeTransformer):
                def visit_Constant(self, node):
                    return ast.parse('wrist_camera.height / 2', mode='eval').body if node.value == 240 else node
            Height().visit(method)
        else:
            method.body[:0] = calibration
    result = ast.unparse(ast.fix_missing_locations(tree)) + '\n'
    assert 'WRIST_LOCAL' not in result and 'FOCAL' not in result
    return result
