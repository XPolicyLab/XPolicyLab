"""Bind kong projection to each episode's current calibrated RGB camera."""
import ast


def edit_kong_vision(text):
    tree = ast.parse(text)
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {'F', 'CAM', 'R'} for t in node.targets))]
    tree.body[0] = ast.Expr(value=ast.Constant('RGB seam instances and ink matching using current camera calibration. Object dimension priors remain pending replacement.'))
    replacements = ast.parse('''
def project(p, *, camera):
    import numpy as np
    if camera is None:
        from physicalrsi_observed_geometry import GeometryUnavailable
        raise GeometryUnavailable('Current kong camera required')
    points = np.asarray(p)
    return camera.project(points)[0].reshape(points.shape[:-1] + (2,))

def plane(uv, z, *, camera):
    from physicalrsi_observed_geometry import GeometryUnavailable
    if camera is None:
        raise GeometryUnavailable('Current kong camera required')
    pixel = np.asarray(uv, float)
    if pixel.shape != (2,) or not np.isfinite(pixel).all() or not np.isfinite(z):
        raise GeometryUnavailable('Finite pixel and plane height required')
    ray = np.linalg.solve(camera.k, np.r_[pixel, 1.])
    ray = camera.transform[:3, :3] @ (ray * [1, -1, -1])
    origin = camera.transform[:3, 3]
    if abs(ray[2]) < 1e-9:
        raise GeometryUnavailable('Camera ray parallel to plane')
    distance = (z - origin[2]) / ray[2]
    if distance <= 0:
        raise GeometryUnavailable('Plane intersection behind camera')
    return origin + distance * ray
''').body
    replacements = {node.name: node for node in replacements}
    functions = {'row_instances', 'flat_faces', 'declaration_stock'}
    class Bind(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            if node.name in replacements:
                return replacements[node.name]
            if node.name in functions:
                node.args.kwonlyargs.append(ast.arg(arg='camera'))
                node.args.kw_defaults.append(None)
                class Calls(ast.NodeTransformer):
                    def visit_Call(self, call):
                        if isinstance(call.func, ast.Name) and call.func.id in {'plane', 'project'}:
                            call.keywords.append(ast.keyword(arg='camera', value=ast.Name(id='camera', ctx=ast.Load())))
                        return self.generic_visit(call)
                return Calls().visit(node)
            return self.generic_visit(node)
        def visit_ClassDef(self, node):
            if node.name == 'Reader':
                for method in node.body:
                    if isinstance(method, ast.FunctionDef) and method.name == '__init__':
                        method.body += ast.parse('self.camera = None').body
                    if isinstance(method, ast.FunctionDef) and method.name == 'observe':
                        for call in ast.walk(method):
                            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id in functions:
                                call.keywords.append(ast.keyword(arg='camera', value=ast.parse('self.camera', mode='eval').body))
            return self.generic_visit(node)
    return ast.unparse(ast.fix_missing_locations(Bind().visit(tree))) + '\n'


def edit_kong_camera_model(text):
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {'project', 'plane', 'declaration_stock'}:
            node.keywords.append(ast.keyword(arg='camera', value=ast.parse('self.reader.camera', mode='eval').body))
    episode = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'EpisodeModel')
    update = next(n for n in episode.body if isinstance(n, ast.FunctionDef) and n.name == 'update_obs')
    update.body += ast.parse('from physicalrsi_observed_geometry import Camera\nself.reader.camera = Camera.from_observation(obs)').body
    return ast.unparse(ast.fix_missing_locations(tree)) + '\n'
