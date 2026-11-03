"""Use the current measured support for language-pour finger checks."""
import ast


def bind_finger_clearance(text, owner, expected):
    tree = ast.parse(text)
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'finger_clearance':
            node.keywords.append(ast.keyword(arg='support_plane', value=ast.parse(owner + '.observed_support_plane', mode='eval').body))
            count += 1
    if count != expected:
        raise ValueError(f'Expected {expected} finger checks; found {count}')
    return ast.unparse(ast.fix_missing_locations(tree)) + '\n'


def edit_finger_geometry(text):
    tree = ast.parse(text)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'finger_clearance')
    replacement = ast.parse('''
def finger_clearance(ee, *, support_plane):
    from physicalrsi_observed_geometry import GeometryUnavailable
    plane = np.asarray(support_plane, dtype=float)
    transform = np.asarray(ee, dtype=float)
    if plane.shape != (3,) or not np.isfinite(plane).all() or transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise GeometryUnavailable('Measured support and finite finger pose required')
    vertices = finger_vertices() @ transform[:3, :3].T + transform[:3, 3]
    # Preserve the controller's vertical-gap metric, evaluated at each vertex.
    return float(np.min(vertices[:, 2] - vertices[:, :2] @ plane[:2] - plane[2]))
''').body[0]
    tree.body[tree.body.index(function)] = replacement
    return ast.unparse(ast.fix_missing_locations(tree)) + '\n'
