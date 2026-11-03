"""Use the detected row edge width for declaration spacing.

The existing row projection still depends on separate height/support priors;
this edit removes only the independent stored tile-width value.
"""
import ast


def edit_kong_width_vision(text):
    tree = ast.parse(text)
    tree.body = [n for n in tree.body if not (isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == 'WIDTH' for t in n.targets))]
    row = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'row_instances')
    count = 0
    for node in ast.walk(row):
        if isinstance(node, ast.Dict) and any(isinstance(k, ast.Constant) and k.value == 'xyz' for k in node.keys):
            node.keys.extend([ast.Constant('width_m'), ast.Constant('width_source')])
            node.values.extend([ast.parse('float(np.linalg.norm(b - a))', mode='eval').body,
                                ast.Constant('current_RGB_row_seams_projected_to_front_top_plane')])
            count += 1
    if count != 1:
        raise ValueError('Expected one kong row instance record')
    return ast.unparse(ast.fix_missing_locations(tree)) + '\n'


def edit_kong_width_model(text):
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == 'vision':
            node.names = [name for name in node.names if name.name != 'WIDTH']
    tree.body = [node for node in tree.body if not (isinstance(node, ast.ImportFrom) and not node.names)]
    class Width(ast.NodeTransformer):
        count = 0
        def visit_Name(self, node):
            if node.id == 'WIDTH' and isinstance(node.ctx, ast.Load):
                self.count += 1
                return ast.parse("rightmost['width_m']", mode='eval').body
            return node
    change = Width()
    tree = change.visit(tree)
    if change.count != 1:
        raise ValueError('Expected one declaration width use')
    return ast.unparse(ast.fix_missing_locations(tree)) + '\n'
