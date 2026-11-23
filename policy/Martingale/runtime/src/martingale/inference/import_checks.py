# Copyright © 2026 Martingale. All Rights Reserved.
import ast
import importlib.util
from pathlib import Path


def check_internal_imports(package_root=None):
    root = Path(package_root) if package_root else Path(__file__).resolve().parents[1]
    checked = set()
    missing = []
    for file in root.rglob('*.py'):
        parts = file.relative_to(root.parent).with_suffix('').parts
        package = '.'.join(parts[:-1])
        for node in ast.walk(ast.parse(file.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                name = '.' * node.level + (node.module or '')
                names = [importlib.util.resolve_name(name, package)]
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                  and node.func.attr == 'import_module' and node.args
                  and isinstance(node.args[0], ast.Constant)
                  and isinstance(node.args[0].value, str)):
                names = [node.args[0].value]
            for name in names:
                if name != root.name and not name.startswith(root.name + '.'):
                    continue
                path = root.parent.joinpath(*name.split('.'))
                checked.add(name)
                if not path.with_suffix('.py').is_file() and not (path / '__init__.py').is_file():
                    missing.append(f'{file.relative_to(root)}:{node.lineno}: {name}')
    if missing:
        raise RuntimeError('Missing packaged internal imports:\n' + '\n'.join(missing))
    return len(checked)


def check_external_imports(package_root=None):
    """Check even deferred unqualified imports, with explicit runtime boundaries.

    ImportError fallback branches are not mandatory dependencies when their
    packaged primary imports work. The cell worker's two root modules are mounted
    by isolation.command; XPolicyLab supplies client_server on the simulator host.
    SAM3 and Torch are separately installed host dependencies, tested by simulator
    preflight rather than this CPU-only package inspection.
    These exceptions are scoped to the exact importer, not blanket module names.
    """
    root = Path(package_root) if package_root else Path(__file__).resolve().parents[1]
    supplied = {
        ('inference/cell_worker.py', 'efficiency_helpers'): root / 'inference/efficiency_helpers.py',
        ('inference/cell_worker.py', 'geometry'): root / 'geometry.py',
        ('hosts/xpolicylab_bridge.py', 'client_server'): None,
        ('hosts/sam3_local.py', 'sam3'): None,
        ('hosts/sam3_local.py', 'torch'): None,
    }
    checked, missing = set(), []
    for file in root.rglob('*.py'):
        tree = ast.parse(file.read_text())
        fallback_nodes = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is not None:
                types = {n.id for n in ast.walk(node.type) if isinstance(n, ast.Name)}
                if types & {'ImportError', 'ModuleNotFoundError'}:
                    for statement in node.body:
                        fallback_nodes.update(ast.walk(statement))
        for node in ast.walk(tree):
            if node in fallback_nodes:
                continue
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                top = name.split('.')[0]
                if top == root.name:
                    continue
                checked.add(top)
                key = (str(file.relative_to(root)), top)
                if key in supplied:
                    target = supplied[key]
                    if target is None or target.is_file():
                        continue
                if importlib.util.find_spec(top) is None:
                    missing.append(f'{file.relative_to(root)}:{node.lineno}: {name}')
    if missing:
        raise RuntimeError('Unresolved external or unqualified imports:\n' + '\n'.join(missing))
    return len(checked)
