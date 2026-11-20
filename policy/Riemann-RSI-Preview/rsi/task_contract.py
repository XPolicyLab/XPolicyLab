"""Validate executable declarations; Git owns source history, not hash snapshots."""
import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def native_action_bindings(mode, *, root=ROOT):
    """Read task-owned run exports from the one executable composition."""
    if not isinstance(mode, str) or not mode.isidentifier():
        raise ValueError('Expected a registered native task identifier')
    root = Path(root).resolve()
    composition = json.loads(contained(root, 'tasks/'+mode+'/composition.json').read_text())
    bindings = composition.get('native_action_tools')
    if (composition.get('task_mode') != mode or not isinstance(bindings, dict) or not bindings
            or any(not isinstance(name, str) or not name.isidentifier()
                   or not isinstance(path, str) or not path.endswith('.py')
                   for name, path in bindings.items())):
        raise ValueError('Native task tools require explicit task-owned Python run exports')
    for path in bindings.values():
        contained(root, path)
    return dict(bindings)


def metadata(mode, *, root=ROOT):
    root = Path(root)
    if not isinstance(mode, str) or not mode.isidentifier():
        raise ValueError('Expected a registered task identifier')
    data = json.loads((root / 'tasks' / mode / 'task.json').read_text())
    if data.get('task') != mode:
        raise ValueError('Task metadata identity differs: ' + mode)
    return data


RULE_SCHEMA = 'riemann-rule-composition-v1'


def is_git_managed(root, mode):
    """Every current task declares Git-managed executable composition."""
    path = Path(root) / 'tasks' / mode / 'composition.json'
    return path.is_file() and json.loads(path.read_text()).get('source_versioning') == 'git'


def verify(root, mode):
    root = Path(root).resolve()
    task = root / 'tasks' / mode
    metadata = json.loads(contained(root, 'tasks/' + mode + '/task.json').read_text())
    composition = json.loads((task / 'composition.json').read_text())
    if metadata.get('task') != mode or composition.get('task_mode') != mode:
        raise ValueError('Task identity and composition must agree')
    from .rule_reuse import implementation_mode
    owner = implementation_mode(root, mode)
    if owner != mode:
        return verify(root, owner)
    if not is_git_managed(root, mode):
        raise ValueError('Expected an explicit Git-managed task contract')
    if composition.get('source_versioning') == 'git' and metadata.get('backend') != 'learned_policy':
        if (composition.get('schema') != RULE_SCHEMA or metadata.get('backend') != 'rule'
                or composition.get('backend') != 'rule'):
            raise ValueError('Current Rule declarations require one schema and backend')
    for name in ('modules', 'source', 'runtime', 'variants', 'varients'):
        if (task / name).exists():
            raise ValueError('Task contains retired scaffolding: ' + name)
    if metadata.get('backend') == 'learned_policy':
        from rsi.skills.runtime import Library
        library = Library(root, mode)
        library.trace.close()
        return len(library._required)
    entries = declared_python_entries(root, mode)
    for relative in entries:
        contained(root, relative)
    closure = static_python_closure(root, mode, entries)
    for relative in closure:
        contained(root, relative)
    skills = composition.get('task_skills', {})
    if (not isinstance(skills, dict) or 'modules' in composition
            or any(not isinstance(composition.get(name), list) for name in ('graph_files', 'resource_files'))
            or not isinstance(composition.get('native_workflow'), str)):
        raise ValueError('Current Rule declarations need canonical skill, graph and resource fields')
    for name, implementation in skills.items():
        if implementation != 'skills/' + name + '/implementation.py':
            raise ValueError('Task skill must have an explicit local implementation')
        contained(root, 'tasks/' + mode + '/' + implementation)
        contained(root, 'tasks/' + mode + '/skills/' + name + '/SKILL.md')
    for relative in (*composition.get('graph_files', []), *composition.get('resource_files', [])):
        contained(task, relative)
    from .workflow_validation import validate
    validate(root, mode, composition)
    return len(closure)


def declared_python_entries(root, mode, *, load=None):
    """Keep explicit source roots separate from their generated import closure."""
    root = Path(root).resolve()
    task = root / 'tasks' / mode
    load = load or (lambda path: json.loads(Path(path).read_text()))
    composition = load(task / 'composition.json')
    declared = composition['python_entry_files']
    if not isinstance(declared, list) or any(not isinstance(p, str) for p in declared):
        raise ValueError('Composition Python entry roots must be a path list')
    if len(declared) != len(set(declared)):
        raise ValueError('Composition Python entry roots must be unique')
    entries = set(declared)
    skills = composition.get('task_skills', {})
    implementations = skills.values() if isinstance(skills, dict) else skills
    for implementation in implementations:
        if isinstance(implementation, str) and not implementation.endswith('.py') and '/' not in implementation:
            implementation = 'skills/' + implementation + '/implementation.py'
        if isinstance(implementation, str) and implementation.endswith('.py'):
            entries.add(str(Path('tasks') / mode / implementation))
    if not entries or any(not isinstance(p, str) or not p.endswith('.py') for p in entries):
        raise ValueError('Declare Python entry files, rather than generated closures')
    for relative in entries:
        contained(root, relative)
    return sorted(entries)


def contained(root, relative):
    path = root / relative
    if (Path(relative).is_absolute() or '..' in Path(relative).parts
            or not path.is_file() or not path.resolve().is_relative_to(root)
            or any((root / parent).is_symlink() for parent in [Path(relative), *Path(relative).parents])):
        raise ValueError('A native skill input must be a contained ordinary file: ' + relative)
    return path


def static_python_closure(root, mode, entries, *, source_reader=None):
    root = Path(root).resolve()
    pending = list(entries)
    found = set()
    while pending:
        relative = pending.pop()
        if relative in found:
            continue
        if (relative.startswith('tasks/') and not relative.startswith('tasks/' + mode + '/')):
            raise ValueError('Extract a cross-task dependency into shared skills: ' + relative)
        path = contained(root, relative)
        found.add(relative)
        if len(found) > 512:
            raise ValueError('Narrow the native task closure instead of copying the framework')
        parts = list(Path(relative).with_suffix('').parts)
        package = parts[:-1]
        for length in range(1, len(package) + 1):
            initializer = '/'.join(package[:length]) + '/__init__.py'
            if (root / initializer).is_file() and initializer not in found:
                pending.append(initializer)
        source = source_reader(path) if source_reader is not None else path.read_bytes()
        for node in ast.walk(ast.parse(source, filename=relative)):
            modules = []
            if isinstance(node, ast.Import):
                modules.extend(alias.name.split('.') for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                prefix = package[:len(package) - node.level + 1] if node.level else []
                target = prefix + (node.module.split('.') if node.module else [])
                modules.append(target)
                modules.extend(target + alias.name.split('.') for alias in node.names if alias.name != '*')
            for module in modules:
                if not module:
                    continue
                name = '/'.join(module)
                for candidate in (name + '.py', name + '/__init__.py'):
                    if (root / candidate).is_file() and candidate not in found:
                        pending.append(candidate)
    return found
