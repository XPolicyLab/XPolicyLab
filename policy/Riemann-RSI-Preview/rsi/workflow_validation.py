"""Resolve executable graph references without importing perception or models."""
import ast
import builtins
from functools import lru_cache
import json
from pathlib import Path

from .task_contract import contained


@lru_cache(maxsize=2048)
def _tree(path, modified, size):
    return ast.parse(Path(path).read_bytes(), filename=path)


def tree(path):
    stat = path.stat()
    return _tree(str(path), stat.st_mtime_ns, stat.st_size)


class Exports:
    def __init__(self, root, mode):
        self.root, self.mode = Path(root), mode

    def module(self, name):
        if name.startswith(('controller', 'bindings.', 'skills.')) and not name.startswith('skills.shared.'):
            name = 'tasks.' + self.mode + '.' + name
        if name.startswith('tasks.') and not name.startswith('tasks.' + self.mode + '.'):
            raise ValueError('Workflow cannot import another task: ' + name)
        relative = name.replace('.', '/')
        for candidate in (relative + '.py', relative + '/__init__.py'):
            if (self.root / candidate).is_file():
                return contained(self.root, candidate)
        raise ValueError('Missing declared workflow module: ' + name)

    def symbol(self, path, symbol, visited=()):
        key = str(path), symbol
        if key in visited or len(visited) > 32:
            raise ValueError('Cyclic workflow export: ' + symbol)
        visited = (*visited, key)
        body = tree(path).body
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == symbol:
                return 'class' if isinstance(node, ast.ClassDef) else 'function'
        package = list(path.relative_to(self.root).with_suffix('').parts[:-1])
        modules = {}
        for node in body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    modules[alias.asname or alias.name] = alias.name
            if not isinstance(node, ast.ImportFrom):
                continue
            base = package[:len(package) - node.level + 1] if node.level else []
            base += node.module.split('.') if node.module else []
            target = '.'.join(base)
            for alias in node.names:
                local = alias.asname or alias.name
                modules[local] = target + '.' + alias.name
                if local == symbol or alias.name == '*':
                    if target == 'builtins':
                        value = getattr(builtins, alias.name, None)
                        if isinstance(value, type) and issubclass(value, BaseException):
                            return 'class'
                    try:
                        return self.symbol(self.module(target), symbol if alias.name == '*' else alias.name, visited)
                    except ValueError:
                        pass
        for node in body:
            if not isinstance(node, ast.Assign):
                continue
            if any(isinstance(t, ast.Name) and t.id == symbol for t in node.targets):
                if isinstance(node.value, ast.Lambda):
                    return 'function'
                if isinstance(node.value, ast.Name):
                    if node.value.id in ('Exception', 'RuntimeError', 'ValueError'):
                        return 'class'
                    return self.symbol(path, node.value.id, visited)
                if (isinstance(node.value, ast.Attribute) and isinstance(node.value.value, ast.Name)
                        and modules.get(node.value.value.id) == 'builtins'):
                    value = getattr(builtins, node.value.attr, None)
                    if isinstance(value, type) and issubclass(value, BaseException):
                        return 'class'
            # Existing functional skill facades alias their complete module.
            if (any(isinstance(t, ast.Subscript) and isinstance(t.value, ast.Attribute)
                    and t.value.attr == 'modules' for t in node.targets)
                    and isinstance(node.value, ast.Name) and node.value.id in modules):
                return self.symbol(self.module(modules[node.value.id]), symbol, visited)
        raise ValueError('Missing exported workflow callable: ' + str(path) + ':' + symbol)

    def check(self, specification, *, error=False):
        if not isinstance(specification, str) or specification.count(':') != 1:
            raise ValueError('Workflow bindings need an explicit module:callable')
        module, symbol = specification.split(':')
        if not symbol.isidentifier() or symbol.startswith('_'):
            raise ValueError('Workflow callable must be a named public export')
        if '/' in module:
            relative = 'tasks/' + self.mode + '/' + module if module.startswith('skills/') and not module.startswith('skills/shared/') else module
            path = contained(self.root, relative)
        else:
            path = self.module(module)
        kind = self.symbol(path, symbol)
        if error and kind != 'class':
            raise ValueError('Workflow exception binding must name a class')
        return type(symbol, (Exception,), {}) if error else lambda *a, **k: None


def validate(root, mode, composition):
    """Preflight names/graphs; runtime retains numerical and observation checks."""
    root = Path(root).resolve()
    task = root / 'tasks' / mode
    exports = Exports(root, mode)
    primary = composition['native_workflow']
    for specification in (composition.get('native_adapter'),):
        if specification:
            exports.check(specification)
    tools = composition.get('native_action_tools', {})

    def tool(name):
        if name not in tools:
            raise ValueError('Workflow calls an undeclared action skill: ' + str(name))
        specification = tools[name]
        exports.check(specification if ':' in specification else specification + ':run')

    def graph(document):
        schema = document.get('schema')
        if schema in ('riemann-declarative-skill-workflow-v1', 'riemann-declarative-skill-workflow-v2',
                      'riemann-declarative-generator-workflow-v1'):
            from harness.runtime import declarative_workflow, declarative_generator_workflow
            bindings = {k: exports.check(v) for k, v in document['bindings'].items()}
            errors = {k: exports.check(v, error=True) for k, v in document['errors'].items()}
            validator = declarative_generator_workflow if 'generator' in schema else declarative_workflow
            validator.validate(document, bindings, errors)
        elif schema in ('riemann-current-observation-programs-v1', 'riemann-mature-teacher-programs-v1'):
            if document.get('package') != 'tasks.' + mode or not document.get('programs'):
                raise ValueError('Workflow programs must use this task package')
            for program in document['programs'].values():
                graph(program)
        elif schema == 'riemann-language-rear-table-episode-graph-v8':
            states = document['states']
            if document.get('entry') not in states:
                raise ValueError('Missing workflow entry state')
            for state in states.values():
                exports.check(state['call'])
                if not state.get('next') or any(t is not None and t not in states for t in state['next'].values()):
                    raise ValueError('Workflow state has an undeclared continuation')
        elif schema == 'organize-reachable-side-push-workflow-r31':
            filename = task / 'skills/original_episode_order/implementation.py'
            assignment = next(n for n in tree(filename).body if isinstance(n, ast.Assign)
                              and any(isinstance(t, ast.Name) and t.id == 'CAPABILITIES' for t in n.targets))
            capabilities = ast.literal_eval(assignment.value)
            for stage in document['primitives']:
                if stage['capability'] not in capabilities:
                    raise ValueError('Unknown original organizing capability')
                module, arguments = capabilities[stage['capability']]
                if stage['arguments'] != arguments:
                    raise ValueError('Original capability arguments differ')
                exports.check('skills.' + module + '.implementation:run')
        elif schema == 'riemann-public-command-workflow-v1':
            for stage in document['stages']:
                tool(stage['skill'])
            for stage in document.get('execution_program', {}).get('stages', []):
                tool(stage['skill'])
            for name in ('view_skill', 'home_skill', 'idle_wrist_reset_skill'):
                if document.get(name):
                    tool(document[name])
            for name in document.get('phase_dispatch', {}).values():
                tool(name)
        else:
            raise ValueError('Unknown executable workflow schema: ' + str(schema))

    for relative in dict.fromkeys([primary, *composition['graph_files']]):
        graph(json.loads(contained(task, relative).read_text()))
