import ast
import atexit
import collections
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import threading
import time
from types import FunctionType, MappingProxyType

def digest(content):
    return hashlib.sha256(content).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def contained(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Skill file escapes the frozen library: ' + relative)
    return path

class Trace:

    def __init__(self, directory, mode, limit=20000):
        self.lock = threading.RLock()
        self.counts = collections.Counter()
        self.errors = collections.Counter()
        self.limit = limit
        self.events = 0
        self.mode = mode
        self.started = time.time()
        self.stream = None
        self.path = None
        if directory is not None:
            directory = Path(directory).resolve()
            directory.mkdir(parents=True, exist_ok=True)
            self.path = directory / (mode + '-' + str(os.getpid()) + '.jsonl')
            self.stream = self.path.open('x', buffering=1)
            atexit.register(self.close)

    def emit(self, slot, skill, phase):
        with self.lock:
            if phase == 'enter':
                self.counts[skill] += 1
            if phase == 'raise':
                self.errors[skill] += 1
            first_use = phase == 'enter' and self.counts[skill] == 1
            if self.stream is not None and (self.events < self.limit or first_use):
                self.stream.write(json.dumps({'task_mode': self.mode, 'slot': slot, 'skill': skill, 'event': phase, 'time_ns': time.monotonic_ns()}) + '\n')
                self.events += 1

    def close(self):
        with self.lock:
            if self.stream is not None:
                self.stream.write(json.dumps({'event': 'summary', 'task_mode': self.mode, 'started_at_unix': self.started, 'finished_at_unix': time.time(), 'calls': dict(self.counts), 'errors': dict(self.errors), 'event_log_truncated': sum(self.counts.values()) * 2 > self.limit}) + '\n')
                self.stream.close()
                self.stream = None

class Library:

    def __init__(self, root, mode, trace_directory=None):
        self.root = Path(root).resolve()
        self.registry_relative = 'configs/skill-registry.json'
        self.source_versioning = 'git'
        self._required = {self.registry_relative}
        self.verify()
        self.registry = read(self.root / self.registry_relative)
        if mode not in self.registry['modes']:
            raise ValueError('Unregistered task/mode; no task-name fallback')
        self.mode = mode
        composition = self.registry['modes'][mode]['composition']
        self._required.add(composition)
        self.verify()
        self.plan = read(contained(self.root, composition))
        self.validate_plan()
        self.bindings = MappingProxyType(dict(self.plan['bindings']))
        self._required.update((record['scaffold'] for record in self.plan['modules']))
        for identifier in self.bindings.values():
            spec = self.registry['skills'][identifier]
            self._required.update((spec[key] for key in ('implementation', 'workflow') if key in spec))
        self.verify()
        self.trace = Trace(trace_directory, mode)
        self._codes = {}
        self._binding_lock = threading.RLock()

    def verify(self, full=False):
        required = set(self._required)
        if full:
            registry = read(contained(self.root, self.registry_relative))
            required.update(record['composition'] for record in registry['modes'].values())
            required.update(record[key] for record in registry['skills'].values()
                            for key in ('implementation', 'workflow') if key in record)
        for relative in required:
            contained(self.root, relative)

    def validate_plan(self):
        if self.plan['task_mode'] != self.mode or self.plan['schema'] != 'riemann-skill-composition-v1':
            raise ValueError('Composition identity/schema mismatch')
        skills = self.registry['skills']
        for module in self.plan['modules']:
            for slot in module['slots']:
                selected = self.plan['bindings'].get(slot['id'])
                if selected not in skills or selected not in slot['allowed']:
                    raise ValueError('Unknown or undeclared skill selection: ' + slot['id'])
                if skills[selected]['interface_sha256'] != slot['interface_sha256']:
                    raise ValueError('Incompatible skill interface: ' + slot['id'])
        declared = {slot['id'] for module in self.plan['modules'] for slot in module['slots']}
        if set(self.plan['bindings']) != declared:
            raise ValueError('Composition has missing or extra skill bindings')


    def bind(self, slot, namespace, class_namespace=None):
        if slot not in self.bindings:
            raise ValueError('Undeclared skill binding')
        skill_id = self.bindings[slot]
        spec = self.registry['skills'][skill_id]
        key = (skill_id, spec['owner'], spec['future_flags'])
        if key not in self._codes:
            node = ast.parse(contained(self.root, spec['implementation']).read_text()).body[0]
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                raise ValueError('Skill implementation must contain one callable definition')
            if spec.get('workflow'):
                from .workflow import compile_nodes
                workflow = read(contained(self.root, spec['workflow']))
                if workflow['schema'] != 'riemann-executable-workflow-v1':
                    raise ValueError('Unknown orchestration workflow schema')
                node.body = compile_nodes(workflow['nodes'])
                if digest(ast.dump(node, include_attributes=False).encode()) != spec['ast_sha256']:
                    raise ValueError('Orchestration graph differs from its declared implementation identity')
            needs_class = class_namespace is not None and any((isinstance(child, ast.Name) and child.id in ('super', '__class__') for child in ast.walk(node)))
            if not needs_class:
                tree = ast.Module(body=[node], type_ignores=[])
            else:
                update = ast.parse('locals().update(__riemann_class_context__)').body[0]
                shell = ast.ClassDef(name='__RiemannSkillShell', bases=[], keywords=[], body=[update, node], decorator_list=[])
                tree = ast.Module(body=[shell], type_ignores=[])
            ast.fix_missing_locations(tree)
            self._codes[key] = (compile(tree, str(contained(self.root, spec['implementation'])), 'exec', flags=spec['future_flags'], dont_inherit=True), needs_class)
        (code, needs_class) = self._codes[key]
        local = dict(class_namespace) if class_namespace is not None and (not needs_class) else {}
        with self._binding_lock:
            previous = namespace.get('__riemann_class_context__')
            existed = '__riemann_class_context__' in namespace
            if class_namespace is not None:
                namespace['__riemann_class_context__'] = dict(class_namespace)
            try:
                exec(code, namespace, local)
            finally:
                if class_namespace is not None:
                    if existed:
                        namespace['__riemann_class_context__'] = previous
                    else:
                        namespace.pop('__riemann_class_context__', None)
        if not needs_class:
            value = local[spec['symbol']]
        else:
            value = vars(local['__RiemannSkillShell'])[spec['symbol']]
        return self._decorate(value, slot, skill_id, spec, namespace)

    def _decorate(self, value, slot, skill_id, spec, namespace):
        if getattr(value, '__riemann_skill_bound__', False):
            return value
        if isinstance(value, staticmethod):
            return staticmethod(self._decorate(value.__func__, slot, skill_id, spec, namespace))
        if isinstance(value, classmethod):
            return classmethod(self._decorate(value.__func__, slot, skill_id, spec, namespace))
        if isinstance(value, property):
            accessors = [self._decorate(function, slot, skill_id, spec, namespace) if function else None for function in (value.fget, value.fset, value.fdel)]
            return property(*accessors, doc=value.__doc__)
        if not callable(value):
            return value
        if hasattr(value, '__qualname__'):
            value.__qualname__ = spec['qualname']

        def prepare():
            if spec['owner'] and inspect.isfunction(value) and ('__class__' in value.__code__.co_freevars):
                owner = namespace[spec['owner']]
                index = value.__code__.co_freevars.index('__class__')
                value.__closure__[index].cell_contents = owner
        if inspect.iscoroutinefunction(value):

            @functools.wraps(value)
            async def invoke(*arguments, **keywords):
                prepare()
                self.trace.emit(slot, skill_id, 'enter')
                try:
                    result = await value(*arguments, **keywords)
                except BaseException:
                    self.trace.emit(slot, skill_id, 'raise')
                    raise
                self.trace.emit(slot, skill_id, 'return')
                return result
        elif inspect.isgeneratorfunction(value):

            @functools.wraps(value)
            def invoke(*arguments, **keywords):
                prepare()
                self.trace.emit(slot, skill_id, 'enter')
                try:
                    result = (yield from value(*arguments, **keywords))
                except BaseException:
                    self.trace.emit(slot, skill_id, 'raise')
                    raise
                self.trace.emit(slot, skill_id, 'return')
                return result
        else:

            @functools.wraps(value)
            def invoke(*arguments, **keywords):
                prepare()
                self.trace.emit(slot, skill_id, 'enter')
                try:
                    result = value(*arguments, **keywords)
                except BaseException:
                    self.trace.emit(slot, skill_id, 'raise')
                    raise
                self.trace.emit(slot, skill_id, 'return')
                return result
        contextual = FunctionType(invoke.__code__, namespace, invoke.__name__, invoke.__defaults__, invoke.__closure__)
        contextual.__kwdefaults__ = invoke.__kwdefaults__
        functools.update_wrapper(contextual, value)
        contextual.__riemann_skill_bound__ = True
        return contextual
