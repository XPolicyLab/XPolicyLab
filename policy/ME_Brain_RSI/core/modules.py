"""Load shared source files into independent task namespaces.

The module store deduplicates bytes, not perception methods. Absolute imports
inside an implementation resolve only within its own registered module graph.
"""
from __future__ import annotations

import ast
import builtins
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import sys
import threading
import types

ROOT = Path(__file__).resolve().parent.parent
_IMPORT = builtins.__import__
_LOCK = threading.RLock()
_SCOPES = {}


class ResourceNames(ast.NodeTransformer):
    def visit_Name(self, node):
        if node.id == '__file__' and isinstance(node.ctx, ast.Load):
            return ast.copy_location(ast.Name(id='__resource_file__', ctx=node.ctx), node)
        return node


class ModuleLoader(importlib.abc.Loader):
    def __init__(self, scope, name, record):
        self.scope, self.name, self.record = scope, name, record

    def create_module(self, spec):
        return None

    def get_filename(self, fullname):
        return str(ROOT / self.record['file'])

    def get_source(self, fullname):
        return (ROOT / self.record['file']).read_text()

    def exec_module(self, module):
        if self.record.get('namespace'):
            module.__path__ = []
            return
        path = ROOT / self.record['file']
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.record['sha256']:
            raise ValueError('Implementation checksum mismatch: ' + self.name)
        module.__file__ = str(path)
        module.__policy_root__ = str(ROOT)
        module.__resource_file__ = str(ROOT / self.record['resource'])
        if self.record['package']:
            module.__path__ = []
        module.__builtins__ = dict(vars(builtins), __import__=self.scope.import_hook)
        tree = ResourceNames().visit(ast.parse(raw, filename=str(path)))
        exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), module.__dict__)


class ScopeFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        for prefix, scope in tuple(_SCOPES.items()):
            if not fullname.startswith(prefix + '.'):
                continue
            name = fullname[len(prefix) + 1:]
            record = scope.modules.get(name)
            if record is None:
                return None
            loader = ModuleLoader(scope, name, record)
            if record.get('namespace'):
                return importlib.machinery.ModuleSpec(fullname, loader, is_package=True)
            return importlib.util.spec_from_loader(fullname, loader, is_package=record.get('package', False))
        return None


_FINDER = ScopeFinder()


class TaskModules:
    def __new__(cls, task):
        prefix = __package__.split('.core')[0] + '._implementations.' + task
        with _LOCK:
            if prefix in _SCOPES:
                return _SCOPES[prefix]
            value = super().__new__(cls)
            value.prefix = prefix
            _SCOPES[prefix] = value
            return value

    def __init__(self, task):
        if hasattr(self, 'modules'):
            return
        self.task = task
        self.modules = json.loads((ROOT / 'tasks' / task / 'modules.json').read_text())
        self.importlib = types.ModuleType('importlib')
        self.importlib.__dict__.update(vars(importlib))
        self.importlib.import_module = self.import_module
        self.importlib.util = types.ModuleType('importlib.util')
        self.importlib.util.__dict__.update(vars(importlib.util))
        self.importlib.util.find_spec = self.find_spec
        with _LOCK:
            parts = self.prefix.split('.')
            start = len(__package__.split('.core')[0].split('.')) + 1
            for i in range(start, len(parts) + 1):
                name = '.'.join(parts[:i])
                if name not in sys.modules:
                    package = types.ModuleType(name)
                    package.__path__ = []
                    package.__package__ = name
                    sys.modules[name] = package
            if _FINDER not in sys.meta_path:
                sys.meta_path.insert(0, _FINDER)

    def import_module(self, name, package=None):
        if name.startswith('.'):
            if package and not package.startswith(self.prefix):
                package = self.prefix + '.' + package
            return importlib.import_module(name, package)
        if name in self.modules:
            return importlib.import_module(self.prefix + '.' + name)
        return importlib.import_module(name, package)

    def find_spec(self, name, package=None):
        if name in self.modules:
            name = self.prefix + '.' + name
        return importlib.util.find_spec(name, package)

    def import_hook(self, name, globals=None, locals=None, fromlist=(), level=0):
        if level:
            return _IMPORT(name, globals, locals, fromlist, level)
        if name == 'importlib' or name == 'importlib.util':
            return self.importlib.util if name.endswith('.util') and fromlist else self.importlib
        if name not in self.modules:
            return _IMPORT(name, globals, locals, fromlist, level)
        module = self.import_module(name)
        if fromlist:
            for member in fromlist:
                if member != '*' and not hasattr(module, member) and name + '.' + member in self.modules:
                    self.import_module(name + '.' + member)
            return module
        return self.import_module(name.split('.')[0])

    def resolve(self, entry):
        module, attribute = entry.split(':', 1)
        return getattr(self.import_module(module), attribute)
