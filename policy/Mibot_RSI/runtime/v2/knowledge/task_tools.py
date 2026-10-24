"""In-task tools: Python modules of functions that a development actor saves with
write_tool and imports with tool(name) in later episodes of the same task, evaluation included.

A tool computes from what it is passed: images and arrays, measured poses, numbers. It never moves the robot.

What this module enforces:
- when a tool is written (check_code; snapshots check it again): at most CODE_LIMIT characters; it parses;
  imports only from ALLOWED_MODULES (absolute, no `import *`, no submodule named in DENIED_MODULE_PARTS); no
  identifier in DENIED_NAMES (robot, the knowledge-writing helpers, open, exec, eval, getattr, os, sys, ...) and no
  dunder name; no attribute in DENIED_ATTRIBUTES (file readers and writers of numpy, scipy and OpenCV, also those
  matching FILE_NAME_PATTERN; system modules that allowed modules re-export, such as statistics.sys or
  collections._sys, and sys.modules; frame and code internals) and no dunder attribute other than
  ALLOWED_DUNDER_ATTRIBUTES; no assignment to an attribute of an imported module; at the top level only imports,
  function and class definitions, assignments and the docstring. What runs at import is then the imports, the
  right-hand sides of assignments, decorators, default argument values and class bodies;
- when it is imported (load): its own namespace, without the robot or any session helper; builtins without
  open, exec, eval, compile, getattr and the like, and an __import__ that admits only ALLOWED_MODULES;
- when one of its functions is called through the returned module: `refuse(value)` (the worker's check) looks at
  each argument and refuses the robot, its arms and grippers, the session helpers and functions defined in the
  model's session (classes' methods and instances of the model's own classes are not inspected).

The tool runs in the worker process under the same audit-hook guard and Landlock sandbox (where the kernel has
Landlock) as the model's own python_exec code (unchanged), so it can read and write no more than that code. These checks keep a tool to its
contract against mistakes and casual use; they are not a sandbox against code written to get around them (for
example str.format reads attributes, dunders included, without an attribute node; it returns only text), and the
model's own code has the same rights anyway.
"""
from __future__ import annotations

import ast
import builtins
import functools
import keyword
import re
import sys
import types

CODE_LIMIT = 16000           # characters of code
DOC_LIMIT = 4000             # characters of doc: what each function takes and returns, how to check its output
TOOL_NAME = re.compile(r'[a-z][a-z0-9_]{1,47}')  # a Python name; skill names always contain a dot

ALLOWED_MODULES = frozenset({
    'math', 'cmath', 'statistics', 'numpy', 'scipy', 'cv2', 'itertools', 'functools', 'collections', 'dataclasses',
    'typing', 'json', 'heapq', 'bisect', 'enum', 'copy', '__future__'})
# Submodules of those that read or write files, load libraries or run tests (scipy.io, numpy.ctypeslib, cv2.dnn, ...).
DENIED_MODULE_PARTS = frozenset({
    'io', 'npyio', 'ctypeslib', 'f2py', 'distutils', 'testing', 'datasets', 'misc', 'dnn', 'utils', 'samples',
    'data', 'format', '_lib'})
# Identifiers a tool does not use: the robot, the harness's knowledge-writing helpers, files, processes, the network,
# and the ways around these checks (getattr with a built name, the interpreter's internals).
DENIED_NAMES = frozenset({
    'robot', 'write_skill', 'retire_skill', 'add_evidence', 'update_brief', 'write_tool', 'retire_tool',
    'log_experience', 'save_analysis', 'save_frame',
    'open', 'exec', 'eval', 'compile', 'input', 'breakpoint', 'help', 'exit', 'quit', 'globals', 'locals', 'vars',
    'getattr', 'setattr', 'delattr',
    'importlib', 'subprocess', 'socket', 'os', 'sys', 'builtins', 'shutil', 'pathlib', 'pickle', 'ctypes', 'gc',
    'inspect'})
DENIED_ATTRIBUTES = frozenset({
    # files: OpenCV, NumPy and SciPy readers and writers
    'imread', 'imreadmulti', 'imreadanimation', 'imcount', 'imwrite', 'imwritemulti', 'imwriteanimation',
    'haveImageReader', 'haveImageWriter', 'VideoCapture', 'VideoWriter', 'FileStorage', 'CascadeClassifier',
    'readOpticalFlow', 'writeOpticalFlow', 'load', 'save', 'savez', 'savez_compressed', 'savetxt', 'loadtxt',
    'genfromtxt', 'fromfile', 'tofile', 'dump', 'memmap', 'open_memmap', 'DataSource', 'fromregex', 'loadmat',
    'savemat', 'system', 'popen',
    # submodules (scipy.io, cv2.dnn, ...); 'data' and 'format' are left to the import check, being common names
    *(DENIED_MODULE_PARTS - {'data', 'format'}),
    # the robot and the session helpers as attributes (e.g. self.robot), and the builtins kept from tools
    'robot', 'write_skill', 'retire_skill', 'add_evidence', 'update_brief', 'write_tool', 'retire_tool',
    'log_experience', 'save_analysis', 'save_frame', 'open', 'exec', 'eval', 'getattr', 'setattr', 'delattr'})
# System modules that allowed modules re-export as attributes (statistics.sys, dataclasses.inspect, collections._sys,
# cv2.os, ...), with their private aliases, and the ways into the interpreter's module table and frames.
SYSTEM_MODULES = frozenset({
    'os', 'sys', 'subprocess', 'socket', 'importlib', 'ctypes', 'gc', 'inspect', 'pickle', 'shutil', 'pathlib',
    'builtins', 'types', 'linecache', 'tokenize', 'pkgutil', 'runpy', 'marshal', 'threading', 'multiprocessing',
    'tempfile', 'urllib', 'http', 'requests', 'posix', 'io'})
SYSTEM_ATTRIBUTES = (SYSTEM_MODULES | {'_' + name for name in SYSTEM_MODULES}
                     | {'modules', '_getframe', 'getframe', '_current_frames', 'environ', 'settrace', 'setprofile',
                        '_thread', '_imp', '_posixsubprocess', 'posixsubprocess'}
                     # code in strings: typing evaluates string annotations (get_type_hints, ForwardRef; singledispatch's
                     # register calls it), operator reads attributes by name (typing.operator.attrgetter)
                     | {'get_type_hints', 'ForwardRef', '_eval_type', 'evaluate_forward_ref', 'singledispatch',
                        'singledispatchmethod', 'operator', '_operator', 'attrgetter', 'methodcaller',
                        # numpy.compat (open_latin1 opens files), OpenCV's G-API sources (read video files)
                        'compat', 'gapi'})
# Frames, code objects and generator internals: a way to the caller's namespace.
INTERNAL_ATTRIBUTES = frozenset({
    'gi_frame', 'gi_code', 'gi_yieldfrom', 'cr_frame', 'cr_code', 'cr_await', 'cr_origin', 'ag_frame', 'ag_code',
    'ag_await', 'f_back', 'f_globals', 'f_locals', 'f_builtins', 'f_code', 'tb_frame', 'tb_next', 'co_code',
    'func_globals', 'func_code'})
# Loaders and savers named by convention (cv2.ml.SVM_load, scipy.sparse.load_npz, np.savez, ...).
# Case-sensitive: OpenCV constants such as MORPH_OPEN are fine.
FILE_NAME_PATTERN = re.compile(r'(?:^|_)(?:load|save|dump)(?:$|_|s?z|txt|mat|npz)|(?:^|_)open(?:$|_)')
IMPORTED_NAMES_DENIED = DENIED_ATTRIBUTES | DENIED_MODULE_PARTS | DENIED_NAMES | SYSTEM_ATTRIBUTES  # from x import y
ALLOWED_DUNDER_ATTRIBUTES = frozenset({'__init__', '__name__', '__doc__'})
ALLOWED_DUNDER_NAMES = frozenset({'__name__', '__doc__'})
DENIED_BUILTINS = frozenset({
    'open', 'exec', 'eval', 'compile', '__import__', 'input', 'breakpoint', 'help', 'exit', 'quit', 'globals',
    'locals', 'vars', 'getattr', 'setattr', 'delattr', 'copyright', 'credits', 'license', '__loader__', '__spec__'})
TOP_LEVEL = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign)
WHY = ('a tool computes from the values it is passed and does not use the robot, the knowledge helpers, files, '
       'processes or the network')


class ToolError(ValueError):
    """A tool that was refused (its code, its doc or its arguments); the message is meant for the model."""


def check_name(name):
    if not isinstance(name, str) or not TOOL_NAME.fullmatch(name) or keyword.iskeyword(name) or name.endswith('_'):
        raise ToolError(f'invalid tool name {name!r}: use a lowercase Python name of 2-48 characters, e.g. '
                        'camera_model')
    return name


def is_tool_name(name):
    """Tool names are Python names; skill names always contain a dot, so the two never collide."""
    try:
        check_name(name)
    except ToolError:
        return False
    return True


def _module_problem(module):
    """Why `module` (a dotted name) may not be imported by a tool, or None."""
    parts = str(module).split('.')
    if parts[0] not in ALLOWED_MODULES:
        return f'a tool imports only from {", ".join(sorted(ALLOWED_MODULES - {"__future__"}))}; not {module}'
    if any(part in DENIED_MODULE_PARTS for part in parts[1:]):
        return f'{module} is not allowed in a tool ({WHY})'
    return None


def _identifier_problem(name):
    if name in DENIED_NAMES:
        return f'{name} is not allowed in a tool ({WHY})'
    if name.startswith('__') and name.endswith('__') and name not in ALLOWED_DUNDER_NAMES:
        return f'{name} is not allowed in a tool (interpreter internals)'
    return None


def _node_problem(node, imported):
    """The problem with one syntax node (without its line), or None; adds the names an import binds to `imported`."""
    if isinstance(node, ast.Import):
        problems = []
        for alias in node.names:
            problems.append(_module_problem(alias.name))
            imported.add((alias.asname or alias.name).split('.')[0])
        return next((p for p in problems if p), None)
    if isinstance(node, ast.ImportFrom):
        if node.level or not node.module:
            return 'relative imports are not allowed in a tool'
        problem = _module_problem(node.module)
        for alias in node.names:
            imported.add(alias.asname or alias.name)
            if alias.name == '*':
                problem = problem or 'import * is not allowed in a tool; import the names you use'
            elif alias.name in IMPORTED_NAMES_DENIED or FILE_NAME_PATTERN.search(alias.name):
                problem = problem or f'{node.module}.{alias.name} is not allowed in a tool ({WHY})'
        return problem
    if isinstance(node, ast.Name):
        return _identifier_problem(node.id)
    if isinstance(node, ast.arg):
        return _identifier_problem(node.arg)
    if isinstance(node, ast.alias) and node.asname:
        return _identifier_problem(node.asname)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        # Methods such as __init__ or __repr__ may be defined; the names in DENIED_NAMES may not.
        return f'{node.name} is not allowed in a tool ({WHY})' if node.name in DENIED_NAMES else None
    if isinstance(node, (ast.Global, ast.Nonlocal)):
        return next((p for p in map(_identifier_problem, node.names) if p), None)
    if isinstance(node, ast.Attribute):
        if node.attr in DENIED_ATTRIBUTES or FILE_NAME_PATTERN.search(node.attr) or node.attr in SYSTEM_ATTRIBUTES:
            return f'the attribute {node.attr} is not allowed in a tool ({WHY})'
        if node.attr in INTERNAL_ATTRIBUTES or (node.attr.startswith('__') and node.attr.endswith('__')
                                                and node.attr not in ALLOWED_DUNDER_ATTRIBUTES):
            return f'the attribute {node.attr} is not allowed in a tool (interpreter internals)'
    return None


def _targets(node):
    """What a statement or clause assigns to or deletes."""
    if isinstance(node, (ast.Assign, ast.Delete)):
        return node.targets
    if isinstance(node, (ast.AugAssign, ast.AnnAssign, ast.For, ast.AsyncFor, ast.comprehension)):
        return [node.target]
    if isinstance(node, ast.withitem) and node.optional_vars is not None:
        return [node.optional_vars]
    return []


def _root(node):
    while isinstance(node, (ast.Attribute, ast.Subscript, ast.Starred)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def check_code(code):
    """The public function and class names of a tool's code. Raises ToolError naming the first problem."""
    if not isinstance(code, str) or not code.strip():
        raise ToolError('code must be the Python source of the tool module')
    if len(code) > CODE_LIMIT:
        raise ToolError(f'the code has {len(code)} characters; a tool is at most {CODE_LIMIT}')
    try:
        tree = ast.parse(code, '<tool>')
    except SyntaxError as exc:
        raise ToolError(f'the code does not parse: {exc.msg} (line {exc.lineno})') from None
    for index, node in enumerate(tree.body):
        docstring = (index == 0 and isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                     and isinstance(node.value.value, str))
        if not docstring and not isinstance(node, TOP_LEVEL):
            raise ToolError(f'line {node.lineno}: at the top level a tool only imports, defines functions and '
                            f'classes, and assigns names; move this {type(node).__name__} statement into a function')
    imported = set()
    nodes = list(ast.walk(tree))
    for node in nodes:
        problem = _node_problem(node, imported)
        if problem:
            raise ToolError(f"line {getattr(node, 'lineno', '?')}: {problem}")
    for node in nodes:
        for target in _targets(node):
            for part in ast.walk(target):
                if isinstance(part, (ast.Attribute, ast.Subscript)) and _root(part) in imported:
                    line = getattr(node, 'lineno', getattr(target, 'lineno', '?'))
                    raise ToolError(f'line {line}: a tool does not change what it imports ({ast.unparse(part)})')
    if not any(isinstance(node, ast.FunctionDef) and not node.name.startswith('_') for node in tree.body):
        raise ToolError('a tool defines at least one public function (a name that does not start with _)')
    return [node.name for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.name.startswith('_')]


def check_doc(doc):
    if not isinstance(doc, str) or not doc.strip() or len(doc) > DOC_LIMIT:
        raise ToolError(f'doc must be non-empty text of at most {DOC_LIMIT} characters: what each function takes '
                        'and returns, and how to check its output')
    return doc.strip()


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level:
        raise ImportError('relative imports are not allowed in a tool')
    problem = _module_problem(name)
    for item in fromlist or ():
        if item in IMPORTED_NAMES_DENIED or item == '*' or FILE_NAME_PATTERN.search(item):
            problem = problem or f'{name}.{item} is not allowed in a tool ({WHY})'
    if problem:
        raise ImportError(problem)
    return builtins.__import__(name, globals, locals, fromlist, level)


def tool_builtins():
    """The builtins a tool's module runs with."""
    allowed = {key: value for key, value in vars(builtins).items() if key not in DENIED_BUILTINS}
    allowed['__import__'] = _guarded_import
    return allowed


def _module_name(name):
    return f'v2_tool_{name}'


def load(name, code, filename, doc='', refuse=None, register=True):
    """Run a tool's code in a fresh module and return a module of its public names.

    filename: where the code is (tracebacks and inspect.getsource read it there). refuse(value) -> reason or None:
    the worker's check of each argument passed to the tool's functions. The returned module holds the public
    functions (wrapped with that check), classes and constants; its __doc__ is the tool's doc. register=False
    leaves sys.modules unchanged (write_tool's trial import).
    """
    check_name(name)
    check_code(code)
    module = types.ModuleType(_module_name(name), doc or None)
    module.__dict__['__builtins__'] = tool_builtins()
    module.__file__ = str(filename)
    previous = sys.modules.get(module.__name__)
    sys.modules[module.__name__] = module  # dataclasses and pickling look a class's module up here
    try:
        exec(compile(code, str(filename), 'exec', dont_inherit=True), module.__dict__)
    except BaseException:
        if previous is None:
            sys.modules.pop(module.__name__, None)
        else:
            sys.modules[module.__name__] = previous
        raise
    if not register:
        if previous is None:
            sys.modules.pop(module.__name__, None)
        else:
            sys.modules[module.__name__] = previous
    view = types.ModuleType(name, doc or None)
    view.__file__ = str(filename)
    for key, value in module.__dict__.items():
        if key.startswith('_') or isinstance(value, types.ModuleType) or type(value).__module__ == '__future__':
            continue
        if isinstance(value, types.FunctionType) and value.__module__ == module.__name__ and refuse is not None:
            value = _checked(name, value, refuse)
        setattr(view, key, value)
    return view


def _checked(tool, function, refuse):
    @functools.wraps(function)
    def call(*args, **kwargs):
        for value in list(args) + list(kwargs.values()):
            reason = refuse(value)
            if reason:
                raise ToolError(f'{tool}.{function.__name__}() was passed {reason}; a tool computes from values '
                                '(images, arrays, measured poses, numbers) and does not move the robot')
        return function(*args, **kwargs)
    return call
