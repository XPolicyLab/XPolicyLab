"""Restricted execution of agent programs.

A program is plain Python checked before it runs: the only imports are numpy and math, no name or attribute may
start with '_', and the builtins that reach outside the program (files, exec/eval, reflection) are absent. Loops
and function bodies are instrumented with a wall-clock deadline check, and every robot call checks it too.
Globals persist across executions of one session, so helpers defined by one program are available to the next.
"""

from __future__ import annotations

import ast
import builtins
import io
import math
import sys
import time
import traceback
from typing import Callable, Iterable, Optional

import numpy as np

PROGRAM_FILE = "<program>"
LIBRARY_PREFIX = "knowin_agent:"
CHECK_NAME = "_knowin_agent_check"
ITER_NAME = "_knowin_agent_iter"
STDOUT_LIMIT = 8000
STDOUT_KEEP = 400_000

MODULES = {"numpy": np, "math": math}
FORBIDDEN_NAMES = frozenset({
    "exec", "eval", "compile", "open", "input", "globals", "locals", "vars", "dir", "getattr", "setattr",
    "delattr", "__import__", "breakpoint", "memoryview", "help", "exit", "quit",
})
# File access and host-module escapes reachable through numpy, plus frame/code introspection.
FORBIDDEN_ATTRS = frozenset({
    "tofile", "dump", "dumps", "load", "loads", "save", "savez", "savez_compressed", "savetxt", "loadtxt",
    "genfromtxt", "fromfile", "fromregex", "recfromtxt", "recfromcsv", "memmap", "DataSource", "source",
    "lib", "core", "compat", "ctypeslib", "ctypes", "f2py", "distutils", "testing", "os", "sys",
    "system", "popen", "exec", "eval", "globals", "locals",
})
FRAME_PREFIXES = ("f_", "gi_", "cr_", "ag_", "tb_", "co_")
SAFE_BUILTIN_NAMES = (
    "abs", "all", "any", "bin", "bool", "callable", "chr", "complex", "dict", "divmod", "enumerate", "filter",
    "float", "format", "frozenset", "hasattr", "hash", "hex", "int", "isinstance", "issubclass", "iter", "len",
    "list", "map", "max", "min", "next", "oct", "ord", "pow", "range", "repr", "reversed", "round", "set",
    "slice", "sorted", "str", "sum", "tuple", "zip",
    "ArithmeticError", "AssertionError", "Exception", "FloatingPointError", "IndexError", "KeyError",
    "LookupError", "NotImplementedError", "OverflowError", "RuntimeError", "StopIteration", "TypeError",
    "ValueError", "ZeroDivisionError",
)


class SandboxError(Exception):
    """A program construct that is not allowed (or does not parse); nothing of the program ran."""

    def __init__(self, message: str, filename: str = PROGRAM_FILE, lineno: Optional[int] = None):
        super().__init__(message)
        self.filename = filename
        self.lineno = lineno


class ProgramStop(BaseException):
    """Ends the whole program. BaseException, so `except Exception` in agent code does not swallow it; the
    program API also latches it, so any later robot call raises it again."""

    reason = "exception"

    def __init__(self, message: str, detail: Optional[dict] = None):
        super().__init__(message)
        self.detail = detail or {}


class ProgramTimeout(ProgramStop):
    reason = "timeout"


def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level or name not in MODULES:
        raise ImportError(f"only numpy and math can be imported (not {name!r})")
    return MODULES[name]


def _check_name(name: Optional[str], node: ast.AST, filename: str, what: str = "name") -> None:
    if name is None:
        return
    if name.startswith("_") and name != "_":
        raise SandboxError(f"{what} {name!r} is not allowed: names starting with '_' are reserved",
                           filename, getattr(node, "lineno", None))
    if what == "name" and name in FORBIDDEN_NAMES:
        raise SandboxError(f"{name!r} is not available in programs", filename, getattr(node, "lineno", None))


def validate(tree: ast.AST, filename: str = PROGRAM_FILE) -> None:
    """Raise SandboxError on the first construct a program may not use."""
    for node in ast.walk(tree):
        line = getattr(node, "lineno", None)
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in MODULES:
                    raise SandboxError(f"import {alias.name} is not allowed: only `import numpy as np` and "
                                       f"`import math`", filename, line)
                _check_name(alias.asname, node, filename)
        elif isinstance(node, ast.ImportFrom):
            if node.level or node.module not in MODULES:
                raise SandboxError(f"from {node.module} import ... is not allowed: only numpy and math",
                                   filename, line)
            for alias in node.names:
                if alias.name == "*":
                    raise SandboxError("star imports are not allowed; name what you import", filename, line)
                if alias.name in FORBIDDEN_ATTRS:
                    raise SandboxError(f"{node.module}.{alias.name} is not available in programs", filename, line)
                _check_name(alias.name, node, filename)
                _check_name(alias.asname, node, filename)
        elif isinstance(node, ast.Name):
            _check_name(node.id, node, filename)
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr.startswith(FRAME_PREFIXES) or node.attr in FORBIDDEN_ATTRS:
                raise SandboxError(f"attribute .{node.attr} is not available in programs", filename, line)
        elif isinstance(node, ast.ClassDef):
            raise SandboxError("class definitions are not available; use functions and dicts", filename, line)
        elif isinstance(node, (ast.AsyncFunctionDef, ast.Await, ast.AsyncFor, ast.AsyncWith)):
            raise SandboxError("async code is not available; use move_to(..., wait=False) and wait()",
                               filename, line)
        elif isinstance(node, ast.FunctionDef):
            _check_name(node.name, node, filename, "function")
        elif isinstance(node, ast.arg):
            _check_name(node.arg, node, filename, "argument")
        elif isinstance(node, ast.keyword):
            _check_name(node.arg, node, filename, "keyword")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            for name in node.names:
                _check_name(name, node, filename)
        elif isinstance(node, ast.ExceptHandler):
            _check_name(node.name, node, filename)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)):
            _check_name(node.name, node, filename)
        elif isinstance(node, ast.MatchMapping):
            _check_name(node.rest, node, filename)


class _Instrument(ast.NodeTransformer):
    """Deadline checks at the top of every loop body and function body; comprehension iterables wrapped."""

    @staticmethod
    def _check(anchor: ast.AST) -> ast.stmt:
        call = ast.Expr(ast.Call(ast.Name(CHECK_NAME, ast.Load()), [], []))
        return ast.copy_location(call, anchor)

    def _loop(self, node):
        self.generic_visit(node)
        node.body.insert(0, self._check(node.body[0] if node.body else node))
        return node

    visit_For = visit_While = _loop

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        docstring = bool(node.body) and isinstance(node.body[0], ast.Expr) and \
            isinstance(getattr(node.body[0], "value", None), ast.Constant) and isinstance(node.body[0].value.value, str)
        node.body.insert(1 if docstring else 0, self._check(node.body[0]))
        return node

    def visit_comprehension(self, node):
        self.generic_visit(node)
        node.iter = ast.copy_location(ast.Call(ast.Name(ITER_NAME, ast.Load()), [node.iter], []), node.iter)
        return node


def compile_program(source: str, filename: str = PROGRAM_FILE):
    """Parse, validate, instrument and compile; raises SandboxError."""
    if not isinstance(source, str):
        raise SandboxError("program code must be a string", filename)
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        raise SandboxError(f"SyntaxError: {exc.msg}", filename, exc.lineno) from None
    validate(tree, filename)
    tree = ast.fix_missing_locations(_Instrument().visit(tree))
    return compile(tree, filename, "exec")


class Sandbox:
    """One persistent restricted namespace."""

    def __init__(self):
        self.deadline: Optional[float] = None
        self._out = io.StringIO()
        self.files: set[str] = {PROGRAM_FILE}
        safe = {name: getattr(builtins, name) for name in SAFE_BUILTIN_NAMES}
        safe.update(print=self._print, __import__=_restricted_import)
        self.globals: dict = {"__builtins__": safe, "__name__": "knowin_agent_program", CHECK_NAME: self.check,
                              ITER_NAME: self._iter, "np": np, "numpy": np, "math": math}

    # -- deadline ----------------------------------------------------------------------

    def check(self) -> None:
        if self.deadline is not None and time.monotonic() > self.deadline:
            raise ProgramTimeout("program exceeded its wall-clock deadline")

    def _iter(self, iterable: Iterable):
        for index, item in enumerate(iterable):
            if not index & 255:
                self.check()
            yield item

    # -- output ---------------------------------------------------------------------------

    def _print(self, *args, sep=" ", end="\n", file=None, flush=False):
        if self._out.tell() < STDOUT_KEEP:
            self._out.write(str(sep).join(str(a) for a in args) + str(end))

    def take_stdout(self) -> tuple[str, int]:
        """(text truncated to STDOUT_LIMIT keeping head and tail, full length); clears the buffer."""
        text = self._out.getvalue()
        self._out = io.StringIO()
        if len(text) <= STDOUT_LIMIT:
            return text, len(text)
        head, tail = 2000, STDOUT_LIMIT - 2000 - 60
        return text[:head] + f"\n...<{len(text) - head - tail} chars truncated>...\n" + text[-tail:], len(text)

    # -- running ----------------------------------------------------------------------------

    def define(self, values: dict) -> None:
        self.globals.update(values)

    def compile(self, source: str, filename: str = PROGRAM_FILE):
        code = compile_program(source, filename)
        self.files.add(filename)
        return code

    def run(self, code) -> None:
        exec(code, self.globals)  # noqa: S102 - validated, instrumented code in a restricted namespace

    # -- locations --------------------------------------------------------------------------

    @staticmethod
    def authored(filename: str) -> bool:
        return not filename.startswith(LIBRARY_PREFIX)

    def program_line(self) -> Optional[int]:
        """Line of the top-level program statement running now (not a line inside a helper)."""
        frame, line = sys._getframe(1), None
        while frame is not None:
            if frame.f_code.co_filename == PROGRAM_FILE:
                line = frame.f_lineno
            frame = frame.f_back
        return line

    def current_location(self) -> dict:
        """Innermost agent-authored frame on the current stack (file, line) plus the program line."""
        frame = sys._getframe(1)
        found = library = None
        while frame is not None:
            name = frame.f_code.co_filename
            if name in self.files:
                if self.authored(name):
                    found = (name, frame.f_lineno)
                    break
                library = library or (name, frame.f_lineno)
            frame = frame.f_back
        found = found or library
        out = {"file": found[0], "line": found[1]} if found else {"file": None, "line": None}
        out["program_line"] = self.program_line()
        return out

    def error_location(self, exc: BaseException) -> dict:
        """Where an exception left agent code: innermost authored frame (else library frame) and the trace."""
        if isinstance(exc, SandboxError):
            return {"file": exc.filename, "line": exc.lineno, "program_line": None, "trace": []}
        trace = [(frame.f_code.co_filename, line, frame.f_code.co_name)
                 for frame, line in traceback.walk_tb(exc.__traceback__) if frame.f_code.co_filename in self.files]
        authored = [row for row in trace if self.authored(row[0])]
        pick = authored[-1] if authored else (trace[-1] if trace else None)
        program = [row for row in trace if row[0] == PROGRAM_FILE]
        return {"file": pick[0] if pick else None, "line": pick[1] if pick else None,
                "program_line": program[0][1] if program else None,
                "trace": [{"file": f, "line": n, "function": fn} for f, n, fn in trace[-8:]]}


def callable_names(namespace: dict, before: Iterable[str]) -> list[str]:
    """Public callables added to a namespace since `before` (names defined by a library source)."""
    seen = set(before)
    return [k for k, v in namespace.items() if k not in seen and not k.startswith("_") and callable(v)]


def wrap_counted(name: str, fn: Callable, counter: Callable[[str], None]) -> Callable:
    def counted(*args, **kwargs):
        counter(name)
        return fn(*args, **kwargs)

    counted.__name__ = name
    counted.__doc__ = fn.__doc__
    counted.__wrapped__ = fn
    return counted
