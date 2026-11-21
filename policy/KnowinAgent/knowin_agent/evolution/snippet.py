"""Offline checks for includable robot snippets (task skills and shared-library helpers).

A snippet is loaded into the restricted program namespace by an `exec` include, so it must follow the same
rules: only numpy and math imports; no name or attribute starting with '_'; none of the escape builtins;
no top-level robot actions (definitions, imports, docstrings and constant assignments only); and it may not
redefine the namespace's own functions.
"""

from __future__ import annotations

import ast
from typing import Iterable

ALLOWED_IMPORTS = frozenset({"numpy", "math"})
FORBIDDEN_NAMES = frozenset({"open", "eval", "exec", "getattr", "setattr", "delattr", "globals", "locals", "vars",
                             "dir", "input", "compile", "breakpoint", "__import__", "help", "memoryview"})
NAMESPACE_FUNCTIONS = frozenset({
    "move_to", "set_gripper", "check_reach", "checkpoint", "subgoal_done", "get_observation", "get_instruction",
    "step", "reset", "grasp_point", "pick", "place", "look_down", "screw_down", "press_at", "best_yaw",
    "release_yaw", "rim_transfer", "release_retreat", "locate", "inspect", "observe", "grasp_candidates",
    "run_skill", "wait", "interact",
})


def _names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.arg):
        return [node.arg]
    if isinstance(node, ast.Attribute):
        return [node.attr]
    if isinstance(node, ast.alias):
        return [node.name, node.asname or node.name]
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [node.name]
    if isinstance(node, ast.keyword) and node.arg:
        return [node.arg]
    return []


def _module_aliases(tree: ast.Module) -> set[str]:
    aliases = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            aliases |= {a.asname or a.name.split(".")[0] for a in node.names}
    return aliases


def _pure_top_level_value(value: ast.AST, modules: set[str]) -> bool:
    """A top-level assignment may only call numpy/math (np.array, math.radians ...)."""
    for node in ast.walk(value):
        if isinstance(node, ast.Call):
            func = node.func
            while isinstance(func, ast.Attribute):
                func = func.value
            if not (isinstance(func, ast.Name) and func.id in modules):
                return False
        if isinstance(node, (ast.Lambda, ast.Await, ast.Yield, ast.YieldFrom, ast.NamedExpr)):
            return False
    return True


def check_snippet(source: str, filename: str = "<snippet>", *,
                  host_names: Iterable[str] = NAMESPACE_FUNCTIONS) -> list[str]:
    """Return the problems found (empty list = the snippet is includable)."""
    try:
        tree = ast.parse(source, filename)
        compile(tree, filename, "exec")
    except SyntaxError as exc:
        return [f"syntax error line {exc.lineno}: {exc.msg}"]
    host = set(host_names)
    problems: list[str] = []
    modules = _module_aliases(tree)
    for top in tree.body:
        line = getattr(top, "lineno", 0)
        if isinstance(top, (ast.FunctionDef, ast.Import, ast.ImportFrom)):
            if isinstance(top, ast.FunctionDef) and top.name in host:
                problems.append(f"line {line}: redefines namespace function {top.name}()")
            continue
        if isinstance(top, ast.Expr) and isinstance(top.value, ast.Constant):
            continue
        if isinstance(top, (ast.Assign, ast.AnnAssign)):
            targets = top.targets if isinstance(top, ast.Assign) else [top.target]
            if any(isinstance(t, ast.Name) and t.id in host for t in targets):
                problems.append(f"line {line}: rebinds a namespace function")
            if top.value is not None and not _pure_top_level_value(top.value, modules):
                problems.append(f"line {line}: top-level statement calls something other than numpy/math "
                                "(no robot actions at include time)")
            continue
        problems.append(f"line {line}: top-level {type(top).__name__} (only definitions, imports, docstrings "
                        "and constants)")
    for node in ast.walk(tree):
        line = getattr(node, "lineno", 0)
        if isinstance(node, ast.Import):
            bad = [a.name for a in node.names if a.name.split(".")[0] not in ALLOWED_IMPORTS]
            if bad:
                problems.append(f"line {line}: import of {bad} (only numpy and math)")
        elif isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                problems.append(f"line {line}: import from {node.module!r} (only numpy and math)")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            problems.append(f"line {line}: {type(node).__name__.lower()} statement")
        for name in _names(node):
            if name.startswith("_"):
                problems.append(f"line {line}: name {name!r} starts with '_'")
            elif name in FORBIDDEN_NAMES:
                problems.append(f"line {line}: forbidden name {name!r}")
    return sorted(set(problems), key=lambda p: (int(p.split()[1].rstrip(":")) if p.startswith("line ") else 0, p))
