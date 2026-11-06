#!/usr/bin/env python3
"""Run RoboTwin evaluation with WSA-specific, non-invasive reliability fixes.

The official RoboTwin source file is read and compiled in memory. It is never
edited on disk. The wrapper makes the requested episode budget depend on
``task_env.test_num`` (the counter used in result reporting), and retries the
known transient renderer-buffer error before handing it to RoboTwin's normal
environment-error path.
"""

from __future__ import annotations

import ast
import os
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any


class _EpisodeBudgetTransformer(ast.NodeTransformer):
    def __init__(self) -> None:
        self.in_eval_remote_policy = False
        self.replacements = 0
        self.already_uses_result_counter = False

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:  # noqa: N802
        previous = self.in_eval_remote_policy
        self.in_eval_remote_policy = node.name == "eval_remote_policy"
        node = self.generic_visit(node)
        self.in_eval_remote_policy = previous
        return node

    def visit_While(self, node: ast.While) -> Any:  # noqa: N802
        if self.in_eval_remote_policy and self._is_budget_test(node.test, "succ_seed"):
            comparison = node.test
            assert isinstance(comparison, ast.Compare)
            comparison.left = ast.Attribute(
                value=ast.Name(id="task_env", ctx=ast.Load()),
                attr="test_num",
                ctx=ast.Load(),
            )
            self.replacements += 1
        elif self.in_eval_remote_policy and self._is_budget_test(node.test, "task_env.test_num"):
            self.already_uses_result_counter = True
        return self.generic_visit(node)

    @staticmethod
    def _is_budget_test(test: ast.expr, left_name: str) -> bool:
        if not (
            isinstance(test, ast.Compare)
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Lt)
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Name)
            and test.comparators[0].id == "test_num"
        ):
            return False
        if left_name == "succ_seed":
            return isinstance(test.left, ast.Name) and test.left.id == "succ_seed"
        return (
            isinstance(test.left, ast.Attribute)
            and isinstance(test.left.value, ast.Name)
            and test.left.value.id == "task_env"
            and test.left.attr == "test_num"
        )


def compile_robotwin_eval(source: str, filename: str):
    tree = ast.parse(source, filename=filename)
    transformer = _EpisodeBudgetTransformer()
    tree = transformer.visit(tree)
    ast.fix_missing_locations(tree)
    if transformer.replacements != 1 and not transformer.already_uses_result_counter:
        raise RuntimeError(
            "Could not locate RoboTwin's eval_remote_policy episode-budget loop. "
            "The upstream script changed; refusing to run without the valid-episode fix."
        )
    return compile(tree, filename=filename, mode="exec"), transformer.replacements


def _install_render_retry(base_task_module: ModuleType) -> None:
    base_task = base_task_module.Base_Task
    original_get_obs = base_task.get_obs
    if getattr(original_get_obs, "_wsa_render_retry", False):
        return

    retries = max(0, int(os.environ.get("WSA_RENDER_BUFFER_RETRIES", "10")))
    delay_s = max(0.0, float(os.environ.get("WSA_RENDER_BUFFER_RETRY_DELAY_S", "0.25")))

    def get_obs_with_render_retry(self):
        for attempt in range(retries + 1):
            try:
                return original_get_obs(self)
            except RuntimeError as exc:
                is_buffer_error = "cannot create buffer" in str(exc).lower()
                if not is_buffer_error or attempt >= retries:
                    raise
                print(
                    "[WSA][renderer] transient 'cannot create buffer'; "
                    f"retrying get_obs ({attempt + 1}/{retries})",
                    flush=True,
                )
                if delay_s:
                    time.sleep(delay_s)

    get_obs_with_render_retry._wsa_render_retry = True
    base_task.get_obs = get_obs_with_render_retry


def _run_official_entrypoint(namespace: dict[str, Any]) -> None:
    parse_args = namespace.get("parse_args")
    parse_bool = namespace.get("parse_bool")
    main_entrypoint = namespace.get("main")
    batch_entrypoint = namespace.get("main_batch")
    if not callable(parse_args) or not callable(parse_bool) or not callable(main_entrypoint):
        raise RuntimeError("RoboTwin evaluation entry points are incomplete or unsupported.")

    # Keep the same startup order as RoboTwin's ``if __name__ == '__main__'``
    # block while allowing the WSA patches to be installed first.
    from test_render import Sapien_TEST

    Sapien_TEST()
    parsed_args = parse_args()
    if parse_bool(parsed_args.get("eval_batch", False)):
        if not callable(batch_entrypoint):
            raise RuntimeError("RoboTwin batch evaluation entry point 'main_batch' was not found.")
        batch_entrypoint(parsed_args)
    else:
        main_entrypoint(parsed_args)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: robotwin_eval_wrapper.py EVAL_SCRIPT [EVAL_ARGS ...]")

    eval_script = Path(sys.argv[1]).expanduser().resolve()
    if not eval_script.is_file():
        raise FileNotFoundError(f"RoboTwin evaluation script not found: {eval_script}")

    robotwin_root = eval_script.parent.parent
    os.chdir(robotwin_root)
    sys.path.insert(0, str(eval_script.parent))
    sys.path.insert(0, str(robotwin_root))
    sys.argv = [str(eval_script), *sys.argv[2:]]

    source = eval_script.read_text(encoding="utf-8")
    code, replacements = compile_robotwin_eval(source, str(eval_script))
    if replacements:
        print(
            "[WSA] RoboTwin valid-episode guard enabled: renderer failures before "
            "the first action do not consume --test-num.",
            flush=True,
        )

    module_name = "_wsa_robotwin_eval"
    eval_module = ModuleType(module_name)
    eval_module.__file__ = str(eval_script)
    eval_module.__package__ = None
    sys.modules[module_name] = eval_module
    namespace: dict[str, Any] = eval_module.__dict__
    exec(code, namespace)

    from envs import _base_task

    _install_render_retry(_base_task)
    _run_official_entrypoint(namespace)


if __name__ == "__main__":
    main()
