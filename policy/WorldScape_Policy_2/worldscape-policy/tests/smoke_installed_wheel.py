"""Smoke-test an installed wheel from outside the source checkout."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


SMOKE = r"""
import importlib
import importlib.abc
import importlib.metadata
import sys
from pathlib import Path

sys.path.insert(0, __INSTALL_ROOT__)
distribution = importlib.metadata.distribution("worldscape-policy")
package_root = Path(distribution.locate_file("")).resolve()
source_root = Path(__SOURCE_ROOT__).resolve()
assert not package_root.is_relative_to(source_root), (package_root, source_root)

blocked_prefixes = (
    "worldscape_policy.compat",
    "worldscape_policy.legacy_server",
    "evaluation",
    "groot",
    "socket_test_optimized_AR",
    "dreamzero_real_robot_server",
)

class BlockLegacyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(
            fullname == prefix or fullname.startswith(prefix + ".")
            for prefix in blocked_prefixes
        ):
            raise AssertionError(f"native entrypoint imported legacy module: {fullname}")
        return None

sys.meta_path.insert(0, BlockLegacyImports())

modules = (
    "worldscape_policy.cli.evaluate",
    "worldscape_policy.cli.hdf5_replay",
    "evals.agilex.action_adapter",
    "evals.agilex.evaluate",
    "evals.agilex.observation_adapter",
    "evals.agilex.robot",
    "evals.agilex.replay",
    "evals.agilex.safety",
    "evals.agilex.server",
    "evals.libero.adapter",
    "evals.libero.task_suite",
    "evals.robotwin2.adapter",
    "worldscape_policy.cli.serve",
    "worldscape_policy.cli.train",
)
for name in modules:
    importlib.import_module(name)

from transformers import Qwen3VLForConditionalGeneration
assert Qwen3VLForConditionalGeneration.__name__ == "Qwen3VLForConditionalGeneration"
files = {str(file) for file in distribution.files or ()}
for excluded_path in (
    "worldscape_policy/compat/",
    "worldscape_policy/legacy_server/",
    "worldscape_policy/legacy_server.py",
    "evaluation/",
    "groot/",
    "socket_test_optimized_AR.py",
    "dreamzero_real_robot_server.py",
):
    assert not any(path.startswith(excluded_path) for path in files), excluded_path

scripts = {
    ep.name: ep
    for ep in distribution.entry_points
    if ep.group == "console_scripts"
}
native = {
    "wsp-eval",
    "wsp-serve",
    "wsp-train",
}
assert native == scripts.keys()
for name in native:
    assert callable(scripts[name].load()), name
"""


def main(wheel: str) -> None:
    wheel_path = Path(wheel).resolve()
    source_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as workdir:
        install_root = Path(workdir) / "site"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--target",
                str(install_root),
                "--no-deps",
                str(wheel_path),
            ],
            check=True,
        )
        script = SMOKE.replace("__SOURCE_ROOT__", repr(str(source_root))).replace(
            "__INSTALL_ROOT__", repr(str(install_root))
        )
        subprocess.run([sys.executable, "-I", "-c", script], cwd=workdir, check=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} WHEEL")
    main(sys.argv[1])
