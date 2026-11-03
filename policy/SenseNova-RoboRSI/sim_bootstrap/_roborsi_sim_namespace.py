"""Anchor RoboDojo's native namespace before Kit registers global importers.

This module changes only this interpreter's ``utils`` imports. It does not write
to the native workspace, Conda environment, or any installed package.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
from pathlib import Path

_REQUIRED_FILES = (
    "utils/load_file.py",
    "env/seed_manager/seed_manager.py",
    "scripts/eval_policy.sh",
)


class BootstrapError(RuntimeError):
    """The explicitly selected native workspace cannot be used safely."""


def _workspace_path(workspace: str | Path) -> Path:
    root = Path(workspace)
    if not root.is_absolute():
        raise BootstrapError("native workspace must be an absolute path")
    return root.resolve()


def inspect_workspace(workspace: str | Path) -> tuple[Path, list[str]]:
    """Check native layout without importing native code or simulator packages."""
    root = _workspace_path(workspace)
    missing = [name for name in _REQUIRED_FILES if not (root / name).is_file()]
    for name in _REQUIRED_FILES:
        source = root / name
        if source.exists() and not source.resolve().is_relative_to(root):
            raise BootstrapError(f"native source escapes workspace: {name}")
    if (root / "utils/__init__.py").exists():
        raise BootstrapError("native utils is no longer a namespace; review bootstrap")
    return root, missing


def _module_source(module: object) -> Path | None:
    source = getattr(module, "__file__", None)
    return Path(source).resolve() if source else None


def _package_paths(module: object) -> tuple[Path, ...]:
    return tuple(Path(path).resolve() for path in getattr(module, "__path__", ()))


class NativeBinding:
    def __init__(self, root: Path, package: object, load_file: object):
        self.root = root
        self.package = package
        self.load_file = load_file

    def verify(self) -> None:
        native_utils = self.root / "utils"
        if (
            sys.modules.get("utils") is not self.package
            or _package_paths(self.package) != (native_utils,)
            or sys.modules.get("utils.load_file") is not self.load_file
            or getattr(self.package, "load_file", None) is not self.load_file
            or _module_source(self.load_file) != native_utils / "load_file.py"
        ):
            raise BootstrapError("native utils binding was replaced after bootstrap")


def pin_native_utils(workspace: str | Path) -> NativeBinding:
    """Bind the selected native package, refusing any already imported foreign one."""
    root, missing = inspect_workspace(workspace)
    if missing:
        raise BootstrapError("native workspace is incomplete: " + ", ".join(missing))
    native_utils = root / "utils"
    package = sys.modules.get("utils")
    if package is not None and (
        _module_source(package) is not None
        or _package_paths(package) != (native_utils,)
    ):
        raise BootstrapError(
            "foreign utils is already imported; refusing to replace it"
        )
    for name, module in tuple(sys.modules.items()):
        if not name.startswith("utils."):
            continue
        source = _module_source(module)
        paths = _package_paths(module)
        if package is None or (
            not (source is not None and source.is_relative_to(native_utils))
            and not (paths and all(path.is_relative_to(native_utils) for path in paths))
        ):
            raise BootstrapError("foreign utils submodule is already imported")
    if package is None:
        spec = importlib.machinery.ModuleSpec("utils", loader=None, is_package=True)
        # A fixed list prevents namespace discovery from merging cv2/XPL utils.
        spec.submodule_search_locations = [str(native_utils)]
        package = importlib.util.module_from_spec(spec)
        sys.modules["utils"] = package
    load_file = sys.modules.get("utils.load_file")
    if load_file is None:
        # Select the checked source explicitly, even if a .pth startup hook has
        # already registered an importer that ignores the package search path.
        spec = importlib.util.spec_from_file_location(
            "utils.load_file", native_utils / "load_file.py"
        )
        if spec is None or spec.loader is None:
            raise BootstrapError("native utils.load_file has no source loader")
        load_file = importlib.util.module_from_spec(spec)
        sys.modules["utils.load_file"] = load_file
        setattr(package, "load_file", load_file)
        try:
            spec.loader.exec_module(load_file)
        except Exception as exc:
            del sys.modules["utils.load_file"]
            delattr(package, "load_file")
            raise BootstrapError(
                f"native utils.load_file import failed ({type(exc).__name__})"
            ) from exc
    binding = NativeBinding(root, package, load_file)
    binding.verify()
    return binding


class _ForeignFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "utils" or fullname == "utils.load_file":
            raise BootstrapError("foreign importer intercepted a pinned native module")
        return None


def check_sim_namespace(
    workspace: str | Path, *, allow_missing_workspace: bool = False
) -> dict:
    """CPU-only install/preflight probe; this does not initialize Kit or a GPU."""
    root, missing = inspect_workspace(workspace)
    if missing and allow_missing_workspace:
        return {
            "status": "skipped",
            "sim_namespace_verified": False,
            "workspace": str(root),
            "missing": missing,
            "reason": "configure a complete native workspace before simulator eval",
        }
    binding = pin_native_utils(root)
    foreign = _ForeignFinder()
    sys.meta_path.insert(0, foreign)
    try:
        importlib.import_module("utils")
        importlib.import_module("utils.load_file")
        binding.verify()
    finally:
        sys.meta_path.remove(foreign)
    return {
        "status": "ok",
        "sim_namespace_verified": True,
        "workspace": str(root),
        "load_file": str(root / "utils/load_file.py"),
        "probe": "native imports survive a later foreign finder; no Kit/App started",
    }
