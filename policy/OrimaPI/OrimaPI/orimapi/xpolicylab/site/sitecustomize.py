# coding=utf-8
"""Pre-bind RoboDojo/XPolicyLab `utils` before Kit appends `cv2/utils` to sys.path."""

import importlib.machinery
import importlib.util
from pathlib import Path
import sys


def _prebind_robodojo_utils() -> None:
    if "utils" in sys.modules:
        return
    roots = []
    for entry in sys.path:
        if not entry:
            continue
        path = Path(entry)
        if (path / "utils" / "cluttered_generator.py").is_file() or (
            path / "utils" / "process_data.py"
        ).is_file():
            roots.append(str(path))
    if not roots:
        return
    spec = importlib.machinery.PathFinder.find_spec("utils", roots)
    if spec is None:
        return
    module = importlib.util.module_from_spec(spec)
    module.__path__ = [str(Path(root) / "utils") for root in roots]
    sys.modules["utils"] = module
    if spec.loader is not None:
        spec.loader.exec_module(module)


_prebind_robodojo_utils()
