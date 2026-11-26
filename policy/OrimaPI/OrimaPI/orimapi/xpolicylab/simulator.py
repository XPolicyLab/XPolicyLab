# coding=utf-8
"""Bootstrap RoboDojo's imports before Kit extends Python's module search path."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys
import traceback


def _bind_simulator_utils(root: Path) -> None:
    directory = root / "utils"
    if not directory.is_dir():
        return
    spec = importlib.machinery.PathFinder.find_spec("utils", [str(root)])
    if spec is None:
        raise ImportError(f"Cannot locate RoboDojo utilities in {directory}")
    module = importlib.util.module_from_spec(spec)
    # Kit adds cv2/utils (a regular package), which otherwise hides RoboDojo's
    # namespace package even when RoboDojo is earlier on sys.path.
    module.__path__ = [str(directory)]
    sys.modules["utils"] = module
    if spec.loader is not None:
        spec.loader.exec_module(module)


def main() -> None:
    try:
        if len(sys.argv) < 2:
            raise ValueError("Provide the RoboDojo simulator entrypoint")
        entry = Path(sys.argv[1]).resolve()
        sys.argv = [str(entry), *sys.argv[2:]]
        sys.path.insert(0, str(entry.parent))
        _bind_simulator_utils(entry.parents[2])
        if os.environ.get("ORIMAPI_RESEARCH_LAYOUT_SELECTION"):
            from orimapi.xpolicylab.layout_sampling import install

            install(os.environ["ORIMAPI_RESEARCH_LAYOUT_SELECTION"])
        runpy.run_path(str(entry), run_name="__main__")
    except BaseException as error:
        if isinstance(error, SystemExit) and error.code in (None, 0):
            raise
        path = os.environ.get("ORIMAPI_SIMULATOR_FAILURE")
        if path:
            # Kit shutdown can replace an exception's exit code with zero.
            Path(path).write_text(
                json.dumps(
                    {
                        "error": f"{type(error).__name__}: {error}",
                        "traceback": traceback.format_exc(),
                    },
                    indent=2,
                )
                + "\n"
            )
        raise


if __name__ == "__main__":
    main()
