"""Opt-in, simulator-prefix-only native namespace bootstrap.

An existing sitecustomize still executes with its original module identity. In
the selected simulator it sees the pinned native utils namespace; if it replaces
that binding, startup fails rather than silently continuing with mixed imports.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import sys
from pathlib import Path


def _chain_sitecustomize() -> None:
    own_dir = Path(__file__).resolve().parent
    search_path = [
        entry for entry in sys.path if Path(entry or os.curdir).resolve() != own_dir
    ]
    spec = importlib.machinery.PathFinder.find_spec("sitecustomize", search_path)
    if spec is None or spec.loader is None:
        return
    downstream = importlib.util.module_from_spec(spec)
    # Preserve __name__, self-imports and loader behavior of the existing hook.
    sys.modules["sitecustomize"] = downstream
    spec.loader.exec_module(downstream)


def _active() -> bool:
    prefix = os.environ.get("ROBORSI_SIM_PYTHON_PREFIX")
    return (
        os.environ.get("ROBORSI_SIM_BOOTSTRAP") == "1"
        and bool(prefix)
        and Path(prefix).is_absolute()
        and Path(prefix).resolve() == Path(sys.prefix).resolve()
    )


if _active():
    try:
        from _roborsi_sim_namespace import BootstrapError, pin_native_utils

        workspace = os.environ.get("ROBORSI_SIM_BENCH_ROOT")
        if not workspace:
            raise BootstrapError(
                "ROBORSI_SIM_BENCH_ROOT is required for simulator eval"
            )
        binding = pin_native_utils(workspace)
        _chain_sitecustomize()
        binding.verify()
    except BaseException as exc:
        # site.py normally reports a sitecustomize exception and then continues.
        # Simulator startup must not continue with a broken namespace binding.
        # External startup errors can contain runtime credentials.
        print(
            f"[ROBORSI sim bootstrap] startup refused: {type(exc).__name__}",
            file=sys.stderr,
            flush=True,
        )
        os._exit(78)
else:
    _chain_sitecustomize()
