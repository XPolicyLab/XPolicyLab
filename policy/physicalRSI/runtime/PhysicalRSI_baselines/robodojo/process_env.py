"""Process environment helpers for source-checkout task adapters."""

import os
from pathlib import Path


def with_project_path(environment):
    """Preserve caller paths while keeping the installed adapter importable.

    Native and model workers run from fresh output directories.  In a source
    checkout a caller-provided ``PYTHONPATH`` may contain only model code, so
    prepend the adapter root without discarding that explicit path.
    """
    result = dict(environment or {})
    root = str(Path(__file__).resolve().parents[2])
    current = result.get("PYTHONPATH")
    result["PYTHONPATH"] = os.pathsep.join(
        [root] + ([current] if current else [])
    )
    return result
