"""Stable public entrypoint for the XBrain-v1 inference-only runtime.

The implementation deliberately retains the internal ``giga_models`` module
layout used by the checkpoint.  Consumers should import from this package so
the policy does not silently bind to an unrelated upstream ``giga-models``
installation.
"""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def runtime_info() -> dict[str, str]:
    """Return the installed distribution and implementation locations."""
    try:
        distribution_version = version("xbrain-v1-inference-runtime")
    except PackageNotFoundError:
        distribution_version = "editable-or-uninstalled"

    import giga_models

    return {
        "distribution": "xbrain-v1-inference-runtime",
        "version": distribution_version,
        "implementation_root": str(Path(giga_models.__file__).resolve().parent),
    }


from giga_models.pipelines.vla.giga_brain_0 import GigaBrain0Pipeline

__all__ = ["GigaBrain0Pipeline", "runtime_info"]
