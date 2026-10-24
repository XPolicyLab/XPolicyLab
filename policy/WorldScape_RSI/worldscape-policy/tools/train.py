"""Bootstrap training and spawned workers before importing model dependencies."""

from __future__ import annotations

import os
from pathlib import Path
import runpy
import socket
import tempfile


def isolate_triton_cache() -> Path:
    """Give DeepSpeed's exit-time autotune writer a process-private directory."""
    root = Path(
        os.environ.get("WSP_TRITON_CACHE_ROOT")
        or os.environ.get("TRITON_CACHE_DIR")
        or str(Path(os.environ.get("WSP_CACHE_DIR", tempfile.gettempdir())) / "triton")
    ).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Preserve the original root across multiprocessing spawn. A child must
    # not reuse its parent's cache or nest new caches inside the parent's leaf.
    os.environ["WSP_TRITON_CACHE_ROOT"] = str(root)
    identity = f"{socket.gethostname()}-rank-{os.environ.get('RANK', '0')}-pid-{os.getpid()}"
    prefix = "".join(c if c.isalnum() or c in "-_." else "_" for c in identity)
    cache = Path(tempfile.mkdtemp(prefix=f"{prefix}-", dir=root))
    os.environ["TRITON_CACHE_DIR"] = str(cache)
    return cache


# Spawn re-executes this file as __mp_main__, before importing worker targets.
# Do not put setup inside the main guard, or import worldscape_policy above it.
# Keep the directory through process exit: DeepSpeed writes from atexit hooks.
isolate_triton_cache()

if __name__ == "__main__":
    runpy.run_module("worldscape_policy.cli.train", run_name="__main__")
