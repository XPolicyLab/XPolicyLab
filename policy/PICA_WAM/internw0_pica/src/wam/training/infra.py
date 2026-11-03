"""Optional compiler setup shared by training and offline preparation."""

from __future__ import annotations

import os
import socket
from pathlib import Path


def configure_compiler_cache(cfg) -> None:
    mode = str(cfg.model.get("mot_compile_mode", "off"))
    if mode == "off":
        return
    root = (
        Path(
            str(cfg.paths.get("compile_cache", Path(cfg.paths.cache) / "torch_compile"))
        )
        .expanduser()
        .resolve()
    )
    # A shared filesystem can expose a freshly renamed extension to one rank
    # before dlopen can see it on another. Give every worker its own artifacts.
    host = f"host_{socket.gethostname()}"
    rank = f"rank_{os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0'))}"
    for name, subdir in (("TORCHINDUCTOR_CACHE_DIR", "inductor"), ("TRITON_CACHE_DIR", "triton")):
        base = Path(os.environ.get(name, str(root / subdir))).expanduser().resolve()
        # Keep repeated setup idempotent, including inherited worker paths.
        if base.name.startswith("rank_") and base.parent.name.startswith("host_"):
            base = base.parent.parent
        os.environ[name] = str(base / host / rank)
    os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
    os.environ.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "1")
    for name in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
        Path(os.environ[name]).mkdir(parents=True, exist_ok=True)
