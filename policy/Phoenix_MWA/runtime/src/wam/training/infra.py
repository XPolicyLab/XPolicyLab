"""Optional compiler setup shared by training and offline preparation."""

from __future__ import annotations

import os
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
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(root / "inductor"))
    os.environ.setdefault("TRITON_CACHE_DIR", str(root / "triton"))
    os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
    os.environ.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "1")
    for name in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
        Path(os.environ[name]).mkdir(parents=True, exist_ok=True)
