"""Launch the native EmbodiedEvo module with the image's offline material cache."""

from __future__ import annotations

import hashlib
import json
import os
import runpy
import sys
import traceback
from pathlib import Path


def configure_materials(root):
    """Alias original USD material URLs to checksum-verified identical bytes."""
    import omni.client

    root = Path(root).resolve()
    records = json.loads((root / "manifest.json").read_text())
    for record in records:
        target = (root / record["relative_path"]).resolve()
        if not target.is_relative_to(root):
            raise ValueError("Material cache path escapes its root")
        if hashlib.sha256(target.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Material cache checksum mismatch: {target.name}")
        omni.client.set_alias(record["source_url"], str(target))


def main():
    argv = sys.argv[1:]
    if len(argv) < 2 or argv[0] != "-m":
        raise ValueError("Use the image interpreter with -m MODULE and --run-dir")
    # Match the image's established import order before launching Isaac.
    import gymnasium  # noqa: F401

    if mirror := os.environ.get("EMBODIEDEVO_MDL_MIRROR"):
        import isaaclab.app

        launcher = isaaclab.app.AppLauncher

        class OfflineLauncher(launcher):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                configure_materials(mirror)

        isaaclab.app.AppLauncher = OfflineLauncher
    module, sys.argv = argv[1], [argv[1], *argv[2:]]
    try:
        runpy.run_module(module, run_name="__main__")
    except Exception as exc:
        if "--run-dir" in sys.argv:
            run = Path(sys.argv[sys.argv.index("--run-dir") + 1])
            run.mkdir(parents=True, exist_ok=True)
            (run / "simulator-error.json").write_text(json.dumps({
                "type": type(exc).__name__, "message": str(exc),
                "traceback": traceback.format_exc(),
            }, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
