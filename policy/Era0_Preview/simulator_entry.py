"""Validated SDK execution with native rendering; never edits SDK files or physics."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
from types import ModuleType

# File-based import keeps this standalone simulator entry usable with Python -P.
_helper_path = Path(__file__).resolve().parent / "sdk_compat.py"
_helper_spec = importlib.util.spec_from_file_location("era0_sdk_compat", _helper_path)
_sdk_compat = importlib.util.module_from_spec(_helper_spec)
_helper_spec.loader.exec_module(_sdk_compat)


def render_config() -> dict:
    return {
        "legacy_render": False,
        "policy_image_lag_updates": 0,
        "camera_timing": "sdk_default",
        "physics": "sdk_unchanged",
        "robodojo_root": str(Path(os.environ["ROBODOJO_ROOT"]).resolve()),
        "execution": _sdk_compat.describe(Path(os.environ["ROBODOJO_ROOT"])),
    }


def record_config(path: Path, config: dict) -> None:
    if path.exists():
        if json.loads(path.read_text()) != config:
            raise ValueError(f"Evaluation render configuration differs from {path}; use a fresh output directory")
    elif path.parent.exists() and next(path.parent.glob("**/_result.json"), None):
        raise ValueError(f"Existing results have no render provenance in {path.parent}; use a fresh output directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n")


def main() -> None:
    config = render_config()
    if len(sys.argv) >= 2 and sys.argv[1] == "--describe":
        if len(sys.argv) == 4 and sys.argv[2] == "--output":
            record_config(Path(sys.argv[3]), config)
        elif len(sys.argv) != 2:
            raise ValueError("Usage: simulator_entry.py --describe [--output PATH]")
        print(json.dumps(config), flush=True)
        return
    root = Path(config["robodojo_root"])
    entry = root / "src/eval_client/main.py"
    execution = _sdk_compat.install(root)
    print(f"[eval_execution] profile={execution['execution_profile']}; IK guard={execution['ik_guard_enabled']}; SDK files unchanged", flush=True)
    print(f"[eval_render] mode={config['camera_timing']}; physics=sdk_unchanged", flush=True)
    # Keep argv[0] pointing at this wrapper: SDK os.execv self-restarts must
    # reinstall the selected execution profile. __file__ points at the SDK entry.
    module = ModuleType("__main__")
    module.__dict__.update(__file__=str(entry), __package__=None,
                           __spec__=None, __cached__=None)
    sys.modules["__main__"] = module
    exec(compile(_sdk_compat.source(root, "src/eval_client/main.py"), str(entry), "exec"), module.__dict__)


if __name__ == "__main__":
    main()
