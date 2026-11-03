"""Process-local SDK execution/render compatibility; never edits SDK files or physics."""
from __future__ import annotations

import importlib
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


_lag_spec = importlib.util.spec_from_file_location("era0_obs_lag", Path(__file__).parent / "obs_lag.py")
_obs_lag = importlib.util.module_from_spec(_lag_spec)
_lag_spec.loader.exec_module(_obs_lag)


def legacy_render_enabled() -> bool:
    return _obs_lag.legacy_render_setting(_obs_lag.observation_lag())


def render_config() -> dict:
    legacy = legacy_render_enabled()
    return {
        "legacy_render": legacy,
        "policy_image_lag_updates": _obs_lag.observation_lag(),
        "camera_timing": "legacy_single_render" if legacy else "sdk_default",
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


def install_legacy_render(root: Path) -> None:
    sync = importlib.import_module("env.camera_manager.capture.render_sync")
    expected = root / "env/camera_manager/capture/render_sync.py"
    if Path(sync.__file__).resolve() != expected.resolve():
        raise RuntimeError(f"Unexpected render_sync module: {sync.__file__}; expected {expected}")

    def skip_zero_delay_args(args):
        # Existing explicit zero-delay flags defeat the legacy mode; fail early.
        kit_args = getattr(args, "kit_args", "") or ""
        if any(f"--{key}=" in kit_args for key in sync.ZERO_DELAY_KIT_SETTINGS):
            raise ValueError("Legacy rendering requires Kit defaults; remove explicit zero-delay Kit overrides")

    def single_render(env, capture_manager=None, min_passes=1, max_passes=16):
        env.render()
        return 1

    sync.add_zero_delay_kit_args = skip_zero_delay_args
    sync.wait_for_latest_cameras = single_render


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
    if config["legacy_render"]:
        install_legacy_render(root)
    print(f"[eval_execution] profile={execution['execution_profile']}; IK guard={execution['ik_guard_enabled']}; SDK files unchanged", flush=True)
    print(f"[eval_render] mode={config['camera_timing']}; physics=sdk_unchanged", flush=True)
    # Keep argv[0] pointing at this wrapper: SDK os.execv self-restarts must
    # reinstall the camera overrides. __file__ still points at the SDK entry.
    module = ModuleType("__main__")
    module.__dict__.update(__file__=str(entry), __package__=None,
                           __spec__=None, __cached__=None)
    sys.modules["__main__"] = module
    exec(compile(_sdk_compat.source(root, "src/eval_client/main.py"), str(entry), "exec"), module.__dict__)


if __name__ == "__main__":
    main()
