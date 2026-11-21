"""Fail before the episode when the configured slow brain cannot run (reviewer m5).

For the ``codex_exec`` wire the login directory named by the provider's ``env`` must exist, the provider's
``program`` must be a Codex on PATH that reports a version (recorded, not pinned: the user's decision of 2026-10-19
replaced the exact-version gate with checks of the login, the needed interface and the call itself), its
``login status`` under that login directory must succeed, and its ``exec`` interface must take ``--json``. The
login directory itself is never opened here; Codex reads it. The call itself is checked once per deploy by the ops
smoke (experiments/openwam-gpt-hybrid/codex_smoke.py), not per attempt.

For the ``claude_code_exec`` wire the variable named by the provider's ``env`` must hold the subscription token
(``claude setup-token``) and the provider's ``program`` must be Claude Code on PATH. The token is never read here
beyond checking that it is set.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess

# The version the eval/stable campaign ran (plan §4; ops/pod/pins.env PIN_CODEX): recorded for comparison in the
# preflight result, no longer required [decision: user, chat 2026-10-19].
CODEX_VERSION = "0.155.1"
LOGIN_STATUS_TIMEOUT_S = 30


def _providers(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except ValueError:
        import yaml  # providers/default.yaml; PyYAML is an OpenWAM dependency of the policy env
        data = yaml.safe_load(text)
    providers = (data or {}).get("providers")
    return providers if isinstance(providers, dict) else {}


def check(brain: dict) -> dict:
    """Raise RuntimeError naming what is missing; otherwise return what was checked."""
    provider = brain["provider"]
    declared = _providers(Path(brain["providers"])).get(provider)
    if not isinstance(declared, dict):
        raise RuntimeError(f"brain provider {provider!r} is not declared in {brain['providers']}")
    if declared.get("api_mode") == "claude_code_exec":
        return _claude_code(provider, declared)
    if declared.get("api_mode") != "codex_exec":
        return {"provider": provider, "api_mode": declared.get("api_mode")}
    variable = declared.get("env") or ""
    home = os.environ.get(variable)
    if not home or not Path(home).is_dir():
        raise RuntimeError(f"{variable or 'the codex provider env'} must name an existing Codex login directory")
    program = declared.get("program") or "codex"
    path = shutil.which(program)
    if path is None:
        raise RuntimeError(f"codex program {program!r} is not on PATH")
    try:
        version = subprocess.run([path, "--version"], stdin=subprocess.DEVNULL, capture_output=True,
                                 text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"{program} --version failed: {exc}") from None
    if not re.search(r"\d+\.\d+", version):
        raise RuntimeError(f"{program} --version reports no version: {version!r}")
    env = {**os.environ, "CODEX_HOME": home}
    try:
        status = subprocess.run([path, "login", "status"], stdin=subprocess.DEVNULL, capture_output=True, text=True,
                                timeout=LOGIN_STATUS_TIMEOUT_S, env=env)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"{program} login status failed under {variable}: {exc}") from None
    if status.returncode != 0:
        raise RuntimeError(f"{program} login status reports no usable login under {variable} "
                           f"(rc {status.returncode}): {(status.stdout + status.stderr).strip()[:200]}")
    try:
        usage = subprocess.run([path, "exec", "--help"], stdin=subprocess.DEVNULL, capture_output=True, text=True,
                               timeout=LOGIN_STATUS_TIMEOUT_S, env=env)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"{program} exec --help failed: {exc}") from None
    if "--json" not in usage.stdout + usage.stderr:
        raise RuntimeError(f"{program} exec takes no --json: the brain lane reads its usage from the JSON event stream")
    return {"provider": provider, "api_mode": "codex_exec", "program": path, "version": version,
            "version_pinned_elsewhere": CODEX_VERSION, "login_status": (status.stdout.strip() or "ok")[:200],
            "login_env": variable}


def _claude_code(provider: str, declared: dict) -> dict:
    variable = declared.get("env") or ""
    if not os.environ.get(variable):
        raise RuntimeError(f"{variable or 'the claude provider env'} must hold a Claude subscription token "
                           "(run `claude setup-token`)")
    program = declared.get("program") or "claude"
    path = shutil.which(program)
    if path is None:
        raise RuntimeError(f"claude program {program!r} is not on PATH")
    try:
        version = subprocess.run([path, "--version"], stdin=subprocess.DEVNULL, capture_output=True,
                                 text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"{program} --version failed: {exc}") from None
    if "Claude Code" not in version:
        raise RuntimeError(f"{program} reports {version!r}, not Claude Code")
    return {"provider": provider, "api_mode": "claude_code_exec", "program": path, "version": version,
            "login_env": variable}
