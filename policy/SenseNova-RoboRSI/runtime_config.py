"""Resolve the immutable RoboDojo Agent and its isolated CLI environment."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

POLICY_DIR = Path(__file__).resolve().parent
MODEL = "gpt-6-astra/azure_L/qwb"
BASE_URL = "https://tokenhub.sensetime.com/v1"
BASE_URL_ENV = "ROBORSI_CODEX_BASE_URL"
TRAJECTORY_ID_ENV = "ROBORSI_TRAJECTORY_ID"


def resolve_base_url() -> str:
    """Resolve the public endpoint before putting it into recorded CLI settings."""
    value = os.environ.get(BASE_URL_ENV, BASE_URL).strip()
    message = (
        f"{BASE_URL_ENV} must be a nonempty HTTP(S) base URL "
        "without embedded credentials, query parameters or fragments"
    )
    try:
        parsed = urlsplit(value)
        valid = (
            parsed.scheme in {"http", "https"}
            and parsed.hostname
            and not parsed.username
            and not parsed.password
            and not parsed.query
            and not parsed.fragment
            and not any(character.isspace() for character in value)
        )
        parsed.port  # Reject malformed ports without echoing the original URL.
    except ValueError:
        raise ValueError(message) from None
    if not valid:
        raise ValueError(message)
    return value


def submission() -> dict[str, Any]:
    """Read nonsecret release pins shipped with this adapter."""
    return json.loads((POLICY_DIR / "submission.json").read_text())


def load_agent_config(
    model_cfg: dict[str, Any],
    *,
    require_key: bool = True,
    episode_dir: Path | None = None,
    trajectory_id: str | None = None,
):
    """Load the pinned preset; credentials never become configuration values."""
    from roborsi.agents.config import AgentConfig

    if require_key and not os.environ.get("CODEX_API_KEY", "").strip():
        raise ValueError("CODEX_API_KEY is required; set it in the runtime environment")
    base_url = resolve_base_url()
    pins = submission()
    cli = pins["cli"]
    if not cli.get("version") or not cli.get("expected_version"):
        raise ValueError("RoboDojo Codex CLI version is missing from submission.json")
    label = model_cfg.get("ckpt_name") or pins["agent_label"]
    if label != pins["agent_label"]:
        raise ValueError(f"ckpt_name must be the Agent label {pins['agent_label']!r}")

    runtime = POLICY_DIR / "runtime"
    state_root = runtime
    state_override = os.environ.get("ROBORSI_RUNTIME_STATE_DIR", "")
    if state_override and episode_dir is None:
        state_root = Path(state_override).expanduser()
        if not state_root.is_absolute():
            raise ValueError("ROBORSI_RUNTIME_STATE_DIR must be an absolute path")
    state_dir = episode_dir if episode_dir is not None else state_root / "preflight"
    codex_home = state_dir / "codex-home"
    workspace = state_dir / "workspace"
    for directory in (codex_home, workspace):
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    # A private CODEX_HOME prevents fallback to the author's login state.
    if (codex_home / "auth.json").exists():
        raise ValueError("Remove auth.json from the adapter's private Codex home")
    environment = {
        "CODEX_HOME": str(codex_home),
        "PATH": os.pathsep.join(
            (str(runtime / "codex/bin"), str(Path(sys.executable).parent), os.defpath)
        ),
        "LANG": "C.UTF-8",
    }
    if trajectory_id is not None:
        if not isinstance(trajectory_id, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", trajectory_id
        ):
            raise ValueError("trajectory_id must be 1-128 safe ASCII characters")
        # Use the episode's identity without mutating the process environment:
        # several Model instances may own separate CLI processes concurrently.
        environment[TRAJECTORY_ID_ENV] = trajectory_id
    # Preserve the selected executor's network proxy and CA trust. Values are
    # resolved only by CodexCliConfig.subprocess_env, never put in an Agent record.
    forwarded = ["CODEX_API_KEY"]
    forwarded.extend(
        name
        for name in (
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "NO_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
            "no_proxy",
            "SSL_CERT_FILE",
            "SSL_CERT_DIR",
            "REQUESTS_CA_BUNDLE",
        )
        if name in os.environ
    )
    overrides = {
        "agent.codex_cli.executable": str(runtime / "codex/bin/codex"),
        "agent.codex_cli.expected_version": cli["expected_version"],
        "agent.codex_cli.expected_sha256": cli["sha256"],
        "agent.codex_cli.env": environment,
        "agent.codex_cli.env_vars": forwarded,
        "agent.working_dir": str(workspace),
    }
    if BASE_URL_ENV in os.environ:
        # This is a literal dotted TOML key inside config_overrides. A JSON
        # pointer preserves that key instead of creating nested dictionaries.
        overrides[
            "/agent/codex_cli/config_overrides/model_providers.tokenhub.base_url"
        ] = base_url
    if trajectory_id is not None:
        overrides[
            "/agent/codex_cli/config_overrides/"
            "model_providers.tokenhub.env_http_headers.X-Trajectory-Id"
        ] = TRAJECTORY_ID_ENV
    config = AgentConfig.from_file(
        POLICY_DIR / "agent.yaml",
        overrides=[f"{key}={json.dumps(value)}" for key, value in overrides.items()],
    )
    options = config.document["agent"]
    provider = options["codex_cli"]["config_overrides"]
    expected = {
        "model_provider": "tokenhub",
        "model_providers.tokenhub.base_url": base_url,
        "model_providers.tokenhub.wire_api": "responses",
        "model_providers.tokenhub.env_key": "CODEX_API_KEY",
    }
    if options["planner"] != "codex_cli" or options["model"] != MODEL:
        raise ValueError("The submitted RoboDojo planner/model has changed")
    if any(provider.get(key) != value for key, value in expected.items()):
        raise ValueError("The submitted Codex responses provider has changed")
    return config
