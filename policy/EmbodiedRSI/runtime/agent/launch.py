"""Build the agent CLI command and thin container for workspace sessions.

The session runner owns process lifecycle, the session socket, and read-only
protocol/knowledge mounts. The policy adapter owns the deployment entrypoint.
The simulator, vendor sources, task configuration and ledgers stay on the host.

``filtered`` uses a Docker bridge so the CLI can reach its model endpoint; it
does not restrict network destinations. ``none`` disables container networking.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
import tomllib
from pathlib import Path

from XPolicyLab.policy.EmbodiedRSI.runtime.config import setting

CONTAINER_WORKSPACE = Path("/workspace")
CONTAINER_HOME = Path("/home/agent")
AGENT_NETWORK = "embodiedrsi-agent"


def agent_argv(
    preset: str, prompt: str, *, model: str | None = None, effort: str | None = None
) -> list[str]:
    """The CLI invocation for a preset, inside the container.

    Model and effort are passed explicitly rather than left to the staged
    config, so the run's own record says which they were.
    """
    if preset == "codex":
        argv = ["codex", "exec", "--sandbox", "workspace-write", "--skip-git-repo-check"]
        if model:
            argv += ["-c", f'model="{model}"']
        if effort:
            argv += ["-c", f'model_reasoning_effort="{effort}"']
        return [*argv, prompt]
    raise ValueError("This release requires the codex preset")


def docker_argv(
    workspace: Path,
    home: Path,
    inner: list[str],
    *,
    image: str,
    network: str,
    cpus: int,
    memory_mb: int,
    container_name: str | None = None,
) -> list[str]:
    """Run the agent CLI in the thin workspace-only image.

    The submitted program is deliberately *not* executed here: the host's
    session server receives source snapshots through the session socket.
    """
    if cpus <= 0 or memory_mb <= 0:
        raise ValueError("Docker CPU and memory limits must be positive")
    workspace = workspace.resolve()
    home = home.resolve()
    name = container_name or f"embodiedrsi-agent-{os.getpid()}-{time.time_ns()}"
    proxy_args = []
    proxy = (setting("AGENT_HTTPS_PROXY")
             or os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY"))
    if network != "none" and proxy:
        proxy_args = [
            "--env", f"HTTPS_PROXY={proxy}",
            "--env", f"HTTP_PROXY={proxy}",
            "--env", "NO_PROXY=localhost,127.0.0.1,::1",
        ]
    # Forward only provider-declared environment keys, by name. Values are
    # inherited by Docker and never interpolated into argv or recorded logs.
    provider_config = tomllib.loads((home / ".codex/config.toml").read_text())
    provider = provider_config.get("model_provider", "openai")
    provider_env = provider_config.get("model_providers", {}).get(provider, {}).get("env_key")
    credential_args = []
    if provider_env:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", provider_env) or not os.environ.get(provider_env):
            raise ValueError("Provider env_key must name a populated environment variable")
        credential_args = ["--env", provider_env]
    return [
        "docker",
        "run",
        "--rm",
        "--init",
        "--read-only",
        "--tmpfs",
        "/tmp:mode=1777",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--workdir",
        str(CONTAINER_WORKSPACE),
        "--env",
        f"HOME={CONTAINER_HOME}",
        "--env",
        f"EAHARNESS_WORKSPACE={CONTAINER_WORKSPACE}",
        "--env",
        "LANG=C.UTF-8",
        "--env",
        "LC_ALL=C.UTF-8",
        "--env",
        f"TERM={os.environ.get('TERM', 'xterm-256color')}",
        "--env",
        f"CODEX_HOME={CONTAINER_HOME / '.codex'}",
        "--env",
        "CODEX_CI=1",
        *proxy_args,
        *credential_args,
        "--mount",
        f"type=bind,src={workspace},dst={CONTAINER_WORKSPACE}",
        "--mount",
        f"type=bind,src={home},dst={CONTAINER_HOME}",
        "--network",
        "none" if network == "none" else AGENT_NETWORK,
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "512",
        "--memory",
        f"{memory_mb}m",
        "--cpus",
        str(cpus),
        *(["--cpuset-cpus", setting("AGENT_CPUSET")]
          if setting("AGENT_CPUSET") else []),
        "--name",
        name,
        image,
        *inner,
    ]


def adapt_for_docker(argv: list[str], *, preset: str) -> list[str]:
    """Drop Codex's own sandbox when the container is already the boundary.

    Codex wraps every command it runs in bwrap. A capability-dropped container
    cannot create that user namespace, so the CLI starts, authenticates, and
    then fails on its first shell command. Granting the capability back would
    weaken the outer boundary, so use Codex's explicit externally-sandboxed
    mode instead: the container keeps its read-only root, dropped capabilities,
    PID limit, resource caps, and the single workspace mount.
    """
    adapted = list(argv)
    if preset != "codex":
        return adapted
    try:
        index = adapted.index("--sandbox")
    except ValueError as exc:
        raise ValueError("codex argv is missing its --sandbox mode") from exc
    if index + 1 >= len(adapted):
        raise ValueError("codex argv has an incomplete --sandbox option")
    del adapted[index : index + 2]
    adapted.insert(2, "--dangerously-bypass-approvals-and-sandbox")
    return adapted


# -- docker preflight ------------------------------------------------------


def _docker(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["docker", *argv], capture_output=True, text=True, check=False)


def require_image(image: str) -> None:
    if shutil.which("docker") is None:
        raise RuntimeError("docker not found; workspace sessions require Docker")
    # Some Docker clients return 0 for templated `info` even when daemon access
    # is denied, yielding empty stdout. Do not misreport that as a missing image.
    daemon = _docker(["version", "--format", "{{.Server.Version}}"])
    if daemon.returncode != 0 or not daemon.stdout.strip():
        detail = daemon.stderr.strip() or "Docker daemon unavailable"
        raise RuntimeError(f"cannot reach the Docker daemon: {detail}")
    if _docker(["image", "inspect", image]).returncode != 0:
        raise RuntimeError(
            f"agent image {image!r} is missing; prepare the thin CLI image "
            "with policy/EmbodiedRSI/runtime/docker/build.sh (see README.md)"
        )


def ensure_network(network: str = AGENT_NETWORK) -> None:
    if _docker(["network", "inspect", network]).returncode == 0:
        return
    created = _docker(["network", "create", "--driver", "bridge", network])
    if created.returncode != 0 and _docker(["network", "inspect", network]).returncode != 0:
        raise RuntimeError(f"cannot create Docker network {network!r}: {created.stderr.strip()}")


def _toml_value(value):
    if isinstance(value, dict):
        return (
            "{ "
            + ", ".join(f"{json.dumps(key)} = {_toml_value(item)}" for key, item in value.items())
            + " }"
        )
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    return json.dumps(value)


def agent_codex_home() -> Path:
    """Resolve the same provider source for agent staging and capacity proxies."""
    override = setting("CODEX_HOME")
    if not override:
        raise ValueError("Set EMBODIEDRSI_CODEX_HOME to an explicit provider/auth directory")
    source = Path(override).expanduser()
    if not (source / "config.toml").is_file():
        raise ValueError(f"Missing container Codex config: {source / 'config.toml'}")
    return source


def stage_home(root: Path) -> Path:
    """A throwaway HOME holding only what the CLI needs to start.

    The operator's real HOME is never mounted: personal plugins, MCP servers,
    memories, prior transcripts and project settings must not reach a measured
    run. What is copied is the credential and the provider routing, because
    without them the CLI cannot reach a model at all.

    ``[projects."..."]`` blocks are dropped on the way in. They are a list of
    host paths the operator trusts, and both the paths and the fact that they
    exist are things the agent has no business seeing.
    """
    home = Path(root)
    codex = home / ".codex"
    codex.mkdir(parents=True, exist_ok=True)

    # Container routing is independent of the operator's interactive Codex.
    # Never mix a dedicated provider config with the host's credentials.
    source = agent_codex_home()
    auth = source / "auth.json"
    if auth.is_file():
        shutil.copy2(auth, codex / "auth.json")
        (codex / "auth.json").chmod(0o600)

    config = source / "config.toml"
    if config.is_file():
        data = tomllib.loads(config.read_text(encoding="utf-8"))
        # Host plugins, sqlite_home, instructions and MCP servers are neither
        # credentials nor provider routing. They must not enter measured runs.
        keys = (
            "model",
            "model_provider",
            "model_providers",
            "model_reasoning_effort",
            "model_context_window",
            "model_auto_compact_token_limit",
            "service_tier",
        )
        kept = [f"{key} = {_toml_value(data[key])}" for key in keys if key in data]
        kept += ["", f'[projects."{CONTAINER_WORKSPACE}"]', 'trust_level = "trusted"', ""]
        (codex / "config.toml").write_text("\n".join(kept), encoding="utf-8")
        (codex / "config.toml").chmod(0o600)
    return home
