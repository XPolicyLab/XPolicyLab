"""Isolation of the program kernel (App. A: each episode runs in an isolated
container holding the kernel and its workspace, which can reach only the model
endpoint and the policy servers).

File system: the kernel starts in new user, mount and PID namespaces (util-linux
unshare; no privileges needed). Its view of the hidden roots (by default the user's
home directory) is empty except for the Python installation and, read-only, the
harness source, so benchmark checkouts, task files, run directories and keys are
not visible. /tmp is a fresh tmpfs that disappears with the episode; the working
directory is /tmp/workspace. Host processes are not visible.

Network and processes: once the session has started, an audit hook lets sockets
connect only to the endpoints named in the kernel options (worker, models, policy
servers) and refuses to start processes or to load native code through ctypes. The
hook guards against programs reaching data and services, not against deliberate
escape from the Python interpreter.
"""

import os
from pathlib import Path
import shlex
import socket
import sys
from urllib.parse import urlsplit


# Audit events a program may not trigger once the kernel is serving.
_BLOCKED_EVENTS = ("subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.fork", "os.forkpty",
                   "os.spawn", "pty.spawn", "ctypes.dlopen", "ctypes.dlsym")
_DEFAULT_PORTS = {"http": 80, "ws": 80, "https": 443, "wss": 443}
# Environment variables the kernel keeps; API keys named in the options are added.
_ENVIRONMENT = ("PATH", "LANG", "LC_ALL", "TZ", "PYTHONPATH", "PYTHONUNBUFFERED", "PYTHONNOUSERSITE",
                "HARNESS_WORKER_TOKEN",
                "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy")


def sandbox_command(command: list[str], *, keep: list[Path], read_only: list[Path],
                    hide: list[Path]) -> list[str]:
    """Wrap `command` so it runs in the namespaces described above. `hide` are the
    roots whose content is replaced by an empty tmpfs; `keep` (read-write) and
    `read_only` paths inside them stay visible."""
    visible = [(Path(p).resolve(), False) for p in keep] + [(Path(p).resolve(), True) for p in read_only]
    roots = [Path(p).resolve() for p in hide]
    lines = ["set -e", "mount -t tmpfs tmpfs /tmp", "mkdir -p /tmp/.keep"]
    for index, (path, _ro) in enumerate(visible):
        lines += [f"mkdir -p /tmp/.keep/{index}", f"mount --rbind {shlex.quote(str(path))} /tmp/.keep/{index}"]
    for root in roots:
        lines.append(f"mount -t tmpfs tmpfs {shlex.quote(str(root))}")
        for index, (path, ro) in enumerate(visible):
            if path == root or root in path.parents:
                target = shlex.quote(str(path))
                lines += [f"mkdir -p {target}", f"mount --rbind /tmp/.keep/{index} {target}"]
                if ro:
                    lines.append(f"mount -o remount,bind,ro {target}")
    lines += ["for d in /tmp/.keep/*; do umount -l \"$d\"; done", "rm -rf /tmp/.keep",
              "mkdir -p /tmp/workspace", "cd /tmp/workspace",
              "exec " + " ".join(shlex.quote(part) for part in command)]
    return ["unshare", "--user", "--map-root-user", "--mount", "--pid", "--fork", "--mount-proc",
            "--kill-child", "--", "/bin/sh", "-c", "\n".join(lines)]


# Packages the kernel imports that may be installed outside the Python prefix
# (editable installs): the policy client of XPolicyLab.
KERNEL_PACKAGES = ("client_server",)
# Modules imported before the network guard is installed: importing them loads
# native code through ctypes, which programs may not do.
KERNEL_IMPORTS = ("client_server.ws.model_client",)


def package_dirs(names=KERNEL_PACKAGES) -> list[Path]:
    """Where these packages live on the host, so the sandbox can show them read-only
    at the same paths (an editable install's finder points there)."""
    import importlib.util

    dirs = []
    for name in names:
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            spec = None
        if spec is not None:
            dirs += [Path(location) for location in (spec.submodule_search_locations or [])]
    return dirs


def hidden_roots(paths: list[Path], extra=()) -> list[Path]:
    """The home directory and, for each path, its highest ancestor owned by this
    user (e.g. /data/<user> for a checkout in it): the user's own directories."""
    roots = {Path.home().resolve()}
    uid = os.getuid()
    for path in paths:
        owned = None
        for parent in [Path(path).resolve(), *Path(path).resolve().parents]:
            try:
                if parent != Path("/") and parent.stat().st_uid == uid:
                    owned = parent
            except OSError:
                pass
        if owned is not None:
            roots.add(owned)
    roots |= {Path(p).expanduser().resolve() for p in extra}
    # Drop roots inside other roots: hiding the outer one hides them.
    return sorted(root for root in roots if not any(other in root.parents for other in roots))


def kernel_environment(options: dict, base: dict) -> dict:
    """The kernel's environment: basic variables, HOME in the empty home, and the
    API keys its model clients need."""
    names = set(_ENVIRONMENT)
    for config in _model_configs(options):
        names.add(config.get("api_key_env") or ("OPENROUTER_API_KEY"
                                                if config.get("backend", "openrouter") == "openrouter" else ""))
        names.add(config.get("base_url_env") or "")
    env = {name: base[name] for name in names if name and name in base}
    env["HOME"] = base.get("HOME", "/tmp")
    env["TMPDIR"] = "/tmp"
    return env


def endpoints(options: dict) -> set[tuple[str, int]]:
    """(host, port) of every service named in the kernel options."""
    from ..core.http import chat_url

    urls = []

    def scan(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ("endpoint", "url", "base_url") and isinstance(item, str):
                    urls.append(item)
                else:
                    scan(item)
        elif isinstance(value, list):
            for item in value:
                scan(item)

    scan(options)
    urls += [chat_url(config.get("backend", "openrouter"), config.get("base_url"))
             for config in _model_configs(options)]
    for name in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        if os.environ.get(name):
            urls.append(os.environ[name])
    result = set()
    for url in urls:
        parts = urlsplit(url if "://" in url else "http://" + url)
        if parts.hostname:
            result.add((parts.hostname, parts.port or _DEFAULT_PORTS.get(parts.scheme, 80)))
    return result


def _model_configs(options: dict) -> list[dict]:
    return [options[name] for name in ("vlm", "pointer", "verifier") if isinstance(options.get(name), dict)]


def install_network_guard(allowed: set[tuple[str, int]]) -> None:
    """Refuse connections to anything but `allowed`, and process creation and
    ctypes, for the rest of the process's life (audit hooks cannot be removed)."""
    import importlib

    for name in KERNEL_IMPORTS:
        try:
            importlib.import_module(name)
        except ImportError:
            pass  # not installed here; skills that need it report the error
    hosts = frozenset(host for host, _port in allowed)
    addresses: dict[str, set[int]] = {}

    def resolve():
        for host, port in allowed:
            try:
                infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
            except OSError:
                continue
            for info in infos:
                addresses.setdefault(info[4][0], set()).add(port)

    def permitted(address) -> bool:
        return (isinstance(address, tuple) and len(address) >= 2
                and address[1] in addresses.get(str(address[0]), ()))

    resolve()

    def hook(event, args):
        if event == "socket.getaddrinfo":
            if str(args[0]) not in hosts and str(args[0]) not in addresses:
                raise PermissionError(f"programs may not resolve {args[0]!r}")
        elif event == "socket.connect":
            if not permitted(args[1]):
                resolve()  # addresses of the allowed names may have changed
                if not permitted(args[1]):
                    raise PermissionError(f"programs may not connect to {args[1]!r}")
        elif event.startswith(_BLOCKED_EVENTS):
            raise PermissionError(f"programs may not use {event}")

    sys.addaudithook(hook)
