from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
from .common import stop_process


def native_binary(configured):
    found = shutil.which(configured) or configured
    path = Path(found).expanduser().resolve()
    if not path.is_file():
        raise ValueError("Codex binary not found; install Codex or set codex_binary")
    with path.open("rb") as stream:
        if stream.read(4) == b"\x7fELF":
            return path
    # npm's executable is a JS shim. Bind its native payload without exposing the user's home.
    package = path.parent.parent
    matches = list(package.glob("node_modules/@openai/codex-*/vendor/*/bin/codex"))
    matches += list(package.glob("vendor/*/codex/codex"))
    if len(matches) != 1:
        raise ValueError("Set codex_binary to the native Linux Codex executable")
    return matches[0].resolve()


def auth_file(cfg):
    home = Path(cfg.get("codex_home") or os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))).expanduser()
    path = home / "auth.json"
    if not path.is_file():
        raise ValueError("Run codex login first using file-backed credential storage; auth.json was not found")
    return path.resolve()


def sandbox_command(cfg, run, private_home, command):
    if not shutil.which("bwrap"):
        raise ValueError("Linux bubblewrap is required: install the bwrap package")
    cmd = ["bwrap", "--die-with-parent", "--new-session", "--unshare-pid", "--unshare-ipc", "--unshare-uts"]
    for path in ("/usr", "/bin", "/sbin", "/lib", "/lib64"):
        if Path(path).exists():
            cmd += ["--ro-bind", path, path]
    for path in ("/etc/ssl", "/etc/resolv.conf", "/etc/hosts", "/etc/nsswitch.conf", "/etc/ld.so.cache"):
        if Path(path).exists():
            cmd += ["--ro-bind", path, path]
    python = Path(cfg["agent_python"])
    runtime = python.parent.parent
    # Venv symlinks may reference a uv-managed base interpreter outside /usr.
    base = python.resolve().parent.parent
    if not str(base).startswith("/usr"):
        cmd += ["--ro-bind", str(base), str(base)]
    binary = native_binary(cfg["codex_binary"])
    cmd += ["--ro-bind", str(runtime), "/runtime",
            "--ro-bind", str(binary), "/codex",
            "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
            "--dir", "/home/agent",
            "--bind", str(private_home), "/home/agent/.codex",
            "--bind", str(auth_file(cfg)), "/home/agent/.codex/auth.json",
            "--ro-bind", str(Path(cfg["training_set"]).resolve()), "/training",
            "--ro-bind", str(run / "bridge"), "/bridge",
            "--ro-bind", str(run / "tools"), "/tools",
            "--bind", str(run / "workspace"), "/workspace",
            "--clearenv", "--setenv", "HOME", "/home/agent",
            "--setenv", "CODEX_HOME", "/home/agent/.codex",
            "--setenv", "PATH", "/runtime/bin:/usr/bin:/bin",
            "--setenv", "PYTHONPATH", "/tools", "--setenv", "LANG", "C.UTF-8",
            "--setenv", "TERM", "xterm", "--chdir", "/workspace"]
    episode = cfg.get("_observation_episode")
    if episode:
        cmd += ["--dir", "/observations", "--ro-bind", str(run / "observations" / episode), "/observations/" + episode]
    else:
        cmd += ["--ro-bind", str(run / "observations"), "/observations"]
    host = binary.with_name("codex-code-mode-host")
    if host.is_file():
        cmd += ["--ro-bind", str(host), "/codex-code-mode-host"]
    # Proxy URLs are forwarded in memory only, never included in run manifests.
    for name in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "NO_PROXY", "https_proxy", "http_proxy", "all_proxy", "no_proxy"):
        if name in os.environ:
            cmd += ["--setenv", name, os.environ[name]]
    return cmd + ["--"] + command


def command(cfg, session=None):
    cmd = ["/codex", "exec"]
    if session:
        cmd += ["resume", session]
    compact_limit = cfg.get("options", {}).get("codex_compact_tokens")
    if compact_limit is not None:
        if type(compact_limit) is not int or compact_limit < 10000:
            raise ValueError("options.codex_compact_tokens must be an integer >= 10000")
        cmd += ["-c", f"model_auto_compact_token_limit={compact_limit}"]
    return cmd + ["--json", "--skip-git-repo-check", "--ignore-user-config", "--ignore-rules",
                  "--dangerously-bypass-approvals-and-sandbox", "--model", cfg["model"],
                  "-c", "model_reasoning_effort=" + json.dumps(cfg["reasoning_effort"]),
                  "-c", 'web_search="disabled"', "-"]


def launch(cfg, run, private_home, segment, prompt, session=None):
    private_home.mkdir(parents=True, exist_ok=True, mode=0o700)
    (private_home / "auth.json").touch(exist_ok=True)
    output = (segment / "model.jsonl").open("w")
    error = (segment / "model.stderr").open("w")
    (segment / "prompt.txt").write_text(prompt)
    process = None
    try:
        process = subprocess.Popen(sandbox_command(cfg, run, private_home, command(cfg, session)),
                                   stdin=subprocess.PIPE, stdout=output, stderr=error, start_new_session=True)
        process.stdin.write(prompt.encode())
        process.stdin.close()
        return process
    except BaseException:
        stop_process(process)
        raise
    finally:
        output.close()
        error.close()


def events_summary(path):
    session, usage = None, []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("type") == "thread.started":
            session = row.get("thread_id")
        if row.get("usage"):
            usage.append(row["usage"])
    return session, usage
