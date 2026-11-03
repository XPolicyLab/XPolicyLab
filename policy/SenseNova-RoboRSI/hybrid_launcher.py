"""Launch the original PolicyLab CLI segments with isolated runtime credentials."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from roborsi.planner.codex_cli import CodexCliPlanner, _terminate_process
from robots.robodojo.agents.icl_pi05.bwrap_transport import BubblewrapArgumentTransport


class HybridCancelled(RuntimeError):
    """The owning Model closed this episode."""


def _sandbox_builder(reference):
    """Extract the actual adaptive launcher without executing its module/main."""
    tree = ast.parse((reference.source_dir / "local_runtime.py").read_text())
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "sandbox_command"
    ]
    if len(functions) != 1:
        raise ValueError("Original adaptive sandbox structure changed")

    class KeyName(ast.NodeTransformer):
        def visit_Constant(self, node):
            if node.value == "ROBOPROBE_MODEL_KEY":
                return ast.copy_location(ast.Constant(value="CODEX_API_KEY"), node)
            return node

    function = KeyName().visit(functions[0])
    return compile(
        ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
        "<original-adaptive-sandbox>",
        "exec",
    )


class HybridCodexCliPlanner(CodexCliPlanner):
    """A CodexCliPlanner launch hook for the original run_segment state machine.

    The inherited solve loop is intentionally unused: it does not implement the
    original controller's resume and live_context_recovery behavior.
    """

    def __init__(
        self, *, reference, bubblewrap: Path, bubblewrap_sha256: str, **kwargs
    ):
        super().__init__(**kwargs)
        self.reference = reference
        self.bubblewrap = Path(bubblewrap).resolve()
        self.bubblewrap_sha256 = bubblewrap_sha256
        self._sandbox_code = _sandbox_builder(reference)
        self._cancelled = threading.Event()
        self._process_lock = threading.Lock()
        self._process = None
        self._spawn = subprocess.Popen
        if self.config.tools != "external" or self.config.cli_args:
            raise ValueError(
                "Original hybrid CLI requires external tools and original CLI flags"
            )
        if self.config.expected_version != "codex-cli 0.160.1":
            raise ValueError("Original hybrid requires codex-cli 0.160.1")
        if (
            self.config.expected_sha256
            != "f34a4d2301892ae96c90097786bfe5dc269f187b6f69faf42a7b357b8c081e35"
        ):
            raise ValueError(
                "Original hybrid requires the reference native CLI SHA-256"
            )
        if "CODEX_API_KEY" not in self.config.env_vars:
            raise ValueError(
                "agent.codex_cli.env_vars must explicitly forward CODEX_API_KEY"
            )

    def _sandbox(self, cfg, run, private_home, command, child_env):
        namespace = {
            "Path": Path,
            "os": SimpleNamespace(environ=child_env),
            "BWRAP": self.bubblewrap,
            "codex": self.reference.runner.codex,
        }
        exec(self._sandbox_code, namespace)
        args = namespace["sandbox_command"](cfg, run, private_home, command)
        if args[0] != str(self.bubblewrap):
            raise ValueError("Original sandbox executable changed")
        boundary = args.index("--")
        # /tmp is a fresh sandbox mount. Restore an explicitly selected Python
        # base after it when an evaluator installs its environment under /tmp.
        base = Path(cfg["agent_python"]).resolve().parent.parent
        triple = ["--ro-bind", str(base), str(base)]
        indices = [i for i in range(boundary - 2) if args[i : i + 3] == triple]
        if str(base).startswith("/tmp/"):
            if len(indices) != 1:
                raise ValueError("Original sandbox Python base bind changed")
            index = indices[0]
            del args[index : index + 3]
            boundary -= 3
            args[boundary:boundary] = triple
            boundary += 3
        names = set(self.config.env_vars)
        names.update(
            name
            for name in self.config.env
            if name not in {"PATH", "HOME", "CODEX_HOME", "LANG"}
        )
        existing = {args[i + 1] for i in range(boundary - 2) if args[i] == "--setenv"}
        for name in sorted(names - existing):
            if name in child_env:
                args[boundary:boundary] = ["--setenv", name, child_env[name]]
                boundary += 3
        return args

    def launch_segment(self, cfg, run, private_home, segment, prompt, session=None):
        """Return the real owned Popen to the unmodified original segment loop."""
        if self._cancelled.is_set():
            raise HybridCancelled("Hybrid episode was cancelled")
        child_env = self.config.subprocess_env()
        if not child_env.get("CODEX_API_KEY", "").strip():
            raise ValueError(
                "CODEX_API_KEY is required; personal Codex login is unavailable"
            )
        identity = self.preflight()
        if identity["native_binary"] is None:
            raise ValueError("Hybrid CLI must be the pinned native ELF executable")
        private_home = Path(private_home)
        private_home.mkdir(parents=True, exist_ok=True, mode=0o700)
        if (private_home / "auth.json").exists() or (
            private_home / "auth.json"
        ).is_symlink():
            raise ValueError("Hybrid private Codex home must not contain auth.json")
        cfg = {**cfg, "codex_binary": identity["executable"], "model": self.model}
        command = self.reference.runner.codex.command(cfg, session)
        if command[-1] != "-":
            raise ValueError("Original CLI must read its prompt from standard input")
        overrides = dict(self.config.config_overrides)
        required = {
            "model_provider": "runtime",
            "model_providers.runtime.wire_api": "responses",
            "model_providers.runtime.env_key": "CODEX_API_KEY",
        }
        if any(overrides.get(key) != value for key, value in required.items()):
            raise ValueError(
                "Hybrid CLI needs the explicitly configured responses provider"
            )
        if not overrides.get("model_providers.runtime.base_url"):
            raise ValueError("Hybrid CLI provider base_url is required")
        extra = []
        for key, value in sorted(overrides.items()):
            if key == "model_auto_compact_token_limit":
                if value != cfg.get("options", {}).get("codex_compact_tokens"):
                    raise ValueError("Agent and reference compaction settings differ")
                continue
            extra.extend(("-c", key + "=" + json.dumps(value, ensure_ascii=False)))
        command[-1:-1] = extra
        args = self._sandbox(cfg, Path(run), private_home, command, child_env)
        transport = BubblewrapArgumentTransport(
            self._spawn, self.bubblewrap, self.bubblewrap_sha256
        )
        (segment / "prompt.txt").write_text(prompt)
        (segment / "cli-identity.json").write_text(
            json.dumps(identity, indent=2) + "\n"
        )
        with (
            (segment / "model.jsonl").open("w") as output,
            (segment / "model.stderr").open("w") as error,
        ):
            process = transport(
                args,
                stdin=subprocess.PIPE,
                stdout=output,
                stderr=error,
                start_new_session=True,
                env={"PATH": os.defpath, "LANG": "C.UTF-8"},
            )
            with self._process_lock:
                self._process = process
                cancelled = self._cancelled.is_set()
            try:
                if cancelled:
                    raise HybridCancelled(
                        "Hybrid episode was cancelled during CLI launch"
                    )
                process.stdin.write(prompt.encode())
                process.stdin.close()
                return process
            except BaseException:
                _terminate_process(process, time.monotonic() + 5)
                raise

    def cancel(self):
        self._cancelled.set()
        with self._process_lock:
            process = self._process
        if process is not None:
            _terminate_process(process, time.monotonic() + 30)
