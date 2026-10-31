"""The review teacher over Claude Code logins: the user's Claude subscription instead of an API key.

The Claude counterpart of ``codex_teacher`` (same contract as ``teacher.Conversation`` + ``teacher.ResponsesClient``, so
``TeacherBrain`` is unchanged). One persistent ``claude -p`` process per episode in stream-json mode keeps the
conversation (Claude Code holds the history and compacts it, as Codex does); one user turn per packet or correction;
``--json-schema`` constrains each turn's answer to the ``robodojo_execute`` arguments (the tool's own strict schema),
which Claude Code returns as the result's ``structured_output`` and the brain treats exactly like a tool call. The frozen
prompt replaces Claude Code's system prompt; no built-in tools, MCP servers, settings files or slash commands are
loaded, so the model has nothing to answer with but the decision.

Credentials are never logged: Claude Code reads the login from ``CLAUDE_CONFIG_DIR``, and the child's environment is
built from a short allowlist (no API key, token or provider switch is inherited). A login directory may instead hold a
long-lived token from ``claude setup-token`` in ``oauth-token`` (mode 600); the child then gets it as
``CLAUDE_CODE_OAUTH_TOKEN``. Such a token is valid for a year and never refreshes, so pods sharing it cannot race each
other's refresh the way a shared ``auth login`` can, and no pod ever needs a login. Accounts: a file (its
path is frozen, its lines are not) lists login directories; when one reports its usage limit or refuses the login, the
same session continues under the next (its transcript is copied into the next login's ``projects`` directory and
resumed); when none is left the refusal is fatal, reported like a spent API budget or a refused key, which the
evaluator already counts as infrastructure.

Protocol checked against Claude Code 2.1.287 (2026-10-02): stream-json user lines carry Messages content (text and
base64 images); each turn ends with a ``result`` line (``subtype``, ``is_error``, ``api_error_status``,
``structured_output``, per-turn ``usage``, ``total_cost_usd``); ``rate_limit_event`` lines carry the login's windows.
"""
from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path

from .review import response_schema, rgb_views, tool_spec

API = "claude_code"


def cc_intro(views: int = 3) -> str:
    """The text before the first packet on this transport, for ``views`` camera views."""
    return ("Act as the autonomous policy agent for this single simulation rollout. This message is the first "
            "observation packet. In this session a robodojo_execute call is made with the StructuredOutput tool: its "
            "input is exactly the JSON arguments of one robodojo_execute call (this turn's output schema), with no "
            "other tool use. The result of the call arrives as the next message. The user authorizes sending this "
            f"episode's {rgb_views(views)}, end-effector state, the OpenWAM proposal, the task text and same-episode "
            "history to the model for online decisions.")


CC_FIRST_INTRO = cc_intro()
# The child's environment: only these are inherited (proxies and certificates a pod may need to reach the API).
ENV_ALLOW = ("PATH", "LANG", "LC_ALL", "TMPDIR", "TZ", "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY", "https_proxy",
             "http_proxy", "no_proxy", "SSL_CERT_FILE", "SSL_CERT_DIR", "NODE_EXTRA_CA_CERTS", "REQUESTS_CA_BUNDLE")

TOKEN_FILE = "oauth-token"  # in a login directory: a `claude setup-token` token, used instead of its auth login


def login_env(home: str) -> dict:
    """The child's credential variable for one login directory: its long-lived token if it has one, else nothing
    (Claude Code then reads the directory's own login). A token readable by group or others is refused."""
    path = Path(home, TOKEN_FILE)
    if not path.exists():
        return {}
    if path.stat().st_mode & 0o077:
        raise ValueError(f"{path} must be readable by its owner only (chmod 600)")
    token = path.read_text(encoding="utf-8").strip()
    if not token or any(c.isspace() for c in token):
        raise ValueError(f"{path} does not hold one token")
    return {"CLAUDE_CODE_OAUTH_TOKEN": token}


def read_accounts(path: str) -> list[str]:
    """Login directories (``CLAUDE_CONFIG_DIR`` values), one absolute path per line (``#`` comments)."""
    homes = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            if not Path(line).is_absolute():
                raise ValueError(f"claude account list {path}: {line!r} is not an absolute path")
            homes.append(line)
    if not homes:
        raise ValueError(f"claude account list {path} names no login directory")
    return homes


def refusal(code: str) -> dict:
    """A login refusal in the provider-error vocabulary the runtime and the evaluator already use."""
    if code == "unauthorized":
        return {"status": 401, "type": "authentication_error", "code": "claude_code_unauthorized"}
    return {"status": 429, "type": "insufficient_quota", "code": "claude_code_usage_limit_exceeded"}


def classify(result: dict, limit: dict | None, signed_in) -> str | None:
    """``usage_limit``, ``unauthorized`` or None (retry) for a failed turn, from structured fields only: the result's
    ``api_error_status``, this turn's own ``rate_limit_event`` (``limit``; None when the turn had none) and, for a
    failure without a status, a fresh ``claude auth status`` (``signed_in()``). Result text never decides: a refusal
    counts as infrastructure for the evaluator, so a model error that merely mentions a limit must stay a retry."""
    status = result.get("api_error_status")
    if status in (401, 403):
        return "unauthorized"
    if status == 429 or (limit or {}).get("status") == "rejected":
        return "usage_limit"
    if status is None and not signed_in():
        return "unauthorized"
    return None


class ClaudeCodeConversation:
    """What ``TeacherBrain`` keeps per episode for this transport: the content of the next user turn (Claude Code holds
    the history). ``items`` is the input/output log the call artifacts record; nothing is pruned here."""

    def __init__(self, cfg: dict, episode_id: str, prompt: str, tool: dict | None = None, views: int = 3):
        self.cfg = cfg
        self.schema = (tool if tool is not None else tool_spec())["parameters"]  # the turn's output schema
        self.views = views  # camera views per packet, as the first packet's text names them
        self.items: list = []
        self.pending: list = []
        self.cache_key = f"oeh-review-{episode_id}"
        self.pending_call: str | None = None
        self.last_input_tokens = 0
        self.prunes = 0
        self.recorded = 0

    def body(self) -> dict:
        return {"model": self.cfg["model"], "input": list(self.pending), "transport": API,
                "reasoning": {"effort": self.cfg["reasoning_effort"]}, "prompt_cache_key": self.cache_key,
                "tools": [], "output_schema": self.schema}

    def _add(self, content: list):
        self.pending.extend(content)
        self.items.extend(content)

    def add_packet(self, text: str, images: list[dict], intro: str | None = None):
        head = [{"type": "text", "text": cc_intro(self.views)}] if intro else []
        blocks = []
        for image in images:
            prefix, _, data = image["image_url"].partition(",")
            media = prefix[len("data:"):].split(";", 1)[0] or "image/png"
            blocks.append({"type": "image", "source": {"type": "base64", "media_type": media, "data": data}})
        self._add(head + [{"type": "text", "text": text}] + blocks)

    def add_error(self, text: str):
        self._add([{"type": "text", "text": text}])

    def add_output(self, output: list) -> dict | None:
        call = None
        for item in output:
            self.items.append(item)
            if item.get("type") == "function_call":
                call = item
        self.pending = []
        return call

    def unrecorded(self) -> list:
        items, self.recorded = self.items[self.recorded:], len(self.items)
        return items

    def prune(self) -> dict | None:
        return None


class _Process:
    """One ``claude -p`` in stream-json mode: lines in, parsed lines out (a reader thread); stderr to a local file."""

    def __init__(self, argv: list[str], env: dict, cwd: Path, stderr_path: Path):
        self._stderr = open(stderr_path, "w", encoding="utf-8")
        self.process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
                                        env=env, cwd=cwd, text=True, bufsize=1)
        self.incoming: queue.Queue = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.process.stdout:
            try:
                self.incoming.put(json.loads(line))
            except ValueError:
                continue
        self.incoming.put({"type": "__closed__"})

    def send(self, content: list):
        try:
            self.process.stdin.write(json.dumps({"type": "user", "message": {"role": "user", "content": content}})
                                     + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, ValueError) as exc:
            raise RuntimeError(f"claude exited (code {self.process.poll()}): {exc}") from None

    def close(self):
        try:
            self.process.stdin.close()
        except OSError:
            pass
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        self._stderr.close()


class ClaudeCodeClient:
    """One ``claude`` process and one session per episode. ``create`` runs one turn and returns it in the Responses
    shape the brain reads (the structured output as a ``robodojo_execute`` call's arguments). Blocking: the brain runs it
    in a worker thread; it never records anything itself."""

    def __init__(self, cfg: dict, *, workdir: Path, prompt: str, accounts: list[str] | None = None,
                 argv_prefix: list[str] | None = None, schema: dict | None = None):
        self.cfg = cfg
        self.closed = threading.Event()
        self.workdir = Path(workdir)
        self.prompt = prompt
        self.schema = schema if schema is not None else response_schema()  # --json-schema: the tool's arguments
        self.accounts = accounts if accounts is not None else read_accounts(cfg["claude_accounts_file"])
        self.argv_prefix = argv_prefix  # tests: a stand-in claude
        self.account = 0
        self.session_id = str(uuid.uuid4())
        self.started_session = False  # the session exists on disk under the current login
        self.proc: _Process | None = None
        self.local = Path(tempfile.mkdtemp(prefix="oeh-claude-"))  # HOME for the child: local disk, not the volume
        self.launches = 0
        self.rate_limit: dict | None = None  # the latest rate_limit_event (recorded with each call)
        self.turn_limit: dict | None = None  # this turn's rate_limit_event only (classification)
        self.switches: list[dict] = []
        self.model_seen: str | None = None

    def _env(self) -> dict:
        env = {k: v for k, v in os.environ.items() if k in ENV_ALLOW}
        env.update(HOME=str(self.local), CLAUDE_CONFIG_DIR=self.accounts[self.account], DISABLE_AUTOUPDATER="1",
                   CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1")
        env.update(login_env(self.accounts[self.account]))
        return env

    def _argv(self) -> list[str]:
        argv = list(self.argv_prefix or [self.cfg.get("claude_program", "claude")])
        argv += ["-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
                 "--model", self.cfg["model"], "--effort", self.cfg["reasoning_effort"],
                 "--system-prompt", self.prompt, "--json-schema", json.dumps(self.schema),
                 "--tools", "", "--strict-mcp-config", "--setting-sources", "", "--disable-slash-commands"]
        argv += ["--resume", self.session_id] if self.started_session else ["--session-id", self.session_id]
        return argv

    def _ensure(self):
        if self.proc is None or self.proc.process.poll() is not None:
            if self.proc is not None:
                self.proc.close()
            self.launches += 1
            self.proc = _Process(self._argv(), self._env(), self.workdir, self.local / f"claude-{self.launches}.stderr")

    def _stop(self):
        if self.proc is not None:
            self.proc.close()
            self.proc = None

    def _session_file(self, home: str) -> Path | None:
        found = sorted(Path(home, "projects").glob(f"*/{self.session_id}.jsonl"))
        return found[0] if found else None

    def _switch_account(self, code: str) -> bool:
        """Move to the next login, carrying the session transcript over; False when none is left."""
        if self.account + 1 >= len(self.accounts):
            return False
        self._stop()
        source = self._session_file(self.accounts[self.account])
        self.switches.append({"from_account": self.account, "to_account": self.account + 1, "code": code,
                              "at": time.time(), "session_carried": source is not None})
        self.account += 1
        if source is not None:
            target = Path(self.accounts[self.account], "projects", source.parent.name, source.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        else:
            self.started_session = False  # nothing to resume: the next login starts the session afresh
        return True

    # -- the brain's interface ----------------------------------------------------------------------------------
    def _auth_status(self) -> dict:
        """``claude auth status`` for the current login (costs no tokens)."""
        from .teacher import TeacherUnavailable
        program = list(self.argv_prefix or [self.cfg.get("claude_program", "claude")])
        try:
            done = subprocess.run(program + ["auth", "status"], env=self._env(), cwd=self.workdir,
                                  capture_output=True, text=True, timeout=60)
            return json.loads(done.stdout or "{}")
        except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
            raise TeacherUnavailable(f"claude auth status failed: {type(exc).__name__}: {exc}") from None

    def _signed_in(self) -> bool:
        try:
            return bool(self._auth_status().get("loggedIn"))
        except Exception:  # an unreadable status is no evidence of a refusal: the turn is retried
            return True

    def preflight(self) -> dict:
        """Each login is signed in. The model itself is confirmed by the first turn's init line. A signed-out login
        hands over to the next."""
        from .teacher import TeacherFatal
        for home in self.accounts:
            try:
                login_env(home)
            except (OSError, ValueError) as exc:
                raise TeacherFatal(f"claude login {home}: {exc}", [refusal("unauthorized")]) from None
        while True:
            status = self._auth_status()
            if status.get("loggedIn"):
                return {"model": self.cfg["model"], "transport": API, "account": self.account,
                        "accounts": len(self.accounts), "auth_method": status.get("authMethod"),
                        "api_provider": status.get("apiProvider")}
            if not self._switch_account("unauthorized"):
                raise TeacherFatal("no claude login is signed in", [refusal("unauthorized")])

    def create(self, body: dict) -> tuple[dict, list[dict]]:
        from .teacher import TeacherFatal, TeacherUnavailable, mask
        attempts = self.cfg.get("max_attempts", 4)
        delays = self.cfg.get("retry_delays_s") or [5]
        errors: list[dict] = []
        tries = 0
        while tries < attempts:
            if self.closed.is_set():
                raise TeacherUnavailable("teacher client closed", errors)
            started = time.monotonic()
            code = None
            try:
                self._ensure()
                result = self._turn(body)
            except (OSError, RuntimeError, TimeoutError) as exc:
                self._stop()
                error = {"status": None, "type": type(exc).__name__, "message": mask(exc)[:400]}
            else:
                self.started_session = True
                if result.get("subtype") == "success" and not result.get("is_error"):
                    return self._responses_shape(result), errors
                code = classify(result, self.turn_limit, self._signed_in)
                error = {"status": result.get("api_error_status"), "type": "claude_code_" + str(result.get("subtype")),
                         "code": code, "message": mask(result.get("result") or "")[:400]}
            tries += 1
            retry = code is None
            errors.append({**error, "attempt": tries, "seconds": round(time.monotonic() - started, 3),
                           "account": self.account, "retry": retry and tries < attempts, "fatal": False})
            if code is not None:
                if self._switch_account(code):
                    tries -= 1  # a fresh login gets the full retry budget for this packet
                    continue
                errors[-1]["fatal"] = True
                raise TeacherFatal(f"every claude login refused: {code}", errors[:-1] + [
                    {**errors[-1], **refusal(code)}])
            if tries >= attempts:
                break
            if self.closed.wait(delays[min(tries - 1, len(delays) - 1)]):
                break
        raise TeacherUnavailable(f"teacher unavailable after {len(errors)} attempt(s): {errors[-1] if errors else ''}",
                                 errors)

    def _turn(self, body: dict) -> dict:
        self.turn_limit = None
        self.proc.send(body["input"])
        deadline = time.monotonic() + self.cfg["request_timeout_s"]
        while True:
            if self.closed.is_set():
                raise RuntimeError("teacher client closed during a turn")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"claude turn did not finish in {self.cfg['request_timeout_s']:g} s")
            try:
                message = self.proc.incoming.get(timeout=min(remaining, 1.0))
            except queue.Empty:
                continue
            kind = message.get("type")
            if kind == "__closed__":
                raise RuntimeError(f"claude exited during a turn (code {self.proc.process.poll()})")
            if kind == "system" and message.get("subtype") == "init":
                self.model_seen = message.get("model")
                if self.model_seen != self.cfg["model"]:
                    raise RuntimeError(f"claude runs {self.model_seen}, not the frozen {self.cfg['model']}")
            elif kind == "rate_limit_event":
                self.turn_limit = message.get("rate_limit_info") or {}
                self.rate_limit = self.turn_limit or self.rate_limit
            elif kind == "result":
                return message

    def _responses_shape(self, result: dict) -> dict:
        usage = result.get("usage") or {}
        uncached = usage.get("input_tokens") or 0
        written = usage.get("cache_creation_input_tokens") or 0
        read = usage.get("cache_read_input_tokens") or 0
        output_tokens = usage.get("output_tokens") or 0
        structured = result.get("structured_output")
        output = []
        if isinstance(structured, dict):
            output.append({"type": "function_call", "call_id": result.get("uuid") or f"turn-{time.time_ns()}",
                           "name": "robodojo_execute", "arguments": json.dumps(structured)})
        limit = self.rate_limit or {}
        return {"id": result.get("uuid"), "status": "completed", "output": output,
                "stop": {"stop_reason": result.get("stop_reason"), "terminal_reason": result.get("terminal_reason")},
                "usage": {"input_tokens": uncached + written + read,
                          "input_tokens_details": {"cached_tokens": read, "cache_creation_tokens": written},
                          "output_tokens": output_tokens,
                          "output_tokens_details": {"reasoning_tokens": (usage.get("output_tokens_details") or {})
                                                    .get("thinking_tokens")},
                          "total_tokens": uncached + written + read + output_tokens},
                "claude_code": {"account": self.account, "session_id": result.get("session_id"),
                                "model": self.model_seen, "num_turns": result.get("num_turns"),
                                "duration_ms": result.get("duration_ms"),
                                "api_equivalent_cost_usd": result.get("total_cost_usd"),
                                "rate_limit": {"status": limit.get("status"), "type": limit.get("rateLimitType"),
                                               "resets_at": limit.get("resetsAt"),
                                               "windows": limit.get("unifiedWindows")},
                                "switches": self.switches}}

    def close(self):
        self.closed.set()
        self._stop()
        shutil.rmtree(self.local, ignore_errors=True)
