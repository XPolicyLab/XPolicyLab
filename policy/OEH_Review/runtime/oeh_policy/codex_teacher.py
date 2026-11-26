"""The review teacher over the Codex app-server: the user's ChatGPT/Codex quota instead of an API key.

Same contract as ``teacher.Conversation`` + ``teacher.ResponsesClient``, so ``TeacherBrain`` is unchanged: one
persistent Codex thread per episode (Codex keeps the history and compacts it), one turn per packet or correction, and
the turn's final message constrained by ``outputSchema`` to the ``robodojo_execute`` arguments (the tool's own strict
schema), which the brain then treats exactly like a tool call. The frozen prompt replaces Codex's base instructions;
the thread has no execution environment, a read-only sandbox and no approvals, and the turn schema leaves the model
nothing to answer with but the decision. Native tool items Codex reports anyway are recorded, never executed by us.

Credentials never pass through this module: the app-server reads the login from ``CODEX_HOME``. Accounts: a file
(its path is frozen, its lines are not) lists login directories; when one reports its usage limit or refuses the login,
the same thread resumes under the next (Codex keeps the thread in the rollout file named by ``thread.path``); when none
is left the refusal is fatal, reported like a spent API budget (``insufficient_quota``) or a refused key (401), which
the evaluator already counts as infrastructure.
"""
from __future__ import annotations

import json
import os
import queue
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from .review import rgb_views, tool_spec


def codex_intro(views: int = 3) -> str:
    """The text before the first packet on this transport, for ``views`` camera views."""
    return ("Act as the autonomous policy agent for this single simulation rollout. This message is the first "
            "observation packet. In this session a robodojo_execute call is written as your final message: exactly "
            "the JSON arguments of one robodojo_execute call (this turn's output schema), with no tool use and no "
            "other text. The result of the call arrives as the next message. The user authorizes sending this "
            f"episode's {rgb_views(views)}, end-effector state, the OpenWAM proposal, the task text and same-episode "
            "history to the model for online decisions.")


CODEX_FIRST_INTRO = codex_intro()
# app-server codexErrorInfo codes, by what may help: another account, a later retry, or nothing.
# A login whose plan does not serve the frozen model (seen 2026-10-20: a prolite login answered gpt-6-astra with 400
# "The '<model>' model is not supported when using Codex with a ChatGPT account") hands over like a login at its
# limit: the code is ours (error_code derives it from the message), not Codex's.
MODEL_NOT_ON_PLAN = "modelNotSupportedOnPlan"
MODEL_NOT_ON_PLAN_TEXT = "model is not supported when using Codex with a ChatGPT account"
ACCOUNT_CODES = {"usageLimitExceeded", "unauthorized", MODEL_NOT_ON_PLAN}
RETRY_CODES = {"rateLimitExceeded", "serverOverloaded", "internalServerError", "other", "httpConnectionFailed",
               "responseStreamConnectionFailed", "responseStreamDisconnected", "responseTooManyFailedAttempts"}
# Everything a teacher does not need; the probes of 2026-09-29 ran with these off (docs/openwam-gpt-hybrid).
FEATURES_OFF = ("apps", "browser_use", "browser_use_external", "computer_use", "image_generation", "plugins",
                "remote_plugin", "goals", "skill_search", "tool_suggest", "in_app_browser", "hooks", "multi_agent",
                "shell_tool", "unified_exec", "sleep_tool", "memories")
SCRUBBED_ENV = ("OPENAI_API_KEY", "CODEX_API_KEY", "OPENAI_BASE_URL", "OPENAI_ORG_ID", "OPENAI_PROJECT_ID")
# An app-server start may refresh a login or a model cache before it answers; a start that fails is tried again.
LAUNCH_ATTEMPTS = 3
LAUNCH_TIMEOUT_S = 120
LAUNCH_BACKOFF_S = (10, 30)
# The brain channel's way onto a prestarted app-server (Rpc.hand_over): the lane in the kernel runs this as its first
# "app-server" and talks JSON-RPC to it over stdio; it copies bytes both ways between its stdio and the socket the
# policy server serves the prestarted process's stdio on, and exits when either side closes.
RELAY_SOURCE = r'''
import os, socket, sys, threading
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.connect(sys.argv[1])
def up():
    try:
        while True:
            data = os.read(0, 65536)
            if not data:
                break
            s.sendall(data)
    except OSError:
        pass
    try:
        s.shutdown(socket.SHUT_WR)
    except OSError:
        pass
threading.Thread(target=up, daemon=True).start()
out = sys.stdout.buffer
try:
    while True:
        data = s.recv(65536)
        if not data:
            break
        out.write(data)
        out.flush()
except OSError:
    pass
'''


def read_accounts(path: str) -> list[str]:
    """Login directories, one absolute path per line (``#`` comments). The file holds paths, never credentials."""
    homes = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            if not Path(line).is_absolute():
                raise ValueError(f"codex account list {path}: {line!r} is not an absolute path")
            homes.append(line)
    if not homes:
        raise ValueError(f"codex account list {path} names no login directory")
    return homes


def error_code(error) -> str | None:
    """The camel-case code of a TurnError's codexErrorInfo (a string, or an object keyed by the code); a plan that does
    not serve the model (reported as code `other` with the message MODEL_NOT_ON_PLAN_TEXT) gets MODEL_NOT_ON_PLAN."""
    info = (error or {}).get("codexErrorInfo") if isinstance(error, dict) else None
    if isinstance(error, dict) and MODEL_NOT_ON_PLAN_TEXT in str(error.get("message") or ""):
        return MODEL_NOT_ON_PLAN
    if isinstance(info, str):
        return info
    if isinstance(info, dict) and info:
        return next(iter(info))
    return None


def refusal(code: str | None) -> dict:
    """A Codex account refusal in the provider-error vocabulary the runtime and the evaluator already use."""
    if code == "unauthorized":
        return {"status": 401, "type": "authentication_error", "code": "codex_unauthorized"}
    if code == MODEL_NOT_ON_PLAN:
        return {"status": 400, "type": "invalid_request_error", "code": "codex_model_not_on_plan"}
    return {"status": 429, "type": "insufficient_quota", "code": "codex_usage_limit_exceeded"}


class Rpc:
    """JSON-RPC 2.0 over the app-server's stdio: responses by id, everything else (notifications and server requests)
    in arrival order. stderr goes to a local file; the client copies it into the attempt at close (logging only)."""

    def __init__(self, argv: list[str], env: dict, stderr_path: Path):
        self._stderr = open(stderr_path, "w", encoding="utf-8")
        self.process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
                                        env=env, text=True, bufsize=1)
        self.incoming: queue.Queue = queue.Queue()
        self._pending: dict[int, queue.Queue] = {}
        self._lock = threading.Lock()
        self._next = 1
        self._out: queue.Queue | None = None  # hand_over: the app-server's lines go to the socket's client
        self._listener: socket.socket | None = None
        self._closed = False
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.process.stdout:
            if self._out is not None:
                self._out.put(line)
                continue
            try:
                message = json.loads(line)
            except ValueError:
                continue
            if "id" in message and "method" not in message:
                box = self._pending.pop(message["id"], None)
                if box is not None:
                    box.put(message)
                continue
            self.incoming.put(message)
        if self._out is not None:
            self._out.put(None)
        self.incoming.put({"method": "__closed__"})

    def hand_over(self, path: str):
        """Serve this (initialized, idle) app-server's stdio on a Unix socket at ``path`` to one client, which then
        owns the conversation; this object only relays from here on. Every line the app-server writes from now on
        goes to that client (held until it connects), every byte the client writes goes to the app-server, and the
        app-server stops when the client goes away."""
        self._out = queue.Queue()
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(path)
        listener.listen(1)
        self._listener = listener
        threading.Thread(target=self._serve, args=(listener,), name="oeh-codex-relay", daemon=True).start()

    def _serve(self, listener: socket.socket):
        try:
            connection, _ = listener.accept()
        except OSError:
            return  # closed before anyone connected
        finally:
            listener.close()

        def down():
            while True:
                line = self._out.get()
                if line is None:
                    break
                try:
                    connection.sendall(line.encode("utf-8"))
                except OSError:
                    break
            try:
                connection.shutdown(socket.SHUT_WR)
            except OSError:
                pass
        threading.Thread(target=down, name="oeh-codex-relay-down", daemon=True).start()
        buffer = b""
        try:
            while True:
                data = connection.recv(65536)
                if not data:
                    break
                buffer += data
                *lines, buffer = buffer.split(b"\n")
                for line in lines:
                    with self._lock:
                        self.process.stdin.write(line.decode("utf-8") + "\n")
                        self.process.stdin.flush()
        except (OSError, ValueError):
            pass
        finally:
            connection.close()
            self.close()

    def send(self, message: dict):
        with self._lock:
            try:
                self.process.stdin.write(json.dumps(message) + "\n")
                self.process.stdin.flush()
            except (BrokenPipeError, ValueError) as exc:
                raise RuntimeError(f"the app-server exited (code {self.process.poll()}): {exc}") from None

    def request(self, method: str, params: dict, timeout: float) -> dict:
        with self._lock:
            identifier, self._next = self._next, self._next + 1
        box: queue.Queue = queue.Queue(maxsize=1)
        self._pending[identifier] = box
        self.send({"jsonrpc": "2.0", "id": identifier, "method": method, "params": params})
        deadline = time.monotonic() + timeout
        while True:
            try:
                reply = box.get(timeout=min(1.0, max(0.0, deadline - time.monotonic())))
                break
            except queue.Empty:
                if self.process.poll() is not None:
                    self._pending.pop(identifier, None)
                    raise RuntimeError(f"app-server {method}: the app-server exited "
                                       f"(code {self.process.returncode})") from None
                if time.monotonic() >= deadline:
                    self._pending.pop(identifier, None)
                    raise TimeoutError(f"app-server {method}: no reply in {timeout:g} s") from None
        if "error" in reply:
            raise RuntimeError(f"app-server {method}: {reply['error']}")
        return reply.get("result") or {}

    def refuse(self, message: dict):
        self.send({"jsonrpc": "2.0", "id": message["id"],
                   "error": {"code": -32601, "message": "the review teacher executes no tools or approvals"}})

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
        if self._listener is not None:
            try:
                self._listener.close()
            except OSError:
                pass
        try:
            self.process.stdin.close()
        except OSError:
            pass
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10)
        self._stderr.close()


class CodexConversation:
    """What ``TeacherBrain`` keeps per episode, for the Codex transport: the inputs of the next turn (Codex holds the
    history). ``items`` is the input/output log the call artifacts record; nothing is ever pruned here."""

    def __init__(self, cfg: dict, episode_id: str, prompt: str, tool: dict | None = None, views: int = 3):
        self.cfg = cfg
        spec = tool if tool is not None else tool_spec()
        self.schema = spec["parameters"]  # the turn's output schema
        self.tool_name = spec.get("name", "robodojo_execute")  # the answer is reported as a call of this tool
        self.views = views  # camera views per packet, as the first packet's text names them
        self.items: list = []
        self.pending: list = []
        self.cache_key = f"oeh-review-{episode_id}"
        self.pending_call: str | None = None
        self.last_input_tokens = 0
        self.prunes = 0
        self.recorded = 0

    def body(self) -> dict:
        return {"model": self.cfg["model"], "input": list(self.pending), "transport": "codex_app_server",
                "reasoning": {"effort": self.cfg["reasoning_effort"]}, "prompt_cache_key": self.cache_key,
                "tools": [], "output_schema": self.schema, "tool_name": self.tool_name}

    def _add(self, content: list):
        self.pending.extend(content)
        self.items.extend(content)

    def add_packet(self, text: str, images: list[dict], intro: str | None = None):
        head = [{"type": "text", "text": codex_intro(self.views)}] if intro else []
        self._add(head + [{"type": "text", "text": text}]
                  + [{"type": "image", "url": image["image_url"]} for image in images])

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


def prestart(cfg: dict, prompt: str) -> "concurrent.futures.Future":
    """Start the app-server and run its preflight in a thread of its own, from the policy server's construction
    (oeh_policy.model.Model): the start's retries (LAUNCH_ATTEMPTS of LAUNCH_TIMEOUT_S) then overlap the OpenWAM load
    and the benchmark's own set-up instead of the first reset, which RoboDojo's client cuts at 120 s (harness audit
    TIME-1; replication round fold_clothes random l1). The future holds (client, preflight result) or the preflight's
    exception; the runtime takes it at its start. The thread's workdir is set by the runtime: the thread itself
    starts only with the first decision."""
    import concurrent.futures
    started = time.monotonic()

    future: concurrent.futures.Future = concurrent.futures.Future()
    future.set_running_or_notify_cancel()

    def run():
        try:
            client = CodexClient(cfg, workdir=Path(tempfile.gettempdir()), prompt=prompt)
        except BaseException as exc:  # noqa: BLE001 - the runtime's start reports it
            future.set_exception(exc)
            return
        try:
            checked = client.preflight()
        except BaseException as exc:  # noqa: BLE001
            client.closed.set()
            if client.server is not None:
                client.server.close()
                client.server = None
            exc.codex_client = client  # the runtime keeps its stderr in the attempt, then closes it
            future.set_exception(exc)
            return
        future.set_result((client, {**checked, "prestarted": True,
                                    "start_seconds": round(time.monotonic() - started, 1)}))

    # A daemon thread: a policy server that exits while the start is still retrying does not wait for it.
    threading.Thread(target=run, name="oeh-codex-prestart", daemon=True).start()
    return future


def discard(future) -> None:
    """Close the app-server of a prestart no runtime took (the server served no episode)."""
    def close(done):
        if not done.cancelled() and done.exception() is None:
            done.result()[0].close()
    future.add_done_callback(close)


class CodexClient:
    """One app-server and one thread per episode. ``create`` runs one turn and returns it in the Responses shape the
    brain reads (the final message as a ``robodojo_execute`` call's arguments). Blocking: the brain runs it in a worker
    thread; it never records anything itself."""

    def __init__(self, cfg: dict, *, workdir: Path, prompt: str, accounts: list[str] | None = None,
                 argv_prefix: list[str] | None = None):
        self.cfg = cfg
        self.closed = threading.Event()
        self.workdir = Path(workdir)
        self.prompt = prompt
        self.accounts = accounts if accounts is not None else read_accounts(cfg["codex_accounts_file"])
        self.argv_prefix = argv_prefix  # tests: a stand-in app-server
        self.account = 0
        self.server: Rpc | None = None
        self.thread_id: str | None = None
        self.thread_path: str | None = None
        self.total: dict | None = None
        self.rate_limits: dict | None = None
        self.switches: list[dict] = []
        self.local = Path(tempfile.mkdtemp(prefix="oeh-codex-"))  # Codex's sqlite state: local disk, not the volume
        self.launches = 0
        self.launch_log: list[dict] = []  # each app-server start: seconds to its handshake, or how it failed
        self.logs_dir: Path | None = None  # set by the runtime: the attempt dir that keeps the app-server's stderr
        self.thread_ready = False  # the running app-server has started or resumed the thread

    # -- process and thread -------------------------------------------------------------------------------------
    def _argv(self) -> list[str]:
        effort = self.cfg["reasoning_effort"]
        argv = list(self.argv_prefix or [self.cfg.get("codex_program", "codex"), "app-server"])
        overrides = [f'model="{self.cfg["model"]}"', f'model_reasoning_effort="{effort}"', "analytics.enabled=false",
                     f'sqlite_home="{self.local / "state"}"', f'log_dir="{self.local / "log"}"']
        overrides += [f"features.{name}=false" for name in FEATURES_OFF]
        for value in overrides:
            argv += ["-c", value]
        return argv

    def _launch(self):
        """Start the app-server and complete its handshake; a start that fails or does not answer is retried (formal
        A imitate l2, 2026-09-30: one `initialize: no reply in 60 s` ended an attempt at its preflight)."""
        errors: list[str] = []
        for attempt in range(LAUNCH_ATTEMPTS):
            if self.closed.is_set():
                break
            env = {k: v for k, v in os.environ.items() if k not in SCRUBBED_ENV}
            env["CODEX_HOME"] = self.accounts[self.account]
            self.launches += 1
            self.thread_ready = False
            started = time.monotonic()
            self.server = Rpc(self._argv(), env, self.local / f"app-server-{self.launches}.stderr")
            try:
                self.server.request("initialize", {"clientInfo": {"name": "oeh-review-teacher", "version": "1"},
                                                   "capabilities": {"experimentalApi": True}}, LAUNCH_TIMEOUT_S)
                self.server.send({"jsonrpc": "2.0", "method": "initialized", "params": {}})
                self.launch_log.append({"launch": self.launches, "seconds": round(time.monotonic() - started, 1)})
                return
            except (OSError, RuntimeError, TimeoutError) as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
                self.launch_log.append({"launch": self.launches, "seconds": round(time.monotonic() - started, 1),
                                        "error": errors[-1][:200]})
                self.server.close()
                self.server = None
            if attempt + 1 < LAUNCH_ATTEMPTS and self.closed.wait(LAUNCH_BACKOFF_S[min(attempt, len(LAUNCH_BACKOFF_S) - 1)]):
                break
        raise RuntimeError(f"the codex app-server did not start in {len(errors)} attempt(s): "
                           f"{errors[-1] if errors else 'client closed'}; launches {self.launch_log}")

    def _thread_params(self) -> dict:
        return {"model": self.cfg["model"], "cwd": str(self.workdir), "approvalPolicy": "never",
                "sandbox": "read-only", "baseInstructions": self.prompt,
                "config": {"model_reasoning_effort": self.cfg["reasoning_effort"]}}

    def _open_thread(self):
        if self.thread_id is None:
            response = self.server.request("thread/start", {**self._thread_params(), "environments": [],
                                                            "ephemeral": False}, 120)
        else:  # a new account continues the same thread from its rollout file
            response = self.server.request("thread/resume", {**self._thread_params(), "threadId": self.thread_id,
                                                             "path": self.thread_path}, 120)
        model, effort = response.get("model"), response.get("reasoningEffort")
        if model != self.cfg["model"] or effort != self.cfg["reasoning_effort"]:
            raise RuntimeError(f"app-server runs {model}/{effort}, not the frozen "
                               f"{self.cfg['model']}/{self.cfg['reasoning_effort']}")
        thread = response.get("thread") or {}
        self.thread_id, self.thread_path = thread["id"], thread.get("path") or self.thread_path

    def _ensure(self):
        if self.server is None or self.server.process.poll() is not None:
            self._launch()
        if not self.thread_ready:
            self._open_thread()
            self.thread_ready = True

    def _switch_account(self, code: str | None) -> bool:
        """Move to the next login; False when none is left."""
        if self.account + 1 >= len(self.accounts):
            return False
        self.switches.append({"from_account": self.account, "to_account": self.account + 1, "code": code,
                              "at": time.time()})
        if self.server is not None:
            self.server.close()
            self.server = None
        self.account += 1
        return True

    # -- the brain's interface ----------------------------------------------------------------------------------
    def preflight(self) -> dict:
        """The login runs and serves the frozen model; an account at its usage limit hands over to the next. When every
        account is at its limit, the first one whose reading shows usable credits (the user's rule: plan windows first,
        then the Pro account's credits) serves the episode on credits."""
        from .teacher import TeacherFatal, TeacherUnavailable
        credit_account, on_credits = None, False
        while True:
            try:
                self._launch()
                started = time.monotonic()
                models = self.server.request("model/list", {}, 60)
                limits = self.server.request("account/rateLimits/read", {}, 60)
                requests_seconds = round(time.monotonic() - started, 1)
            except (OSError, RuntimeError, TimeoutError) as exc:
                raise TeacherUnavailable(f"codex app-server preflight failed: {exc}") from None
            names = [m.get("id") or m.get("model") for m in (models.get("data") or [])]
            if self.cfg["model"] not in names:
                raise TeacherFatal(f"codex does not serve {self.cfg['model']}", [
                    {"status": 404, "type": "invalid_request_error", "code": "model_not_found"}])
            snapshot = limits.get("rateLimits") or limits
            self.rate_limits = snapshot
            reached = bool(snapshot.get("rateLimitReachedType"))
            if not reached or (on_credits and has_credits(snapshot)):
                result = {"model": self.cfg["model"], "transport": "codex_app_server", "account": self.account,
                          "accounts": len(self.accounts), "rate_limits": limit_summary(snapshot),
                          "launches": list(self.launch_log), "requests_seconds": requests_seconds}
                if reached:
                    result["on_credits"] = True
                return result
            if credit_account is None and has_credits(snapshot):
                credit_account = self.account
            if self._switch_account("usageLimitExceeded"):
                continue
            if credit_account is None or on_credits:
                raise TeacherFatal("every codex account is at its usage limit", [refusal("usageLimitExceeded")])
            on_credits = True
            if credit_account != self.account:  # back to the login with credits
                self.switches.append({"from_account": self.account, "to_account": credit_account, "code": "credits",
                                      "at": time.time()})
                if self.server is not None:
                    self.server.close()
                    self.server = None
                self.account = credit_account
                continue
            result = {"model": self.cfg["model"], "transport": "codex_app_server", "account": self.account,
                      "accounts": len(self.accounts), "rate_limits": limit_summary(snapshot),
                      "launches": list(self.launch_log), "requests_seconds": requests_seconds, "on_credits": True}
            return result

    def reset_context(self):
        """Begin a new server-side thread at an explicit semantic checkpoint.

        The caller supplies the current observation and deterministic task state;
        old thread artifacts remain available for audit. Never reset during a turn.
        """
        self.thread_id = self.thread_path = None
        self.thread_ready = False
        self.total = {}

    def create(self, body: dict) -> tuple[dict, list[dict]]:
        from .teacher import TeacherFatal, TeacherUnavailable
        attempts = self.cfg.get("max_attempts", 4)
        delays = self.cfg.get("retry_delays_s") or [5]
        errors: list[dict] = []
        tries = 0
        while tries < attempts:
            if self.closed.is_set():
                raise TeacherUnavailable("teacher client closed", errors)
            started = time.monotonic()
            try:
                self._ensure()
                turn, final, natives, usage = self._turn(body)
            except (OSError, RuntimeError, TimeoutError) as exc:
                code, turn = None, None
                error = {"status": None, "type": type(exc).__name__, "message": str(exc)[:400]}
            else:
                status = turn.get("status")
                if status == "completed":
                    output = []
                    if final:
                        # The final message is the arguments of the conversation's own tool (its outputSchema);
                        # a non-Review session (Q2 autorevision) names a different tool (round-2 smoke, 2026-10-21).
                        output.append({"type": "function_call", "call_id": turn["id"],
                                       "name": body.get("tool_name", "robodojo_execute"), "arguments": final})
                    return ({"id": turn["id"], "status": "completed", "output": output, "usage": usage,
                             "codex": {"account": self.account, "thread_id": self.thread_id,
                                       "duration_ms": turn.get("durationMs"), "native_items": natives,
                                       "compaction_telemetry_enabled":True,
                                       "rate_limits": limit_summary(self.rate_limits), "switches": self.switches}},
                            errors)
                code = error_code(turn.get("error"))
                error = {"status": None, "type": "codex_turn_" + str(status), "code": code,
                         "message": str((turn.get("error") or {}).get("message") or "")[:400]}
            tries += 1
            retry = code not in ACCOUNT_CODES and (code in RETRY_CODES or code is None)
            errors.append({**error, "attempt": tries, "seconds": round(time.monotonic() - started, 3),
                           "account": self.account, "retry": retry and tries < attempts, "fatal": False})
            if code in ACCOUNT_CODES:
                if self._switch_account(code):
                    tries -= 1  # a fresh account gets the full retry budget for this packet
                    continue
                errors[-1]["fatal"] = True
                raise TeacherFatal(f"every codex account refused: {code}", errors[:-1] + [
                    {**errors[-1], **refusal(code)}])
            if not retry or tries >= attempts:
                break
            if self.closed.wait(delays[min(tries - 1, len(delays) - 1)]):
                break
        raise TeacherUnavailable(f"teacher unavailable after {len(errors)} attempt(s): {errors[-1] if errors else ''}",
                                 errors)

    def _turn(self, body: dict):
        before = dict(self.total or {})
        before_turn=getattr(self,"before_model_turn",None)
        if before_turn is not None:before_turn()
        result = self.server.request("turn/start", {"threadId": self.thread_id, "input": body["input"],
                                                    "outputSchema": body["output_schema"],
                                                    "effort": self.cfg["reasoning_effort"]}, self.cfg.get("start_timeout_s", 60))
        turn_id = (result.get("turn") or {})["id"]
        deadline = time.monotonic() + self.cfg["request_timeout_s"]
        final, natives, interrupted = None, [], False
        while True:
            if self.closed.is_set() and not interrupted:
                self._interrupt(turn_id)
                interrupted = True
                deadline = min(deadline, time.monotonic() + self.cfg.get("interrupt_grace_s", 30))
            remaining = deadline - time.monotonic()
            if remaining <= 0 and not interrupted:
                self._interrupt(turn_id)
                interrupted = True
                deadline = time.monotonic() + self.cfg.get("interrupt_grace_s", 30)  # bounded interrupt grace
                remaining = self.cfg.get("interrupt_grace_s", 30)
            elif remaining <= 0:
                raise TimeoutError("codex turn neither completed nor ended after an interrupt")
            try:
                message = self.server.incoming.get(timeout=min(remaining, 1.0))
            except queue.Empty:
                continue
            method, params = message.get("method"), message.get("params") or {}
            if method == "__closed__":
                raise RuntimeError("codex app-server exited during a turn")
            if "id" in message:  # a server request (approval, tool call): the teacher executes nothing
                natives.append({"request": method})
                self.server.refuse(message)
                continue
            if params.get("threadId") not in (None, self.thread_id):
                continue
            if method == "item/completed":
                item = params.get("item") or {}
                kind = item.get("type")
                if kind == "agentMessage":
                    if item.get("text"):
                        final = item["text"]
                elif kind not in ("reasoning", "userMessage"):
                    natives.append({"item": kind, "status": item.get("status")})
                    if kind == "contextCompaction":
                        natives.append({"event": "thread/compacted", "turn_id": turn_id, "source": "contextCompaction"})
            elif method == "thread/compacted":
                natives.append({"event":method,"turn_id":turn_id})
            elif method == "thread/tokenUsage/updated":
                self.total = (params.get("tokenUsage") or {}).get("total") or self.total
                observer=getattr(self,"model_usage_observer",None)
                if observer is not None:
                    from .llm_budget import LLMTokenBudgetExceeded
                    try:observer(usage_delta(before,self.total or {}))
                    except LLMTokenBudgetExceeded:
                        self._interrupt(turn_id)
                        raise
            elif method == "account/rateLimits/updated":
                self.rate_limits = {**(self.rate_limits or {}),
                                    **{k: v for k, v in (params.get("rateLimits") or {}).items() if v is not None}}
            elif method == "turn/completed" and (params.get("turn") or {}).get("id") == turn_id:
                return params["turn"], final, natives, usage_delta(before, self.total or {})

    def _interrupt(self, turn_id: str):
        try:
            self.server.request("turn/interrupt", {"threadId": self.thread_id, "turnId": turn_id}, self.cfg.get("interrupt_timeout_s", 30))
        except (OSError, RuntimeError, TimeoutError):
            pass

    def hand_over(self, path: str) -> dict:
        """Give this client's running app-server (started and preflighted, no thread yet) to the brain channel's lane
        (oeh-brain teacher/codex_app_server.rs, `adopt`): what the lane needs to carry on exactly where this client
        would, the relay it runs in place of its first app-server start included. This client keeps the process
        and its temp dir until ``close``."""
        if self.server is None or self.server.process.poll() is not None or self.thread_ready:
            raise RuntimeError("no idle app-server to hand over")
        self.server.hand_over(path)
        return {"argv": [sys.executable, "-c", RELAY_SOURCE, path], "account": self.account,
                "rate_limits": self.rate_limits, "next_id": self.server._next}

    def close(self):
        self.closed.set()
        if self.server is not None:
            self.server.close()
            self.server = None
        self.keep_logs()
        shutil.rmtree(self.local, ignore_errors=True)

    def keep_logs(self):
        """Copy the app-server's stderr (one file per start) into the attempt, for diagnosing slow or failed starts.
        Logging only: nothing reads it back."""
        if self.logs_dir is None:
            return
        try:
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            for path in sorted(self.local.glob("app-server-*.stderr")):
                shutil.copyfile(path, self.logs_dir / path.name)
        except OSError:
            pass


def usage_delta(before: dict, after: dict) -> dict:
    """One turn's tokens (the thread's cumulative total, after minus before) in the Responses usage shape."""
    def d(key):
        return max(0, int(after.get(key) or 0) - int(before.get(key) or 0))
    return {"input_tokens": d("inputTokens"), "input_tokens_details": {"cached_tokens": d("cachedInputTokens")},
            "output_tokens": d("outputTokens"), "output_tokens_details": {"reasoning_tokens": d("reasoningOutputTokens")},
            "total_tokens": d("totalTokens")}


def has_credits(snapshot: dict | None) -> bool:
    """Whether a rate-limit reading shows usable credits (account/rateLimits/read: credits.hasCredits or
    credits.unlimited), so that a login at its plan limit still serves turns."""
    credits = (snapshot or {}).get("credits")
    return isinstance(credits, dict) and bool(credits.get("hasCredits") or credits.get("unlimited"))


def limit_summary(snapshot: dict | None) -> dict | None:
    if not snapshot:
        return None
    windows = {}
    for key in ("primary", "secondary"):
        window = snapshot.get(key) or None
        if window:
            windows[key] = {"used_percent": window.get("usedPercent"), "window_minutes": window.get("windowDurationMins"),
                            "resets_at": window.get("resetsAt")}
    return {"limit_id": snapshot.get("limitId"), "plan": snapshot.get("planType"),
            "reached": snapshot.get("rateLimitReachedType"), **windows,
            "credits": credit_summary(snapshot.get("credits"))}


def credit_summary(credits) -> dict | None:
    if not isinstance(credits, dict):
        return None
    return {"has_credits": credits.get("hasCredits"), "unlimited": credits.get("unlimited"),
            "balance": credits.get("balance")}
