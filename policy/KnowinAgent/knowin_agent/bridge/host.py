"""Episode host for externally driven agents (a coding-agent CLI per role, talking MCP).

The host owns the backend, runtime and episode state. Simulator calls must stay
on the main thread, so socket handler threads only enqueue jobs; the main thread
runs them. While Task is blocked in `interact`, the main thread keeps pumping
jobs so the Local agent's calls are served.
"""

from __future__ import annotations

import json
import os
import queue
import socketserver
import subprocess
import threading
import time
from concurrent.futures import Future
from pathlib import Path
from typing import Callable, Optional

from knowin_agent.agents.context import ContextBuilder
from knowin_agent.agents.controller import EpisodeLimits
from knowin_agent.agents.tools import ControlTransfer, ToolResult, execute, local_toolset, task_toolset
from knowin_agent.core.events import EventKind
from knowin_agent.core.recorder import RunRecorder
from knowin_agent.runtime.api import RobotRuntime, ToolOutcome
from knowin_agent.runtime.contract import InteractionContract

WHY = {"type": "string", "description": "one or two sentences: what the evidence shows and why this call"}
MCP_NOTE = """
## Tool protocol
You act on the robot only through the `knowin_agent` MCP tools (programs go through `exec`). Every tool result ends with a
fresh snapshot of the robot (status text and the current camera images); read it before the next call. Put your
rationale in each call's `why` argument. Your working directory is the episode workspace: read `skills/`,
`skills/shared/` and `lessons/` with your file tools; in practice, write task skills and lessons there yourself.
Shell commands cannot reach the robot.
"""
# Notes and documentation lookups do not count toward the decision cap: in 10-31 practice `remember` was 22-30%
# of all decisions, and five PGs stopped at 150 decisions with 16-70% of their native actions unused.
UNCAPPED_TOOLS = ("remember", "api_reference", "skill_info")
DECISION_WARNING = 15


def decision_cap(limits: EpisodeLimits, kit=None) -> int:
    """The Task decision cap: the configured one, raised to cover the execution budget (counted programs plus
    as many uncounted measure-only ones) and a share of single-call tools."""
    executions = getattr(kit, "max_executions", None) or limits.max_executions
    if not executions:
        return int(limits.max_task_decisions)
    return max(int(limits.max_task_decisions), 2 * int(executions) + 40)


class MainThreadJobs:
    def __init__(self):
        self._jobs: "queue.Queue[tuple[Callable, Future]]" = queue.Queue()

    def submit(self, fn: Callable) -> object:
        future: Future = Future()
        self._jobs.put((fn, future))
        return future.result()

    def pump(self, timeout: float = 0.2) -> int:
        done = 0
        deadline = time.time() + timeout
        while True:
            remaining = deadline - time.time()
            try:
                fn, future = self._jobs.get(timeout=max(0.0, remaining)) if done == 0 else self._jobs.get_nowait()
            except queue.Empty:
                return done
            try:
                future.set_result(fn())
            except BaseException as exc:  # noqa: BLE001 - surfaced to the caller thread
                future.set_exception(exc)
            done += 1


class EpisodeHost:
    def __init__(self, runtime: RobotRuntime, library, *, recorder: RunRecorder, limits: EpisodeLimits,
                 task_brief: str = "", launcher: Optional[Callable[[str, str], subprocess.Popen]] = None,
                 image_side: int = 448, kit=None):
        self.rt = runtime
        self.library = library
        self.recorder = recorder
        self.limits = limits
        self.launcher = launcher
        self.kit = kit
        self.jobs = MainThreadJobs()
        sections = ()
        if kit is not None:
            from knowin_agent.assembly import task_sections

            sections = task_sections(kit, inline_plan=False)
        self.context = ContextBuilder(runtime, library, task_brief=task_brief, image_side=image_side,
                                      recorder=recorder, task_sections=sections)
        if kit is not None:
            from knowin_agent.assembly import build_task_tools

            tools = build_task_tools(kit, remember=self._remember, interact=self._interact, finish=self._finish,
                                     file_tools=False)
        else:
            tools = task_toolset(runtime, library, remember=self._remember, interact=self._interact,
                                 finish=self._finish)
        self.task_tools = {t.name: t for t in tools}
        self.local_tools: dict = {}
        self.contract: Optional[InteractionContract] = None
        self.delegation: Optional[str] = None
        self.local_report: Optional[dict] = None
        self.decisions = {"task": 0, "local": 0, "task_capped": 0}
        self.decision_cap = decision_cap(limits, kit)
        self.delegations: list[dict] = []
        self.finish_report: Optional[dict] = None
        self.agent_resumes: list[dict] = []
        self.started = time.time()
        self.rt.wall_window = (self.started, float(self.limits.wall_seconds))

    def _attempt_may_restart(self) -> bool:
        from knowin_agent.assembly import attempt_may_restart

        return attempt_may_restart(self.kit)

    def _episode_done(self) -> bool:
        return self.finish_report is not None or (self.rt.executor.episode_over and not self._attempt_may_restart())

    MAX_AGENT_RESUMES = 10
    RESUME_BACKOFF_SECONDS = (60, 120, 180, 240, 300)

    def _wait_to_resume(self, error: str) -> bool:
        """Wait out the backoff for the next resume (robot jobs keep being served); False when no resume is
        left or the wall-clock budget would run out first."""
        attempt = len(self.agent_resumes)
        if attempt >= self.MAX_AGENT_RESUMES:
            return False
        delay = self.RESUME_BACKOFF_SECONDS[min(attempt, len(self.RESUME_BACKOFF_SECONDS) - 1)]
        if time.time() + delay + 60 > self.started + self.limits.wall_seconds:
            return False
        self.agent_resumes.append({"attempt": attempt + 1, "error": error[:300], "tick": self.rt.tick,
                                   "after_seconds": round(time.time() - self.started, 1), "delay": delay})
        self.recorder.record("episode", phase="agent_resume", **self.agent_resumes[-1])
        until = time.time() + delay
        while time.time() < until and not self._episode_done():
            self.jobs.pump(0.2)
        return not self._episode_done()

    # -- requests (handler threads) --------------------------------------------------------------

    def handle(self, request: dict) -> dict:
        op = request.get("op")
        role = request.get("role", "task")
        if op == "tools":
            return {"tools": self.jobs.submit(lambda: self.tool_schemas(role))}
        if op == "prompt":
            return {"prompt": self.jobs.submit(lambda: self.prompt(role))}
        if op == "call":
            return self.jobs.submit(lambda: self.call(role, request["name"], dict(request.get("arguments") or {})))
        if op == "status":
            return {"finished": self.finish_report is not None, "episode_over": self._episode_done()}
        return {"error": f"unknown op {op!r}"}

    # -- main-thread work ------------------------------------------------------------------------

    def _tools(self, role: str) -> dict:
        if role == "task":
            return self.task_tools
        if self.contract is None:
            return {}
        return self.local_tools

    def tool_schemas(self, role: str) -> list[dict]:
        out = []
        for spec in self._tools(role).values():
            schema = spec.schema()["function"]
            params = json.loads(json.dumps(schema["parameters"]))
            params["properties"]["why"] = WHY
            out.append({"name": schema["name"], "description": schema["description"], "inputSchema": params})
        return out

    def prompt(self, role: str) -> str:
        return self._agent_paths(self._prompt(role))

    def _prompt(self, role: str) -> str:
        text = self.context.system_prompt(role, self.contract) + "\n" + MCP_NOTE
        if role == "task":
            text += "\n## Instruction\n" + self.rt.ctx.instruction + "\n\nStart by looking at the scene (any tool " \
                    "result includes the current images); call `finish` when done.\n"
        else:
            text += "\nStart with `observe` or a small correction; call `return_to_task` when done.\n"
        return text

    def call(self, role: str, name: str, arguments: dict) -> dict:
        tools = self._tools(role)
        why = str(arguments.pop("why", "") or "")
        if role == "local" and self.contract is None:
            return self._reply(role, ToolResult(name, False, {"error": "no active local interaction"}))
        if role == "local" and self.decisions["local_current"] >= self.contract.max_rounds and name != "return_to_task":
            return self._reply(role, ToolResult(name, False, {"error": "round limit reached; call return_to_task"}))
        if name not in tools:
            return self._reply(role, ToolResult(name, False, {"error": f"unknown tool {name!r}"}))
        self.decisions[role] += 1
        if role == "local":
            self.decisions["local_current"] += 1
        elif name not in UNCAPPED_TOOLS:
            self.decisions["task_capped"] += 1
        tick_before = self.rt.tick
        try:
            result = execute(tools[name], arguments)
        except ControlTransfer as control:
            result = ToolResult(name, True, control.payload)
            if control.kind == "return":
                self.local_report = dict(control.payload)
        self.recorder.record("tool_calls", scope=role, delegation=self.delegation if role == "local" else None,
                             decision=self.decisions[role], tool=name, arguments=arguments, rationale=why,
                             ok=result.ok, status=result.data.get("status"), result=result.data,
                             tick_before=tick_before, tick_after=self.rt.tick, obs_id=self.rt.obs.obs_id,
                             score_host_only=self._score_after(tick_before),
                             ground_truth_host_only=self._ground_truth_after(tick_before))
        return self._reply(role, result)

    def _score_after(self, tick_before: int) -> Optional[float]:
        """The native Score after a call that moved the robot, for the operator's replay only: it is written to
        the trace, never into the reply, so a test agent still sees no score."""
        if self.rt.tick == tick_before:
            return None
        try:
            return self.rt.backend.evaluation().get("score")
        except Exception:  # noqa: BLE001 - telemetry must not break the episode
            return None

    def _ground_truth_after(self, tick_before: int) -> Optional[dict]:
        """Every scoring predicate and object pose after a call that moved the robot: trace only, for judging
        reviews offline (and, behind a switch, as an oracle review input); never in any agent's reply."""
        probe = getattr(self.rt.backend, "ground_truth", None)
        if self.rt.tick == tick_before or probe is None:
            return None
        try:
            return probe()
        except Exception as exc:  # noqa: BLE001 - telemetry must not break the episode
            return {"error": f"{type(exc).__name__}: {exc}"[:200]}

    def _reply(self, role: str, result: ToolResult) -> dict:
        consumer = "task" if role == "task" else f"local:{self.delegation}"
        note = ""
        if role == "task":
            left = self.decision_cap - self.decisions["task_capped"]
            note = f"[Decisions] {left} of {self.decision_cap} left (remember/api_reference do not count)"
            if left <= DECISION_WARNING:
                note += "; the episode stops at 0: secure the tiers you hold and return home first"
        snapshot = self.context.observation(role if role == "task" or self.contract else "task", consumer,
                                            decision=self.decisions.get(role, 0), contract=self.contract,
                                            extra_images=result.images, note=note)
        texts, images = [], []
        for part in snapshot["content"]:
            if part["type"] == "text":
                texts.append(part["text"])
            elif part["type"] == "image_url":
                images.append(part["image_url"]["url"].split(",", 1)[1])
        done = self._episode_done()
        return {"ok": result.ok, "result": self._agent_paths(result.text()),
                "context": self._agent_paths("\n".join(texts)), "images": images, "episode_done": done}

    def _agent_paths(self, text: str) -> str:
        """A contained agent sees its workspace at /workspace: host paths never reach it."""
        kit = self.kit
        if kit is None or kit.workspace is None:
            return text
        from knowin_agent.assembly import agent_workspace

        shown = agent_workspace(kit)
        host = str(kit.workspace.root)
        return text if shown == host else text.replace(host, shown)

    # -- control tools ---------------------------------------------------------------------------

    def _remember(self, args: dict) -> ToolOutcome:
        note = self.rt.ctx.notes.write(args["key"], args["text"], "task", self.rt.tick)
        return ToolOutcome({"status": "stored", "key": note.key})

    def _finish(self, args: dict):
        from knowin_agent.assembly import completion_refused

        refusal = completion_refused(self.rt, args.get("outcome"))
        if refusal is not None:
            return ToolOutcome(refusal)
        self.finish_report = {"outcome": args["outcome"], "summary": args["summary"], "tick": self.rt.tick}
        raise ControlTransfer("finish", {"status": "finished", **self.finish_report,
                                         "note": "the episode is over; stop now"})

    @staticmethod
    def _return(args: dict):
        raise ControlTransfer("return", {"status": args["status"], "report": args["report"],
                                         "evidence": args.get("evidence", "")})

    def _interact(self, args: dict) -> ToolOutcome:
        if self.launcher is None:
            raise ValueError("this host has no Local agent launcher")
        contract = InteractionContract.from_args(args)
        contract.max_rounds = min(contract.max_rounds, self.limits.max_local_rounds)
        contract.open({arm: state.ee_pose.xyz for arm, state in self.rt.obs.arms.items()})
        self.delegation = f"d{len(self.delegations) + 1}"
        start_tick = self.rt.tick
        self.recorder.record("delegations", delegation=self.delegation, phase="open", tick=start_tick,
                             contract=contract.describe())
        self.rt.bus.publish(EventKind.DELEGATED, "task", f"delegated '{contract.goal}' ({self.delegation})",
                            tick=start_tick)
        self.rt.bus.mark_seen(f"local:{self.delegation}")
        self.contract, self.local_report = contract, None
        self.local_tools = {t.name: t for t in local_toolset(self.rt, self.library, contract, return_to_task=self._return)}
        self.decisions["local_current"] = 0
        self.rt.contract, self.rt.active_scope, self.rt.track_owner = contract, "local", "local"
        process = self.launcher("local", self.prompt("local"))
        report = None
        try:
            while report is None:
                self.jobs.pump(0.2)
                if self.local_report is not None:
                    report = self.local_report
                elif self.rt.executor.episode_over:
                    report = {"status": "episode_over", "report": "the episode ended during local interaction"}
                elif process.poll() is not None:
                    report = {"status": "unresolved", "report": f"local agent exited (code {process.returncode}) "
                                                                "without return_to_task"}
                elif self.decisions["local_current"] >= contract.max_rounds + 2:
                    report = {"status": "round_limit", "report": f"Local used its {contract.max_rounds} rounds"}
                elif time.time() - self.started > self.limits.wall_seconds:
                    report = {"status": "unresolved", "report": "episode wall-clock budget exhausted"}
        finally:
            _stop(process)
            self.rt.contract, self.rt.active_scope, self.rt.track_owner = None, "task", "task"
            self.contract = None
        report = dict(report, delegation=self.delegation, local_decisions=self.decisions["local_current"],
                      ticks_used=self.rt.tick - start_tick)
        self.delegations.append({"delegation": self.delegation, "goal": contract.goal, "status": report["status"],
                                 "rounds": self.decisions["local_current"], "ticks": report["ticks_used"]})
        self.recorder.record("delegations", delegation=self.delegation, phase="return", tick=self.rt.tick, report=report)
        self.rt.bus.publish(EventKind.RETURNED, "local", f"local returned {report['status']} ({self.delegation})",
                            tick=self.rt.tick)
        return ToolOutcome(report)

    # -- episode loop (main thread) -------------------------------------------------------------

    def run(self, task_process_factory: Callable[[], subprocess.Popen],
            resume: Optional[Callable[[subprocess.Popen], subprocess.Popen]] = None) -> dict:
        """Drive the Task agent until it finishes or a limit ends the episode. When `resume` is given, a Task
        session that the model service cut off (session_error) is continued after a backoff, up to
        MAX_AGENT_RESUMES times within the wall-clock budget, instead of ending the episode."""
        self.library.freeze()
        self.rt.note_start_poses()
        self.recorder.record("episode", phase="start", mode="external_agents", instruction=self.rt.ctx.instruction)
        process = task_process_factory()
        stop = None
        try:
            while stop is None:
                self.jobs.pump(0.2)
                if self.finish_report is not None:
                    stop = "finished"
                elif self.rt.executor.episode_over and not self._attempt_may_restart():
                    stop = f"episode_over:{self.rt.executor.over_reason}"
                elif process.poll() is not None:
                    error = session_error(process) if resume is not None else None
                    if error is not None and self._wait_to_resume(error):
                        process = resume(process)
                        continue
                    stop = f"task_agent_exited:{process.returncode}"
                elif time.time() - self.started > self.limits.wall_seconds:
                    stop = "wall_budget"
                elif self.decisions["task_capped"] >= self.decision_cap:
                    stop = "task_decision_limit"
        finally:
            _stop(process)
        try:
            evaluation = self.rt.backend.evaluation()
        except Exception as exc:  # noqa: BLE001
            evaluation = {"error": str(exc)}
        result = {"stop_reason": stop, "finish_report": self.finish_report, "agent_resumes": self.agent_resumes,
                  "decisions": {"task": self.decisions["task"], "local": self.decisions["local"],
                                "task_capped": self.decisions["task_capped"], "task_cap": self.decision_cap},
                  "delegations": self.delegations, "budget": self.rt.budget(),
                  "wall_seconds": round(time.time() - self.started, 1), "evaluation_host_only": evaluation}
        self.recorder.record("episode", phase="end", **result)
        self.recorder.write_json("result.json", result)
        return result


def _stop(process: Optional[subprocess.Popen], grace: float = 10.0) -> None:
    if process is None:
        return
    container = getattr(process, "knowin_agent_container", None)
    if container and process.poll() is None:
        subprocess.run(["docker", "stop", "-t", str(int(grace)), container], capture_output=True)
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=grace)
    home = getattr(process, "knowin_agent_codex_home", None)
    if home is not None:
        (Path(home) / "auth.json").unlink(missing_ok=True)


class _Handler(socketserver.StreamRequestHandler):
    def handle(self):
        for line in self.rfile:
            if not line.strip():
                continue
            try:
                response = self.server.host.handle(json.loads(line))
            except Exception as exc:  # noqa: BLE001 - every request gets an answer
                response = {"error": f"{type(exc).__name__}: {exc}"}
            self.wfile.write(json.dumps(response, default=str).encode() + b"\n")
            self.wfile.flush()


class HostServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True

    def __init__(self, path: str, host: EpisodeHost, mode: int = 0o600):
        if os.path.exists(path):
            os.unlink(path)
        super().__init__(path, _Handler)
        os.chmod(path, mode)
        self.host = host

    def start(self) -> threading.Thread:
        thread = threading.Thread(target=self.serve_forever, daemon=True)
        thread.start()
        return thread


SESSION_CONFIG = 'cli_auth_credentials_store = "file"\nweb_search = "disabled"\n'
# Codex features that reach beyond the workspace and the model (connectors, browsers, computer use, plugins,
# image generation, dependency installs). A measured run gets the shell, images and its MCP tools only.
DISABLED_FEATURES = ("apps", "browser_use", "browser_use_external", "browser_use_full_cdp_access", "computer_use",
                     "image_generation", "in_app_browser", "plugins", "remote_plugin", "plugin_sharing",
                     "skill_mcp_dependency_install", "workspace_dependencies")


def _session_config(text: str) -> str:
    """The operator's Codex config for a measured run: web search off, file credentials, the features above
    off, and no [projects."<host path>"] trust tables (host paths are nothing the agent should see)."""
    out, skipping = [], False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            skipping = stripped.startswith(("[projects.", "[features]"))
        if skipping or stripped.split("=", 1)[0].strip() in ("web_search", "cli_auth_credentials_store"):
            continue
        out.append(line)
    features = "\n[features]\n" + "".join(f"{name} = false\n" for name in DISABLED_FEATURES)
    return SESSION_CONFIG + "\n".join(out) + "\n" + features


def _session_home(source: Path, target: Path) -> Path:
    """Private CODEX_HOME per session: auth and a cleaned config copied in; auth removed again by _stop."""
    import shutil

    target.mkdir(mode=0o700, parents=True, exist_ok=True)
    if (source / "auth.json").exists():
        shutil.copy2(source / "auth.json", target / "auth.json")
        os.chmod(target / "auth.json", 0o600)
    config = (source / "config.toml").read_text(encoding="utf-8") if (source / "config.toml").exists() else ""
    (target / "config.toml").write_text(_session_config(config), encoding="utf-8")
    os.chmod(target / "config.toml", 0o600)
    return target


def _host_ip() -> str:
    return subprocess.check_output(["hostname", "-I"], text=True).split()[0]


def _ensure_network(network: str, internal: bool = False) -> None:
    if subprocess.run(["docker", "network", "inspect", network], capture_output=True).returncode == 0:
        return
    subprocess.run(["docker", "network", "create", "--driver", "bridge", *(["--internal"] if internal else []),
                    network], capture_output=True)
    if subprocess.run(["docker", "network", "inspect", network], capture_output=True).returncode != 0:
        raise RuntimeError(f"cannot create Docker network {network!r}")


def docker_command(iso: dict, inner: list[str], *, name: str, run_name: str, workspace: Optional[Path],
                   writable: list[str], read_only: list[str], home: Path, socket_dir: str, pythonpath: str,
                   proxy_url: str) -> list[str]:
    """`docker run` for one agent session, the H4/H5 L20 container policy: read-only root, tmpfs /tmp, all
    capabilities dropped, no-new-privileges, pid/CPU/memory caps, a network whose only exit is the model
    allow-list proxy, and only the workspace (read-only, plus the phase's writable folders) mounted."""
    mounts = []
    if workspace is not None:
        mounts += ["--mount", f"type=bind,src={Path(workspace).resolve()},dst=/workspace,readonly"]
        for sub in writable:
            if (Path(workspace) / sub).is_dir():
                mounts += ["--mount", f"type=bind,src={(Path(workspace) / sub).resolve()},dst=/workspace/{sub}"]
        for sub in read_only:
            if (Path(workspace) / sub).is_dir():
                mounts += ["--mount", f"type=bind,src={(Path(workspace) / sub).resolve()},dst=/workspace/{sub},readonly"]
    mounts += ["--mount", f"type=bind,src={Path(home).resolve()},dst=/home/agent/.codex",
               "--mount", f"type=bind,src={socket_dir},dst=/run/knowin_agent,readonly",
               "--mount", f"type=bind,src={Path(pythonpath).resolve()},dst=/opt/knowin_agent/src,readonly",
               "--mount", f"type=bind,src={iso['codex_vendor']},dst=/opt/codex,readonly"]
    env = {"HOME": "/home/agent", "CODEX_HOME": "/home/agent/.codex", "CODEX_CI": "1", "LANG": "C.UTF-8",
           "LC_ALL": "C.UTF-8", "TERM": "xterm-256color",
           "PATH": "/opt/codex/bin:/opt/codex/codex-path:/usr/local/bin:/usr/bin:/bin",
           "HTTPS_PROXY": proxy_url, "HTTP_PROXY": proxy_url, "NO_PROXY": "localhost,127.0.0.1,::1"}
    return ["docker", "run", "--rm", "-i", "--init", "--read-only", "--tmpfs", "/tmp:mode=1777",
            "--user", "0:0", "--workdir", "/workspace" if workspace is not None else "/tmp",
            *[x for key, value in env.items() for x in ("--env", f"{key}={value}")],
            "--network", iso.get("internal_network") or iso.get("network", "knowin_agent-agent"), "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges", "--pids-limit", str(int(iso.get("pids", 512))),
            "--memory", f"{int(iso.get('memory_mb', 8192))}m", "--cpus", str(iso.get("cpus", 4)),
            "--label", f"knowin_agent.run={run_name}", "--name", name, *mounts, iso["image"], *inner]


def _session_events(process: subprocess.Popen) -> list:
    thread = getattr(process, "knowin_agent_log_thread", None)
    if thread is not None:
        thread.join(timeout=5)
    workdir = getattr(process, "knowin_agent_workdir", None)
    if workdir is None or not (Path(workdir) / "codex.jsonl").is_file():
        return []
    events = []
    for line in (Path(workdir) / "codex.jsonl").read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def session_thread(process: subprocess.Popen) -> Optional[str]:
    for event in _session_events(process):
        if event.get("type") == "thread.started" and event.get("thread_id"):
            return event["thread_id"]
    return None


def session_error(process: subprocess.Popen) -> Optional[str]:
    """The model-service error that ended an exited session (quota, rate limit, capacity, network), or None
    when its last turn completed, i.e. the agent itself stopped."""
    for event in reversed(_session_events(process)):
        kind = event.get("type")
        if kind == "turn.completed":
            return None
        if kind == "turn.failed":
            return str((event.get("error") or {}).get("message") or "turn failed")
        if kind == "error" and event.get("message"):
            return str(event["message"])
    return None


RESUME_PROMPT = ("Your previous turn was cut off by a model service error, not by you. The episode is still "
                 "running: the robot and scene are as your last completed tool call left them (a call that was "
                 "in flight may or may not have run, so check before repeating it). Continue the task.")


def _timestamp_lines(source, log) -> None:
    """Copy `codex exec --json` output to the log, adding the receive time `_ts` to every JSON event
    (Codex's own events carry none, and the replay needs them)."""
    try:
        for raw in iter(source.readline, b""):
            try:
                event = json.loads(raw)
                line = json.dumps({**event, "_ts": round(time.time(), 3)}, ensure_ascii=False).encode() + b"\n" \
                    if isinstance(event, dict) else raw
            except ValueError:
                line = raw
            log.write(line)
            log.flush()
    finally:
        log.close()


def codex_launcher(agent: dict, socket_path: str, run_dir: Path, *, workspace: Optional[Path] = None,
                   phase: str = "test") -> Callable[[str, str], subprocess.Popen]:
    """Launch `codex exec` for a role with the knowin_agent MCP server attached. agent keys: binary, model, effort,
    python, pythonpath, extra_args, env, tool_timeout_sec, codex_home_source, sandbox, isolation.

    isolation {"kind": "docker", image, codex_vendor, network, cpus, memory_mb, pids, upstream_proxy}: run each
    session in a container with the H4/H5 policy (docker_command); the practice Task agent may write skills/,
    lessons/ and scratch/, a test Task agent only scratch/, Local nothing. Without it Codex runs on the host
    under `sandbox` (default workspace-write for the practice Task agent, else read-only); hosts whose kernel
    refuses unprivileged user namespaces need "danger-full-access", which leaves the host fully exposed."""
    binary = agent.get("binary", "codex")
    python = agent.get("python", "python3")
    pythonpath = agent.get("pythonpath", str(Path(__file__).resolve().parents[2]))
    timeout = int(agent.get("tool_timeout_sec", 7200))
    iso = agent.get("isolation") or {}
    contained = iso.get("kind") == "docker"
    sessions = {"count": 0}
    proxy = {}
    if contained:
        from knowin_agent.bridge.model_proxy import ModelProxy, start_relay

        host, port = str(iso.get("upstream_proxy", "127.0.0.1:17890")).rsplit(":", 1)
        server = ModelProxy(iso.get("bind") or _host_ip(), upstream=(host, int(port)))
        proxy["server"], proxy["url"] = server, server.url
        if iso.get("internal_network"):
            # Agents get only the internal network; its one way out is the relay to the allow-list proxy.
            _ensure_network(iso["internal_network"], internal=True)
            _ensure_network(iso.get("network", "knowin_agent-agent"))
            proxy["relay"] = f"knowin_agent-relay-{os.getpid()}"
            address = start_relay(proxy["relay"], image=iso["image"], internal=iso["internal_network"],
                                  outbound=iso.get("network", "knowin_agent-agent"),
                                  target=(server.server_address[0], server.server_port),
                                  labels={"knowin_agent.run": run_dir.name})
            credentials = server.url.split("//", 1)[1].rsplit("@", 1)[0]
            proxy["url"] = f"http://{credentials}@{address}"
        else:
            _ensure_network(iso.get("network", "knowin_agent-agent"))

    def launch(role: str, prompt: str, resume_of: Optional[subprocess.Popen] = None) -> subprocess.Popen:
        """A new session, or (resume_of) the same Codex thread continued in the same session directory with
        freshly copied credentials, so a session cut off by the model service picks up where it stopped."""
        thread = session_thread(resume_of) if resume_of is not None else None
        if resume_of is not None and not thread:
            raise ValueError("the session to resume never started a thread")
        sessions["count"] += 1
        workdir = resume_of.knowin_agent_workdir if resume_of is not None else run_dir / f"agent-{sessions['count']:02d}-{role}"
        workdir.mkdir(parents=True, exist_ok=True)
        home = None
        if agent.get("codex_home_source"):
            home = _session_home(Path(agent["codex_home_source"]), run_dir / "private" / workdir.name)
        mcp_socket = f"/run/knowin_agent/{os.path.basename(socket_path)}" if contained else socket_path
        mcp_args = json.dumps(["-m", "knowin_agent.bridge.mcp", "--socket", mcp_socket, "--role", role])
        mcp_python, mcp_path = ("python3", "/opt/knowin_agent/src") if contained else (python, pythonpath)
        cwd = workspace if (role == "task" and workspace is not None) else workdir
        if contained:
            head = ["codex", "exec", "--dangerously-bypass-approvals-and-sandbox", "--skip-git-repo-check", "--json",
                    "-C", "/workspace" if workspace is not None else "/tmp"]
        else:
            sandbox = agent.get("sandbox") or (
                "workspace-write" if (role == "task" and workspace is not None and phase == "practice") else "read-only")
            head = [binary, "exec", "--skip-git-repo-check", "--json", "-C", str(cwd), "--sandbox", sandbox]
        command = head + ["-c", f'mcp_servers.knowin_agent.command="{mcp_python}"',
                          "-c", f"mcp_servers.knowin_agent.args={mcp_args}",
                          "-c", f'mcp_servers.knowin_agent.env={{PYTHONPATH="{mcp_path}"}}',
                          "-c", f"mcp_servers.knowin_agent.tool_timeout_sec={timeout}",
                          "-c", "mcp_servers.knowin_agent.startup_timeout_sec=60",
                          "-c", 'mcp_servers.knowin_agent.default_tools_approval_mode="approve"',
                          "-c", "mcp_servers.knowin_agent.supports_parallel_tool_calls=false"]
        if agent.get("model"):
            command += ["-m", agent["model"]]
        if agent.get("effort"):
            command += ["-c", f'model_reasoning_effort="{agent["effort"]}"']
        command += list(agent.get("extra_args", [])) + (["resume", thread, "-"] if thread else ["-"])
        container = None
        if contained:
            if home is None:
                raise ValueError("docker isolation needs agent.codex_home_source")
            container = f"knowin_agent-codex-{os.getpid()}-{sessions['count']}-{role}"
            writable = (["skills", "lessons", "scratch"] if phase == "practice" else ["scratch"]) if role == "task" else []
            command = docker_command(iso, command, name=container, run_name=run_dir.name,
                                     workspace=workspace, writable=writable,
                                     read_only=["skills/shared"] if "skills" in writable else [], home=home,
                                     socket_dir=os.path.dirname(socket_path), pythonpath=pythonpath,
                                     proxy_url=proxy["url"])
        log = open(workdir / "codex.jsonl", "ab" if thread else "wb")
        env = {**os.environ, **{k: str(v) for k, v in agent.get("env", {}).items()}}
        if home is not None and not contained:
            env["CODEX_HOME"] = str(home)
        if thread:
            with open(workdir / "prompt.md", "a", encoding="utf-8") as handle:
                handle.write(f"\n\n<!-- resumed thread {thread} -->\n{prompt}")
        else:
            (workdir / "prompt.md").write_text(prompt, encoding="utf-8")
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   env=env, cwd=workdir)
        process.knowin_agent_log_thread = threading.Thread(target=_timestamp_lines, args=(process.stdout, log), daemon=True)
        process.knowin_agent_log_thread.start()
        process.knowin_agent_codex_home = home
        process.knowin_agent_container = container
        process.knowin_agent_workdir = workdir
        process.stdin.write(prompt.encode())
        process.stdin.close()
        return process

    launch.proxy = proxy
    return launch


def run_hosted_episode(config: dict, run_dir: Path, *, backend=None) -> dict:
    """Boot the backend (or take ``backend``, e.g. the official evaluator's env), serve tools on a socket, and
    drive Task (and Local) through codex exec."""
    from knowin_agent.assembly import assemble
    from knowin_agent.episode import build_limits, finish_run

    run_dir = Path(run_dir)
    recorder = RunRecorder(run_dir)
    recorder.write_json("config.json", config)
    stride = int(config.get("video_stride", 2))
    frames = None
    if stride > 0:
        from knowin_agent.backends.robodojo.video import FrameRecorder

        frames = FrameRecorder(stride=stride)
    if backend is not None and frames is not None and getattr(backend, "frame_sink", None) is None:
        backend.frame_sink = frames
    kit = assemble(config, run_dir, backend=backend, frame_sink=frames, recorder=recorder)
    # Its own directory under an unlistable parent (as H4's session socket): a contained agent gets only this
    # directory mounted.
    import tempfile

    socket_path = kit.config.get("socket")
    if not socket_path:
        parent = Path("/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()) / f"knowin_agent-sock-{os.getuid()}"
        parent.mkdir(mode=0o711, exist_ok=True)
        parent.chmod(0o711)
        socket_path = os.path.join(tempfile.mkdtemp(prefix=f"{os.getpid()}-", dir=parent), "knowin_agent.sock")
    launcher = None
    try:
        workdir = kit.workspace.root if kit.workspace is not None else None
        launcher = codex_launcher(kit.config.get("agent", {}), socket_path, run_dir, workspace=workdir,
                                  phase=kit.phase)
        host = EpisodeHost(kit.runtime, kit.library, recorder=recorder, limits=build_limits(kit.config.get("limits")),
                           task_brief=kit.brief_text, launcher=launcher,
                           image_side=int(kit.config.get("image_side", 448)), kit=kit)
        contained = (kit.config.get("agent", {}).get("isolation") or {}).get("kind") == "docker"
        if contained:
            # Docker remaps container root to another uid: the directory must be traversable and the socket
            # connectable by it (as for H4's session socket). The directory holds nothing else.
            os.chmod(os.path.dirname(socket_path), 0o711)
        server = HostServer(socket_path, host, mode=0o666 if contained else 0o600)
        server.start()
        try:
            result = host.run(lambda: launcher("task", host.prompt("task")),
                              resume=lambda cut: launcher("task", RESUME_PROMPT, resume_of=cut))
        finally:
            server.shutdown()
            server.server_close()
        return finish_run(kit, run_dir, result)
    finally:
        # Everything but the simulator first: closing Isaac Sim can end the process outright.
        proxy = getattr(launcher, "proxy", None) or {}
        if proxy.get("relay"):
            from knowin_agent.bridge.model_proxy import stop_relay

            stop_relay(proxy["relay"])
        if proxy.get("server") is not None:
            proxy["server"].shutdown()
            proxy["server"].server_close()
        if os.path.exists(socket_path):
            os.unlink(socket_path)
        if not kit.config.get("socket"):
            try:
                os.rmdir(os.path.dirname(socket_path))
            except OSError:
                pass
        if frames is not None and frames.frames:
            frames.write(run_dir / "video", fps=kit.backend.profile.control_hz)
        kit.backend.close()
