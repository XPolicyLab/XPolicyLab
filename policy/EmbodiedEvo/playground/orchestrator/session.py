"""Host-owned persistent simulator. Requests arrive through a Unix socket."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import signal
import socket
import time
import traceback
from pathlib import Path

from environments import execute
from orchestrator.config import SESSION_PROTOCOL, config_from_dict
from orchestrator.files import append_jsonl, capture_writable_changes, write_json
from orchestrator.scoring import capture as capture_scoring
from orchestrator.session_client import receive, send
from simulators.observations import export_observation
from simulators.video import write_videos

PUBLIC_OUTPUT_LIMIT = 256 * 1024


def bind_instruction_budgets(workspace, env, budget):
    """Fill only explicit budget fields, once, before the agent reads instruction.md."""
    path = workspace / "instruction.md"
    if not path.is_file():
        return
    before = path.read_text()
    if "{{EXECUTION_BUDGET}}" not in before and "{{NATIVE_ACTION_LIMIT}}" not in before:
        return
    limit = getattr(env, "native_step_limit", env.max_steps)
    if env.max_steps is None or type(limit) is not int or limit <= 0:
        raise ValueError("Instruction requires a finite native action limit from the simulator")
    if type(budget) is not int or budget <= 0:
        raise ValueError("Instruction requires a positive execution-request budget")
    after = before.replace("{{EXECUTION_BUDGET}}", str(budget))
    after = after.replace("{{NATIVE_ACTION_LIMIT}}", str(limit))
    path.write_text(after)
    write_json(workspace.parent / "instruction-budget.json", {
        "execution_budget": budget,
        "native_action_limit": limit,
        "source_sha256": hashlib.sha256(before.encode()).hexdigest(),
        "instruction_sha256": hashlib.sha256(after.encode()).hexdigest(),
    })


class ExecutionTimeout(BaseException):
    """Cannot be swallowed by an agent program's except Exception."""


@contextlib.contextmanager
def deadline(seconds):
    timed_out = False

    def expired(signum, frame):
        nonlocal timed_out
        timed_out = True
        raise ExecutionTimeout("Execution timed out; the session cannot be resumed")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
        # A bare `except:` can catch BaseException. Returning afterward must
        # not turn an expired execution into a valid result.
        if timed_out:
            raise ExecutionTimeout("Execution timed out; the session cannot be resumed")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


class Session:
    def __init__(
        self, env, run, *, phase, seed, budget, seconds, execution_seconds, fps=15, event=None
    ):
        self.env, self.run = env, Path(run)
        self.workspace = self.run / "workspace"
        self.phase, self.budget = phase, budget
        self.seconds, self.execution_seconds = seconds, execution_seconds
        self.fps, self.event = fps, event or (lambda value: None)
        self.used = 0
        self.closed = False
        self.valid = True
        self.reason = "active"
        self.latest = "observations/000000"
        self.event_seq = 0
        self._writable_state = {}
        self._scoring = None
        self._scoring_finalized = False
        self.env.execution_phase = phase
        self.env.reset(seed=seed)
        bind_instruction_budgets(self.workspace, self.env, self.budget)
        self._scoring = capture_scoring(self.env, self.run, execution=0)
        export_observation(env, self.workspace / self.latest, workspace=self.workspace)
        self.started = time.monotonic()
        self.save()

    @staticmethod
    def _request_summary(request):
        summary = {"op": request.get("op")}
        files = request.get("files")
        if isinstance(files, list):
            summary["files"] = [
                {
                    "path": item.get("path"),
                    "bytes": len(item.get("code", "")) if isinstance(item, dict) else None,
                    "sha256": hashlib.sha256(item.get("code", "").encode()).hexdigest()
                    if isinstance(item, dict) and isinstance(item.get("code"), str)
                    else None,
                }
                for item in files
            ]
        return summary

    def record_rpc(self, request, response):
        self.event_seq += 1
        event = {
            "seq": self.event_seq,
            "timestamp": time.time(),
            "phase": self.phase,
            "request": self._request_summary(request),
        }
        execution = response.get("execution") if isinstance(response, dict) else None
        if isinstance(execution, int):
            event["code_ref"] = f"workspace/observations/{execution:06d}/code.py"
            observation = response.get("observation")
            if isinstance(observation, str):
                event["feedback_ref"] = f"workspace/{observation}"
            event["result"] = {
                key: response.get(key)
                for key in (
                    "ok",
                    "success",
                    "executions_used",
                    "native_steps_used",
                    "error",
                )
                if key in response
            }
        elif isinstance(response, dict) and not response.get("ok", True):
            event["error"] = response.get("error")
        append_jsonl(self.run / "events.jsonl", event)

    def status(self):
        remaining = (
            None
            if self.seconds is None
            else max(0.0, self.seconds - (time.monotonic() - self.started))
        )
        return {
            "ok": True,
            "phase": self.phase,
            "closed": self.closed,
            "reason": self.reason,
            "execution_budget": self.budget,
            "executions_used": self.used,
            "executions_remaining": max(0, self.budget - self.used),
            "seconds_remaining": None if remaining is None else round(remaining, 3),
            "execution_timeout_sec": self.execution_seconds,
            "native_action_limit": (
                None if self.env.max_steps is None
                else getattr(self.env, "native_step_limit", self.env.max_steps)
            ),
            "native_steps_used": self.env.control_steps,
            "native_steps_remaining": (
                None if self.env.max_steps is None
                else max(0, self.env.max_steps - self.env.control_steps)
            ),
            "success": bool(self.env.success),
            "terminated": bool(self.env.terminated),
            "truncated": bool(self.env.truncated),
            "observation": self.latest,
        }

    def save(self):
        if self.closed and self.valid and not self._scoring_finalized:
            self._scoring = capture_scoring(self.env, self.run, execution=self.used, final=True)
            self._scoring_finalized = True
        write_json(
            self.run / "session-result.json",
            {
                **self.status(),
                "protocol": SESSION_PROTOCOL,
                "valid": self.valid,
                "task_completed": bool(self.env.success),
                **(self._scoring or {}),
            },
        )

    def handle(self, request):
        request_started = time.monotonic()
        op = request.get("op")
        if not isinstance(op, str):
            raise ValueError("Operation must be a string")
        if op == "status":
            return self.status()
        if op == "finish":
            self.closed, self.reason = True, "agent_finished"
            self.save()
            return self.status()
        if op not in {"exec", "observe", "instruction", "reset"}:
            raise ValueError("Unknown session operation")
        if self.closed:
            raise ValueError("Session is closed")
        remaining = self.status()["seconds_remaining"]
        if remaining is not None and remaining <= 0:
            raise ValueError("Session time budget exhausted; finish your work")
        if op == "reset" and self.phase != "playground":
            raise PermissionError("reset() is forbidden in Test")
        if self.used >= self.budget:
            raise ValueError("Execution budget exhausted; status and finish remain available")
        fixed = {
            "observe": "get_observation()",
            "instruction": "print(get_instruction())",
            "reset": "reset()",
        }
        files = request.get("files") if op == "exec" else [{"path": op + ".py", "code": fixed[op]}]
        if not isinstance(files, list) or not files or len(files) > 256:
            raise ValueError("Expected 1 to 256 source files")
        if any(
            not isinstance(f, dict)
            or not isinstance(f.get("code"), str)
            or not isinstance(f.get("path"), str)
            for f in files
        ):
            raise ValueError("Invalid source snapshot")
        # Labels are metadata only. The server never opens an agent-supplied path.
        code = "\n\n".join(f["code"] for f in files)
        self.env.validate_action(code)
        self.used += 1
        number = f"{self.used:06d}"
        folder = self.workspace / "observations" / number
        folder.mkdir(parents=True, exist_ok=True)
        # Preserve the exact source sent to the simulator in the agent-visible
        # feedback directory.  This is the submission history; later edits to
        # submission/solution.py cannot change it.
        (folder / "code.py").write_text(code)
        write_json(
            folder / "sources.json",
            [
                {
                    "path": item["path"],
                    "bytes": len(item["code"]),
                    "sha256": hashlib.sha256(item["code"].encode()).hexdigest(),
                }
                for item in files
            ],
        )
        self.reason = "executing"
        self.save()
        execution_timeout = (
            self.execution_seconds if remaining is None else min(self.execution_seconds, remaining)
        )
        self.event(
            {
                "event": "executing",
                "timeout": execution_timeout,
            }
        )
        try:
            simulation_started = time.monotonic()
            with deadline(execution_timeout):
                result = execute(self.env, code)
            simulation_execution_sec = time.monotonic() - simulation_started
            # Persist before image/video encoding: feedback failure must not
            # discard a successfully captured native evaluator state.
            self._scoring = capture_scoring(self.env, self.run, execution=self.used)
            feedback_started = time.monotonic()
            export_observation(self.env, folder, workspace=self.workspace)
            videos = write_videos(result.frames, folder, stem="execution", fps=self.fps)
            self._writable_state = capture_writable_changes(
                self.workspace, folder, self._writable_state
            )
            feedback_capture_sec = time.monotonic() - feedback_started
            self.latest = str(folder.relative_to(self.workspace))
            self.reason = "active"
            public = {
                **self.status(),
                "ok": result.ok,
                # Keep the JSON RPC response below provider reverse-proxy
                # request limits. Full stdout/stderr remain in the observation
                # directory for local inspection.
                "stdout": result.stdout[-PUBLIC_OUTPUT_LIMIT:],
                "stderr": result.stderr[-PUBLIC_OUTPUT_LIMIT:],
                "execution": self.used,
                "videos": {k: str(Path(v).relative_to(self.workspace)) for k, v in videos.items()},
                "timing": {
                    "simulation_execution_sec": round(simulation_execution_sec, 6),
                    "feedback_capture_sec": round(feedback_capture_sec, 6),
                    "request_handler_sec": round(time.monotonic() - request_started, 6),
                },
            }
            (folder / "stdout.txt").write_text(result.stdout)
            (folder / "stderr.txt").write_text(result.stderr)
            write_json(folder / "result.json", public)
            with (self.run / "ledger.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "execution": self.used,
                            "reward": result.reward,
                            "success": result.success,
                            "ok": result.ok,
                            "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
                            "timing": public["timing"],
                            **(self._scoring or {}),
                        }
                    )
                    + "\n"
                )
            self.save()
            return public
        except BaseException:
            self.closed, self.valid, self.reason = True, False, "execution_failed"
            self.save()
            raise
        finally:
            self.event({"event": "idle"})


def serve(cfg, run, socket_path, *, phase, seed, budget, seconds, execution_seconds, ready_fd):
    from simulators.backend import build

    def event(value):
        os.write(ready_fd, (json.dumps(value) + "\n").encode())

    env = None
    try:
        env = build(cfg, seed=seed)
        session = Session(
            env,
            run,
            phase=phase,
            seed=seed,
            budget=budget,
            seconds=seconds,
            execution_seconds=execution_seconds,
            fps=cfg.feedback_fps,
            event=event,
        )
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(str(socket_path))
            os.chmod(socket_path, 0o600)
            listener.listen(8)
            listener.settimeout(0.5)
            event({"event": "ready"})
            while not session.closed:
                try:
                    connection, _ = listener.accept()
                except TimeoutError:
                    remaining = session.status()["seconds_remaining"]
                    if remaining is not None and remaining <= 0:
                        session.closed, session.reason = True, "time_budget"
                    continue
                # Buffered stream close can retry a failed flush. Suppress disconnects
                # around both send and close, without repeating the handled request.
                with (
                    connection,
                    contextlib.suppress(BrokenPipeError, ConnectionResetError, TimeoutError),
                ):
                    connection.settimeout(10)  # An incomplete request cannot hold the service open.
                    with connection.makefile("rwb") as stream:
                        request = None
                        try:
                            request = receive(stream)
                            response = session.handle(request)
                        except (ValueError, PermissionError, ConnectionError, OSError) as exc:
                            response = {
                                "ok": False,
                                "error": str(exc),
                                **{"status": session.status()},
                            }
                        except ExecutionTimeout as exc:
                            response = {"ok": False, "error": str(exc), "status": session.status()}
                        if request is not None:
                            session.record_rpc(request, response)
                        # A disconnected client never rolls back or replays physical actions.
                        send(stream, response)
            session.save()
            return 0 if session.valid else 1
    except (Exception, ExecutionTimeout) as exc:
        # Keep infrastructure tracebacks outside the agent workspace. Otherwise
        # DEVNULL child output reduces initialization/reset faults to a timeout.
        write_json(Path(run) / "simulator-error.json", {
            "type": type(exc).__name__, "message": str(exc),
            "traceback": traceback.format_exc(), "at": time.time(),
        })
        raise
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            os.close(ready_fd)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--phase", choices=["playground", "test"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument("--execution-seconds", type=float, required=True)
    parser.add_argument("--ready-fd", type=int, required=True)
    args = parser.parse_args(argv)
    cfg = config_from_dict(json.loads(args.config.read_text()))
    return serve(
        cfg,
        args.run_dir,
        args.socket,
        phase=args.phase,
        seed=args.seed,
        budget=args.budget,
        seconds=args.seconds,
        execution_seconds=args.execution_seconds,
        ready_fd=args.ready_fd,
    )


if __name__ == "__main__":
    raise SystemExit(main())
