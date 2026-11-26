# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transport original ICL requests through Runner decisions and feedback."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import queue
import socketserver
import subprocess
import tempfile
import threading
import uuid
from pathlib import Path

from roborsi.evaluation.media import CaseMediaChannel
from roborsi.evaluation.messages import (
    Action,
    BenchmarkQuery,
    EpisodeStop,
    Observation,
    QueryResult,
    StepResult,
    Task,
)


class _Closed(RuntimeError):
    pass


class IclRoboRSIAgent:
    """Own a controller process, yielding every native operation to the Runner."""

    def __init__(
        self, *, request: dict, directory: Path, media_dir: Path, command=None
    ):
        self.request = copy.deepcopy(request)
        self.directory = directory.absolute()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.media = CaseMediaChannel(media_dir)
        self.cfg = json.loads(Path(request["config"]).read_text())
        self._command = command
        self._outputs = queue.Queue()
        self._acks = queue.Queue()
        self._closed = threading.Event()
        self._cleanup_complete = False
        self._cleanup_lock = threading.Lock()
        self._task = None
        self._current = None
        self._pending = None
        self._process = None
        self._server = None
        self._socket_dir = None
        self._observation_dir = None
        self._monitor = None
        self._stdout = None
        self._stderr = None
        self._terminal = False
        self._lifecycle = None
        self._owner = Path(request["runs"]).absolute()
        self._offset = self.cfg["options"].get("replay", {}).get("control_step", 0)

    def reset(self, task: Task):
        """Pin the exact original instance before starting any controller."""
        if self._task is not None or self._closed.is_set():
            raise ValueError("Original ICL Agent owns one episode")
        episode = self.cfg["test_episodes"][0]
        limit = (
            self.cfg["options"]["review_window_steps"]
            * self.cfg["options"]["review_max_windows"]
        )
        if (
            task.benchmark != "robodojo"
            or task.task_id != episode["task"]
            or task.seed != episode["layout_id"]
            or task.native.get("eval_seed") != episode["eval_seed"]
            or task.max_steps != limit - self._offset
        ):
            raise ValueError("Agent task differs from original ICL instance")
        if not task.capabilities.get("recoverable_action_rejections", False):
            raise ValueError("Original ICL requires non-moving validation feedback")
        self._task = task

    def _identity(self):
        return {
            "benchmark": self._current.benchmark,
            "episode_id": self._current.episode_id,
            "step_index": self._current.step_index,
        }

    def _start(self):
        if self.request.get("experiment_root"):
            from .runtime_records import runtime_lock

            self._lifecycle = runtime_lock(
                Path(self.request["experiment_root"]), Path(self.request["case_dir"])
            )
            self._lifecycle.__enter__()
        self._socket_dir = tempfile.TemporaryDirectory(
            prefix="roborsi-icl-", dir="/tmp"
        )
        path = Path(self._socket_dir.name) / "backend.sock"
        agent = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                self.connection.settimeout(agent.cfg["request_seconds"])
                try:
                    request = json.loads(self.rfile.readline())
                    reply = agent._dispatch(request)
                except Exception:
                    import traceback

                    with (agent.directory / "private-transport-errors.log").open(
                        "a"
                    ) as stream:
                        traceback.print_exc(file=stream)
                    reply = {
                        "error": "Original ICL private transport failed; do not repeat action"
                    }
                try:
                    self.wfile.write(
                        (json.dumps(reply, allow_nan=False) + "\n").encode()
                    )
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self._server = socketserver.UnixStreamServer(str(path), Handler)
        self._server_thread = threading.Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": 0.1},
            daemon=True,
        )
        self._server_thread.start()
        request_path = self.directory / "controller-request.json"
        self.request["framework_episode_id"] = self._current.episode_id
        self.request["attempt_dir"] = str(self.directory.parents[1])
        self.request["native_config_sha256"] = hashlib.sha256(
            Path(self.request["config"]).read_bytes()
        ).hexdigest()
        request_path.write_text(json.dumps(self.request, indent=2))
        env = dict(os.environ)
        # The private HTTP client lives in the parent. The original controller
        # uses only its Unix socket and never needs the evaluator credential.
        for name in ("ROBORSI_BENCHMARK_TOKEN", "RPENT_BENCHMARK_TOKEN"):
            env.pop(name, None)
        source_root = self.request.get("framework_source_root")
        if source_root:
            env["PYTHONPATH"] = (
                str(source_root) + os.pathsep + env.get("PYTHONPATH", "")
            )
        env["ROBORSI_ICL_PRIVATE_SOCKET"] = str(path)
        env["ROBORSI_ICL_REQUEST_SECONDS"] = str(self.cfg["request_seconds"])
        self._stdout = (self.directory / "controller.stdout.log").open("wb")
        self._stderr = (self.directory / "controller.stderr.log").open("wb")
        argv = (
            self._command(request_path)
            if self._command
            else [
                self.request["controller_python"],
                "-m",
                "robots.robodojo.agents.icl_pi05.host",
                str(request_path),
            ]
        )
        self._process = subprocess.Popen(
            argv,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=self._stdout,
            stderr=self._stderr,
            start_new_session=True,
        )
        if self.request.get("experiment_root"):
            from .runtime_records import record_process

            try:
                record_process(
                    Path(self.request["experiment_root"]),
                    Path(self.request["case_dir"]),
                    "controller",
                    self._process.pid,
                )
            except BaseException:
                self.close()
                raise

        def monitor():
            code = self._process.wait()
            self._outputs.put(
                _Closed(
                    f"Original ICL controller exited ({code}); see its retained summary/logs"
                )
            )

        self._monitor = threading.Thread(target=monitor, daemon=True)
        self._monitor.start()

    def _await_ack(self, decision):
        self._outputs.put(decision)
        while not self._closed.is_set():
            try:
                value = self._acks.get(timeout=0.1)
                if isinstance(value, Exception):
                    raise value
                return value
            except queue.Empty:
                pass
        raise _Closed("Original ICL bridge closed")

    def _copy_transition(self, observation: Observation):
        reply = copy.deepcopy(observation.data["icl_transition"])
        for name, value in observation.data.get("icl_media", {}).items():
            relative = Path(name.removeprefix("obs://"))
            if (
                not name.startswith("obs://")
                or relative.is_absolute()
                or ".." in relative.parts
            ):
                raise ValueError("Invalid original observation reference")
            from roborsi.evaluation.messages import MediaRef

            ref = MediaRef.from_dict(value)
            if ref not in observation.media:
                raise ValueError(
                    "Original image mapping lacks a matching media reference"
                )
            payload = self.media.read(
                ref, observation.episode_id, observation.step_index
            )
            destination = self._observation_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.is_symlink() or destination.parent.is_symlink():
                raise ValueError("Original observation destination is linked")
            if destination.exists() and destination.read_bytes() != payload:
                raise ValueError("Original immutable image changed at the same step")
            destination.write_bytes(payload)
        if (
            reply["observation"]["control_step"]
            != observation.step_index + self._offset
        ):
            raise ValueError("Original and Runner control steps differ")
        return reply

    def _dispatch(self, request):
        if self._closed.is_set():
            raise _Closed("Original ICL bridge closed")
        op = request.get("op")
        if op == "reset":
            if self._observation_dir is not None:
                raise ValueError("Original controller attempted a second native reset")
            if request["episode"] != self.cfg["test_episodes"][0]:
                raise ValueError("Original initialization identity differs")
            target = Path(request["observation_dir"]).absolute()
            if (
                not target.is_relative_to(self._owner)
                or target.is_symlink()
                or not target.is_dir()
            ):
                raise ValueError("Original observation directory is outside this run")
            self._observation_dir = target
            run = target.parents[1]
            binding = {
                "framework_episode_id": self._current.episode_id,
                "native_config_sha256": self.request.get("native_config_sha256"),
                "source_run": str(run),
                "case_dir": self.request.get("case_dir"),
                "attempt_dir": self.request.get("attempt_dir"),
            }
            if self.request.get("experiment_root"):
                (run / "roborsi-controller.json").write_text(
                    json.dumps(binding, indent=2)
                )
            transition = self._copy_transition(self._current)
            if transition["done"]:
                self._outputs.put(
                    EpisodeStop(**self._identity(), reason="original_initial_terminal")
                )
            return transition
        if op == "vla_propose":
            query = BenchmarkQuery(
                **self._identity(),
                query_id=uuid.uuid4().hex,
                method="icl_vla_propose",
                arguments={"subgoal": request["subgoal"]},
            )
            value = self._await_ack(query)
            if "icl_invalid_action" in value.data:
                return {"invalid_action": value.data["icl_invalid_action"]}
            return value.data["icl_reply"]
        if op == "step":
            value = self._await_ack(
                Action(**self._identity(), data={"icl_action": request["action"]})
            )
            if value.data.get("action_rejected") is True:
                return {"invalid_action": value.data["icl_invalid_action"]}
            return self._copy_transition(value)
        if op == "finish":
            if self._terminal:
                return self._copy_transition(self._current)
            value = self._await_ack(
                EpisodeStop(
                    **self._identity(),
                    reason=request.get("original_reason", "original_icl_finish"),
                )
            )
            return self._copy_transition(value)
        if op == "close":
            return {"closed": True}
        raise ValueError("Unsupported original ICL backend operation")

    def _next(self):
        while not self._closed.is_set():
            try:
                value = self._outputs.get(timeout=0.1)
                if isinstance(value, Exception):
                    raise value
                self._pending = value
                return value
            except queue.Empty:
                pass
        raise _Closed("Original ICL bridge closed")

    def act(self, observation: Observation):
        """Supply one acknowledged transition, then wait for a real decision."""
        if self._task is None:
            raise ValueError("Original ICL Agent is not reset")
        if self._current is not None and (
            observation.benchmark,
            observation.episode_id,
        ) != (self._current.benchmark, self._current.episode_id):
            raise ValueError("Original ICL episode changed")
        self._current = observation
        if self._process is None:
            if observation.step_index != 0:
                raise ValueError("Runner must initialize its episode index at zero")
            self._start()
        else:
            if not isinstance(self._pending, Action):
                raise ValueError("Original ICL has no action awaiting feedback")
            expected = self._pending.step_index + (
                0 if observation.data.get("action_rejected") is True else 1
            )
            if observation.step_index != expected:
                raise ValueError("Original ICL action feedback is out of order")
            self._pending = None
            self._acks.put(observation)
        return self._next()

    def resume(self, reply: QueryResult):
        """This method supplies Query feedback; it is not experiment resume."""
        pending = self._pending
        if not isinstance(pending, BenchmarkQuery) or (
            reply.benchmark,
            reply.episode_id,
            reply.step_index,
            reply.query_id,
        ) != (
            pending.benchmark,
            pending.episode_id,
            pending.step_index,
            pending.query_id,
        ):
            raise ValueError("Original ICL query feedback differs")
        self._pending = None
        self._acks.put(reply)
        return self._next()

    def observe_end(self, step: StepResult):
        """Deliver final ACK and let the original controller finish its records."""
        if not (step.terminated or step.truncated) or self._pending is None:
            raise ValueError("Original ICL terminal feedback is not pending")
        expected = self._pending.step_index + (
            1 if isinstance(self._pending, Action) else 0
        )
        if step.observation.step_index != expected:
            raise ValueError("Original ICL terminal feedback step differs")
        if (step.observation.benchmark, step.observation.episode_id) != (
            self._pending.benchmark,
            self._pending.episode_id,
        ):
            raise ValueError("Original ICL terminal feedback episode differs")
        if step.observation.data.get("icl_transition", {}).get("done") is not True:
            raise ValueError("Original ICL terminal feedback lacks native done")
        self._terminal = True
        self._current = step.observation
        self._pending = None
        self._acks.put(step.observation)
        # No new terminal timeout: original run_segment owns its remaining wall
        # budget, final review and 15-second acknowledgement allowance.
        self._process.wait()
        receipt = Path(self.request["receipt"])
        if not receipt.is_file():
            raise _Closed("Original ICL controller did not save its final receipt")

    def close(self):
        """Cancel only this controller and release its private transport."""
        with self._cleanup_lock:
            if self._cleanup_complete:
                return
            self._closed.set()
            self._acks.put(_Closed("Original ICL bridge cancelled"))
            if self._process is not None and self._process.poll() is None:
                import signal
                import time

                from roborsi.planner.codex_cli import _terminate_process

                # Let the unchanged controller's KeyboardInterrupt handler
                # retain its checkpoint/summary before bounded hard cleanup.
                try:
                    self._process.send_signal(signal.SIGINT)
                    self._process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    pass
                except ProcessLookupError:
                    pass
                _terminate_process(self._process, time.monotonic() + 30)
            if self._server is not None:
                self._server.shutdown()
                self._server.server_close()
                self._server_thread.join(2)
            if self._monitor is not None:
                self._monitor.join(2)
            if self._stdout is not None:
                self._stdout.close()
                self._stderr.close()
            if self._socket_dir is not None:
                self._socket_dir.cleanup()
            if self._lifecycle is not None:
                self._lifecycle.__exit__(None, None, None)
                self._lifecycle = None
            if self.request.get("experiment_root"):
                from .runtime_records import snapshot_runtime

                snapshot_runtime(
                    Path(self.request["experiment_root"]),
                    Path(self.request["case_dir"]),
                )
            self._cleanup_complete = True
