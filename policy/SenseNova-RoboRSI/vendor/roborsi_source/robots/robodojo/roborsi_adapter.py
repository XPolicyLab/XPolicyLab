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

"""RoboDojo's native evaluator in a private process behind Benchmark HTTP."""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path
from typing import Any

from roborsi.evaluation.messages import (
    Action,
    BenchmarkQuery,
    EpisodeStop,
    NativeResult,
    QueryResult,
    StartResult,
    StepResult,
    Task,
)
from roborsi.utils.config import get_repo_root
from roborsi.utils.daemon import ProcessDaemon, pick_free_port
from roborsi.utils.rpc import make_rpc_client, wait_for_ready

ACTION_SCHEMA = {
    "oneOf": [
        {
            "type": "object",
            "properties": {
                **{
                    f"{arm}_{kind}": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": size,
                        "maxItems": size,
                        "description": "Absolute joint targets in radians"
                        if kind == "arm_joint_state"
                        else "World xyz in metres followed by quaternion wxyz",
                    }
                    for arm in ("left", "right")
                },
                **{
                    f"{arm}_ee_joint_state": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 1,
                        "maxItems": 1,
                        "description": "Native normalized gripper target; native clipping and scaling apply",
                    }
                    for arm in ("left", "right")
                },
            },
            "required": [
                f"{arm}_{field}"
                for arm in ("left", "right")
                for field in (kind, "ee_joint_state")
            ],
            "additionalProperties": False,
        }
        for kind, size in (("arm_joint_state", 6), ("ee_pose", 7))
    ],
}
EVALUATION_RULES = {
    "success_metric": "native_success",
    "score_metric": "native_episode_score",
    "aggregation": "case_mean",
    "error_denominator": "valid",
}


class RoboDojoRoboRSIAdapter:
    """Own a simulator process; files stay inside the Benchmark service."""

    def __init__(
        self,
        directory: Path,
        *,
        factory: str = "robots.robodojo.native:make_runtime",
        operation_timeout_s: float = 900.0,
    ) -> None:
        if not math.isfinite(operation_timeout_s) or operation_timeout_s <= 0:
            raise ValueError("operation timeout must be finite and positive")
        self.directory = directory.resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "native").mkdir(exist_ok=True)
        self._timeout = operation_timeout_s
        self._closed = False
        port = pick_free_port()
        self.daemon = ProcessDaemon(
            "roborsi-robodojo",
            [
                sys.executable,
                "-m",
                "robots.robodojo.worker",
                "--port",
                str(port),
                "--directory",
                str(self.directory),
                "--factory",
                factory,
            ],
            cwd=str(self.directory / "native"),
            log_path=str(self.directory / "native/worker.log"),
            prepend_rlinf_path=False,
            env_overrides={
                "PYTHONPATH": str(get_repo_root())
                + os.pathsep
                + os.environ.get("PYTHONPATH", "")
            },
        )
        self._rpc = make_rpc_client(f"http://127.0.0.1:{port}")
        try:
            self.daemon.start()
            wait_for_ready(self._rpc, timeout_s=30, daemon=self.daemon)
        except BaseException:
            self.close()
            raise

    def _call(self, method: str, payload: dict) -> dict:
        if self._closed:
            raise RuntimeError("RoboDojo process is closed")
        return self._rpc.call("request", (method, payload), timeout_s=self._timeout)

    def inspect(self, specification: dict[str, Any]) -> Task:
        """Prepare a native scene and export its identity without scoring it."""
        return Task.from_dict(self._call("inspect", specification))

    def start(self, task: Task, episode_id: str) -> StartResult:
        return StartResult.from_dict(
            self._call("start", {"task": task.as_dict(), "episode_id": episode_id})
        )

    def step(self, action: Action) -> StepResult:
        return StepResult.from_dict(self._call("step", action.as_dict()))

    def finish(self) -> NativeResult:
        return NativeResult.from_dict(self._call("finish", {}))

    def stop(self, request: EpisodeStop) -> StepResult:
        """Ask the native owner to terminate without an extra environment step."""
        return StepResult.from_dict(self._call("stop", request.as_dict()))

    def query(self, request: BenchmarkQuery) -> QueryResult:
        """Ask the native owner to plan without advancing the environment."""
        return QueryResult.from_dict(self._call("query", request.as_dict()))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self.daemon.poll() is None:
                try:
                    self._rpc.call("request", ("close", {}), timeout_s=5)
                    self._rpc.call("shutdown", timeout_s=1)
                except Exception:
                    pass
                # The process may exit before sending its shutdown response.
                end = time.monotonic() + 1
                while self.daemon.poll() is None and time.monotonic() < end:
                    time.sleep(0.02)
        finally:
            try:
                self.daemon.stop(timeout=5)
            finally:
                self._rpc.close()
