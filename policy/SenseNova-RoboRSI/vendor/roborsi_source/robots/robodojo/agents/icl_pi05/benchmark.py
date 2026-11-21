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

"""Expose the unchanged ICL native adapter through the RoboRSI boundary."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import uuid
from pathlib import Path

from roborsi.evaluation.media import CaseMediaChannel
from roborsi.evaluation.messages import (
    Action,
    BenchmarkQuery,
    EpisodeStop,
    NativeResult,
    Observation,
    QueryResult,
    StartResult,
    StepResult,
    Task,
)


def original_backend(source: Path):
    """Load the private original transport without replacing global policylab."""
    name = f"_icl_original_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(
        name,
        source / "code/policylab/__init__.py",
        submodule_search_locations=[str(source / "code/policylab")],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    backend = __import__(f"{name}.backend", fromlist=["Backend"])
    api = __import__(f"{name}.api", fromlist=["InvalidAction"])
    return backend.Backend, api.InvalidAction


class IclBenchmarkAdapter:
    """Own one original simulator worker; preserve its transition and score."""

    def __init__(
        self, config: Path, directory: Path, media_dir: Path, *, backend_factory=None
    ):
        self.config = json.loads(config.read_text())
        self._config_sha = hashlib.sha256(config.read_bytes()).hexdigest()
        self.directory = directory.absolute()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.media = CaseMediaChannel(media_dir)
        self._backend_factory = backend_factory
        self._backend = None
        self._invalid_action = ValueError
        self._task = None
        self._episode_id = None
        self._reply = None
        self._offset = 0
        self._finished = None
        self._closed = False
        self._step = 0
        self._protocol_failure = False
        self.observations = self.directory / "observations/evaluation-0001"
        self.records = self.directory / "episodes/evaluation-0001"

    def _identity(self, value):
        if (value.benchmark, value.episode_id, value.step_index) != (
            self._task.benchmark,
            self._episode_id,
            self._step,
        ):
            raise ValueError("Original ICL request identity/step differs")

    def start(self, task: Task, episode_id: str) -> StartResult:
        """Initialize only this service's original worker and native episode."""
        if self._backend is not None or self._closed:
            raise ValueError("Original ICL benchmark owns one episode")
        episode = self.config["test_episodes"][0]
        full_limit = (
            self.config["options"]["review_window_steps"]
            * self.config["options"]["review_max_windows"]
        )
        replay = self.config["options"].get("replay")
        self._offset = replay["control_step"] if replay else 0
        if (
            task.benchmark != "robodojo"
            or task.task_id != episode["task"]
            or task.seed != episode["layout_id"]
            or task.native.get("eval_seed") != episode["eval_seed"]
            or task.native.get("config") != episode["robot"]
            or task.native.get("reset_seed", episode["reset_seed"])
            != episode["reset_seed"]
            or task.max_steps != full_limit - self._offset
        ):
            raise ValueError("Runner task differs from original ICL episode/budget")
        if not task.capabilities.get("recoverable_action_rejections", False):
            raise ValueError("Original ICL requires recoverable action feedback")
        self._task, self._episode_id = task, episode_id
        self.observations.mkdir(parents=True)
        (self.records / "native").mkdir(parents=True)
        (self.directory / "config.json").write_text(json.dumps(self.config, indent=2))
        (self.records / "episode.json").write_text(json.dumps(episode))
        source = Path(self.config["adapter"].rsplit(":", 1)[0]).parent
        if self._backend_factory is None:
            factory, self._invalid_action = original_backend(source)
        else:
            factory = self._backend_factory
        self._backend = factory(self.config, self.directory)
        self._backend.request(op="initialize")
        self._reply = self._backend.request(
            op="reset",
            episode=episode,
            record_dir=str(self.records / "native"),
            observation_dir=str(self.observations),
        )
        if self._reply["observation"]["control_step"] != self._offset:
            raise ValueError(
                "Original initialization did not reach requested checkpoint"
            )
        if (
            self._reply["observation"].get("step_limit") != full_limit
            or self._reply["observation"].get("instruction") != task.instruction
        ):
            raise ValueError("Actual original instruction/horizon differs from catalog")
        return StartResult(
            task=task, observation=self._observation(), capabilities=task.capabilities
        )

    def _observation(self, *, rejected: str | None = None) -> Observation:
        refs, mapping = [], {}

        def collect(value):
            if isinstance(value, dict):
                for child in value.values():
                    collect(child)
            elif isinstance(value, list):
                for child in value:
                    collect(child)
            elif isinstance(value, str) and value.startswith("obs://"):
                relative = Path(value[6:])
                path = self.observations / relative
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or path.is_symlink()
                ):
                    raise ValueError("Unsafe original observation image")
                if value not in mapping:
                    reference = self.media.write(
                        self._episode_id,
                        self._step,
                        path.read_bytes(),
                        "image/png",
                    )
                    refs.append(reference)
                    mapping[value] = reference.as_dict()

        collect(self._reply["observation"])
        data = {"icl_transition": copy.deepcopy(self._reply), "icl_media": mapping}
        if rejected is not None:
            data["action_rejected"] = True
            data["icl_invalid_action"] = rejected
        return Observation(
            benchmark=self._task.benchmark,
            episode_id=self._episode_id,
            step_index=self._step,
            data=data,
            media=tuple(refs),
        )

    def query(self, request: BenchmarkQuery) -> QueryResult:
        """Forward an original proposal without advancing native physics."""
        self._identity(request)
        if request.method != "icl_vla_propose" or set(request.arguments) != {"subgoal"}:
            raise ValueError("Unsupported original ICL query")
        try:
            data = {
                "icl_reply": self._backend.request(
                    op="vla_propose", **request.arguments
                )
            }
        except self._invalid_action as exc:
            data = {"icl_invalid_action": str(exc)}
        return QueryResult(
            benchmark=request.benchmark,
            episode_id=request.episode_id,
            step_index=request.step_index,
            query_id=request.query_id,
            data=data,
        )

    def step(self, action: Action) -> StepResult:
        """Forward one original step, keeping recoverable zero-movement errors."""
        self._identity(action)
        if set(action.data) != {"icl_action"}:
            raise ValueError("Original ICL action envelope differs")
        try:
            reply = self._backend.request(op="step", action=action.data["icl_action"])
        except self._invalid_action as exc:
            return StepResult(
                observation=self._observation(rejected=str(exc)),
                terminated=False,
                truncated=False,
                native={"action_rejected": True},
            )
        if reply["observation"]["control_step"] != self._offset + self._step + 1:
            raise ValueError("Original native step acknowledgement differs")
        self._step += 1
        self._reply = reply
        return self._step_result()

    def _step_result(self):
        done = self._reply["done"]
        return StepResult(
            observation=self._observation(),
            terminated=done,
            truncated=False,
            termination_reason="native_terminal" if done else None,
            native={"icl_transition": copy.deepcopy(self._reply)},
        )

    def stop(self, request: EpisodeStop) -> StepResult:
        """Use the original finish operation, never another early-stop scorer."""
        self._identity(request)
        self._protocol_failure = request.reason == "reset_requested"
        if not self._reply["done"]:
            self._reply = self._backend.request(op="finish")
        if not self._reply["done"]:
            raise ValueError("Original native finish did not terminate")
        return self._step_result()

    def finish(self) -> NativeResult:
        """Publish the original native result without calling finish twice."""
        if not self._reply or not self._reply["done"]:
            raise ValueError("Original ICL episode has no native terminal result")
        if self._finished is None:
            metrics = self._reply["metrics"]
            self._finished = NativeResult(
                task=self._task,
                episode_id=self._episode_id,
                success=False if self._protocol_failure else metrics["success"],
                score=metrics["score"],
                details={
                    "original_metrics": copy.deepcopy(metrics),
                    "failure_type": "reset_requested"
                    if self._protocol_failure
                    else None,
                    "original_control_step": self._reply["observation"]["control_step"],
                },
            )
            (self.directory / "native-result.json").write_text(
                json.dumps(self._finished.as_dict(), indent=2)
            )
        return self._finished

    def close(self):
        """Close only this adapter's worker."""
        if not self._closed:
            if self._backend is not None:
                self._backend.close()
                process = getattr(self._backend, "process", None)
                verified = process is not None and process.poll() is not None
                if process is not None and not verified:
                    raise RuntimeError("Original native worker is still running")
                (self.directory / "worker-close.json").write_text(
                    json.dumps(
                        {
                            "native_closed": verified,
                            "pid": process.pid if process is not None else None,
                            "returncode": process.returncode
                            if process is not None
                            else None,
                            "episode_id": self._episode_id,
                            "native_config_sha256": self._config_sha,
                            "identity": self._task.as_dict()
                            if self._task is not None
                            else None,
                        },
                        indent=2,
                    )
                )
            self._closed = True
