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

"""Keep native initialization, rendering, control and scoring on the main thread."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import queue
import threading
from concurrent.futures import Future
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

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
from roborsi.evaluation.service_host import load_factory
from roborsi.utils.daemon import watch_parent_death
from roborsi.utils.rpc import RpcFacade
from roborsi.utils.serialization import to_numpy_tree
from robots.robodojo.episode import mark_incomplete_episode_failed
from robots.robodojo.roborsi_adapter import ACTION_SCHEMA


class StopEpisode(Exception):
    """End inspection or cancellation without producing a native score."""


def json_state(value: Any) -> Any:
    """Keep small native state in JSON; camera arrays use PNG artifacts."""
    import numpy as np

    value = to_numpy_tree(value)

    def convert(item):
        if isinstance(item, np.ndarray):
            if item.nbytes > 65536:
                raise ValueError("large RoboDojo state needs an artifact")
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, dict):
            return {key: convert(v) for key, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [convert(v) for v in item]
        return item

    result = convert(value)
    json.dumps(result, allow_nan=False)
    return result


class EpisodeWorker(RpcFacade):
    """RPC requests wait while the native policy callback drives one episode."""

    def __init__(self, directory: Path, runtime: Any) -> None:
        super().__init__()
        self.directory = directory
        self.runtime = runtime
        self.requests = queue.Queue()
        self._rpc["request"] = self.request
        self._media = CaseMediaChannel(directory / "media")
        self._started = False

    def request(self, method: str, payload: dict) -> dict:
        future = Future()
        self.requests.put((method, payload, future))
        return future.result()

    def _observation(
        self, task: Task, episode_id: str, step: int, obs: dict
    ) -> Observation:
        import numpy as np
        from PIL import Image

        expected = task.capabilities["observation_channels"]
        if sorted(obs["vision"]) != expected:
            raise ValueError("RoboDojo cameras differ from the exported catalog")
        if obs["instruction"] != task.instruction:
            raise ValueError("RoboDojo instruction differs from the exported catalog")
        media = []
        for camera in expected:
            values = np.asarray(obs["vision"][camera]["color"])
            if values.dtype != np.uint8 or values.ndim != 3 or values.shape[2] != 3:
                raise ValueError("RoboDojo cameras must return RGB uint8 arrays")
            output = io.BytesIO()
            Image.fromarray(values).save(output, format="PNG")
            media.append(
                self._media.write(episode_id, step, output.getvalue(), "image/png")
            )
        data = json_state({key: value for key, value in obs.items() if key != "vision"})
        data["media_fields"] = {name: i for i, name in enumerate(expected)}
        # The native policy renders cameras in source insertion order; the
        # catalog's sorted channel list only identifies the media references.
        data["camera_order"] = list(obs["vision"])
        return Observation(
            benchmark="robodojo",
            episode_id=episode_id,
            step_index=step,
            data=data,
            media=tuple(media),
        )

    def _episode(self, method: str, payload: dict, initial: Future) -> None:
        requested = Task.from_dict(payload["task"]) if method == "start" else None
        specification = (
            payload
            if requested is None
            else {
                "task_id": requested.task_id,
                "layout_id": requested.seed,
                "eval_seed": requested.native["eval_seed"],
                "max_steps": requested.max_steps,
            }
        )
        pending_finish = None
        pending_close = None
        actual = None
        episode_id = payload.get("episode_id", "inspect")

        def interact(env, task: Task, observation: dict):
            nonlocal pending_finish, pending_close, actual
            actual = task
            if method == "inspect":
                initial.set_result(task.as_dict())
                raise StopEpisode()
            if requested != actual:
                raise ValueError(
                    "RoboDojo native task, layout, instruction or limits differ"
                )
            steps = 0
            if env.take_action_cnt != [0] or env.end_flag != [False]:
                raise ValueError("RoboDojo reset must start an unfinished episode")
            last_observation = self._observation(task, episode_id, 0, observation)
            initial.set_result(
                StartResult(
                    task=task,
                    observation=last_observation,
                    capabilities=task.capabilities,
                ).as_dict()
            )
            while True:
                operation, message, future = self.requests.get()
                try:
                    if operation == "close":
                        pending_close = future
                        raise StopEpisode()
                    if operation == "finish":
                        if not env.end_flag[0]:
                            raise ValueError("RoboDojo must end before scoring")
                        pending_finish = future
                        return
                    if operation == "query":
                        request = BenchmarkQuery.from_dict(message)
                        if (
                            request.benchmark,
                            request.episode_id,
                            request.step_index,
                        ) != ("robodojo", episode_id, steps):
                            raise ValueError(
                                "RoboDojo query identity or sequence differs"
                            )

                        def check_query_state():
                            if (
                                env.take_action_cnt != [steps]
                                or env.env_seeds != [task.seed]
                                or env.end_flag != [False]
                                or env.unstable_envs
                            ):
                                raise ValueError(
                                    "RoboDojo query changed or lost the active scene"
                                )

                        check_query_state()
                        data = self.runtime.query(
                            env, request.method, request.arguments
                        )
                        check_query_state()
                        future.set_result(
                            QueryResult(
                                benchmark="robodojo",
                                episode_id=episode_id,
                                step_index=steps,
                                query_id=request.query_id,
                                data=data,
                            ).as_dict()
                        )
                        continue
                    if operation == "stop":
                        request = EpisodeStop.from_dict(message)
                        if (
                            request.benchmark,
                            request.episode_id,
                            request.step_index,
                        ) != (
                            "robodojo",
                            episode_id,
                            steps,
                        ):
                            raise ValueError(
                                "RoboDojo stop identity or sequence differs"
                            )
                        if request.reason not in {
                            "give_up",
                            "budget_exhausted",
                            "capability_failure",
                            "done",
                        }:
                            raise ValueError(
                                "RoboDojo stop reason is not a policy termination"
                            )
                        if env.end_flag[0]:
                            raise ValueError("RoboDojo episode has already ended")
                        mark_incomplete_episode_failed(env)
                        if not env.end_flag[0]:
                            raise ValueError(
                                "RoboDojo stop did not terminate the native episode"
                            )
                        if env.take_action_cnt != [steps] or env.env_seeds != [
                            task.seed
                        ]:
                            raise ValueError(
                                "RoboDojo stop changed the native counter or layout"
                            )
                        if env.unstable_envs:
                            raise ValueError("RoboDojo excluded an unstable layout")
                        future.set_result(
                            StepResult(
                                observation=last_observation,
                                terminated=True,
                                truncated=False,
                                termination_reason="native_success"
                                if env.success[0]
                                else request.reason,
                                native={
                                    "policy_stop": {
                                        "reason": request.reason,
                                        "details": request.details,
                                    },
                                    "env_steps": steps,
                                },
                            ).as_dict()
                        )
                        continue
                    if operation != "step" or env.end_flag[0]:
                        raise ValueError("RoboDojo episode is not executable")
                    action = Action.from_dict(message)
                    if (action.benchmark, action.episode_id, action.step_index) != (
                        "robodojo",
                        episode_id,
                        steps,
                    ):
                        raise ValueError("RoboDojo action identity or sequence differs")
                    Draft202012Validator(ACTION_SCHEMA).validate(action.data)
                    json.dumps(action.data, allow_nan=False)
                    env.take_action(action.data)
                    steps += 1
                    if env.take_action_cnt != [steps] or env.env_seeds != [task.seed]:
                        raise ValueError("RoboDojo native counter or layout differs")
                    if env.unstable_envs:
                        raise ValueError("RoboDojo excluded an unstable layout")
                    done = bool(env.end_flag[0])
                    success = done and bool(env.success[0])
                    limit = done and not success and steps >= task.max_steps
                    last_observation = self._observation(
                        task, episode_id, steps, env.get_obs()
                    )
                    future.set_result(
                        StepResult(
                            observation=last_observation,
                            terminated=done and not limit,
                            truncated=limit,
                            termination_reason=(
                                "episode_limit"
                                if limit
                                else "native_success"
                                if success
                                else "native_failure"
                            )
                            if done
                            else None,
                        ).as_dict()
                    )
                except StopEpisode:
                    raise
                except BaseException as exc:
                    future.set_exception(exc)
                    raise

        try:
            native = self.runtime.run(specification, interact)
            if pending_finish is None or actual is None:
                raise ValueError("RoboDojo evaluation returned before finish")
            path = Path(native["source"])
            source = path.resolve().relative_to(self.directory.resolve()).as_posix()
            # This is the native evaluator's complete saved result, not Agent output.
            raw = json.loads(path.read_text())
            rows = list(raw["details"].values())
            if (
                len(rows) != 1
                or rows[0]["layout_id"] != actual.seed
                or raw["eval_time"] != 1
            ):
                raise ValueError("RoboDojo must score exactly the requested layout")
            row = rows[0]
            result = NativeResult(
                task=actual,
                episode_id=episode_id,
                score=row["score"],
                success=row["success"],
                details=raw,
                source=source,
                source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            pending_finish.set_result(result.as_dict())
        except StopEpisode:
            if pending_close is not None:
                pending_close.set_result({})
        except BaseException as exc:
            target = pending_finish if pending_finish is not None else initial
            if not target.done():
                target.set_exception(exc)

    def close(self) -> None:
        self._shutdown_event.set()
        self.requests.put(("close", {}, Future()))

    def run(self) -> None:
        """Consume native work on the calling thread, one episode per process."""
        while not self._shutdown_event.is_set():
            try:
                method, payload, future = self.requests.get(timeout=0.1)
            except queue.Empty:
                continue
            if method == "close":
                future.set_result({})
            elif method not in {"start", "inspect"} or self._started:
                future.set_exception(ValueError("RoboDojo process accepts one episode"))
            else:
                self._started = True
                self._episode(method, payload, future)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--factory", required=True)
    args = parser.parse_args()
    worker = EpisodeWorker(args.directory, load_factory(args.factory)())

    def parent_died():
        worker.close()
        timer = threading.Timer(5, lambda: os._exit(1))
        timer.daemon = True
        timer.start()

    watch_parent_death(parent_died)
    threading.Thread(
        target=worker.serve,
        kwargs={
            "transport": "http",
            "host": "127.0.0.1",
            "port": args.port,
            "parent_watch": False,
        },
        daemon=True,
    ).start()
    try:
        worker.run()
    finally:
        close = getattr(worker.runtime, "close", None)
        if close is not None:
            close()


if __name__ == "__main__":
    main()
