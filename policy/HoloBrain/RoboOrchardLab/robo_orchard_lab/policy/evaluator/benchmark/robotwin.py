# Project RoboOrchard
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

from __future__ import annotations
import copy
import os
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, cast

import torch
from pydantic import Field
from robo_orchard_core.utils.config import (
    ClassConfig,
    ClassType,
    Config,
    ConfigInstanceOf,
)
from robo_orchard_core.utils.logging import LoggerManager
from robo_orchard_core.utils.ray import RayRemoteClassConfig

from robo_orchard_lab.envs.robotwin import (
    RoboTwinEnvCfg,
    RoboTwinEnvStepReturn,
)
from robo_orchard_lab.envs.robotwin.env import (
    EVAL_INSTRUCTION_NUM,
    config_robotwin_path,
)
from robo_orchard_lab.policy.base import PolicyConfig, PolicyMixin
from robo_orchard_lab.policy.evaluator.base import PolicyEvaluatorConfig
from robo_orchard_lab.policy.evaluator.benchmark.backend import (
    LocalBenchmarkBackend,
    LocalBenchmarkBackendConfig,
    RemoteBenchmarkBackend,
    RemoteBenchmarkBackendConfig,
)
from robo_orchard_lab.policy.evaluator.benchmark.core import (
    BenchmarkAttemptError,
    BenchmarkAttemptRequest,
    BenchmarkDriver,
    BenchmarkEpisode,
    BenchmarkEpisodeRecord,
    BenchmarkEvaluateFailedEvent,
    BenchmarkEvaluateSucceededEvent,
    BenchmarkPrepareFailedEvent,
    BenchmarkPrepareJob,
    BenchmarkPrepareSucceededEvent,
    BenchmarkResult,
    BenchmarkTerminalEvent,
)
from robo_orchard_lab.policy.evaluator.metric_contracts import (
    EvaluatorMetrics,
)
from robo_orchard_lab.utils.state import State, StateSaveLoadMixin

__all__ = [
    "RoboTwinBenchmarkDriver",
    "RoboTwinBenchmarkEvaluator",
    "RoboTwinBenchmarkEvaluatorCfg",
    "RoboTwinLocalBenchmarkBackendCfg",
    "RoboTwinRemoteBenchmarkBackendCfg",
    "SEM_TASKS_16",
    "SuccessRateInfo",
    "SuccessRateMetric",
]

logger = LoggerManager().get_child(__name__)


SEM_TASKS_16 = (
    "adjust_bottle",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "dump_bin_bigbin",
    "handover_mic",
    "lift_pot",
    "move_pillbottle_pad",
    "open_laptop",
    "open_microwave",
    "place_cans_plasticbox",
    "place_dual_shoes",
    "place_empty_cup",
    "rotate_qrcode",
    "stack_blocks_three",
    "stack_bowls_three",
)


@dataclass
class SuccessRateInfo:
    """Aggregate RoboTwin success counts for one task.

    ``SuccessRateMetric`` stores one instance per task name. The record keeps
    both the counters used for aggregate benchmark metrics and the per-seed
    info list used for final episode records and debugging.
    """

    task_name: str
    """RoboTwin task name represented by this aggregate."""

    success_count: int
    """Number of successful terminal episodes for this task."""

    total_count: int
    """Number of terminal episodes observed for this task."""

    info_list: list[dict[str, Any]]
    """Per-episode seed and success annotations retained for reporting."""

    def success_rate(self) -> float:
        if self.total_count == 0:
            return 0.0
        return float(self.success_count) / self.total_count

    def merge(
        self,
        infos: Iterable[SuccessRateInfo],
    ) -> None:
        if not infos:
            raise ValueError("No SuccessRateInfo to merge.")
        ret = self
        for info in infos:
            if ret.task_name != info.task_name:
                raise ValueError(
                    "Cannot merge SuccessRateInfo with different task names."
                )
            ret.success_count += info.success_count
            ret.total_count += info.total_count
            ret.info_list.extend(info.info_list)

    def summary(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "success_count": self.success_count,
            "total_count": self.total_count,
            "success_rate": self.success_rate(),
        }


class SuccessRateMetric(StateSaveLoadMixin):
    """Metric that records terminal RoboTwin task success rates.

    The evaluator updates this metric only at terminal evaluator timing. That
    terminal boundary may come from RoboTwin ``terminated`` / ``truncated``
    flags or from evaluator-level limits such as ``max_steps``. The metric
    expects ``RoboTwinEnvStepReturn.info`` to include task and seed metadata,
    then records one success/failure sample for the task. The metric supports
    ``State`` save/load so remote benchmark workers can return snapshots for
    driver-side aggregation after each attempt.
    """

    info: dict[str, SuccessRateInfo]
    """Per-task success aggregates keyed by RoboTwin task name."""

    last_update_info: dict | None
    """Most recent per-episode success annotation, if any."""

    def __init__(self) -> None:
        self.info = {}
        self.last_update_info = None

    def reset(self, **kwargs: Any) -> None:
        self.info.clear()
        self.last_update_info = None

    def update(self, action: Any, step_return: RoboTwinEnvStepReturn) -> None:
        del action
        if step_return.info is None:
            raise ValueError(
                "RoboTwin success-rate metric requires step_return.info."
            )
        required_info_keys = (
            "task",
            "seed",
            "start_seed",
            "resolved_start_seed",
            "offset_seed",
        )
        missing_info_keys = [
            key for key in required_info_keys if key not in step_return.info
        ]
        if missing_info_keys:
            raise ValueError(
                "RoboTwin success-rate metric requires step_return.info "
                "to include keys "
                f"{required_info_keys}. Missing: {missing_info_keys}."
            )

        task_name: str = step_return.info["task"]
        success = bool(step_return.rewards)
        if task_name not in self.info:
            self.info[task_name] = SuccessRateInfo(
                task_name=task_name,
                success_count=0,
                total_count=0,
                info_list=[],
            )
        task_info = self.info[task_name]
        task_info.total_count += 1
        if success:
            task_info.success_count += 1
        seed_info = {
            "task_name": task_name,
            "seed": step_return.info["seed"],
            "start_seed": step_return.info["start_seed"],
            "resolved_start_seed": step_return.info["resolved_start_seed"],
            "offset_seed": step_return.info["offset_seed"],
            "success": success,
        }
        task_info.info_list.append(seed_info)
        self.last_update_info = seed_info

    def compute(self) -> dict[str, Any]:
        success_rates: list[float] = []
        summaries = []
        for _, info in self.info.items():
            success_rates.append(info.success_rate())
            summaries.append(info.summary())

        average_success_rate = (
            sum(success_rates) / len(success_rates) if success_rates else 0.0
        )
        return {
            "tasks": summaries,
            "average_success_rate": average_success_rate,
            "last_update": self.last_update_info,
        }

    def _get_state(self) -> dict[str, object]:
        return {
            "info": {
                task_name: {
                    "task_name": info.task_name,
                    "success_count": info.success_count,
                    "total_count": info.total_count,
                    "info_list": copy.deepcopy(info.info_list),
                }
                for task_name, info in self.info.items()
            },
            "last_update_info": copy.deepcopy(self.last_update_info),
        }

    def _set_state(self, state: State) -> None:
        payload = state.state
        if not isinstance(payload, dict):
            raise TypeError(
                "SuccessRateMetric state payload must be a dict. "
                f"Got {type(payload).__name__}."
            )

        serialized_info = payload.get("info")
        if not isinstance(serialized_info, dict):
            raise TypeError(
                "SuccessRateMetric state field `info` must be a "
                f"dict. Got {type(serialized_info).__name__}."
            )

        restored_info: dict[str, SuccessRateInfo] = {}
        for task_name, task_state in serialized_info.items():
            if isinstance(task_state, SuccessRateInfo):
                restored_info[task_name] = copy.deepcopy(task_state)
                continue
            if not isinstance(task_state, dict):
                raise TypeError(
                    "SuccessRateMetric task state must be a dict or "
                    f"SuccessRateInfo. Got {type(task_state).__name__} for "
                    f"task `{task_name}`."
                )
            restored_info[task_name] = SuccessRateInfo(
                task_name=task_state.get("task_name", task_name),
                success_count=task_state["success_count"],
                total_count=task_state["total_count"],
                info_list=copy.deepcopy(task_state.get("info_list", [])),
            )

        last_update_info = payload.get("last_update_info")
        if last_update_info is not None and not isinstance(
            last_update_info, dict
        ):
            raise TypeError(
                "SuccessRateMetric state field `last_update_info` "
                "must be a dict or None. Got "
                f"{type(last_update_info).__name__}."
            )

        self.info = restored_info
        self.last_update_info = copy.deepcopy(last_update_info)

    def to(self, *args: Any, **kwargs: Any) -> None:
        pass

    def merge(
        self,
        metrics: Iterable[SuccessRateMetric],
    ) -> None:
        ret = self
        for metric in metrics:
            for task_name, info in metric.info.items():
                if task_name not in ret.info:
                    ret.info[task_name] = info
                else:
                    ret.info[task_name].merge([info])


@dataclass(slots=True)
class _RoboTwinTaskState:
    task_name: str
    next_episode_id: int = 0
    next_offset_seed: int = 0
    prepare_active: bool = False
    retry_queue: deque[BenchmarkPrepareJob] = field(default_factory=deque)


@dataclass(slots=True)
class _RoboTwinAttemptResetContext:
    env_cfg: RoboTwinEnvCfg
    env_reset_input: dict[str, Any]
    offset_seed: int


@dataclass(slots=True)
class _RoboTwinEpisodeState:
    episode: BenchmarkEpisode
    attempt_index: int = 0
    attempt_errors: list[dict[str, Any]] = field(default_factory=list)
    retry_reset_context: _RoboTwinAttemptResetContext | None = None


class RoboTwinBenchmarkDriver(BenchmarkDriver):
    """Own RoboTwin-specific benchmark readiness, retry, and aggregation.

    The generic backend owns remote workers and the prepare/evaluate pipeline.
    This driver owns logical RoboTwin episodes, per-task reset offset
    advancement, bounded retry, artifact paths, and aggregate success-rate
    metric merging. Backend callbacks are expected to be invoked serially.

    Ready job scheduling is task-major: every ``get_ready_jobs(...)`` call
    scans ``cfg.task_names`` from first to last, returns at most one prepare
    job per task, and gives earlier task names priority whenever they have
    ready retry or new episode work. Retry jobs for a task are emitted before
    new logical episodes for that same task.
    """

    def __init__(self, cfg: RoboTwinBenchmarkEvaluatorCfg) -> None:
        self.cfg = cfg
        self._task_order = tuple(cfg.task_names)
        self._task_states = {
            task_name: _RoboTwinTaskState(task_name=task_name)
            for task_name in self._task_order
        }
        self._episode_states: dict[str, _RoboTwinEpisodeState] = {}
        self._records: list[BenchmarkEpisodeRecord] = []
        self._metric = SuccessRateMetric()
        self._completion_logged = False

    def has_unfinished_work(self) -> bool:
        """Return whether any logical episode or retry remains unfinished."""

        if self._episode_states:
            return True
        return any(
            task_state.next_episode_id < self.cfg.episode_num
            or task_state.prepare_active
            or bool(task_state.retry_queue)
            for task_state in self._task_states.values()
        )

    def get_ready_jobs(
        self,
        max_jobs: int | None = None,
    ) -> Sequence[BenchmarkPrepareJob]:
        """Return ready jobs in ``task_names`` priority order.

        The driver scans tasks from first to last on every call. A task whose
        prepare is already active is skipped, which preserves RoboTwin's
        per-task prepare ordering while still allowing prepared rollouts to
        execute concurrently in the backend. For each ready task, pending
        retry work is returned before a new logical episode. ``max_jobs``
        limits the number of tasks selected by this call.
        """

        if max_jobs is not None and max_jobs <= 0:
            return []

        jobs: list[BenchmarkPrepareJob] = []
        for task_name in self._task_order:
            if max_jobs is not None and len(jobs) >= max_jobs:
                break
            task_state = self._task_states[task_name]
            if task_state.prepare_active:
                continue

            job = self._pop_ready_job(task_state)
            if job is None:
                continue

            task_state.prepare_active = True
            jobs.append(job)
        return jobs

    def make_attempt_request(
        self,
        job: BenchmarkPrepareJob,
    ) -> BenchmarkAttemptRequest:
        """Allocate RoboTwin reset inputs once a worker slot is reserved."""

        task_name = job.episode.group_key
        episode_state = self._episode_states[job.episode.episode_key]
        if episode_state.retry_reset_context is not None:
            return self._make_pinned_attempt_request(
                job,
                episode_state.retry_reset_context,
            )

        task_state = self._task_states[task_name]
        attempted_offset_seed = task_state.next_offset_seed
        env_reset_input = self._make_reset_input(
            episode=job.episode,
            offset_seed=attempted_offset_seed,
        )

        return BenchmarkAttemptRequest(
            episode=job.episode,
            attempt_index=job.attempt_index,
            env_cfg=self._make_env_cfg(task_name),
            env_reset_input=env_reset_input,
            max_steps=job.max_steps,
            policy_reset_input=job.policy_reset_input,
            metric_reset_input=job.metric_reset_input,
            metadata={
                **job.metadata,
                "attempted_offset_seed": attempted_offset_seed,
                "reset_context_pinned": False,
            },
        )

    def on_attempt_prepared(
        self,
        event: BenchmarkPrepareSucceededEvent,
    ) -> None:
        """Update task seed bookkeeping after a successful RoboTwin reset.

        Fresh prepares advance the seed frontier. Pinned retry prepares reuse
        the captured reset context and only clear the active-prepare marker.
        """

        task_state = self._task_states[event.request.episode.group_key]
        if self._request_reset_context_pinned(event.request):
            task_state.prepare_active = False
            return

        actual_offset_seed = event.reset_info["offset_seed"]
        task_state.next_offset_seed = max(
            task_state.next_offset_seed,
            int(actual_offset_seed) + 1,
        )
        task_state.prepare_active = False

    def on_terminal_event(
        self,
        event: BenchmarkTerminalEvent,
    ) -> None:
        """Handle retry, final records, and metric aggregation."""

        episode = event.request.episode
        episode_state = self._episode_states[episode.episode_key]
        task_state = self._task_states[episode.group_key]
        if self.cfg.fail_fast and isinstance(
            event,
            (BenchmarkPrepareFailedEvent, BenchmarkEvaluateFailedEvent),
        ):
            raise BenchmarkAttemptError(event)

        if isinstance(
            event,
            (BenchmarkPrepareFailedEvent, BenchmarkEvaluateFailedEvent),
        ):
            self._request_reset_context_pinned(event.request)

        if isinstance(event, BenchmarkPrepareFailedEvent):
            task_state.prepare_active = False

        if isinstance(event, BenchmarkEvaluateSucceededEvent):
            worker_metric = event.worker_metrics.get_metric("success_rate")
            if not isinstance(worker_metric, SuccessRateMetric):
                raise TypeError(
                    "Worker metrics must include a SuccessRateMetric named "
                    f"'success_rate'. Got {type(worker_metric).__name__}."
                )
            self._metric.merge([worker_metric])
            self._record_episode_success(event, worker_metric, episode_state)
            self._episode_states.pop(episode.episode_key, None)
            return

        failure_event = cast(
            BenchmarkPrepareFailedEvent | BenchmarkEvaluateFailedEvent,
            event,
        )
        self._append_attempt_error(episode_state, failure_event)
        if episode_state.attempt_index < self.cfg.max_retries:
            if self.cfg.log_progress:
                self._log_attempt_retry(failure_event)
            if isinstance(failure_event, BenchmarkEvaluateFailedEvent):
                episode_state.retry_reset_context = (
                    self._make_retry_reset_context(failure_event)
                )
            episode_state.attempt_index += 1
            task_state.retry_queue.append(self._make_job(episode_state))
            return

        self._record_episode_failure(failure_event, episode_state)
        self._episode_states.pop(episode.episode_key, None)

    def result(self) -> BenchmarkResult:
        """Return aggregate success-rate metrics and final episode records."""

        result = BenchmarkResult(
            metrics=self._metric.compute(),
            episodes=list(self._records),
            metadata={
                "task_names": list(self._task_order),
                "episode_num": self.cfg.episode_num,
                "max_retries": self.cfg.max_retries,
                "config_type": self.cfg.config_type,
                "start_seed": self.cfg.start_seed,
            },
        )
        if self.cfg.log_progress and not self.has_unfinished_work():
            self._log_benchmark_completed(result)
        return result

    def make_worker_metrics(self) -> EvaluatorMetrics:
        """Build the per-worker metric surface used by remote evaluators."""

        return EvaluatorMetrics.from_metric(
            SuccessRateMetric(),
            name="success_rate",
        )

    def _pop_ready_job(
        self,
        task_state: _RoboTwinTaskState,
    ) -> BenchmarkPrepareJob | None:
        if task_state.retry_queue:
            return task_state.retry_queue.popleft()
        if task_state.next_episode_id >= self.cfg.episode_num:
            return None

        episode_id = task_state.next_episode_id
        task_state.next_episode_id += 1
        episode = BenchmarkEpisode(
            episode_key=f"{task_state.task_name}/episode-{episode_id}",
            group_key=task_state.task_name,
            episode_id=episode_id,
            metadata={
                "task_name": task_state.task_name,
                "config_type": self.cfg.config_type,
            },
        )
        episode_state = _RoboTwinEpisodeState(episode=episode)
        self._episode_states[episode.episode_key] = episode_state
        return self._make_job(episode_state)

    def _make_job(
        self,
        episode_state: _RoboTwinEpisodeState,
    ) -> BenchmarkPrepareJob:
        return BenchmarkPrepareJob(
            episode=episode_state.episode,
            attempt_index=episode_state.attempt_index,
            max_steps=self.cfg.max_steps,
            metadata={
                "task_name": episode_state.episode.group_key,
                "config_type": self.cfg.config_type,
            },
        )

    def _make_pinned_attempt_request(
        self,
        job: BenchmarkPrepareJob,
        reset_context: _RoboTwinAttemptResetContext,
    ) -> BenchmarkAttemptRequest:
        return BenchmarkAttemptRequest(
            episode=job.episode,
            attempt_index=job.attempt_index,
            env_cfg=copy.deepcopy(reset_context.env_cfg),
            env_reset_input=copy.deepcopy(reset_context.env_reset_input),
            max_steps=job.max_steps,
            policy_reset_input=job.policy_reset_input,
            metric_reset_input=job.metric_reset_input,
            metadata={
                **job.metadata,
                "attempted_offset_seed": reset_context.offset_seed,
                "reset_context_pinned": True,
            },
        )

    def _make_env_cfg(self, task_name: str) -> RoboTwinEnvCfg:
        task_config_path = os.path.join(
            config_robotwin_path(),
            "task_config",
            f"{self.cfg.config_type}.yml",
        )
        if not os.path.exists(task_config_path):
            raise FileNotFoundError(
                f"Task config file not found: {task_config_path}"
            )

        return RoboTwinEnvCfg(
            task_name=task_name,
            check_expert=True,
            check_task_init=False,
            eval_mode=True,
            max_instruction_num=EVAL_INSTRUCTION_NUM,
            format_datatypes=self.cfg.format_datatypes,
            action_type=self.cfg.action_type,
            task_config_path=task_config_path,
            seed=self.cfg.start_seed,
        )

    def _make_reset_input(
        self,
        *,
        episode: BenchmarkEpisode,
        offset_seed: int,
    ) -> dict[str, Any]:
        video_dir = None
        if self.cfg.artifact_root_dir is not None:
            video_dir = os.path.join(
                self.cfg.artifact_root_dir,
                episode.group_key,
                self.cfg.config_type,
            )

        return {
            "offset_seed": offset_seed,
            "task_name": episode.group_key,
            "episode_id": episode.episode_id,
            "clear_cache": True,
            "return_obs": True,
            "video_dir": video_dir,
        }

    def _request_reset_context_pinned(
        self,
        request: BenchmarkAttemptRequest,
    ) -> bool:
        try:
            pinned = request.metadata["reset_context_pinned"]
        except KeyError as exc:
            raise RuntimeError(
                "RoboTwin benchmark request metadata must include "
                "`reset_context_pinned`."
            ) from exc
        if not isinstance(pinned, bool):
            raise RuntimeError(
                "RoboTwin benchmark request metadata "
                "`reset_context_pinned` must be a bool. Got "
                f"{type(pinned).__name__}."
            )
        return pinned

    def _make_retry_reset_context(
        self,
        event: BenchmarkEvaluateFailedEvent,
    ) -> _RoboTwinAttemptResetContext:
        env_reset_input = copy.deepcopy(event.request.env_reset_input)
        if not isinstance(env_reset_input, dict):
            raise TypeError(
                "RoboTwin benchmark retry reset context requires dict "
                "env_reset_input. Got "
                f"{type(env_reset_input).__name__}."
            )
        offset_seed = int(
            event.reset_info.get(
                "offset_seed",
                event.request.metadata["attempted_offset_seed"],
            )
        )
        env_reset_input["offset_seed"] = offset_seed
        return _RoboTwinAttemptResetContext(
            env_cfg=copy.deepcopy(cast(RoboTwinEnvCfg, event.request.env_cfg)),
            env_reset_input=env_reset_input,
            offset_seed=offset_seed,
        )

    def _record_episode_success(
        self,
        event: BenchmarkEvaluateSucceededEvent,
        worker_metric: SuccessRateMetric,
        episode_state: _RoboTwinEpisodeState,
    ) -> None:
        last_update = worker_metric.last_update_info
        if not isinstance(last_update, dict):
            raise RuntimeError(
                "SuccessRateMetric did not record last_update_info for "
                f"episode {event.request.episode.episode_key}."
            )
        episode_metrics = dict(event.episode_metrics)
        episode_metrics.setdefault("success_rate", worker_metric.compute())
        self._records.append(
            BenchmarkEpisodeRecord(
                episode=event.request.episode,
                succeeded=bool(last_update.get("success", False)),
                attempts=event.request.attempt_index + 1,
                episode_metrics=episode_metrics,
                attempt_errors=list(episode_state.attempt_errors),
            )
        )
        if self.cfg.log_progress:
            self._log_episode_completed(
                event.request.episode,
                succeeded=bool(last_update.get("success", False)),
                attempts=event.request.attempt_index + 1,
                offset_seed=self._event_offset_seed(event),
            )

    def _append_attempt_error(
        self,
        episode_state: _RoboTwinEpisodeState,
        event: BenchmarkPrepareFailedEvent | BenchmarkEvaluateFailedEvent,
    ) -> None:
        phase = (
            "prepare"
            if isinstance(event, BenchmarkPrepareFailedEvent)
            else "evaluate"
        )
        episode_state.attempt_errors.append(
            {
                "attempt_index": event.request.attempt_index,
                "phase": phase,
                "attempted_offset_seed": event.request.metadata.get(
                    "attempted_offset_seed"
                ),
                "reset_info": dict(event.reset_info),
                "error_type": event.error_type,
                "error_message": event.error_message,
            }
        )

    def _record_episode_failure(
        self,
        event: BenchmarkPrepareFailedEvent | BenchmarkEvaluateFailedEvent,
        episode_state: _RoboTwinEpisodeState,
    ) -> None:
        self._record_failed_metric(event)
        self._records.append(
            BenchmarkEpisodeRecord(
                episode=event.request.episode,
                succeeded=False,
                attempts=event.request.attempt_index + 1,
                error_type=event.error_type,
                error_message=event.error_message,
                attempt_errors=list(episode_state.attempt_errors),
            )
        )
        if self.cfg.log_progress:
            self._log_episode_completed(
                event.request.episode,
                succeeded=False,
                attempts=event.request.attempt_index + 1,
                offset_seed=self._event_offset_seed(event),
            )

    def _record_failed_metric(
        self,
        event: BenchmarkPrepareFailedEvent | BenchmarkEvaluateFailedEvent,
    ) -> None:
        task_name = event.request.episode.group_key
        task_info = self._metric.info.get(task_name)
        if task_info is None:
            task_info = SuccessRateInfo(
                task_name=task_name,
                success_count=0,
                total_count=0,
                info_list=[],
            )
            self._metric.info[task_name] = task_info

        failure_info = {
            "task_name": task_name,
            "episode_id": event.request.episode.episode_id,
            "attempt_index": event.request.attempt_index,
            "attempted_offset_seed": event.request.metadata.get(
                "attempted_offset_seed"
            ),
            "offset_seed": event.reset_info.get("offset_seed"),
            "success": False,
            "error_type": event.error_type,
            "error_message": event.error_message,
        }
        task_info.total_count += 1
        task_info.info_list.append(failure_info)
        self._metric.last_update_info = failure_info

    def _log_attempt_retry(
        self,
        event: BenchmarkPrepareFailedEvent | BenchmarkEvaluateFailedEvent,
    ) -> None:
        phase = (
            "prepare"
            if isinstance(event, BenchmarkPrepareFailedEvent)
            else "evaluate"
        )
        episode = event.request.episode
        logger.warning(
            "RoboTwin benchmark attempt failed, retrying: "
            "task=%s episode=%s attempt=%s/%s phase=%s offset_seed=%s "
            "error=%s",
            episode.group_key,
            episode.episode_id,
            event.request.attempt_index + 1,
            self.cfg.max_retries + 1,
            phase,
            self._event_offset_seed(event),
            self._format_event_error(event),
        )

    def _format_event_error(
        self,
        event: BenchmarkPrepareFailedEvent | BenchmarkEvaluateFailedEvent,
    ) -> str:
        error_type = event.error_type or type(event).__name__
        if event.error_message:
            return f"{error_type}: {event.error_message}"
        return error_type

    def _log_episode_completed(
        self,
        episode: BenchmarkEpisode,
        *,
        succeeded: bool,
        attempts: int,
        offset_seed: object,
    ) -> None:
        task_info = self._metric.info.get(episode.group_key)
        task_success_rate = (
            task_info.success_rate() if task_info is not None else 0.0
        )
        task_completed = sum(
            1
            for record in self._records
            if record.episode.group_key == episode.group_key
        )
        total_completed = len(self._records)
        total_episode_num = len(self._task_order) * self.cfg.episode_num
        if (
            succeeded
            and total_completed != total_episode_num
            and total_completed % self.cfg.progress_log_every_n_episodes != 0
        ):
            return

        logger.info(
            "RoboTwin benchmark episode completed: "
            "task=%s episode=%s success=%s attempts=%s offset_seed=%s "
            "task_success_rate=%.3f task_progress=%s/%s "
            "total_progress=%s/%s",
            episode.group_key,
            episode.episode_id,
            succeeded,
            attempts,
            offset_seed,
            task_success_rate,
            task_completed,
            self.cfg.episode_num,
            total_completed,
            total_episode_num,
        )

    def _log_benchmark_completed(self, result: BenchmarkResult) -> None:
        if self._completion_logged:
            return
        self._completion_logged = True
        logger.info(
            "RoboTwin benchmark completed: tasks=%s episodes=%s "
            "average_success_rate=%.3f",
            len(self._task_order),
            len(self._task_order) * self.cfg.episode_num,
            float(result.metrics["average_success_rate"]),
        )

    def _event_offset_seed(
        self,
        event: (
            BenchmarkPrepareFailedEvent
            | BenchmarkEvaluateSucceededEvent
            | BenchmarkEvaluateFailedEvent
        ),
    ) -> object:
        return event.reset_info.get(
            "offset_seed",
            event.request.metadata.get("attempted_offset_seed"),
        )


class RoboTwinBenchmarkEvaluator:
    """Run a complete RoboTwin benchmark for one policy.

    Use this evaluator when a policy already fits the
    :class:`~robo_orchard_lab.policy.base.PolicyMixin` /
    :class:`~robo_orchard_lab.policy.base.PolicyConfig` evaluation surface and
    should be measured over one or more RoboTwin tasks. The evaluator is the
    user-facing benchmark entrypoint: callers provide a policy, and receive a
    :class:`BenchmarkResult` containing aggregate success-rate metrics and one
    final record per logical RoboTwin episode.

    Compared with RoboTwin's official local single-worker evaluator, this
    class keeps benchmark orchestration reusable and testable. It uses the
    shared benchmark backend pipeline for prepared reset-before-rollout
    execution and cleanup. The default local backend runs one current-process
    worker for debugging and smoke tests; the remote backend adds Ray workers,
    timeouts, and worker replacement. RoboTwin domain policy remains in
    :class:`RoboTwinBenchmarkDriver`: task readiness, bounded retry,
    offset-seed advancement, artifact paths, and success-rate aggregation.

    The evaluator does not load policy-specific checkpoints through RoboTwin's
    deploy-policy module conventions. Those concerns belong to the policy
    construction path. It also does not hide RoboTwin installation
    requirements: RoboTwin must be installed, and the environment wrapper must
    be able to resolve the configured RoboTwin task names and task config
    files.

    Example::

        cfg = RoboTwinBenchmarkEvaluatorCfg(
            task_names=["<robotwin_task>"],
            episode_num=100,
            artifact_root_dir="outputs/robotwin_eval",
        )
        result = cfg().evaluate(policy)
        success_rate = result.metrics["average_success_rate"]
    """

    cfg: RoboTwinBenchmarkEvaluatorCfg
    InitFromConfig: bool = True

    def __init__(self, cfg: RoboTwinBenchmarkEvaluatorCfg) -> None:
        self.cfg = cfg

    def evaluate(
        self,
        policy_or_cfg: PolicyConfig | PolicyMixin,
        *,
        device: str | torch.device | None = None,
    ) -> BenchmarkResult:
        """Run the configured RoboTwin benchmark and return final results."""

        driver = RoboTwinBenchmarkDriver(self.cfg)
        backend_cfg = self.cfg.backend
        evaluator_cfg = PolicyEvaluatorConfig(
            reconfigure_env_force_recreate=False,
        )
        if isinstance(backend_cfg, RoboTwinLocalBenchmarkBackendCfg):
            with LocalBenchmarkBackend(
                LocalBenchmarkBackendConfig(
                    evaluator_cfg=evaluator_cfg,
                    policy_or_cfg=policy_or_cfg,
                    worker_metrics_factory=driver.make_worker_metrics,
                    device=device,
                )
            ) as backend:
                return backend.run(driver)

        if isinstance(backend_cfg, RoboTwinRemoteBenchmarkBackendCfg):
            remote_cfg = evaluator_cfg.as_remote(
                remote_class_config=backend_cfg.remote_class_config,
                ray_init_config=backend_cfg.ray_init_config,
                rollout_timeout_s=backend_cfg.rollout_timeout_s,
                reset_timeout_s=backend_cfg.reset_timeout_s,
                timeout_grace_retries=backend_cfg.timeout_grace_retries,
            )
            with RemoteBenchmarkBackend(
                RemoteBenchmarkBackendConfig(
                    remote_cfg=remote_cfg,
                    num_workers=backend_cfg.num_parallel_envs,
                    policy_or_cfg=policy_or_cfg,
                    worker_metrics_factory=driver.make_worker_metrics,
                    device=device,
                )
            ) as backend:
                return backend.run(driver)

        raise TypeError(
            "Unsupported RoboTwin benchmark backend config: "
            f"{type(backend_cfg).__name__}."
        )


class RoboTwinLocalBenchmarkBackendCfg(Config):
    """Select the current-process single-worker RoboTwin backend."""


class RoboTwinRemoteBenchmarkBackendCfg(Config):
    """Configure the remote Ray-backed RoboTwin benchmark backend."""

    num_parallel_envs: int = Field(default=1, ge=1)
    """Maximum number of remote RoboTwin environments in flight."""

    rollout_timeout_s: float | None = Field(default=120.0, gt=0)
    """Per-call timeout for rollout/evaluate calls; ``None`` disables it."""

    reset_timeout_s: float | None = Field(default=1200.0, gt=0)
    """Per-call timeout for setup, reconfigure, reset, and metric calls."""

    timeout_grace_retries: int = Field(default=1, ge=0)
    """Extra waits after a remote timeout before replacing the worker."""

    remote_class_config: RayRemoteClassConfig = Field(
        default_factory=lambda: RayRemoteClassConfig(
            num_cpus=8,
            num_gpus=1,
            memory=16 * 1024**3,
        )
    )
    """Ray actor resource request for each remote policy evaluator worker."""

    ray_init_config: dict[str, Any] | None = None
    """Optional Ray config forwarded to remote evaluator setup."""


class RoboTwinBenchmarkEvaluatorCfg(ClassConfig[RoboTwinBenchmarkEvaluator]):
    """Configure :class:`RoboTwinBenchmarkEvaluator`.

    This is the main user-facing contract for RoboTwin benchmark evaluation.
    It describes which tasks to run, how many final logical episodes to record
    per task, how retries advance RoboTwin offset seeds, which backend runs
    the worker pipeline, and where optional artifacts should be written.

    Failed attempts are retried up to ``max_retries``. When all retries are
    exhausted, the logical episode is still counted in ``episode_num`` and the
    evaluator advances to the next logical episode. Set ``fail_fast=True`` to
    raise :class:`BenchmarkAttemptError` on the first infrastructure failure
    instead of retrying or returning partial benchmark results.

    Runtime details live under ``backend``. The default
    :class:`RoboTwinLocalBenchmarkBackendCfg` runs one current-process worker
    without Ray or hard timeout isolation. Use
    :class:`RoboTwinRemoteBenchmarkBackendCfg` for Ray-backed parallel workers
    with reset and rollout per-call timeouts.
    """

    class_type: ClassType[RoboTwinBenchmarkEvaluator] = (
        RoboTwinBenchmarkEvaluator
    )

    task_names: list[str]
    """RoboTwin task names to evaluate.

    The list must be non-empty and contain no duplicates. Task names are
    resolved by the RoboTwin env at worker runtime, so config validation can
    run in environments where RoboTwin is not installed locally. The evaluator
    reports per-task success-rate metrics using these names as task identities.
    The order is also the driver's prepare scheduling priority: earlier tasks
    are selected first whenever they have ready work.
    """

    episode_num: int = Field(default=100, ge=0)
    """Number of final logical episodes to record per task.

    A logical episode is counted after success or after all retries fail. This
    mirrors benchmark accounting: a permanently failed episode still consumes
    one slot in the task's denominator.
    """

    max_retries: int = Field(default=3, ge=0)
    """Maximum retry count for each logical episode after a failed attempt.

    The maximum number of concrete attempts for one logical episode is
    ``max_retries + 1``. Prepare failures retry from the current task seed
    frontier without advancing it. Evaluate failures retry from a pinned copy
    of the failed attempt's reset context so remote parallel work cannot
    cause the retry to pick up a later episode's seed frontier. ``fail_fast``
    takes precedence and aborts before retry scheduling.
    """

    max_steps: int = Field(default=1500, gt=0)
    """Rollout step cap passed to each concrete evaluation attempt."""

    config_type: Literal["demo_clean", "demo_randomized"] = "demo_clean"
    """RoboTwin task config set loaded from ``task_config/<config_type>.yml``.

    Use ``"demo_clean"`` for the standard deterministic config set and
    ``"demo_randomized"`` when evaluating the randomized RoboTwin task config.
    """

    start_seed: int = 0
    """Caller-facing RoboTwin benchmark start seed.

    The env resolves this into RoboTwin's evaluation seed range. The driver
    then advances actual runtime coverage through per-task ``offset_seed``
    values returned by reset.
    """

    format_datatypes: bool = True
    """Whether RoboTwin formats observations into typed data objects."""

    action_type: Literal["qpos", "ee"] = "qpos"
    """Policy action representation expected by the RoboTwin env.

    ``"qpos"`` means joint target positions. ``"ee"`` means RoboTwin
    end-effector pose actions.
    """

    backend: (
        ConfigInstanceOf[RoboTwinLocalBenchmarkBackendCfg]
        | ConfigInstanceOf[RoboTwinRemoteBenchmarkBackendCfg]
    ) = Field(default_factory=RoboTwinLocalBenchmarkBackendCfg)
    """Backend runtime config selecting local or remote evaluation."""

    fail_fast: bool = False
    """Raise on first infrastructure failure instead of bounded retry."""

    log_progress: bool = True
    """Emit benchmark progress and retry logs through the module logger."""

    progress_log_every_n_episodes: int = Field(default=1, ge=1)
    """Log successful episode progress every N completed episodes.

    Retry warnings, failed terminal episodes, the final episode, and the final
    benchmark summary are still logged whenever ``log_progress`` is true.
    """

    artifact_root_dir: str | None = None
    """Optional root directory for RoboTwin episode artifacts such as videos.

    When set, reset receives the task/config artifact directory. The env owns
    the final video file name because the actual RoboTwin runtime seed is only
    known after reset.
    """

    def __post_init__(self) -> None:
        if not self.task_names:
            raise ValueError("RoboTwin benchmark requires at least one task.")
        if len(set(self.task_names)) != len(self.task_names):
            raise ValueError("RoboTwin benchmark task_names must be unique.")
