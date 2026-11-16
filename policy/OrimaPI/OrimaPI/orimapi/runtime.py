# coding=utf-8
"""A single-owner feedback loop with bounded background language-model calls."""

from __future__ import annotations

import math
import time
import uuid
from concurrent.futures import (
    FIRST_COMPLETED,
    Executor,
    Future,
    TimeoutError,
    wait,
)
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable

from .state import (
    Decision,
    DecisionKind,
    Mode,
    Observation,
    PolicySelection,
    SupervisorRequest,
    json_value,
)
from .trace import TraceWriter
from .supervisor import SupervisorEpisodeError, SupervisorExecutor
from .xpolicylab.evaluation import fast_trajectory_chunk_size


@dataclass(frozen=True)
class RuntimeConfig:
    mode: Mode = Mode.CLOSED_LOOP
    exec_horizon: int = 5
    # Legacy field name: completed chunks in blocking mode, action steps in async.
    replan_interval_steps: int = 25
    max_observation_age_steps: int = 50
    max_unreviewed_steps: int = 150
    supervisor_timeout_s: float = 30.0
    max_consecutive_failures: int = 3
    max_supervisor_calls: int = 100
    feedback_history_size: int = 8
    supervisor_workers: int = 1
    supervisor_schedule: str = "blocking"
    supervisor_retry_backoff_s: float = 1.0
    record_frames: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", Mode(self.mode))
        if type(self.record_frames) is not bool:
            raise ValueError("record_frames must be a boolean")
        for name in self.__dataclass_fields__:
            if name in {
                "mode",
                "supervisor_timeout_s",
                "supervisor_schedule",
                "supervisor_retry_backoff_s",
                "record_frames",
            }:
                continue
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            type(self.supervisor_timeout_s) not in (int, float)
            or not math.isfinite(self.supervisor_timeout_s)
            or self.supervisor_timeout_s <= 0
        ):
            raise ValueError("supervisor_timeout_s must be positive and finite")
        if self.supervisor_schedule not in {"blocking", "async"}:
            raise ValueError("supervisor_schedule must be blocking or async")
        if (
            type(self.supervisor_retry_backoff_s) not in (int, float)
            or not math.isfinite(self.supervisor_retry_backoff_s)
            or self.supervisor_retry_backoff_s < 0
        ):
            raise ValueError(
                "supervisor_retry_backoff_s must be nonnegative and finite"
            )


class Runtime:
    """Call observe/actions/reset serially from the evaluation adapter.

    Background workers return decisions only. They never mutate episode state
    or call the action model. Prediction tails are discarded after each call,
    so no unsubmitted actions survive into a later plan.
    """

    def __init__(
        self,
        *,
        policy=None,
        policy_library: dict | None = None,
        fixed_policy: str | None = None,
        supervisor,
        trace: TraceWriter,
        config: RuntimeConfig | None = None,
        executor: Executor | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.config = config or RuntimeConfig()
        self._blocking = (
            self.config.mode == Mode.CLOSED_LOOP
            and self.config.supervisor_schedule == "blocking"
        )
        if (policy is None) == (policy_library is None):
            raise ValueError(
                "Supply exactly one policy library or a fixed policy instance"
            )
        if fixed_policy is not None and policy_library is None:
            raise ValueError("fixed_policy requires a policy library")
        self.fixed_policy = fixed_policy
        self._selects_policy = policy_library is not None and fixed_policy is None
        self._definitions = (
            deepcopy(policy_library["candidates"]) if policy_library is not None else {}
        )
        self._factory = (
            policy_library["factory"] if policy_library is not None else None
        )
        self._candidates = {
            key: candidate.capability()
            for key, candidate in self._definitions.items()
            if self.config.mode == Mode.DIRECT or candidate.supports_subgoals
        }
        # None names a directly injected policy. Named candidates load only after selection.
        self._policies = {None: policy} if policy is not None else {}
        self._capabilities = {}
        if policy_library is not None and not self._candidates:
            raise ValueError(
                "No policy candidates support the configured supervision mode"
            )
        if fixed_policy is not None and fixed_policy not in self._candidates:
            raise ValueError("fixed_policy is unknown or incompatible with subgoals")
        needs_supervisor = self._selects_policy or self.config.mode != Mode.DIRECT
        if needs_supervisor and supervisor is None:
            raise ValueError(
                "A supervisor is required for policy selection and subgoal supervision"
            )
        self.supervisor, self.trace = supervisor, trace
        if policy_library is not None:
            trace.write(
                "policy_catalogued",
                candidates=self._candidates,
                fixed_policy=fixed_policy,
            )
        self._clock = clock
        self._executor = executor
        self._owns_executor = executor is None
        if executor is None and needs_supervisor:
            self._executor = SupervisorExecutor(self.config.supervisor_workers)
        self._episodes: dict[int, SimpleNamespace] = {}
        self._unbound_observations: dict[int, list[Observation]] = {}
        # Pending entries are (request, future, start time); retired entries keep
        # only the environment and running future until its worker is released.
        self._pending = {}
        self._retired: list[tuple[int, Future]] = []
        self._closed = False

    def state(self, env_id: int) -> SimpleNamespace:
        return deepcopy(self._episodes[env_id])

    def update_observation(self, observation: Observation) -> None:
        self._open()
        observation = deepcopy(observation)
        state = self._episodes.get(observation.env_id)
        if state is None:
            state = SimpleNamespace(
                env_id=observation.env_id,
                episode_id=uuid.uuid4().hex,
                task_goal=observation.instruction,
                latest=observation,
                selected_policy=None,
                policy_capability=None,
                active_subgoal=None,
                plan_version=0,
                completed_subgoals=[],
                recent_feedback=[],
                last_request_step=None,
                last_reviewed_step=None,
                last_reviewed_chunk=None,
                completed_action_chunks=0,
                next_action_step=None,
                supervisor_calls=0,
                consecutive_failures=0,
                submitted_actions=0,
                error=None,
                supervisor_error=False,
                retry_at=0.0,
            )
            if self.config.mode == Mode.DIRECT and not self._selects_policy:
                state.active_subgoal = state.task_goal
                state.plan_version = 1
            self._episodes[observation.env_id] = state
            self.trace.write(
                "episode_started",
                env_id=state.env_id,
                episode_id=state.episode_id,
                task_goal=state.task_goal,
                mode=self.config.mode.value,
            )
        else:
            if state.error:
                self._raise_episode_error(state)
            if observation.instruction != state.task_goal:
                raise ValueError(
                    "Original task instruction changed; reset the episode first"
                )
            if observation.step <= state.latest.step:
                raise ValueError(
                    "Observation steps must strictly increase within an episode"
                )
            state.latest = observation
        # A delivered chunk counts only after its final action is observed.
        if (
            state.next_action_step is not None
            and observation.step >= state.next_action_step
        ):
            state.completed_action_chunks += 1
            state.next_action_step = None
        try:
            if self._definitions and state.selected_policy is None:
                if self.fixed_policy is not None:
                    self._bind_policy(state, self.fixed_policy)
                else:
                    buffered = self._unbound_observations.setdefault(state.env_id, [])
                    if (
                        buffered
                        and observation.step - buffered[0].step
                        > self.config.max_observation_age_steps
                    ):
                        raise ValueError(
                            "Observations exceeded the freshness budget before policy selection"
                        )
                    buffered.append(observation)
            # Every physical observation reaches stateful skills, including frames
            # received between action-model inference calls.
            if state.selected_policy in self._policies:
                self._policies[state.selected_policy].observe(deepcopy(observation))
        except Exception as error:
            state.error = f"{type(error).__name__}: {error}"
            self.trace.write(
                "episode_failed",
                env_id=state.env_id,
                episode_id=state.episode_id,
                error=state.error,
            )
            raise
        self.trace.write(
            "observation_received",
            env_id=state.env_id,
            episode_id=state.episode_id,
            step=observation.step,
        )
        self._request_if_due(state)

    def get_actions(
        self, env_id: int, *, wait_for_capacity: bool = False
    ) -> list[dict]:
        self._open()
        state = self._episodes[env_id]
        if state.error:
            self._raise_episode_error(state)
        if (
            state.next_action_step is not None
            and state.latest.step < state.next_action_step
        ):
            raise RuntimeError("The previous action chunk has not finished executing")
        self._poll(state, wait=self._review_required(state))
        self._request_if_due(state)
        # Selection and initial subgoal are separate calls. Only the owner thread
        # loads the chosen policy, between those two calls and before any action.
        capacity_deadline = None
        while self._review_required(state) and not state.error:
            if state.env_id in self._pending:
                self._poll(state, wait=True)
                if state.env_id in self._pending:
                    break
            elif wait_for_capacity or self._blocking:
                running = [
                    future
                    for _, future, _ in self._pending.values()
                    if not future.done()
                ]
                running.extend(
                    future for _, future in self._retired if not future.done()
                )
                if not running:
                    # A worker may have completed between the capacity check and here.
                    if self._blocking and state.retry_at > self._clock():
                        time.sleep(max(0.0, state.retry_at - self._clock()))
                    self._request_if_due(state)
                    if state.env_id not in self._pending:
                        break
                    continue
                if capacity_deadline is None:
                    capacity_deadline = self._clock() + self.config.supervisor_timeout_s
                done, _ = wait(
                    running,
                    timeout=max(0.0, capacity_deadline - self._clock()),
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    state.error = "Supervisor capacity wait timed out"
                    state.supervisor_error = True
                    break
                capacity_deadline = None
            else:
                break
            self._request_if_due(state)
        if state.error:
            self._raise_episode_error(state)
        if self._review_required(state):
            raise SupervisorEpisodeError(
                "No supervision decision is ready; supervisor capacity is busy"
            )
        if (
            self.config.mode == Mode.CLOSED_LOOP
            and not self._blocking
            and (
                state.latest.step - state.last_reviewed_step
                > self.config.max_unreviewed_steps
            )
        ):
            state.error = "Current subgoal exceeded the unreviewed execution budget"
            state.supervisor_error = True
            self.trace.write(
                "episode_failed",
                env_id=env_id,
                episode_id=state.episode_id,
                error=state.error,
            )
            self._raise_episode_error(state)
        started = self._clock()
        action_budget = max(self.config.exec_horizon, fast_trajectory_chunk_size())
        try:
            predicted = self._policies[state.selected_policy].predict(
                deepcopy(state.latest),
                prompt=state.active_subgoal,
                max_actions=action_budget,
            )
            actions = json_value(predicted)
            if (
                not isinstance(actions, list)
                or not actions
                or any(not isinstance(row, dict) or not row for row in actions)
            ):
                raise ValueError(
                    "Action policy must return nonempty action dictionaries"
                )
            actions = actions[:action_budget]
        except Exception as error:
            state.error = f"{type(error).__name__}: {error}"
            state.supervisor_error = isinstance(error, SupervisorEpisodeError)
            self.trace.write(
                "episode_failed",
                env_id=env_id,
                episode_id=state.episode_id,
                error=state.error,
            )
            raise
        state.next_action_step = state.latest.step + len(actions)
        state.submitted_actions += len(actions)
        self.trace.write(
            "actions_submitted",
            env_id=env_id,
            episode_id=state.episode_id,
            observation_step=state.latest.step,
            plan_version=state.plan_version,
            policy_prompt=state.active_subgoal,
            policy_id=state.selected_policy,
            completed_action_chunks=state.completed_action_chunks,
            actions=actions,
            prediction_steps=len(predicted),
            elapsed_s=self._clock() - started,
        )
        return actions

    def end_episode(self, env_id: int, result: dict) -> None:
        """Accept an official evaluator result for logging, never as a policy input."""
        self._open()
        state = self._episodes[env_id]
        self.trace.write(
            "episode_ended",
            env_id=env_id,
            episode_id=state.episode_id,
            official_result=json_value(result),
            policy_id=state.selected_policy,
            supervisor_calls=state.supervisor_calls,
            submitted_actions=state.submitted_actions,
            completed_action_chunks=state.completed_action_chunks,
        )
        self.reset(env_id)

    def reset(self, env_id: int | None = None) -> None:
        self._open()
        indices = list(self._episodes) if env_id is None else [env_id]
        selected = None
        for index in indices:
            self._retire(index, "episode_reset")
            self._unbound_observations.pop(index, None)
            state = self._episodes.pop(index, None)
            if state is not None:
                selected = state.selected_policy
                self.trace.write(
                    "episode_reset", env_id=index, episode_id=state.episode_id
                )
        policies = (
            self._policies.values()
            if env_id is None
            else ([self._policies[selected]] if selected in self._policies else [])
        )
        failure = None
        for policy in policies:
            try:
                policy.reset(env_id)
            except Exception as error:
                failure = failure or error
        if failure:
            raise failure

    def close(self) -> None:
        if self._closed:
            return
        try:
            for index in list(self._pending):
                self._retire(index, "runtime_closed")
            failure = None
            for policy in self._policies.values():
                try:
                    policy.close()
                except Exception as error:
                    failure = failure or error
            if failure:
                raise failure
        finally:
            self._policies.clear()
            self._capabilities.clear()
            self._unbound_observations.clear()
            if self._owns_executor and self._executor is not None:
                self._executor.shutdown(wait=False, cancel_futures=True)
            self._closed = True
            self.trace.write("runtime_closed")

    def _open(self) -> None:
        if self._closed:
            raise RuntimeError("Runtime is closed")

    def _bind_policy(self, state, policy_id: str) -> None:
        if state.selected_policy is not None:
            raise ValueError("The episode policy is fixed until reset")
        started = self._clock()
        if policy_id not in self._candidates:
            raise ValueError("Selected policy is unknown or incompatible with subgoals")
        self._load_policy(policy_id)
        capability = deepcopy(self._capabilities[policy_id])
        state.selected_policy = policy_id
        state.policy_capability = capability
        begin = getattr(self._policies[policy_id], "begin_episode", None)
        if callable(begin):
            begin(state.env_id, state.episode_id)
        self.trace.write(
            "policy_bound",
            env_id=state.env_id,
            episode_id=state.episode_id,
            policy_id=policy_id,
            capability=capability,
            elapsed_s=self._clock() - started,
            selection="agent" if self._selects_policy else "fixed",
        )
        for observation in self._unbound_observations.pop(state.env_id, []):
            self._policies[policy_id].observe(deepcopy(observation))
        if self.config.mode == Mode.DIRECT:
            state.active_subgoal = state.task_goal
            state.plan_version = 1

    def prepare_job(self, seed):
        """Prepare a top-level backend; code-policy lazily chooses its sub-skills."""
        return self.prepare_fixed_policy(seed)

    def prepare_fixed_policy(self, seed):
        """Preload once; a job boundary resets state/RNG, never weights or JIT."""
        self._open()
        if self.fixed_policy is None or not (
            self.config.mode == Mode.DIRECT or self._blocking
        ):
            raise ValueError(
                "Suite preparation requires a fixed direct or blocking closed-loop policy"
            )
        self.reset()
        self.drain_supervisor(self.config.supervisor_timeout_s)
        self._load_policy(self.fixed_policy)
        policy = self._policies[self.fixed_policy]
        metrics = policy.prepare_job(seed)
        self.trace.write("policy_prepared", policy_id=self.fixed_policy, **metrics)
        return metrics

    def drain_supervisor(self, timeout):
        """Retire reviews without applying them, then await worker capacity."""
        for env_id in list(self._pending):
            self._retire(env_id, "job_finished")
        futures = [future for _, future in self._retired if not future.done()]
        if futures and wait(futures, timeout=timeout)[1]:
            raise SupervisorEpisodeError(
                "Supervisor requests did not drain; cannot start another job"
            )
        self._retired.clear()

    def _review_required(self, state):
        return (
            state.active_subgoal is None
            or self._blocking
            and state.completed_action_chunks
            >= state.last_reviewed_chunk + self.config.replan_interval_steps
        )

    @staticmethod
    def _raise_episode_error(state):
        error_type = SupervisorEpisodeError if state.supervisor_error else RuntimeError
        raise error_type(f"Episode failed; reset required: {state.error}")

    def _load_policy(self, policy_id):
        started = self._clock()
        if policy_id not in self._policies:
            candidate = self._definitions[policy_id]
            policy = self._factory(candidate.config)
            try:
                bind_trace = getattr(policy, "bind_trace", None)
                if callable(bind_trace):
                    bind_trace(self.trace, policy_id=policy_id)
                policy.load()
                capability = policy.describe()
                if not isinstance(capability, dict):
                    raise ValueError("The loaded policy must describe its capabilities")
                if (
                    self.config.mode != Mode.DIRECT
                    and capability.get("supports_subgoals") is not True
                ):
                    raise ValueError(
                        "The selected policy does not support subgoals as declared"
                    )
                self._capabilities[policy_id] = {
                    **deepcopy(capability),
                    "policy_id": policy_id,
                    "description": candidate.description,
                    "implementation_description": capability.get("description", ""),
                }
            except BaseException:
                policy.close()
                raise
            self._policies[policy_id] = policy
            self.trace.write(
                "policy_loaded", policy_id=policy_id, elapsed_s=self._clock() - started
            )

    def _request_if_due(self, state) -> None:
        selecting = self._selects_policy and state.selected_policy is None
        if (
            (self.config.mode == Mode.DIRECT and not selecting)
            or state.error
            or state.env_id in self._pending
        ):
            return
        if self.config.mode == Mode.ONCE and state.active_subgoal is not None:
            return
        if self._blocking:
            if not self._review_required(state) or state.retry_at > self._clock():
                return
        elif (
            state.active_subgoal is not None
            and state.last_request_step is not None
            and (
                state.latest.step - state.last_request_step
                < self.config.replan_interval_steps
            )
        ):
            return
        if state.supervisor_calls >= self.config.max_supervisor_calls:
            state.error = "Supervisor call budget exhausted"
            state.supervisor_error = True
            return
        self._retired = [
            (env_id, future) for env_id, future in self._retired if not future.done()
        ]
        if any(env_id == state.env_id for env_id, _ in self._retired):
            return
        active = sum(not future.done() for _, future, _ in self._pending.values())
        if active + len(self._retired) >= self.config.supervisor_workers:
            return
        memory = {
            name: deepcopy(getattr(state, name))
            for name in (
                "episode_id",
                "task_goal",
                "selected_policy",
                "active_subgoal",
                "plan_version",
                "completed_subgoals",
                "recent_feedback",
                "completed_action_chunks",
            )
        }
        memory["last_observation_step"] = state.latest.step
        request = SupervisorRequest(
            uuid.uuid4().hex,
            state.episode_id,
            state.plan_version,
            deepcopy(state.latest),
            memory,
            policy=deepcopy(state.policy_capability),
            candidates=deepcopy(self._candidates) if selecting else None,
        )
        self.trace.record_request(request, save_frames=self.config.record_frames)
        started = self._clock()
        future = self._executor.submit(self._run_supervisor, request)
        self._pending[state.env_id] = request, future, started
        state.last_request_step = state.latest.step
        state.supervisor_calls += 1

    def _run_supervisor(self, request: SupervisorRequest) -> tuple:
        try:
            if request.candidates is not None:
                decision = self.supervisor.select_policy(request)
                if not isinstance(decision, PolicySelection):
                    raise ValueError(
                        "Supervisor must return a validated PolicySelection"
                    )
            else:
                decision = self.supervisor.decide(request)
                if not isinstance(decision, Decision):
                    raise ValueError("Supervisor must return a validated Decision")
            return decision, None, self._clock()
        except Exception as error:
            return None, error, self._clock()

    def _poll(self, state, *, wait: bool = False) -> None:
        pending = self._pending.get(state.env_id)
        if pending is None:
            return
        request, future, started = pending
        deadline = started + self.config.supervisor_timeout_s
        if wait and not future.done():
            try:
                future.result(timeout=max(0.0, deadline - self._clock()))
            except TimeoutError:
                pass
        if not future.done():
            if self._clock() >= deadline:
                self._retire(state.env_id, "request_timeout")
                self._feedback_error(state, "Supervisor request timed out")
            return
        self._pending.pop(state.env_id)
        try:
            decision, error, finished_at = future.result()
        except Exception as error:
            self._feedback_error(
                state, f"Supervisor worker failed: {type(error).__name__}"
            )
            return
        self.trace.write(
            "supervisor_finished",
            env_id=state.env_id,
            episode_id=request.episode_id,
            request_id=request.request_id,
            elapsed_s=finished_at - started,
            decision=decision.to_dict() if decision else None,
            error=f"{type(error).__name__}: {error}" if error else None,
        )
        reason = None
        if request.episode_id != state.episode_id:
            reason = "episode_changed"
        elif request.plan_version != state.plan_version:
            reason = "plan_changed"
        elif (
            state.latest.step - request.observation.step
            > self.config.max_observation_age_steps
        ):
            reason = "observation_too_old"
        elif finished_at > deadline:
            reason = "request_timeout"
        if reason:
            self.trace.write(
                "supervisor_discarded",
                env_id=state.env_id,
                episode_id=state.episode_id,
                request_id=request.request_id,
                reason=reason,
            )
            self._feedback_error(state, reason)
            return
        if error:
            self._feedback_error(state, error)
            return
        try:
            if request.candidates is not None:
                self.trace.write(
                    "policy_selected",
                    env_id=state.env_id,
                    episode_id=state.episode_id,
                    request_id=request.request_id,
                    selection=decision.to_dict(),
                )
                try:
                    self._bind_policy(state, decision.policy_id)
                except Exception as error:
                    state.error = (
                        f"Policy binding failed: {type(error).__name__}: {error}"
                    )
                    self.trace.write(
                        "episode_failed",
                        env_id=state.env_id,
                        episode_id=state.episode_id,
                        error=state.error,
                    )
            else:
                self._apply(state, decision, request)
        except ValueError as error:
            self._feedback_error(state, str(error))

    def _apply(self, state, decision: Decision, request: SupervisorRequest) -> None:
        previous = state.active_subgoal
        supported = (state.policy_capability or {}).get("supported_subgoals")
        if supported and decision.subgoal not in supported:
            raise ValueError("Subgoal is not supported by the bound policy")
        if decision.decision == DecisionKind.KEEP and (
            previous is None or decision.subgoal != previous
        ):
            raise ValueError("keep requires the exact existing subgoal")
        if decision.previous_subgoal_completed and previous is None:
            raise ValueError("There is no previous subgoal to mark completed")
        if decision.previous_subgoal_completed:
            state.completed_subgoals.append(
                {
                    "subgoal": previous,
                    "reason": decision.reason,
                    "step": state.latest.step,
                }
            )
        if decision.decision != DecisionKind.KEEP:
            state.active_subgoal = decision.subgoal
            state.plan_version += 1
        state.last_reviewed_step = request.observation.step
        state.last_reviewed_chunk = request.memory["completed_action_chunks"]
        state.consecutive_failures = 0
        state.recent_feedback.append({"step": state.latest.step, **decision.to_dict()})
        del state.recent_feedback[: -self.config.feedback_history_size]
        self.trace.write(
            "decision_applied",
            env_id=state.env_id,
            episode_id=state.episode_id,
            request_id=request.request_id,
            observation_step=request.observation.step,
            applied_step=state.latest.step,
            completed_action_chunks=state.completed_action_chunks,
            plan_version=state.plan_version,
            policy_id=state.selected_policy,
            decision=decision.to_dict(),
        )

    def _feedback_error(self, state, message: Exception | str) -> None:
        retryable = getattr(message, "retryable", True)
        message = str(message)
        state.consecutive_failures += 1
        state.recent_feedback.append({"step": state.latest.step, "error": message})
        del state.recent_feedback[: -self.config.feedback_history_size]
        self.trace.write(
            "supervisor_failed",
            env_id=state.env_id,
            episode_id=state.episode_id,
            step=state.latest.step,
            error=message,
        )
        if (
            (state.active_subgoal is None and not self._blocking)
            or state.consecutive_failures >= self.config.max_consecutive_failures
            or not retryable
        ):
            state.error = message
            state.supervisor_error = True
        elif self._blocking:
            delay = min(
                self.config.supervisor_timeout_s,
                self.config.supervisor_retry_backoff_s
                * 2 ** min(state.consecutive_failures - 1, 16),
            )
            state.retry_at = self._clock() + delay
            self.trace.write(
                "supervisor_retry_scheduled",
                env_id=state.env_id,
                episode_id=state.episode_id,
                step=state.latest.step,
                delay_s=delay,
                consecutive_failures=state.consecutive_failures,
            )

    def _retire(self, env_id: int, reason: str) -> None:
        pending = self._pending.pop(env_id, None)
        if pending is None:
            return
        request, future, started = pending
        if not future.cancel() and not future.done():
            self._retired.append((env_id, future))
        self.trace.write(
            "supervisor_discarded",
            env_id=env_id,
            episode_id=request.episode_id,
            request_id=request.request_id,
            elapsed_s=self._clock() - started,
            reason=reason,
        )
