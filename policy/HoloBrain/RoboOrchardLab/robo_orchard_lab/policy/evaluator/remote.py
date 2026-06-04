# Project RoboOrchard
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
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
import concurrent.futures
import math
import weakref
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, NoReturn

import ray
import torch
from pydantic import Field
from ray._raylet import ObjectRefGenerator
from ray.exceptions import GetTimeoutError, RayActorError, RayTaskError
from robo_orchard_core.utils.config import ClassType, ConfigInstanceOf
from robo_orchard_core.utils.logging import LoggerManager
from robo_orchard_core.utils.ray import (
    RayRemoteInstance,
    RayRemoteInstanceConfig,
)

from robo_orchard_lab.envs.base import EnvBaseCfg
from robo_orchard_lab.policy.base import PolicyConfig, PolicyMixin
from robo_orchard_lab.policy.evaluator.base import (
    PolicyEvaluationError,
    PolicyEvaluationExecutionError,
    PolicyEvaluationRemoteTimeoutError,
    PolicyEvaluationWorkerLostError,
    PolicyEvaluator,
    PolicyEvaluatorConfig,
    RollOutStopCondition,
    evaluate_rollout_stop_condition,
)
from robo_orchard_lab.policy.evaluator.contracts import (
    EnvResetInput,
    EnvStartInput,
    EpisodeResult,
    EvaluationStatus,
    PolicyResetInput,
    TerminalReason,
)
from robo_orchard_lab.policy.evaluator.metric_contracts import EvaluatorMetrics

__all__ = ["PolicyEvaluatorRemote", "PolicyEvaluatorRemoteConfig"]

logger = LoggerManager().get_child(__name__)


def _format_remote_timeout_message(
    operation: str,
    timeout_s: float | None,
    *,
    wait_count: int = 1,
) -> str:
    if timeout_s is None:
        return f"Remote policy evaluator {operation} timed out."
    if wait_count <= 1:
        return (
            f"Remote policy evaluator {operation} timed out after "
            f"{timeout_s}s."
        )
    return (
        f"Remote policy evaluator {operation} timed out after "
        f"{wait_count} consecutive {timeout_s}s waits."
    )


def _finalize_remote_actor(remote: Any) -> None:
    if remote is None:
        return
    try:
        if not ray.is_initialized():
            return
        ray.kill(remote, no_restart=True)
    except Exception:
        logger.exception("Failed to finalize remote policy evaluator actor.")


class PolicyEvaluatorRemote(RayRemoteInstance[PolicyEvaluator]):
    """Synchronous Ray facade for one remote :class:`PolicyEvaluator`.

    Use this facade when callers need the same evaluator public surface as
    :class:`PolicyEvaluator` but want env and policy execution isolated in a
    Ray actor. The wrapper forwards setup, reconfigure, reset, rollout,
    and metric calls through a synchronous API while mapping Ray timeout and
    worker-loss failures into public evaluator exceptions.

    The wrapper owns the Ray actor handle, but not benchmark retry
    orchestration or actor snapshot recovery. Higher layers that need
    failure isolation should close or replace the wrapper after timeout or
    worker-loss failures.

    The actor is a disposable runtime resource. Use :meth:`close` or a
    ``with`` block to release it; close is idempotent and kills the actor
    directly instead of queuing a graceful remote close call behind a
    potentially stuck reset or rollout.

    One streaming operation may be active per wrapper instance. Exhaust or
    close the generator returned by :meth:`make_episode_evaluation` before
    calling another method on the same facade.
    """

    cfg: PolicyEvaluatorRemoteConfig
    InitFromConfig: bool = True

    def __init__(
        self,
        cfg: PolicyEvaluatorRemoteConfig,
    ) -> None:
        super().__init__(cfg)
        self._stream_active = False
        self._closed = False
        self._remote_finalizer: weakref.finalize | None = None
        self._set_remote_finalizer(getattr(self, "_remote", None))

    def __enter__(self) -> PolicyEvaluatorRemote:
        """Enter an explicit remote evaluator lifecycle scope."""

        self._ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """Release the owned Ray actor when leaving a lifecycle scope."""

        del exc_type, exc, tb
        self.close()

    def close(self) -> None:
        """Release the owned Ray actor and make this facade unusable.

        Cleanup is best-effort and idempotent. The actor is killed with
        ``no_restart=True`` and the finalizer is detached so explicit close
        remains the primary lifecycle contract.
        """

        if getattr(self, "_closed", False):
            return
        self._closed = True
        self._stream_active = False

        self._kill_remote_actor()

    def setup(
        self,
        env_cfg: EnvBaseCfg,
        policy_or_cfg: PolicyConfig | PolicyMixin,
        metrics: EvaluatorMetrics,
        device: str | torch.device | None = None,
        *,
        force_recreate: bool | None = None,
        timeout_s: float | None = None,
    ) -> None:
        """Set up the remote evaluator.

        ``force_recreate`` is forwarded only when explicitly provided;
        ``None`` lets the remote evaluator follow its
        :class:`PolicyEvaluatorConfig`.
        """

        self._ensure_no_active_stream()
        timeout = self._resolve_timeout_s(
            timeout_s,
            default_timeout_s=self.cfg.reset_timeout_s,
        )
        setup_kwargs = {
            "env_cfg": env_cfg,
            "policy_or_cfg": policy_or_cfg,
            "metrics": metrics,
            "device": device,
        }
        if force_recreate is not None:
            setup_kwargs["force_recreate"] = force_recreate
        self._get(
            self.remote.setup.remote(**setup_kwargs),
            timeout=timeout,
            timeout_message="Remote policy evaluator setup timed out.",
        )

    def reconfigure_metrics(
        self,
        metrics: EvaluatorMetrics,
        *,
        timeout_s: float | None = None,
    ) -> None:
        """Reconfigure the remote metrics."""

        self._ensure_no_active_stream()
        timeout = self._resolve_timeout_s(
            timeout_s,
            default_timeout_s=self.cfg.reset_timeout_s,
        )
        self._get(
            self.remote.reconfigure_metrics.remote(metrics),
            timeout=timeout,
            timeout_message=(
                "Remote policy evaluator metrics reconfigure timed out."
            ),
        )

    def reconfigure_env(
        self,
        env_cfg: EnvBaseCfg,
        *,
        force_recreate: bool | None = None,
        timeout_s: float | None = None,
    ) -> None:
        """Reconfigure the remote environment.

        ``force_recreate`` is forwarded only when explicitly provided;
        ``None`` lets the remote evaluator follow its
        :class:`PolicyEvaluatorConfig`.
        """

        self._ensure_no_active_stream()
        timeout = self._resolve_timeout_s(
            timeout_s,
            default_timeout_s=self.cfg.reset_timeout_s,
        )
        reconfigure_kwargs: dict[str, Any] = {}
        if force_recreate is not None:
            reconfigure_kwargs["force_recreate"] = force_recreate
        self._get(
            self.remote.reconfigure_env.remote(
                env_cfg,
                **reconfigure_kwargs,
            ),
            timeout=timeout,
            timeout_message=(
                "Remote policy evaluator environment reconfigure timed out."
            ),
        )

    def reconfigure_policy(
        self,
        policy_or_cfg: PolicyConfig | PolicyMixin,
        device: str | torch.device | None = None,
        *,
        timeout_s: float | None = None,
    ) -> None:
        """Reconfigure the remote policy."""

        self._ensure_no_active_stream()
        timeout = self._resolve_timeout_s(
            timeout_s,
            default_timeout_s=self.cfg.reset_timeout_s,
        )
        self._get(
            self.remote.reconfigure_policy.remote(
                policy_or_cfg,
                device=device,
            ),
            timeout=timeout,
            timeout_message=(
                "Remote policy evaluator policy reconfigure timed out."
            ),
        )

    def evaluate_episode(
        self,
        max_steps: int,
        env_reset_input: EnvStartInput = None,
        policy_reset_input: PolicyResetInput = None,
        rollout_stop_condition: RollOutStopCondition = evaluate_rollout_stop_condition,  # noqa: E501
        *,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Evaluate one complete episode on the remote actor.

        The return value matches :meth:`PolicyEvaluator.evaluate_episode`: it
        is the remote evaluator's current metrics dict after the episode
        finishes and terminal metrics are updated, not necessarily a delta
        for only this episode.

        This non-streaming facade consumes
        :meth:`make_episode_evaluation` to keep ``rollout_timeout_s`` and
        ``timeout_s`` scoped to remote rollout waits instead of the whole
        episode call. Final metric computation is fetched after the stream
        completes. Ordinary evaluator execution failures restore only the
        remote metric runtime state to the pre-attempt value. Ray timeout and
        worker-loss errors leave worker replacement to the caller because the
        in-flight actor may still be busy or unavailable.

        Args:
            max_steps: Maximum number of environment steps for the episode.
            env_reset_input: Episode-start input forwarded to the remote
                evaluator. ``PreparedEnvStart`` still skips env reset while
                resetting the policy.
            policy_reset_input: Optional kwargs for the remote per-episode
                policy reset.
            rollout_stop_condition: Stop predicate passed to remote rollout.
            timeout_s: Optional rollout-wait timeout override for this
                remote episode call. Final metric computation uses the
                ordinary remote metric timeout.

        Returns:
            The metrics dict produced by the remote evaluator after the
            episode finishes.
        """

        self._ensure_no_active_stream()
        timeout = self._resolve_timeout_s(
            timeout_s,
            default_timeout_s=self.cfg.rollout_timeout_s,
        )
        metric_timeout = self._resolve_timeout_s(
            None,
            default_timeout_s=self.cfg.reset_timeout_s,
        )
        metric_state = self._get(
            self.remote._export_metric_runtime_state.remote(),
            timeout=metric_timeout,
            timeout_message=(
                "Remote policy evaluator metric state export timed out."
            ),
        )
        episode_steps = 0
        try:
            for step_count in self._make_episode_evaluation(
                max_steps=max_steps,
                env_reset_input=env_reset_input,
                policy_reset_input=policy_reset_input,
                rollout_steps=max_steps,
                rollout_stop_condition=rollout_stop_condition,
                timeout_s=timeout,
                timeout_operation="evaluate_episode",
            ):
                episode_steps += step_count
        except (
            PolicyEvaluationRemoteTimeoutError,
            PolicyEvaluationWorkerLostError,
        ):
            raise
        except PolicyEvaluationExecutionError as exception:
            self._restore_remote_metric_runtime_state_after_failure(
                metric_state,
                timeout=metric_timeout,
                original_exception=exception,
            )
            raise
        except Exception as exception:
            self._restore_remote_metric_runtime_state_after_failure(
                metric_state,
                timeout=metric_timeout,
                original_exception=exception,
            )
            raise PolicyEvaluationExecutionError(
                "Remote policy evaluation failed during episode execution.",
                result=EpisodeResult(
                    status=EvaluationStatus.FAILED,
                    terminal_reason=TerminalReason.ERROR,
                    episode_steps=episode_steps,
                    metrics={},
                ),
                cause_type=type(exception).__name__,
                cause_message=str(exception),
            ) from exception

        try:
            return self.compute_metrics()
        except (
            PolicyEvaluationRemoteTimeoutError,
            PolicyEvaluationWorkerLostError,
        ):
            raise
        except Exception as exception:
            self._restore_remote_metric_runtime_state_after_failure(
                metric_state,
                timeout=metric_timeout,
                original_exception=exception,
            )
            raise PolicyEvaluationExecutionError(
                "Failed to compute policy evaluation metrics.",
                result=EpisodeResult(
                    status=EvaluationStatus.FAILED,
                    terminal_reason=TerminalReason.ERROR,
                    episode_steps=episode_steps,
                    metrics={},
                ),
                cause_type=type(exception).__name__,
                cause_message=str(exception),
            ) from exception

    def make_episode_evaluation(
        self,
        max_steps: int,
        env_reset_input: EnvStartInput = None,
        policy_reset_input: PolicyResetInput = None,
        rollout_steps: int = 5,
        rollout_stop_condition: RollOutStopCondition = evaluate_rollout_stop_condition,  # noqa: E501
        *,
        timeout_s: float | None = None,
    ) -> Generator[int, None, None]:
        """Yield rollout step counts from the remote evaluator."""

        yield from self._make_episode_evaluation(
            max_steps=max_steps,
            env_reset_input=env_reset_input,
            policy_reset_input=policy_reset_input,
            rollout_steps=rollout_steps,
            rollout_stop_condition=rollout_stop_condition,
            timeout_s=timeout_s,
            timeout_operation="make_episode_evaluation",
        )

    def _make_episode_evaluation(
        self,
        max_steps: int,
        env_reset_input: EnvStartInput,
        policy_reset_input: PolicyResetInput,
        rollout_steps: int,
        rollout_stop_condition: RollOutStopCondition,
        *,
        timeout_s: float | None,
        timeout_operation: str,
    ) -> Generator[int, None, None]:
        gen = None
        timeout = self._resolve_timeout_s(
            timeout_s,
            default_timeout_s=self.cfg.rollout_timeout_s,
        )
        with self._stream_operation():
            try:
                gen = self.remote.make_episode_evaluation.remote(
                    max_steps=max_steps,
                    env_reset_input=env_reset_input,
                    policy_reset_input=policy_reset_input,
                    rollout_steps=rollout_steps,
                    rollout_stop_condition=rollout_stop_condition,
                )
                if isinstance(gen, ObjectRefGenerator):
                    while True:
                        try:
                            ref = self._next_remote_stream_ref(
                                gen,
                                timeout=timeout,
                                timeout_operation=timeout_operation,
                            )
                        except StopIteration:
                            break
                        yield self._get(
                            ref,
                            timeout=timeout,
                            timeout_message=_format_remote_timeout_message(
                                timeout_operation,
                                timeout,
                                wait_count=self._timeout_wait_count(),
                            ),
                        )
                else:
                    for ref in gen:
                        yield self._get(
                            ref,
                            timeout=timeout,
                            timeout_message=_format_remote_timeout_message(
                                timeout_operation,
                                timeout,
                                wait_count=self._timeout_wait_count(),
                            ),
                        )
            except KeyboardInterrupt:
                self.close()
                raise
            finally:
                if gen is not None:
                    close = getattr(gen, "close", None)
                    if callable(close):
                        try:
                            close()
                        except NotImplementedError:
                            pass

    def _next_remote_stream_ref(
        self,
        gen: ObjectRefGenerator,
        *,
        timeout: float | None,
        timeout_operation: str,
    ) -> Any:
        timeout_message = _format_remote_timeout_message(
            timeout_operation,
            timeout,
            wait_count=self._timeout_wait_count(),
        )
        wait_count = 1 if timeout is None else self._timeout_wait_count()
        for _ in range(wait_count):
            try:
                ready, _ = ray.wait(
                    [gen],
                    timeout=timeout,
                    fetch_local=False,
                )
                if ready:
                    return next(gen)
            except KeyboardInterrupt:
                self.close()
                raise
            except (RayTaskError, RayActorError) as exception:
                self._raise_mapped_remote_failure(exception)

        raise PolicyEvaluationRemoteTimeoutError(timeout_message)

    def _restore_remote_metric_runtime_state(
        self,
        state: Any,
        *,
        timeout: float | None,
    ) -> None:
        self._get(
            self.remote._restore_metric_runtime_state.remote(state),
            timeout=timeout,
            timeout_message=(
                "Remote policy evaluator metric state restore timed out."
            ),
        )

    def _restore_remote_metric_runtime_state_after_failure(
        self,
        state: Any,
        *,
        timeout: float | None,
        original_exception: BaseException,
    ) -> None:
        try:
            self._restore_remote_metric_runtime_state(
                state,
                timeout=timeout,
            )
        except Exception as restore_exception:
            add_note = getattr(original_exception, "add_note", None)
            if callable(add_note):
                add_note(
                    "Remote policy evaluator metric state rollback failed: "
                    f"{type(restore_exception).__name__}: "
                    f"{restore_exception}"
                )

    def reset_metrics(
        self,
        *,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Reset the remote metrics."""

        self._ensure_no_active_stream()
        timeout = self._resolve_timeout_s(
            timeout_s,
            default_timeout_s=self.cfg.reset_timeout_s,
        )
        self._get(
            self.remote.reset_metrics.remote(**kwargs),
            timeout=timeout,
            timeout_message="Remote policy evaluator reset timed out.",
        )

    def reset_policy(
        self,
        *,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Reset the remote policy."""

        self._ensure_no_active_stream()
        timeout = self._resolve_timeout_s(
            timeout_s,
            default_timeout_s=self.cfg.reset_timeout_s,
        )
        self._get(
            self.remote.reset_policy.remote(**kwargs),
            timeout=timeout,
            timeout_message="Remote policy evaluator reset timed out.",
        )

    def reset_env(
        self,
        env_reset_input: EnvResetInput = None,
        *,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """Reset the remote environment.

        ``env_reset_input`` is forwarded as the evaluator reset contract:
        ``None`` calls the remote env reset with no kwargs, ``dict`` provides
        reset kwargs, and ``State`` is left for the remote evaluator's
        state-reset capability checks. Legacy ``reset_env(seed=...)`` calls
        are normalized to a dict input.
        """

        self._ensure_no_active_stream()
        if kwargs:
            if env_reset_input is not None:
                raise ValueError(
                    "Pass either env_reset_input or reset kwargs to "
                    "PolicyEvaluatorRemote.reset_env(...), not both."
                )
            env_reset_input = dict(kwargs)
        timeout = self._resolve_timeout_s(
            timeout_s,
            default_timeout_s=self.cfg.reset_timeout_s,
        )
        return self._get(
            self.remote.reset_env.remote(env_reset_input=env_reset_input),
            timeout=timeout,
            timeout_message="Remote policy evaluator reset timed out.",
        )

    def get_metrics(
        self,
        *,
        timeout_s: float | None = None,
    ) -> EvaluatorMetrics | None:
        """Return the configured remote evaluator metrics."""

        self._ensure_no_active_stream()
        timeout = self._resolve_timeout_s(
            timeout_s,
            default_timeout_s=self.cfg.reset_timeout_s,
        )
        return self._get(
            self.remote.get_metrics.remote(),
            timeout=timeout,
        )

    def compute_metrics(
        self,
        *,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Compute and return remote evaluator metrics."""

        self._ensure_no_active_stream()
        timeout = self._resolve_timeout_s(
            timeout_s,
            default_timeout_s=self.cfg.reset_timeout_s,
        )
        return self._get(
            self.remote.compute_metrics.remote(),
            timeout=timeout,
            timeout_message=(
                "Remote policy evaluator metrics compute timed out."
            ),
        )

    @contextmanager
    def _stream_operation(self) -> Generator[None, None, None]:
        self._ensure_no_active_stream()
        self._stream_active = True
        try:
            yield
        finally:
            self._stream_active = False

    def _ensure_no_active_stream(self) -> None:
        self._ensure_open()
        if self._stream_active:
            raise PolicyEvaluationError(
                "Another remote episode stream is active on this evaluator. "
                "Exhaust or close it before running another operation."
            )

    def _ensure_open(self) -> None:
        if getattr(self, "_closed", False):
            raise PolicyEvaluationError(
                "Remote policy evaluator is closed and cannot be reused."
            )

    def _resolve_timeout_s(
        self,
        timeout_s: float | None,
        *,
        default_timeout_s: float | None,
    ) -> float | None:
        if timeout_s is None:
            return default_timeout_s
        if timeout_s <= 0 or not math.isfinite(timeout_s):
            raise ValueError("timeout_s must be a positive number or None.")
        return timeout_s

    def _get(
        self,
        ref: Any,
        timeout: float | None = None,
        *,
        timeout_message: str = "Remote policy evaluation timed out.",
    ) -> Any:
        wait_count = 1 if timeout is None else self._timeout_wait_count()
        last_timeout: BaseException | None = None
        for _ in range(wait_count):
            try:
                return ray.get(ref, timeout=timeout)
            except (concurrent.futures.TimeoutError, GetTimeoutError) as exc:
                last_timeout = exc
                continue
            except KeyboardInterrupt:
                self.close()
                raise
            except (RayTaskError, RayActorError) as exception:
                self._raise_mapped_remote_failure(exception)

        raise PolicyEvaluationRemoteTimeoutError(
            timeout_message
        ) from last_timeout

    def _timeout_wait_count(self) -> int:
        return self.cfg.timeout_grace_retries + 1

    def _unwrap_task_error(self, exception: RayTaskError) -> BaseException:
        cause = getattr(exception, "cause", None)
        if isinstance(cause, BaseException):
            return cause
        return exception.as_instanceof_cause()

    def _raise_worker_lost(self, exception: BaseException) -> NoReturn:
        raise PolicyEvaluationWorkerLostError(
            "Remote policy evaluator worker was lost."
        ) from exception

    def _raise_mapped_remote_failure(
        self,
        exception: RayTaskError | RayActorError,
    ) -> NoReturn:
        if isinstance(exception, RayActorError):
            self._raise_worker_lost(exception)

        cause = self._unwrap_task_error(exception)
        if isinstance(cause, PolicyEvaluationError):
            raise cause from exception
        if isinstance(cause, RayActorError):
            self._raise_worker_lost(exception)
        raise cause from exception

    def _kill_remote_actor(self) -> None:
        remote = getattr(self, "_remote", None)
        try:
            if remote is not None:
                ray.kill(remote, no_restart=True)
        except Exception:
            logger.exception("Failed to kill remote policy evaluator actor.")
        finally:
            self._detach_remote_finalizer()
            self._remote = None  # type: ignore[assignment]
            self._remote_checked = False

    def _set_remote_finalizer(self, remote: Any) -> None:
        self._detach_remote_finalizer()
        self._remote_finalizer = weakref.finalize(
            self,
            _finalize_remote_actor,
            remote,
        )

    def _detach_remote_finalizer(self) -> None:
        finalizer = getattr(self, "_remote_finalizer", None)
        if finalizer is not None and finalizer.alive:
            finalizer.detach()
        self._remote_finalizer = None


class PolicyEvaluatorRemoteConfig(
    RayRemoteInstanceConfig[PolicyEvaluatorRemote, PolicyEvaluatorConfig]
):
    """Configure the Ray facade around one PolicyEvaluator actor.

    The nested ``instance_config`` remains the source of truth for local
    evaluator behavior such as env reuse. This config owns only remote
    transport concerns: Ray actor resources, Ray initialization inherited
    from :class:`RayRemoteInstanceConfig`, and default per-call timeouts.
    Higher-level benchmark orchestrators should pass per-call timeout
    overrides only when they intentionally need a narrower or wider bound.
    """

    class_type: ClassType[PolicyEvaluatorRemote] = PolicyEvaluatorRemote
    instance_config: ConfigInstanceOf[PolicyEvaluatorConfig]
    rollout_timeout_s: float | None = Field(default=120.0, gt=0)
    """Timeout for remote episode and rollout-stream waits."""
    reset_timeout_s: float | None = Field(default=1200.0, gt=0)
    """Timeout for remote setup, reset, reconfigure, and metric calls."""
    timeout_grace_retries: int = Field(default=1, ge=0)
    """Extra waits after a remote timeout before reporting failure."""
