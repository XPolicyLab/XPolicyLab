# Project RoboOrchard
#
# Copyright (c) 2025 Horizon Robotics. All Rights Reserved.
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
import weakref
from types import TracebackType
from typing import TYPE_CHECKING, Any, Callable, Generator, Protocol, cast

import torch
from robo_orchard_core.utils.config import (
    ClassConfig,
)
from robo_orchard_core.utils.logging import LoggerManager
from robo_orchard_core.utils.ray import RayRemoteClassConfig
from typing_extensions import Self

from robo_orchard_lab.envs.base import EnvBase, EnvBaseCfg, EnvStepReturn
from robo_orchard_lab.envs.state import (
    EnvStateScope,
    require_env_supports_state_scope,
)
from robo_orchard_lab.policy.base import (
    PolicyConfig,
    PolicyMixin,
)
from robo_orchard_lab.policy.evaluator.contracts import (
    EnvResetInput,
    EnvStartInput,
    EpisodeResult,
    EvaluationRequest,
    PolicyResetInput,
)
from robo_orchard_lab.policy.evaluator.episode import (
    _resolve_step_ret_terminal_flags,
    _run_episode_loop as _run_policy_episode_loop,
    evaluate_episode as evaluate_policy_episode,
)
from robo_orchard_lab.policy.evaluator.metric_contracts import (
    EvaluatorMetrics,
)
from robo_orchard_lab.policy.evaluator.recovery import (
    PolicyEvaluatorRecoveryManager,
    PolicyEvaluatorRecoverySnapshot,
)
from robo_orchard_lab.utils.state import State

if TYPE_CHECKING:
    from robo_orchard_lab.policy.evaluator.remote import (
        PolicyEvaluatorRemoteConfig,
    )

__all__ = [
    "PolicyEvaluationError",
    "PolicyEvaluationExecutionError",
    "PolicyEvaluationRemoteTimeoutError",
    "PolicyEvaluationWorkerLostError",
    "PolicyEvaluator",
    "PolicyEvaluatorConfig",
    "RollOutStopCondition",
    "evaluate_rollout_stop_condition",
]

RollOutStopCondition = Callable[
    [EnvStepReturn | tuple[Any, Any, bool, bool, dict[str, Any]]], bool
]

logger = LoggerManager().get_child(__name__)


class PolicyEvaluationError(RuntimeError):
    """Base exception for public policy-evaluation failures."""


class PolicyEvaluationExecutionError(PolicyEvaluationError):
    """Raised when an episode fails during local evaluator execution."""

    result: EpisodeResult
    cause_type: str | None
    cause_message: str | None

    def __init__(
        self,
        message: str,
        result: EpisodeResult,
        cause_type: str | None = None,
        cause_message: str | None = None,
    ) -> None:
        super().__init__(message)
        self.result = result
        self.cause_type = cause_type
        self.cause_message = cause_message

    def __str__(self) -> str:
        message = super().__str__()
        if self.cause_type is None and self.cause_message is None:
            return message
        cause = self.cause_type or "Exception"
        if self.cause_message:
            cause = f"{cause}: {self.cause_message}"
        return f"{message} Cause: {cause}"

    def __reduce__(
        self,
    ) -> tuple[
        type[PolicyEvaluationExecutionError],
        tuple[str, EpisodeResult, str | None, str | None],
    ]:  # noqa: E501
        return (
            self.__class__,
            (
                self.args[0],
                self.result,
                self.cause_type,
                self.cause_message,
            ),
        )


class PolicyEvaluationRemoteTimeoutError(PolicyEvaluationError):
    """Raised when a remote evaluation exceeds its timeout."""


class PolicyEvaluationWorkerLostError(PolicyEvaluationError):
    """Raised when a remote evaluator worker becomes unavailable."""


def evaluate_rollout_stop_condition(
    step_ret: EnvStepReturn | tuple[Any, Any, bool, bool, dict[str, Any]],
) -> bool:
    """Determine whether to stop the rollout based on terminal conditions.

    Returns:
        bool: True if the rollout should stop, False otherwise.
    """
    terminated, truncated = _resolve_step_ret_terminal_flags(step_ret)
    return terminated or truncated


def _close_env(env: EnvBase | None) -> None:
    if env is None:
        return
    close = getattr(env, "close", None)
    if not callable(close):
        return
    try:
        close()
    except Exception:
        logger.exception("Failed to close policy evaluator env.")


class _ConfigBackedEnv(Protocol):
    cfg: EnvBaseCfg


class PolicyEvaluator:
    """Run single policy-evaluation episodes against one configured env.

    Use this facade when a caller wants the evaluator to own env creation,
    episode-start normalization, policy reset, rollout execution, metric
    timing dispatch, and metric-state rollback after ordinary execution
    failures. Configure it with :meth:`setup`, or reconfigure env, policy, and
    metrics independently before calling :meth:`evaluate_episode` or
    :meth:`make_episode_evaluation`.

    The evaluator owns envs created from env configs and closes them on
    replacement, :meth:`close`, or context-manager exit. Policy and metric
    objects are borrowed after setup; their public contracts do not define a
    close lifecycle, so close detaches them without calling ``close()``.

    :meth:`evaluate_episode` is the non-streaming path. It returns the
    current computed metric dict after terminal metric updates and restores
    metric runtime state if the episode raises
    :class:`PolicyEvaluationExecutionError`. :meth:`make_episode_evaluation`
    streams rollout batch sizes and intentionally does not provide transparent
    rollback after partial output has been yielded.

    Example::

        with PolicyEvaluator(cfg) as evaluator:
            evaluator.setup(env_cfg, policy_cfg, evaluator_metrics)
            metrics = evaluator.evaluate_episode(max_steps=1000)

    Args:
        cfg (PolicyEvaluatorConfig): Configuration for the
            PolicyEvaluator instance.

    """

    InitFromConfig: bool = True

    cfg: PolicyEvaluatorConfig
    _env: EnvBase | None
    _policy: PolicyMixin | None
    _evaluator_metrics_value: EvaluatorMetrics | None
    _recovery_manager: PolicyEvaluatorRecoveryManager
    _env_finalizer: weakref.finalize | None

    def __init__(
        self,
        cfg: PolicyEvaluatorConfig,
    ) -> None:
        self.cfg = cfg
        self._env = None
        self._policy = None
        self._evaluator_metrics_value = None
        self._recovery_manager = PolicyEvaluatorRecoveryManager()
        self._env_finalizer = None

    def _refresh_env_finalizer(self) -> None:
        """Track the evaluator-owned env for best-effort finalizer cleanup."""

        if self._env_finalizer is not None and self._env_finalizer.alive:
            self._env_finalizer.detach()
        self._env_finalizer = None
        if self._env is None:
            return
        self._env_finalizer = weakref.finalize(
            self,
            _close_env,
            self._env,
        )

    @property
    def env(self) -> EnvBase:
        if self._env is None:
            raise RuntimeError(
                "Environment is not configured. Cannot access environment."
            )
        return self._env

    @env.setter
    def env(self, env: EnvBase) -> None:
        old_env = self._env
        if old_env is not None and old_env is not env:
            _close_env(old_env)
        self._env = env
        self._refresh_env_finalizer()

    @property
    def policy(self) -> PolicyMixin:
        if self._policy is None:
            raise RuntimeError(
                "Policy is not configured. Cannot access policy."
            )
        return self._policy

    @policy.setter
    def policy(self, policy: PolicyMixin) -> None:
        self._policy = policy

    @property
    def _evaluator_metrics(self) -> EvaluatorMetrics:
        if self._evaluator_metrics_value is None:
            raise RuntimeError(
                "Metrics are not configured. Cannot access evaluator "
                "metric runtime."
            )
        return self._evaluator_metrics_value

    @property
    def metrics(self) -> EvaluatorMetrics:
        return self._evaluator_metrics

    def _is_ready(self) -> bool:
        return (
            self._env is not None
            and self._policy is not None
            and self._evaluator_metrics_value is not None
        )

    def close(self) -> None:
        """Release evaluator-owned env resources and detach runtime objects.

        Close is idempotent and best-effort. Envs are evaluator-owned because
        they are created from configs. Policy and metric public contracts do
        not include ``close()``, so they are detached without being closed.
        The evaluator is left unconfigured after close; call ``setup(...)``
        again before further evaluation.
        """

        env = self._env
        self._env = None
        self._policy = None
        self._evaluator_metrics_value = None
        self._refresh_env_finalizer()
        _close_env(env)

    def __enter__(self) -> Self:
        """Return this evaluator for context-managed resource ownership."""

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close evaluator-owned resources when leaving a ``with`` block."""

        del exc_type, exc, tb
        self.close()

    def setup(
        self,
        env_cfg: EnvBaseCfg,
        policy_or_cfg: PolicyConfig | PolicyMixin,
        metrics: EvaluatorMetrics,
        device: str | torch.device | None = None,
        *,
        force_recreate: bool | None = None,
    ):
        """Setup the evaluator with the current configuration.

        Args:
            env_cfg (EnvBaseCfg): Environment configuration used to create
                or reuse the evaluator-owned env.
            policy_or_cfg (PolicyConfig | PolicyMixin): Policy config or
                instance used by the evaluator.
            metrics (EvaluatorMetrics): Metric surface borrowed by the
                evaluator.
            device (str | torch.device | None, optional): Optional policy
                device. Default is None.
            force_recreate (bool | None, optional): Passed to
                :meth:`reconfigure_env`. ``None`` follows
                :class:`PolicyEvaluatorConfig`. Default is None.
        """
        self.reconfigure_env(env_cfg, force_recreate=force_recreate)
        self.reconfigure_policy(policy_or_cfg, device=device)
        self.reconfigure_metrics(metrics)

    def reconfigure_metrics(self, metrics: EvaluatorMetrics) -> None:
        """Reconfigure the metrics with a new set of metrics.

        The evaluator borrows the supplied metric surface. Replacing metrics
        detaches the old surface but does not call ``close()`` on it or on its
        child metrics.

        Args:
            metrics (EvaluatorMetrics): Metric surface borrowed by the
                evaluator.
        """
        if not isinstance(metrics, EvaluatorMetrics):
            raise TypeError(
                "PolicyEvaluator metrics must be provided as "
                "EvaluatorMetrics. Use EvaluatorMetrics.from_metric(...) or "
                "EvaluatorMetrics.from_metric_dict(...) for shorthand "
                f"construction. Got {type(metrics).__name__}."
            )
        self._evaluator_metrics_value = metrics
        self._recovery_manager.update_metrics(metrics)

    def reconfigure_env(
        self,
        env_cfg: EnvBaseCfg,
        *,
        force_recreate: bool | None = None,
    ) -> None:
        """Reconfigure the environment with a new configuration.

        We only provide reconfiguration via configuration here because for
        most cases, the environment does not support pickling/unpickling.
        By default the evaluator follows
        ``cfg.reconfigure_env_force_recreate``. When recreation is not forced,
        the evaluator skips env teardown only if the current env exposes a
        ``cfg`` whose JSON string matches ``env_cfg``.

        Args:
            env_cfg (EnvBaseCfg): The new configuration for the environment.
            force_recreate (bool | None, optional): ``True`` always closes
                and recreates the env. ``False`` reuses the current env when
                its JSON config string matches ``env_cfg``. ``None`` follows
                ``cfg.reconfigure_env_force_recreate``. Default is None.
        """
        if force_recreate is None:
            force_recreate = self.cfg.reconfigure_env_force_recreate
        if not force_recreate and self._env is not None:
            current_env = cast(_ConfigBackedEnv, self._env)
            if current_env.cfg.to_str(format="json") == env_cfg.to_str(
                format="json"
            ):
                self._recovery_manager.update_env_cfg(env_cfg)
                return

        old_env = self._env
        if old_env is not None:
            _close_env(old_env)
            self._env = None
            self._refresh_env_finalizer()
        self.env = env_cfg()
        self._recovery_manager.update_env_cfg(env_cfg)

    def reconfigure_policy(
        self,
        policy_or_cfg: PolicyConfig | PolicyMixin,
        device: str | torch.device | None = None,
    ) -> None:
        """Reconfigure the policy with a new configuration.

        The policy public contract does not include ``close()``. Replacing a
        policy detaches the old policy object without closing it, regardless
        of whether it was created from a config or supplied directly.

        Args:
            policy_or_cfg (PolicyConfig | PolicyMixin): The new configuration
                for the policy or a policy instance.
        """
        if isinstance(policy_or_cfg, PolicyMixin):
            self._policy = policy_or_cfg
        else:
            self._policy = policy_or_cfg()
        if device is not None:
            self.policy.to(device=device)
        self._recovery_manager.update_policy(
            policy_or_cfg,
            device=device,
        )

    def evaluate_episode(
        self,
        max_steps: int,
        env_reset_input: EnvStartInput = None,
        policy_reset_input: PolicyResetInput = None,
        rollout_stop_condition: RollOutStopCondition = evaluate_rollout_stop_condition,  # noqa: E501
    ) -> dict[str, Any]:
        """Evaluate one complete episode and return computed metrics.

        The return value is the current ``EvaluatorMetrics.compute()`` result
        after this episode finishes and terminal metrics have been updated.
        It is not guaranteed to be a per-episode delta: cumulative metrics
        may include state from earlier successful episodes until callers reset
        or replace the metric surface.

        This is the non-streaming evaluator path. If one episode attempt
        fails, the evaluator restores metric state to the pre-episode value
        before re-raising the public execution error. Env and policy still
        follow their normal per-episode start contract; PreparedEnvStart skips
        env reset but still resets policy. This method does not promise
        generic env or policy runtime rollback.

        If the env implements ``finalize_episode()``, the evaluator calls it
        once through best-effort env finalization when the episode attempt
        exits, including error paths. Finalization failures are logged and do
        not replace the episode result or execution error.

        Args:
            max_steps: Maximum number of environment steps to run before the
                episode is treated as max-step terminated.
            env_reset_input: Episode-start input. ``None`` resets the env
                normally, ``dict`` is forwarded as reset kwargs, ``State``
                calls the env state-reset path, and ``PreparedEnvStart``
                starts from an already-reset observation.
            policy_reset_input: Optional kwargs for the per-episode policy
                reset.
            rollout_stop_condition: Stop predicate passed to the env rollout.

        Returns:
            The dict produced by ``compute_metrics()`` after the episode.
        """
        request = self._make_evaluation_request(
            max_steps=max_steps,
            env_reset_input=env_reset_input,
            policy_reset_input=policy_reset_input,
            rollout_steps=max_steps,
            rollout_stop_condition=rollout_stop_condition,
        )
        pre_episode_metric_state = self._evaluator_metrics.get_state()
        try:
            return evaluate_policy_episode(self, request).metrics
        except PolicyEvaluationExecutionError:
            self._evaluator_metrics.load_state(pre_episode_metric_state)
            raise

    def make_episode_evaluation(
        self,
        max_steps: int,
        env_reset_input: EnvStartInput = None,
        policy_reset_input: PolicyResetInput = None,
        rollout_steps: int = 5,
        rollout_stop_condition: RollOutStopCondition = evaluate_rollout_stop_condition,  # noqa: E501
    ) -> Generator[int, None, None]:
        """Yield rollout batch sizes while evaluating one episode.

        The evaluator resets env and policy once at episode start, then
        yields each rollout batch size until the episode terminates or reaches
        `max_steps`. Metrics dispatch according to the configured
        `EvaluatorMetrics` channels:
        - `STEP` metrics update during env callbacks
        - `TERMINAL` metrics update on the final step

        Args:
            max_steps (int): The maximum number of steps to evaluate
                the policy for.
            env_reset_input (EnvStartInput, optional): Episode-start input.
                Dict values call ``env.reset(**kwargs)``; State values call
                ``env.reset_from_state(state)``; PreparedEnvStart values use
                the provided observation without resetting env. Defaults to
                None.
            policy_reset_input (dict, optional): Policy reset input.
                Dict values call ``policy.reset(**kwargs)``. Defaults to None.
            rollout_steps (int, optional): The number of steps to roll
                out in each iteration. Defaults to 5.
        yields:
            int: The number of steps taken in each rollout.

        This streaming interface does not provide transparent rollback or
        replay on failure. Once rollout counts have been yielded, the caller
        should treat that partial output as already published.

        If the env implements ``finalize_episode()``, the generator finalizes
        it once through best-effort env finalization when the episode loop
        exits. To release episode-local artifacts promptly, callers must
        either consume this generator to completion or explicitly close it.

        """
        request = self._make_evaluation_request(
            max_steps=max_steps,
            env_reset_input=env_reset_input,
            policy_reset_input=policy_reset_input,
            rollout_steps=rollout_steps,
            rollout_stop_condition=rollout_stop_condition,
        )
        yield from _run_policy_episode_loop(self, request)

    def _make_evaluation_request(
        self,
        max_steps: int,
        env_reset_input: EnvStartInput,
        policy_reset_input: PolicyResetInput,
        rollout_steps: int,
        rollout_stop_condition: RollOutStopCondition,
    ) -> EvaluationRequest:
        return EvaluationRequest(
            max_steps=max_steps,
            rollout_steps=rollout_steps,
            env_reset_input=env_reset_input,
            policy_reset_input=policy_reset_input,
            rollout_stop_condition=rollout_stop_condition,
        )

    def reset_metrics(self, **kwargs) -> None:
        """Reset all metrics.

        Args:
            kwargs: Additional arguments to pass to the
                metrics' reset method.
        """
        if self._evaluator_metrics_value is None:
            raise RuntimeError(
                "Metrics are not configured. Cannot reset metrics."
            )
        return self._evaluator_metrics.reset(**kwargs)

    def reset_env(
        self,
        env_reset_input: EnvResetInput = None,
        **kwargs: Any,
    ) -> Any:
        """Reset the environment from a canonical reset-triggering input.

        ``None`` calls ``env.reset()``. Dict inputs call
        ``env.reset(**env_reset_input)``. State inputs require POST_RESET env
        state support and call ``env.reset_from_state(state)``. Legacy
        ``reset_env(**kwargs)`` calls are normalized to the dict path.

        Args:
            env_reset_input (EnvResetInput, optional): Reset input that
                actually triggers an env reset. Defaults to None.
            kwargs: Legacy env reset keyword arguments.
        """

        if self._env is None:
            raise RuntimeError(
                "Environment is not configured. Cannot reset environment."
            )
        if kwargs:
            if env_reset_input is not None:
                raise TypeError(
                    "reset_env() accepts either env_reset_input or legacy "
                    "reset kwargs, not both."
                )
            env_reset_input = dict(kwargs)

        if env_reset_input is None:
            return self._env.reset()
        if isinstance(env_reset_input, dict):
            return self._env.reset(**dict(env_reset_input))
        if isinstance(env_reset_input, State):
            stateful_env = require_env_supports_state_scope(
                self._env,
                EnvStateScope.POST_RESET,
            )
            return stateful_env.reset_from_state(env_reset_input)
        raise TypeError(
            "env_reset_input must be dict, State, or None. "
            f"Got {type(env_reset_input).__name__}."
        )

    def reset_policy(self, **kwargs) -> None:
        """Reset the policy.

        Args:
            kwargs: Additional arguments to pass to the policy's
                reset method.

        """

        if self._policy is None:
            raise RuntimeError(
                "Policy is not configured. Cannot reset policy."
            )
        return self._policy.reset(**kwargs)

    def get_metrics(self) -> EvaluatorMetrics | None:
        """Get the current metrics.

        Returns:
            EvaluatorMetrics | None: The configured metric surface,
                or None when the evaluator has not been set up yet.
        """
        if self._evaluator_metrics_value is None:
            return None
        return self._evaluator_metrics

    def compute_metrics(self) -> dict[str, Any]:
        """Compute all configured metrics and return the public result."""
        return self._evaluator_metrics.compute()

    def _export_metric_runtime_state(self) -> State:
        """Capture evaluator metric runtime state for private rollback."""
        return copy.deepcopy(self._evaluator_metrics.get_state())

    def _restore_metric_runtime_state(self, state: State) -> None:
        """Restore evaluator metric runtime state for private rollback."""
        if not isinstance(state, State):
            raise TypeError(
                "Metric runtime state must be a State payload. "
                f"Got {type(state).__name__}."
            )
        self._evaluator_metrics.load_state(copy.deepcopy(state))

    def _export_recovery_snapshot(self) -> PolicyEvaluatorRecoverySnapshot:
        """Capture one evaluator-owned recovery snapshot."""
        return self._recovery_manager.export_snapshot(self)

    def _restore_recovery_snapshot(
        self,
        snapshot: PolicyEvaluatorRecoverySnapshot,
    ) -> None:
        """Restore one evaluator-owned recovery snapshot."""
        self._recovery_manager.restore_snapshot(self, snapshot)


class PolicyEvaluatorConfig(ClassConfig[PolicyEvaluator]):
    """Configure construction and default env reuse for PolicyEvaluator.

    ``PolicyEvaluatorConfig`` intentionally does not store env, policy, or
    metric instances. Those are runtime resources passed to
    :meth:`PolicyEvaluator.setup` or the individual reconfigure methods.
    Keeping the config focused on evaluator behavior lets the same evaluator
    config be reused by local and remote wrappers while callers still choose
    the episode resources at setup time.

    Use :meth:`as_remote` to wrap the same local evaluator configuration in a
    :class:`PolicyEvaluatorRemoteConfig` without duplicating env-reuse
    defaults.

    """

    class_type: type[PolicyEvaluator] = PolicyEvaluator

    reconfigure_env_force_recreate: bool = True
    """Default env recreation policy for ``reconfigure_env``.

    ``True`` preserves the historical behavior: every reconfigure call closes
    the current env and creates a fresh one. ``False`` allows
    ``reconfigure_env(force_recreate=None)`` to reuse the current env when it
    exposes a ``cfg`` that serializes to the same JSON env config string.
    """

    def as_remote(
        self,
        remote_class_config: RayRemoteClassConfig | None = None,
        ray_init_config: dict[str, Any] | None = None,
        check_init_timeout: int = 60,
        rollout_timeout_s: float | None = 120.0,
        reset_timeout_s: float | None = 1200.0,
        timeout_grace_retries: int = 1,
    ) -> PolicyEvaluatorRemoteConfig:
        """Build a remote evaluator config that uses this local config.

        Args:
            remote_class_config (RayRemoteClassConfig, optional): Ray actor
                resource config for the remote wrapper. Default is a default
                ``RayRemoteClassConfig`` instance.
            ray_init_config (dict[str, Any] | None, optional): Optional Ray
                initialization config forwarded to the remote instance.
                Default is None.
            check_init_timeout (int, optional): Timeout for remote actor
                initialization checks. Default is 60.
            rollout_timeout_s (float | None, optional): Default remote
                rollout timeout. Default is 120.0.
            reset_timeout_s (float | None, optional): Default remote setup,
                reset, recovery, and metric-call timeout. Default is 1200.0.
            timeout_grace_retries (int, optional): Extra timeout waits before
                reporting a remote timeout. Default is 1.

        Returns:
            PolicyEvaluatorRemoteConfig: Remote facade config that references
            this local evaluator config.
        """

        from robo_orchard_lab.policy.evaluator.remote import (
            PolicyEvaluatorRemoteConfig,
        )

        if remote_class_config is None:
            remote_class_config = RayRemoteClassConfig()
        return PolicyEvaluatorRemoteConfig(
            instance_config=self,
            remote_class_config=remote_class_config,
            ray_init_config=ray_init_config,
            check_init_timeout=check_init_timeout,
            rollout_timeout_s=rollout_timeout_s,
            reset_timeout_s=reset_timeout_s,
            timeout_grace_retries=timeout_grace_retries,
        )
