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
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from robo_orchard_lab.policy.base import PolicyMixin
from robo_orchard_lab.policy.evaluator.metric_contracts import (
    EvaluatorMetrics,
)
from robo_orchard_lab.utils.state import State, validate_recovery_state

__all__ = [
    "PolicyEvaluatorRecoveryManager",
    "PolicyEvaluatorRecoverySnapshot",
]


class _UnsetValue(Enum):
    TOKEN = "UNSET"


_UNSET = _UnsetValue.TOKEN


def _detach_runtime_state(state: State, *, context: str) -> State:
    """Validate and clone a runtime State payload for evaluator snapshots."""

    validate_recovery_state(state, context=context)
    return copy.deepcopy(state)


@dataclass(slots=True)
class PolicyEvaluatorRecoverySnapshot:
    """Evaluator-owned payload for actor recreate and runtime restore.

    Snapshots carry the reconstruction inputs needed to rebuild env, policy,
    and metric surfaces, plus optional runtime ``State`` payloads for policy
    and metrics. The generic evaluator recovery contract recreates envs from
    config; it does not promise generic env runtime-state restore.
    """

    env_cfg: Any = _UNSET
    """Env config used to recreate or reconfigure the evaluator env."""

    policy_recovery_input: Any = _UNSET
    """Policy config or stable policy reconstruction input."""

    metrics_recovery_input: Any = _UNSET
    """Deep-copied EvaluatorMetrics reconstruction metadata."""

    device: Any = _UNSET
    """Optional policy device forwarded during policy reconstruction."""

    policy_runtime_state: Any = _UNSET
    """Optional policy runtime State captured through PolicyMixin."""

    metric_runtime_state: Any = _UNSET
    """Optional EvaluatorMetrics runtime State captured for restore."""

    @property
    def is_complete(self) -> bool:
        return (
            self.env_cfg is not _UNSET
            and self.policy_recovery_input is not _UNSET
            and self.metrics_recovery_input is not _UNSET
        )


class _RecoveryOwner(Protocol):
    _policy: PolicyMixin | None
    _evaluator_metrics_value: EvaluatorMetrics | None

    def setup(
        self,
        env_cfg: Any,
        policy_or_cfg: Any,
        metrics: EvaluatorMetrics,
        device: Any = None,
        *,
        force_recreate: bool | None = None,
    ) -> None: ...

    def reconfigure_env(
        self,
        env_cfg: Any,
        *,
        force_recreate: bool | None = None,
    ) -> None: ...

    def reconfigure_policy(
        self,
        policy_or_cfg: Any,
        device: Any = None,
    ) -> None: ...

    def reconfigure_metrics(self, metrics: EvaluatorMetrics) -> None: ...


class PolicyEvaluatorRecoveryManager:
    """Track evaluator reconstruction metadata and build recovery snapshots.

    ``PolicyEvaluator`` uses this manager behind private export/restore seams
    so local and remote wrappers share one recovery contract. The manager
    records only setup and reconfigure inputs it needs for reconstruction; it
    captures mutable runtime state at snapshot-export time through policy and
    metric ``State`` APIs.
    """

    def __init__(self) -> None:
        self._env_cfg: Any = _UNSET
        self._policy_recovery_input: Any = _UNSET
        self._metrics_recovery_input: Any = _UNSET
        self._device: Any = _UNSET

    def update_env_cfg(self, env_cfg: Any) -> None:
        self._env_cfg = env_cfg

    def update_policy(
        self,
        policy_or_cfg: Any,
        *,
        device: Any,
    ) -> None:
        if isinstance(policy_or_cfg, PolicyMixin):
            policy_cfg = getattr(policy_or_cfg, "cfg", None)
            if policy_cfg is None:
                raise TypeError(
                    "Evaluator recovery requires live policy instances to "
                    "expose a stable `cfg` for reconstruction."
                )
            self._policy_recovery_input = policy_cfg
        else:
            self._policy_recovery_input = policy_or_cfg
        self._device = device

    def update_metrics(self, metrics: EvaluatorMetrics) -> None:
        if not isinstance(metrics, EvaluatorMetrics):
            raise TypeError(
                "PolicyEvaluator recovery requires EvaluatorMetrics "
                "reconstruction metadata. Got "
                f"{type(metrics).__name__}."
            )
        self._metrics_recovery_input = self._clone_metrics_recovery_input(
            metrics
        )

    def export_snapshot(
        self,
        evaluator: _RecoveryOwner,
    ) -> PolicyEvaluatorRecoverySnapshot:
        policy_runtime_state = _UNSET
        metric_runtime_state = _UNSET

        if self._policy_recovery_input is not _UNSET:
            policy = evaluator._policy
            if policy is None:
                raise RuntimeError(
                    "Policy is not configured. Cannot export evaluator "
                    "recovery snapshot."
                )
            policy_runtime_state = _detach_runtime_state(
                policy.get_state(),
                context="policy runtime state snapshot",
            )

        if self._metrics_recovery_input is not _UNSET:
            metrics = evaluator._evaluator_metrics_value
            if metrics is None:
                raise RuntimeError(
                    "Metrics are not configured. Cannot export evaluator "
                    "recovery snapshot."
                )
            metric_runtime_state = _detach_runtime_state(
                metrics.get_state(),
                context="metric runtime state snapshot",
            )

        return PolicyEvaluatorRecoverySnapshot(
            env_cfg=self._copy_recovery_input(self._env_cfg, "env_cfg"),
            policy_recovery_input=self._copy_recovery_input(
                self._policy_recovery_input,
                "policy_recovery_input",
            ),
            metrics_recovery_input=self._copy_field(
                self._metrics_recovery_input
            ),
            device=self._copy_field(self._device),
            policy_runtime_state=policy_runtime_state,
            metric_runtime_state=metric_runtime_state,
        )

    def restore_snapshot(
        self,
        evaluator: _RecoveryOwner,
        snapshot: PolicyEvaluatorRecoverySnapshot,
    ) -> None:
        if not isinstance(snapshot, PolicyEvaluatorRecoverySnapshot):
            raise TypeError(
                "Evaluator recovery snapshot must be a "
                "PolicyEvaluatorRecoverySnapshot payload. Got "
                f"{type(snapshot).__name__}."
            )

        if snapshot.is_complete:
            evaluator.setup(
                env_cfg=snapshot.env_cfg,
                policy_or_cfg=snapshot.policy_recovery_input,
                metrics=snapshot.metrics_recovery_input,
                device=(
                    None if snapshot.device is _UNSET else snapshot.device
                ),
            )
        else:
            if snapshot.env_cfg is not _UNSET:
                evaluator.reconfigure_env(snapshot.env_cfg)
            if snapshot.policy_recovery_input is not _UNSET:
                evaluator.reconfigure_policy(
                    snapshot.policy_recovery_input,
                    device=(
                        None if snapshot.device is _UNSET else snapshot.device
                    ),
                )
            if snapshot.metrics_recovery_input is not _UNSET:
                evaluator.reconfigure_metrics(snapshot.metrics_recovery_input)

        if snapshot.policy_runtime_state is not _UNSET:
            policy = evaluator._policy
            if policy is None:
                raise RuntimeError(
                    "Policy is not configured. Cannot restore evaluator "
                    "policy runtime state."
                )
            policy.load_state(snapshot.policy_runtime_state)

        if snapshot.metric_runtime_state is not _UNSET:
            metrics = evaluator._evaluator_metrics_value
            if metrics is None:
                raise RuntimeError(
                    "Metrics are not configured. Cannot restore evaluator "
                    "metric runtime state."
                )
            metrics.load_state(snapshot.metric_runtime_state)

    def _clone_metrics_recovery_input(
        self,
        metrics: EvaluatorMetrics,
    ) -> EvaluatorMetrics:
        try:
            return copy.deepcopy(metrics)
        except Exception as exception:
            raise TypeError(
                "Evaluator recovery requires metrics that can be deep-copied "
                "as EvaluatorMetrics reconstruction metadata."
            ) from exception

    def _copy_recovery_input(self, value: Any, field_name: str) -> Any:
        if value is _UNSET:
            return _UNSET
        try:
            return copy.deepcopy(value)
        except Exception as exception:
            raise TypeError(
                f"Evaluator recovery could not clone `{field_name}`."
            ) from exception

    def _copy_field(self, value: Any) -> Any:
        if value is _UNSET:
            return _UNSET
        return copy.deepcopy(value)
