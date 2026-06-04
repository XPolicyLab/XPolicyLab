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

"""Env-domain contracts for State-backed episode starts."""

from __future__ import annotations
from collections.abc import Iterable
from enum import Enum
from typing import Any, Protocol, cast, runtime_checkable

from robo_orchard_lab.utils.state import (
    State,
    StateRuntimeMixin,
    StateRuntimeProtocol,
    validate_recovery_state,
)

__all__ = [
    "ENV_STATE_SCOPE_KEY",
    "EnvStateScope",
    "StatefulEnvMixin",
    "StatefulEnvProtocol",
    "require_env_supports_state_scope",
]

ENV_STATE_SCOPE_KEY = "scope"


class EnvStateScope(str, Enum):
    """Lifecycle scope described by an env runtime State payload."""

    POST_RESET = "post_reset"
    MID_EPISODE = "mid_episode"


@runtime_checkable
class StatefulEnvProtocol(StateRuntimeProtocol, Protocol):
    """Env capability for starting an episode from a State payload.

    ``reset_from_state(...)`` is intentionally separate from generic
    ``load_state(State)``: evaluators need a reset-shaped ``(obs, info)``
    result to begin rollout, while ``load_state(...)`` remains the generic
    no-return runtime apply API.
    """

    supported_state_scopes: frozenset[EnvStateScope]

    def reset_from_state(self, state: State) -> tuple[Any, dict[str, Any]]:
        """Restore an episode start and return the reset-shaped result."""
        ...


class StatefulEnvMixin(StateRuntimeMixin):
    """Reusable env State helpers without default env-state serialization."""

    supported_state_scopes: frozenset[EnvStateScope] = frozenset()

    def _get_state(self) -> State:
        raise NotImplementedError(
            f"{type(self).__name__} must define an explicit env State schema."
        )

    def _set_state(self, state: State) -> None:
        del state
        raise NotImplementedError(
            f"{type(self).__name__} must define explicit env State restore."
        )

    @staticmethod
    def _normalize_state_scopes(
        scopes: Iterable[EnvStateScope | str],
    ) -> frozenset[EnvStateScope]:
        try:
            return frozenset(EnvStateScope(item) for item in scopes)
        except TypeError as exc:
            raise TypeError(
                "Env supported_state_scopes must be an iterable of "
                "EnvStateScope values."
            ) from exc
        except ValueError as exc:
            raise ValueError(
                "Env supported_state_scopes contains an unsupported scope."
            ) from exc

    @staticmethod
    def get_env_state_scope(state: State) -> EnvStateScope:
        """Return the env lifecycle scope carried by a State payload."""

        validate_recovery_state(state, context="Env state")
        payload = state.state
        if ENV_STATE_SCOPE_KEY not in payload:
            raise ValueError(
                f"Env state requires `{ENV_STATE_SCOPE_KEY}` in State.state."
            )
        raw_scope = payload[ENV_STATE_SCOPE_KEY]
        try:
            return EnvStateScope(raw_scope)
        except ValueError as exc:
            raise ValueError(
                f"Unsupported env state scope: {raw_scope!r}."
            ) from exc

    @classmethod
    def require_supported_state_scope(cls, scope: EnvStateScope) -> None:
        """Validate that this env class declares support for a scope."""

        normalized_scopes = cls._normalize_state_scopes(
            cls.supported_state_scopes
        )
        if scope not in normalized_scopes:
            raise ValueError(
                f"Env {cls.__name__} does not support state scope "
                f"{scope.value!r}."
            )


def require_env_supports_state_scope(
    env: object,
    scope: EnvStateScope,
) -> StatefulEnvProtocol:
    """Validate that an env can start an episode from the requested scope."""

    if not isinstance(env, StatefulEnvProtocol):
        raise TypeError(
            "Env state reset requires StatefulEnvProtocol "
            "(`supported_state_scopes`, State runtime methods, and "
            f"`reset_from_state(...)`). Got {type(env).__name__}."
        )

    stateful_env = cast(StatefulEnvProtocol, env)
    normalized_scopes = StatefulEnvMixin._normalize_state_scopes(
        stateful_env.supported_state_scopes
    )
    if scope not in normalized_scopes:
        raise ValueError(
            f"Env {type(env).__name__} does not support state scope "
            f"{scope.value!r}."
        )
    return stateful_env
