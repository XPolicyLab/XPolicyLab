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

"""Lab env base contracts and lightweight core env re-exports."""

from __future__ import annotations
from typing import Protocol, runtime_checkable

from robo_orchard_core.envs.env_base import EnvBase, EnvBaseCfg, EnvStepReturn
from robo_orchard_core.utils.logging import LoggerManager

logger = LoggerManager().get_child(__name__)

__all__ = [
    "EnvBase",
    "EnvBaseCfg",
    "EnvStepReturn",
    "EpisodeFinalizableEnvProtocol",
    "finalize_env_episode",
]


@runtime_checkable
class EpisodeFinalizableEnvProtocol(Protocol):
    """Env capability for finalizing episode-local resources.

    Implementations must make this method idempotent. Calling it when no
    episode is active, after reset failure, or after a previous finalization
    should be safe. Finalization must not close the environment runtime.
    """

    def finalize_episode(self) -> None:
        """Finalize episode-local resources without closing the environment."""
        ...


def finalize_env_episode(env: object) -> None:
    """Best-effort finalize one env episode when the env supports it."""

    if not isinstance(env, EpisodeFinalizableEnvProtocol):
        return
    try:
        env.finalize_episode()
    except Exception:
        logger.exception("Failed to finalize env episode.")
