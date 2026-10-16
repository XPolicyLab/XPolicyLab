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

"""Native termination shared by direct and remote RoboDojo owners."""

from typing import Any

from roborsi.utils.logging import get_logger

logger = get_logger("robodojo.episode")


def mark_incomplete_episode_failed(task_env: Any) -> None:
    """Apply RoboProbe's early-stop rule while retaining native final checks."""
    if task_env.is_episode_end():
        return
    running = (
        task_env.get_running_env_idx_list()
        if hasattr(task_env, "get_running_env_idx_list")
        else [0]
    )
    success = getattr(task_env, "success", None)
    if success is None:
        raise RuntimeError(
            "L3 inspect policy stopped before official termination and the "
            "environment does not expose a failure state."
        )
    for env_idx in running:
        success[env_idx] = False
    task_env.is_episode_end()
    logger.info("policy stopped early; marked envs failed: %s", running)
