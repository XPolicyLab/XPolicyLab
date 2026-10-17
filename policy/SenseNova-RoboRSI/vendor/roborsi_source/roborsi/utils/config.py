# Copyright 2026 The HEART Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Path resolution and environment-variable configuration."""

from __future__ import annotations

import os
from pathlib import Path


def get_project_env(name: str, default: str | None = None) -> str | None:
    """Read the explicitly named environment variable without aliases."""
    return os.environ.get(name, default)


def require_project_env(name: str) -> str:
    """Read required environment configuration, raising KeyError when absent."""
    value = get_project_env(name)
    if value is None:
        raise KeyError(name)
    return value


# ============================================================================
# Repository / package roots
# ============================================================================


def get_repo_root() -> Path:
    """Return the RoboRSI repository root directory.

    Resolution: ``ROBORSI_REPO_ROOT``, then the parent of the ``roborsi/`` package.
    """
    env = get_project_env("ROBORSI_REPO_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    # config.py lives at <repo>/roborsi/utils/config.py
    return Path(__file__).resolve().parents[2]


# ============================================================================
# Paths derived from the repo root  (callable so tests can override)
# ============================================================================


def get_memory_dir(robot_name: str) -> Path:
    """Return the persistent, cross-run memory directory for a robot."""
    return get_repo_root() / "memory" / robot_name


def get_pi05_checkpoint_path() -> str:
    return os.environ.get("PI05_CHECKPOINT_PATH", "")


def get_libero_type() -> str:
    return os.environ.get("LIBERO_TYPE", "pro")


def get_rlinf_repo_path() -> Path:
    """Return the RLinf path that RPC servers import."""
    env = get_project_env("ROBORSI_RLINF_ROOT") or os.environ.get("RLINF_REPO_PATH")
    if env:
        return Path(env).expanduser().resolve()
    return (get_repo_root().parent / "rlinf").resolve()
