# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os
import pathlib
import re
import subprocess
from typing import Any, Dict
import wandb
from wandb import Settings
from awomo.awomo.training.distributed.init_utils import initialize_distributed


def get_git_commit(directory: pathlib.Path) -> str | None:
    revision = None
    try:
        if (directory / ".git").exists():
            revision = subprocess.run(
                ["git", "-c", f"safe.directory={directory}", "rev-parse", "HEAD"],
                cwd=directory, check=True, capture_output=True, text=True, timeout=5,
            ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    revision = revision or os.environ.get("GIT_COMMIT", "")
    return revision if re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", revision) else None


def _get_model_name(cfg_model):
    if cfg_model.get("pretrained_model_name_or_path", None) is not None:
        return cfg_model.pretrained_model_name_or_path
    elif cfg_model.get("config", None) is not None:
        if isinstance(cfg_model.config, str):
            return cfg_model.config
        return cfg_model.config.get("pretrained_model_name_or_path", None)
    else:
        return None


def build_distributed(cfg_dist: Dict[str, Any]) -> "DistInfo":  # noqa: F821
    backend = cfg_dist.get("backend", "nccl")
    timeout = cfg_dist.get("timeout_minutes", 1)
    return initialize_distributed(backend=backend, timeout_minutes=timeout)


def build_wandb(cfg) -> wandb.Run:
    assert cfg.get("wandb", None) is not None
    kwargs = cfg.wandb.to_dict()
    if kwargs.get("name", "") == "":
        kwargs["name"] = "_".join(_get_model_name(cfg.model).split("/")[-2:])
    kwargs.setdefault("save_code", True)
    config = cfg.to_dict()
    revision = get_git_commit(pathlib.Path(__file__).resolve().parents[2])
    if revision:
        config["git_commit"] = revision
    if os.environ.get("SOURCE_DIGEST"):
        config["source_digest"] = os.environ["SOURCE_DIGEST"]
    run = wandb.init(
        **kwargs,
        config=config,
        settings=Settings(silent=True),
    )
    return run
