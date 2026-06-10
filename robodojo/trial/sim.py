"""Load simulation trial environments from a pluggable factory."""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from robodojo.trial.env import TrialEnv


@dataclass(frozen=True)
class SimEnvConfig:
    task_name: str
    env_cfg_type: str
    seed: int | None = None
    device_id: str | None = None
    additional_info: str | None = None
    case_meta: dict[str, Any] | None = None
    root_dir: str | None = None
    sim_env_factory: str | None = None


def _import_callable(path: str) -> Callable[..., TrialEnv]:
    module_path, _, attr = path.partition(":")
    if not module_path or not attr:
        raise ValueError(
            f"Invalid sim env factory {path!r}; expected 'module.path:callable'"
        )
    module = importlib.import_module(module_path)
    factory = getattr(module, attr)
    if not callable(factory):
        raise TypeError(f"{path} is not callable")
    return factory


def resolve_sim_env_factory(config: SimEnvConfig) -> str | None:
    if config.sim_env_factory:
        return config.sim_env_factory
    env_factory = os.environ.get("ROBODOJO_SIM_ENV_FACTORY")
    if env_factory:
        return env_factory
    if config.root_dir:
        candidate = Path(config.root_dir) / "scripts" / "robodojo_sim_env.py"
        if candidate.is_file():
            return "robodojo_sim_env:create_trial_env"
    return None


def create_sim_trial_env(config: SimEnvConfig) -> TrialEnv:
    factory_path = resolve_sim_env_factory(config)
    if factory_path is None:
        raise RuntimeError(
            "eval_env=sim requires a simulation factory. Set dispatch.sim_env_factory, "
            "ROBODOJO_SIM_ENV_FACTORY, or place scripts/robodojo_sim_env.py under root_dir."
        )

    if config.root_dir:
        root = str(Path(config.root_dir).resolve())
        import sys

        if root not in sys.path:
            sys.path.insert(0, root)

    factory = _import_callable(factory_path)
    return factory(
        task_name=config.task_name,
        env_cfg_type=config.env_cfg_type,
        seed=config.seed,
        device_id=config.device_id,
        additional_info=config.additional_info,
        case_meta=config.case_meta or {},
    )
