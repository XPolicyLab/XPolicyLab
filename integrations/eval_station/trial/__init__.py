"""Dispatch → deploy_cfg mapping helpers."""

from eval_station.trial.config import (
    TrialRunConfig,
    build_trial_run_config,
    normalize_policy_name,
)

__all__ = [
    "TrialRunConfig",
    "build_trial_run_config",
    "normalize_policy_name",
]
