"""Trial execution: env backends, config, and policy episode runner."""

from robodojo.trial.cli import add_trial_env_arguments
from robodojo.trial.env import (
    DebugTrialEnv,
    TrialEnv,
    build_mock_observation,
    validate_robot_state_dict,
)
from robodojo.trial.runner import (
    ActionRecorder,
    TrialRunConfig,
    build_trial_run_config,
    create_trial_env,
    normalize_policy_name,
    run_eval_episode,
    run_policy_trial,
)
from robodojo.trial.sim import SimEnvConfig, create_sim_trial_env, resolve_sim_env_factory

__all__ = [
    "ActionRecorder",
    "DebugTrialEnv",
    "SimEnvConfig",
    "TrialEnv",
    "TrialRunConfig",
    "add_trial_env_arguments",
    "build_mock_observation",
    "build_trial_run_config",
    "create_sim_trial_env",
    "create_trial_env",
    "normalize_policy_name",
    "resolve_sim_env_factory",
    "run_eval_episode",
    "run_policy_trial",
    "validate_robot_state_dict",
]
