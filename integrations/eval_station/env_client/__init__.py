"""Env-side trial execution: HTTP contract, in-process runner, and WS adapter."""

from eval_station.env_client.api import (
    DEBUG_ENV_CLIENT_DEPLOY_CFG_KEYS,
    EnvClientBaselineConfig,
    HealthResponse,
    TrialRunRequest,
    TrialRunResponse,
    baseline_deploy_cfg_view,
    debug_env_client_deploy_cfg_view,
    dispatch_trial_to_deploy_cfg,
    dispatch_trial_to_request,
    trial_request_to_deploy_cfg,
)
from eval_station.env_client.runner import (
    DebugTrialRunner,
    EnvTrialRunner,
    StopCheckFactory,
    TrialRunnerError,
    TrialRunnerFn,
    make_dispatch_trial_runner,
    reset_idle_env,
    run_debug_trial,
    run_real_trial,
)
from eval_station.env_client.ws_adapter import RoboDojoModelClient

__all__ = [
    "DEBUG_ENV_CLIENT_DEPLOY_CFG_KEYS",
    "DebugTrialRunner",
    "EnvTrialRunner",
    "EnvClientBaselineConfig",
    "HealthResponse",
    "RoboDojoModelClient",
    "TrialRunRequest",
    "TrialRunResponse",
    "StopCheckFactory",
    "TrialRunnerError",
    "TrialRunnerFn",
    "baseline_deploy_cfg_view",
    "debug_env_client_deploy_cfg_view",
    "dispatch_trial_to_deploy_cfg",
    "dispatch_trial_to_request",
    "make_dispatch_trial_runner",
    "reset_idle_env",
    "run_debug_trial",
    "run_real_trial",
    "trial_request_to_deploy_cfg",
]
