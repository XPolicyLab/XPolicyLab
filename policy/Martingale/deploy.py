# Derived from the PyRUA_Lean policy adapter in XPolicyLab (Apache License 2.0); see runtime/THIRD_PARTY_NOTICES.md.
# Modifications Copyright © 2026 Martingale. All Rights Reserved.
# This file has been modified by Martingale.
"""One environment per process; bounded automatic retry of safe reads only.

The policy answers WAIT while the model is reasoning; the loop then checks for the episode end and
asks again without stepping the simulator. Reading the observation, the episode-end check and reading
the verdict do not change anything, so a failed attempt is repeated up to SAFE_ATTEMPTS times,
SAFE_BACKOFF_S seconds apart. Every other call (reset, update_obs, get_action, action validation,
take_action, trial_end) runs once. Its exception, or the last exception of a safe read whose attempts
are used up, propagates to the evaluation client, as in XPolicyLab's demo_policy. There is no
operator entry point and no way to supply an action, a result or a verdict from outside.
"""

from __future__ import annotations

from typing import Any, Callable
import os
import time

import numpy as np
from XPolicyLab.utils.process_data import get_robot_action_dim_info, get_robot_state_schema

SAFE_ATTEMPTS = 3      # attempts of a read that changes nothing
SAFE_BACKOFF_S = 1.5   # pause between two attempts (seconds)


def _pictures_behind() -> int:
    """How many observations the client's pictures run behind its state.

    RoboDojo's client used to hand out the frame rendered before the current physics state:
    an observation's pictures showed the robot one observation earlier than its state did
    (RoboDojo issue 38).  Since ``env/camera_manager/capture/render_sync.py``
    it renders the current state before it captures.  The policy pairs a picture with the
    state it shows, so it has to know which client this is.
    """

    try:
        from env.camera_manager.capture import render_sync  # noqa: F401 - RoboDojo's own
    except Exception:  # noqa: BLE001 - an older client, or not a RoboDojo client at all
        return 1
    return 0


def _status(task_env: Any, env_idx: int = 0) -> dict[str, Any]:
    info: dict[str, Any] = {"pictures_behind": _pictures_behind()}
    # RoboDojo's environment counts per environment; XPolicyLab's debug environment has one
    # counter and its own names
    taken = getattr(task_env, "take_action_cnt", None)
    try:
        info["step"] = int(taken[env_idx])
    except (TypeError, IndexError, ValueError):
        if isinstance(getattr(task_env, "episode_step", None), int):
            info["step"] = task_env.episode_step
    return info


def _waiting(actions: Any) -> bool:
    return (
        isinstance(actions, (list, tuple))
        and len(actions) == 1
        and isinstance(actions[0], dict)
        and actions[0].get("martingale") == "wait"
    )


def _safe(operation: Callable[[], Any]) -> Any:
    """Run a read that changes nothing; repeat it after a short pause when it raises."""
    for attempt in range(1, SAFE_ATTEMPTS + 1):
        try:
            return operation()
        except Exception:  # noqa: BLE001 - any read failure; the last one propagates
            if attempt == SAFE_ATTEMPTS:
                raise
            time.sleep(SAFE_BACKOFF_S)


def _episode_verdict(task_env: Any) -> dict[str, Any]:
    if not task_env.is_episode_end():
        raise RuntimeError("Episode is still active")
    if os.environ.get("EVAL_ENV_TYPE") == "debug":
        if task_env.episode_step < task_env.episode_step_limit:
            return {**_status(task_env), "interrupted": True}
        return {**_status(task_env), "debug_completed": True}
    success = task_env.success[0]
    if not isinstance(success, (bool, np.bool_)):
        raise ValueError("Missing boolean verdict; cannot invent a failure")
    return {**_status(task_env), "success": bool(success)}


def _validate(actions: Any, schema: dict[str, int]) -> None:
    if not isinstance(actions, (list, tuple)) or not actions or not all(isinstance(a, dict) for a in actions):
        raise ValueError("Empty or malformed policy action")
    for action in actions:
        if action.keys() != schema.keys():
            raise ValueError(f"Expected action fields {list(schema)}, got {list(action)}")
        for key, width in schema.items():
            value = np.asarray(action[key])
            if value.shape != (width,) or value.dtype.kind not in "iuf" or not np.isfinite(value).all():
                raise ValueError(f"{key} requires a finite real vector of length {width}")


def eval_one_episode(TASK_ENV: Any, model_client: Any) -> None:
    # Robot metadata is installed on the environment host in split deployments.
    env_cfg_type = TASK_ENV.deploy_cfg.get("env_cfg_type") or "arx_x5"
    schema = get_robot_state_schema("joint", get_robot_action_dim_info(env_cfg_type))
    model_client.call(func_name="reset")
    while not _safe(TASK_ENV.is_episode_end):
        obs = _safe(TASK_ENV.get_obs)
        info = dict(obs.get("additional_info") or {})
        info["martingale"] = _status(TASK_ENV)
        obs["additional_info"] = info
        model_client.call(func_name="update_obs", obs=obs)
        while True:
            actions = model_client.call(func_name="get_action")
            if (isinstance(actions, (list, tuple)) and len(actions) == 1
                    and isinstance(actions[0], dict) and actions[0].get('martingale') == 'geometry_fitting_complete'):
                # Calibration has its own terminal receipt, never a task verdict.
                model_client.call(func_name="trial_end", obs={"geometry_fitting": "complete"})
                return
            if _waiting(actions):
                if _safe(TASK_ENV.is_episode_end):
                    break
                continue
            _validate(actions, schema)
            break
        if _safe(TASK_ENV.is_episode_end):
            break
        for action in actions:
            TASK_ENV.take_action(action)
            if _safe(TASK_ENV.is_episode_end):
                break
    if os.environ.get('MARTINGALE_RUN_GEOMETRY_FITTING') == '1':
        model_client.call(func_name="trial_end", obs={"geometry_fitting": "incomplete"})
        return
    verdict = _safe(lambda: _episode_verdict(TASK_ENV))
    model_client.call(func_name="trial_end", obs=verdict)


def eval_one_episode_batch(TASK_ENV: Any, model_client: Any) -> None:
    num_envs = int(getattr(TASK_ENV, "num_envs", 1) or 1)
    if num_envs > 1:
        raise RuntimeError(
            f"RoboDojo started {num_envs} environments; Martingale drives one conversation per "
            "environment.  Keep eval_batch: false in deploy.yml."
        )
    eval_one_episode(TASK_ENV, model_client)
