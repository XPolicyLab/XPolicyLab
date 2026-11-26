"""The evaluator-side loop for HarnessTree's instruction-selected programs.

Poll through waits and bind each motion to the observation it used. Send only the
policy observation and the control status needed for motion budgeting, RGB/state
alignment and receipts. The evaluator's terminal verdict goes through ``trial_end``.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from .bridge import MOTION_BINDING, STATUS_KEY, receipt_matches, robot_observation
from .instructions import instruction_sources


def _pictures_behind() -> int:
    """How many observations the client's pictures run behind its state.

    RoboDojo's client used to hand out the frame rendered before the current physics state:
    an observation's pictures showed the robot one observation earlier than its state did
    (RoboDojo issue 38).  Since ``env/camera_manager/capture/render_sync.py`` (commit 6e3190f)
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
    for name in ("step_lim", "episode_step_limit"):
        try:
            info["step_limit"] = int(getattr(task_env, name))
            break
        except (TypeError, ValueError, AttributeError):
            continue
    return info


def _policy_observation(obs: dict[str, Any], status: dict[str, Any]) -> dict[str, Any]:
    payload = robot_observation(obs)
    detailed, short = instruction_sources(obs)
    if detailed:
        payload["extra_info"] = {"detailed_instruction": detailed}
    if short:
        payload["instruction"] = short
    payload["additional_info"] = {STATUS_KEY: status}
    return payload


def _waiting(actions: Any) -> bool:
    return (
        isinstance(actions, (list, tuple))
        and len(actions) == 1
        and isinstance(actions[0], dict)
        and bool(actions[0].get("harness_wait"))
    )


def _finalize_episode(task_env: Any) -> None:
    """Preserve the pinned native client's final-check behavior for private runs.

    This state write is a protocol residual, not an evaluator-owned stop API.
    ``trial_end`` only reports the verdict to the policy. Replace this hook with
    an approved evaluator request that runs the final check, seals the episode
    and returns the verdict without an extra motion or a budget change.
    """
    if task_env.is_episode_end() or not hasattr(task_env, "success"):
        return
    running = getattr(task_env, "get_running_env_idx_list", lambda: [0])
    for env_idx in running():
        task_env.success[env_idx] = False
    task_env.is_episode_end()


def eval_one_episode(TASK_ENV: Any, model_client: Any) -> None:
    model_client.call(func_name="reset")
    episode_id = uuid4().hex
    observation_step = 0
    receipt = None
    client_error = None
    try:
        while not TASK_ENV.is_episode_end():
            obs = TASK_ENV.get_obs()
            status = dict(_status(TASK_ENV), episode_id=episode_id,
                          observation_step=observation_step, receipt=receipt)
            model_client.call(func_name="update_obs", obs=_policy_observation(obs, status))
            actions = model_client.call(func_name="get_action")
            while _waiting(actions):
                actions = model_client.call(func_name="get_action")
            if not actions:
                break  # the program has finished
            if len(actions) != 1:
                raise RuntimeError("expected one action per observation")
            action = dict(actions[0])
            binding = action.pop(MOTION_BINDING, None)
            if receipt_matches(binding, episode_id, observation_step):
                TASK_ENV.take_action(action)
                receipt = binding
                observation_step += 1
            else:
                raise RuntimeError("motion is not bound to the current episode observation")
    except Exception as exc:
        client_error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        _finalize_episode(TASK_ENV)
        verdict = dict(_status(TASK_ENV), episode_id=episode_id, receipt=receipt)
        if client_error is not None:
            verdict["client_error"] = client_error
        try:
            verdict["success"] = bool(TASK_ENV.success[0])
        except (TypeError, IndexError, AttributeError):
            verdict["success"] = False
        try:
            model_client.call(func_name="trial_end", obs=verdict)
        except Exception as exc:  # noqa: BLE001 - the trial stands whether or not it was heard
            print(f"[HarnessTree] the verdict did not reach the policy server: {exc}", flush=True)


def eval_one_episode_batch(TASK_ENV: Any, model_client: Any) -> None:
    num_envs = int(getattr(TASK_ENV, "num_envs", 1) or 1)
    if num_envs > 1:
        raise RuntimeError(
            f"RoboDojo started {num_envs} environments; HarnessTree drives one conversation per "
            "environment.  Keep eval_batch: false in deploy.yml."
        )
    eval_one_episode(TASK_ENV, model_client)
