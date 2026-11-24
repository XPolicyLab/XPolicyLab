"""The evaluation loop of the PyRUA_Lean adapter.

The policy is a program a language model writes while the episode runs: between two actions it
may think for minutes, and it decides itself when it has finished.  The loop is demo_policy's
with three differences, each because of that:

* ``get_action`` is asked again while it answers "wait" (the policy server never holds one
  request longer than a few seconds, so the client's request timeout stays what it is);
* an empty answer means the program has finished: the episode ends there and RoboDojo judges
  the scene as it stands;
* every observation carries the episode's step count and limit, which the program needs to
  budget its motions, and whether this client's pictures run an observation behind its state
  (``additional_info["pyrualean"]``); the benchmark's verdict is reported to the policy
  server once the episode is over (the protocol's ``trial_end``).

Nothing else is read from the environment: the observation is the one RoboDojo's client
builds, the actions are joint targets.
"""

from __future__ import annotations

from typing import Any


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
    for name in ("task_name", "eval_seed"):
        value = getattr(task_env, name, None)
        if value is not None:
            info[name] = value
    seeds = getattr(task_env, "env_seeds", None)
    try:
        info["layout"] = int(seeds[env_idx])
    except (TypeError, IndexError, ValueError):
        pass
    return info


def _waiting(actions: Any) -> bool:
    return (
        isinstance(actions, (list, tuple))
        and len(actions) == 1
        and isinstance(actions[0], dict)
        and actions[0].get("pyrualean") == "wait"
    )


def _judge_as_it_stands(task_env: Any) -> None:
    """End an episode the program has finished: RoboDojo runs its final checks now."""

    if task_env.is_episode_end() or not hasattr(task_env, "success"):
        return  # over already, or an environment with nothing to judge (the debug one)
    running = (
        task_env.get_running_env_idx_list()
        if hasattr(task_env, "get_running_env_idx_list")
        else [0]
    )
    for env_idx in running:
        task_env.success[env_idx] = False
    task_env.is_episode_end()


def eval_one_episode(TASK_ENV: Any, model_client: Any) -> None:
    model_client.call(func_name="reset")
    try:
        while not TASK_ENV.is_episode_end():
            obs = TASK_ENV.get_obs()
            info = dict(obs.get("additional_info") or {})
            info["pyrualean"] = _status(TASK_ENV)
            obs["additional_info"] = info
            model_client.call(func_name="update_obs", obs=obs)
            actions = model_client.call(func_name="get_action")
            while _waiting(actions):
                actions = model_client.call(func_name="get_action")
            if not actions:
                break  # the program has finished
            for action in actions:
                TASK_ENV.take_action(action)
                if TASK_ENV.is_episode_end():
                    break
    finally:
        _judge_as_it_stands(TASK_ENV)
        verdict = dict(_status(TASK_ENV))
        try:
            verdict["success"] = bool(TASK_ENV.success[0])
        except (TypeError, IndexError, AttributeError):
            verdict["success"] = False
        try:
            model_client.call(func_name="trial_end", obs=verdict)
        except Exception as exc:  # noqa: BLE001 - the trial stands whether or not it was heard
            print(f"[PyRUA_Lean] the verdict did not reach the policy server: {exc}", flush=True)


def eval_one_episode_batch(TASK_ENV: Any, model_client: Any) -> None:
    num_envs = int(getattr(TASK_ENV, "num_envs", 1) or 1)
    if num_envs > 1:
        raise RuntimeError(
            f"RoboDojo started {num_envs} environments; PyRUA_Lean drives one conversation per "
            "environment.  Keep eval_batch: false in deploy.yml."
        )
    eval_one_episode(TASK_ENV, model_client)
