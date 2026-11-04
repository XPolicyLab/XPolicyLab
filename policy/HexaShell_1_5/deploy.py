"""RoboDojo-side episode loop for HexaShell (runs in the simulator's Python; stdlib only).

The policy is a language model that writes and runs code while the episode is on, so this loop
differs from a plain policy loop in three ways:

1. ``get_action`` may answer "wait" (the model is still thinking): ask again, the simulation
   stands still meanwhile.
2. An empty answer means the agent is done: leave the loop and let RoboDojo judge the scene.
3. Every observation carries the task name and the episode's step count and limit (the agent
   budgets its motions with them), and nothing else of the simulator: no seed, layout or scene
   state. The verdict goes back to the policy server when the episode is over, for the record.
"""

from __future__ import annotations

from typing import Any

STATUS_KEY = "hexashell"


def _renders_current_state() -> bool:
    """Newer RoboDojo clients render the current physics state before capturing the pictures;
    older ones hand out the frame of the previous observation."""
    try:
        import importlib

        importlib.import_module("env.camera_manager.capture.render_sync")
        return True
    except Exception:  # noqa: BLE001
        return False


_FRESH_PICTURES = None


def _episode_status(env: Any) -> dict:
    global _FRESH_PICTURES
    if _FRESH_PICTURES is None:
        _FRESH_PICTURES = _renders_current_state()
    status = {"pictures_behind": 0 if _FRESH_PICTURES else 1}
    counts = getattr(env, "take_action_cnt", None)
    if counts is not None:
        try:
            status["step"] = int(counts[0])
        except (TypeError, IndexError, ValueError):
            pass
    elif isinstance(getattr(env, "episode_step", None), int):
        status["step"] = env.episode_step
    for attr in ("step_lim", "episode_step_limit"):
        value = getattr(env, attr, None)
        if value is not None:
            try:
                status["step_limit"] = int(value)
                break
            except (TypeError, ValueError):
                continue
    task = getattr(env, "task_name", None)
    if task is not None:
        status["task_name"] = task
    return status


def _is_wait(answer: Any) -> bool:
    return (isinstance(answer, (list, tuple)) and len(answer) == 1 and isinstance(answer[0], dict)
            and answer[0].get(STATUS_KEY) == "wait")


def _close_out(env: Any) -> None:
    """The agent stopped before RoboDojo ended the episode: run the final judgement now."""
    if env.is_episode_end() or not hasattr(env, "success"):
        return
    running = env.get_running_env_idx_list() if hasattr(env, "get_running_env_idx_list") else [0]
    for idx in running:
        env.success[idx] = False
    env.is_episode_end()


def eval_one_episode(TASK_ENV: Any, model_client: Any) -> None:
    model_client.call(func_name="reset")
    try:
        while not TASK_ENV.is_episode_end():
            obs = TASK_ENV.get_obs()
            extra = dict(obs.get("additional_info") or {})
            extra[STATUS_KEY] = _episode_status(TASK_ENV)
            obs["additional_info"] = extra
            model_client.call(func_name="update_obs", obs=obs)
            answer = model_client.call(func_name="get_action")
            while _is_wait(answer):
                answer = model_client.call(func_name="get_action")
            if not answer:
                break
            for action in answer:
                TASK_ENV.take_action(action)
                if TASK_ENV.is_episode_end():
                    break
    finally:
        _close_out(TASK_ENV)
        verdict = _episode_status(TASK_ENV)
        try:
            verdict["success"] = bool(TASK_ENV.success[0])
        except (TypeError, IndexError, AttributeError):
            verdict["success"] = False
        try:  # RoboDojo's progress score (0..100) as it will record it; a plain getter
            scores = TASK_ENV.reward_manager.get_score() if hasattr(TASK_ENV, "get_score") else None
            if scores is not None:
                verdict["progress"] = 100.0 if verdict["success"] else float(scores[0])
        except Exception:  # noqa: BLE001
            pass
        try:
            model_client.call(func_name="trial_end", obs=verdict)
        except Exception as exc:  # noqa: BLE001 - the trial stands either way
            print(f"[HexaShell] verdict not delivered to the policy server: {exc}", flush=True)


def eval_one_episode_batch(TASK_ENV: Any, model_client: Any) -> None:
    n = int(getattr(TASK_ENV, "num_envs", 1) or 1)
    if n > 1:
        raise RuntimeError(f"HexaShell runs one conversation per environment; got {n} environments. "
                           "Set eval_batch: false in deploy.yml.")
    eval_one_episode(TASK_ENV, model_client)
