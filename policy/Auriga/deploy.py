"""RoboDojo-side episode loop for Auriga (runs in the simulator's Python; stdlib + numpy only).

Auriga is a language-model agent harness that runs entirely inside the policy server
(``model.py``): the Codex session, the tools, the motion planner (RoboDojo's own cuRobo planner
class, vendored) and the perception services. This loop is ``policy/demo_policy/deploy.py`` with
three documented divergences:

1. ``get_action`` may answer WAIT while the model is thinking. The loop asks again without
   stepping the simulator, so physics stays paused during reasoning (the semantics Auriga was
   developed with). Each answer arrives within ``poll_s`` (default 20 s), well inside the RoboDojo
   client's fixed 120 s request timeout.
2. An empty answer means the agent declared the episode finished: the loop exits and RoboDojo's
   own final check judges the scene (same close-out as when the step limit is reached).
3. Every observation carries ``additional_info["auriga"]``: the native step count and step limit,
   and robot proprioception that the stock ``arx_x5`` observation does not include (measured
   gripper jaw position and arm joint velocities). Nothing about objects, the layout, the seed or
   the evaluator is read or sent.

When the episode is over, the verdict is reported through the standard ``trial_end`` call, for
the run record only; the policy never sees it during the episode.
"""

from __future__ import annotations

from typing import Any

STATUS_KEY = "auriga"
WAIT_KEY = "auriga_wait"


def _as_list(value: Any) -> list:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _proprioception(env: Any) -> dict:
    """Robot-only measurements missing from the stock observation (no task or scene state)."""
    manager = getattr(env, "robot_manager", None)
    if manager is None:  # XPolicyLab debug client
        return {}
    out: dict = {}
    for index, robot in enumerate(getattr(manager, "robot_list", [])):
        if getattr(robot, "type", None) != "target":
            continue
        side = str(robot.arm_name).split("_")[0]
        item: dict = {}
        try:
            # Measured gripper joint positions in metres (RoboDojo RobotManager public getter).
            item["gripper_position_m"] = _as_list(manager.get_end_effector_real_val(robot, [0])[0])
        except Exception:  # noqa: BLE001 - optional field
            pass
        try:
            asset = manager.robot_key[index]
            item["arm_qdot_rad_s"] = _as_list(asset.data.joint_vel[0, robot.arm_joint_indices])
        except Exception:  # noqa: BLE001 - optional field
            pass
        out[side] = item
    return out


def _status(env: Any) -> dict:
    status: dict = {}
    counts = getattr(env, "take_action_cnt", None)
    if counts is not None:
        try:
            status["native_step"] = int(counts[0])
        except (TypeError, IndexError, ValueError):
            pass
    elif isinstance(getattr(env, "episode_step", None), int):  # debug client
        status["native_step"] = env.episode_step
    for attr in ("step_lim", "episode_step_limit"):
        value = getattr(env, attr, None)
        if value is not None:
            try:
                status["native_step_limit"] = int(value)
                break
            except (TypeError, ValueError):
                continue
    task = getattr(env, "task_name", None)
    if task is not None:
        status["task_name"] = str(task)
    status["proprioception"] = _proprioception(env)
    return status


def _observe(env: Any) -> dict:
    obs = env.get_obs()
    extra = dict(obs.get("additional_info") or {})
    extra[STATUS_KEY] = _status(env)
    obs["additional_info"] = extra
    return obs


def _is_wait(answer: Any) -> bool:
    return (isinstance(answer, (list, tuple)) and len(answer) == 1 and isinstance(answer[0], dict)
            and answer[0].get(WAIT_KEY) is True)


def _close_out(env: Any) -> None:
    """The agent stopped before RoboDojo ended the episode: run RoboDojo's final judgement now."""
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
            model_client.call(func_name="update_obs", obs=_observe(TASK_ENV))
            answer = model_client.call(func_name="get_action")
            while _is_wait(answer):
                answer = model_client.call(func_name="get_action")
            if not answer:
                break
            for action_idx, action in enumerate(answer):
                TASK_ENV.take_action(action)
                if TASK_ENV.is_episode_end() or action_idx + 1 == len(answer):
                    break
                model_client.call(func_name="update_obs", obs=_observe(TASK_ENV))
    finally:
        _close_out(TASK_ENV)
        verdict = {"native_step": _status(TASK_ENV).get("native_step")}
        try:
            verdict["success"] = bool(TASK_ENV.success[0])
        except (TypeError, IndexError, AttributeError):
            verdict["success"] = None
        try:
            model_client.call(func_name="trial_end", obs=verdict)
        except Exception as exc:  # noqa: BLE001 - the trial result stands either way
            print(f"[Auriga] verdict not delivered to the policy server: {exc}", flush=True)


def eval_one_episode_batch(TASK_ENV: Any, model_client: Any) -> None:
    n = int(getattr(TASK_ENV, "num_envs", 1) or 1)
    if n > 1:
        raise RuntimeError(f"Auriga runs one agent session per environment; got {n} environments. "
                           "Keep eval_batch: false in deploy.yml.")
    eval_one_episode(TASK_ENV, model_client)
