"""Generic XPolicyLab episode loop (aligned with policy/demo_policy/deploy.py).

XPolicyLab's debug client uses this loop. Simulator evaluation instead runs the
pinned native RoboTwin evaluator through setup_eval_env_client.sh, whose own
loop calls prepare_case before each trial so the goal matches the scene.
"""


def eval_one_episode(TASK_ENV, model_client):

    model_client.call(func_name="reset")  # reset policy

    while not TASK_ENV.is_episode_end():  # Check whether the episode ends
        obs = TASK_ENV.get_obs()  # Get Observation
        model_client.call(func_name="update_obs", obs=obs)  # Update Observation
        actions = model_client.call(func_name="get_action")  # Get Action according to observation chunk

        for action_idx, action in enumerate(actions):
            TASK_ENV.take_action(action)

            if TASK_ENV.is_episode_end() or action_idx + 1 == len(actions):
                break

            obs = TASK_ENV.get_obs()
            model_client.call(func_name="update_obs", obs=obs)


def eval_one_episode_batch(TASK_ENV, model_client):
    raise RuntimeError("HumanWAM-Goal keeps per-episode goal and history state; set eval_batch: false")
