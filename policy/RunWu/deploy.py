"""XPolicyLab/policy/RunWu/deploy.py: the official loop of policy/demo_policy/deploy.py, with one end_episode call at the end.

XPolicyLab does not call the policy's close(), and the server gets no end-of-episode signal; the explicit end_episode lets
every episode (including the last one) finish cleanly. Other policies also add custom calls in deploy.py.
This file runs inside the RoboDojo simulation process and does not import runwu_rsi.
"""


def eval_one_episode(TASK_ENV, model_client):
    model_client.call(func_name="reset")

    while not TASK_ENV.is_episode_end():
        obs = TASK_ENV.get_obs()
        model_client.call(func_name="update_obs", obs=obs)
        actions = model_client.call(func_name="get_action")

        for action_idx, action in enumerate(actions):
            TASK_ENV.take_action(action)

            if TASK_ENV.is_episode_end() or action_idx + 1 == len(actions):
                break

            obs = TASK_ENV.get_obs()
            model_client.call(func_name="update_obs", obs=obs)

    model_client.call(func_name="end_episode")
