import time


def eval_one_episode(TASK_ENV, model_client):
    env, client = TASK_ENV, model_client
    client.call(func_name='reset')

    def observation(final=False):
        obs = env.get_obs_batch(last_frame=True)[0] if final else env.get_obs()
        return {**obs, 'step': int(env.take_action_cnt[0]), 'step_limit': int(env.step_lim)}

    observed = True
    while not env.end_flag[0]:
        if observed:
            client.call(func_name='update_obs', obs=observation())
        result = client.call(func_name='poll_action')
        observed = result['status'] == 'action'
        if observed:
            env.take_action(result['action'])
        elif result['status'] == 'finished':
            env.is_episode_end()
            break
        else:
            time.sleep(0.1)
    client.call(func_name='cache_final_observation', obs=observation(final=True))
    client.call(func_name='trial_end', obs={'env_idx': 0, 'success': bool(env.success[0]),
        'step': int(env.take_action_cnt[0]), 'step_limit': int(env.step_lim),
        'layout_id': int(env.env_seeds[0]), 'unstable': 0 in env.unstable_envs})
