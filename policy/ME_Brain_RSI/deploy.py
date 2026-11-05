"""Central deployment dispatch; task-specific control loops keep their timing."""
from .routing import configure


class DecodedEnvironment:
    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    def get_obs(self, *args, **kwargs):
        from XPolicyLab.utils.process_data import decode_obs_images
        return decode_obs_images(self.env.get_obs(*args, **kwargs))

    def get_obs_batch(self, *args, **kwargs):
        from XPolicyLab.utils.process_data import decode_obs_images
        return decode_obs_images(self.env.get_obs_batch(*args, **kwargs))


def _run(env, client, batch):
    from .core.runtime import apply, prepare
    info = client.call(func_name='runtime_info')
    config, route = configure(apply(info))
    if route['driver'] == 'vla':
        from . import vla_deploy
        return (vla_deploy.eval_one_episode_batch if batch else vla_deploy.eval_one_episode)(env, client)
    if batch != route['eval_batch']:
        raise ValueError('Wrong eval_batch for the selected task')
    if getattr(env, 'num_envs', 1) != 1:
        raise ValueError('This strategy requires one environment per process')
    config, run = prepare(config, route)
    if route['deploy']:
        from .core.modules import TaskModules
        module = TaskModules(route['canonical_task']).import_module(route['deploy'])
        return (module.eval_one_episode_batch if batch else module.eval_one_episode)(DecodedEnvironment(env), client)
    if route['driver'] == 'equation':
        return _batch_loop(env, client)
    client.call(func_name='reset')
    while not env.is_episode_end():
        client.call(func_name='update_obs', obs=env.get_obs())
        for action in client.call(func_name='get_action'):
            env.take_action(action)
            if env.is_episode_end():
                break
    if route['driver'] == 'model':
        client.call(func_name='update_obs', obs=env.get_obs())
        client.call(func_name='finish_episode')


def _batch_loop(env, client):
    client.call(func_name='reset')
    while not env.is_episode_end():
        indices = env.get_running_env_idx_list()
        if len(indices) != 1:
            raise ValueError('This batch protocol requires one active environment')
        client.call(func_name='update_obs_batch', obs=env.get_obs_batch(indices))
        chunks = client.call(func_name='get_action_batch', obs=indices)
        if len(chunks) != 1 or not chunks[0]:
            raise ValueError('Invalid action batch')
        for action in chunks[0]:
            env.take_action_batch([action], indices)
            running = set(env.get_running_env_idx_list()) if not env.is_episode_end() else set()
            ended = [i for i in indices if i not in running]
            if ended:
                client.call(func_name='mark_natural_end_batch', obs=ended)
                break


def eval_one_episode(TASK_ENV, model_client):
    return _run(TASK_ENV, model_client, False)


def eval_one_episode_batch(TASK_ENV, model_client):
    return _run(TASK_ENV, model_client, True)
