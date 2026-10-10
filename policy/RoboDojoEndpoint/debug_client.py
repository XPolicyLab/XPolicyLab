"""CPU interface check using upstream RGB fixtures and opaque session IDs."""
import argparse
import os
from uuid import uuid4

from XPolicyLab.utils.debug_env_client import TestEnv


class InstructionTestEnv(TestEnv):
    def __init__(self, cfg, num_envs):
        super().__init__(cfg)
        self.num_envs = num_envs

    def reset(self):
        super().reset()
        self._steps = [0] * self.num_envs

    def get_running_env_idx_list(self):
        return [index for index, step in enumerate(self._steps) if step < 12 + index]

    def take_action_batch(self, actions, indices):
        super().take_action_batch(actions, indices)
        for index in indices:
            self._steps[index] += 1

    def is_episode_end(self):
        return not self.get_running_env_idx_list()

    def get_obs(self, env_idx=0):
        obs = super().get_obs(env_idx)
        obs['instruction'] = os.environ.get('DEBUG_INSTRUCTION', 'Stack the three bowls together.')
        return obs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-cfg-type', required=True)
    parser.add_argument('--host', required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--episodes', type=int, default=2)
    parser.add_argument('--num-envs', type=int, default=8)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('episodes must be positive')
    if not 1 <= args.num_envs <= 8:
        parser.error('num-envs must be between 1 and 8')
    session = uuid4().hex
    env = InstructionTestEnv(dict(
        env_cfg_type=args.env_cfg_type, policy_name='RoboDojoEndpoint', protocol='ws',
        policy_server_url=None, host=args.host, port=args.port,
        evaluation_id=f'debug-{session}', trial_id=f'trial-{session}',
        action_case_id=None, repeat_index=None,
        obs_encoded=os.environ.get('DEBUG_OBS_ENCODED', '0').lower() in ('1', 'true', 'yes'),
    ), args.num_envs)
    try:
        for _ in range(args.episodes):
            env.reset()
            env.eval_one_episode_batch()
            env.finish_episode()
    finally:
        env.model_client.close()


if __name__ == '__main__':
    main()
