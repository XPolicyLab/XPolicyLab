"""CPU interface check using upstream RGB fixtures and opaque session IDs."""
import argparse
import os
from uuid import uuid4

from XPolicyLab.utils.debug_env_client import TestEnv


class InstructionTestEnv(TestEnv):
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
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('episodes must be positive')
    session = uuid4().hex
    env = InstructionTestEnv(dict(
        env_cfg_type=args.env_cfg_type, policy_name='RoboDojoEndpoint', protocol='ws',
        policy_server_url=None, host=args.host, port=args.port,
        evaluation_id=f'debug-{session}', trial_id=f'trial-{session}',
        action_case_id=None, repeat_index=None,
        obs_encoded=os.environ.get('DEBUG_OBS_ENCODED', '0').lower() in ('1', 'true', 'yes'),
    ))
    try:
        for _ in range(args.episodes):
            env.reset()
            env.eval_one_episode()
            env.finish_episode()
    finally:
        env.model_client.close()


if __name__ == '__main__':
    main()
