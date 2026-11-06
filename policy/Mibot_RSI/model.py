"""Policy-server model of Mibot-RSI: it serves no actions.

The agent runs inside the RoboDojo evaluation client (deploy.py); RoboDojo still connects to a policy server and
calls reset() before every episode, so this Model answers that contract and refuses any request for actions.
"""
from XPolicyLab.model_template import ModelTemplate


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = model_cfg

    def reset(self):
        return None

    def update_obs(self, obs):
        del obs

    def update_obs_batch(self, obs_list):
        del obs_list

    def get_action(self):
        raise RuntimeError('Mibot-RSI serves no actions: the agent runs in the evaluation client (deploy.py)')

    def get_action_batch(self, env_idx_list=None):
        del env_idx_list
        raise RuntimeError('Mibot-RSI serves no batched actions; run it with eval_batch: false')
