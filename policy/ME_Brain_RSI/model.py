"""Standard XPolicyLab entry point backed by one task coordinator."""
from XPolicyLab.model_template import ModelTemplate
from .adaptor import ME_Brain_Adaptor


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.adaptor = ME_Brain_Adaptor(model_cfg)

    def runtime_info(self):
        return self.adaptor.runtime_info()

    def update_obs(self, obs):
        return self.adaptor.update_obs(obs)

    def update_obs_batch(self, obs):
        return self.adaptor.update_obs_batch(obs)

    def get_action(self, **kwargs):
        return self.adaptor.get_action(**kwargs)

    def get_action_batch(self, env_idx_list=None, **kwargs):
        return self.adaptor.get_action_batch(env_idx_list, **kwargs)

    def reset(self):
        return self.adaptor.reset()

    def reset_obsrvationwindows(self):
        return self.reset()

    def mark_natural_end_batch(self, env_idx_list):
        return self.adaptor.optional('mark_natural_end_batch', env_idx_list)

    def finish_episode(self):
        return self.adaptor.optional('finish_episode')

    def prepare_case(self, case_meta=None):
        return self.adaptor.optional('prepare_case', case_meta)

    def on_trial_end(self, result=None):
        return self.adaptor.optional('on_trial_end', result)

    def close(self):
        return self.adaptor.optional('close')
