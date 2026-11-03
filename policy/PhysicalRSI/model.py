"""Public XPolicyLab model contract with an owned PhysicalRSI runtime."""
from XPolicyLab.model_template import ModelTemplate
from PhysicalRSI_baselines.robodojo.skill_model import Model as SkillModel


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        super().__init__()
        self.model = SkillModel(model_cfg)

    def update_obs(self, obs):
        return self.model.update_obs(obs)

    def update_obs_batch(self, obs_list):
        return self.model.update_obs_batch(obs_list)

    def get_action(self):
        return self.model.get_action()

    def get_action_batch(self, env_idx_list=None):
        return self.model.get_action_batch(env_idx_list)

    def reset(self):
        return self.model.reset()

    def on_trial_end(self, result=None):
        return self.model.on_trial_end(result)

    def physicalrsi_identity(self):
        return self.model.physicalrsi_identity()

    def close(self):
        return self.model.close()


__all__ = ['Model']
