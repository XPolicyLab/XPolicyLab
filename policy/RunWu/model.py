"""XPolicyLab/policy/RunWu/model.py: the Model(ModelTemplate) that XPolicyLab requires; the runtime is in the runwu_rsi package."""
import os
from pathlib import Path

from XPolicyLab.model_template import ModelTemplate

from runwu_rsi.runtime.adapters.xpolicylab import Model as RuntimeModel


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        super().__init__()
        policy_dir = Path(__file__).resolve().parent
        workspace_assets = policy_dir.parents[2] / "Assets"          # <RoboDojo workspace>/XPolicyLab/policy/RunWu
        if workspace_assets.is_dir():                                # the robot URDF is read from the workspace's assets
            os.environ.setdefault("ROBODOJO_ASSETS_ROOT", str(workspace_assets))
        if not model_cfg.get("result_dir"):             # the official launcher never sets it: keep traces next to the adapter
            model_cfg = {**model_cfg, "result_dir": str(policy_dir / "results")}
        self.model = RuntimeModel(model_cfg, policy_dir=policy_dir)

    def reset(self):
        self.model.reset()

    def update_obs(self, obs):
        self.model.update_obs(obs)

    def get_action(self):
        return self.model.get_action()

    def update_obs_batch(self, obs_list):       # batched evaluation is not supported: deploy.yml keeps eval_batch: false
        raise NotImplementedError("RunWu runs one environment per policy server; keep eval_batch: false")

    def get_action_batch(self, env_idx_list=None):
        raise NotImplementedError("RunWu runs one environment per policy server; keep eval_batch: false")

    def end_episode(self):                      # called by deploy.py after the official loop ends
        self.model.end_episode()
