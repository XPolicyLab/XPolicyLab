"""XPolicyLab entry point for the installed MoPA policy."""

from pathlib import Path

from mopa.integrations.xpolicylab import ModelAdapter
from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.process_data import get_robot_action_dim_info
from mopa.data import mobile


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        if model_cfg.get("action_type") != "joint":
            raise ValueError("MoPA supports action_type=joint only")
        env_cfg_type = model_cfg["env_cfg_type"]
        # The mobile layout adds poses and base channels to the joint schema.
        dim_info = (mobile.metadata()["robot_action_dim_info"]
                    if str(env_cfg_type).lower() in {"m92uw", "mobile", "mobile_m92uw"}
                    else get_robot_action_dim_info(env_cfg_type))
        policy_dir = Path(__file__).resolve().parent
        checkpoint = resolve_checkpoint_root(
            model_cfg, policy_dir / "checkpoints", policy_dir=policy_dir,
        )
        self.adapter = ModelAdapter(checkpoint, dim_info, model_cfg)

    def reset(self):
        self.adapter.reset()

    def update_obs(self, obs):
        self.adapter.update_obs(obs)

    def update_obs_batch(self, obs_list):
        self.adapter.update_obs_batch(obs_list)

    def get_action(self):
        return self.adapter.get_action()

    def get_action_batch(self, env_idx_list=None):
        return self.adapter.get_action_batch(env_idx_list)
