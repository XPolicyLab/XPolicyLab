"""Keep the XPolicyLab registration boundary separate from model inference."""

from pathlib import Path

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import (
    DEFAULT_EXPLICIT_KEYS,
    resolve_checkpoint_root,
)

from prismwam.integrations.xpolicylab import PrismWAMAdapter


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        super().__init__()
        args = dict(model_cfg)
        policy_dir = Path(__file__).resolve().parent
        args["policy_dir"] = str(policy_dir)
        args["model_path"] = str(
            resolve_checkpoint_root(
                args,
                policy_dir / "checkpoints",
                policy_dir=policy_dir,
                explicit_keys=("model_path", *DEFAULT_EXPLICIT_KEYS),
            )
        )
        self.model = PrismWAMAdapter(args)

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

    def prepare_case(self, *args, **kwargs):
        return self.model.prepare_case(*args, **kwargs)
