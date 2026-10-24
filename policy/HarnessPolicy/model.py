"""Harness Policy as an XPolicyLab policy (exported by `harness export xpolicylab`).

The policy is the frozen Harness Policy agent (agent.json): a model writes programs that
call perception, motion and learned skills; this Model hands XPolicyLab's observations to
those programs and their actions back (harness_policy/connectors/xpolicylab.py). model.py
never decodes images: the policy server has decoded every observation it forwards.
"""

from pathlib import Path
import sys

from XPolicyLab.model_template import ModelTemplate

PACKAGE = Path(__file__).resolve().parent
_HARNESS = PACKAGE / "the-harness"
for _path in (_HARNESS / "src", _HARNESS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from harness_policy.connectors.package import HarnessModel  # noqa: E402


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = model_cfg
        self.harness = HarnessModel(model_cfg, PACKAGE)

    def update_obs(self, obs):
        self.harness.update_obs(obs)

    def update_obs_batch(self, obs_list):
        self.harness.update_obs_batch(obs_list)

    def get_action(self):
        return self.harness.get_action()

    def get_action_batch(self, env_idx_list=None):
        return self.harness.get_action_batch(env_idx_list)

    def reset(self):
        self.harness.reset()
