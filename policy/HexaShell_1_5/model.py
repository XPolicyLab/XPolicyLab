"""XPolicyLab entry point of HexaShell-1.5 (runs inside the policy server).

The agent harness (the ``hexashell`` package, its prompts and task notes) lives in this directory, so it
is put on sys.path here and ``hexashell`` imports resolve wherever the policy server was started from.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from XPolicyLab.model_template import ModelTemplate  # noqa: E402

from hexashell.xpl.adapter import PolicyHost  # noqa: E402


class Model(ModelTemplate):
    def __init__(self, deploy_cfg):
        super().__init__()
        self.host = PolicyHost(dict(deploy_cfg))
        self.model = self.host

    def reset(self):
        return self.host.reset()

    def update_obs(self, obs):
        return self.host.update_obs(obs)

    def get_action(self):
        return self.host.get_action()

    def on_trial_end(self, result=None):
        return self.host.on_trial_end(result)

    def update_obs_batch(self, obs_list):
        raise NotImplementedError("HexaShell-1.5 drives one environment per policy server (eval_batch: false)")

    def get_action_batch(self, env_idx_list=None):
        raise NotImplementedError("HexaShell-1.5 drives one environment per policy server (eval_batch: false)")
