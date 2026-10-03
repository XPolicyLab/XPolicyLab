"""Optional XPolicyLab boundary; the application never imports this module."""

import numpy as np
from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import get_robot_action_dim_info

from .policy_model import Model as PolicyModel


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        super().__init__()
        if (model_cfg.get("env_cfg_type"), model_cfg.get("action_type")) != (
            "arx_x5",
            "joint",
        ):
            raise ValueError("physicalRSI currently supports arx_x5 joint actions")
        if model_cfg.get("task_name") != model_cfg["physicalrsi"]["task"]:
            raise ValueError("Deployment task differs from frozen policy task")
        dimensions = get_robot_action_dim_info(model_cfg["env_cfg_type"])
        if len(dimensions["arm_dim"]) != 2 or len(dimensions["ee_dim"]) != 2:
            raise ValueError("Expected dual-arm robot metadata")
        self.dimensions = {
            f"{side}_{part}_joint_state": dimensions[field][i]
            for i, side in enumerate(("left", "right"))
            for part, field in (("arm", "arm_dim"), ("ee", "ee_dim"))
        }
        self.model = PolicyModel(model_cfg)

    def _chunk(self, chunk):
        if not isinstance(chunk, list) or not chunk:
            raise ValueError("Expected nonempty action chunk")
        result = []
        for action in chunk:
            if not isinstance(action, dict) or set(action) != set(self.dimensions):
                raise ValueError("Joint action keys differ from robot contract")
            row = {}
            for key, size in self.dimensions.items():
                value = np.asarray(action[key])
                if (
                    value.shape != (size,)
                    or value.dtype.kind not in "fiu"
                    or not np.isfinite(value).all()
                ):
                    raise ValueError(f"Invalid joint action: {key}")
                row[key] = value.copy()
            result.append(row)
        return result

    def update_obs(self, obs):
        return self.model.update_obs(obs)

    def update_obs_batch(self, obs_list):
        return self.model.update_obs_batch(obs_list)

    def get_action(self):
        return self._chunk(self.model.get_action())

    def get_action_batch(self, env_idx_list=None):
        return [
            self._chunk(chunk) for chunk in self.model.get_action_batch(env_idx_list)
        ]

    def reset(self):
        return self.model.reset()

    def on_trial_end(self, result=None):
        return self.model.on_trial_end(result)

    def physicalrsi_identity(self):
        return self.model.physicalrsi_identity()

    def close(self):
        return self.model.close()
