"""XPolicyLab-compatible client for a separately hosted inference service."""

from __future__ import annotations

import os
from uuid import uuid4

import numpy as np

from XPolicyLab.client_server.ws import WsModelClient
from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import get_robot_action_dim_info


class Model(ModelTemplate):
    """Forward model calls; observation processing and inference stay remote.

    The normal eval scripts use the frame-preserving WebSocket bridge. This
    class also supports applications that instantiate the Model API directly.
    """

    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = dict(model_cfg)
        if model_cfg.get("action_type") != "joint":
            raise ValueError("RoboDojoEndpoint supports action_type=joint")
        if model_cfg.get("env_cfg_type") != "arx_x5":
            raise ValueError("RoboDojoEndpoint supports env_cfg_type=arx_x5")
        if model_cfg.get("eval_batch", False):
            raise ValueError("Use independent sessions with eval_batch=false")
        url = os.environ.get("POLICY_ENDPOINT_URL")
        if not url or not url.startswith(("ws://", "wss://")):
            raise ValueError("Set POLICY_ENDPOINT_URL to the supplied endpoint")

        dimensions = get_robot_action_dim_info(model_cfg["env_cfg_type"])
        if len(dimensions["arm_dim"]) != 2 or len(dimensions["ee_dim"]) != 2:
            raise ValueError("Expected a dual-arm robot configuration")
        self.action_dimensions = {}
        for index, prefix in enumerate(("left_", "right_")):
            self.action_dimensions[prefix + "arm_joint_state"] = dimensions["arm_dim"][index]
            self.action_dimensions[prefix + "ee_joint_state"] = dimensions["ee_dim"][index]

        session = uuid4().hex
        task_name = model_cfg.get("task_name")
        case_id = (
            f"{task_name.strip()}_case"
            if isinstance(task_name, str) and task_name.strip() and len(task_name) <= 128
            else f"case-{session}"
        )
        self.client = WsModelClient(
            url=url,
            evaluation_id=f"endpoint-model-{session}",
            trial_id=f"trial-{session}",
            action_case_id=case_id,
        )

    def update_obs(self, obs):
        # XPolicyLab already supplies RGB arrays: forward without decoding.
        return self.client.call(func_name="update_obs", obs=obs)

    def get_action(self):
        actions = self.client.call(func_name="get_action")
        if not isinstance(actions, list) or not actions:
            raise ValueError("Endpoint must return a nonempty action chunk")
        for action in actions:
            if not isinstance(action, dict) or set(action) != set(self.action_dimensions):
                raise ValueError("Endpoint returned unexpected action keys")
            for key, dimension in self.action_dimensions.items():
                value = np.asarray(action[key])
                if value.shape != (dimension,) or not np.isfinite(value).all():
                    raise ValueError("Endpoint returned invalid joint action dimensions or values")
                action[key] = value
        return actions

    def reset(self):
        return self.client.call(func_name="reset")

    def update_obs_batch(self, obs_list):
        raise NotImplementedError("Use one independent session per environment")

    def get_action_batch(self, env_idx_list=None):
        raise NotImplementedError("Use one independent session per environment")

    def prepare_case(self, case_meta=None):
        return self.client.call(func_name="prepare_case", obs=case_meta)

    def on_trial_end(self, result=None):
        return self.client.call(func_name="trial_end", obs=result)

    def close(self):
        self.client.close()
