import numpy as np
from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import get_robot_action_dim_info, get_batch_size, get_action_dim, get_robot_state_schema


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        # Store the configuration
        self.model_cfg = model_cfg
        self.action_type = model_cfg["action_type"]
        self.env_cfg_type = model_cfg["env_cfg_type"]

        self.action_dim = get_action_dim(self.env_cfg_type, self.action_type) # get the total dim of the action

        # Get robot action dimension metadata
        # Example:
        # {
        #     "arm_dim": [7] or [7, 7],
        #     "ee_dim": [1] or [1, 1]
        # }
        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        self.batch_size = get_batch_size(self.env_cfg_type)

        # The number of arm and EE entries must match, e.g. both are 2 for dual-arm robots
        assert len(self.robot_action_dim_info["arm_dim"]) == len(self.robot_action_dim_info["ee_dim"]), \
            "Arm and EE action dimensions must match"

        print(f"[Model] Model successfully initialized with action type: {self.action_type}")

    def update_obs(self, obs):
        # Update a single observation here if needed
        print("[Model] Received observation")
        pass

    def update_obs_batch(self, obs_list):
        # Update a batch of observations here if needed
        print(f"[Model] Received observation batch of size: {len(obs_list)}")
        pass

    def get_action(self):
        schema = get_robot_state_schema(self.action_type, self.robot_action_dim_info)
        action_list = []
        for _ in range(2):  # Return a two-step action chunk.
            action = {key: np.zeros(width, dtype=np.float32) for key, width in schema.items()}
            if self.action_type == "ee":
                for key in action:
                    if key.endswith("ee_pose"):
                        action[key][3] = 1.0  # Identity quaternion [qw, qx, qy, qz].
            action_list.append(action)

        print("[Model] Generated action")
        return action_list

    def get_action_batch(self, env_idx_list=None):
        # Batch size follows the running environment index list; fall back to the env_cfg default batch_size when omitted
        batch_size = len(env_idx_list) if env_idx_list is not None else self.batch_size
        action_batch = [self.get_action() for _ in range(batch_size)]

        print(f"[Model] Generated action batch of size: {batch_size}")
        return action_batch

    def reset(self):
        # Reset model state here if it has internal state, such as an RNN hidden state
        print("[Model] Model successfully reset")
        pass
