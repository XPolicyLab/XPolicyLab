import numpy as np
from XPolicyLab.model_template import ModelTemplate

class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = model_cfg
        self.action_type = model_cfg["action_type"]
        # Initialize your policy model here according to model_cfg
    
    def update_obs(self, obs):
        # Update your model's observation here if needed
        print("[Model] Received observation")
        pass
    
    def update_obs_batch(self, obs_list):
        # Update your model's observation here if needed
        print("[Model] Received observation batch of size:", len(obs_list))
        pass

    def get_action(self):
        # Generate action according to your model and current observation
        # This is a dummy action for demonstration, replace it with your model's action
        if self.action_type == "joint":
            left_key = "left_arm_joint_state"
            right_key = "right_arm_joint_state"
        elif self.action_type == "ee":
            left_key = "left_ee_pose"
            right_key = "right_ee_pose"
        else:
            raise ValueError(f"Unsupported action_type: {self.action_type}")

        action_dict = {
            left_key: np.zeros(7),
            "left_ee_joint_state": np.zeros(1),
            right_key: np.zeros(7),
            "right_ee_joint_state": np.zeros(1),
        }

        print("[Model] Generated action")
        return action_dict

    def get_action_batch(self):
        # Generate action batch according to your model and current observation batch
        # This is a dummy action batch for demonstration, replace it with your model's action batch
        batch_size = 4  # Example batch size
        if self.action_type == "joint":
            left_key = "left_arm_joint_state"
            right_key = "right_arm_joint_state"
        elif self.action_type == "ee":
            left_key = "left_ee_pose"
            right_key = "right_ee_pose"
        else:
            raise ValueError(f"Unsupported action_type: {self.action_type}")

        action_batch = []
        for _ in range(batch_size):
            action_dict = {
                left_key: np.zeros(7),
                "left_ee_joint_state": np.zeros(1),
                right_key: np.zeros(7),
                "right_ee_joint_state": np.zeros(1),
            }
            action_batch.append(action_dict)

        print("[Model] Generated action batch of size:", batch_size)
        return action_batch 

    def reset(self):
        # Reset your model's internal state if needed
        print("[Model] Model successfully reset")
        pass