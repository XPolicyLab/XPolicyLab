import numpy as np
from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import get_robot_action_dim_info, get_batch_size, get_action_dim


class Model(ModelTemplate):
    # Add "ppo" and/or "grpo" after _sample_policy in rl.py returns real log-probs.
    # Put the trainable nn.Module on self.model or self.policy.
    RL_MODES = ("eval",)

    def __init__(self, model_cfg):
        # Store the configuration
        self.model_cfg = model_cfg
        self.action_type = model_cfg["action_type"]
        self.env_cfg_type = model_cfg["env_cfg_type"]

        self.action_dim = get_action_dim(self.env_cfg_type) # get the total dim of the action

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
        # Select action keys from the arm count and action type
        num_arms = len(self.robot_action_dim_info["arm_dim"])

        if num_arms == 1:  # single arm
            arm_keys = ["arm_joint_state"] if self.action_type == "joint" else ["ee_pose"]
            ee_keys = ["ee_joint_state"]

        elif num_arms == 2:  # dual arm
            arm_keys = ["left_arm_joint_state", "right_arm_joint_state"] if self.action_type == "joint" else ["left_ee_pose", "right_ee_pose"]
            ee_keys = ["left_ee_joint_state", "right_ee_joint_state"]

        else:
            raise NotImplementedError(f"Unsupported number of arms: {num_arms}")

        steps = 2  # current exampleonlygenerate 2 action, by
        action_list = []

        for _ in range(steps):
            action_dict = {}

            for i, (arm_key, ee_key) in enumerate(zip(arm_keys, ee_keys)):
                # Arm action
                # joint mode: dimensions are determined by arm_dim
                # ee mode: defaults to a 7-D pose [x, y, z, qw, qx, qy, qz]
                if self.action_type == "joint":
                    action_dict[arm_key] = np.zeros(
                        self.robot_action_dim_info["arm_dim"][i],
                        dtype=np.float32,
                    )
                else:
                    action_dict[arm_key] = np.array(
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                        dtype=np.float32,
                    )

                # Gripper / end-effector joint action
                action_dict[ee_key] = np.zeros(
                    self.robot_action_dim_info["ee_dim"][i],
                    dtype=np.float32,
                )

            action_list.append(action_dict)

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

    def predict_action_batch_rl(self, obs_list, mode="train", compute_values=True, **kwargs):
        # Fill _sample_policy in rl.py. sample_actions_rl already returns this shape.
        # mode "train" samples and records log-probs; mode "eval" is deterministic.
        # compute_values is for PPO. Add "ppo" or "grpo" to RL_MODES when the
        # log-probs are real; until then RLinf eval keeps using get_action_batch.
        from .rl import sample_actions_rl

        outputs = sample_actions_rl(
            self, obs_list, mode=mode, compute_values=compute_values
        )
        return outputs["actions"], {
            "prev_logprobs": outputs["prev_logprobs"],
            "prev_values": outputs["prev_values"],
            "forward_inputs": outputs["forward_inputs"],
        }

    def compute_logprob_value(self, forward_inputs, compute_values=False, **kwargs):
        # Recompute the rollout sample with gradients. Fill _log_prob_value in rl.py.
        from .rl import get_log_prob_value

        return get_log_prob_value(
            self, forward_inputs, compute_values=compute_values
        )
