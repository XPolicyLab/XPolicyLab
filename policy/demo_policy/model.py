import numpy as np
from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import get_robot_action_dim_info, get_batch_size, get_action_dim


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        # 保存配置
        self.model_cfg = model_cfg
        self.action_type = model_cfg["action_type"]
        self.env_cfg_type = model_cfg["env_cfg_type"]

        self.action_dim = get_action_dim(self.env_cfg_type) # get the total dim of the action

        # 获取机器人动作维度信息
        # 示例:
        # {
        #     "arm_dim": [7] 或 [7, 7],
        #     "ee_dim": [1] 或 [1, 1]
        # }
        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        self.batch_size = get_batch_size(self.env_cfg_type)

        # arm 和 ee 的数量应一致，例如双臂时都应为 2
        assert len(self.robot_action_dim_info["arm_dim"]) == len(self.robot_action_dim_info["ee_dim"]), \
            "Arm and EE action dimensions must match"

        print(f"[Model] Model successfully initialized with action type: {self.action_type}")

    def update_obs(self, obs):
        # 如有需要，可在这里更新单条观测
        print("[Model] Received observation")
        pass

    def update_obs_batch(self, obs_list):
        # 如有需要，可在这里更新一批观测
        print(f"[Model] Received observation batch of size: {len(obs_list)}")
        pass

    def get_action(self):
        # 根据机械臂数量和动作类型确定 action key
        num_arms = len(self.robot_action_dim_info["arm_dim"])

        if num_arms == 1:  # 单臂
            arm_keys = ["arm_joint_state"] if self.action_type == "joint" else ["ee_pose"]
            ee_keys = ["ee_joint_state"]

        elif num_arms == 2:  # 双臂
            arm_keys = ["left_arm_joint_state", "right_arm_joint_state"] if self.action_type == "joint" else ["left_ee_pose", "right_ee_pose"]
            ee_keys = ["left_ee_joint_state", "right_ee_joint_state"]

        else:
            raise NotImplementedError(f"Unsupported number of arms: {num_arms}")

        steps = 1  # 当前示例只生成 1 步动作，可按需修改
        action_list = []

        for _ in range(steps):
            action_dict = {}

            for i, (arm_key, ee_key) in enumerate(zip(arm_keys, ee_keys)):
                # 机械臂动作
                # joint 模式: 维度由 arm_dim 决定
                # ee 模式: 默认使用 7 维位姿 [x, y, z, qw, qx, qy, qz]
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

                # 夹爪 / 末端执行器关节动作
                action_dict[ee_key] = np.zeros(
                    self.robot_action_dim_info["ee_dim"][i],
                    dtype=np.float32,
                )

            action_list.append(action_dict)

        print("[Model] Generated action")
        return action_list

    def get_action_batch(self, env_idx_list=None):
        # batch 大小由运行中的环境索引列表决定；缺省时回退到 env_cfg 的默认 batch_size
        batch_size = len(env_idx_list) if env_idx_list is not None else self.batch_size
        action_batch = [self.get_action() for _ in range(batch_size)]

        print(f"[Model] Generated action batch of size: {batch_size}")
        return action_batch

    def reset(self):
        # 如模型有内部状态（如 RNN hidden state），可在此重置
        print("[Model] Model successfully reset")
        pass
