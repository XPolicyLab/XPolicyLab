import numpy as np
import os
from robot.utils.base.data_handler import debug_print, dict_to_list, hdf5_groups_to_dict
import numpy as np

def state_transform(data):
    state = np.concatenate([
        np.array(data["left_arm"]["joint"]).reshape(-1),
        np.array(data["left_arm"]["gripper"]).reshape(-1),
        np.array(data["right_arm"]["joint"]).reshape(-1),
        np.array(data["right_arm"]["gripper"]).reshape(-1)
    ])
    return state

class REPLAY:
    def __init__(self, hdf5_path, chunk_size=500):
        self.raw_episode = dict_to_list(hdf5_groups_to_dict(hdf5_path))
        self.chunk_size = chunk_size
        self.ptr = 0
        self.episode = self._get_full_actions()

    def _interpolate(self, start, end, num):
        """线性插值，不包含终点"""
        return np.linspace(start, end, num=num, endpoint=False)

    def _get_full_actions(self):
        base_state = np.array(
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            dtype=np.float32
        )

        start_state = state_transform(self.raw_episode[0])
        end_state = state_transform(self.raw_episode[-1])

        actions = []

        # base → first
        actions.extend(self._interpolate(base_state, start_state, 10))

        # episode
        for ep in self.raw_episode:
            actions.append(state_transform(ep))

        # last → base
        actions.extend(self._interpolate(end_state, base_state, 30))

        return np.asarray(actions)   # (T, 14)

    def infer(self):
        action_chunk = []

        for _ in range(self.chunk_size):
            action = self.episode[self.ptr]
            action_chunk.append(action)

            self.ptr += 1
            if self.ptr >= len(self.episode):
                self.ptr = 0  # 循环

        return np.asarray(action_chunk)

    def reset(self):
        self.ptr = 0

class Your_Policy:
    def __init__(self, deploy_cfg=None):
        # Initialize your policy model here
        self.deploy_cfg = deploy_cfg
        self.model = REPLAY(os.path.join('./data', deploy_cfg['task_name'], deploy_cfg['base_cfg'], f"{deploy_cfg['ckpt_setting']}.hdf5"))
        
    def update_obs(self, obs):
        self.last_obs = obs

    def get_action(self, obs=None):
        if obs is not None:
            self.update_obs(obs)
        
        actions = self.model.infer()

        ret_actions = []
        for action in actions:
            ret_action = {
                "arm": {
                    "left_arm": {
                        "joint": action[:6],
                        "gripper": action[6],
                    },
                    "right_arm": {
                        "joint": action[7:13],
                        "gripper": action[13],
                    }
                }
            }
            ret_actions.append(ret_action)

        return ret_actions

    def set_language(self, instruction):
        # Set the language instruction for the model here
        self.instruction = instruction

    def reset(self):
        # Reset the observation cache or window here
        self.model.reset()
        debug_print("YOUR_POLICY", "REPLAY model reset success!", "INFO")