import numpy as np
from robot.utils.base.data_handler import debug_print
import numpy as np

STATE_POINTS = [
    [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1.],
    [0.3, 0., 0.3, 0., 0., 0., 0., 0.3, 0., 0.3, 0., 0., 0., 0.],
    [0.0, 0., 0.3, 0.1, 0., 0., 0., 0.0, 0., 0.3, 0.1, 0., 0., 0.],
]

class RUN_POINT:
    def __init__(self, state_points, interp_frames=30, chunk_size=10):
        self.state_points = np.asarray(state_points, dtype=np.float32)
        self.interp_frames = interp_frames
        self.chunk_size = chunk_size

        self.episode = self._get_full_actions()
        self.ptr = 0

    def _interpolate(self, start, end, num):
        """线性插值，不包含终点"""
        return np.linspace(start, end, num=num, endpoint=False)

    def _get_full_actions(self):
        actions = []
        n = len(self.state_points)

        for i in range(n):
            start = self.state_points[i]
            end = self.state_points[(i + 1) % n]  # 关键：回到第一个
            seg = self._interpolate(start, end, self.interp_frames)
            actions.append(seg)

        actions = np.concatenate(actions, axis=0)
        return actions  # (T, 14)

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
        self.model = RUN_POINT(STATE_POINTS)
        
    def update_obs(self, obs):
        # 如果你后面要用历史窗口，可以存在这里
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
        debug_print("YOUR_POLICY", "RUN_POINT model reset success!", "INFO")