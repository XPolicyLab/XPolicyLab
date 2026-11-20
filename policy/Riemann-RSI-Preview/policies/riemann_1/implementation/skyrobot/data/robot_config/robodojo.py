from skyrobot.data.robot_config.base_config import base_data_config
from skyrobot.data.robot_config.base_config import EmbodimentId
from skyrobot.data.robot_config.base_config import Embodiment_DIM
import torch

class RoboDojo(base_data_config):
    embodiment = 'RoboDojo'
    embodiment_id: int = EmbodimentId.RoboDojo
    action_dim = Embodiment_DIM.action_dims[embodiment_id]
    state_dim = Embodiment_DIM.state_dims[embodiment_id]
    obs_cam_keys: list = ['observation.images.cam_high', 'observation.images.cam_left_wrist', 'observation.images.cam_right_wrist']

    action_indices: list = (
        list(range(0, 16))
    )  # 16 dims
    state_indices: list = (
        list(range(0, 16))
    )  # 16 dims

    def combine_views(self, video_dict: dict) -> torch.Tensor:
        """Inverted-T concat. Each raw view is first letterbox-resized to a
        common per-view size (480x640) via `self.resize`, then arranged as:
            ┌──[ pad ][ top_head ][ pad ]──┐   <- top row centered, W/2 black border on each side
            ├──────────────┬───────────────┤
            │   hand_left  │   hand_right  │   <- bottom row: left/right concatenated directly, width 2W
            └──────────────┴───────────────┘
        Per-view input: [F, C, H_raw, W_raw] (raw resolutions may differ).
        Output: [F, C, 2H, 2W] = [F, C, 960, 1280].
        Caller (dataset `_resize_frames`) will letterbox this combined image
        down to the model target (video_height, video_width).
        """
        top = self._resize(video_dict[self.obs_cam_keys[0]], 256, 320)    # [F, C, 256, 320]
        left = self._resize(video_dict[self.obs_cam_keys[1]], 128, 160)   # [F, C, 128, 160]
        right = self._resize(video_dict[self.obs_cam_keys[2]], 128, 160)  # [F, C, 128, 160]

        bottom = torch.cat([left, right], dim=-1)    # [F, C, 128, 320]
        return torch.cat([top, bottom], dim=-2)      # [F, C, 384, 320]
