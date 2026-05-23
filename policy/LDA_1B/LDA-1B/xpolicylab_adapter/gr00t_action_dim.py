"""Print the gr00t canonical action_dim for an env_cfg_type.

LDA's mixture dataloader pads every action key to a fixed per-key width
(pad_action_state_with_key: arm->7, gripper_close->1, gripper_width->6, ...) and
concatenates them, so the model's action_dim must equal that per-key padded SUM,
NOT the robot's raw physical action dim (XPolicyLab/utils/get_action_dim.sh).

For arx_x5 (action_keys = left_arm, left_gripper_close, right_arm,
right_gripper_close) this is 7 + 1 + 7 + 1 = 16.

Usage (run in the LDA_1B conda env, where the `lda` package is importable):
    python gr00t_action_dim.py <env_cfg_type>
"""

import sys

import numpy as np

from lda.dataloader.gr00t_lerobot.data_config import ROBOT_TYPE_CONFIG_MAP
from lda.dataloader.gr00t_lerobot.datasets import pad_action_state_with_key


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: python gr00t_action_dim.py <env_cfg_type>")
    env_cfg_type = sys.argv[1]
    cfg = ROBOT_TYPE_CONFIG_MAP[env_cfg_type]
    # Width each action key is padded to, summed over the embodiment's action_keys.
    total = sum(
        int(pad_action_state_with_key(np.zeros((1, 1)), key)[0].shape[1])
        for key in cfg.action_keys
    )
    print(total)


if __name__ == "__main__":
    main()
