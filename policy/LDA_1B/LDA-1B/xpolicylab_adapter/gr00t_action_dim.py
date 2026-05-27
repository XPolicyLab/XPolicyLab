"""Print the LDA action_dim for an env_cfg_type.

Most LDA mixture datasets pad action keys to fixed per-key widths. XPolicyLab's
arx_x5 is a single custom robot path and keeps the 6-D arm keys unpadded, so its
model action_dim matches the raw physical action dim.

For arx_x5 (action_keys = left_arm, left_gripper_close, right_arm,
right_gripper_close) this is 6 + 1 + 6 + 1 = 14.

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
    pad_arm_to_7 = env_cfg_type != "arx_x5"
    # Width each action key is padded to, summed over the embodiment's action_keys.
    total = sum(
        int(pad_action_state_with_key(np.zeros((1, 1)), key, pad_arm_to_7=pad_arm_to_7)[0].shape[1])
        for key in cfg.action_keys
    )
    print(total)


if __name__ == "__main__":
    main()
