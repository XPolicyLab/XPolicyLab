"""Run one synthetic observation through the standard XPolicyLab Model API."""

import argparse
import time

import numpy as np
import yaml

from XPolicyLab.policy.XBrain_v1.model import Model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-cfg-type", choices=("piper_x", "piper", "arx_x5"), required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config.update(
        env_cfg_type=args.env_cfg_type,
        action_type="joint",
        action_horizon=50,
        prompt_max_length=120,
    )

    model = Model(config)
    model.reset()
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    observation = {
        "vision": {
            "cam_head": {"color": image},
            "cam_left_wrist": {"color": image},
            "cam_right_wrist": {"color": image},
        },
        "state": np.zeros(14, dtype=np.float32),
        "instruction": "Complete the task described by the instruction.",
    }
    model.update_obs(observation)
    started = time.monotonic()
    actions = model.get_action()
    elapsed = time.monotonic() - started

    assert len(actions) == 50
    expected_shapes = {
        "left_arm_joint_state": (6,),
        "left_ee_joint_state": (1,),
        "right_arm_joint_state": (6,),
        "right_ee_joint_state": (1,),
    }
    for action in actions:
        assert set(action) == set(expected_shapes)
        for key, shape in expected_shapes.items():
            value = np.asarray(action[key])
            assert value.shape == shape, (key, value.shape)
            assert np.isfinite(value).all(), key
    print(
        f"POLICY_INTERFACE_OK robot={args.env_cfg_type} seconds={elapsed:.3f} "
        "action_count=50 action_dim=14 finite=true"
    )


if __name__ == "__main__":
    main()
