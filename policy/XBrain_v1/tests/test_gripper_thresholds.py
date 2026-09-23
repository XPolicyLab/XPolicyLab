"""Check prompt-specific left/right gripper threshold behavior."""

from pathlib import Path

import numpy as np

from XPolicyLab.policy.XBrain_v1.gripper_thresholds import (
    apply_gripper_thresholds,
    load_gripper_thresholds,
    normalize_prompt,
)


ROOT = Path(__file__).resolve().parent.parent
EXPECTED_TASK_COUNTS = {"piper_x": 6, "piper": 6, "arx_x5": 6}


def main():
    config_path = ROOT / "gripper_thresholds.json"
    for env_cfg_type, expected_count in EXPECTED_TASK_COUNTS.items():
        thresholds = load_gripper_thresholds(config_path, env_cfg_type)
        assert len(thresholds) == expected_count

    piper_x = load_gripper_thresholds(config_path, "piper_x")
    sweep_prompt = (
        "  PICK up the broom, hand it over to the right hand, then use the "
        "dustpan to sweep the blocks  "
    )
    rule = piper_x[normalize_prompt(sweep_prompt)]
    assert rule.task == "sweep_blocks"
    assert rule.left == 0.35
    assert rule.right == 0.3

    actions = np.zeros((3, 14), dtype=np.float32)
    actions[:, 6] = [0.349, 0.35, 0.351]
    actions[:, 13] = [0.299, 0.3, 0.301]
    filtered = apply_gripper_thresholds(actions, rule)

    np.testing.assert_allclose(filtered[:, 6], [0.0, 0.35, 0.351])
    np.testing.assert_allclose(filtered[:, 13], [0.0, 0.3, 0.301])
    np.testing.assert_allclose(actions[:, 6], [0.349, 0.35, 0.351])
    np.testing.assert_allclose(actions[:, 13], [0.299, 0.3, 0.301])
    print("GRIPPER_THRESHOLDS_OK tasks=18 left_right_independent=true")


if __name__ == "__main__":
    main()
