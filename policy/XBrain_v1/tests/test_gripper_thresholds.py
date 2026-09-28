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

    actions = np.arange(6 * 14, dtype=np.float32).reshape(6, 14) / 100
    actions[:, 6] = [0.349, 0.35, 0.351, 0.34, 0.36, 0.9]
    actions[:, 13] = [0.299, 0.3, 0.301, 0.31, 0.29, 0.8]
    original = actions.copy()
    filtered = apply_gripper_thresholds(actions, rule)

    np.testing.assert_allclose(filtered[:, 6], [0.0, 0.455, 0.4563, 0.0, 0.468, 1.17])
    np.testing.assert_allclose(filtered[:, 13], [0.0, 0.39, 0.3913, 0.403, 0.0, 1.04])
    arm_columns = list(range(6)) + list(range(7, 13))
    np.testing.assert_array_equal(filtered[:, arm_columns], original[:, arm_columns])
    np.testing.assert_array_equal(actions, original)
    assert filtered.dtype == actions.dtype
    print("GRIPPER_THRESHOLDS_OK tasks=18 left_right_independent=true at_or_above_scale=1.3")


if __name__ == "__main__":
    main()
