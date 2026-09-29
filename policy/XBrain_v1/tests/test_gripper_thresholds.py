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
EXPECTED_OVERRIDES = {
    ("piper_x", "pack_objects_into_backpack"): (0.15, 0.15, 1.1, 1.1),
    ("piper_x", "classify_objects"): (0.25, 0.25, 1.3, 1.3),
    ("piper_x", "sweep_blocks"): (0.35, 0.30, 1.3, 1.0),
    ("piper", "fill_pen_holder"): (0.35, 0.35, 1.1, 1.1),
    ("piper", "put_objects_into_basket"): (0.33, 0.33, 1.15, 1.15),
    ("piper", "insert_charger"): (0.20, 0.20, 1.1, 1.1),
    ("piper", "stack_and_cover_blocks"): (0.45, 0.45, 1.1, 1.1),
    ("arx_x5", "pack_and_pour_fruit"): (0.30, 0.30, 1.1, 1.1),
}


def main():
    config_path = ROOT / "gripper_thresholds.json"
    for env_cfg_type, expected_count in EXPECTED_TASK_COUNTS.items():
        thresholds = load_gripper_thresholds(config_path, env_cfg_type)
        assert len(thresholds) == expected_count
        for rule in thresholds.values():
            if (env_cfg_type, rule.task) not in EXPECTED_OVERRIDES:
                assert rule.left_scale == rule.right_scale == 1.3

    for (robot, task), expected in EXPECTED_OVERRIDES.items():
        rules = load_gripper_thresholds(config_path, robot)
        rule = next(rule for rule in rules.values() if rule.task == task)
        assert (rule.left, rule.right, rule.left_scale, rule.right_scale) == expected
        left, right, left_scale, right_scale = expected
        values = np.zeros((3, 14), dtype=np.float32)
        values[:, 6] = [left - 0.01, left, left + 0.01]
        values[:, 13] = [right + 0.01, right, right - 0.01]
        result = apply_gripper_thresholds(values, rule)
        np.testing.assert_allclose(result[:, 6], [0, left * left_scale, (left + 0.01) * left_scale], rtol=1e-6)
        np.testing.assert_allclose(result[:, 13], [(right + 0.01) * right_scale, right * right_scale, 0], rtol=1e-6)

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
    np.testing.assert_allclose(filtered[:, 13], [0.0, 0.3, 0.301, 0.31, 0.0, 0.8])
    arm_columns = list(range(6)) + list(range(7, 13))
    np.testing.assert_array_equal(filtered[:, arm_columns], original[:, arm_columns])
    np.testing.assert_array_equal(actions, original)
    assert filtered.dtype == actions.dtype
    print("GRIPPER_THRESHOLDS_OK tasks=18 left_right_independent=true task_specific_scales=true")


if __name__ == "__main__":
    main()
