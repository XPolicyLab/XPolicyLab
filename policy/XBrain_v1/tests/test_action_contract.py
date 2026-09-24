"""Check the published XBrain-v1 action semantics and robot mappings."""

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
EXPECTED_MASK = [True] * 6 + [False] + [True] * 6 + [False]


def main() -> None:
    expected_frequency = {"piper_x": 25, "piper": 30, "arx_x5": 30}
    for name, embodiment in (("piper_x", 6), ("piper", 6), ("arx_x5", 0)):
        config = yaml.safe_load(
            (ROOT / "runtime_config" / f"{name}.yaml").read_text(encoding="utf-8")
        )
        assert config["embodiment_id"] == embodiment
        assert config["state_dim"] == 14
        assert config["action_chunk_size"] == 50
        assert config["action_horizon"] == 30
        assert config["control_frequency_hz"] == expected_frequency[name]
        assert config["training_arm_action_mode"] == "delta"
        assert config["policy_output_arm_action_mode"] == "absolute"
        assert config["gripper_action_mode"] == "absolute"
        assert config["executor_applies_joint_delta"] is False
        assert config["delta_mask"] == EXPECTED_MASK
    print("ACTION_CONTRACT_OK")


if __name__ == "__main__":
    main()
