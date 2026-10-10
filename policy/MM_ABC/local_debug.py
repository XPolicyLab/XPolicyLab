"""In-process debug for the MM_ABC adapter: no server, no port.

Drives ``Model`` through reset / update_obs / get_action (and the batch path)
with synthetic Mobile observations and validates every returned action.

    PYTHONPATH=<workspace>:<workspace>/XPolicyLab \\
      python -m XPolicyLab.policy.MM_ABC.local_debug                     # mock
    PYTHONPATH=... python -m XPolicyLab.policy.MM_ABC.local_debug \\
      --checkpoint_path <MM-ABC run or milestone dir>                    # real
"""

from __future__ import annotations

import argparse

from XPolicyLab.policy.MM_ABC.debug_mobile_client import SyntheticSource, validate_action
from XPolicyLab.policy.MM_ABC.model import Model

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_path", default=None, help="omit for mock mode")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--gpu_id", default=None)
    args = ap.parse_args()

    model = Model({
        "policy_name": "MM_ABC",
        "action_type": "joint",
        "mock": args.checkpoint_path is None,
        "checkpoint_path": args.checkpoint_path,
        "gpu_id": args.gpu_id,
    })
    src = SyntheticSource(seed=0, image_hw=(720, 1280))

    model.reset()
    for step in range(args.steps):
        model.update_obs(src.obs(step, 0, "array"))
        chunk = model.get_action()
        assert isinstance(chunk, list) and chunk, "get_action must return a non-empty list"
        for action in chunk:
            validate_action(action, model.adapter.action_key_style)
        print(f"[step {step}] {len(chunk)} actions, left_arm_joint_state[0]={chunk[0]['left_arm_joint_state'].round(3)}")

    model.reset()
    model.update_obs_batch([src.obs(0, i, "array") for i in range(3)])
    batch = model.get_action_batch([0, 1, 2])
    assert len(batch) == 3
    for chunk in batch:
        for action in chunk:
            validate_action(action, model.adapter.action_key_style)
    print(f"[batch] 3 envs x {len(batch[0])} actions")
    print("\nLOCAL DEBUG PASSED: action keys and shapes match the Mobile contract.")

if __name__ == "__main__":
    main()
