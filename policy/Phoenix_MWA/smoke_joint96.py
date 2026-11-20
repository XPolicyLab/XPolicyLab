"""Load a real Joint96 checkpoint and predict from synthetic RGB/state inputs.

This does not connect to a robot or simulator and does not measure task success.
Run as python -m XPolicyLab.policy.Phoenix_MWA.smoke_joint96 from the parent
workspace, which must contain the benchmark's env_cfg/ robot definitions.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .model import Model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument('--robot-type', choices=('arx_x5', 'piper', 'piper_x'), default='arx_x5')
    parser.add_argument('--rounds', type=int, default=1, help='Sequential acknowledged inference chunks')
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error('--rounds must be positive')
    started = time.perf_counter()
    model = Model(dict(
        runtime_mode="joint96_real_phoenix", checkpoint_path=str(args.weights.resolve()),
        expected_step=args.expected_step, audit_dir=str(args.audit_dir.resolve()),
        env_cfg_type=args.robot_type, action_type="joint", action_horizon=32,
        replan_steps=32, num_inference_steps=10, mixed_precision="bf16",
        device="cuda:0", seed=0, allow_dummy_policy=False,
    ))
    loaded = time.perf_counter()
    image = np.full((480, 640, 3), [23, 81, 190], dtype=np.uint8)
    obs = dict(
        vision={name: {"color": image.copy()} for name in ("cam_head", "cam_left_wrist", "cam_right_wrist")},
        state=dict(left_arm_joint_state=np.zeros(6, dtype=np.float32),
                   left_ee_joint_state=np.array([0.5], dtype=np.float32),
                   right_arm_joint_state=np.zeros(6, dtype=np.float32),
                   right_ee_joint_state=np.array([0.5], dtype=np.float32)),
        instruction="cover the blocks",
    )
    model.update_obs(obs)
    actions = model.get_action()
    assert len(actions) == model.replan_steps
    for action in actions:
        assert set(action) == set(obs["state"])
        for key, value in action.items():
            assert value.shape == obs["state"][key].shape, (key, value.shape)
            assert np.isfinite(value).all(), key
    inferred = time.perf_counter()
    assert len(model.session.pending_model_actions) == model.replan_steps
    for _ in actions:
        model.update_obs(obs)
    assert not model.session.pending_model_actions
    inference_seconds = [inferred - loaded]
    for _ in range(args.rounds - 1):
        started_round = time.perf_counter()
        next_actions = model.get_action()
        inference_seconds.append(time.perf_counter() - started_round)
        assert len(next_actions) == model.replan_steps
        for action in next_actions:
            assert set(action) == set(obs['state'])
            for key, value in action.items():
                assert value.shape == obs['state'][key].shape and np.isfinite(value).all()
        assert len(model.session.pending_model_actions) == model.replan_steps
        for _ in next_actions:
            model.update_obs(obs)
        assert not model.session.pending_model_actions
    model.reset()
    assert model.last_obs is None and not model.session.pending_model_actions
    result = dict(
        status="passed", step=args.expected_step, dummy_policy=False,
        robot_type=args.robot_type, embodiment_id=model.runtime.embodiment_id,
        robot_support=model.runtime.robot_support,
        input="synthetic_rgb_and_joint_state", robot_or_simulator_connected=False,
        task_success_measured=False, actions_returned=len(actions),
        action_shapes={key: list(value.shape) for key, value in actions[0].items()},
        load_seconds=loaded - started, first_inference_seconds=inferred - loaded,
        inference_rounds=args.rounds, inference_seconds=inference_seconds,
        cuda_device=torch.cuda.get_device_name(0),
        peak_allocated_gib=torch.cuda.max_memory_allocated(0) / 2**30,
        acknowledgments_and_reset="passed",
    )
    args.audit_dir.mkdir(parents=True, exist_ok=True)
    (args.audit_dir / "SMOKE_RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
    source = Path(__file__).resolve().parent / "joint96_source"
    imported = sorted({str(Path(module.__file__).resolve().relative_to(source))
        for module in tuple(sys.modules.values())
        if getattr(module, "__file__", None) and Path(module.__file__).resolve().is_relative_to(source)})
    (args.audit_dir / "IMPORTED_SOURCE_FILES.json").write_text(json.dumps(imported, indent=2) + "\n")
    np.savez(args.audit_dir / "ACTIONS.npz", **{key: np.stack([a[key] for a in actions]) for key in actions[0]})
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
