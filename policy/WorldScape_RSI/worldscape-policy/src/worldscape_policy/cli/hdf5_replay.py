from __future__ import annotations

import argparse

import torch

from evals.common.hdf5_replay import (
    HDF5ReplayConfig,
    run_hdf5_replay,
)
from worldscape_policy.native_builder import (
    build_wan22_policy_from_checkpoint,
    checkpoint_mode,
    checkpoint_supports_mode,
)
from worldscape_policy.rollout.session import PolicyRuntime
from worldscape_policy.types import InteractionMode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run native WorldScape Policy inference on an HDF5 replay"
    )
    parser.add_argument("checkpoint")
    parser.add_argument("hdf5")
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--embodiment-id", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--visual-input-range",
        choices=("zero_one", "minus_one_one", "uint8"),
        default="zero_one",
    )
    parser.add_argument("--mode", choices=("auto", "interactive"))
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--num-history-frames", type=int, default=4)
    parser.add_argument("--action-mode", choices=("eef",), default="eef")
    parser.add_argument("--seed", type=int, default=1140)
    args = parser.parse_args()

    checkpoint_primary_mode = checkpoint_mode(
        args.checkpoint, validate_artifacts=False
    )
    mode = InteractionMode.parse(
        args.mode if args.mode is not None else checkpoint_primary_mode
    )
    if not checkpoint_supports_mode(checkpoint_primary_mode, mode):
        parser.error(
            f"checkpoint mode is {checkpoint_primary_mode.value!r}, "
            f"not requested mode {mode.value!r}"
        )
    policy = build_wan22_policy_from_checkpoint(
        args.checkpoint,
        visual_input_range=args.visual_input_range,
        device=args.device,
        expected_mode=mode,
    )
    generator = torch.Generator(device=torch.device(args.device)).manual_seed(
        args.seed
    )
    outputs = run_hdf5_replay(
        PolicyRuntime(policy),
        HDF5ReplayConfig(
            path=args.hdf5,
            mode=mode,
            instruction=args.instruction,
            embodiment_id=args.embodiment_id,
            max_steps=args.max_steps,
            num_history_frames=args.num_history_frames,
            action_mode=args.action_mode,
        ),
        generator=generator,
    )
    print(f"Completed {len(outputs)} native replay prediction(s)")


if __name__ == "__main__":
    main()
