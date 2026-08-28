"""Reload a KinRT checkpoint and run one offline RoboDojo inference."""

from __future__ import annotations

import argparse
import importlib.machinery
import json
from pathlib import Path
import sys
import time
import types

import numpy as np
# The offline check does not use XPolicyLab's image codecs. Providing the module
# placeholder keeps its state-packing utilities usable on headless OpenPI nodes.
cv2_placeholder = types.ModuleType("cv2")
cv2_placeholder.__spec__ = importlib.machinery.ModuleSpec("cv2", loader=None)
sys.modules["cv2"] = cv2_placeholder

from XPolicyLab.policy.KinRT.model import Model


CAMERA_NAMES = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, default=100)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--repo-id", default="RoboDojo-KinRT-demo-arx_x5-joint")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--action-chunk-size", type=int, default=50)
    parser.add_argument("--train-config-name", default="kinrt_lora_robodojo")
    parser.add_argument("--actions-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset_root is None:
        observation = {
            "images": {
                camera: np.zeros((3, 480, 640), dtype=np.uint8)
                for camera in CAMERA_NAMES
            },
            "state": np.zeros(14, dtype=np.float32),
            "instruction": "stack the bowls",
        }
        input_source = "synthetic"
    else:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset(args.repo_id, root=args.dataset_root)
        sample = dataset[args.sample_index]
        observation = {
            "images": {
                camera: _to_numpy(sample[f"observation.images.{camera}"])
                for camera in CAMERA_NAMES
            },
            "state": _to_numpy(sample["observation.state"]),
            "instruction": sample["task"],
        }
        input_source = str(args.dataset_root)

    load_started = time.perf_counter()
    model = Model(
        {
            "action_type": "joint",
            "env_cfg_type": "arx_x5",
            "checkpoint_path": str(args.checkpoint_root),
            "checkpoint_num": args.checkpoint_step,
            "train_config_name": args.train_config_name,
            "repo_id": args.repo_id,
            "action_chunk_size": args.action_chunk_size,
        }
    )
    model_load_seconds = time.perf_counter() - load_started
    model.update_obs(observation)
    inference_started = time.perf_counter()
    structured_actions = model.get_action()
    inference_seconds = time.perf_counter() - inference_started
    action_keys = (
        "left_arm_joint_state",
        "left_ee_joint_state",
        "right_arm_joint_state",
        "right_ee_joint_state",
    )
    actions = np.stack(
        [np.concatenate([np.asarray(action[key]) for key in action_keys]) for action in structured_actions]
    )

    expected_shape = (args.action_chunk_size, 14)
    if actions.shape != expected_shape:
        raise RuntimeError(f"Expected action shape {expected_shape}, got {actions.shape}.")
    if not np.isfinite(actions).all():
        raise RuntimeError("Inference returned NaN or infinite actions.")
    if args.actions_output is not None:
        args.actions_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.actions_output, actions)

    print(
        json.dumps(
            {
                "checkpoint_step": args.checkpoint_step,
                "train_config_name": args.train_config_name,
                "sample_index": args.sample_index,
                "instruction": observation["instruction"],
                "input_source": input_source,
                "state_shape": list(observation["state"].shape),
                "image_shapes": {
                    camera: list(observation["images"][camera].shape)
                    for camera in CAMERA_NAMES
                },
                "action_shape": list(actions.shape),
                "action_min": float(actions.min()),
                "action_max": float(actions.max()),
                "action_mean": float(actions.mean()),
                "model_load_seconds": model_load_seconds,
                "inference_seconds": inference_seconds,
                "actions_output": str(args.actions_output) if args.actions_output else None,
                "finite": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
