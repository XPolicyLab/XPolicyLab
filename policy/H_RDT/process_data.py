import argparse
import csv
import json
import os
import shutil
from pathlib import Path

import cv2
import h5py
import numpy as np

from XPolicyLab.utils.load_file import load_hdf5
from XPolicyLab.utils.process_data import (
    decode_image_bit,
    get_robot_action_dim_info,
    pack_robot_state,
)


def _encode_jpeg(image):
    success, encoded = cv2.imencode(".jpg", image)
    if not success:
        raise ValueError("Failed to encode image as JPEG.")
    return np.asarray(encoded, dtype=np.uint8)


def _camera_frames(data, camera_name):
    frames = []
    for image_bits in data["vision"][camera_name]["colors"]:
        image = decode_image_bit(image_bits)
        image = cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA)
        frames.append(_encode_jpeg(image))
    return frames


def _write_vlen_images(group, name, encoded_frames):
    dtype = h5py.vlen_dtype(np.dtype("uint8"))
    dataset = group.create_dataset(name, (len(encoded_frames),), dtype=dtype)
    for idx, encoded in enumerate(encoded_frames):
        dataset[idx] = encoded


def _split_joint_vector(joint_vector, robot_action_dim_info):
    arm_dims = robot_action_dim_info["arm_dim"]
    ee_dims = robot_action_dim_info["ee_dim"]
    if len(arm_dims) != 2 or len(ee_dims) != 2:
        raise ValueError("H_RDT training currently expects a dual-arm joint action space.")

    left_arm_dim, right_arm_dim = arm_dims
    left_ee_dim, right_ee_dim = ee_dims
    offset = 0
    left_arm = joint_vector[:, offset : offset + left_arm_dim]
    offset += left_arm_dim
    left_gripper = joint_vector[:, offset : offset + left_ee_dim]
    offset += left_ee_dim
    right_arm = joint_vector[:, offset : offset + right_arm_dim]
    offset += right_arm_dim
    right_gripper = joint_vector[:, offset : offset + right_ee_dim]
    return left_arm, left_gripper, right_arm, right_gripper


def _instruction_from_episode(data, task_name):
    instruction = data.get("instruction") or data.get("instructions")
    if isinstance(instruction, (list, tuple, np.ndarray)) and len(instruction) > 0:
        return str(instruction[0])
    if instruction:
        return str(instruction)
    return task_name.replace("_", " ")


def _update_task_instruction_csv(csv_path, task_name, instruction):
    rows = []
    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            rows = [row for row in reader if row.get("task_name") != task_name]

    rows.append({"task_name": task_name, "instruction": instruction})
    rows.sort(key=lambda row: row["task_name"])

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["task_name", "instruction"])
        writer.writeheader()
        writer.writerows(rows)


def _write_stats(stats_path, action_array):
    action_min = np.min(action_array, axis=0).astype(float).tolist()
    action_max = np.max(action_array, axis=0).astype(float).tolist()
    stats = {
        "robotwin_agilex": {
            "min": action_min,
            "max": action_max,
            "file_count": int(action_array.shape[0]),
            "total_files_scanned": int(action_array.shape[0]),
            "action_dim": int(action_array.shape[1]),
        }
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as fp:
        json.dump(stats, fp, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name")
    parser.add_argument("task_name")
    parser.add_argument("env_cfg_type")
    parser.add_argument("expert_data_num", type=int)
    parser.add_argument("action_type")
    args = parser.parse_args()

    if args.action_type != "joint":
        raise ValueError("H_RDT process_data.py currently supports only action_type='joint'.")

    script_dir = Path(__file__).resolve().parent
    data_root = script_dir.parents[2] / "data" / args.dataset_name / args.task_name / args.env_cfg_type
    source_episode_dir = data_root / "data"
    if not source_episode_dir.exists():
        raise FileNotFoundError(f"XPolicyLab source data directory not found: {source_episode_dir}")

    output_root = script_dir / "data" / f"{args.dataset_name}-{args.task_name}-{args.env_cfg_type}-{args.expert_data_num}-{args.action_type}"
    output_episode_dir = output_root / args.task_name / "demo_clean" / "data"
    if output_root.exists():
        shutil.rmtree(output_root)
    output_episode_dir.mkdir(parents=True, exist_ok=True)

    robot_action_dim_info = get_robot_action_dim_info(args.env_cfg_type)
    all_actions = []
    instruction = None

    for episode_idx in range(args.expert_data_num):
        input_path = source_episode_dir / f"episode_{episode_idx:07d}.hdf5"
        if not input_path.exists():
            raise FileNotFoundError(f"Missing source episode: {input_path}")

        data = load_hdf5(str(input_path))
        if instruction is None:
            instruction = _instruction_from_episode(data, args.task_name)

        action_all = pack_robot_state(
            data,
            args.action_type,
            robot_action_dim_info,
            source_type="dataset",
            state_type="action",
        ).astype(np.float32)
        all_actions.append(action_all)
        left_arm, left_gripper, right_arm, right_gripper = _split_joint_vector(
            action_all,
            robot_action_dim_info,
        )

        output_path = output_episode_dir / f"episode_{episode_idx:07d}.hdf5"
        with h5py.File(output_path, "w", libver="latest") as fp:
            joint_action = fp.create_group("joint_action")
            joint_action.create_dataset("left_arm", data=left_arm, dtype="float32")
            joint_action.create_dataset("left_gripper", data=left_gripper, dtype="float32")
            joint_action.create_dataset("right_arm", data=right_arm, dtype="float32")
            joint_action.create_dataset("right_gripper", data=right_gripper, dtype="float32")

            observation = fp.create_group("observation")
            camera_paths = {
                "head_camera": "cam_head",
                "left_camera": "cam_left_wrist",
                "right_camera": "cam_right_wrist",
            }
            for hrdt_camera_name, xpolicy_camera_name in camera_paths.items():
                camera_group = observation.create_group(hrdt_camera_name)
                encoded_frames = _camera_frames(data, xpolicy_camera_name)
                _write_vlen_images(camera_group, "rgb", encoded_frames)

        print(f"[H_RDT] processed episode {episode_idx + 1}/{args.expert_data_num}: {output_path}")

    all_actions = np.concatenate(all_actions, axis=0)
    _write_stats(output_root / "stats.json", all_actions)
    _update_task_instruction_csv(
        script_dir / "H_RDT" / "datasets" / "robotwin2" / "task_instructions.csv",
        args.task_name,
        instruction or args.task_name.replace("_", " "),
    )

    print(f"[H_RDT] processed data root: {output_root}")
    print(f"[H_RDT] stats path: {output_root / 'stats.json'}")


if __name__ == "__main__":
    main()
