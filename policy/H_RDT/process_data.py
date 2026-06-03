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
        image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)
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


def _discover_task_names(source_root):
    if source_root is None:
        raise ValueError("XPOLICY_HRDT_TASKS=all requires XPOLICY_HRDT_SOURCE_ROOT.")

    root = Path(source_root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"XPOLICY_HRDT_SOURCE_ROOT does not exist: {root}")

    task_names = sorted(
        path.name
        for path in root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    if not task_names:
        raise ValueError(f"No task directories found under XPOLICY_HRDT_SOURCE_ROOT: {root}")
    return task_names


def _task_names_from_env(default_task_name, source_root):
    task_names_text = os.environ.get("XPOLICY_HRDT_TASKS") or os.environ.get("XPOLICY_HRDT_TASK_NAME")
    if task_names_text is None:
        return [default_task_name]

    if task_names_text.strip().lower() == "all":
        return _discover_task_names(source_root)

    task_names = [
        task_name.strip()
        for task_name in task_names_text.replace(",", " ").split()
        if task_name.strip()
    ]
    if not task_names:
        raise ValueError("XPOLICY_HRDT_TASKS/XPOLICY_HRDT_TASK_NAME did not contain any task names.")
    return task_names


def _resolve_source_episode_dir(script_dir, dataset_name, task_name, env_cfg_type):
    source_root = os.environ.get("XPOLICY_HRDT_SOURCE_ROOT")
    if source_root is None:
        candidates = [
            script_dir.parents[2] / "data" / dataset_name / task_name / env_cfg_type / "data",
        ]
    else:
        root = Path(source_root).expanduser()
        candidates = [
            root / task_name / "data" / task_name / env_cfg_type / "data",
            root / "data" / task_name / env_cfg_type / "data",
            root / dataset_name / task_name / env_cfg_type / "data",
            root / task_name / env_cfg_type / "data",
            root / task_name / "data",
        ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    candidate_text = "\n".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"Could not find source episode directory for task={task_name}. Tried:\n{candidate_text}"
    )


def _episode_files(source_episode_dir, expert_data_num):
    episode_files = sorted(
        path
        for path in source_episode_dir.glob("episode_*.hdf5")
        if not path.name.endswith("_tmp.hdf5")
    )
    if not episode_files:
        episode_files = sorted(source_episode_dir.glob("episode_*.hdf5"))

    if len(episode_files) < expert_data_num:
        raise FileNotFoundError(
            f"Need {expert_data_num} episodes under {source_episode_dir}, found {len(episode_files)}."
        )
    return episode_files[:expert_data_num]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name")
    parser.add_argument("ckpt_name")
    parser.add_argument("env_cfg_type")
    parser.add_argument("expert_data_num", type=int)
    parser.add_argument("action_type")
    args = parser.parse_args()

    if args.action_type != "joint":
        raise ValueError("H_RDT process_data.py currently supports only action_type='joint'.")

    script_dir = Path(__file__).resolve().parent
    source_root = os.environ.get("XPOLICY_HRDT_SOURCE_ROOT")
    task_names = _task_names_from_env(args.ckpt_name, source_root)

    output_root = script_dir / "data" / f"{args.dataset_name}-{args.ckpt_name}-{args.env_cfg_type}-{args.expert_data_num}-{args.action_type}"
    if output_root.exists():
        shutil.rmtree(output_root)

    robot_action_dim_info = get_robot_action_dim_info(args.env_cfg_type)
    all_actions = []
    task_instructions = {}

    for task_name in task_names:
        source_episode_dir = _resolve_source_episode_dir(
            script_dir,
            args.dataset_name,
            task_name,
            args.env_cfg_type,
        )
        episode_files = _episode_files(source_episode_dir, args.expert_data_num)

        output_episode_dir = output_root / task_name / "demo_clean" / "data"
        output_episode_dir.mkdir(parents=True, exist_ok=True)
        instruction = None

        for episode_idx, input_path in enumerate(episode_files):
            data = load_hdf5(str(input_path))
            if instruction is None:
                instruction = _instruction_from_episode(data, task_name)

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

            print(
                f"[H_RDT] processed task={task_name} episode "
                f"{episode_idx + 1}/{args.expert_data_num}: {output_path}"
            )

        task_instructions[task_name] = instruction or task_name.replace("_", " ")

    all_actions = np.concatenate(all_actions, axis=0)
    _write_stats(output_root / "stats.json", all_actions)
    for task_name, instruction in task_instructions.items():
        _update_task_instruction_csv(
            script_dir / "H_RDT" / "datasets" / "robotwin2" / "task_instructions.csv",
            task_name,
            instruction,
        )

    print(f"[H_RDT] processed data root: {output_root}")
    print(f"[H_RDT] processed tasks: {', '.join(task_names)}")
    print(f"[H_RDT] stats path: {output_root / 'stats.json'}")


if __name__ == "__main__":
    main()
