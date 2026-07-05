"""Convert XPolicyLab HDF5 episodes into Galaxea LeRobot format (v3).

Uses the upstream LeRobotDataset writer for byte-compatible output.
Output matches ``xpolicylab/dual_arm_joint_robodojo``:
  - flat 14-dim ``observation.state`` / ``action``
  - cameras ``cam_high``, ``cam_left_wrist``, ``cam_right_wrist`` (RGB HWC 480x640)
"""

import argparse
import glob
import os
import shutil

import cv2
import numpy as np
from tqdm import tqdm

from XPolicyLab.utils.load_file import load_hdf5
from XPolicyLab.utils.process_data import (
    decode_image_bit,
    get_robot_action_dim_info,
    pack_robot_state,
)

from galaxea_fm.data.lerobot.lerobot_dataset_v3 import LeRobotDataset

STD_W, STD_H = 640, 480
CAM_MAP = {
    "cam_head": "cam_high",
    "cam_left_wrist": "cam_left_wrist",
    "cam_right_wrist": "cam_right_wrist",
}


def _state_keys(robot_action_dim_info: dict):
    arm_dims = robot_action_dim_info["arm_dim"]
    ee_dims = robot_action_dim_info["ee_dim"]
    if len(arm_dims) == 1:
        return [("left_arm", arm_dims[0]), ("left_gripper", ee_dims[0])]
    return [
        ("left_arm", arm_dims[0]),
        ("left_gripper", ee_dims[0]),
        ("right_arm", arm_dims[1]),
        ("right_gripper", ee_dims[1]),
    ]


def _build_features(key_dims):
    vec_dim = sum(dim for _, dim in key_dims)
    features = {
        "observation.state": {"dtype": "float32", "shape": (vec_dim,), "names": None},
        "action": {"dtype": "float32", "shape": (vec_dim,), "names": None},
    }
    for cam_key in CAM_MAP.values():
        features[f"observation.images.{cam_key}"] = {
            "dtype": "video",
            "shape": (STD_H, STD_W, 3),
            "names": ["height", "width", "channel"],
        }
    return features


def _standardize(rgb: np.ndarray) -> np.ndarray:
    # decode_image_bit already yields RGB HWC for XPolicyLab HDF5 (see README §4.4).
    img = cv2.resize(rgb, (STD_W, STD_H), interpolation=cv2.INTER_AREA)
    assert img.shape == (STD_H, STD_W, 3), img.shape
    return np.ascontiguousarray(img, dtype=np.uint8)


def _episode_paths(load_dir: str, max_episodes: int):
    paths = sorted(glob.glob(os.path.join(load_dir, "data", "episode_*.hdf5")))
    if max_episodes and max_episodes > 0:
        paths = paths[:max_episodes]
    return paths


def _resolve_task_load_dir(batch_root: str, task: str, env_cfg_type: str) -> str | None:
    """Find task/<env_cfg_type>/data (exact subdir name only)."""
    load_dir = os.path.join(batch_root, task, env_cfg_type)
    if os.path.isdir(os.path.join(load_dir, "data")):
        return load_dir
    return None


def _instruction_from_episode(data: dict, fallback: str) -> str:
    inst = data.get("instruction")
    if inst is None:
        return fallback
    if isinstance(inst, bytes):
        inst = inst.decode("utf-8")
    elif hasattr(inst, "item"):
        inst = inst.item()
    inst = str(inst).strip()
    return inst if inst else fallback


def _add_episode(dataset, hdf5_path, key_dims, robot_action_dim_info, action_type, instruction, position):
    data = load_hdf5(hdf5_path)
    instruction = _instruction_from_episode(data, instruction)
    state_all = pack_robot_state(
        data, action_type, robot_action_dim_info, source_type="dataset", state_type="state"
    )
    action_all = pack_robot_state(
        data, action_type, robot_action_dim_info, source_type="dataset", state_type="action"
    )
    decoded = {
        cam_key: decode_image_bit(data["vision"][cam_src]["colors"])
        for cam_src, cam_key in CAM_MAP.items()
    }

    num_frames = state_all.shape[0]
    for t in tqdm(range(num_frames), desc=position, leave=False):
        frame = {
            "observation.state": state_all[t].astype(np.float32),
            "action": action_all[t].astype(np.float32),
        }
        for cam_key, frames in decoded.items():
            frame[f"observation.images.{cam_key}"] = _standardize(frames[t])
        frame["task"] = instruction
        dataset.add_frame(frame)
    dataset.save_episode()
    return num_frames


def _workspace_and_policy_dirs(here: str) -> tuple[str, str]:
    policy_dir = os.path.abspath(os.path.join(here, "..", ".."))
    workspace = os.environ.get("XPOLICYLAB_WORKSPACE")
    if workspace:
        workspace = os.path.abspath(workspace)
    else:
        workspace = os.path.abspath(os.path.join(policy_dir, "..", ".."))
        if not os.path.isdir(os.path.join(workspace, "data")) and os.path.isdir(
            os.path.join(workspace, "..", "data")
        ):
            workspace = os.path.abspath(os.path.join(workspace, ".."))
    return workspace, policy_dir


def _dataset_tag(bench_name: str, ckpt_name: str, env_cfg_type: str, action_type: str) -> str:
    return f"{bench_name}-{ckpt_name}-{env_cfg_type}-{action_type}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bench_name")
    parser.add_argument("ckpt_name")
    parser.add_argument("env_cfg_type")
    parser.add_argument("expert_data_num", type=int, help="single: #episodes; batch: max per task (0=all)")
    parser.add_argument("action_type", choices=["joint"])
    parser.add_argument("--src_root", default=None, help="defaults to <workspace>/data")
    parser.add_argument("--batch_root", default=None)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--out_root", default=None)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--robot_type", default=None, help="defaults to env_cfg_type")
    parser.add_argument("--instruction", default=None)
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    workspace_root, policy_dir = _workspace_and_policy_dirs(here)
    data_out_dir = os.path.join(policy_dir, "data")

    robot_action_dim_info = get_robot_action_dim_info(args.env_cfg_type)
    key_dims = _state_keys(robot_action_dim_info)
    features = _build_features(key_dims)

    if args.batch_root:
        batch_root = os.path.abspath(args.batch_root)
        all_tasks = sorted(
            d for d in os.listdir(batch_root)
            if os.path.isdir(os.path.join(batch_root, d))
            and _resolve_task_load_dir(batch_root, d, args.env_cfg_type) is not None
        )
        tasks = [t for t in all_tasks if (args.tasks is None or t in args.tasks)]
        if not tasks:
            raise SystemExit(
                f"no tasks with {args.env_cfg_type}/data under {batch_root}"
            )
        tag = _dataset_tag(args.bench_name, args.ckpt_name, args.env_cfg_type, args.action_type)
        plan = []
        for task in tasks:
            load_dir = _resolve_task_load_dir(batch_root, task, args.env_cfg_type)
            instruction = args.instruction or task.replace("_", " ")
            for ep_path in _episode_paths(load_dir, args.expert_data_num):
                plan.append((ep_path, instruction, task))
    else:
        src_root = args.src_root or os.path.join(workspace_root, "data")
        load_dir = os.path.join(src_root, args.bench_name, args.ckpt_name, args.env_cfg_type)
        tag = _dataset_tag(args.bench_name, args.ckpt_name, args.env_cfg_type, args.action_type)
        instruction = args.instruction or args.ckpt_name.replace("_", " ")
        plan = [(p, instruction, args.ckpt_name) for p in _episode_paths(load_dir, args.expert_data_num)]

    if not plan:
        raise SystemExit("no episodes found to convert")

    out_root = os.path.abspath(args.out_root or os.path.join(data_out_dir, f"{tag}-lerobot"))
    if os.path.exists(out_root):
        shutil.rmtree(out_root)
    os.makedirs(os.path.dirname(out_root), exist_ok=True)

    dataset = LeRobotDataset.create(
        repo_id=f"xpolicylab/{tag}".replace(" ", "_"),
        fps=args.fps,
        features=features,
        root=out_root,
        robot_type=args.robot_type or args.env_cfg_type,
        use_videos=True,
    )

    n_tasks = len({label for _, _, label in plan})
    print(f"[convert] mode={'batch' if args.batch_root else 'single'}  episodes={len(plan)}  tasks={n_tasks}")
    print(f"[convert] out={out_root}  keys={[k for k, _ in key_dims]}")

    total_frames = 0
    for idx, (ep_path, instruction, label) in enumerate(tqdm(plan, desc="episodes")):
        if not os.path.exists(ep_path):
            raise FileNotFoundError(f"missing episode file: {ep_path}")
        total_frames += _add_episode(
            dataset, ep_path, key_dims, robot_action_dim_info, args.action_type,
            instruction, position=f"{label}[{idx + 1}/{len(plan)}]",
        )

    dataset.meta._close_writer()
    print(f"[convert] done: {len(plan)} episodes / {total_frames} frames / {n_tasks} tasks -> {out_root}")


if __name__ == "__main__":
    main()
