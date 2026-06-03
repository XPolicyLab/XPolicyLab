"""Convert XPolicyLab HDF5 episodes into the Galaxea LeRobot (v2.1/v3) format.

There is no HDF5 converter upstream, so this writes datasets with the *upstream*
LeRobotDataset writer (create / add_frame / save_episode) to guarantee the
output is byte-compatible with galaxea_fm's reader.

Two modes:
  * single (default): one (dataset_name, task_name, env_cfg_type) -> one dataset
        <src_root>/<dataset_name>/<task_name>/<env_cfg_type>/data/episode_*.hdf5
  * batch (--batch_root DIR): merge ALL tasks under DIR into ONE multi-task
        dataset, with per-episode instruction = task dir name. Expects layout
        DIR/<task>/<env_cfg_type>/data/episode_*.hdf5

Per-frame source fields:
  data['state'|'action']            -> packed via pack_robot_state(source_type='dataset')
  data['vision'][cam]['colors'][t]  -> jpeg bytes (BGR after decode)

Output feature keys consumed by base_lerobot_dataset.py:
  observation.state.<key> / action.<key>      (left_arm/left_gripper/...)
  observation.images.<cam>                     (head_rgb/left_wrist_rgb/right_wrist_rgb)

All camera frames are standardized to RGB HWC (240, 320, 3) -- the XPolicyLab
image standard -- matching the deploy-time preprocessing in model.py.
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

STD_W, STD_H = 320, 240
# XPolicyLab dataset camera -> upstream image feature key.
CAM_MAP = {
    "cam_head": "head_rgb",
    "cam_left_wrist": "left_wrist_rgb",
    "cam_right_wrist": "right_wrist_rgb",
}


def _state_keys(robot_action_dim_info: dict):
    """Galaxea state/action keys + dims, in XPolicyLab packed order.

    Packed order is [arm_0, ee_0, arm_1, ee_1] (see process_data.pack_robot_state),
    which equals Galaxea's [left_arm, left_gripper, right_arm, right_gripper].
    """
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
    features = {}
    for name, dim in key_dims:
        features[f"observation.state.{name}"] = {"dtype": "float32", "shape": (dim,), "names": None}
    for name, dim in key_dims:
        features[f"action.{name}"] = {"dtype": "float32", "shape": (dim,), "names": None}
    for cam_key in CAM_MAP.values():
        features[f"observation.images.{cam_key}"] = {
            "dtype": "video",
            "shape": (STD_H, STD_W, 3),
            "names": ["height", "width", "channel"],
        }
    return features


def _split(packed_row, key_dims):
    out, offset = {}, 0
    for name, dim in key_dims:
        out[name] = packed_row[offset:offset + dim].astype(np.float32)
        offset += dim
    return out


def _standardize(bgr_image) -> np.ndarray:
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (STD_W, STD_H), interpolation=cv2.INTER_AREA)
    assert rgb.shape == (STD_H, STD_W, 3), rgb.shape
    return np.ascontiguousarray(rgb, dtype=np.uint8)


def _episode_paths(load_dir: str, max_episodes: int):
    """Sorted episode hdf5 paths under <load_dir>/data, capped to max_episodes (<=0 = all)."""
    paths = sorted(glob.glob(os.path.join(load_dir, "data", "episode_*.hdf5")))
    if max_episodes and max_episodes > 0:
        paths = paths[:max_episodes]
    return paths


def _add_episode(dataset, hdf5_path, key_dims, robot_action_dim_info, action_type, instruction, position):
    """Add one episode (all frames) to the dataset and flush it. Returns frame count."""
    data = load_hdf5(hdf5_path)
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
        frame = {}
        for name, value in _split(state_all[t], key_dims).items():
            frame[f"observation.state.{name}"] = value
        for name, value in _split(action_all[t], key_dims).items():
            frame[f"action.{name}"] = value
        for cam_key, frames in decoded.items():
            frame[f"observation.images.{cam_key}"] = _standardize(frames[t])
        frame["task"] = instruction
        dataset.add_frame(frame)
    dataset.save_episode()
    return num_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_name")
    parser.add_argument("task_name", help="single mode: task; batch mode: ignored (use 'all')")
    parser.add_argument("env_cfg_type")
    parser.add_argument("expert_data_num", type=int, help="single: #episodes; batch: max episodes per task (0=all)")
    parser.add_argument("action_type", choices=["joint", "ee"])
    parser.add_argument("--src_root", default=None, help="single mode root; defaults to <repo_root>/data")
    parser.add_argument("--batch_root", default=None,
                        help="batch mode: merge all <task>/<env_cfg_type> under this dir into one dataset")
    parser.add_argument("--tasks", nargs="*", default=None, help="batch mode: optional subset of task dir names")
    parser.add_argument("--out_root", default=None)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--robot_type", default=None, help="defaults to env_cfg_type")
    parser.add_argument("--instruction", default=None,
                        help="single: episode instruction (default task_name); "
                             "batch: if set, used for ALL episodes instead of per-task name")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    # here -> .../XPolicyLab/policy/GalaxeaVLA/GalaxeaVLA/xpolicylab_adapter
    # repo_root (5 up) -> .../xspark-data/zijian (where data/ lives, like DP's ../../../data)
    repo_root = os.path.abspath(os.path.join(here, "..", "..", "..", "..", ".."))
    data_out_dir = os.path.abspath(os.path.join(here, "..", "..", "data"))

    robot_action_dim_info = get_robot_action_dim_info(args.env_cfg_type)
    key_dims = _state_keys(robot_action_dim_info)
    features = _build_features(key_dims)

    # ---- resolve mode -> list of (load_dir, instruction, label) episodes plan ----
    if args.batch_root:
        batch_root = os.path.abspath(args.batch_root)
        all_tasks = sorted(
            d for d in os.listdir(batch_root)
            if os.path.isdir(os.path.join(batch_root, d, args.env_cfg_type, "data"))
        )
        tasks = [t for t in all_tasks if (args.tasks is None or t in args.tasks)]
        if not tasks:
            raise SystemExit(f"no tasks with {args.env_cfg_type}/data under {batch_root}")
        tag = f"{args.dataset_name}-{args.env_cfg_type}-{args.action_type}"
        plan = []
        for task in tasks:
            load_dir = os.path.join(batch_root, task, args.env_cfg_type)
            instruction = args.instruction or task.replace("_", " ")
            for ep_path in _episode_paths(load_dir, args.expert_data_num):
                plan.append((ep_path, instruction, task))
    else:
        src_root = args.src_root or os.path.join(repo_root, "data")
        load_dir = os.path.join(src_root, args.dataset_name, args.task_name, args.env_cfg_type)
        tag = f"{args.dataset_name}-{args.task_name}-{args.env_cfg_type}-{args.expert_data_num}-{args.action_type}"
        instruction = args.instruction or args.task_name.replace("_", " ")
        plan = [(p, instruction, args.task_name) for p in _episode_paths(load_dir, args.expert_data_num)]

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

    # Flush buffered episode metadata explicitly (metadata is buffered and would
    # otherwise only flush in __del__, which is fragile).
    dataset.meta._close_writer()
    print(f"[convert] done: {len(plan)} episodes / {total_frames} frames / {n_tasks} tasks -> {out_root}")


if __name__ == "__main__":
    main()
