import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

XPOLICYLAB_ROOT = Path(__file__).resolve().parents[5]
if str(XPOLICYLAB_ROOT) not in sys.path:
    sys.path.insert(0, str(XPOLICYLAB_ROOT))

from XPolicyLab.utils.process_data import decode_image_bit

# HoloBrain camera name -> XPolicyLab HDF5 vision key
CAMERA_MAP = {
    "front_camera": "cam_head",
    "left_camera": "cam_left_wrist",
    "right_camera": "cam_right_wrist",
}


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_yaml_robot_name(path):
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["config"]["robot"]


def _robot_info(project_root, env_cfg_type):
    env_cfg_dir = project_root / "env_cfg"
    robot_infos = _read_json(env_cfg_dir / "robot" / "_robot_info.json")
    env_cfg_path = env_cfg_dir / f"{env_cfg_type}.yml"

    if env_cfg_path.exists():
        robot_name = _read_yaml_robot_name(env_cfg_path)
    elif env_cfg_type in robot_infos:
        robot_name = env_cfg_type
    else:
        robot_cfg_path = env_cfg_dir / "robot" / f"{env_cfg_type}.yml"
        raise FileNotFoundError(
            f"Could not resolve env_cfg_type '{env_cfg_type}'. Expected either "
            f"{env_cfg_path}, a key in {env_cfg_dir / 'robot' / '_robot_info.json'}, "
            f"or robot config {robot_cfg_path}."
        )

    if robot_name not in robot_infos:
        raise KeyError(f"Robot '{robot_name}' is missing from {env_cfg_dir / 'robot' / '_robot_info.json'}")
    return robot_infos[robot_name]


def _state_keys(action_type, arm_count):
    if arm_count == 1:
        arm_keys = ["arm_joint_states"] if action_type == "joint" else ["ee_poses"]
        return arm_keys, ["ee_joint_states"]
    if action_type == "joint":
        return ["left_arm_joint_states", "right_arm_joint_states"], [
            "left_ee_joint_states",
            "right_ee_joint_states",
        ]
    return ["left_ee_poses", "right_ee_poses"], [
        "left_ee_joint_states",
        "right_ee_joint_states",
    ]


def _read_state_array(group, key, expected_dim):
    if key not in group:
        raise KeyError(f"Missing key {group.name}/{key}")
    arr = np.asarray(group[key], dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.shape[-1] == expected_dim:
        return arr
    if arr.shape[-1] > expected_dim:
        return arr[..., :expected_dim]
    pad_width = [(0, 0)] * arr.ndim
    pad_width[-1] = (0, expected_dim - arr.shape[-1])
    return np.pad(arr, pad_width, mode="constant")


def _pack_state(group, action_type, robot_info):
    arm_keys, ee_keys = _state_keys(action_type, len(robot_info["arm_dim"]))
    parts = []
    for arm_key, ee_key, arm_dim, ee_dim in zip(
        arm_keys, ee_keys, robot_info["arm_dim"], robot_info["ee_dim"]
    ):
        parts.append(_read_state_array(group, arm_key, arm_dim))
        parts.append(_read_state_array(group, ee_key, ee_dim))
    return np.concatenate(parts, axis=-1).astype(np.float32)


def _read_instruction(src, fallback):
    for key in ("instruction", "instructions"):
        if key not in src:
            continue
        value = src[key][()]
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if isinstance(value, np.ndarray) and value.shape == ():
            item = value.item()
            if isinstance(item, bytes):
                return item.decode("utf-8", errors="replace")
            return str(item)
        return str(value)
    return fallback


def _decode_frame(frame_bytes):
    raw = bytes(frame_bytes).rstrip(b"\0")
    img = decode_image_bit(raw)
    if img is None:
        raise ValueError("Failed to decode image bytes")
    img = cv2.resize(img, (320, 240), interpolation=cv2.INTER_AREA)
    if img.shape != (240, 320, 3):
        raise ValueError(f"Expected image shape (240, 320, 3), got {img.shape}")
    return img.astype(np.uint8)


def _encode_png(img_bgr):
    ok, encoded = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError("Failed to encode frame as PNG")
    return encoded.astype(np.uint8)


def _resize_depth(depth):
    depth = np.asarray(depth)
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth = cv2.resize(depth, (320, 240), interpolation=cv2.INTER_NEAREST)
    return depth.astype(np.uint16)


def _scale_intrinsic(intrinsic, source_shape):
    src_h, src_w = source_shape[:2]
    out = np.asarray(intrinsic, dtype=np.float32).copy()
    out[0, :] *= 320.0 / float(src_w)
    out[1, :] *= 240.0 / float(src_h)
    out[2, 2] = 1.0
    return out


def _to_holobrain_world2cam(extrinsic):
    convention = os.environ.get(
        "XPOLICY_HOLOBRAIN_EXTRINSIC_CONVENTION", "cam2world_opengl"
    ).lower()
    if convention in {"world2cam", "world_to_cam", "w2c"}:
        return extrinsic
    if convention in {"cam2world", "camera_to_world", "c2w"}:
        return np.linalg.inv(extrinsic)
    if convention in {"cam2world_opengl", "c2w_opengl", "sapien"}:
        # XPolicyLab/SAPIEN store camera pose as cam->world in OpenGL
        # convention (X right, Y up, Z backward). HoloBrain projection uses
        # OpenCV (X right, Y down, Z forward), so we invert and flip the
        # camera-frame Y and Z axes.
        flip = np.diag([1.0, -1.0, -1.0, 1.0]).astype(extrinsic.dtype)
        return flip @ np.linalg.inv(extrinsic)
    raise ValueError(
        "XPOLICY_HOLOBRAIN_EXTRINSIC_CONVENTION must be one of "
        "cam2world/world2cam/cam2world_opengl."
    )


def _episode_number(path):
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return int(digits) if digits else 0


def _write_string_dataset(group, name, value):
    dtype = h5py.string_dtype(encoding="utf-8")
    group.create_dataset(name, data=value, dtype=dtype)


def _convert_episode(src_path, raw_out_path, npz_out_path, robot_info, action_type, instruction):
    with h5py.File(src_path, "r") as src:
        state = _pack_state(src["state"], action_type, robot_info)
        action_src = src["action"] if "action" in src else src["state"]
        action = _pack_state(action_src, action_type, robot_info)
        horizon = min(len(state), len(action))
        state = state[:horizon]
        action = action[:horizon]

        episode_instruction = _read_instruction(src, instruction)
        rgb_arrays = {}
        depth_arrays = {}
        intrinsic = {}
        extrinsic = {}
        encoded_rgbs = {}
        encoded_depths = {}

        for holobrain_cam, xpolicy_cam in CAMERA_MAP.items():
            cam = src["vision"][xpolicy_cam]
            source_shape = tuple(np.asarray(cam["shape"]).tolist()) if "shape" in cam else (480, 640, 3)
            frames = []
            frame_encoded = []
            for frame in cam["colors"][:horizon]:
                img = _decode_frame(frame)
                frames.append(img)
                frame_encoded.append(_encode_png(img))

            depths = []
            depth_encoded = []
            depth_key = "approximate_depths" if "approximate_depths" in cam else "depths"
            if depth_key in cam:
                for depth in cam[depth_key][:horizon]:
                    depth_u16 = _resize_depth(depth)
                    ok, encoded = cv2.imencode(".png", depth_u16)
                    if not ok:
                        raise RuntimeError("Failed to encode depth frame as PNG")
                    depths.append(depth_u16)
                    depth_encoded.append(encoded.astype(np.uint8))
            else:
                zero_depth = np.zeros((240, 320), dtype=np.uint16)
                for _ in range(horizon):
                    ok, encoded = cv2.imencode(".png", zero_depth)
                    if not ok:
                        raise RuntimeError("Failed to encode zero depth frame as PNG")
                    depths.append(zero_depth)
                    depth_encoded.append(encoded.astype(np.uint8))

            rgb_arrays[holobrain_cam] = np.stack(frames, axis=0)
            depth_arrays[holobrain_cam] = np.stack(depths, axis=0)
            intrinsic[holobrain_cam] = _scale_intrinsic(cam["intrinsic_matrix"][()], source_shape)
            ext = np.asarray(cam["extrinsic_matrix"], dtype=np.float32)
            extrinsic[holobrain_cam] = np.stack(
                [_to_holobrain_world2cam(x) for x in ext[:horizon]], axis=0
            ).astype(np.float32)
            encoded_rgbs[holobrain_cam] = frame_encoded
            encoded_depths[holobrain_cam] = depth_encoded

    # Raw layout accepted by robo_orchard_lab.dataset.robotwin.robotwin_packer.
    with h5py.File(raw_out_path, "w") as dst:
        dst.create_dataset("endpose", data=np.zeros((horizon, 2, 8), dtype=np.float32))
        joint_action = dst.create_group("joint_action")
        joint_action.create_dataset("vector", data=action, dtype=np.float32)
        obs = dst.create_group("observation")
        for cam_name in CAMERA_MAP:
            cam_group = obs.create_group(cam_name)
            cam_group.create_dataset("rgb", data=np.asarray(encoded_rgbs[cam_name], dtype=object), dtype=h5py.vlen_dtype(np.dtype("uint8")))
            cam_group.create_dataset("depth", data=np.stack(depth_arrays[cam_name], axis=0), dtype=np.uint16)
            cam_group.create_dataset("intrinsic_cv", data=np.repeat(intrinsic[cam_name][None, :, :], horizon, axis=0))
            cam_group.create_dataset("extrinsic_cv", data=extrinsic[cam_name])
        _write_string_dataset(dst, "instruction", episode_instruction)

    np.savez_compressed(
        npz_out_path,
        state=state,
        action=action,
        instruction=np.asarray([episode_instruction]),
        **{f"rgb_{name}": value for name, value in rgb_arrays.items()},
        **{f"depth_{name}": value for name, value in depth_arrays.items()},
        **{f"intrinsic_{name}": value for name, value in intrinsic.items()},
        **{f"extrinsic_{name}": value for name, value in extrinsic.items()},
    )

    return {
        "source": str(src_path),
        "raw_hdf5": str(raw_out_path),
        "npz": str(npz_out_path),
        "frames": int(horizon),
        "instruction": episode_instruction,
        "state_dim": int(state.shape[-1]),
        "action_dim": int(action.shape[-1]),
        "rgb_shape": [240, 320, 3],
        "cameras": list(CAMERA_MAP.keys()),
    }


def _write_stats(out_dir, episode_npz_paths):
    states = []
    actions = []
    for path in episode_npz_paths:
        data = np.load(path)
        states.append(data["state"])
        actions.append(data["action"])
    state = np.concatenate(states, axis=0)
    action = np.concatenate(actions, axis=0)
    stats = {
        "state_mean": state.mean(axis=0).tolist(),
        "state_std": state.std(axis=0).tolist(),
        "state_min": state.min(axis=0).tolist(),
        "state_max": state.max(axis=0).tolist(),
        "action_mean": action.mean(axis=0).tolist(),
        "action_std": action.std(axis=0).tolist(),
        "action_min": action.min(axis=0).tolist(),
        "action_max": action.max(axis=0).tolist(),
    }
    with open(out_dir / "dataset_statistics.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Convert XPolicyLab HDF5 data for HoloBrain smoke training/data checks.")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--env-cfg-type", required=True)
    parser.add_argument("--expert-data-num", type=int, required=True)
    parser.add_argument("--action-type", choices=("joint", "ee"), required=True)
    parser.add_argument("--config-name", default="demo_clean")
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.action_type != "joint":
        raise ValueError("HoloBrain XPolicyLab conversion currently supports action_type=joint only.")

    project_root = Path(args.project_root).resolve()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_paths = sorted((input_dir / "data").glob("*.hdf5"))
    if args.expert_data_num > 0:
        episode_paths = episode_paths[: args.expert_data_num]
    if not episode_paths:
        raise FileNotFoundError(f"No HDF5 episodes found under {input_dir / 'data'}")

    robot_info = _robot_info(project_root, args.env_cfg_type)
    fallback_instruction = args.instruction or args.task_name.replace("_", " ")

    raw_root = output_dir / "robotwin_packer_input"
    raw_data_dir = raw_root / args.task_name / args.config_name / "data"
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    npz_dir = output_dir / "xpolicylab_npz"
    npz_dir.mkdir(parents=True, exist_ok=True)

    seed_values = []
    manifest = {
        "source_dataset": str(input_dir),
        "task_name": args.task_name,
        "env_cfg_type": args.env_cfg_type,
        "action_type": args.action_type,
        "config_name": args.config_name,
        "robot_info": robot_info,
        "image_shape": [240, 320, 3],
        "image_color": "BGR",
        "robotwin_packer_command": (
            "python3 -m robo_orchard_lab.dataset.robotwin.robotwin_packer "
            f"--input_path {raw_root} --output_path {output_dir / 'lmdb'} "
            f"--task_names {args.task_name} --config_name {args.config_name}"
        ),
        "episodes": [],
    }

    iterator = enumerate(episode_paths)
    if tqdm is not None:
        iterator = tqdm(
            iterator,
            total=len(episode_paths),
            desc="Converting episodes",
            unit="episode",
        )
    for out_idx, src_path in iterator:
        ep_num = _episode_number(src_path)
        seed_values.append(str(ep_num))
        raw_out = raw_data_dir / f"episode{out_idx}.hdf5"
        npz_out = npz_dir / f"episode_{out_idx:07d}.npz"
        manifest["episodes"].append(
            _convert_episode(
                src_path,
                raw_out,
                npz_out,
                robot_info,
                args.action_type,
                fallback_instruction,
            )
        )

    seed_file = raw_root / args.task_name / args.config_name / "seed.txt"
    seed_file.write_text(" ".join(seed_values), encoding="utf-8")

    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    _write_stats(output_dir, sorted(npz_dir.glob("*.npz")))

    print(f"Converted {len(episode_paths)} episodes")
    print(f"Output: {output_dir}")
    print(f"Manifest: {output_dir / 'manifest.json'}")
    print("Next optional step:")
    print(manifest["robotwin_packer_command"])


if __name__ == "__main__":
    main()
