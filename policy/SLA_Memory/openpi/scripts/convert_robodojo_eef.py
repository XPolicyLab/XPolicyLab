import argparse
from pathlib import Path

import av
import h5py
from lerobot.common.datasets import image_writer
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from tqdm import tqdm
from XPolicyLab.utils.data_loader import load

# Converts official RoboDojo HDF5 episodes (<bench data dir>/<task>/<robot>/data, simulation or real robot) into one
# LeRobot v2.1 dataset with end-effector state and actions. Based on XPolicyLab's scripts/transform_lerobot_v21_format.py,
# with these differences:
# 1. Written for the lerobot 0.1.0 vendored in packages/lerobot (the official script targets 0.3.x).
# 2. state and action are the 16-dim end-effector poses plus grippers, from the HDF5 state and action groups
#    (the official script uses joint angles).
# 3. fps comes from each episode's additional_info/frequency, falling back to the frame rate of that episode's official
#    preview video; all episodes must agree.
# Videos use the lerobot 0.1.0 encode_video_frames defaults (libsvtav1, g=2, crf=30), like the official script; images
# stay 480x640. lerobot writes every frame to a temporary PNG before encoding; this script lowers the PNG compression
# level from PIL's default 6 to 1 (see _write_image_png_level1).
# Needs the parent directory of XPolicyLab on PYTHONPATH.
# Usage: convert_robodojo_eef.py <bench data dir> <robot> <output dataset dir> <task> [<task> ...] [--max-episodes N]

# LeRobot camera name -> camera name under the HDF5 vision group, same as the official CAMERA_CANDIDATES.
CAMERA_CANDIDATES = {"cam_high": "cam_head", "cam_left_wrist": "cam_left_wrist", "cam_right_wrist": "cam_right_wrist"}
POSE_PARTS = ("left_ee_poses", "left_ee_joint_states", "right_ee_poses", "right_ee_joint_states")
# Per arm: xyz position, quaternion with w first, gripper.
MOTORS = [f"{arm}_{name}" for arm in ("l", "r") for name in ("x", "y", "z", "w", "wx", "wy", "wz", "g")]
IMAGE_HEIGHT, IMAGE_WIDTH = 480, 640


def _write_image_png_level1(image, fpath):
    # lerobot's image_writer.write_image (including printing on failure) plus compress_level=1. PNG is lossless, so
    # the pixels and the encoded video are unchanged, and saving one image takes about 22 ms of CPU instead of 50 ms.
    try:
        image_writer.image_array_to_pil_image(image).save(fpath, compress_level=1)
    except Exception as e:
        print(f"Error writing image {fpath}: {e}")


# The image writer threads look up write_image in the image_writer module on every call, so replacing it takes effect.
image_writer.write_image = _write_image_png_level1


def _episode_fps(path):
    with h5py.File(path, "r") as f:
        if "additional_info/frequency" in f:
            return float(f["additional_info/frequency"][()])
    preview = path.parent.parent / "preview_video" / path.name.replace(".hdf5", "_cam_head.mp4")
    with av.open(str(preview)) as container:
        return float(container.streams.video[0].average_rate)


def create_empty_dataset(root, repo_id, robot_type, fps):
    features = {
        "observation.state": {"dtype": "float32", "shape": (len(MOTORS),), "names": [MOTORS]},
        "action": {"dtype": "float32", "shape": (len(MOTORS),), "names": [MOTORS]},
    }
    for cam in CAMERA_CANDIDATES:
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": (3, IMAGE_HEIGHT, IMAGE_WIDTH),
            "names": ["channels", "height", "width"],
        }
    # LeRobotDatasetMetadata.create raises if the directory exists (mkdir exist_ok=False), so nothing is overwritten.
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=root,
        robot_type=robot_type,
        features=features,
        use_videos=True,
        tolerance_s=0.0001,
        image_writer_processes=0,
        image_writer_threads=4,
    )


def _pose16(data, group, num_frames):
    parts = [np.asarray(data[group][k], dtype=np.float32).reshape(num_frames, -1) for k in POSE_PARTS]
    out = np.concatenate(parts, axis=1)
    assert out.shape == (num_frames, len(MOTORS)), (group, out.shape)
    return out


def release_dataset_memory(dataset):
    # lerobot appends every saved episode to hf_dataset, so memory keeps growing; clear it after each episode,
    # like the official script.
    dataset.hf_dataset = dataset.create_hf_dataset()


def convert_one(input_path, dataset):
    data = load(str(input_path), data_type="RoboDojo", data_version="v1.0")
    num_frames = data["state"]["left_ee_poses"].shape[0]
    state = _pose16(data, "state", num_frames)
    action = _pose16(data, "action", num_frames)
    # Every episode has exactly one instruction (the official _choose_instruction just takes it in that case).
    instruction = data["instruction"]
    assert isinstance(instruction, str) and instruction, (input_path, instruction)

    # load already decodes the images to RGB with decode_image_bit, for both storage formats.
    images = {cam: data["vision"][src]["colors"] for cam, src in CAMERA_CANDIDATES.items()}
    for cam, frames in images.items():
        assert frames.shape == (num_frames, IMAGE_HEIGHT, IMAGE_WIDTH, 3) and frames.dtype == np.uint8, (cam, frames.shape)

    for i in range(num_frames):
        frame = {"observation.state": state[i], "action": action[i], "task": instruction}
        for cam, frames in images.items():
            frame[f"observation.images.{cam}"] = frames[i]
        dataset.add_frame(frame)
    dataset.save_episode()
    release_dataset_memory(dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src_root", help="Benchmark data directory, e.g. data/RoboDojo or data/RoboDojo_real")
    parser.add_argument("robot")
    parser.add_argument("out_dir", help="Output dataset directory; it must not exist yet")
    parser.add_argument("tasks", nargs="+", help="Tasks written into this one dataset")
    parser.add_argument("--max-episodes", type=int, default=None, help="Convert at most this many episodes per task")
    args = parser.parse_args()

    input_files = []
    for task in args.tasks:
        task_files = sorted((Path(args.src_root) / task / args.robot / "data").glob("episode_*.hdf5"))
        assert task_files, f"No HDF5 files found in {args.src_root}/{task}/{args.robot}/data"
        input_files += task_files[: args.max_episodes]
    fps_set = {_episode_fps(p) for p in input_files}
    assert len(fps_set) == 1, f"Episodes disagree on fps: {fps_set}"
    fps = fps_set.pop()
    assert fps == int(fps), f"fps is not an integer: {fps}"
    fps = int(fps)

    out_dir = Path(args.out_dir)
    repo_id = out_dir.name
    print(f"{repo_id}: {len(args.tasks)} tasks, {len(input_files)} episodes, fps {fps}", flush=True)
    dataset = create_empty_dataset(out_dir, repo_id, args.robot, fps)
    for input_path in tqdm(input_files, desc=repo_id, mininterval=60):
        convert_one(input_path, dataset)
    dataset.stop_image_writer()
    print(f"Finished {repo_id}: {dataset.meta.total_episodes} episodes, {dataset.meta.total_frames} frames", flush=True)


if __name__ == "__main__":
    main()
