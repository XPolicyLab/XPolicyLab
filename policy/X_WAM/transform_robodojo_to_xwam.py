#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Convert a RoboDojo (HDF5) dataset directly into the xwam format.

This merges three previously-sequential scripts into a single per-episode pass:
  1) HDF5 -> mibot EE format
  2) end-effector coordinate-frame local axis redefinition
  3) mibot EE -> xwam format
Pipeline per episode: read HDF5 -> decode and write H264 videos ->
pack the 6 EE fields -> apply the frame transform -> write the xwam JSON.
No intermediate mibot / mibot_ee datasets are produced.

Notes
-----
- xwam keeps only the 6 EE fields (left/right ee_pos / ee_rotm / gripper_pos),
  so direct / quaternion / arm_joint intermediates are never computed and
  arm_joint is not even read from the HDF5.
- The frame redefinition affects the rotation matrix only: R_new = R_cur @ P;
  position and gripper are unchanged.
- actions = proprios without the first frame ([1:]): actions[t] == proprios[t+1],
  length T-1.
- instructions is taken directly from the raw HDF5 instruction text.
- Videos are written under the xwam camera naming and relative paths
  (head/left/right_camera) so observations' rgb_path matches the files on disk.

Usage
-----
    source /path/to/wan_env/bin/activate
    python XPolicyLab/policy/X_WAM/transform_robodojo_to_xwam.py \
        --input-dir  data/RoboDojo \
        --output-dir xwam_datasets/RoboDojo \
        --env-cfg-type arx_x5 \
        --workers 16 [--limit 3] [--clean]
"""

import argparse
import json
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import h5py
import imageio
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHUNK_SIZE = 1000           # at most 1000 episodes per chunk (mibot/xwam convention)
CRF = "18"                  # H264 quality (lower is sharper; 18 is visually lossless)

# RoboDojo camera name -> (xwam observations field, video subdir, type).
# Only 3 views are kept; cam_third_view is intentionally not converted.
CAMERA_MAP = [
    ("cam_head", "head_camera", "static"),
    ("cam_left_wrist", "left_camera", "dynamic"),
    ("cam_right_wrist", "right_camera", "dynamic"),
]

# End-effector local axis redefinition matrix P = Rx(+90 deg) @ Rz(+90 deg), det=+1.
P = np.array([
    [0.0, -1.0,  0.0],
    [0.0,  0.0, -1.0],
    [1.0,  0.0,  0.0],
])


# ---------------------------------------------------------------------------
# Geometry / state packing
# ---------------------------------------------------------------------------

def ee_pose_to_pos_rotm(ee_poses: np.ndarray):
    """ee_poses (T,7)[x,y,z,qw,qx,qy,qz] -> (pos (T,3), rotm (T,3,3)).

    The rotation matrix already has the frame redefinition applied: R_new = R_cur @ P.
    """
    ee_poses = np.asarray(ee_poses, dtype=np.float64)
    pos = ee_poses[:, 0:3]
    quat_wxyz = ee_poses[:, 3:7]
    R_cur = R.from_quat(quat_wxyz[:, [1, 2, 3, 0]]).as_matrix()   # scipy uses xyzw
    R_new = R_cur @ P
    return pos, R_new


def build_arm_ee(ee_poses: np.ndarray, gripper: np.ndarray):
    """Build the 3 xwam EE fields for one arm: ee_pos (T,3) / ee_rotm (T,9) / gripper_pos (T,1)."""
    pos, R_new = ee_pose_to_pos_rotm(ee_poses)
    rotm_flat = R_new.reshape(R_new.shape[0], 9)
    grip = np.asarray(gripper, dtype=np.float64).reshape(-1, 1)
    return pos, rotm_flat, grip


# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------

def write_video(frames_bgr_list, out_path: Path, fps: float):
    """Write a list of BGR frames to an H264 mp4 via imageio."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(out_path),
        fps=fps,
        codec="libx264",
        format="FFMPEG",
        pixelformat="yuv420p",
        macro_block_size=None,        # do not force resolution to a multiple of 16
        output_params=["-crf", CRF],
    )
    try:
        for bgr in frames_bgr_list:
            writer.append_data(bgr)
    finally:
        writer.close()


def decode_camera_frames(colors_dataset) -> list:
    """Decode a RoboDojo camera's colors (T,) JPEG byte-string dataset into a list of BGR frames."""
    frames = []
    for i in range(colors_dataset.shape[0]):
        buf = colors_dataset[i]
        if isinstance(buf, (bytes, bytearray, np.bytes_)):
            buf = bytes(buf).rstrip(b"\x00")
        img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"JPEG decode failed at frame {i}")
        frames.append(img)
    return frames


# ---------------------------------------------------------------------------
# Single-episode conversion
# ---------------------------------------------------------------------------

def convert_one(job: dict) -> dict:
    """Read HDF5 -> write videos -> pack EE -> apply frame transform -> write xwam JSON."""
    src_path = job["src_path"]
    task_name = job["task_name"]
    src_index = job["src_index"]
    global_index = job["global_index"]
    out_dir = Path(job["out_dir"])

    try:
        chunk_id = global_index // CHUNK_SIZE
        chunk_name = f"chunk-{chunk_id:04d}"
        ep_name = f"episode_{global_index:07d}"
        json_path = out_dir / "data" / chunk_name / f"{ep_name}.json"

        with h5py.File(src_path, "r") as f:
            instr_text = f["instruction"][()]
            if isinstance(instr_text, (bytes, bytearray)):
                instr_text = instr_text.decode("utf-8")
            fps = float(int(f["additional_info/frequency"][()]))

            st = f["state"]
            T = st["left_arm_joint_states"].shape[0]

            left_pos, left_rotm, left_grip = build_arm_ee(
                st["left_ee_poses"][:], st["left_ee_joint_states"][:])
            right_pos, right_rotm, right_grip = build_arm_ee(
                st["right_ee_poses"][:], st["right_ee_joint_states"][:])

            # Videos: decode and write H264 mp4 per camera (xwam naming + relative path).
            observations = {}
            for cam_name, obs_field, cam_type in CAMERA_MAP:
                if cam_name not in f["vision"]:
                    observations[obs_field] = None
                    continue
                frames = decode_camera_frames(f[f"vision/{cam_name}/colors"])
                rel = f"video/{obs_field}/{chunk_name}/{ep_name}.mp4"
                write_video(frames, out_dir / rel, fps)
                observations[obs_field] = {
                    "type": cam_type,
                    "rgb_path": rel,
                    "depth_path": rel,   # depth points to the rgb path as required
                    "start": 0,
                    "end": T,
                    "fps": fps,
                }

        # proprios (length T, 6 EE fields)
        proprios = {
            "left_ee_pos": left_pos,
            "left_ee_rotm": left_rotm,
            "left_gripper_pos": left_grip,
            "right_ee_pos": right_pos,
            "right_ee_rotm": right_rotm,
            "right_gripper_pos": right_grip,
        }
        # actions = proprios without the first frame (Ta=T-1, actions[t]==proprios[t+1])
        actions = {k: v[1:] for k, v in proprios.items()}

        episode = {
            "num_frames": int(T),
            "instructions": [instr_text.strip()] if instr_text.strip() else [],
            "observations": observations,
            "proprios": {k: v.tolist() for k, v in proprios.items()},
            "actions": {k: v.tolist() for k, v in actions.items()},
        }

        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as fp:
            json.dump(episode, fp, ensure_ascii=False, indent=4)

        return {"ok": True, "global_index": global_index,
                "task": task_name, "src_index": src_index, "T": int(T)}

    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "global_index": global_index,
                "task": task_name, "src_index": src_index,
                "src_path": str(src_path),
                "error": f"{exc}\n{traceback.format_exc()}"}


# ---------------------------------------------------------------------------
# Job collection / main
# ---------------------------------------------------------------------------

def collect_jobs(input_dir: Path, out_dir: Path, env_cfg_type: str):
    """Scan all tasks' hdf5 files and assign globally-continuous indices, sorted by (task, in-task index)."""
    tasks = sorted(p.name for p in input_dir.iterdir() if p.is_dir())
    jobs = []
    global_index = 0
    for task_name in tasks:
        data_dir = input_dir / task_name / env_cfg_type / "data"
        if not data_dir.is_dir():
            continue
        for src_index, ep_path in enumerate(sorted(data_dir.glob("episode_*.hdf5"))):
            jobs.append({
                "src_path": str(ep_path),
                "task_name": task_name,
                "src_index": src_index,
                "global_index": global_index,
                "out_dir": str(out_dir),
            })
            global_index += 1
    return jobs, tasks


def main():
    parser = argparse.ArgumentParser(description="Convert RoboDojo HDF5 -> xwam format (one-shot).")
    parser.add_argument("--input-dir", type=str, default="data/RoboDojo",
                        help="RoboDojo root directory (contains per-task subdirs)")
    parser.add_argument("--output-dir", type=str,
                        default="xwam_datasets/RoboDojo",
                        help="xwam dataset output directory")
    parser.add_argument("--workers", type=int, default=16, help="number of parallel processes")
    parser.add_argument("--limit", type=int, default=0,
                        help="convert only the first N episodes (0 = all), for sampling")
    parser.add_argument("--env-cfg-type", type=str, default="arx_x5",
                        help="robot/environment config subdir under each task")
    parser.add_argument("--clean", action="store_true", help="clear the output directory before converting")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    out_dir = Path(args.output_dir).resolve()

    if not input_dir.is_dir():
        print(f"[ERROR] input directory does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)

    if args.clean and out_dir.exists():
        import shutil
        print(f"[INFO] clearing output directory: {out_dir}")
        shutil.rmtree(out_dir)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)

    jobs, tasks = collect_jobs(input_dir, out_dir, args.env_cfg_type)
    if args.limit > 0:
        jobs = jobs[:args.limit]

    print(f"[INFO] tasks: {len(tasks)} | episodes to convert: {len(jobs)} | "
          f"env_cfg_type: {args.env_cfg_type} | workers: {args.workers} | output: {out_dir}")

    failures = []
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(convert_one, job) for job in jobs]
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="Converting", unit="ep", smoothing=0.05):
            res = fut.result()
            results.append(res)
            if not res["ok"]:
                failures.append(res)

    n_ok = sum(1 for r in results if r["ok"])
    print(f"\n[DONE] succeeded {n_ok}/{len(jobs)} -> {out_dir}")
    if failures:
        print(f"[FAIL] {len(failures)} failed:")
        for fl in failures[:20]:
            print(f"  - {fl['task']}#{fl['src_index']} ({fl['src_path']}): "
                  f"{fl['error'].splitlines()[0]}")
        with open(out_dir / "conversion_failures.json", "w") as fp:
            json.dump(failures, fp, ensure_ascii=False, indent=2)
        sys.exit(2)


if __name__ == "__main__":
    main()
