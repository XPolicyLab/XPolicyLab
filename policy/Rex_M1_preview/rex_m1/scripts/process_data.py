#!/usr/bin/env python
"""Convert RoboDojo official HDF5 episodes into the xr1 JSON training format.

Handles both releases:
    RoboDojo       simulation, one robot (arx_x5)
    RoboDojo_real  real robots: piper_x, piper, arx_x5

Only tasks that carry the requested `--env-cfg-type` robot are converted. In
RoboDojo_real each task belongs to exactly one robot (6 tasks each), so a
piper_x run leaves the piper and arx_x5 tasks untouched.

Per-release differences handled below:
- EE reframe matrix: simulation shares one; the real robots each need their own
  (a mismatch trains rotations the deployed adapter will not reproduce).
- EE action: RoboDojo_real records `action/*_ee_poses` and it is used directly;
  simulation has none, so the pose action is emulated by a next-state shift.
- fps: read from `additional_info/frequency`, falling back to 30 because
  several RoboDojo_real piper_x tasks omit that group.

Source layout (per task, under the bench data root):
    <SRC>/<task_name>/<env_cfg_type>/data/episode_XXXXXXX.hdf5

Output layout (under the destination root):
    <DST>/data/chunk-XXXX/episode_XXXXXXX.json
    <DST>/videos/chunk-XXXX/episode_XXXXXXX.{ego,wrist_left,wrist_right}.mp4

No path is hard-coded in this script. Everything is anchored on this file's own
location: SCRIPT_DIR -> <xr1>/scripts, XR1_DIR -> <xr1>. The data root is
discovered by walking upward from XR1_DIR looking for `data/<bench_name>`.

The video paths recorded inside each JSON, and the `paths` entry in the
generated config, are written as absolute paths so training does not depend on
the directory it is launched from. Moving the dataset therefore requires
rewriting those JSONs.

Camera mapping, matching the inference prompt built in `model.py`:
    ego         <- vision/cam_head        ("# Ego View")
    wrist_left  <- vision/cam_left_wrist  ("# Left-Wrist View")
    wrist_right <- vision/cam_right_wrist ("# Right-Wrist View")
`vision/cam_third_view` is ignored.

Written fields are exactly the ones `mibot/data/datasets/json_dataset.py`
reads, no more: num_frames, time, trajectory_type, instruction.general,
observations.{ego,wrist_left,wrist_right}[].{path,start,crop_bbox},
proprios.{left,right}_{ee_pos,ee_rotm,gripper_pos,arm_joint},
proprios.waist_pos, actions.{left,right}_{ee_pos,ee_rotm,gripper_pos},
actions.waist_pos and actions.base_vel.

ARX X5 has neither a torso joint nor a mobile base, so waist_pos and base_vel
are emitted as all-zero columns: packed action dimensions 16-19 stay at a
constant delta of 0 instead of crashing the loader on a null.
`actions.{left,right}_arm_joint` is omitted because the loader packs actions
in end-effector space only.

Action definition:
- gripper: taken straight from the source `action/*_ee_joint_states`.
- ee_pos / ee_rotm: from `action/*_ee_poses` when recorded (RoboDojo_real),
  otherwise emulated from the proprio by a next-state shift,
  action[t] = proprio[t+1], with the last frame duplicated (RoboDojo sim).

EE pose reframe: local axes are renamed to the MiBot convention through
R_new = R_cur @ P, using the same per-bench/per-robot matrices the inference
adapter applies (`_EEF_REFRAME_P_ROBODOJO` and `_EEF_REFRAME_P_ROBODOJO_REAL`
in model.py), so training and deployment agree on the end-effector frame.

Normalization statistics are recomputed from the converted episodes and
emitted as a ready-to-use hydra data config, because the shipped
`configs/data/load_washer.yaml` statistics belong to a different robot. The
rules follow `scripts/compute_robodojo_stats.py`:

- gripper, both sides: hard-coded onto [-1, 1] from the measured [0, 1] travel,
  never estimated from data. The state side (quantile) gets q01 = 0, q99 = 1.0.
  The action side (gaussian) gets mean = 0, std = 1.0 rather than the reference
  script's 0.5 / 0.5, because this loader packs a gripper delta while the
  reference chain normalizes the absolute value; see the GRIPPER_ACTION_*
  constants for the derivation.
- EE action (ee_pos / ee_aa): gaussian estimated per (step, dim) from moving
  frames only (|x| >= 1e-4), so idle frames do not spike the distribution. A
  column with fewer than 10 moving frames falls back to mean 0 / std 1.
- remaining state (arm joints): 1st/99th percentiles.

Only whole frames are sampled (a full action_length horizon), so tail-padded
frames stay out of the statistics. Everything runs in float64 and is written
as .16e so the YAML round trip is lossless.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm

# --- Path anchors -----------------------------------------------------------
# Everything below is derived from this file's own location, never from cwd.
SCRIPT_DIR = Path(__file__).resolve().parent          # <rex_m1>/scripts
XR1_DIR = SCRIPT_DIR.parent                           # <xr1>
CONFIG_DATA_DIR = XR1_DIR / "configs" / "data"

# Import the loader's own packing helpers rather than reimplementing them, so
# the statistics are computed with exactly the math training will apply. This
# makes the script runnable from any cwd without a preset PYTHONPATH.
if str(XR1_DIR) not in sys.path:
    sys.path.insert(0, str(XR1_DIR))

# Stored image bits decode only through XPolicyLab's decode_image_bit; the
# checkout root (made importable by its XPolicyLab.py shim) is four levels up.
# Modified by the Rex-M1-preview authors (2026): path depth only, after the vendored
# tree was flattened one level (upstream had this file one directory deeper).
XPOLICYLAB_ROOT = SCRIPT_DIR.parents[3]
if str(XPOLICYLAB_ROOT) not in sys.path:
    sys.path.insert(0, str(XPOLICYLAB_ROOT))

from XPolicyLab.utils.process_data import decode_image_bit  # noqa: E402

from mibot.utils.io import (  # noqa: E402  (import needs the sys.path tweak above)
    ACTION_DIM,
    ACTION_PARTS,
    STATE_DIM,
    rotm2aa_batch,
)

# Benches this script understands. RoboDojo is simulation, RoboDojo_real is the
# real-robot release; they differ in the EE reframe, whether an end-effector
# action is recorded, and the fallback fps.
SIM_BENCH = "RoboDojo"
REAL_BENCH = "RoboDojo_real"
SUPPORTED_BENCHES = (SIM_BENCH, REAL_BENCH)

# Source roots searched for, relative to each ancestor of XR1_DIR, per bench.
SRC_REL_TEMPLATES = ("data/{bench}", "final_data/{bench}")
DST_REL_TEMPLATE = "data/{bench}_xr1"

# fps used when the HDF5 carries no additional_info/frequency. Several
# RoboDojo_real piper_x tasks omit that group entirely.
FALLBACK_FPS = 30.0

# Chunk size for the data/chunk-XXXX chunk directories.
CHUNK_SIZE = 1000

# Slices of the packed 60-dim state vector, mirroring io.compose_state().
STATE_SLICES = {
    "left_arm_joint": slice(0, 7),
    "left_gripper": slice(7, 8),
    "right_arm_joint": slice(8, 15),
    "right_gripper": slice(15, 16),
}

# --- Statistics rules (ported from scripts/compute_robodojo_stats.py) -------
# Frames where |x| < MOTION_EPS count as stationary and are dropped before the
# gaussian action statistics, so idle frames do not spike the distribution.
MOTION_EPS = 1e-4
# A (step, dim) column needs at least this many moving frames to be estimated;
# below that it degrades to the identity transform (mean 0, std 1).
MIN_MOTION_FRAMES = 10
# Floor on the estimated std, guarding against a near-degenerate column.
MIN_STD = 1e-6
# Measured RoboDojo gripper travel is [0, GRIPPER_MAX]; both the action and the
# state side are hard-coded onto [-1, 1] rather than estimated from data.
GRIPPER_MAX = 1.0
# The reference script hard-codes the gripper action to mean = std = 0.5,
# because its transform chain normalizes the ABSOLUTE gripper value
# (`GetAbsAction` in configs/data/sources/robodojo_delta.yaml), for which
# (x - 0.5) / 0.5 maps [0, 1] onto [-1, 1].
#
# The xr1 loader instead packs a gripper DELTA (`_arm_action` in
# json_dataset.py subtracts the current proprio), whose range is
# [-GRIPPER_MAX, +GRIPPER_MAX] and is already centred on 0. Reusing 0.5/0.5
# there would shift the whole channel to a mean of -1, so the same rule
# (hard-code, map the measured travel onto [-1, 1]) becomes mean 0, std
# GRIPPER_MAX for the delta convention.
GRIPPER_ACTION_MEAN = 0.0
GRIPPER_ACTION_STD = GRIPPER_MAX

# Packed action dimensions estimated per (step, dim) from moving frames only.
# The xr1 layout carries no arm-joint action slots, so the reference script's
# action_*_arm_joint keys have no counterpart here.
STEP_GAUSSIAN_ACTION_PARTS = (
    "left_ee_pos", "left_ee_aa", "right_ee_pos", "right_ee_aa",
)
# Packed action dimensions hard-coded to mean = std = GRIPPER_MAX / 2.
GRIPPER_ACTION_PARTS = ("left_gripper", "right_gripper")
# Packed state dimensions hard-coded to q01 = 0, q99 = GRIPPER_MAX.
GRIPPER_STATE_SLICES = ("left_gripper", "right_gripper")
# Packed state dimensions estimated as 1st/99th percentiles.
QUANTILE_STATE_SLICES = ("left_arm_joint", "right_arm_joint")

# Prompt text, byte-for-byte identical to the inference prompt assembled in
# model.py so the fine-tuned model sees the same wording at deploy time.
# The dataloader appends " /no_cot" to the human turn and rewrites the gpt
# turn to "<cot></cot>" on the fly, so they are left out here.
CONVERSATION_TEMPLATE = (
    "The following observations are captured from multiple views.\n"
    "# Ego View\n"
    "<image>\n"
    "# Left-Wrist View\n"
    "<image>\n"
    "# Right-Wrist View\n"
    "<image>\n"
    "Generate robot actions for the task:\n"
)

# JSON camera key -> HDF5 vision group. Order matches the <image> placeholders.
CAM_MAP = (
    ("ego", "cam_head"),
    ("wrist_left", "cam_left_wrist"),
    ("wrist_right", "cam_right_wrist"),
)

# Local-axis reframe of the end-effector pose: R_new = R_cur @ P, det(P) = +1;
# position is left untouched. These mirror _EEF_REFRAME_P_ROBODOJO and
# _EEF_REFRAME_P_ROBODOJO_REAL in model.py, so training and deployment agree on
# the end-effector frame.
#
# Simulation shares one matrix across robots, P = Rx(+90deg) @ Rz(+90deg).
EEF_REFRAME_P_SIM = np.array(
    [
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)

# The real robots each need their own matrix; picking the wrong one silently
# trains on end-effector rotations the deployed adapter will not reproduce.
#   piper_x: P = Rz(-90deg)
#   piper:   P = Rz(180deg)
#   arx_x5:  P = Ry(+90deg) @ Rz(180deg)
EEF_REFRAME_P_REAL = {
    "piper_x": np.array(
        [
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ),
    "piper": np.array(
        [
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ),
    "arx_x5": np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    ),
}


def eef_reframe_p(bench_name: str, env_cfg_type: str) -> np.ndarray:
    """Reframe matrix for one bench/robot pair, mirroring model.py."""
    if bench_name == SIM_BENCH:
        return EEF_REFRAME_P_SIM
    if bench_name == REAL_BENCH:
        if env_cfg_type not in EEF_REFRAME_P_REAL:
            raise SystemExit(
                f"[process_data] unsupported env_cfg_type {env_cfg_type!r} for "
                f"bench {bench_name}. Supported: "
                f"{', '.join(sorted(EEF_REFRAME_P_REAL))}"
            )
        return EEF_REFRAME_P_REAL[env_cfg_type]
    raise SystemExit(
        f"[process_data] unsupported bench_name {bench_name!r}. "
        f"Supported: {', '.join(SUPPORTED_BENCHES)}"
    )


# --- Path resolution --------------------------------------------------------


def resolve_source_root(explicit: str | None, bench_name: str) -> Path:
    """Locate the HDF5 root for one bench without hard-coding an absolute path.

    An explicit --src wins (relative values resolve against cwd, as a user
    typing a path on the command line expects). Otherwise walk upward from
    XR1_DIR looking for `data/<bench>`, which finds the shared dataset root
    regardless of how deeply the policy is vendored.
    """
    if explicit:
        root = Path(explicit).expanduser()
        if not root.is_absolute():
            root = (Path.cwd() / root).resolve()
        if not root.is_dir():
            raise SystemExit(f"[process_data] --src is not a directory: {root}")
        return root

    relatives = [template.format(bench=bench_name) for template in SRC_REL_TEMPLATES]
    for base in (XR1_DIR, *XR1_DIR.parents):
        for rel in relatives:
            candidate = base / rel
            if candidate.is_dir():
                return candidate.resolve()

    raise SystemExit(
        f"[process_data] could not locate the {bench_name} data root.\n"
        f"  searched for [{', '.join(relatives)}] walking up from {XR1_DIR}\n"
        f"  pass --src <path> explicitly."
    )


def resolve_dest_root(explicit: str | None, bench_name: str) -> Path:
    """Destination root; defaults to <xr1>/data/<bench>_xr1."""
    if explicit:
        root = Path(explicit).expanduser()
        if not root.is_absolute():
            root = (Path.cwd() / root).resolve()
        return root
    return (XR1_DIR / DST_REL_TEMPLATE.format(bench=bench_name.lower())).resolve()


def video_path_for_json(abs_video: Path) -> str:
    """Absolute path recorded inside the JSON.

    Absolute paths make the dataset independent of the directory training is
    launched from, at the cost of having to rewrite the JSONs if the dataset
    ever moves.
    """
    return str(abs_video)


# --- Rotation / action math -------------------------------------------------


def quat_wxyz_to_rotm(quats):
    """Vectorized quaternion (wxyz) -> rotation matrix. (T,4) -> (T,3,3)."""
    quats = np.asarray(quats, dtype=np.float64)
    norms = np.linalg.norm(quats, axis=-1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    q = quats / norms
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    rotm = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    rotm[..., 0, 0] = 1 - 2 * (y * y + z * z)
    rotm[..., 0, 1] = 2 * (x * y - z * w)
    rotm[..., 0, 2] = 2 * (x * z + y * w)
    rotm[..., 1, 0] = 2 * (x * y + z * w)
    rotm[..., 1, 1] = 1 - 2 * (x * x + z * z)
    rotm[..., 1, 2] = 2 * (y * z - x * w)
    rotm[..., 2, 0] = 2 * (x * z - y * w)
    rotm[..., 2, 1] = 2 * (y * z + x * w)
    rotm[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return rotm


def build_ee_arrays(endpose, reframe_p):
    """(T,7) xyz+quat_wxyz -> (pos (T,3) list, flattened rotm (T,9) list).

    Position passes through unchanged; the rotation gets the bench/robot local
    axis reframe supplied in `reframe_p`.
    """
    endpose = np.asarray(endpose, dtype=np.float64)
    pos = endpose[:, :3]
    rotm = quat_wxyz_to_rotm(endpose[:, 3:7]) @ reframe_p
    return pos.tolist(), rotm.reshape(-1, 9).tolist()


def next_state_repeat(seq):
    """action[t] = proprio[t+1]; the final frame is duplicated."""
    if len(seq) <= 1:
        return [list(row) for row in seq]
    shifted = [list(row) for row in seq[1:]]
    shifted.append(list(seq[-1]))
    return shifted


# --- Video / image I/O ------------------------------------------------------


def decode_jpeg_seq(colors_dataset):
    """RoboDojo cam colors: (T,) image bit strings -> list of RGB uint8 frames.

    decode_image_bit resolves both stored byte formats to RGB, which is what
    write_video and the official loader both want, so no channel conversion
    belongs anywhere in this path.
    """
    return [decode_image_bit(colors_dataset[index]) for index in range(colors_dataset.shape[0])]


def write_video(frames_rgb, out_path: Path, fps: float):
    """Write RGB frames as H.264 with gop=1 so decord can seek to any frame.

    Frames must be RGB, and both ends of the pipeline say so: imageio's ffmpeg
    writer takes RGB, and training reads these files back through decord in
    `mibot/data/datasets/json_dataset.py`, which returns RGB too. Handing this
    BGR would silently train on reversed channels while evaluation feeds RGB.
    (The earlier hand-rolled cv2.imdecode here happened to yield RGB on the
    legacy byte format but would yield BGR on marked standard buffers, which is
    why decoding now goes through decode_image_bit.)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        os.fspath(out_path),
        fps=fps,
        codec="libx264",
        macro_block_size=1,
        ffmpeg_params=[
            "-crf", "18",
            "-g", "1",
            "-pix_fmt", "yuv420p",
            "-threads", "1",
            "-loglevel", "error",
        ],
    )
    try:
        for frame in frames_rgb:
            writer.append_data(frame)
    finally:
        writer.close()


def normalize_instruction(text) -> str:
    """Trim and force a single trailing period."""
    return str(text).strip().rstrip(".") + "."


def build_general_prompt(instruction: str):
    """instruction.general: one prompt whose <image> order matches CAM_MAP."""
    return [
        {
            "images": [f"observations.{cam_key}" for cam_key, _ in CAM_MAP],
            "conversations": [
                {"from": "human", "value": CONVERSATION_TEMPLATE + instruction},
                {"from": "gpt", "value": ""},
            ],
        }
    ]


# --- Job planning -----------------------------------------------------------


@dataclass
class Job:
    src_hdf5: Path
    task_name: str
    src_episode_index: int
    global_index: int

    @property
    def chunk_id(self) -> int:
        return self.global_index // CHUNK_SIZE

    @property
    def stem(self) -> str:
        return f"episode_{self.global_index:07d}"

    def dst_json(self, dst_root: Path) -> Path:
        return dst_root / "data" / f"chunk-{self.chunk_id:04d}" / f"{self.stem}.json"

    def dst_video(self, dst_root: Path, cam_key: str) -> Path:
        return (
            dst_root / "videos" / f"chunk-{self.chunk_id:04d}"
            / f"{self.stem}.{cam_key}.mp4"
        )


def task_robots(src_root: Path, task: str):
    """Robot subdirectories that carry data for one task."""
    task_dir = src_root / task
    if not task_dir.is_dir():
        return []
    return sorted(
        entry.name for entry in task_dir.iterdir()
        if entry.is_dir() and (entry / "data").is_dir()
    )


def available_robots(src_root: Path):
    """Every robot subdirectory seen across all tasks, with its task count."""
    counts = {}
    for entry in sorted(src_root.iterdir()):
        if not entry.is_dir():
            continue
        for robot in task_robots(src_root, entry.name):
            counts[robot] = counts.get(robot, 0) + 1
    return [f"{robot} ({count} tasks)" for robot, count in sorted(counts.items())]


def discover_jobs(src_root: Path, env_cfg_type: str, tasks_filter, episodes_per_task: int = 0):
    """Walk <src_root>/<task>/<env_cfg_type>/data/episode_*.hdf5.

    Episodes are numbered globally in sorted task order so the numbering is
    reproducible across runs and independent of filesystem listing order.
    `episodes_per_task` caps each task independently, matching how the
    XPolicyLab `expert_data_num` argument behaves for the other policies.
    """
    # Only tasks that actually carry a <task>/<env_cfg_type>/data directory are
    # eligible. In RoboDojo_real each task belongs to exactly one robot, so this
    # is what keeps a piper_x run from also converting the piper and arx_x5
    # tasks sitting next to it.
    all_tasks = sorted(entry.name for entry in src_root.iterdir() if entry.is_dir())
    tasks = [
        task for task in all_tasks if (src_root / task / env_cfg_type / "data").is_dir()
    ]
    if not tasks:
        raise SystemExit(
            f"[process_data] no task under {src_root} has a "
            f"{env_cfg_type}/data directory.\n"
            f"  available robots: {', '.join(available_robots(src_root)) or '(none)'}"
        )

    if tasks_filter:
        wanted = [name.strip() for name in tasks_filter.split(",") if name.strip()]
        # Distinguish "no such task" from "task exists but is another robot's",
        # which is the mistake worth an explicit message.
        unknown = [name for name in wanted if name not in all_tasks]
        if unknown:
            raise SystemExit(
                f"[process_data] tasks not found under {src_root}: "
                f"{', '.join(unknown)}"
            )
        other_robot = [name for name in wanted if name not in tasks]
        if other_robot:
            details = ", ".join(
                f"{name} (has {', '.join(task_robots(src_root, name)) or 'no robot dir'})"
                for name in other_robot
            )
            raise SystemExit(
                f"[process_data] these tasks carry no {env_cfg_type} data: {details}\n"
                f"  pass the matching env_cfg_type, or drop them from the task list."
            )
        tasks = [name for name in tasks if name in wanted]

    jobs = []
    for task in tasks:
        data_dir = src_root / task / env_cfg_type / "data"
        if not data_dir.is_dir():
            continue
        files = sorted(
            (path for path in data_dir.iterdir()
             if path.name.startswith("episode_") and path.suffix == ".hdf5"),
            key=lambda path: int(path.stem[len("episode_"):]),
        )
        if episodes_per_task > 0:
            if len(files) < episodes_per_task:
                print(
                    f"[process_data] warning: task {task!r} has only {len(files)} "
                    f"episodes, fewer than the requested {episodes_per_task}",
                    flush=True,
                )
            files = files[:episodes_per_task]
        for src_index, path in enumerate(files):
            jobs.append(Job(
                src_hdf5=path,
                task_name=task,
                src_episode_index=src_index,
                global_index=len(jobs),
            ))
    return jobs


# --- Conversion -------------------------------------------------------------


def read_episode(src_hdf5: Path):
    """Read one RoboDojo HDF5 into plain arrays plus decoded video frames."""
    with h5py.File(src_hdf5, "r") as handle:
        instruction = handle["instruction"][()]
        if isinstance(instruction, (bytes, bytearray, np.bytes_)):
            instruction = bytes(instruction).decode("utf-8")

        # Several RoboDojo_real piper_x tasks ship no additional_info group.
        if "additional_info/frequency" in handle:
            fps = float(int(handle["additional_info/frequency"][()]))
        else:
            fps = FALLBACK_FPS

        state = handle["state"]
        action = handle["action"]
        arrays = {
            "left_pose": state["left_ee_poses"][:],
            "right_pose": state["right_ee_poses"][:],
            "left_grip_state": state["left_ee_joint_states"][:],
            "right_grip_state": state["right_ee_joint_states"][:],
            "left_arm_state": state["left_arm_joint_states"][:],
            "right_arm_state": state["right_arm_joint_states"][:],
            "left_grip_action": action["left_ee_joint_states"][:],
            "right_grip_action": action["right_ee_joint_states"][:],
        }

        # RoboDojo_real records a real end-effector action; simulation does not,
        # and there the pose action is emulated by a next-state shift instead.
        if "left_ee_poses" in action and "right_ee_poses" in action:
            arrays["left_pose_action"] = action["left_ee_poses"][:]
            arrays["right_pose_action"] = action["right_ee_poses"][:]

        num_frames = int(arrays["left_pose"].shape[0])
        if num_frames <= 0:
            raise RuntimeError(f"{src_hdf5}: episode has no frames")
        for name, array in arrays.items():
            if array.shape[0] != num_frames:
                raise RuntimeError(
                    f"{src_hdf5}: {name} length {array.shape[0]} != {num_frames}"
                )

        frames_by_cam = {}
        for cam_key, src_key in CAM_MAP:
            if src_key not in handle["vision"]:
                raise RuntimeError(f"{src_hdf5}: missing vision/{src_key}")
            frames = decode_jpeg_seq(handle[f"vision/{src_key}/colors"])
            if len(frames) != num_frames:
                raise RuntimeError(
                    f"{src_hdf5}: {src_key} has {len(frames)} frames != {num_frames}"
                )
            frames_by_cam[cam_key] = frames

    return instruction, fps, num_frames, arrays, frames_by_cam


def build_trajectory(job: Job, dst_root: Path, instruction, num_frames, arrays, reframe_p):
    """Assemble the JSON payload for one episode."""
    left_pos, left_rotm = build_ee_arrays(arrays["left_pose"], reframe_p)
    right_pos, right_rotm = build_ee_arrays(arrays["right_pose"], reframe_p)

    # Prefer the recorded end-effector action (RoboDojo_real); fall back to the
    # next-state shift when the source has none (RoboDojo simulation).
    if "left_pose_action" in arrays:
        left_pos_action, left_rotm_action = build_ee_arrays(
            arrays["left_pose_action"], reframe_p
        )
        right_pos_action, right_rotm_action = build_ee_arrays(
            arrays["right_pose_action"], reframe_p
        )
    else:
        left_pos_action, left_rotm_action = (
            next_state_repeat(left_pos), next_state_repeat(left_rotm)
        )
        right_pos_action, right_rotm_action = (
            next_state_repeat(right_pos), next_state_repeat(right_rotm)
        )

    def column(array):
        return np.asarray(array, dtype=np.float64).reshape(-1, 1).tolist()

    def rows(array):
        return np.asarray(array, dtype=np.float64).tolist()

    # ARX X5 has no torso joint and no mobile base: emit explicit zero columns
    # so the loader's waist/base delta terms are a constant 0 instead of null.
    zeros_1d = [[0.0] for _ in range(num_frames)]
    zeros_3d = [[0.0, 0.0, 0.0] for _ in range(num_frames)]

    proprios = {
        "left_ee_pos": left_pos,
        "left_ee_rotm": left_rotm,
        "left_arm_joint": rows(arrays["left_arm_state"]),
        "left_gripper_pos": column(arrays["left_grip_state"]),
        "right_ee_pos": right_pos,
        "right_ee_rotm": right_rotm,
        "right_arm_joint": rows(arrays["right_arm_state"]),
        "right_gripper_pos": column(arrays["right_grip_state"]),
        "waist_pos": zeros_1d,
    }

    # Gripper actions always come from the source action group; the pose action
    # is either recorded (real) or emulated above (simulation).
    actions = {
        "left_ee_pos": left_pos_action,
        "left_ee_rotm": left_rotm_action,
        "left_gripper_pos": column(arrays["left_grip_action"]),
        "right_ee_pos": right_pos_action,
        "right_ee_rotm": right_rotm_action,
        "right_gripper_pos": column(arrays["right_grip_action"]),
        "waist_pos": zeros_1d,
        "base_vel": zeros_3d,
    }

    observations = {
        cam_key: [{
            "path": video_path_for_json(job.dst_video(dst_root, cam_key)),
            "start": 0,
            "crop_bbox": None,
        }]
        for cam_key, _ in CAM_MAP
    }

    return {
        "trajectory_type": "success",
        "time": datetime.now().strftime("%Y-%m-%d_%H_%M_%S"),
        "num_frames": num_frames,
        "instruction": {"general": build_general_prompt(normalize_instruction(instruction))},
        "observations": observations,
        "proprios": proprios,
        "actions": actions,
    }


def convert_one(job: Job, dst_root: Path, fps_override, reframe_p):
    """Convert a single episode. Returns (global_index, num_frames, error)."""
    try:
        instruction, fps, num_frames, arrays, frames_by_cam = read_episode(job.src_hdf5)
        if fps_override:
            fps = float(fps_override)

        traj = build_trajectory(job, dst_root, instruction, num_frames, arrays, reframe_p)

        for cam_key, _ in CAM_MAP:
            write_video(frames_by_cam[cam_key], job.dst_video(dst_root, cam_key), fps)

        json_path = job.dst_json(dst_root)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = json_path.with_suffix(".json.tmp")
        with open(tmp_path, "w") as handle:
            json.dump(traj, handle, ensure_ascii=False)
        os.replace(tmp_path, json_path)
        return job.global_index, num_frames, None
    except Exception as exc:
        return job.global_index, 0, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"


# --- Normalization statistics ----------------------------------------------
# The shipped configs/data/load_washer.yaml statistics come from a different
# robot, so training on this data with them would clip the gripper and joint
# channels to a constant. These helpers recompute them from what was just
# written, following the same rules as scripts/compute_robodojo_stats.py:
#
#   1. gripper, both sides: hard-coded onto [-1, 1] from the measured travel.
#      action (gaussian) mean = std = GRIPPER_MAX / 2; state (quantile)
#      q01 = 0, q99 = GRIPPER_MAX.
#   2. EE action (ee_pos / ee_aa): gaussian per (step, dim), estimated from
#      moving frames only so idle frames do not spike the distribution.
#   3. remaining state (arm_joint): 1st/99th percentiles.
#
# Only whole frames are sampled (actual_length == action_length), so the
# tail-padded frames never enter the statistics. Everything runs in float64.


def episode_action_deltas(traj, action_length: int):
    """Packed relative action deltas for the whole frames of one episode.

    Returns an (n, action_length, 60) array where n counts only frames with a
    full action_length horizon, matching the reference script's
    `range(0, num_frames - ideal + 1)` window.
    """
    num_frames = int(traj["num_frames"])
    proprios, actions = traj["proprios"], traj["actions"]

    arrays = {
        key: np.asarray(proprios[key], dtype=np.float64)
        for key in ("left_ee_pos", "left_ee_rotm", "left_gripper_pos",
                    "right_ee_pos", "right_ee_rotm", "right_gripper_pos", "waist_pos")
    }
    targets = {
        key: np.asarray(actions[key], dtype=np.float64)
        for key in ("left_ee_pos", "left_ee_rotm", "left_gripper_pos",
                    "right_ee_pos", "right_ee_rotm", "right_gripper_pos",
                    "waist_pos", "base_vel")
    }
    rotms = {
        side: arrays[f"{side}_ee_rotm"].reshape(-1, 3, 3) for side in ("left", "right")
    }
    target_rotms = {
        side: targets[f"{side}_ee_rotm"].reshape(-1, 3, 3) for side in ("left", "right")
    }

    # Packing slices are read from io.ACTION_PARTS so this stays correct if the
    # 60-dim layout is ever revised.
    parts = dict(ACTION_PARTS)

    # Only frames with a full horizon; no tail padding enters the statistics.
    frames = range(0, max(0, num_frames - action_length + 1))
    if len(frames) == 0:
        return np.zeros((0, action_length, ACTION_DIM), dtype=np.float64)

    out = np.zeros((len(frames), action_length, ACTION_DIM), dtype=np.float64)

    for row, frame in enumerate(frames):
        stop = frame + action_length
        packed = out[row]

        for side in ("left", "right"):
            rotm = rotms[side][frame]
            pos = arrays[f"{side}_ee_pos"][frame]
            packed[:, parts[f"{side}_ee_pos"]] = (
                rotm.T @ (targets[f"{side}_ee_pos"][frame:stop] - pos).T
            ).T
            packed[:, parts[f"{side}_ee_aa"]] = rotm2aa_batch(
                rotm.T @ target_rotms[side][frame:stop]
            )
            packed[:, parts[f"{side}_gripper"]] = (
                targets[f"{side}_gripper_pos"][frame:stop]
                - arrays[f"{side}_gripper_pos"][frame]
            )

        packed[:, parts["waist"]] = (
            targets["waist_pos"][frame:stop] - arrays["waist_pos"][frame]
        )
        packed[:, parts["base"]] = targets["base_vel"][frame:stop]

    return out


def episode_state_matrix(traj, action_length: int):
    """Packed (n, 60) state for the whole frames of one episode.

    Sampled over the same window as episode_action_deltas so the state and
    action statistics describe the same set of frames.
    """
    proprios = traj["proprios"]
    num_frames = int(traj["num_frames"])
    keep = max(0, num_frames - action_length + 1)
    state = np.zeros((num_frames, STATE_DIM), dtype=np.float64)

    for side, joint_slice, grip_slice in (
        ("left", STATE_SLICES["left_arm_joint"], STATE_SLICES["left_gripper"]),
        ("right", STATE_SLICES["right_arm_joint"], STATE_SLICES["right_gripper"]),
    ):
        joints = np.asarray(proprios[f"{side}_arm_joint"], dtype=np.float64)
        if joints.shape[1] > joint_slice.stop - joint_slice.start:
            raise ValueError(
                f"{side}_arm_joint has {joints.shape[1]} dims, "
                f"more than the {joint_slice.stop - joint_slice.start} the state layout allows"
            )
        state[:, joint_slice.start:joint_slice.start + joints.shape[1]] = joints
        state[:, grip_slice] = np.asarray(
            proprios[f"{side}_gripper_pos"], dtype=np.float64
        ).reshape(-1, 1)

    return state[:keep]


def stats_worker(json_path: str, action_length: int):
    """Per-episode samples for the statistics pass, run in a worker process."""
    try:
        with open(json_path, "r") as handle:
            traj = json.load(handle)
        actions = episode_action_deltas(traj, action_length)
        state = episode_state_matrix(traj, action_length)
        return actions, state, None
    except Exception as exc:
        return None, None, f"{json_path}: {type(exc).__name__}: {exc}"


def step_gaussian(samples, columns):
    """Per-(step, dim) mean/std from moving frames only.

    `samples` is (N, T, 60); `columns` lists the packed dimensions to estimate.
    A column with fewer than MIN_MOTION_FRAMES moving samples degrades to the
    identity transform (mean 0, std 1). Returns the arrays plus a per-column
    report of (step, dim, total, kept).
    """
    steps = samples.shape[1]
    mean = np.zeros((steps, ACTION_DIM), dtype=np.float64)
    std = np.ones((steps, ACTION_DIM), dtype=np.float64)
    report = []

    for step in range(steps):
        for dim in columns:
            values = samples[:, step, dim]
            moving = values[np.abs(values) >= MOTION_EPS]
            report.append((step, dim, values.size, moving.size))
            if moving.size < MIN_MOTION_FRAMES:
                continue
            mean[step, dim] = moving.mean()
            std[step, dim] = max(moving.std(), MIN_STD)

    return mean, std, report


def compute_stats(json_files, action_length: int, workers: int):
    """Normalization statistics, following scripts/compute_robodojo_stats.py."""
    action_chunks = []
    state_chunks = []
    failures = []

    print(
        f"[process_data] computing normalization statistics over "
        f"{len(json_files)} episodes (action_length={action_length})",
        flush=True,
    )
    frames = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(stats_worker, path, action_length): path for path in json_files
        }
        with tqdm(total=len(futures), desc="stats", unit="ep", smoothing=0.05) as bar:
            for future in as_completed(futures):
                actions, state, error = future.result()
                if error:
                    failures.append(error)
                elif actions.shape[0] > 0:
                    action_chunks.append(actions)
                    state_chunks.append(state)
                    frames += actions.shape[0]
                bar.update(1)
                bar.set_postfix(failed=len(failures), frames=frames)

    if failures:
        for message in failures[:10]:
            print(f"[process_data]   stats failure: {message}", flush=True)
        raise SystemExit(f"[process_data] statistics failed on {len(failures)} episodes")
    if not action_chunks:
        raise SystemExit(
            f"[process_data] no episode is at least {action_length} frames long, "
            f"so no whole-frame sample exists for the statistics"
        )

    actions = np.concatenate(action_chunks, axis=0)
    states = np.concatenate(state_chunks, axis=0)
    print(f"[process_data] collected {actions.shape[0]} whole frames", flush=True)

    parts = dict(ACTION_PARTS)

    # 1. EE action: gaussian per (step, dim), moving frames only.
    gaussian_columns = [
        dim for name in STEP_GAUSSIAN_ACTION_PARTS
        for dim in range(parts[name].start, parts[name].stop)
    ]
    mean, std, report = step_gaussian(actions, gaussian_columns)
    kept = np.mean([row[3] / row[2] for row in report]) * 100 if report else 0.0
    degraded = [(row[0], row[1]) for row in report if row[3] < MIN_MOTION_FRAMES]
    print(
        f"[process_data]   EE action gaussian: kept {kept:.1f}% moving frames, "
        f"std range [{std[:, gaussian_columns].min():.6f}, "
        f"{std[:, gaussian_columns].max():.6f}]",
        flush=True,
    )
    if degraded:
        print(
            f"[process_data]   warning: {len(degraded)} (step, dim) columns had "
            f"fewer than {MIN_MOTION_FRAMES} moving frames and fall back to "
            f"mean 0 / std 1",
            flush=True,
        )

    # 2. gripper action: hard-coded so the delta's [-GRIPPER_MAX, +GRIPPER_MAX]
    #    travel maps onto [-1, 1]. See the GRIPPER_ACTION_* notes above for why
    #    this is 0 / GRIPPER_MAX rather than the reference script's 0.5 / 0.5.
    for name in GRIPPER_ACTION_PARTS:
        mean[:, parts[name]] = GRIPPER_ACTION_MEAN
        std[:, parts[name]] = GRIPPER_ACTION_STD
    print(
        f"[process_data]   gripper action: hard-coded mean = {GRIPPER_ACTION_MEAN}, "
        f"std = {GRIPPER_ACTION_STD} (delta convention)",
        flush=True,
    )

    q01 = np.zeros((1, STATE_DIM), dtype=np.float64)
    q99 = np.zeros((1, STATE_DIM), dtype=np.float64)

    # 3. gripper state: hard-coded quantiles over the measured travel.
    for name in GRIPPER_STATE_SLICES:
        q01[:, STATE_SLICES[name]] = 0.0
        q99[:, STATE_SLICES[name]] = GRIPPER_MAX
    print(
        f"[process_data]   gripper state: hard-coded q01 = 0, q99 = {GRIPPER_MAX}",
        flush=True,
    )

    # 4. remaining state (arm joints): 1st/99th percentiles.
    quantile_columns = [
        dim for name in QUANTILE_STATE_SLICES
        for dim in range(STATE_SLICES[name].start, STATE_SLICES[name].stop)
    ]
    q01[0, quantile_columns] = np.percentile(states[:, quantile_columns], 1, axis=0)
    q99[0, quantile_columns] = np.percentile(states[:, quantile_columns], 99, axis=0)

    # validate_quantiles() rejects q99 <= q01 unless both are exactly zero, so
    # collapse constant columns to 0/0; normalize_quantile passes those through.
    degenerate = q99 <= q01
    q01[degenerate] = 0.0
    q99[degenerate] = 0.0
    print(
        f"[process_data]   arm joint state: quantile over {len(quantile_columns)} dims, "
        f"{int(degenerate.sum())} constant dims zeroed",
        flush=True,
    )

    return mean, std, q01, q99


def format_matrix(matrix, indent: str) -> str:
    """Render a 2D array as a YAML block list of inline lists.

    Values use .16e so the float64 statistics survive a YAML round trip
    without losing precision, matching the reference script's snippet output.
    """
    lines = []
    for row in np.asarray(matrix):
        values = ", ".join(f"{float(value):+.16e}" for value in row)
        lines.append(f"{indent}- [{values}]")
    return "\n".join(lines)


def write_data_config(
    config_path: Path,
    json_dirs,
    mean,
    std,
    q01,
    q99,
    action_length: int,
    batch_size: int,
):
    """Emit a hydra data config wired to the converted dataset."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    paths_block = "\n".join(f"      - {path}" for path in json_dirs)

    content = f"""# @package _global_
# Generated by scripts/process_data.py from the RoboDojo official HDF5 release.
# mean/std: packed relative actions, shape ({action_length}, {ACTION_DIM}).
#   EE dims are gaussian per (step, dim) over moving frames only (|x| >= {MOTION_EPS:g});
#   gripper dims are hard-coded to mean = {GRIPPER_ACTION_MEAN}, std = {GRIPPER_ACTION_STD}, mapping the
#     delta travel [-{GRIPPER_MAX}, +{GRIPPER_MAX}] onto [-1, 1];
#   unused dims stay at mean 0 / std 1 (identity).
# q01/q99: packed robot state, shape (1, {STATE_DIM}).
#   gripper dims are hard-coded to 0 / {GRIPPER_MAX}; arm joint dims are 1st/99th
#   percentiles; constant dims are written as 0/0, which the loader passes through.

data:
  type: BaseDataModule
  params:
    type: json
    max_steps: ${{trainer.max_steps}}
    train_datasets:
      batch_size: {batch_size}
      action_length: {action_length}
      paths:
{paths_block}
      mean:
{format_matrix(mean, "      ")}
      std:
{format_matrix(std, "      ")}
      q01:
{format_matrix(q01, "      ")}
      q99:
{format_matrix(q99, "      ")}
"""
    config_path.write_text(content)


# --- CLI --------------------------------------------------------------------


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert RoboDojo official HDF5 episodes to the xr1 JSON format.",
    )
    parser.add_argument(
        "--bench-name", default=SIM_BENCH, choices=list(SUPPORTED_BENCHES),
        help=f"dataset release to convert (default: {SIM_BENCH})",
    )
    parser.add_argument(
        "--src", default=None,
        help="HDF5 root holding <task>/<env_cfg_type>/data/*.hdf5 "
             "(default: nearest data/<bench_name> found above the xr1 directory)",
    )
    parser.add_argument(
        "--dst", default=None,
        help="output root (default: <xr1>/data/<bench_name lowercased>_xr1)",
    )
    parser.add_argument(
        "--env-cfg-type", default="arx_x5",
        help="robot subdirectory under each task; only tasks carrying this "
             f"robot are converted. {REAL_BENCH} ships "
             f"{', '.join(sorted(EEF_REFRAME_P_REAL))} (default: arx_x5)",
    )
    parser.add_argument(
        "--tasks", default=None,
        help="comma-separated task names to convert (default: every task that "
             "carries the requested robot)",
    )
    parser.add_argument(
        "--episodes-per-task", type=int, default=0,
        help="convert at most N episodes from each task; 0 means all "
             "(this is what the XPolicyLab expert_data_num argument maps to)",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="hard cap on the total number of episodes, applied after "
             "--episodes-per-task; 0 means no cap",
    )
    parser.add_argument(
        "--workers", type=int, default=0,
        help="worker processes; 0 means os.cpu_count() // 2",
    )
    parser.add_argument(
        "--fps", type=float, default=0.0,
        help="override the video fps; 0 uses the HDF5 recording frequency",
    )
    parser.add_argument(
        "--action-length", type=int, default=30,
        help="action chunk length the statistics are computed for (default: 30)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="batch_size written into the generated data config (default: 16)",
    )
    parser.add_argument(
        "--config-name", default="robodojo",
        help="basename of the generated config under configs/data (default: robodojo)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="reconvert episodes whose JSON and videos already exist",
    )
    parser.add_argument(
        "--skip-stats", action="store_true",
        help="only convert episodes; do not compute statistics or write a config",
    )
    parser.add_argument(
        "--stats-only", action="store_true",
        help="skip conversion; recompute statistics from the existing output",
    )
    return parser.parse_args(argv)


def resolve_workers(requested: int) -> int:
    if requested > 0:
        return requested
    return max(2, min(64, (os.cpu_count() or 8) // 2))


def existing_json_files(dst_root: Path):
    data_dir = dst_root / "data"
    if not data_dir.is_dir():
        return []
    return sorted(os.fspath(path) for path in data_dir.glob("chunk-*/episode_*.json"))


def run_conversion(args, src_root: Path, dst_root: Path, workers: int):
    """Convert HDF5 episodes to JSON + MP4. Returns the JSON paths written."""
    jobs = discover_jobs(
        src_root, args.env_cfg_type, args.tasks, args.episodes_per_task
    )
    if not jobs:
        raise SystemExit(
            f"[process_data] no episodes found under {src_root} "
            f"with env_cfg_type={args.env_cfg_type!r}"
        )
    names = sorted({job.task_name for job in jobs})
    suffix = (
        f" (capped at {args.episodes_per_task} per task)"
        if args.episodes_per_task > 0 else ""
    )
    print(
        f"[process_data] selected {len(jobs)} episodes from {len(names)} "
        f"{args.env_cfg_type} tasks{suffix}: {', '.join(names)}",
        flush=True,
    )

    if args.limit > 0:
        jobs = jobs[: args.limit]
        print(f"[process_data] --limit applied: {len(jobs)} episodes", flush=True)

    planned = [os.fspath(job.dst_json(dst_root)) for job in jobs]

    if not args.overwrite:
        pending = []
        for job in jobs:
            videos_ready = all(
                job.dst_video(dst_root, cam_key).exists() for cam_key, _ in CAM_MAP
            )
            if job.dst_json(dst_root).exists() and videos_ready:
                continue
            pending.append(job)
        print(
            f"[process_data] {len(jobs) - len(pending)} already converted, "
            f"{len(pending)} to do",
            flush=True,
        )
        jobs = pending

    if not jobs:
        print("[process_data] nothing to convert", flush=True)
        return planned

    fps_override = args.fps if args.fps > 0 else None
    reframe_p = eef_reframe_p(args.bench_name, args.env_cfg_type)
    failures = []
    total_frames = 0

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(convert_one, job, dst_root, fps_override, reframe_p)
            for job in jobs
        ]
        with tqdm(total=len(futures), desc="convert", unit="ep", smoothing=0.05) as bar:
            for future in as_completed(futures):
                index, frames, error = future.result()
                if error:
                    failures.append((index, error))
                    bar.write(f"[process_data] FAIL episode {index}: {error.splitlines()[0]}")
                else:
                    total_frames += frames
                bar.update(1)
                bar.set_postfix(failed=len(failures), frames=total_frames)

    if failures:
        print(f"[process_data] {len(failures)} episodes failed", flush=True)
        for index, error in failures[:10]:
            print(f"[process_data]   episode {index}: {error.splitlines()[0]}", flush=True)
        raise SystemExit(1)

    print(
        f"[process_data] converted {len(jobs)} episodes, {total_frames} frames",
        flush=True,
    )
    return planned


def main(argv=None):
    args = parse_args(argv)
    workers = resolve_workers(args.workers)

    # Fail before any conversion work if the bench/robot pair has no matrix.
    eef_reframe_p(args.bench_name, args.env_cfg_type)

    dst_root = resolve_dest_root(args.dst, args.bench_name)
    print(f"[process_data] xr1 dir     = {XR1_DIR}", flush=True)
    print(f"[process_data] bench       = {args.bench_name}", flush=True)
    print(f"[process_data] robot       = {args.env_cfg_type}", flush=True)
    print(f"[process_data] destination = {dst_root}", flush=True)
    print(f"[process_data] workers     = {workers}", flush=True)

    if args.stats_only:
        json_files = existing_json_files(dst_root)
        if not json_files:
            raise SystemExit(
                f"[process_data] --stats-only found no JSON under {dst_root / 'data'}"
            )
        print(f"[process_data] --stats-only: {len(json_files)} existing episodes", flush=True)
    else:
        src_root = resolve_source_root(args.src, args.bench_name)
        print(f"[process_data] source      = {src_root}", flush=True)
        print(
            f"[process_data] robots found: "
            f"{', '.join(available_robots(src_root)) or '(none)'}",
            flush=True,
        )
        json_files = run_conversion(args, src_root, dst_root, workers)

    if args.skip_stats:
        print("[process_data] --skip-stats: no config written", flush=True)
        return

    json_files = [path for path in json_files if os.path.exists(path)]
    if not json_files:
        raise SystemExit("[process_data] no converted episodes available for statistics")

    mean, std, q01, q99 = compute_stats(json_files, args.action_length, workers)

    config_path = CONFIG_DATA_DIR / f"{args.config_name}.yaml"
    # Absolute, so the config does not depend on where training is launched.
    config_data_path = str(dst_root / "data")

    write_data_config(
        config_path, [config_data_path], mean, std, q01, q99,
        args.action_length, args.batch_size,
    )

    rel_config = os.path.relpath(config_path, XR1_DIR)
    print(f"[process_data] wrote data config: {rel_config}", flush=True)
    print(
        f"[process_data] train with: bash scripts/train.sh data={args.config_name}",
        flush=True,
    )


if __name__ == "__main__":
    sys.exit(main())
