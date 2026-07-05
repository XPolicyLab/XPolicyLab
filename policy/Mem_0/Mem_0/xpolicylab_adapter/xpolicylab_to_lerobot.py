"""
Convert XPolicyLab trajectory HDF5 directly into the Mem_0 LeRobot training
format (one step -- no RMBench-format intermediate).

DP-style entrypoint (called by ../process_data.sh):

    python xpolicylab_to_lerobot.py <bench_name> <ckpt_name> <env_cfg_type> \
        <expert_data_num> <action_type> --task_type {M1,Mn} [--instruction "..."] \
        [--language_annotation PATH]

Reads:  <ROOT>/data/<bench_name>/<ckpt_name>/<env_cfg_type>/data/episode_*.hdf5
        via XPolicyLab.utils.load_file.load_hdf5 (default sample: data/RoboDojo/test_data/arx_x5)
Writes: policy/Mem_0/data/<dataset>-<ckpt>-<env>-<action>-lerobot
        (legacy: Mem_0/lerobot_datasets/... when MEM0_LEGACY_PATHS=1)

State/action are packed with XPolicyLab's dual-arm joint convention (14-dim:
[LA(6),LGrip,RA(6),RGrip]) and expanded to Mem_0's 16-dim model layout
([LA(6),pad,RA(6),pad,LGrip,RGrip]). Every head-camera frame is standardized to
RGB HWC (240, 320, 3).

Sub-task annotation (Mem_0 trains on `subtask` language + `subtask_end`):

- ``--task_type M1`` (single-stage): one instruction for the whole episode
  (HDF5 ``instruction``, else --instruction, else <task_name>); subtask_end=1 on
  the final 8 frames.
- ``--task_type Mn`` (multi-stage): per-segment sub-task language consumed from a
  language_annotation.json in the RMBench reference format
  (``{"episode_<i>": [[subtask_text, duration], ...]}``). ``global_task`` is the
  HDF5 ``instruction``. subtask_end=1 within 8 frames of each segment boundary.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

ADAPTER_DIR = os.path.dirname(os.path.abspath(__file__))
UPSTREAM_DIR = os.path.dirname(ADAPTER_DIR)               # policy/Mem_0/Mem_0
POLICY_DIR = os.path.dirname(UPSTREAM_DIR)              # policy/Mem_0
ROOT_DIR = os.path.abspath(os.path.join(UPSTREAM_DIR, "..", "..", "..", ".."))  # workspace root (data/)
ANNOTATIONS_ROOT = os.path.join(UPSTREAM_DIR, "language_annotations")
LANGUAGE_ANNOTATION_ROOT = os.path.join(ADAPTER_DIR, "language_annotation")
for p in (ROOT_DIR, UPSTREAM_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from XPolicyLab.utils.load_file import load_hdf5  # noqa: E402
from XPolicyLab.utils.process_data import (  # noqa: E402
    decode_image_bit,
    get_robot_action_dim_info,
    pack_robot_state,
)

STD_W, STD_H = 320, 240
SUBTASK_END_WINDOW = 8  # frames before a (sub)task boundary flagged subtask_end=1
DEFAULT_VCODEC = "h264"

STATE_NAMES = [
    "left_joint_0", "left_joint_1", "left_joint_2", "left_joint_3", "left_joint_4",
    "left_joint_5", "left_joint_6", "right_joint_0", "right_joint_1", "right_joint_2",
    "right_joint_3", "right_joint_4", "right_joint_5", "right_joint_6",
    "left_gripper", "right_gripper",
]

FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (16,), "names": STATE_NAMES},
    "action": {"dtype": "float32", "shape": (16,), "names": STATE_NAMES},
    "observation.image.head_camera": {
        "dtype": "video", "shape": (STD_H, STD_W, 3),
        "names": ["height", "width", "channels"],
    },
    "subtask": {"dtype": "string", "shape": (1,), "names": ["subtask_annotation"]},
    "global_task": {"dtype": "string", "shape": (1,), "names": ["global_task_annotation"]},
    "subtask_end": {"dtype": "int32", "shape": (1,), "names": ["subtask_end_flag"]},
    "episode_id": {"dtype": "int32", "shape": (1,), "names": ["episode_id"]},
}


def read_instruction(data: dict, fallback: str) -> str:
    """Resolve global_task from HDF5 ``instruction`` (RMBench TASK_INSTRUCTIONS equivalent)."""
    inst = data.get("instruction")
    if isinstance(inst, bytes):
        inst = inst.decode("utf-8", errors="replace")
    if isinstance(inst, str) and inst.strip():
        return inst.strip()
    return fallback


def _packed14_to_model16(packed14: np.ndarray) -> np.ndarray:
    """[LA(6),LGrip,RA(6),RGrip] -> Mem_0 model layout [LA(6),pad,RA(6),pad,LGrip,RGrip]."""
    out = np.zeros((packed14.shape[0], 16), dtype=np.float32)
    out[:, 0:6] = packed14[:, 0:6]
    out[:, 6] = 0.0
    out[:, 7:13] = packed14[:, 7:13]
    out[:, 13] = 0.0
    out[:, 14] = packed14[:, 6]
    out[:, 15] = packed14[:, 13]
    return out


def resolve_dataset_out_dir(
    bench_name: str,
    ckpt_name: str,
    env_cfg_type: str,
    expert_data_num: int,
    action_type: str,
) -> Path:
    """Default LeRobot output root (README §4.2); legacy when MEM0_LEGACY_PATHS=1."""
    tag = f"{bench_name}-{ckpt_name}-{env_cfg_type}-{action_type}"
    if os.environ.get("MEM0_LEGACY_PATHS") == "1":
        legacy_name = (
            f"{bench_name}-{ckpt_name}-{env_cfg_type}-{expert_data_num}-{action_type}"
        )
        return Path(UPSTREAM_DIR) / "lerobot_datasets" / legacy_name
    return Path(POLICY_DIR) / "data" / f"{tag}-lerobot"


def _decode_rgb(img_bit) -> np.ndarray:
    """Encoded bytes -> RGB HWC (240, 320, 3) uint8 (decode_image_bit already returns RGB)."""
    img = decode_image_bit(img_bit)
    assert img.ndim == 3 and img.shape[-1] == 3, f"Expected HxWx3, got {img.shape}"
    img = cv2.resize(img, (STD_W, STD_H), interpolation=cv2.INTER_AREA)
    assert img.shape == (STD_H, STD_W, 3)
    return img


def _load_preview_frames(preview_path: Path, expected_length: int) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(preview_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open preview video: {preview_path}")
    frames: List[np.ndarray] = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (STD_W, STD_H), interpolation=cv2.INTER_AREA)
        frames.append(rgb.astype(np.uint8))
    cap.release()
    if len(frames) != expected_length:
        raise ValueError(
            f"Preview frame count mismatch for {preview_path}: "
            f"expected {expected_length}, got {len(frames)}"
        )
    return frames


def _segment_boundaries(episode_annotation, episode_length: int):
    """[[text, duration], ...] -> [(start, end, text)] consecutive segments (clamped)."""
    boundaries, cur = [], 0
    for text, duration in episode_annotation:
        start = cur
        end = min(cur + int(duration) - 1, episode_length - 1)
        boundaries.append((start, end, text))
        cur = end + 1
        if cur >= episode_length:
            break
    if boundaries:
        s, _e, t = boundaries[-1]
        boundaries[-1] = (s, episode_length - 1, t)
    return boundaries


def pack_episode_state_action(
    data: dict,
    action_type: str,
    robot_action_dim_info: dict,
) -> Tuple[np.ndarray, np.ndarray, int]:
    state14 = np.asarray(
        pack_robot_state(
            data, action_type, robot_action_dim_info,
            source_type="dataset", state_type="state",
        ),
        dtype=np.float32,
    )
    action14 = np.asarray(
        pack_robot_state(
            data, action_type, robot_action_dim_info,
            source_type="dataset", state_type="action",
        ),
        dtype=np.float32,
    )
    state16 = _packed14_to_model16(state14)
    action16 = _packed14_to_model16(action14)
    return state16, action16, int(state16.shape[0])


def load_episode_images(
    data: dict,
    task_dir: Path,
    episode_idx: int,
    *,
    camera: str = "cam_head",
    use_preview: bool = True,
    episode_length: int,
) -> List[np.ndarray]:
    if use_preview:
        preview_path = (
            task_dir / "preview_video" / f"episode_{episode_idx:07d}_{camera}.mp4"
        )
        if preview_path.is_file():
            return _load_preview_frames(preview_path, episode_length)
    colors = data["vision"][camera]["colors"]
    return [_decode_rgb(colors[i]) for i in range(episode_length)]


def resolve_mn_annotation_path(
    task_name: str,
    bench_name: str,
    env_cfg_type: str,
    language_annotation: Optional[str] = None,
    annotation_root: Optional[str] = None,
) -> Path:
    if language_annotation:
        return Path(language_annotation)
    if annotation_root:
        candidate = Path(annotation_root) / task_name / "language_annotation.json"
        if candidate.is_file():
            return candidate
    for candidate in (
        Path(LANGUAGE_ANNOTATION_ROOT) / task_name / "language_annotation.json",
        Path(ANNOTATIONS_ROOT) / bench_name / task_name / env_cfg_type / "language_annotation.json",
    ):
        if candidate.is_file():
            return candidate
    return Path(LANGUAGE_ANNOTATION_ROOT) / task_name / "language_annotation.json"


def episode_hdf5_path(task_dir: Path, episode_idx: int) -> Path:
    return task_dir / "data" / f"episode_{episode_idx:07d}.hdf5"


def create_mem0_lerobot_dataset(
    repo_id: str,
    out_root: Path,
    *,
    vcodec: str = DEFAULT_VCODEC,
    fps: int = 30,
):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.video_utils import encode_video_frames

    dataset = LeRobotDataset.create(
        repo_id=repo_id, fps=fps, features=FEATURES, root=out_root, use_videos=True,
    )
    dataset._vcodec = vcodec  # noqa: SLF001

    def _encode_episode_videos(episode_index: int) -> None:
        for key in dataset.meta.video_keys:
            video_path = dataset.root / dataset.meta.get_video_file_path(episode_index, key)
            if video_path.is_file():
                continue
            img_dir = dataset._get_image_file_path(  # noqa: SLF001
                episode_index=episode_index, image_key=key, frame_index=0,
            ).parent
            encode_video_frames(
                img_dir, video_path, dataset.fps, overwrite=True, vcodec=dataset._vcodec,
            )
            shutil.rmtree(img_dir)
        if len(dataset.meta.video_keys) > 0 and episode_index == 0:
            from lerobot.datasets.utils import write_info
            dataset.meta.update_video_info()
            write_info(dataset.meta.info, dataset.meta.root)

    dataset.encode_episode_videos = _encode_episode_videos  # type: ignore[method-assign]
    return dataset


def convert_episode_frames(
    dataset,
    *,
    task_name: str,
    task_type: str,
    episode_idx: int,
    state16: np.ndarray,
    action16: np.ndarray,
    images: List[np.ndarray],
    global_task: str,
    boundaries: Optional[list] = None,
) -> int:
    episode_length = state16.shape[0]
    for frame_idx in range(episode_length):
        if task_type == "Mn":
            subtask, subtask_end = global_task, 0
            for start, end, text in boundaries or []:
                if start <= frame_idx <= end:
                    subtask = text
                    subtask_end = 1 if (end - frame_idx) < SUBTASK_END_WINDOW else 0
                    break
        else:
            subtask = global_task
            subtask_end = 1 if (episode_length - frame_idx) <= SUBTASK_END_WINDOW else 0

        dataset.add_frame(
            {
                "observation.state": state16[frame_idx],
                "action": action16[frame_idx],
                "observation.image.head_camera": images[frame_idx],
                "subtask": subtask,
                "global_task": global_task,
                "subtask_end": np.array([subtask_end], dtype=np.int32),
                "episode_id": np.array([episode_idx], dtype=np.int32),
            },
            task=task_name,
        )
    dataset.save_episode()
    return episode_length


def main() -> None:
    parser = argparse.ArgumentParser(description="XPolicyLab HDF5 -> Mem_0 LeRobot dataset")
    parser.add_argument("bench_name", type=str)
    parser.add_argument("ckpt_name", type=str,
                        help="Experiment/raw-task key; HDF5 source dir under data/<dataset>/<ckpt_name>/")
    parser.add_argument("env_cfg_type", type=str)
    parser.add_argument("expert_data_num", type=int)
    parser.add_argument("action_type", type=str, help="'joint' (Mem_0 default) or 'ee'")
    parser.add_argument("--task_type", choices=["M1", "Mn"], required=True,
                        help="M1: single-stage; Mn: multi-stage with per-segment sub-tasks")
    parser.add_argument("--instruction", default=None,
                        help="Override global_task (default: HDF5 instruction, else task_name)")
    parser.add_argument("--language_annotation", default=None,
                        help="Mn segmentation JSON {episode_<i>:[[text,duration],...]}")
    parser.add_argument("--camera", default="cam_head",
                        help="vision camera key holding the head view (default cam_head)")
    parser.add_argument("--no-use-preview", action="store_true",
                        help="Decode JPEG frames from HDF5 instead of preview mp4")
    parser.add_argument("--vcodec", default=DEFAULT_VCODEC,
                        help=f"Video codec for LeRobot encoding (default: {DEFAULT_VCODEC})")
    args = parser.parse_args()

    robot_action_dim_info = get_robot_action_dim_info(args.env_cfg_type)
    assert len(robot_action_dim_info["arm_dim"]) == 2, (
        f"Mem_0 expects a dual-arm robot; env_cfg_type={args.env_cfg_type} gave "
        f"arm_dim={robot_action_dim_info['arm_dim']}."
    )

    task_dir = Path(ROOT_DIR) / "data" / args.bench_name / args.ckpt_name / args.env_cfg_type
    if not task_dir.is_dir():
        raise FileNotFoundError(
            f"Source data dir not found: {task_dir}\n"
            "Expected data/<bench_name>/<ckpt_name>/<env_cfg_type>/data/episode_*.hdf5."
        )

    annotations = {}
    if args.task_type == "Mn":
        ann_path = resolve_mn_annotation_path(
            args.ckpt_name, args.bench_name, args.env_cfg_type, args.language_annotation,
        )
        if not ann_path.is_file():
            raise FileNotFoundError(
                f"Mn task needs sub-task annotations but {ann_path} is missing."
            )
        annotations = json.loads(ann_path.read_text(encoding="utf-8"))

    out_name = f"{args.bench_name}-{args.ckpt_name}-{args.env_cfg_type}-{args.action_type}"
    out_root = resolve_dataset_out_dir(
        args.bench_name, args.ckpt_name, args.env_cfg_type,
        args.expert_data_num, args.action_type,
    )
    if out_root.exists():
        shutil.rmtree(out_root)

    dataset = create_mem0_lerobot_dataset(out_name, out_root, vcodec=args.vcodec)
    use_preview = not args.no_use_preview

    written, total_frames = 0, 0
    bar = tqdm(range(args.expert_data_num), desc=f"convert {out_name} [{args.task_type}]",
               unit="ep", dynamic_ncols=True)
    for episode_idx in bar:
        load_path = episode_hdf5_path(task_dir, episode_idx)
        if not load_path.is_file():
            tqdm.write(f"[convert] skip missing {load_path}")
            continue

        data = load_hdf5(str(load_path))
        global_task = args.instruction or read_instruction(data, args.ckpt_name)
        state16, action16, episode_length = pack_episode_state_action(
            data, args.action_type, robot_action_dim_info,
        )
        images = load_episode_images(
            data, task_dir, episode_idx, camera=args.camera,
            use_preview=use_preview, episode_length=episode_length,
        )

        boundaries = None
        if args.task_type == "Mn":
            ep_ann = annotations.get(f"episode_{episode_idx}")
            if not ep_ann:
                tqdm.write(f"[convert] skip episode {episode_idx}: no annotation entry")
                continue
            boundaries = _segment_boundaries(ep_ann, episode_length)

        nframes = convert_episode_frames(
            dataset,
            task_name=args.ckpt_name,
            task_type=args.task_type,
            episode_idx=episode_idx,
            state16=state16,
            action16=action16,
            images=images,
            global_task=global_task,
            boundaries=boundaries,
        )
        written += 1
        total_frames += nframes
        bar.set_postfix(frames=nframes, total=total_frames, episodes=written)
    bar.close()

    if written == 0:
        raise RuntimeError(f"No episodes converted from {task_dir}; check expert_data_num / paths.")
    tqdm.write(f"[convert] wrote {written} episodes / {total_frames} frames -> {out_root}")


if __name__ == "__main__":
    main()
