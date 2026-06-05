"""
Merge every task listed in xpolicylab_adapter/task_config.json into ONE cotrain
Mem_0 LeRobot dataset. M1 / Mn tasks follow the same semantics as the official
RMBench converters (M1_dataset_to_lerobot.py / Mn_dataset_to_lerobot.py), with
XPolicyLab HDF5 key mapping via pack_robot_state / vision.cam_head.

Called by ../process_data_batch.sh:

    python xpolicylab_batch_to_lerobot.py <dataset_name> <env_cfg_type> \
        <expert_data_num> <action_type> \
        --m1_tasks t1,t2 --mn_tasks t3,t4 \
        --annotation_root <path/to/language_annotation> \
        [--dataset_id NAME] [--vcodec h264] [--no-use-preview]

``global_task`` is read from each episode's HDF5 ``instruction`` field.
Mn sub-tasks come from ``<annotation_root>/<task>/language_annotation.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from tqdm import tqdm

ADAPTER_DIR = Path(__file__).resolve().parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(UPSTREAM_DIR) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_DIR))

from XPolicyLab.utils.load_file import load_hdf5  # noqa: E402
from XPolicyLab.utils.process_data import get_robot_action_dim_info  # noqa: E402
from xpolicylab_to_lerobot import (  # noqa: E402
    DEFAULT_VCODEC,
    POLICY_DIR,
    ROOT_DIR,
    UPSTREAM_DIR,
    _segment_boundaries,
    convert_episode_frames,
    create_mem0_lerobot_dataset,
    episode_hdf5_path,
    load_episode_images,
    pack_episode_state_action,
    read_instruction,
    resolve_mn_annotation_path,
)


def _split_tasks(raw: str) -> list[str]:
    return [t.strip() for t in raw.split(",") if t.strip()]


def _default_dataset_id(
    dataset_name: str,
    env_cfg_type: str,
    expert_data_num: int,
    action_type: str,
) -> str:
    return f"{dataset_name}-cotrain-{env_cfg_type}-{expert_data_num}-{action_type}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="XPolicyLab multi-task HDF5 -> single Mem_0 cotrain LeRobot dataset",
    )
    parser.add_argument("dataset_name", type=str)
    parser.add_argument("env_cfg_type", type=str)
    parser.add_argument("expert_data_num", type=int)
    parser.add_argument("action_type", type=str, help="'joint' or 'ee'")
    parser.add_argument("--m1_tasks", default="", help="Comma-separated M1 task names")
    parser.add_argument("--mn_tasks", default="", help="Comma-separated Mn task names")
    parser.add_argument(
        "--annotation_root", required=True,
        help="Root dir containing <task>/language_annotation.json for Mn tasks",
    )
    parser.add_argument("--dataset_id", default=None,
                        help="Output tag (default <dataset>-cotrain-<env>-<N>-<action>)")
    parser.add_argument("--camera", default="cam_head")
    parser.add_argument(
        "--no-use-preview", action="store_true",
        help="Decode JPEG frames from HDF5 instead of preview mp4",
    )
    parser.add_argument(
        "--vcodec", default=DEFAULT_VCODEC,
        help=f"Video codec for LeRobot encoding (default: {DEFAULT_VCODEC})",
    )
    args = parser.parse_args()

    m1_tasks = _split_tasks(args.m1_tasks)
    mn_tasks = _split_tasks(args.mn_tasks)
    if not m1_tasks and not mn_tasks:
        raise ValueError("At least one task must be listed under --m1_tasks or --mn_tasks.")

    robot_action_dim_info = get_robot_action_dim_info(args.env_cfg_type)
    if len(robot_action_dim_info["arm_dim"]) != 2:
        raise ValueError(
            f"Mem_0 expects a dual-arm robot; env_cfg_type={args.env_cfg_type} gave "
            f"arm_dim={robot_action_dim_info['arm_dim']}."
        )

    dataset_id = args.dataset_id or _default_dataset_id(
        args.dataset_name, args.env_cfg_type, args.expert_data_num, args.action_type,
    )
    if os.environ.get("MEM0_LEGACY_PATHS") == "1":
        out_root = Path(UPSTREAM_DIR) / "lerobot_datasets" / dataset_id
    else:
        out_root = Path(POLICY_DIR) / "data" / f"{dataset_id}-lerobot"
    if out_root.exists():
        shutil.rmtree(out_root)

    dataset = create_mem0_lerobot_dataset(dataset_id, out_root, vcodec=args.vcodec)
    use_preview = not args.no_use_preview
    annotation_root = Path(args.annotation_root)

    mn_annotations: dict[str, dict] = {}
    for task_name in mn_tasks:
        ann_path = resolve_mn_annotation_path(
            task_name, args.dataset_name, args.env_cfg_type,
            annotation_root=str(annotation_root),
        )
        if not ann_path.is_file():
            raise FileNotFoundError(f"Mn annotation missing: {ann_path}")
        mn_annotations[task_name] = json.loads(ann_path.read_text(encoding="utf-8"))

    jobs: list[tuple[str, str]] = [(t, "M1") for t in m1_tasks] + [(t, "Mn") for t in mn_tasks]
    written, total_frames, skipped = 0, 0, 0

    bar = tqdm(jobs, desc=f"cotrain {dataset_id}", unit="task", dynamic_ncols=True)
    for task_name, task_type in bar:
        task_dir = ROOT_DIR / "data" / args.dataset_name / task_name / args.env_cfg_type
        if not task_dir.is_dir():
            raise FileNotFoundError(f"Missing task data dir: {task_dir}")

        annotations = mn_annotations.get(task_name, {})
        task_written = 0

        ep_bar = tqdm(
            range(args.expert_data_num),
            desc=f"  {task_name}[{task_type}]",
            leave=False,
            unit="ep",
        )
        for episode_idx in ep_bar:
            hdf5_path = episode_hdf5_path(task_dir, episode_idx)
            if not hdf5_path.is_file():
                tqdm.write(f"[batch] skip missing {hdf5_path}")
                skipped += 1
                continue

            data = load_hdf5(str(hdf5_path))
            global_task = read_instruction(data, task_name)
            state16, action16, episode_length = pack_episode_state_action(
                data, args.action_type, robot_action_dim_info,
            )
            images = load_episode_images(
                data, task_dir, episode_idx, camera=args.camera,
                use_preview=use_preview, episode_length=episode_length,
            )

            boundaries = None
            if task_type == "Mn":
                ep_ann = annotations.get(f"episode_{episode_idx}")
                if not ep_ann:
                    tqdm.write(
                        f"[batch] skip {task_name}/episode_{episode_idx}: no annotation entry"
                    )
                    skipped += 1
                    continue
                boundaries = _segment_boundaries(ep_ann, episode_length)

            nframes = convert_episode_frames(
                dataset,
                task_name=task_name,
                task_type=task_type,
                episode_idx=episode_idx,
                state16=state16,
                action16=action16,
                images=images,
                global_task=global_task,
                boundaries=boundaries,
            )
            written += 1
            task_written += 1
            total_frames += nframes
            ep_bar.set_postfix(written=task_written, frames=total_frames)

        bar.set_postfix(episodes=written, frames=total_frames, skipped=skipped)

    bar.close()
    if written == 0:
        raise RuntimeError("No episodes converted; check task lists, paths, and expert_data_num.")
    tqdm.write(
        f"[batch] cotrain dataset ready: {written} episodes, {total_frames} frames, "
        f"{skipped} skipped -> {out_root}"
    )


if __name__ == "__main__":
    main()
