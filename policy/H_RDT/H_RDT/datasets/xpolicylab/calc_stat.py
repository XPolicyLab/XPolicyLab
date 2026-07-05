#!/usr/bin/env python3
import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


CURRENT_DIR = Path(__file__).resolve().parent
HRDT_ROOT = CURRENT_DIR.parents[1]
XPOLICYLAB_ROOT = CURRENT_DIR.parents[4]
DEMO_ENV_ROOT = CURRENT_DIR.parents[5]
sys.path.extend([str(HRDT_ROOT), str(XPOLICYLAB_ROOT), str(DEMO_ENV_ROOT)])

from XPolicyLab.utils.process_data import get_robot_action_dim_info, pack_robot_state


def _candidate_task_data_dirs(data_root, raw_bench_name, task_name, env_cfg_type):
    return [
        data_root / raw_bench_name / task_name / env_cfg_type / "data",
        data_root / task_name / env_cfg_type / "data",
        data_root / task_name / "data" / task_name / env_cfg_type / "data",
        data_root / task_name / "data",
    ]


def _find_hdf5_files(data_dir):
    files = sorted(data_dir.glob("episode_*.hdf5"))
    if not files:
        files = sorted(data_dir.glob("*.hdf5"))
    if not files:
        files = sorted(data_dir.glob("*.h5"))
    return files


def _resolve_task_data_dir(data_root, raw_bench_name, task_name, env_cfg_type):
    for candidate in _candidate_task_data_dirs(data_root, raw_bench_name, task_name, env_cfg_type):
        if candidate.is_dir() and _find_hdf5_files(candidate):
            return candidate
    return None


def _discover_task_names(data_root, raw_bench_name):
    search_root = data_root / raw_bench_name
    if not search_root.exists():
        search_root = data_root

    return sorted(
        path.name
        for path in search_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )


def _parse_tasks(tasks_text, data_root, raw_bench_name):
    if not tasks_text or tasks_text.strip().lower() == "all":
        return _discover_task_names(data_root, raw_bench_name)
    return [
        task.strip()
        for task in tasks_text.replace(",", " ").split()
        if task.strip()
    ]


def _load_action_data(hdf5_path, action_type, robot_action_dim_info):
    with h5py.File(hdf5_path, "r", swmr=True, libver="latest") as fp:
        action_group = fp["action"]
        action = {key: action_group[key][:] for key in action_group.keys()}
    return pack_robot_state(
        {"action": action},
        action_type,
        robot_action_dim_info,
        source_type="dataset",
        state_type="action",
    ).astype(np.float32)


def collect_actions(data_root, raw_bench_name, env_cfg_type, action_type, tasks, max_episodes):
    robot_action_dim_info = get_robot_action_dim_info(env_cfg_type)
    all_actions = []
    task_file_counts = {}
    skipped_tasks = []

    for task_name in tasks:
        data_dir = _resolve_task_data_dir(data_root, raw_bench_name, task_name, env_cfg_type)
        if data_dir is None:
            skipped_tasks.append(task_name)
            continue

        episode_files = _find_hdf5_files(data_dir)
        if max_episodes is not None:
            episode_files = episode_files[:max_episodes]

        task_file_counts[task_name] = len(episode_files)
        for hdf5_path in tqdm(episode_files, desc=f"Task {task_name}"):
            all_actions.append(_load_action_data(hdf5_path, action_type, robot_action_dim_info))

    if not all_actions:
        raise FileNotFoundError(
            f"No XPolicyLab action data found under {data_root} for tasks: {', '.join(tasks)}"
        )

    return np.concatenate(all_actions, axis=0), task_file_counts, skipped_tasks


def write_stats(output_path, actions, task_file_counts, skipped_tasks, elapsed_time, env_cfg_type, action_type):
    q01 = np.quantile(actions, 0.01, axis=0)
    q99 = np.quantile(actions, 0.99, axis=0)
    action_min = np.min(actions, axis=0)
    action_max = np.max(actions, axis=0)

    stats = {
        "xpolicylab": {
            "q01": q01.astype(float).tolist(),
            "q99": q99.astype(float).tolist(),
            "min": action_min.astype(float).tolist(),
            "max": action_max.astype(float).tolist(),
            "action_dim": int(actions.shape[1]),
            "frame_count": int(actions.shape[0]),
            "file_count": int(sum(task_file_counts.values())),
            "total_files_scanned": int(sum(task_file_counts.values())),
            "task_file_counts": task_file_counts,
            "skipped_tasks": skipped_tasks,
            "env_cfg_type": env_cfg_type,
            "action_type": action_type,
            "normalization": "clip_to_q01_q99_then_map_to_minus1_1",
            "processing_time_seconds": elapsed_time,
        }
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(stats, fp, indent=4, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Calculate XPolicyLab q01/q99 action statistics.")
    parser.add_argument("--data_root", required=True, help="XPolicyLab source dataset root.")
    parser.add_argument("--raw_bench_name", default="RoboDojo")
    parser.add_argument("--env_cfg_type", default="arx_x5")
    parser.add_argument("--action_type", default="joint")
    parser.add_argument("--tasks", default="all", help="Task names separated by comma/space, or 'all'.")
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument(
        "--output_path",
        default=str(CURRENT_DIR / "stats.json"),
        help="Output stats.json path.",
    )
    args = parser.parse_args()

    if args.action_type != "joint":
        raise ValueError("XPolicyLab H_RDT stats currently supports only action_type='joint'.")

    data_root = Path(args.data_root).expanduser().resolve()
    output_path = Path(args.output_path).expanduser()
    tasks = _parse_tasks(args.tasks, data_root, args.raw_bench_name)

    print(f"[xpolicylab] data root: {data_root}")
    print(f"[xpolicylab] raw dataset: {args.raw_bench_name}")
    print(f"[xpolicylab] env cfg: {args.env_cfg_type}")
    print(f"[xpolicylab] tasks: {', '.join(tasks)}")
    print(f"[xpolicylab] max episodes per task: {args.max_episodes}")

    start_time = time.time()
    actions, task_file_counts, skipped_tasks = collect_actions(
        data_root=data_root,
        raw_bench_name=args.raw_bench_name,
        env_cfg_type=args.env_cfg_type,
        action_type=args.action_type,
        tasks=tasks,
        max_episodes=args.max_episodes,
    )
    elapsed_time = time.time() - start_time

    write_stats(
        output_path=output_path,
        actions=actions,
        task_file_counts=task_file_counts,
        skipped_tasks=skipped_tasks,
        elapsed_time=elapsed_time,
        env_cfg_type=args.env_cfg_type,
        action_type=args.action_type,
    )

    print(f"[xpolicylab] wrote stats to: {output_path}")
    print(f"[xpolicylab] action shape: {actions.shape}")
    print(f"[xpolicylab] processed files: {sum(task_file_counts.values())}")
    if skipped_tasks:
        print(f"[xpolicylab] skipped tasks without data: {', '.join(skipped_tasks)}")


if __name__ == "__main__":
    main()
