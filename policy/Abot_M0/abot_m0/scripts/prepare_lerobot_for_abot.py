#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


REQUIRED_META_FILES = (
    "meta/info.json",
    "meta/modality.json",
    "meta/episodes.jsonl",
    "meta/tasks.jsonl",
)

REQUIRED_STATE_KEYS = (
    "left_joints",
    "right_joints",
    "left_gripper",
    "right_gripper",
)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def save_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def ensure_required_files(dataset_dir: Path) -> None:
    missing = [rel for rel in REQUIRED_META_FILES if not (dataset_dir / rel).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")


def ensure_info_has_paths(info: dict) -> None:
    for key in ("data_path", "video_path"):
        if key not in info:
            raise ValueError(f"meta/info.json is missing `{key}`")


def ensure_video_dirs(dataset_dir: Path, video_keys: list[str]) -> None:
    missing = []
    for video_key in video_keys:
        rel = Path("videos/chunk-000") / video_key
        if not (dataset_dir / rel).exists():
            missing.append(rel.as_posix())
    if missing:
        raise FileNotFoundError(f"Missing video directories: {missing}")


def ensure_state_action_layout(modality: dict) -> None:
    for branch in ("state", "action"):
        if branch not in modality:
            raise ValueError(
                f"meta/modality.json is missing `{branch}`. "
                "The quick-start path expects robotwin-like state/action slices."
            )
        missing = [key for key in REQUIRED_STATE_KEYS if key not in modality[branch]]
        if missing:
            raise ValueError(
                f"meta/modality.json missing {branch} keys: {missing}. "
                "Expected keys are left_joints, right_joints, left_gripper, right_gripper."
            )


def update_video_and_language_mappings(
    dataset_dir: Path,
    cam_high_key: str,
    cam_left_key: str,
    cam_right_key: str,
    annotation_source_key: str,
) -> None:
    modality_path = dataset_dir / "meta/modality.json"
    modality = load_json(modality_path)

    ensure_state_action_layout(modality)

    modality.setdefault("video", {})
    modality["video"]["cam_high"] = {"original_key": cam_high_key}
    modality["video"]["cam_left_wrist"] = {"original_key": cam_left_key}
    modality["video"]["cam_right_wrist"] = {"original_key": cam_right_key}

    modality.setdefault("annotation", {})
    modality["annotation"]["human.action.task_description"] = {
        "original_key": annotation_source_key
    }
    save_json(modality_path, modality)


def optionally_normalize_task_text(dataset_dir: Path, task_text: str | None) -> None:
    if not task_text:
        return

    tasks_path = dataset_dir / "meta/tasks.jsonl"
    tasks = load_jsonl(tasks_path)
    if not tasks:
        tasks = [{"task_index": 0, "task": task_text}]
    else:
        for row in tasks:
            row["task"] = task_text
    save_jsonl(tasks_path, tasks)

    episodes_path = dataset_dir / "meta/episodes.jsonl"
    episodes = load_jsonl(episodes_path)
    updated = False
    for row in episodes:
        if row.get("tasks") != [task_text]:
            row["tasks"] = [task_text]
            updated = True
        for action_cfg in row.get("action_config", []):
            if action_cfg.get("action_text") != task_text:
                action_cfg["action_text"] = task_text
                updated = True
    if updated:
        save_jsonl(episodes_path, episodes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Path to a LeRobot dataset root")
    parser.add_argument("--task", type=str, default=None, help="Optional single task text to write into tasks.jsonl and episodes.jsonl")
    parser.add_argument("--cam-high-key", type=str, default="observation.images.cam_high")
    parser.add_argument("--cam-left-key", type=str, default="observation.images.cam_left_wrist")
    parser.add_argument("--cam-right-key", type=str, default="observation.images.cam_right_wrist")
    parser.add_argument("--annotation-source-key", type=str, default="task_index")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    ensure_required_files(dataset_dir)
    info = load_json(dataset_dir / "meta/info.json")
    ensure_info_has_paths(info)
    ensure_video_dirs(dataset_dir, [args.cam_high_key, args.cam_left_key, args.cam_right_key])
    update_video_and_language_mappings(
        dataset_dir,
        cam_high_key=args.cam_high_key,
        cam_left_key=args.cam_left_key,
        cam_right_key=args.cam_right_key,
        annotation_source_key=args.annotation_source_key,
    )
    optionally_normalize_task_text(dataset_dir, args.task)

    print(f"Dataset ready for ABot quick-start training: {dataset_dir}")
    print("Verified meta files, video layout, and robotwin-style modality mapping.")


if __name__ == "__main__":
    main()