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

REQUIRED_VIDEO_DIRS = (
    "videos/chunk-000/observation.images.cam_high",
    "videos/chunk-000/observation.images.cam_left_wrist",
    "videos/chunk-000/observation.images.cam_right_wrist",
)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _save_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_dataset(dataset_dir: Path) -> None:
    missing = [rel for rel in REQUIRED_META_FILES if not (dataset_dir / rel).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required metadata files: {missing}")

    missing_video_dirs = [rel for rel in REQUIRED_VIDEO_DIRS if not (dataset_dir / rel).exists()]
    if missing_video_dirs:
        raise FileNotFoundError(f"Missing required video directories: {missing_video_dirs}")

    info = _load_json(dataset_dir / "meta/info.json")
    if "video_path" not in info:
        raise ValueError("meta/info.json is missing `video_path`")
    if "data_path" not in info:
        raise ValueError("meta/info.json is missing `data_path`")


def normalize_modality(dataset_dir: Path) -> None:
    modality_path = dataset_dir / "meta/modality.json"
    modality = _load_json(modality_path)

    modality.setdefault("video", {})
    modality["video"].setdefault("cam_high", {"original_key": "observation.images.cam_high"})
    modality["video"].setdefault("cam_left_wrist", {"original_key": "observation.images.cam_left_wrist"})
    modality["video"].setdefault("cam_right_wrist", {"original_key": "observation.images.cam_right_wrist"})

    modality.setdefault("annotation", {})
    modality["annotation"].setdefault(
        "human.action.task_description",
        {"original_key": "task_index"},
    )

    _save_json(modality_path, modality)


def normalize_task_text(dataset_dir: Path, task_text: str) -> None:
    tasks_path = dataset_dir / "meta/tasks.jsonl"
    episodes_path = dataset_dir / "meta/episodes.jsonl"

    tasks = _load_jsonl(tasks_path)
    if not tasks:
        tasks = [{"task_index": 0, "task": task_text}]
    else:
        for row in tasks:
            row["task"] = task_text
    _save_jsonl(tasks_path, tasks)

    episodes = _load_jsonl(episodes_path)
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
        _save_jsonl(episodes_path, episodes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Path to a LeRobot dataset root")
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Optional single task text. If set, overwrites all entries in tasks.jsonl and episodes.jsonl.",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    validate_dataset(dataset_dir)
    normalize_modality(dataset_dir)
    if args.task is not None:
        normalize_task_text(dataset_dir, args.task)

    print(f"Dataset ready for ABot training: {dataset_dir}")
    if args.task is not None:
        print("Verified metadata and video layout; task text and modality mapping have been normalized.")
    else:
        print("Verified metadata and video layout; existing task text preserved; modality mapping normalized.")


if __name__ == "__main__":
    main()