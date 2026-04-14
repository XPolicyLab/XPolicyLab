#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Add LingBot-VA action_config fields to a LeRobot episodes.jsonl file."
    )
    parser.add_argument("--dataset-root", required=True, help="Root of the LeRobot dataset.")
    parser.add_argument(
        "--default-action-text",
        default="robot manipulation task",
        help="Fallback action text when an episode has no usable tasks field.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing action_config instead of keeping it.",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Write a .bak backup before modifying the file.",
    )
    return parser.parse_args()


def normalize_text(text):
    return " ".join(str(text).strip().split())


def build_action_text(record, default_action_text):
    tasks = record.get("tasks")

    if isinstance(tasks, list):
        for task in tasks:
            task_text = normalize_text(task)
            if task_text:
                return task_text

    if isinstance(tasks, str):
        task_text = normalize_text(tasks)
        if task_text:
            return task_text

    return normalize_text(default_action_text)


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        raise FileNotFoundError(f"episodes.jsonl not found: {episodes_path}")

    original_text = episodes_path.read_text(encoding="utf-8")
    lines = original_text.splitlines()
    updated = []

    for raw_line in lines:
        if not raw_line.strip():
            continue

        record = json.loads(raw_line)
        has_action_config = isinstance(record.get("action_config"), list) and len(record["action_config"]) > 0

        if has_action_config and not args.overwrite:
            updated.append(record)
            continue

        length = int(record.get("length", 0))
        if length <= 1:
            raise ValueError(
                f"Episode {record.get('episode_index')} has invalid length={length}; expected at least 2 frames."
            )

        record["action_config"] = [
            {
                "start_frame": 0,
                "end_frame": length,
                "action_text": build_action_text(record, args.default_action_text),
            }
        ]
        updated.append(record)

    if args.backup:
        backup_path = episodes_path.with_suffix(episodes_path.suffix + ".bak")
        backup_path.write_text(original_text, encoding="utf-8")

    content = "\n".join(json.dumps(item, ensure_ascii=False) for item in updated) + "\n"
    episodes_path.write_text(content, encoding="utf-8")

    print(f"Updated: {episodes_path}")
    print(f"Episodes processed: {len(updated)}")


if __name__ == "__main__":
    main()
