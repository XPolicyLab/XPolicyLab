import argparse
import csv
from pathlib import Path

import h5py


def _decode_instruction(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "item"):
        value = value.item()
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
    return str(value)


def _task_name_from_episode(data_root, episode_path, env_cfg_type):
    relative_parts = episode_path.relative_to(data_root).parts
    if env_cfg_type in relative_parts:
        env_index = relative_parts.index(env_cfg_type)
        if env_index > 0:
            return relative_parts[env_index - 1]
    return relative_parts[0]


def extract_task_instructions(data_root, env_cfg_type):
    data_root = Path(data_root).expanduser().resolve()
    task_instructions = {}

    for episode_path in sorted(data_root.rglob("episode_*.hdf5")):
        task_name = _task_name_from_episode(data_root, episode_path, env_cfg_type)
        if task_name in task_instructions:
            continue

        with h5py.File(episode_path, "r") as fp:
            if "instruction" not in fp:
                raise KeyError(f"Missing instruction in {episode_path}")
            task_instructions[task_name] = _decode_instruction(fp["instruction"][()])

    if not task_instructions:
        raise FileNotFoundError(f"No episode_*.hdf5 files found under {data_root}")

    return task_instructions


def write_task_instructions(task_instructions, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["task_name", "instruction"])
        writer.writeheader()
        for task_name, instruction in sorted(task_instructions.items()):
            writer.writerow({"task_name": task_name, "instruction": instruction})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root")
    parser.add_argument("--env_cfg_type", default="arx_x5")
    parser.add_argument(
        "--output",
        default="task_instructions.csv",
        help="Output CSV path. Defaults to datasets/xpolicylab/task_instructions.csv.",
    )
    args = parser.parse_args()

    task_instructions = extract_task_instructions(args.data_root, args.env_cfg_type)
    write_task_instructions(task_instructions, args.output)

    print(f"[xpolicylab] wrote {len(task_instructions)} task instructions to {args.output}")


if __name__ == "__main__":
    main()
