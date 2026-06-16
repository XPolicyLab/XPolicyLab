"""
Generate Mem_0 inference normalization stats from a LeRobot dataset.

Run in the Mem_0 policy conda env:

    python gen_norm_stats.py --repo_id <data/...-lerobot> --ckpt_name <ckpt_name>

Writes policy/Mem_0/assets/<ckpt_name>/norm_stats.json (and global_instruction.txt).
Falls back to Mem_0/assets/<ckpt_name>/ when MEM0_LEGACY_PATHS=1.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ADAPTER_DIR = os.path.dirname(os.path.abspath(__file__))
UPSTREAM_DIR = os.path.dirname(ADAPTER_DIR)
POLICY_DIR = os.path.dirname(UPSTREAM_DIR)
if UPSTREAM_DIR not in sys.path:
    sys.path.insert(0, UPSTREAM_DIR)

from source.dataloader.dataset_min_max import LeRobot_Dataset  # noqa: E402


def _format_list_fixed_10(values) -> str:
    import numpy as np

    flat = np.asarray(values).reshape(-1).tolist()
    return "[" + ", ".join(f"{float(v):.10f}" for v in flat) + "]"


def _resolve_assets_dir(ckpt_name: str) -> str:
    if os.environ.get("MEM0_LEGACY_PATHS") == "1":
        return os.path.join(UPSTREAM_DIR, "assets", ckpt_name)
    return os.path.join(POLICY_DIR, "assets", ckpt_name)


def save_norm_stats_to_policy_assets(dataset: LeRobot_Dataset, ckpt_name: str, lang: str) -> str:
    target_dir = _resolve_assets_dir(ckpt_name)
    os.makedirs(target_dir, exist_ok=True)
    norm_path = os.path.join(target_dir, "norm_stats.json")
    lang_path = os.path.join(target_dir, "global_instruction.txt")

    stats = dataset.dataset.meta.stats
    state_min = stats["observation.state"]["min"]
    state_max = stats["observation.state"]["max"]
    action_min = stats["action"]["min"]
    action_max = stats["action"]["max"]

    with open(lang_path, "w", encoding="utf-8") as f:
        f.write(lang)

    with open(norm_path, "w", encoding="utf-8") as f:
        f.write("{\n")
        f.write(f'  "state_min": {_format_list_fixed_10(state_min)},\n')
        f.write(f'  "state_max": {_format_list_fixed_10(state_max)},\n')
        f.write(f'  "action_min": {_format_list_fixed_10(action_min)},\n')
        f.write(f'  "action_max": {_format_list_fixed_10(action_max)}\n')
        f.write("}\n")

    return norm_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Mem_0 norm_stats.json")
    parser.add_argument("--repo_id", required=True, help="path to the LeRobot dataset")
    parser.add_argument("--ckpt_name", required=True, help="asset folder name under policy/Mem_0/assets/")
    args = parser.parse_args()

    dataset = LeRobot_Dataset(
        repo_id=str(os.path.expanduser(args.repo_id)),
        features_to_load=[
            "observation.image.head_camera",
            "observation.state",
            "action",
            "subtask",
            "subtask_end",
            "episode_id",
        ],
    )
    sample = dataset[0]
    norm_path = save_norm_stats_to_policy_assets(dataset, args.ckpt_name, sample["lang"])
    print(f"[norm] instruction: {sample['lang']}")
    print(f"[norm] saved norm stats -> {norm_path}")


if __name__ == "__main__":
    main()
