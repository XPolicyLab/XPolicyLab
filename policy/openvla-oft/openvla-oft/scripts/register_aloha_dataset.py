"""
Register an ALOHA dataset in the OpenVLA-OFT OXE config registry.

This script adds idempotent entries to:
  - prismatic/vla/datasets/rlds/oxe/configs.py
  - prismatic/vla/datasets/rlds/oxe/transforms.py
  - prismatic/vla/datasets/rlds/oxe/mixtures.py
"""

import argparse
from pathlib import Path


CONFIG_ENTRY_TEMPLATE = """    "{dataset_name}": {{
        "image_obs_keys": {{"primary": "image", "secondary": None, "left_wrist": "left_wrist_image", "right_wrist": "right_wrist_image"}},
        "depth_obs_keys": {{"primary": None, "secondary": None, "wrist": None}},
        "state_obs_keys": ["state"],
        "state_encoding": StateEncoding.JOINT_BIMANUAL,
        "action_encoding": ActionEncoding.JOINT_POS_BIMANUAL,
    }},
"""

TRANSFORM_ENTRY_TEMPLATE = '    "{dataset_name}": aloha_dataset_transform,\n'

MIXTURE_ENTRY_TEMPLATE = """    "{dataset_name}": [
        ("{dataset_name}", 1.0),
    ],
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", required=True, help="ALOHA dataset name to register.")
    parser.add_argument(
        "--repo_root",
        default="/mnt/pfs/pg4hw0/niantian/openvla-oft",
        help="Path to the openvla-oft repository root.",
    )
    return parser.parse_args()


def insert_before_last_marker(file_path: Path, marker: str, new_block: str, dataset_name: str) -> bool:
    text = file_path.read_text(encoding="utf-8")
    if f'"{dataset_name}"' in text:
        return False
    marker_idx = text.rfind(marker)
    if marker_idx == -1:
        raise ValueError(f"Marker `{marker}` not found in {file_path}")
    updated = text[:marker_idx] + new_block + text[marker_idx:]
    file_path.write_text(updated, encoding="utf-8")
    return True


def main():
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    dataset_name = args.dataset_name

    configs_path = repo_root / "prismatic" / "vla" / "datasets" / "rlds" / "oxe" / "configs.py"
    transforms_path = repo_root / "prismatic" / "vla" / "datasets" / "rlds" / "oxe" / "transforms.py"
    mixtures_path = repo_root / "prismatic" / "vla" / "datasets" / "rlds" / "oxe" / "mixtures.py"

    changed_configs = insert_before_last_marker(
        configs_path,
        "\n}",
        CONFIG_ENTRY_TEMPLATE.format(dataset_name=dataset_name),
        dataset_name,
    )
    changed_transforms = insert_before_last_marker(
        transforms_path,
        "\n}",
        TRANSFORM_ENTRY_TEMPLATE.format(dataset_name=dataset_name),
        dataset_name,
    )
    changed_mixtures = insert_before_last_marker(
        mixtures_path,
        "# fmt: on",
        MIXTURE_ENTRY_TEMPLATE.format(dataset_name=dataset_name),
        dataset_name,
    )

    print(f"dataset_name={dataset_name}")
    print(f"configs.py updated: {changed_configs}")
    print(f"transforms.py updated: {changed_transforms}")
    print(f"mixtures.py updated: {changed_mixtures}")


if __name__ == "__main__":
    main()
