#!/usr/bin/env python3
"""Generate a Dexbotic data_source registration file for a converted dataset."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=str, help="Converted Dexdata root directory")
    parser.add_argument("data_key", type=str, help="Dataset key, e.g. RoboDojo-cotrain-arx_x5-3500-joint")
    parser.add_argument(
        "output_path",
        type=str,
        help="Output .py path under dexbotic/dexbotic/data/data_source/",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    content = f'''"""Auto-generated Dexbotic data source for {args.data_key}."""

from dexbotic.data.data_source.register import register_dataset

ROBODOJO_DATASET = {{
    "{args.data_key}": {{
        "data_path_prefix": "{data_root / "video"}",
        "annotations": "{data_root}",
        "frequency": 1,
    }},
}}

meta_data = {{
    "non_delta_mask": [6, 20],
    "periodic_mask": None,
    "periodic_range": None,
}}

register_dataset(ROBODOJO_DATASET, meta_data=meta_data, prefix="robodojo")
'''
    output_path.write_text(content, encoding="utf-8")
    registered_name = f"robodojo_{args.data_key}"
    print(f"Wrote {output_path}")
    print(f"Registered dataset name: {registered_name}")


if __name__ == "__main__":
    main()
