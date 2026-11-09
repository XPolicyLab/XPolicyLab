#!/usr/bin/env python3
"""Write RoboDojo development layouts, disjoint from the evaluation packs.

    python scripts/robodojo_make_dev_layouts.py --out /path/to/dev_layouts/arx_x5 \
        --task stack_bowls --task classify_objects --seeds 0-9

Each file is ``<out>/<task>_<seed>.json`` in the Eval_Layout format, so a worker
started with ``layout_source = "procedural"`` and ``dev_layout_dir = <out>``
replays it through RoboDojo's normal replay path. Existing files are kept, so
a layout never changes once written. The same (task, seed) always generates
the same layout. Needs numpy, shapely, rtree, transforms3d and PyYAML from the
project environment; does not start Isaac Sim.
"""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benches.robodojo.codec import ENV_CONFIG  # noqa: E402
from benches.robodojo.devlayouts import OBJECT_TYPES, dev_layout_path, ensure_dev_layout  # noqa: E402
from benches.robodojo.tasks import task_catalog, validate_task  # noqa: E402


def seeds(text):
    out = []
    for part in text.split(","):
        low, _, high = part.partition("-")
        out.extend(range(int(low), int(high or low) + 1))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkout", default=os.environ.get("HARNESS_ROBODOJO_ROOT"),
                        help="RoboDojo checkout (default: $HARNESS_ROBODOJO_ROOT)")
    parser.add_argument("--out", required=True, help="dev layout directory")
    parser.add_argument("--task", action="append", required=True, help="task name; repeat for several")
    parser.add_argument("--seeds", type=seeds, default=seeds("0-9"), help="e.g. 0-9 or 0,3,5 (default 0-9)")
    args = parser.parse_args()
    if not args.checkout:
        parser.error("--checkout or HARNESS_ROBODOJO_ROOT is required")
    checkout = Path(args.checkout).absolute()
    catalog = task_catalog(checkout)
    failed = 0
    for task in args.task:
        validate_task(task, catalog)
        for seed in args.seeds:
            existed = dev_layout_path(args.out, task, seed).exists()
            try:
                path = ensure_dev_layout(checkout, ENV_CONFIG, args.out, task, seed)
            except Exception as error:
                failed += 1
                print(f"FAIL {task} {seed}: {type(error).__name__}: {error}", flush=True)
                continue
            layout = json.loads(path.read_text())
            count = sum(len(v) for kind in OBJECT_TYPES for v in (layout.get(kind) or {}).values())
            print(f"{'kept' if existed else 'wrote'} {path} objects={count}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
