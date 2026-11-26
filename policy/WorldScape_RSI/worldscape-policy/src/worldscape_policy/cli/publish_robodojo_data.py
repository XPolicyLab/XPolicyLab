"""Publish append-only RoboDojo snapshots for a running training job."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

from worldscape_policy.data.robodojo_snapshots import (
    DEFAULT_PUBLISH_DIR, DEFAULT_WATCH_ROOT, SnapshotStore,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-index", required=True, type=Path)
    parser.add_argument("--watch-root", action="append", type=Path,
                        help="Repeatable producer directories; files must be closed before atomic final publication")
    parser.add_argument("--publish-dir", type=Path, default=DEFAULT_PUBLISH_DIR)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="Publish one update and exit (default)")
    mode.add_argument("--watch", action="store_true", help="Continue polling for finalized episodes")
    args = parser.parse_args(argv)
    if not math.isfinite(args.poll_seconds) or args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    store = SnapshotStore(args.publish_dir)
    try:
        with store.publisher_lock():
            initial = store.initialize(args.base_index)
            print(json.dumps({"event": "initialized", "descriptor": initial}), flush=True)
            while True:
                result = store.publish(args.watch_root or [DEFAULT_WATCH_ROOT])
                print(json.dumps({**result, "rejected_count": len(result["rejected"]),
                                  "rejected": result["rejected"][:32]}), flush=True)
                if not args.watch:
                    return 0
                time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
