#!/usr/bin/env python
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REAL_LIFT2_DIR = THIS_DIR.parent / "Real_Lift2_Example"

real_lift2_dir = str(REAL_LIFT2_DIR)
if real_lift2_dir not in sys.path:
    sys.path.insert(0, real_lift2_dir)

from model_server import main as serve_main
from model_server import parse_args as parse_shared_args


def parse_args():
    args = parse_shared_args()
    args.stats_key = args.stats_key or os.environ.get("STATS_KEY") or "real_piper"
    if args.rtc_enabled:
        raise ValueError("Real Piper example model_server_sync.py is sync-only; disable --rtc_enabled.")
    args.rtc_enabled = False
    return args


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    serve_main(parse_args())
