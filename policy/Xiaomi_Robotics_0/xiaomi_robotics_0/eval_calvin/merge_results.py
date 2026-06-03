# Copyright (C) 2026 Xiaomi Corporation.
import dataclasses
import itertools
import logging
import os
import pickle
import sys
import json

from pathlib import Path
from typing import Optional, Tuple

import tyro

from calvin_agent.evaluation.utils import print_and_save


def get_logger(name: str) -> logging.Logger:
    """Create a simple console logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


logger = get_logger(__name__)


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # IO / logging settings (core settings)
    #################################################################################################################
    eval_log_dir: str = None  # REQUIRED: User-specified path to eval log directory (no default)
    save_file: str = "results_calvin_merged.json"  # Filename for merged evaluation results.

    #################################################################################################################
    # Optional settings
    #################################################################################################################
    world_size: Optional[int] = None  # Expected number of ranks. If None, determined from existing pkl files.


def merge_results(args: Args):
    """Merge per-rank pickle results and compute final metrics.

    This function loads all `rank_{rank}_results.pkl` files from the
    user-specified eval_log_dir, concatenates results and sequences across ranks,
    and calls `print_and_save` to compute and store the final aggregated evaluation metrics.

    Args:
        args: Parsed dataclass arguments.
    """
    eval_log_dir = args.eval_log_dir
    eval_log_path = Path(eval_log_dir)
    result_path = eval_log_path

    if not eval_log_path.exists():
        logger.error("Eval log dir %s does not exist, merge aborted.", eval_log_dir)
        return

    # Find all rank result files like rank_0_results.pkl, rank_1_results.pkl, ...
    pkl_files = sorted(eval_log_path.glob("rank_*_results.pkl"))

    if not pkl_files:
        logger.error("No rank_*_results.pkl files found in %s, merge aborted.", eval_log_dir)
        return

    if args.world_size is not None and len(pkl_files) != args.world_size:
        logger.warning("Found %d pkl files, but world_size=%d.", len(pkl_files), args.world_size)

    all_results = []
    all_eval_sequences = []

    logger.info("Merging %d rank result files under %s", len(pkl_files), eval_log_dir)

    for pkl_path in pkl_files:
        logger.info("Loading %s", pkl_path)
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        if "results" not in data or "sequences" not in data:
            logger.warning(
                "File %s missing 'results' or 'sequences' keys, skip.",
                pkl_path,
            )
            continue

        all_results.append(data["results"])
        all_eval_sequences.append(data["sequences"])

    if not all_results:
        logger.error("No valid data loaded from rank result files, merge aborted.")
        return

    # Flatten list-of-lists
    flat_results = list(itertools.chain.from_iterable(all_results))
    flat_sequences = list(itertools.chain.from_iterable(all_eval_sequences))

    logger.info(
        "Evaluating gathered results with %d samples in total.",
        len(flat_results),
    )

    # Compute overall metrics and save to result_path
    print_and_save(flat_results, flat_sequences, result_path, None)

    logger.info("Merged results saved to %s", result_path)


if __name__ == "__main__":
    args = tyro.cli(Args)

    if args.eval_log_dir is None:
        raise ValueError("Please specify eval_log_dir")

    merge_results(args)
