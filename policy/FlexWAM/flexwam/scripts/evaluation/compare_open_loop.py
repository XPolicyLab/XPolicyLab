#!/usr/bin/env python3
"""Compare two one-sample open-loop dumps."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load(path: Path):
    data = np.load(path)
    summary = json.loads(path.with_suffix(".json").read_text())
    return data, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True, help="Left open-loop .npz dump.")
    parser.add_argument("--right", type=Path, required=True, help="Right open-loop .npz dump.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Comparison JSON path. Defaults to <left-parent>/compare.json.",
    )
    args = parser.parse_args()
    left_path = args.left
    right_path = args.right
    left, left_s = _load(left_path)
    right, right_s = _load(right_path)

    pred_l = left["pred_action"].astype(np.float64)
    pred_r = right["pred_action"].astype(np.float64)
    gt_l = left["gt_action"].astype(np.float64)
    gt_r = right["gt_action"].astype(np.float64)
    norm_l = left["pred_normed"].astype(np.float64)
    norm_r = right["pred_normed"].astype(np.float64)

    pred_diff = np.abs(pred_l - pred_r)
    gt_diff = np.abs(gt_l - gt_r)
    norm_diff = np.abs(norm_l - norm_r)
    report = {
        "left": left_s,
        "right": right_s,
        "same_prompt": left_s.get("prompt") == right_s.get("prompt"),
        "gt_max_abs": float(gt_diff.max()),
        "gt_mean_abs": float(gt_diff.mean()),
        "pred_max_abs": float(pred_diff.max()),
        "pred_mean_abs": float(pred_diff.mean()),
        "pred_rms": float(np.sqrt(np.square(pred_l - pred_r).mean())),
        "pred_allclose_1e5": bool(np.allclose(pred_l, pred_r, rtol=1e-5, atol=1e-5)),
        "pred_allclose_1e3": bool(np.allclose(pred_l, pred_r, rtol=1e-3, atol=1e-3)),
        "pred_allclose_1e2": bool(np.allclose(pred_l, pred_r, rtol=1e-2, atol=1e-2)),
        "normed_max_abs": float(norm_diff.max()),
        "normed_mean_abs": float(norm_diff.mean()),
        "mae_delta": float(left_s["mae"] - right_s["mae"]),
        "first_step_l2_delta": float(left_s["first_step_l2"] - right_s["first_step_l2"]),
    }
    out = args.output if args.output is not None else left_path.parent / "compare.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
