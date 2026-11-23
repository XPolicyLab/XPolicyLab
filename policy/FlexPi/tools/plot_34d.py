"""Re-plot 34D GT vs predicted trajectories from a finished openloop_replay.py run.

    python plot_34d.py <out_dir>/ep001200 [<out_dir>/ep003000 ...]

Needs only numpy/pandas/matplotlib/av (no model, no GPU).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from openloop_replay import DEFAULT_DATASET, Episode, plot_34d  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ep_dirs", nargs="+")
    ap.add_argument("--dataset-root", default=str(DEFAULT_DATASET))
    ap.add_argument("--every", type=int, default=32, help="cadence for the second, non-overlapping plot")
    args = ap.parse_args()
    for ep_dir in map(Path, args.ep_dirs):
        res = np.load(ep_dir / "replay.npz")
        ep = Episode(Path(args.dataset_root), int(ep_dir.name.removeprefix("ep")))
        ts, pred32, valid = res["t"].tolist(), res["pred32"], res["valid"].tolist()
        plot_34d(ep, ts, pred32, valid, ep_dir / "gt_vs_pred_34d.png")
        plot_34d(ep, ts, pred32, valid, ep_dir / f"gt_vs_pred_34d_every{args.every}.png", every=args.every)
        print(f"wrote {ep_dir}/gt_vs_pred_34d.png, gt_vs_pred_34d_every{args.every}.png")


if __name__ == "__main__":
    main()
