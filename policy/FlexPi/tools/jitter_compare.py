"""Compare action-chunk jitter between checkpoint steps from FLEXPI_DUMP_ACTIONS dumps.

    python jitter_compare.py <OUT>          # <OUT>/step<N>/actions/ep<NNN>_env<i>.npz

Joint units are rad, one step = one 25 Hz action. Metrics, all over the 12 arm joints:

  intra-chunk (every predicted chunk, all 32 steps -- what the model asked for)
    vel        mean |d a|                      per-step motion, the scale reference
    acc        mean |d2 a|                     raw jerkiness
    rough      acc / vel                       jerkiness per unit motion (speed-invariant)
    wiggle     RMS residual after a cubic fit  high-frequency content inside a chunk
    flips      velocity sign reversals per joint per 100 steps (|d a| > 1e-3 on both sides)

  chunk boundary (consecutive executed chunks)
    jump       mean |a_k[0] - a_{k-1}[last executed]|
    jump/vel   the same in units of a typical within-chunk step (1 = seamless)

  executed (measured joint state after every step)
    exec_acc   mean |d2 q| over the episode
    bnd/int    mean |d2 q| at the 2 steps around a chunk boundary / elsewhere

  gripper
    grip_flips command direction reversals per gripper per 100 steps (|d g| > 0.02)
"""

import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

JOINTS14 = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
GRIPS14 = [6, 13]
Q32 = slice(20, 32)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3, fixed order
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def _flips(x: np.ndarray, eps: float) -> int:
    """Direction reversals along axis 0 of a [T, J] velocity array, both sides above eps."""
    s = np.sign(np.where(np.abs(x) > eps, x, 0.0))
    return int(((s[1:] * s[:-1]) < 0).sum())


def episode_metrics(d) -> dict:
    rd, at, st = d["rd14"], d["chunk_at"], d["states"]
    K, T, _ = rd.shape
    a = rd[..., JOINTS14]                                  # [K, T, 12]
    da, d2a = np.diff(a, axis=1), np.diff(a, n=2, axis=1)
    t = np.arange(T)
    wig = []
    for k in range(K):
        coef = np.polyfit(t, a[k], 3)                      # [4, 12]
        fit = np.stack([np.polyval(coef[:, j], t) for j in range(a.shape[-1])], axis=1)
        wig.append(np.sqrt(((a[k] - fit) ** 2).mean()))
    vel = float(np.abs(da).mean())

    # executed chunks: chunk k ran for chunk_at[k+1] - chunk_at[k] steps (the last one's
    # length is unknown -- the dump ends at the last inference -- so it is excluded)
    jumps = []
    for k in range(1, K):
        n_prev = int(at[k] - at[k - 1])
        if 1 <= n_prev <= T:
            jumps.append(np.abs(a[k, 0] - a[k - 1, n_prev - 1]).mean())

    q = st[:, Q32]
    d2q = np.abs(np.diff(q, n=2, axis=0)).mean(axis=1)    # d2q[i] is centred on state i+1
    bnd = np.zeros(len(d2q), dtype=bool)
    for k in range(1, K):
        for c in (at[k] - 1, at[k]):                       # the two steps straddling the switch
            if 0 <= c < len(bnd):
                bnd[c] = True
    g = rd[..., GRIPS14]
    return {
        "chunks": int(K),
        "steps": int(len(st)),
        "vel": vel,
        "acc": float(np.abs(d2a).mean()),
        "rough": float(np.abs(d2a).mean() / max(vel, 1e-9)),
        "wiggle": float(np.mean(wig)),
        "flips": 100.0 * sum(_flips(da[k], 1e-3) for k in range(K)) / (K * (T - 1) * len(JOINTS14)),
        "jump": float(np.mean(jumps)) if jumps else float("nan"),
        "jump/vel": float(np.mean(jumps) / max(vel, 1e-9)) if jumps else float("nan"),
        "exec_acc": float(d2q.mean()) if len(d2q) else float("nan"),
        "bnd/int": float(d2q[bnd].mean() / max(d2q[~bnd].mean(), 1e-9)) if bnd.any() and (~bnd).any() else float("nan"),
        "grip_flips": 100.0 * sum(_flips(np.diff(g[k], axis=0), 0.02) for k in range(K)) / (K * (T - 1) * len(GRIPS14)),
    }


def load(out: Path):
    runs = {}
    for step_dir in sorted(out.glob("step*"), key=lambda p: int(p.name[4:])):
        files = sorted((step_dir / "actions").glob("ep*_env*.npz"))
        if files:
            runs[step_dir.name] = {f.stem: dict(np.load(f)) for f in files}
    return runs


def results_json(out: Path, bench_root: Path):
    res = {}
    for step_dir in out.glob("step*"):
        rid_file = step_dir / "run_id"
        if not rid_file.exists():
            continue
        rid = rid_file.read_text().strip()
        for f in glob.glob(str(bench_root / f"eval_result/RoboDojo/*/FlexPi/*/*/{rid}_*/_result.json")):
            task = Path(f).parent.name[len(rid) + 1:]
            r = json.loads(Path(f).read_text())
            res[(step_dir.name, task)] = (r["success_rate"], r["score"])
    return res


def plot(runs, out: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = list(runs)
    episodes = sorted({ep for r in runs.values() for ep in r})
    names = ["L_j0", "L_j1", "L_j2", "L_j3", "L_j4", "L_j5", "R_j0", "R_j1", "R_j2", "R_j3", "R_j4", "R_j5"]

    def style(ax):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color="#e6e5e0", linewidth=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=7)

    for ep in episodes:
        # 1) commanded targets as executed, per joint, one colour per ckpt; chunk starts dotted
        fig, axes = plt.subplots(6, 2, figsize=(14, 13), sharex=True, facecolor=SURFACE)
        for si, step in enumerate(steps):
            d = runs[step].get(ep)
            if d is None:
                continue
            rd, at = d["rd14"], d["chunk_at"]
            for j, col in enumerate(JOINTS14):
                ax = axes[j % 6, j // 6]
                for k in range(len(at)):
                    n = int(at[k + 1] - at[k]) if k + 1 < len(at) else rd.shape[1]
                    xs = at[k] + 1 + np.arange(n)
                    ax.plot(xs, rd[k, :n, col], color=COLORS[si], linewidth=1.3,
                            label=step if (j == 0 and k == 0) else None)
                    ax.axvline(at[k], color=COLORS[si], linewidth=0.4, alpha=0.35, linestyle=":")
        for j in range(12):
            style(axes[j % 6, j // 6])
            axes[j % 6, j // 6].set_title(names[j] + " (rad)", fontsize=8, color=INK, loc="left")
        axes[0, 0].legend(fontsize=8, frameon=False)
        for ax in axes[-1]:
            ax.set_xlabel("step (25 Hz)", fontsize=8, color=MUTED)
        instr = str(next(iter(runs.values())).get(ep, {}).get("instruction", ""))[:70]
        fig.suptitle(f"{ep}: executed joint targets, chunk starts dotted -- {instr}", fontsize=10, color=INK)
        fig.tight_layout()
        fig.savefig(out / f"targets_{ep}.png", dpi=100)
        plt.close(fig)

        # 2) |d2 q| of the measured state over time: jitter shows as spikes, boundary spikes line up with dots
        fig, axes = plt.subplots(len(steps), 1, figsize=(14, 2.6 * len(steps)), sharex=True, sharey=True,
                                 facecolor=SURFACE, squeeze=False)
        for si, step in enumerate(steps):
            ax = axes[si, 0]
            style(ax)
            d = runs[step].get(ep)
            if d is None:
                continue
            d2q = np.abs(np.diff(d["states"][:, Q32], n=2, axis=0)).mean(axis=1)
            ax.plot(np.arange(1, len(d2q) + 1), d2q, color=COLORS[si], linewidth=1.0)
            for c in d["chunk_at"][1:]:
                ax.axvline(c, color=MUTED, linewidth=0.5, alpha=0.5, linestyle=":")
            ax.set_title(f"{step}: mean |d2 q| of measured joints (rad/step^2), chunk switches dotted",
                         fontsize=8, color=INK, loc="left")
        axes[-1, 0].set_xlabel("step (25 Hz)", fontsize=8, color=MUTED)
        fig.tight_layout()
        fig.savefig(out / f"exec_accel_{ep}.png", dpi=100)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--bench-root", default=str(Path(__file__).resolve().parents[4]),
                    help="benchmark checkout holding eval_result/ (default: the one this adapter sits in)")
    args = ap.parse_args()
    out = Path(args.out)
    runs = load(out)
    if not runs:
        raise SystemExit(f"no dumps under {out}/step*/actions")

    keys = ["chunks", "vel", "acc", "rough", "wiggle", "flips", "jump", "jump/vel", "exec_acc", "bnd/int", "grip_flips"]
    table = {step: {ep: episode_metrics(d) for ep, d in eps.items()} for step, eps in runs.items()}
    res = results_json(out, Path(args.bench_root))

    print(f"{'step':>11} {'episode':>12} " + " ".join(f"{k:>10}" for k in keys) + "  instruction")
    agg = defaultdict(list)
    for step, eps in table.items():
        for ep, m in eps.items():
            agg[step].append(m)
            instr = str(runs[step][ep]["instruction"])[:40]
            print(f"{step:>11} {ep:>12} " + " ".join(f"{m[k]:>10.4f}" for k in keys) + f"  {instr}")
    print()
    print(f"{'step':>11} {'MEAN':>12} " + " ".join(f"{k:>10}" for k in keys))
    summary = {}
    for step, ms in agg.items():
        mean = {k: float(np.nanmean([m[k] for m in ms])) for k in keys}
        summary[step] = mean
        print(f"{step:>11} {'':>12} " + " ".join(f"{mean[k]:>10.4f}" for k in keys))
    if res:
        print()
        for (step, task), (sr, score) in sorted(res.items()):
            print(f"{step:>11} {task:>24}  success_rate {sr:.2f}  score {score}")

    (out / "jitter_metrics.json").write_text(json.dumps(
        {"per_episode": table, "mean": summary,
         "results": {f"{s}/{t}": v for (s, t), v in res.items()}}, indent=1))
    plot(runs, out)
    print(f"\nwrote {out}/jitter_metrics.json, targets_*.png, exec_accel_*.png")


if __name__ == "__main__":
    main()
