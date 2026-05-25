#!/usr/bin/env python3
"""Compare loss curves between LoRA and full fine-tuning runs.

Usage:
    python scripts/compare_loss.py \
        --lora-log ./checkpoints/dreamzero_droid_lora/loss_log.jsonl \
        --full-log ./checkpoints/dreamzero_droid_full_finetune/loss_log.jsonl \
        [--plot loss_comparison.png]
"""

import argparse
import json


def load_loss_log(path):
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def print_comparison_table(lora_entries, full_entries):
    # Index by step
    lora_by_step = {e["step"]: e for e in lora_entries}
    full_by_step = {e["step"]: e for e in full_entries}
    all_steps = sorted(set(lora_by_step.keys()) | set(full_by_step.keys()))

    header = f"{'Step':>6}  {'LoRA Loss':>10}  {'Full Loss':>10}  {'LoRA Dyn':>10}  {'Full Dyn':>10}  {'LoRA Act':>10}  {'Full Act':>10}"
    print(header)
    print("-" * len(header))

    for step in all_steps:
        lora = lora_by_step.get(step, {})
        full = full_by_step.get(step, {})

        def fmt(d, key):
            v = d.get(key)
            return f"{v:10.4f}" if v is not None else f"{'—':>10}"

        print(
            f"{step:>6}  "
            f"{fmt(lora, 'loss')}  {fmt(full, 'loss')}  "
            f"{fmt(lora, 'dynamics_loss_avg')}  {fmt(full, 'dynamics_loss_avg')}  "
            f"{fmt(lora, 'action_loss_avg')}  {fmt(full, 'action_loss_avg')}"
        )


def plot_comparison(lora_entries, full_entries, output_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot generation.")
        print("Install with: pip install matplotlib")
        return

    metrics = [
        ("loss", "Total Loss"),
        ("dynamics_loss_avg", "Dynamics Loss"),
        ("action_loss_avg", "Action Loss"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]

    for ax, (key, title) in zip(axes, metrics):
        lora_steps = [e["step"] for e in lora_entries if key in e]
        lora_vals = [e[key] for e in lora_entries if key in e]
        full_steps = [e["step"] for e in full_entries if key in e]
        full_vals = [e[key] for e in full_entries if key in e]

        if lora_steps:
            ax.plot(lora_steps, lora_vals, label="LoRA", marker="o", markersize=3)
        if full_steps:
            ax.plot(full_steps, full_vals, label="Full FT", marker="s", markersize=3)

        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare LoRA vs full fine-tuning loss curves")
    parser.add_argument("--lora-log", required=True, help="Path to LoRA run loss_log.jsonl")
    parser.add_argument("--full-log", required=True, help="Path to full FT run loss_log.jsonl")
    parser.add_argument("--plot", default=None, help="Output path for comparison plot (e.g., loss_comparison.png)")
    args = parser.parse_args()

    lora_entries = load_loss_log(args.lora_log)
    full_entries = load_loss_log(args.full_log)

    print(f"LoRA: {len(lora_entries)} log entries")
    print(f"Full: {len(full_entries)} log entries")
    print()

    print_comparison_table(lora_entries, full_entries)

    if args.plot:
        plot_comparison(lora_entries, full_entries, args.plot)


if __name__ == "__main__":
    main()
