"""
Render a ready-to-run execution-module training config from the committed
template, filling the fields the upstream README asks you to edit by hand
(vla_dataset.RMBench.repo_id, trainer.checkpoint_dir, trainer.wandb_run_name,
batch_size, train_steps, seed). The committed template is never mutated; the
resolved config is written next to the run's checkpoints.

Used by ../../train.sh (train_module=execution|both); can also be run standalone.
"""

import argparse
import os
import sys

from omegaconf import OmegaConf

ADAPTER_DIR = os.path.dirname(os.path.abspath(__file__))
UPSTREAM_DIR = os.path.dirname(ADAPTER_DIR)
TEMPLATE = os.path.join(UPSTREAM_DIR, "source", "config", "execution_module_train.yaml")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Mem_0 execution-module train config")
    parser.add_argument("--repo_id", required=True, help="LeRobot dataset path")
    parser.add_argument("--checkpoint_dir", required=True, help="checkpoint dir (relative to upstream root)")
    parser.add_argument("--wandb_run_name", required=True)
    parser.add_argument("--out", required=True, help="output config path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=56)
    parser.add_argument("--train_steps", type=int, default=30000)
    parser.add_argument("--norm_stats_path", default=None)
    parser.add_argument("--enable_wandb", default="true")
    parser.add_argument("--is_debug", default="false")
    args = parser.parse_args()

    def _b(v):
        return str(v).strip().lower() in {"1", "true", "yes", "t"}

    cfg = OmegaConf.load(TEMPLATE)
    cfg.seed = args.seed
    cfg.is_debug = _b(args.is_debug)
    cfg.trainer.checkpoint_dir = args.checkpoint_dir
    cfg.trainer.wandb_run_name = args.wandb_run_name
    cfg.trainer.batch_size = args.batch_size
    cfg.trainer.train_steps = args.train_steps
    cfg.trainer.enable_wandb = _b(args.enable_wandb)
    cfg.vla_dataset.RMBench.repo_id = os.path.abspath(os.path.expanduser(args.repo_id))
    if args.norm_stats_path:
        cfg.dataloader.norm_stats_path = os.path.abspath(os.path.expanduser(args.norm_stats_path))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    OmegaConf.save(cfg, args.out)
    print(f"[config] wrote {args.out}")
    print(f"[config]   repo_id        = {cfg.vla_dataset.RMBench.repo_id}")
    print(f"[config]   checkpoint_dir = {cfg.trainer.checkpoint_dir} (under upstream root)")
    print(f"[config]   batch_size={cfg.trainer.batch_size} train_steps={cfg.trainer.train_steps} "
          f"seed={cfg.seed} is_debug={cfg.is_debug}")


if __name__ == "__main__":
    main()
