"""Standard XPolicyLab entry into the externally defined OpenDM experiment."""

import argparse
import json
import os
from pathlib import Path

from XPolicyLab.utils.checkpoint_resolver import build_run_dir_name
from .recipe import accumulation_steps, scheduler_kwargs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bench-name", "ckpt-name", "env-cfg-type", "action-type"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--action-dim", type=int, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--base-model", type=Path)
    parser.add_argument("--norm-stats-root", type=Path)
    parser.add_argument("--per-device-batch", type=int, default=8)
    parser.add_argument("--global-batch", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--save-steps", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--stop-after",
        type=int,
        help="Stop and save at this optimizer step; resume by rerunning without it",
    )
    parser.add_argument(
        "--sdpa",
        action="store_true",
        help="Use SDPA for all attention without requiring flash-attn",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the complete resolved recipe without loading weights",
    )
    args = parser.parse_args()
    assert (args.bench_name, args.env_cfg_type, args.action_type) == ("RoboDojo", "arx_x5", "joint")
    root = Path(__file__).resolve().parent.parent
    data_root = args.data_root or root / "data" / build_run_dir_name(vars(args), include_seed=False)
    output = root / "checkpoints" / build_run_dir_name(vars(args))
    from .robodojo_train import DM05Exp

    exp = DM05Exp()
    exp.model_config.model_name_or_path = str(
        (args.base_model or root / "checkpoints/DM05-MEM").resolve()
    )
    exp.data_config.data_root = str(data_root.resolve())
    exp.data_config.norm_stats_root = str(
        (args.norm_stats_root or root / "checkpoints/robodojo-norm").resolve()
    )
    trainer = exp.trainer_config
    trainer.output_dir = str(output)
    trainer.run_name = output.name
    trainer.seed = args.seed
    trainer.per_device_train_batch_size = args.per_device_batch
    trainer.gradient_accumulation_steps = accumulation_steps(
        int(os.environ["WORLD_SIZE"]), args.per_device_batch, args.global_batch
    )
    trainer.num_train_steps = args.steps
    trainer.lr_scheduler_kwargs = scheduler_kwargs()
    trainer.save_steps = args.save_steps
    trainer.dataloader_num_workers = args.workers
    trainer.dataloader_persistent_workers = args.workers > 0
    trainer.dataloader_prefetch_factor = 2 if args.workers else None
    if args.sdpa:
        exp.model_config.llm_attn_implementation = "sdpa"
        exp.model_config.vision_attn_implementation = "sdpa"
    if args.dry_run:
        from dataclasses import asdict
        from enum import Enum

        print(
            json.dumps(
                asdict(exp), indent=2, default=lambda v: v.value if isinstance(v, Enum) else str(v)
            )
        )
        return
    manifest = json.loads((data_root / "dataset.json").read_text())
    assert (
        manifest["action_dim"] == args.action_dim
    ), "Training and conversion robot dimensions disagree"
    exp.train_release(stop_after=args.stop_after)


if __name__ == "__main__":
    main()
