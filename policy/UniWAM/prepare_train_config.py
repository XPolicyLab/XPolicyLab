from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--wan", required=True)
    parser.add_argument("--vlm", required=True)
    parser.add_argument("--finetune", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--report-to", default="tensorboard")
    args = parser.parse_args()

    with Path(args.base).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["dataset"]["dataset_dir"] = str(Path(args.dataset).resolve())
    cfg["dataset"]["max_episodes"] = args.max_episodes
    cfg["model"]["wan"].update(
        {
            "config_path": str(Path(args.wan).resolve()),
            "checkpoint_path": str(Path(args.wan).resolve()),
            "vae_path": str(Path(args.wan).resolve() / "Wan2.2_VAE.pth"),
        }
    )
    cfg["model"]["vlm"]["checkpoint_path"] = str(Path(args.vlm).resolve())
    cfg["finetune"]["checkpoint_path"] = str(Path(args.finetune).resolve())
    cfg["resume"]["checkpoint_path"] = None
    cfg["training"]["max_steps"] = args.max_steps
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["system"]["num_workers"] = args.num_workers
    cfg["system"]["checkpoint_dir"] = str(Path(args.checkpoint_dir).resolve())
    cfg["system"]["exact_checkpoint_dir"] = True
    cfg["logging"]["run_name"] = args.run_name
    cfg["logging"]["report_to"] = args.report_to

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


if __name__ == "__main__":
    main()
