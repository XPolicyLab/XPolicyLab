#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from omegaconf import OmegaConf

def main():
    parser = argparse.ArgumentParser(description="Export filtered training YAML to config.json in a checkpoint directory")
    parser.add_argument("--yaml", required=True, help="Path to training YAML (e.g., configs/robotwin.yaml)")
    parser.add_argument("--ckpt_dir", required=True, help="Path to checkpoint directory (e.g., .../checkpoint_step_40000)")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.yaml)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Filter only requested sections
    common = cfg_dict.get("common", {})
    model = cfg_dict.get("model", {})
    filtered = {
        "common": common,
        "action_expert": model.get("action_expert", {}),
        "und_expert": model.get("und_expert", {}),
        "time_distribution": model.get("time_distribution", {}),
        "ema": model.get("ema", {}),
    }

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_path = ckpt_dir / "config.json"
    with open(out_path, "w") as f:
        json.dump(filtered, f, indent=2)
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()

