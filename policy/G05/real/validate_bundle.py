from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

NAMED_STATS_BY_EMBODIMENT = {
    "robodojo_arx_x5": "dataset_stats_robodojo_real_arx_x5_h32.json",
    "robodojo_piper": "dataset_stats_robodojo_real_piper_h32.json",
    "robodojo_piper_x": "dataset_stats_robodojo_real_piper_x_h32.json",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--embodiment", required=True)
    args = parser.parse_args()

    root = Path(args.bundle).expanduser().resolve()
    try:
        named_stats_name = NAMED_STATS_BY_EMBODIMENT[args.embodiment]
    except KeyError as exc:
        raise ValueError(f"unsupported embodiment: {args.embodiment}") from exc
    checkpoint = root / "checkpoints" / "checkpoint.pt"
    config = root / ".hydra" / "config.yaml"
    public_config = root / "config.yaml"
    stats = root / "dataset_stats.json"
    named_stats = root / named_stats_name
    required = [
        checkpoint,
        config,
        public_config,
        stats,
        named_stats,
        root / "input_processor",
        root / "action_tokenizer_hf",
        root / "inference_runtime" / "src" / "g05",
        root / "inference_runtime" / "scripts" / "serve_policy.py",
        root / "inference_runtime" / "third_party" / "galaxea_dataset" / "src",
        root / "inference_runtime" / "third_party" / "galaxea_tokenizer" / "src",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"incomplete G05 real bundle; missing: {missing}")

    cfg = OmegaConf.load(config)
    if OmegaConf.to_container(cfg, resolve=False) != OmegaConf.to_container(
        OmegaConf.load(public_config), resolve=False
    ):
        raise ValueError("config.yaml and .hydra/config.yaml differ")
    datasets = cfg.data.get("datasets")
    if datasets is None or list(datasets) != [args.embodiment]:
        raise ValueError(
            f"bundle must contain exactly data.datasets.{args.embodiment}; "
            f"got {list(datasets) if datasets is not None else None}"
        )
    if str(datasets[args.embodiment].embodiment) != args.embodiment:
        raise ValueError("dataset embodiment mismatch")
    if int(datasets[args.embodiment].action_size) != 32:
        raise ValueError("action horizon must be 32")
    payload = json.loads(stats.read_text(encoding="utf-8"))
    if set(payload) != {args.embodiment}:
        raise ValueError(f"stats keys={sorted(payload)} do not match {args.embodiment}")
    if json.loads(named_stats.read_text(encoding="utf-8")) != payload:
        raise ValueError("named and flat dataset stats differ")
    print(f"validated G05 real bundle: embodiment={args.embodiment} frequency=25Hz")


if __name__ == "__main__":
    main()
