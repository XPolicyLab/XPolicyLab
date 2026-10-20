"""Offline precompute of state/action normalization statistics.

Always launch this through the Spark_0 adapter so every dataloader
environment variable (SPARK_0_XPOLICY_*, SPARK_0_EMBODIMENT_DIR, data root)
matches the real training launch bit-for-bit — the content-addressed cache
file names depend on them:

    bash scripts/precompute_stats.sh <train_config> [gpus] [overrides]
    # equivalently: SPARK_0_PRECOMPUTE_STATS_ONLY=1 bash train.sh ...

Instantiating the datasets runs the exact statistics pipeline training uses
(raw parquet stats + converted pose stats + tactile-pressure q99 when
``pressure.scale`` is auto) and publishes the caches under
<dataset>/meta/stats_xpolicy*__<hash>.json. With
SPARK_0_REQUIRE_PRECOMPUTED_STATS=1 (the train.sh default), training then
only reads those files and fails fast when one is missing, instead of
scanning the dataset on rank 0 at launch.

No GPU and no videos are touched: dataset __init__ reads metadata and parquet
columns only.
"""

import argparse
import os

from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_yaml", required=True, help="Resolved training YAML")
    parser.add_argument(
        "--stats-impl",
        choices=["vectorized", "loop"],
        default=None,
        help="Converted-pose statistics implementation (sets SPARK_0_STATS_IMPL). "
        "'vectorized' (default) uses whole-episode tensor ops; 'loop' is the "
        "original per-base-index Python loop, kept for A/B verification. Both "
        "must produce identical values, so they share the same cache hash.",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore existing stats caches and rebuild them (sets "
        "SPARK_0_STATS_FORCE_RECOMPUTE=1). Needed for implementation A/Bs: "
        "without it an existing cache is simply loaded and nothing is computed.",
    )
    args, overrides = parser.parse_known_args()

    if args.stats_impl is not None:
        os.environ["SPARK_0_STATS_IMPL"] = args.stats_impl
    if args.force_recompute:
        os.environ["SPARK_0_STATS_FORCE_RECOMPUTE"] = "1"

    cfg = OmegaConf.load(args.config_yaml)
    # Forwarded OmegaConf dotlist overrides (same syntax as train.sh passes to
    # the trainer), e.g. `datasets.vla_data.state_history_frames=5` to build
    # the cache for an ablation arm.
    dotlist = [item.lstrip("-") for item in overrides if "=" in item]
    if dotlist:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))
    vla_dataset_cfg = cfg.datasets.vla_data

    # Imported here so --help stays cheap and the registry env vars exported
    # by train.sh are already in place when the data registry module loads.
    from starVLA.dataloader.lerobot_datasets import get_vla_dataset

    dataset = get_vla_dataset(
        data_cfg=vla_dataset_cfg,
        balance_dataset_weights=bool(vla_dataset_cfg.get("balance_dataset_weights", False)),
        balance_trajectory_weights=bool(vla_dataset_cfg.get("balance_trajectory_weights", False)),
        seed=int(getattr(cfg, "seed", None) or 0),
        framework=cfg.get("framework"),
    )

    if not dataset.datasets:
        raise RuntimeError(
            "No datasets were built; check data_mixture / data_mix and the "
            "SPARK_0_XPOLICY_* environment (launch via train.sh)."
        )

    print("[precompute_stats] statistics caches ready:")
    for single in dataset.datasets:
        cache_files = sorted(single.dataset_path.glob("meta/stats_xpolicy*__*.json"))
        print(f"  {single.dataset_path} ({len(cache_files)} cache files)")
        for cache_file in cache_files:
            print(f"    {cache_file}")
    print(
        "[precompute_stats] done. Training launched through train.sh "
        "(SPARK_0_REQUIRE_PRECOMPUTED_STATS=1) will now read these files and "
        "refuse to recompute."
    )


if __name__ == "__main__":
    main()
