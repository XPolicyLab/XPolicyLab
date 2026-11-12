"""Pi05 SFT: RLinf v0.4 native network/optimizer + existing OpenPI data semantics.

The dry run imports neither OpenPI nor Ray. The worker uses the official
OpenPI PyTorch data loader, not the official JAX or legacy PyTorch network.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import shutil

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from XPolicyLab.infra.RLinf.core.validation import checked_rlinf_revision
from XPolicyLab.utils.checkpoint_resolver import build_run_dir_name

POLICY_DIR = Path(__file__).resolve().parents[1]
ADAPTER_DIR = POLICY_DIR.parents[1] / "infra" / "RLinf"


def compose_config(bench, ckpt, robot, action, seed, gpu_ids, overrides=()):
    if robot not in ("arx_x5", "aloha_agilex") or action != "joint":
        raise ValueError(
            "Pi05 SFT currently supports arx_x5/aloha_agilex joint control."
        )
    for name in (bench, ckpt):
        if not name or name in (".", "..") or "/" in name or "\\" in name:
            raise ValueError("bench_name and ckpt_name must be simple run names.")
    devices = gpu_ids.split(",")
    if not all(device.isdecimal() for device in devices) or len(set(devices)) != len(
        devices
    ):
        raise ValueError("gpu_id must contain distinct comma-separated GPU indices.")
    fields = dict(
        bench_name=bench,
        ckpt_name=ckpt,
        env_cfg_type=robot,
        action_type=action,
        seed=seed,
    )
    run_name = build_run_dir_name(fields)
    os.environ["RLINF_PATH"] = str(ADAPTER_DIR / "RLinf")
    with initialize_config_dir(
        version_base="1.1", config_dir=str(POLICY_DIR / "rlinf" / "config")
    ):
        cfg = compose(config_name="pi05_sft", overrides=list(overrides))
    # The positional contract owns run identity and placement; Hydra overrides
    # control training hyperparameters, model/data paths and resume_dir.
    # Keep this outside the JAX run root, which train.sh may overwrite.
    cfg.runner.logger.log_path = str(POLICY_DIR / "checkpoints" / "pytorch_sft")
    cfg.runner.logger.experiment_name = run_name
    cfg.actor.seed = int(seed)
    cfg.actor.model.action_dim = int(os.environ["PI05_SFT_ACTION_DIM"])
    cfg.cluster.component_placement = {"actor": f"0-{len(devices) - 1}"}
    if OmegaConf.is_missing(cfg.data, "train_data_paths"):
        cfg.data.train_data_paths = os.environ.get(
            "OPENPI_LEROBOT_REPO_ID", build_run_dir_name(fields, include_seed=False)
        )
    validate_config(cfg, len(devices))
    return cfg


def validate_config(cfg, world_size):
    model, fsdp = cfg.actor.model, cfg.actor.fsdp_config
    checks = (
        (cfg.runner.task_type == "sft", "runner.task_type must be sft"),
        (
            model.model_type == "openpi" and model.pi05 and model.openpi.task == "sft",
            "SFT requires the native openpi Pi05 model with task=sft",
        ),
        (
            not model.is_lora and not model.add_value_head,
            "This entry implements full SFT without LoRA or a value head",
        ),
        (
            cfg.actor.training_backend == "fsdp"
            and fsdp.strategy == "fsdp"
            and fsdp.sharding_strategy in ("no_shard", "full_shard", "shard_grad_op")
            and fsdp.use_orig_params,
            "Pi05 v0.4 requires FSDP1 with no_shard, full_shard or shard_grad_op and use_orig_params=true",
        ),
        (
            model.precision == "fp32" and model.openpi.embed_dtype == "bf16",
            "Keep FP32 master parameters and BF16 backbone activations",
        ),
        (
            cfg.runner.val_check_interval == -1 and not cfg.data.get("val_data_paths"),
            "RLinf v0.4 VLA SFT has no validation loop; use separate policy evaluation",
        ),
        (
            cfg.runner.max_steps > 0 and cfg.runner.save_interval > 0,
            "max_steps and save_interval must be positive",
        ),
        (cfg.cluster.num_nodes == 1, "This launcher supports a single training node"),
        (
            cfg.actor.micro_batch_size > 0
            and cfg.actor.global_batch_size > 0
            and cfg.actor.global_batch_size % (cfg.actor.micro_batch_size * world_size)
            == 0,
            "global_batch_size must be divisible by micro_batch_size * GPU count",
        ),
        (
            model.openpi.action_horizon > 0
            and model.num_action_chunks > 0
            and model.num_action_chunks <= model.openpi.action_horizon,
            "Action chunk must be positive and no larger than the network horizon",
        ),
    )
    for valid, message in checks:
        if not valid:
            raise ValueError(message)


def norm_stats_file(cfg):
    path = Path(cfg.actor.model.openpi_data.norm_stats_path).expanduser()
    return path / "norm_stats.json" if path.is_dir() else path


def build_train_config(cfg, world_size, *, require_stats=True):
    """Reuse the selected OpenPI transforms without mutating its registry."""
    from openpi.training import config as official

    actual = Path(official.__file__).resolve()
    if not actual.is_relative_to(POLICY_DIR / "openpi" / "src"):
        raise RuntimeError(
            f"Expected the bundled OpenPI data config, imported {actual}"
        )
    source = official.get_config(cfg.actor.model.openpi.config_name)
    model = cfg.actor.model
    expected = {
        "pi05": model.pi05,
        "action_horizon": model.openpi.action_horizon,
        "action_dim": model.openpi.model_action_dim,
        "max_token_len": model.openpi.max_token_len,
        "paligemma_variant": model.openpi.paligemma_variant,
        "action_expert_variant": model.openpi.action_expert_variant,
        "discrete_state_input": model.openpi.discrete_state_input,
    }
    for field, value in expected.items():
        if getattr(source.model, field) != value:
            raise ValueError(
                f"OpenPI TrainConfig.{field} differs from native SFT model: "
                f"{getattr(source.model, field)!r} != {value!r}"
            )
    if (
        not isinstance(source.data, official.LeRobotAlohaDataConfig)
        or source.data.adapt_to_pi
    ):
        raise ValueError("Use the simulator Aloha data config with adapt_to_pi=False.")
    stats = norm_stats_file(cfg).resolve()
    if stats.name != "norm_stats.json" or (require_stats and not stats.is_file()):
        raise FileNotFoundError(f"Supply task-specific norm_stats.json: {stats}")
    dataset = str(cfg.data.train_data_paths)
    # LeRobot accepts absolute local paths; HF cache IDs are also preserved.
    if Path(dataset).expanduser().is_dir():
        dataset = str(Path(dataset).expanduser().resolve())
    data = dataclasses.replace(
        source.data,
        repo_id=dataset,
        assets=dataclasses.replace(
            source.data.assets,
            assets_dir=str(stats.parent.parent),
            asset_id=stats.parent.name,
        ),
    )
    return dataclasses.replace(
        source,
        data=data,
        batch_size=int(cfg.actor.micro_batch_size) * world_size,
        seed=int(cfg.actor.seed),
        num_workers=int(cfg.data.num_workers),
    )


def build_dataloader(cfg, world_size):
    from openpi.training.data_loader import create_data_loader

    config = build_train_config(cfg, world_size)
    loader = create_data_loader(config, framework="pytorch", shuffle=True)
    return loader, loader.data_config()


class _StatsInputs:
    def __call__(self, data):
        return {key: data[key] for key in ("state", "actions")}


def compute_norm_stats(cfg):
    """Compute stats after the same joint-delta transform used by SFT and RL."""
    from openpi.shared import normalize
    from openpi.training import data_loader
    from torch.utils.data import DataLoader

    config = build_train_config(cfg, 1, require_stats=False)
    target = norm_stats_file(cfg)
    if target.exists():
        raise FileExistsError(f"{target} already exists; choose a new stats directory.")
    data = config.data.create(config.assets_dirs, config.model)
    dataset = data_loader.create_torch_dataset(
        data, config.model.action_horizon, config.model
    )
    dataset = data_loader.TransformedDataset(
        dataset,
        [
            *data.repack_transforms.inputs,
            *data.data_transforms.inputs,
            _StatsInputs(),
        ],
    )
    if len(dataset) == 0:
        raise ValueError(
            "Cannot compute normalization statistics from an empty dataset"
        )
    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    for batch in DataLoader(dataset, batch_size=64, num_workers=config.num_workers):
        for key in stats:
            stats[key].update(batch[key].numpy())
    normalize.save(
        target.parent, {key: value.get_statistics() for key, value in stats.items()}
    )
    return target


def save_metadata(cfg, destination):
    """Keep the exact stats and training config alongside a native checkpoint."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if norm_stats_file(cfg).resolve() != (destination / "norm_stats.json").resolve():
        shutil.copy2(norm_stats_file(cfg), destination / "norm_stats.json")
    OmegaConf.save(cfg, destination / "sft_config.yaml", resolve=True)
    (destination / "sft_metadata.json").write_text(
        json.dumps(
            {
                "backend": "rlinf_v0_4",
                "rlinf_revision": checked_rlinf_revision(),
                "train_config_name": cfg.actor.model.openpi.config_name,
                "data": str(cfg.data.train_data_paths),
                "norm_stats_path": "norm_stats.json",
                "action_horizon": cfg.actor.model.openpi.action_horizon,
                "action_dim": cfg.actor.model.action_dim,
            },
            indent=2,
        )
        + "\n"
    )


def preflight(cfg):
    """Validate real inputs before starting Ray or constructing a large model."""
    model_path = cfg.actor.model.model_path
    if not model_path:
        raise ValueError(
            "Set POLICY_MODEL_PATH or actor.model.model_path to the Pi0.5 PyTorch "
            "checkpoint used to initialize SFT."
        )
    from rlinf.models.embodiment.openpi.checkpoint import (
        resolve_full_weights,
        resolve_model_safetensors,
    )

    path = Path(model_path).expanduser()
    if resolve_full_weights(path) is None and resolve_model_safetensors(path) is None:
        raise FileNotFoundError(
            f"Native PyTorch model weights not found at {path}; "
            "JAX params/ is not a PyTorch checkpoint."
        )
    build_train_config(cfg, 1)
    output = Path(cfg.runner.logger.log_path) / cfg.runner.logger.experiment_name
    resume = cfg.runner.resume_dir
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(
            f"{output} already contains a run; use a new ckpt_name or runner.resume_dir."
        )
    if resume:
        from rlinf.utils.checkpoint import parse_global_step_from_checkpoint_path

        step = parse_global_step_from_checkpoint_path(resume)
        if not (Path(resume) / "actor" / "dcp_checkpoint").is_dir():
            raise FileNotFoundError(f"No resumable actor/dcp_checkpoint in {resume}")
        if step >= cfg.runner.max_steps:
            raise ValueError("runner.max_steps must exceed the resumed global step")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bench", "ckpt", "robot", "action", "seed", "gpu_ids"):
        parser.add_argument(name)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate/print config without weights, GPU or Ray",
    )
    mode.add_argument(
        "--compute-norm-stats",
        action="store_true",
        help="Compute dataset stats on CPU without model weights",
    )
    args, overrides = parser.parse_known_args()
    checked_rlinf_revision()
    cfg = compose_config(
        args.bench,
        args.ckpt,
        args.robot,
        args.action,
        args.seed,
        args.gpu_ids,
        overrides,
    )
    print(OmegaConf.to_yaml(cfg, resolve=True))
    if args.dry_run:
        return
    if args.compute_norm_stats:
        print(f"Wrote normalization statistics to {compute_norm_stats(cfg)}")
        return
    output = preflight(cfg)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Full Pi05 SFT requires CUDA; use --dry-run for CPU config validation."
        )
    import torch.multiprocessing as mp
    from rlinf.config import validate_cfg
    from rlinf.runners.sft_runner import SFTRunner
    from rlinf.scheduler import Cluster
    from rlinf.utils.placement import HybridComponentPlacement
    from XPolicyLab.policy.Pi_05.rlinf.sft_worker import Pi05SftWorker

    mp.set_start_method("spawn", force=True)
    cfg = validate_cfg(cfg)
    save_metadata(cfg, output)
    # Workers and later checkpoints share the immutable run-local stats copy.
    cfg.actor.model.openpi_data.norm_stats_path = str(output / "norm_stats.json")
    OmegaConf.save(cfg, output / "sft_config.yaml", resolve=True)
    cluster = Cluster(cluster_cfg=cfg.cluster)
    placement = HybridComponentPlacement(cfg, cluster).get_strategy("actor")
    actor = Pi05SftWorker.create_group(cfg).launch(
        cluster, name=cfg.actor.group_name, placement_strategy=placement
    )
    runner = SFTRunner(cfg, actor)
    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
