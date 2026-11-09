"""Thin entry points around SLA's released Robotwin training and statistics code."""
import argparse
import dataclasses
import glob
import json
import logging
import os
from pathlib import Path

import numpy as np

from XPolicyLab.utils.checkpoint_resolver import build_run_dir_name
from .adapter_utils import POLICY_DIR, validate_robot_dims, validate_support


def dataset_name(args):
    validate_support(args.bench_name, args.env_cfg_type, args.action_type)
    cfg = vars(args)
    name = build_run_dir_name(cfg, include_seed=False)
    if not name or any(c in args.ckpt_name for c in "/\\") or args.ckpt_name in (".", ".."):
        raise ValueError("Training/data ckpt_name must be a short run name, not a path.")
    return name


def validate_dataset(path, action_dim):
    info_path = path / "meta/info.json"
    if not info_path.is_file():
        raise ValueError(f"Not a prepared LeRobot dataset: {path}")
    info = json.loads(info_path.read_text())
    if info.get("codebase_version") != "v2.1":
        raise ValueError(f"SLA requires LeRobot v2.1: {path}")
    features = info.get("features", {})
    for key in ("observation.state", "action"):
        if features.get(key, {}).get("shape") != [action_dim]:
            raise ValueError(f"Expected ARX-X5 joint+gripper vectors in {key}: {path}")
    for camera in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
        feature = features.get(f"observation.images.{camera}", {})
        shape = feature.get("shape", [])
        if feature.get("dtype") != "video" or len(shape) != 3 or shape[0] != 3:
            raise ValueError(f"Expected official CHW RGB video feature for {camera}: {path}")
    if info.get("fps", 0) <= 0 or info.get("total_episodes", 0) <= 0:
        raise ValueError(f"Dataset has no episodes or invalid FPS: {path}")
    if not (path / "data").is_dir() or not (path / "videos").is_dir():
        raise ValueError(f"Dataset is missing data/ or videos/: {path}")
    return info


def make_config(args, *, training=False):
    from openpi.training import config as upstream, weight_loaders

    name = dataset_name(args)
    data_dir = POLICY_DIR / "data" / name
    manifest = json.loads((data_dir / "dataset.json").read_text())
    config = upstream.get_config("train_robotwin")
    config = dataclasses.replace(
        config,
        data=dataclasses.replace(
            config.data, repo_id=str(data_dir / "datasets" / "*"),
            assets=upstream.AssetsConfig(assets_dir=str(POLICY_DIR / "assets"), asset_id=name),
        ),
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    if not manifest["sources"]:
        raise ValueError("Prepared dataset manifest is empty.")
    if not training:
        return config

    @dataclasses.dataclass(frozen=True)
    class RunConfig(upstream.TrainConfig):
        output_directory: str = ""

        @property
        def checkpoint_dir(self):
            return Path(self.output_directory)

    gpu_count = len(args.gpu_ids.split(","))
    fsdp_devices = args.fsdp_devices or gpu_count
    if fsdp_devices < 1 or gpu_count % fsdp_devices or args.batch_size % gpu_count:
        raise ValueError("FSDP must divide visible GPUs; batch size must be divisible by visible GPU count.")
    if args.align_vggt_devices is not None and args.align_vggt_devices > gpu_count:
        raise ValueError("align_vggt_devices cannot exceed visible GPU count.")
    if args.action_dim != manifest["action_dim"]:
        raise ValueError("Training robot registration disagrees with the prepared dataset action dimension.")
    base = Path(args.base_checkpoint).expanduser().resolve()
    teacher = Path(args.omnivggt_checkpoint).expanduser().resolve()
    dino = Path(args.dinov2_checkpoint).expanduser().resolve()
    if not (teacher / "OmniVGGT.safetensors").is_file():
        raise FileNotFoundError(f"Missing OmniVGGT asset: {teacher}; run download_checkpoints.sh training.")
    if not dino.is_file():
        raise FileNotFoundError(f"Missing DINOv2 initialization asset: {dino}; run download_checkpoints.sh training.")
    os.environ["SLA_DINOV2_WEIGHTS"] = str(dino)
    if not args.smoke_test and not (base / "params").is_dir():
        raise FileNotFoundError(f"Missing SLA base params: {base}; run download_checkpoints.sh training.")
    if args.smoke_test and not args.ckpt_name.startswith("smoke"):
        raise ValueError("--smoke-test requires a ckpt_name starting with 'smoke' to isolate random weights.")
    run_name = build_run_dir_name(vars(args))
    output = POLICY_DIR / "checkpoints" / run_name
    if output.exists() and not args.resume:
        raise FileExistsError(f"Run already exists: {output}. Choose a new ckpt_name or pass --resume.")
    values = {f.name: getattr(config, f.name) for f in dataclasses.fields(config) if f.init}
    values.update(
        output_directory=str(output), exp_name=run_name, seed=args.seed,
        weight_loader=weight_loaders.NoOpWeightLoader() if args.smoke_test else weight_loaders.CheckpointWeightLoader(str(base / "params")),
        vggt_weight_path=str(teacher), align_target_model="omnivggt",
        fsdp_devices=fsdp_devices, align_vggt_devices=args.align_vggt_devices or gpu_count,
        align_feature_batch_size=args.align_feature_batch_size,
        num_train_steps=args.num_train_steps, save_interval=args.save_interval,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_interval=1 if args.smoke_test else config.log_interval,
        overwrite=False, resume=args.resume, wandb_enabled=False,
        policy_metadata={"policy": "SLA", "random_weight_smoke_test": args.smoke_test},
    )
    return RunConfig(**values)


def prepare(args):
    from openpi.shared import normalize
    from scripts import compute_norm_stats
    from XPolicyLab.utils.process_data import get_robot_action_dim_info

    name = dataset_name(args)
    dims = get_robot_action_dim_info(args.env_cfg_type)
    validate_robot_dims(dims)
    action_dim = sum(dims["arm_dim"]) + sum(dims["ee_dim"])
    pattern = str(Path(args.source).expanduser().absolute())
    sources = sorted({Path(p).resolve() for p in glob.glob(pattern)})
    # The official RoboTwin archive contains one v2.1 directory per task.
    if len(sources) == 1 and not (sources[0] / "meta/info.json").exists():
        sources = sorted(p.parent.parent for p in sources[0].glob("*/meta/info.json"))
    if not sources:
        raise ValueError(f"No LeRobot datasets found: {args.source}")
    for source in sources:
        validate_dataset(source, action_dim)
    directory = POLICY_DIR / "data" / name
    manifest_path = directory / "dataset.json"
    manifest = {"version": "v2.1", "sources": [str(p) for p in sources], "action_dim": action_dim}
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise FileExistsError(f"{name} already refers to other data. Choose a different ckpt_name.")
    links = directory / "datasets"
    links.mkdir(parents=True, exist_ok=True)
    for i, source in enumerate(sources):
        link = links / f"{i:04d}"
        if link.is_symlink() and link.resolve() == source:
            continue
        if link.exists() or link.is_symlink():
            raise FileExistsError(link)
        link.symlink_to(source, target_is_directory=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    config = make_config(args)
    data_cfg = config.data.create(config.assets_dirs, config.model)
    if args.max_frames is not None and args.max_frames < args.batch_size:
        raise ValueError("max_frames must be at least batch_size.")
    loader, _ = compute_norm_stats.create_torch_dataloader(
        data_cfg, config.model.action_horizon, args.batch_size, config.model, args.num_workers, args.max_frames,
    )
    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    batches = 0
    for batch in loader:
        for key, accumulator in stats.items():
            accumulator.update(np.asarray(batch[key]))
        batches += 1
    if not batches:
        raise ValueError("No normalization batches; reduce batch size or provide more frames.")
    result = {key: value.get_statistics() for key, value in stats.items()}
    destination = POLICY_DIR / "assets" / name
    normalize.save(destination, result)
    print(f"[SLA] Prepared {len(sources)} datasets; {batches} statistics batches; assets={destination}")


def train(args):
    from scripts import train_align
    config = make_config(args, training=True)
    print(f"[SLA] checkpoint_dir={config.checkpoint_dir}; random_weight_smoke_test={args.smoke_test}")
    train_align.main(config)


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "train"):
        sub = commands.add_parser(command)
        for name in ("bench_name", "ckpt_name", "env_cfg_type", "action_type"):
            sub.add_argument(name)
        sub.add_argument("--batch-size", type=int, default=256)
        sub.add_argument("--num-workers", type=int, default=32)
        if command == "prepare":
            sub.add_argument("source", help="Prepared v2.1 dataset, task-directory parent, or quoted glob")
            sub.add_argument("--max-frames", type=int)
        else:
            sub.add_argument("seed", type=int)
            sub.add_argument("gpu_ids")
            sub.add_argument("--action-dim", type=int, required=True, help="Provided by utils/get_action_dim.sh")
            sub.add_argument("--base-checkpoint", default=str(POLICY_DIR / "checkpoints/SLA_base"))
            sub.add_argument("--omnivggt-checkpoint", default=str(POLICY_DIR / "checkpoints/omnivggt"))
            sub.add_argument("--dinov2-checkpoint", default=str(POLICY_DIR / "checkpoints/dinov2/dinov2_vitl14_reg4_pretrain.pth"))
            sub.add_argument("--num-train-steps", type=int, default=30000)
            sub.add_argument("--save-interval", type=int, default=30000)
            sub.add_argument("--fsdp-devices", type=int)
            sub.add_argument("--align-vggt-devices", type=int)
            sub.add_argument("--align-feature-batch-size", type=int, default=32)
            sub.add_argument("--gradient-accumulation-steps", type=int, default=1)
            sub.add_argument("--resume", action="store_true")
            sub.add_argument("--smoke-test", action="store_true", help="Random initialization; never use for benchmark results")
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("batch size must be positive and workers must be nonnegative")
    for name in ("max_frames", "num_train_steps", "save_interval", "fsdp_devices", "align_vggt_devices", "align_feature_batch_size", "gradient_accumulation_steps"):
        value = getattr(args, name, None)
        if value is not None and value < 1:
            parser.error(f"{name.replace('_', '-')} must be positive")
    # Keep all upstream relative side effects (e.g. wandb logs) inside this adapter.
    os.chdir(POLICY_DIR)
    (prepare if args.command == "prepare" else train)(args)


if __name__ == "__main__":
    main()
