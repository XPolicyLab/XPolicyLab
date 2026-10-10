"""Distributed Clean50 training entry point."""

from __future__ import annotations

import argparse
import logging
import math
import os
from pathlib import Path

import torch
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from .config import DatasetConfig, TactileConfig, TrainingConfig
from .data import Clean50Dataset, collate
from .model import MEDEXTrainingModel


LOG = logging.getLogger("me_dex.train")


def learning_rate_scale(step: int, *, warmup_steps: int, max_steps: int) -> float:
    if step < warmup_steps:
        return max(1, step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def load_initialization(model: MEDEXTrainingModel, checkpoint: Path) -> None:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    source = state.get("module", state)
    target = model.state_dict()
    prefixes = (
        "video_model.wan_model.",
        "action_expert.time_embedding.",
        "action_expert.time_projection.",
        "action_expert.blocks.",
    )
    selected = {
        f"model.{key}": value.to(dtype=target[f"model.{key}"].dtype)
        for key, value in source.items()
        if key.startswith(prefixes) and f"model.{key}" in target
    }
    if not selected:
        raise RuntimeError(f"No compatible initialization weights found in {checkpoint}")
    model.load_state_dict(selected, strict=False)
    LOG.info("Loaded %d backbone and action tensors", len(selected))


def load_config(path: Path) -> tuple[TrainingConfig, Path, Path]:
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    data = values["data"]
    model = values["model"]
    training = values["training"]
    dataset = DatasetConfig(
        clean_root=Path(data["clean_root"]),
        text_cache=Path(data["text_cache"]),
        quality_manifest=Path(data["quality_manifest"]),
        episodes_per_task=int(data["episodes_per_task"]),
        samples_per_episode=int(data["samples_per_episode"]),
    )
    tactile = TactileConfig(
        checkpoint=Path(model["tactile_checkpoint"]),
        slot_ids=tuple(int(value) for value in model["tactile_slot_ids"]),
        latent_dim=int(model["tactile_latent_dim"]),
        frame_count=int(model["tactile_frame_count"]),
        queries_per_frame=int(model["tactile_queries_per_frame"]),
        observed_frames=int(model["tactile_observed_frames"]),
        hidden_size=int(model["hidden_size"]),
    )
    config = TrainingConfig(
        dataset=dataset,
        tactile=tactile,
        initialization_checkpoint=Path(model["initialization_checkpoint"]),
        output_dir=Path(training["output_dir"]),
        topology=str(model["topology"]),
        h_bridge_joint_start_layer=int(model.get("h_bridge_joint_start_layer", 8)),
        h_bridge_joint_end_layer=int(model.get("h_bridge_joint_end_layer", 22)),
        batch_size=int(training["batch_size"]),
        world_size=int(training["world_size"]),
        max_steps=int(training["max_steps"]),
        learning_rate=float(training["learning_rate"]),
        warmup_steps=int(training["warmup_steps"]),
        save_interval=int(training["save_interval"]),
    )
    return config, Path(values.get("wan_config", "checkpoints/Wan2.2-TI2V-5B")), Path(
        values.get("vae_path", "checkpoints/Wan2.2-TI2V-5B/Wan2.2_VAE.pth")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("training/configs/clean50_uni.yaml"))
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group("nccl")
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    config, wan_config, vae_path = load_config(args.config)
    if world_size != config.world_size:
        raise ValueError(f"Expected {config.world_size} processes, got {world_size}")

    model = MEDEXTrainingModel(config, wan_config, vae_path)
    model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: learning_rate_scale(
            step, warmup_steps=config.warmup_steps, max_steps=config.max_steps
        ),
    )
    start_step = 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.module.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        if "scheduler" in state:
            scheduler.load_state_dict(state["scheduler"])
        start_step = int(state["step"])
    else:
        load_initialization(model.module, config.initialization_checkpoint)

    dataset = Clean50Dataset(
        config.dataset.clean_root,
        config.dataset.text_cache,
        config.dataset.quality_manifest,
        samples_per_episode=config.dataset.samples_per_episode,
    )
    sampler = DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate,
        persistent_workers=True,
        prefetch_factor=4,
    )

    step = start_step
    epoch = 0
    model.train()
    while step < config.max_steps:
        sampler.set_epoch(epoch)
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            losses = model(batch)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            scheduler.step()
            step += 1
            if rank == 0 and step % 10 == 0:
                LOG.info(
                    "step=%d loss=%.5f video=%.5f action=%.5f tactile=%.5f",
                    step,
                    losses["loss"].item(),
                    losses["video_loss"].item(),
                    losses["action_loss"].item(),
                    losses["tactile_loss"].item(),
                )
            if rank == 0 and (step % config.save_interval == 0 or step == config.max_steps):
                config.output_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "step": step,
                        "model": model.module.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    },
                    config.output_dir / f"checkpoint_step_{step}.pt",
                )
            if step == config.max_steps:
                break
        epoch += 1
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
