"""Minimal distributed post-training loop for Focus-VLWA."""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import math
import os
import random
import shutil
import time

import numpy as np
import torch
from safetensors.torch import save_model
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from focus_vlwa.configs.model import load_model_config
from focus_vlwa.configs.training import PostTrainingConfig
from focus_vlwa.data.dataset import FocusVLWAPostTrainingDataset, collate_focus_vlwa_batch
from focus_vlwa.data.normalization import load_norm_stats
from focus_vlwa.inference.tokenizer import FocusVLWATokenizer
from focus_vlwa.model.checkpoint import load_checkpoint_weights
from focus_vlwa.model.focus_vlwa import FocusVLWA
from focus_vlwa.model.observation import FocusVLWAObservation
from focus_vlwa.post_training.sampler import _ResumableDistributedSampler, batch_change_resume_cursor


def _distributed_setup() -> tuple[int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    return rank, world_size, device


def _learning_rate(step: int, config: PostTrainingConfig) -> float:
    if step < config.warmup_steps:
        initial = config.peak_learning_rate / (config.warmup_steps + 1)
        return initial + (config.peak_learning_rate - initial) * step / max(1, config.warmup_steps)
    progress = min(1.0, (step - config.warmup_steps) / max(1, config.decay_steps - config.warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.final_learning_rate + (config.peak_learning_rate - config.final_learning_rate) * cosine


def _to_device(observation: FocusVLWAObservation, device: torch.device) -> FocusVLWAObservation:
    values = {}
    for field in dataclasses.fields(observation):
        value = getattr(observation, field.name)
        if isinstance(value, dict):
            value = {key: tensor.to(device, non_blocking=True) for key, tensor in value.items()}
        elif isinstance(value, torch.Tensor):
            value = value.to(device, non_blocking=True)
        values[field.name] = value
    return FocusVLWAObservation(**values)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return values.mean()
    expanded = mask.to(dtype=values.dtype, device=values.device)
    while expanded.ndim < values.ndim:
        expanded = expanded.unsqueeze(-1)
    return values.sum() / expanded.expand_as(values).sum().clamp_min(1.0)


def _combined_loss(
    losses: dict[str, torch.Tensor],
    observation: FocusVLWAObservation,
    model: FocusVLWA,
    *,
    balance_world_model: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    if isinstance(losses, torch.Tensor):
        losses = {"action": losses}
    action_loss = _masked_mean(losses["action"], observation.action_mask)

    def action_scale(value: torch.Tensor) -> torch.Tensor:
        if not balance_world_model:
            return value
        return value * (action_loss.detach() / value.detach().clamp_min(1e-6))

    event_loss = (_masked_mean(losses["world_model_event"], observation.event_action_mask)
                  if "world_model_event" in losses else action_loss.new_zeros(()))
    state_loss = (_masked_mean(losses["world_model_state"], observation.world_state_mask)
                  if "world_model_state" in losses else action_loss.new_zeros(()))
    world_model_loss = model.world_model_event_loss_weight * action_scale(
        event_loss
    ) + model.world_model_state_loss_weight * action_scale(state_loss)
    total = action_loss + model.world_model_loss_weight * world_model_loss
    metrics = {
        "loss": float(total.detach()),
        "action_loss": float(action_loss.detach()),
        "world_model_event_loss": float(event_loss.detach()),
        "world_model_state_loss": float(state_loss.detach()),
    }
    return total, metrics


def _save_checkpoint(
    model: FocusVLWA,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: PostTrainingConfig,
) -> None:
    final = config.output_dir / str(step)
    temporary = config.output_dir / f".tmp-{step}"
    if final.exists() or temporary.exists():
        raise FileExistsError(f"Checkpoint destination already exists: {final} or {temporary}")
    temporary.mkdir(parents=True, exist_ok=False)
    save_model(model, temporary / "model.safetensors")
    torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
    (temporary / "metadata.json").write_text(
        json.dumps({"global_step": step, "training_config": dataclasses.asdict(config)}, default=str, indent=2)
    )
    (temporary / "model_config.json").write_text(json.dumps(dataclasses.asdict(model.config), indent=2))
    asset_target = temporary / "assets" / config.norm_stats_dir.name
    asset_target.mkdir(parents=True)
    shutil.copy2(config.norm_stats_dir / "norm_stats.json", asset_target / "norm_stats.json")
    temporary.replace(final)


def run_post_training(config: PostTrainingConfig) -> None:
    """Run Focus-VLWA post-training on one or more CUDA devices."""
    if not torch.cuda.is_available():
        raise RuntimeError("Focus-VLWA post-training requires CUDA")
    rank, world_size, device = _distributed_setup()
    if config.batch_size % world_size:
        raise ValueError("Global batch size must be divisible by WORLD_SIZE")
    random.seed(config.seed + rank)
    np.random.seed(config.seed + rank)
    torch.manual_seed(config.seed + rank)

    checkpoint = config.resume_checkpoint or config.init_checkpoint
    if config.resume_checkpoint is not None and not (checkpoint / "metadata.json").is_file():
        raise ValueError("Resume requires a training checkpoint containing metadata.json")
    model_config = load_model_config(checkpoint)
    overrides = {
        name: getattr(config, name)
        for name in (
            "history_mode",
            "max_token_len",
            "world_model_loss_weight",
            "world_model_event_loss_weight",
            "world_model_state_loss_weight",
        )
        if getattr(config, name) is not None
    }
    model_config = dataclasses.replace(model_config, **overrides)
    tokenizer = FocusVLWATokenizer(model_config.max_token_len, config.tokenizer_path)
    norm_stats = load_norm_stats(config.norm_stats_dir.parent.parent, config.norm_stats_dir.name)
    dataset = FocusVLWAPostTrainingDataset(config.dataset, norm_stats, tokenizer, model_config)
    if len(dataset) < config.batch_size:
        raise ValueError("Dataset must contain at least one complete global batch")
    sampler = _ResumableDistributedSampler(dataset, world_size, rank, shuffle=True, seed=config.seed, drop_last=True)
    local_batch_size = config.batch_size // world_size
    loader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=config.num_workers,
        persistent_workers=config.num_workers > 0,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_focus_vlwa_batch,
    )

    model = FocusVLWA(model_config)
    if config.parameter_precision == "float32":
        model.float()
    load_checkpoint_weights(model, checkpoint, init_world_model_from_action=config.init_world_model_from_action)
    model.to(device)
    if (
        config.freeze_world_model_expert
        or model.world_model_loss_weight == 0
        or (model.world_model_event_loss_weight == 0 and model.world_model_state_loss_weight == 0)
    ):
        skip_supervision = (model.world_model_loss_weight == 0 or
                            (model.world_model_event_loss_weight == 0 and model.world_model_state_loss_weight == 0))
        model.freeze_world_model_expert(skip_supervision=skip_supervision)
    model.gradient_checkpointing_enable()
    model.train()
    train_model: FocusVLWA | DistributedDataParallel = model
    if world_size > 1:
        train_model = DistributedDataParallel(
            model, device_ids=[device.index], gradient_as_bucket_view=True,
            find_unused_parameters=config.parameter_precision == "float32",
            static_graph=config.parameter_precision is None
        )
    optimizer = torch.optim.AdamW(
        train_model.parameters(),
        lr=config.peak_learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=config.weight_decay,
        **({"foreach": False} if config.parameter_precision == "float32" else {}),
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    step = 0
    epoch = 0
    sample_offset = 0
    if config.resume_checkpoint is not None:
        metadata = json.loads((checkpoint / "metadata.json").read_text())
        step = int(metadata["global_step"])
        if (checkpoint / "model_config.json").is_file() and load_model_config(checkpoint) != model_config:
            raise ValueError("Resume must preserve the checkpoint model profile and loss configuration")
        previous = metadata.get("training_config", {})
        old_batch = previous.get("batch_size", config.batch_size)
        if old_batch != config.batch_size and config.resume_batch_change_step is None:
            raise ValueError("Changed batch size requires an explicit resume cursor anchor")
        optimizer.load_state_dict(torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=True))
        if config.parameter_precision == "float32":
            for group in optimizer.param_groups:
                group["foreach"] = False
        epoch, batches = divmod(step, len(loader))
        sample_offset = batches * local_batch_size
        if config.resume_batch_change_step is not None:
            if config.resume_old_batch_size % world_size:
                raise ValueError("Old global batch size must be divisible by WORLD_SIZE")
            epoch, sample_offset = batch_change_resume_cursor(
                step, config.resume_batch_change_step, config.resume_old_batch_size // world_size,
                local_batch_size, sampler.num_samples,
            )
    while step < config.num_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
            sampler.start_index = sample_offset
        for observation, actions in loader:
            if step >= config.num_steps:
                break
            observation = _to_device(observation, device)
            actions = actions.to(device, non_blocking=True)
            learning_rate = _learning_rate(step, config)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            context = (torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False)
                       if config.parameter_precision == "float32" else contextlib.nullcontext())
            with context:
                raw_losses = train_model(observation, actions)
            loss, metrics = _combined_loss(
                raw_losses,
                observation,
                model,
                balance_world_model=config.world_model_loss_balance,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite post-training loss")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(train_model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            step += 1

            if rank == 0 and (step == 1 or step % config.log_interval == 0):
                logging.info(
                    "step=%s loss=%.6f action=%.6f world_event=%.6f world_state=%.6f lr=%.3e grad=%.3f elapsed=%.1fs",
                    step,
                    metrics["loss"],
                    metrics["action_loss"],
                    metrics["world_model_event_loss"],
                    metrics["world_model_state_loss"],
                    learning_rate,
                    float(gradient_norm),
                    time.monotonic() - started,
                )
            if rank == 0 and (step % config.save_interval == 0 or step == config.num_steps):
                _save_checkpoint(model, optimizer, step, config)
        epoch += 1
        sample_offset = 0

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
