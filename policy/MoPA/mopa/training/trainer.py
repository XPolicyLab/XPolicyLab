"""Flow-matching optimization and MoPA checkpoint serialization."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def collate_samples(samples: list[dict]) -> dict:
    return {
        "images": [sample["images"] for sample in samples],
        "instructions": [sample["instruction"] for sample in samples],
        "state": torch.from_numpy(np.stack([sample["state"] for sample in samples])),
        "actions": torch.from_numpy(np.stack([sample["actions"] for sample in samples])),
    }


def save_checkpoint(model: torch.nn.Module, config: dict, statistics: dict, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    # Replace complete files so an interrupted torch.save cannot leave a corrupt model.pt.
    temporary = directory / "model.pt.tmp"
    torch.save(model.state_dict(), temporary)
    temporary.replace(directory / "model.pt")
    for name, value in (("config.json", config), ("dataset_statistics.json", statistics)):
        temporary = directory / f"{name}.tmp"
        temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        temporary.replace(directory / name)


def train_model(model: torch.nn.Module, dataset: Dataset, config: dict, output_dir: Path, *,
                device: str = "cpu", max_steps: int = 100000, batch_size: int = 8,
                learning_rate: float = 1e-4, backbone_learning_rate: float = 1e-5,
                weight_decay: float = 1e-8, max_grad_norm: float = 1.0,
                num_workers: int = 0, save_interval: int = 10000, log_interval: int = 100,
                warmup_steps: int = 1000, seed: int = 0) -> float:
    """Optimize flow-matching loss and save policy checkpoints."""
    if max_steps <= 0 or batch_size <= 0 or learning_rate <= 0 or backbone_learning_rate <= 0:
        raise ValueError("Training steps, batch size, and learning rates must be positive.")
    if num_workers < 0 or save_interval < 0 or log_interval <= 0 or warmup_steps < 0:
        raise ValueError("Invalid worker/save/log/warmup setting.")
    if max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive.")
    model.to(device)
    model.train()
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                        collate_fn=collate_samples, generator=generator)
    backbone, head = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (backbone if name.startswith("backbone.") else head).append(parameter)
    groups = []
    if backbone:
        groups.append({"params": backbone, "lr": backbone_learning_rate})
    if head:
        groups.append({"params": head, "lr": learning_rate})
    if not groups:
        raise ValueError("The model has no trainable parameters.")
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1e-8, weight_decay=weight_decay)
    warmup_steps = min(warmup_steps, max_steps - 1)

    def make_lr_factor(initial_lr):
        # Use an absolute 1e-6 learning-rate floor for every parameter group.
        floor_ratio = min(1.0, 1e-6 / initial_lr)

        def lr_factor(step):
            if warmup_steps and step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
            cosine = 0.5 * (1 + math.cos(math.pi * min(progress, 1)))
            return floor_ratio + (1 - floor_ratio) * cosine

        return lr_factor

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, [make_lr_factor(group["lr"]) for group in optimizer.param_groups])
    iterator = iter(loader)
    loss_value = float("nan")
    for step in range(1, max_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch["state"] = batch["state"].to(device)
        batch["actions"] = batch["actions"].to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch)["action_loss"]
        if loss.ndim != 0 or not torch.isfinite(loss).item():
            raise FloatingPointError(f"Invalid action loss at step {step}: {loss.detach()}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        loss_value = float(loss.detach().cpu())
        if step == 1 or step % log_interval == 0 or step == max_steps:
            print(f"[MoPA] step={step}/{max_steps} action_loss={loss_value:.6f}", flush=True)
        if save_interval and step % save_interval == 0 and step < max_steps:
            save_checkpoint(model, config, dataset.statistics, output_dir / f"steps_{step}")
    save_checkpoint(model, config, dataset.statistics, output_dir)
    return loss_value
