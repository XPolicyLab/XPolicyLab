from typing import Dict, Optional, Tuple, Union
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from accelerate.logging import get_logger

import torch
from ema_pytorch import EMA
from torch.nn.parallel import DistributedDataParallel as DDP

from galaxea_fm.utils.normalizer import (
    load_dataset_stats_from_json, 
    save_dataset_stats_to_json,
)

logger = get_logger(__name__)


def load_pretrained_model(
    pretrained_model_path: Path | str, 
    model: DDP, 
):
    """
    Safely load state dict with proper handling of shape mismatches.

    Args:
        model: The model to load weights into
        state_dict: State dict from checkpoint
    """
    pretrained_model_path = Path(pretrained_model_path)
    pretrained_dict = torch.load(pretrained_model_path / "model.pt", weights_only=True, map_location='cpu')
    model_dict = model.module.state_dict()

    # Check for unexpected and shape mismatches and filter valid keys
    # Unexpected keys must be handled here, not by `load_state_dict`
    unexpected_keys, mismatched_key_shapes, match_key_tensors = [], {}, {}

    for key, ckpt_param in pretrained_dict.items():
        if key not in model_dict:
            unexpected_keys.append(key)
        elif model_dict[key].shape != ckpt_param.shape:
            mismatched_key_shapes[key] = (model_dict[key].shape, ckpt_param.shape)
        else:
            match_key_tensors[key] = ckpt_param

    # Load shape matched tensors, get missing keys which excludes shape mismatched ones
    incompatible = model.module.load_state_dict(match_key_tensors, strict=False)
    assert len(incompatible.unexpected_keys) == 0, "The filtered state dict should be a subset of model keys."
    missing_keys = list(set(incompatible.missing_keys) - set(mismatched_key_shapes.keys()))

    # Log summary
    logger.info(f"Successfully loaded keys for model: {len(match_key_tensors)} / {len(model_dict)}")

    if missing_keys:
        logger.warning(f"Name unfounded keys for model: {len(missing_keys)} / {len(model_dict)}")
        for k in missing_keys:
            logger.warning(f"  {k}")

    if mismatched_key_shapes:
        logger.warning(f"Name found but shape mismatched keys for model: {len(mismatched_key_shapes)} / {len(model_dict)}")
        for k, (model_shape, ckpt_shape) in mismatched_key_shapes.items():
            logger.warning(f"  {k}: model shape {model_shape}, checkpoint shape {ckpt_shape}")

    if unexpected_keys:
        logger.warning(f"Unexpected keys from pretrained checkpoint: {len(unexpected_keys)} / {len(pretrained_dict)}")
        for k in unexpected_keys:
            logger.warning(f"  {k}")

    return model


def load_embedded_dataset_stats(
    model_path: Path | str
) -> Dict:
    """Load dataset statistics from a pretrained or checkpointed model../dataset_stats.json)."""
    model_path = Path(model_path)
    return load_dataset_stats_from_json(model_path / "dataset_stats.json")


def save_checkpoint(
    path: Path,
    step: int,
    epoch: int,
    batch_idx: int,
    model: Union[DDP, torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    ema_model: Optional[EMA],
    dataset_stats: Optional[Dict] = None,
    cfg: Optional[DictConfig] = None,
) -> None:
    """
    Save checkpoint in directory-based format.
    Directory structure:
        checkpoints/
        ├── step_N/                      # Deployment directory
        │   ├── model.pt                 # Model weights
        │   ├── ema_model.pt             # EMA weights (if enabled)
        │   ├── dataset_stats.json       # Normalization stats
        │   └── config.yaml              # Config
        └── trainer_state_step_N.pt      # Trainer state (optimizer + scheduler)
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Get model state dict (handle DDP wrapper)
    if isinstance(model, DDP):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    torch.save(model_state_dict, path / "model.pt")
    save_dataset_stats_to_json(dataset_stats, path / "dataset_stats.json")
    OmegaConf.save(cfg, path / "config.yaml")

    if ema_model is not None:
        torch.save(ema_model.ema_model.state_dict(), path / "ema_model.pt")

    trainer_state = {
        "step": step,
        "epoch": epoch,
        "batch_idx": batch_idx,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    torch.save(trainer_state, path.parent / f"trainer_state_step_{step}.pt")


def load_checkpoint_for_eval(
    checkpoint_path: Path | str,
    model: torch.nn.Module,
    device: str = "cpu",
) -> Tuple[torch.nn.Module, Dict]:
    """
    Load checkpoint for evaluation, supporting both legacy (.pt file) and new (directory) formats.

    Args:
        checkpoint_path: Path to checkpoint (either .pt file or directory)
        model: Model to load weights into
        device: Device to load weights to

    Returns:
        tuple: (model with loaded weights, dataset_stats)
    """
    checkpoint_path = Path(checkpoint_path)

    if checkpoint_path.is_dir():
        # New format: directory with model.pt and dataset_stats.json
        logger.info(f"Loading checkpoint from directory (new format): {checkpoint_path}")
        state_dict = torch.load(checkpoint_path / "model.pt", map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        dataset_stats = load_dataset_stats_from_json(checkpoint_path / "dataset_stats.json")
    else:
        # Legacy format: single .pt file
        logger.info(f"Loading checkpoint from file (legacy format): {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict["model_state_dict"], strict=True)
        dataset_stats = load_dataset_stats_from_json(checkpoint_path.parent.parent / "dataset_stats.json")

    return model, dataset_stats


def resume_checkpoint(
    checkpoint_path: str,
    model: DDP,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    ema_model: Optional[EMA],
    device_id: int,
) -> Tuple[int, int, int]:
    """
    Resume full training state from directory-based checkpoint.
    Directory structure:
        checkpoints/
        ├── step_N/                      # Deployment directory
        │   ├── model.pt                 # Model weights
        │   ├── ema_model.pt             # EMA weights (if enabled)
        │   ├── dataset_stats.json       # Normalization stats
        │   └── config.yaml              # Config
        └── trainer_state_step_N.pt      # Trainer state (optimizer + scheduler)
    
    Returns:
        tuple: (step, epoch, batch_idx)
    """
    checkpoint_path = Path(checkpoint_path)
    model_state = torch.load(checkpoint_path / "model.pt", weights_only=True, map_location=f"cuda:{device_id}")
    model.module.load_state_dict(model_state)
    del model_state

    step_from_name = int(checkpoint_path.name.split("_")[1])
    trainer_state = torch.load(checkpoint_path.parent / f"trainer_state_step_{step_from_name}.pt", weights_only=True, map_location=f"cuda:{device_id}")
    optimizer.load_state_dict(trainer_state["optimizer_state_dict"])
    scheduler.load_state_dict(trainer_state["scheduler_state_dict"])
    step = trainer_state["step"]
    epoch = trainer_state["epoch"]
    batch_idx = trainer_state["batch_idx"]

    del trainer_state

    if ema_model is not None:
        ema_path = checkpoint_path / "ema_model.pt"
        assert ema_path.exists(), f"Trying to load EMA model but state does not exist at {ema_path}"
        ema_state = torch.load(ema_path, weights_only=True, map_location=f"cuda:{device_id}")
        ema_model.ema_model.load_state_dict(ema_state)
        del ema_state

    torch.cuda.empty_cache()

    return step, epoch, batch_idx