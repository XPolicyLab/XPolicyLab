from typing import Dict, Optional, Tuple, Union
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from accelerate.logging import get_logger

import torch
from ema_pytorch import EMA
from torch.nn.parallel import DistributedDataParallel as DDP

from galaxea_fm.utils.normalizer import load_dataset_stats_from_json

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
    assert Path(pretrained_model_path).suffix == ".pt"
    pretrained_dict = torch.load(pretrained_model_path, weights_only=True, map_location='cpu')["model_state_dict"]
    model_dict = model.module.state_dict()

    # Check for shape mismatches and filter valid keys
    mismatched_keys = []
    filtered_state_dict = {}

    for key, ckpt_param in pretrained_dict.items():
        if key in model_dict:
            if model_dict[key].shape == ckpt_param.shape:
                filtered_state_dict[key] = ckpt_param
            else:
                mismatched_keys.append((key, model_dict[key].shape, ckpt_param.shape))
                logger.warning(
                    f"Shape mismatch for {key}: model={model_dict[key].shape}, "
                    f"checkpoint={ckpt_param.shape} - keeping random initialization"
                )

    # Load filtered state dict and get missing/unexpected keys
    incompatible = model.module.load_state_dict(filtered_state_dict, strict=False)

    # Log summary
    if incompatible.missing_keys:
        logger.warning(f"Missing keys (keeping random init): {len(incompatible.missing_keys)} parameters")
    if incompatible.unexpected_keys:
        logger.warning(f"Unexpected keys in checkpoint: {len(incompatible.unexpected_keys)} parameters")
    if mismatched_keys:
        logger.warning(f"Shape mismatched keys (keeping random init): {len(mismatched_keys)} parameters")

    loaded_params = len(filtered_state_dict)
    total_params = len(model_dict)
    logger.info(f"Successfully loaded {loaded_params}/{total_params} parameters from checkpoint")

    return model


def load_embedded_dataset_stats(
    model_path: Path | str
) -> Dict:
    """Load dataset statistics from legacy .pt checkpoint (looks at {ckpt}/../dataset_stats.json)."""
    model_path = Path(model_path)
    assert model_path.suffix == ".pt"
    return load_dataset_stats_from_json(model_path.parent.parent / "dataset_stats.json")


# def save_checkpoint(
#     checkpoint_path: Path | str,
#     step: int,
#     epoch: int,
#     batch_idx: int,
#     model: Union[DDP, torch.nn.Module],
#     optimizer: torch.optim.Optimizer,
#     scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
#     ema_model: Optional[EMA],
#     dataset_stats: Optional[Dict] = None,  # ignored, for interface compatibility
#     cfg: Optional[DictConfig] = None,  # ignored, for interface compatibility
# ) -> None:
#     """
#     Save training checkpoint to disk (legacy single-file format).
#     """
#     checkpoint_path = Path(checkpoint_path)
#     assert checkpoint_path.suffix == ".pt"
#     checkpoint_path.parent.mkdir(exist_ok=True)

#     # Get model state dict (handle DDP wrapper)
#     if isinstance(model, DDP):
#         model_state_dict = model.module.state_dict()
#     else:
#         model_state_dict = model.state_dict()

#     state = {
#         "step": step,
#         "epoch": epoch,
#         "batch_idx": batch_idx,
#         "model_state_dict": model_state_dict,
#         "optimizer_state_dict": optimizer.state_dict(),
#         "scheduler_state_dict": scheduler.state_dict(),
#         "ema_model_state_dict": ema_model.ema_model.state_dict() if ema_model is not None else None,
#     }
#     torch.save(state, checkpoint_path)


def resume_checkpoint(
    checkpoint_path: str,
    model: DDP,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    ema_model: Optional[EMA],
    device_id: int,
) -> Tuple[int, int, int]:
    """
    Resume full training state from checkpoint (legacy single-file format).

    Returns:
        tuple: (step, epoch, batch_idx)
    """
    checkpoint_path = Path(checkpoint_path)
    assert checkpoint_path.suffix == ".pt"
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=f"cuda:{device_id}")

    model.module.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if ema_model is not None:
        if checkpoint.get("ema_model_state_dict") is not None:
            ema_model.ema_model.load_state_dict(checkpoint["ema_model_state_dict"])
        else:
            logger.warning("EMA model not found in checkpoint, skipping EMA load")

    step = checkpoint["step"]
    epoch = checkpoint["epoch"]
    batch_idx = checkpoint["batch_idx"]

    del checkpoint
    torch.cuda.empty_cache()

    logger.info(f"Resumed from step {step}, epoch {epoch}, batch_idx {batch_idx}")
    return step, epoch, batch_idx