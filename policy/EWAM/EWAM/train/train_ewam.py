#!/usr/bin/env python3
# Training script for EWAM (WAN + Action Expert + VLM Direct MoT). The flow-matching
# loss variant is selected by `model.loss_reweight` in the config and handled inside the
# model itself (models/ewam.py):
#   false (default) -> plain uniform-mean-MSE flow matching (discrete uniform timestep sampling)
#   true            -> Gaussian timestep weighting + shifted timestep sampling (reweighted)

import os
import re
import sys
import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import warnings

# Set CUDA memory management environment variables to avoid fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import wandb
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin, ProjectConfiguration, set_seed
import yaml
from omegaconf import OmegaConf

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from data.dataset import create_dataset, collate_fn
from utils.scheduler import create_scheduler
from sample import evaluate_model, log_evaluation_metrics
from models.ewam import Ewam, EwamConfig

logger = logging.getLogger(__name__)


def setup_logging(rank: int = 0, log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=f'[Rank {rank}] %(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    warnings.filterwarnings(
        "ignore",
        message=r"No device id is provided via `init_process_group` or `barrier`.*",
        category=UserWarning,
    )


def _convert_stat_to_ewam_format(stat: dict) -> dict:
    """Convert flat stat.json format to the EWAM dataset_stats.json format.

    Input (flat format from stat.json):
        {"observation.state": {"min": [...], "max": [...]}, "action": {"min": [...], "max": [...]}}

    Output (EWAM format):
        {"state": {"default": {"global_min": [...], "global_max": [...], "min": [...], "max": [...]}},
         "action": {"default": {"global_min": [...], "global_max": [...], "min": [...], "max": [...]}}}

    Stats are over the RAW dims (e.g. 7-D action / 8-D state for LIBERO, before UnifyTo16
    padding), so eval restores 16-D model output to the raw dims before denormalizing
    with these stats.
    """
    ewam_stats = {}
    key_map = {
        "observation.state": "state",
        "action": "action",
    }
    for src_key, dst_key in key_map.items():
        if src_key in stat:
            entry = stat[src_key]
            mins = entry.get("min", entry.get("global_min"))
            maxs = entry.get("max", entry.get("global_max"))
            if mins is not None and maxs is not None:
                ewam_stats[dst_key] = {
                    "default": {
                        "global_min": mins,
                        "global_max": maxs,
                        "min": mins,
                        "max": maxs,
                    }
                }
    return ewam_stats


def load_config(config_path: str) -> OmegaConf:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = OmegaConf.load(config_path)

    # Calculate derived parameters
    config.common.action_chunk_size = config.common.num_video_frames * config.common.video_action_freq_ratio

    logger.info(f"Loaded config from {config_path}")
    logger.info(f"Dataset type: {config.dataset.type}")
    logger.info(f"Training mode: {config.model.training_mode}")
    logger.info(f"Action chunk size: {config.common.action_chunk_size}")
    logger.info(f"Video frames: {config.common.num_video_frames}")
    logger.info(f"WAN checkpoint path: {config.model.wan.checkpoint_path if hasattr(config.model, 'wan') and hasattr(config.model.wan, 'checkpoint_path') else 'NOT FOUND'}")

    return config


def setup_distributed():
    """Setup distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')

        return rank, world_size, local_rank
    else:
        return 0, 1, 0


class EwamTrainer:
    """Trainer class for Ewam (WAN + Action + VLM Direct MoT)."""

    def __init__(
        self,
        model: Ewam,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        device: str = "cuda",
        rank: int = 0,
        world_size: int = 1,
        checkpoint_dir: str = "./checkpoints_wan_vlm",
        log_interval: int = 100,
        save_interval: int = 1000,
        val_interval: int = 1000,
        report_to: str = "wandb",
        tb_writer: Optional[SummaryWriter] = None,
        accelerator: Optional[Any] = None,
        config: Optional[Any] = None,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.rank = rank
        self.world_size = world_size

        self.dtype = torch.bfloat16
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.val_interval = val_interval
        self.report_to = report_to
        self.tb_writer = tb_writer
        self.accelerator = accelerator
        self.config = config

        if rank == 0:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.global_step = 0
        self.epoch = 0

        # Track best action L2 loss for special checkpoint saving
        self.best_action_l2_loss = float('inf')
        self.best_action_l2_step = 0

        logger.info(f"EwamDirect Trainer initialized on rank {rank}/{world_size}")
        logger.info(f"Logging backends: {report_to}")

    def save_checkpoint(self, suffix: str = "", is_best_checkpoint: bool = False):
        """Save complete training state using accelerator."""
        if is_best_checkpoint and suffix == "_best_action_l2":
            # For best checkpoint, use fixed name (overwrite old)
            checkpoint_dir = self.checkpoint_dir / "best_action_l2"
        else:
            checkpoint_dir = self.checkpoint_dir / f"checkpoint_step_{self.global_step}{suffix}"

        # Use accelerator to save complete training state
        self.accelerator.save_state(str(checkpoint_dir))
        logger.info(f"Checkpoint saved to {checkpoint_dir}")

        # Also save a config.json alongside weights for reproducibility
        try:
            from omegaconf import OmegaConf as _OmegaConf
            cfg_dict = _OmegaConf.to_container(self.config, resolve=True) if self.config is not None else {}
            # Filter only requested sections
            common = cfg_dict.get("common", {})
            model = cfg_dict.get("model", {})
            filtered = {
                "common": common,
                "action_expert": model.get("action_expert", {}),
                "qwen3_expert": model.get("qwen3_expert", {}),
                "time_distribution": model.get("time_distribution", {}),
            }
            import json as _json
            with open(checkpoint_dir / "config.json", "w") as f:
                _json.dump(filtered, f, indent=2)
            logger.info(f"Wrote config.json to {checkpoint_dir}")
        except Exception as e:
            logger.warning(f"Failed to write config.json: {e}")

    def load_checkpoint(self, checkpoint_path: str, reset_scheduler: bool = True):
        """
        Load checkpoint and resume training.

        Args:
            checkpoint_path: Path to checkpoint directory
            reset_scheduler: If True, reset scheduler to new config instead of loading from checkpoint
        """
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint path {checkpoint_path} does not exist")
            return

        logger.info(f"Loading checkpoint from {checkpoint_path}")

        # Extract step number from checkpoint path (e.g., checkpoint_step_125000)
        step_match = re.search(r'step_(\d+)', checkpoint_path)
        if step_match:
            self.global_step = int(step_match.group(1))
            logger.info(f"Resuming from step {self.global_step}")
        else:
            logger.warning(f"Could not extract step number from {checkpoint_path}, starting from step 0")

        # Load using accelerator
        self.accelerator.load_state(checkpoint_path)
        logger.info(f"Checkpoint loaded successfully from {checkpoint_path}")

        # Reset scheduler with new config if requested
        if reset_scheduler and self.config is not None and self.scheduler is not None:
            logger.info("Resetting scheduler to new configuration (not using checkpoint scheduler state)...")

            unwrapped_scheduler = self.scheduler
            if hasattr(self.scheduler, 'module'):
                unwrapped_scheduler = self.scheduler.module

            if hasattr(unwrapped_scheduler, 'warm_up_steps'):
                unwrapped_scheduler.warm_up_steps = self.config.training.warmup_steps
                unwrapped_scheduler.cycle_length = self.config.training.cycle_length
                unwrapped_scheduler.f_max = self.config.training.f_max
                unwrapped_scheduler.f_min = self.config.training.f_min
                unwrapped_scheduler.base_lrs = [group['lr'] for group in self.optimizer.param_groups]
                unwrapped_scheduler.step_count = 0

                logger.info("Scheduler reset complete")
                logger.info(f"Learning rate will be updated by scheduler on first training step")

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Single training step for EwamDirect (complete step like original train.py)."""
        # Handle DDP wrapper
        model = self.model.module if hasattr(self.model, 'module') else self.model

        self.optimizer.zero_grad()

        # Prepare batch
        first_frame = batch['first_frame'].to(self.device, dtype=self.dtype)
        video_frames = batch['video_frames'].to(self.device, dtype=self.dtype)
        action_sequence = batch['action_sequence'].to(self.device, dtype=self.dtype)
        state = batch.get('initial_state')
        if state is not None:
            state = state.to(self.device, dtype=self.dtype)

        # Language embeddings (T5 for WAN cross attention)
        language_embeddings = batch.get('language_embedding')
        if language_embeddings is not None:
            language_embeddings = language_embeddings.to(self.device, dtype=self.dtype)

        # VLM inputs
        vlm_inputs = batch.get('vlm_inputs')
        if vlm_inputs is not None:
            vlm_inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in vlm_inputs.items()}

        # Action dim pad mask (UnifyTo16): [B, D] bool, True = padded dim
        action_dim_is_pad = batch.get('action_dim_is_pad')
        if action_dim_is_pad is not None:
            action_dim_is_pad = action_dim_is_pad.to(self.device)

        # Model forward
        loss_dict = model.training_step(
            first_frame=first_frame,
            video_frames=video_frames,
            state=state,
            actions=action_sequence,
            language_embeddings=language_embeddings,
            vlm_inputs=vlm_inputs,
            action_dim_is_pad=action_dim_is_pad,
            return_dict=True
        )

        # Backward pass
        loss = loss_dict['total_loss']
        self.accelerator.backward(loss)

        # Gradient clipping
        grad_clip_norm = getattr(self.config.training, 'grad_clip_norm', 1.0)
        if grad_clip_norm > 0:
            self.accelerator.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip_norm
            )

        # Optimizer step
        self.optimizer.step()

        if self.scheduler:
            self.scheduler.step()

        return {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}

    def train(self, max_steps: int, resume_from: Optional[str] = None, reset_scheduler: Optional[bool] = None):
        """
        Main training loop.

        Args:
            max_steps: Maximum number of training steps
            resume_from: Path to checkpoint to resume from
            reset_scheduler: If True, reset scheduler to new config. If None, use config value
        """
        # Load checkpoint if specified
        if resume_from:
            if reset_scheduler is None:
                reset_scheduler = True  # Default behavior
            self.load_checkpoint(resume_from, reset_scheduler=reset_scheduler)

        logger.info(f"Starting EwamDirect training for {max_steps} steps")

        start_time = time.time()

        grad_accum_steps = getattr(self.config.training, 'gradient_accumulation_steps', 1)
        if self.rank == 0:
            logger.info(f"Gradient accumulation steps: {grad_accum_steps}")

        # Step-based training loop (same as original train.py to avoid epoch sync issues)
        data_iter = iter(self.train_dataloader)
        epoch = 0

        while self.global_step < max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                # End of epoch, restart dataloader
                epoch += 1
                self.epoch = epoch
                if hasattr(self.train_dataloader.sampler, 'set_epoch'):
                    self.train_dataloader.sampler.set_epoch(epoch)
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)

            if batch is None:
                continue

            # Training step
            step_start_time = time.time()

            loss_dict = self.train_step(batch)  # Now includes optimizer.step

            # Profile breakdown (log every 5 steps)
            model = self.model.module if hasattr(self.model, 'module') else self.model
            if self.rank == 0 and hasattr(model, '_timing') and self.global_step % 5 == 0:
                pass  # Skip profiling for now

            step_time = time.time() - step_start_time
            self.global_step += 1

            # Log each rank's global_step (commented out)
            # logger.info(f"Rank {self.rank}: global_step={self.global_step}")

            # Logging
            if self.global_step % self.log_interval == 0 and self.rank == 0:
                self.log_metrics(loss_dict, step_time)

            # Validation
            if self.global_step % self.val_interval == 0 and self.val_dataloader is not None:
                should_save_best = False
                if self.rank == 0:
                    # Unwrap model from DDP for inference_step
                    unwrapped_model = self.model.module if hasattr(self.model, 'module') else self.model
                    # Pass dataset stats for denormalized L1/L2 metrics. The dataset's stat is the
                    # aggregated flat format {observation.state: {min,max}, action: {min,max}} over
                    # the RAW action dims. evaluate_model masks padded dims (action_dim_is_pad from
                    # batch) before denormalizing, so raw-dim stats apply correctly.
                    val_dataset_stats = None
                    try:
                        ds = self.train_dataloader.dataset
                        if hasattr(ds, 'stat') and ds.stat is not None:
                            val_dataset_stats = ds.stat
                    except Exception:
                        val_dataset_stats = None
                    val_metrics = evaluate_model(
                        unwrapped_model, self.val_dataloader, self.accelerator, self.config,
                        num_eval_batches=2,
                        dataset_stats=val_dataset_stats,
                    )
                    logger.info(f"Validation - Step {self.global_step}")
                    log_evaluation_metrics(val_metrics, self.tb_writer, self.accelerator, self.global_step)

                    # Check if action L2 loss is the best
                    action_l2_key = 'action_l2_error'
                    if action_l2_key in val_metrics:
                        current_action_l2 = val_metrics[action_l2_key]
                        if current_action_l2 < self.best_action_l2_loss:
                            self.best_action_l2_loss = current_action_l2
                            self.best_action_l2_step = self.global_step
                            logger.info(f"✓ New best action L2 loss: {current_action_l2:.6f} at step {self.global_step}")
                            should_save_best = True

                # Synchronize all processes
                if dist.is_available() and dist.is_initialized():
                    should_save_best_tensor = torch.tensor([1.0 if should_save_best else 0.0], device='cuda')
                    dist.all_reduce(should_save_best_tensor, op=dist.ReduceOp.MAX)
                    should_save_best = should_save_best_tensor.item() > 0.5

                    try:
                        dist.barrier(device_ids=[torch.cuda.current_device()])
                    except TypeError:
                        dist.barrier()

                # Save best checkpoint (all ranks participate)
                if should_save_best:
                    self.save_checkpoint(suffix="_best_action_l2", is_best_checkpoint=True)

            # Save checkpoint
            if self.global_step % self.save_interval == 0:
                self.save_checkpoint()

            # Check for max steps
            if self.global_step >= max_steps:
                logger.info(f"Reached max_steps {max_steps}, training complete!")
                self.save_checkpoint()
                return

        total_time = time.time() - start_time
        if self.rank == 0:
            logger.info(f"Training completed in {total_time:.2f}s ({self.global_step} steps)")
            self.save_checkpoint()
            # Log best action L2 checkpoint info
            if self.best_action_l2_step > 0:
                logger.info(f"Best action L2 loss: {self.best_action_l2_loss:.6f} at step {self.best_action_l2_step}")
                logger.info(f"Best action L2 checkpoint saved to: best_action_l2")

    def log_metrics(self, loss_dict: Dict[str, float], elapsed_time: float):
        """Log training metrics."""
        lrs = [g['lr'] for g in self.optimizer.param_groups]
        lr_main = lrs[0] if len(lrs) > 0 else 0.0
        lr_wan = lrs[1] if len(lrs) > 1 else lr_main
        lr_vlm = lrs[2] if len(lrs) > 2 else lr_main

        if self.rank == 0:
            logger.info(
                f"Step {self.global_step}/{self.config.training.max_steps} (Epoch {self.epoch}), "
                f"Loss: {loss_dict['total_loss']:.4f} "
                f"(Video: {loss_dict['video_loss']:.4f}, Action: {loss_dict['action_loss']:.4f}), "
                f"LR(main/wan/vlm)={lr_main:.2e}/{lr_wan:.2e}/{lr_vlm:.2e}, "
                f"Time: {elapsed_time:.2f}s"
            )

            # Log to WandB - include all loss_dict metrics
            if self.report_to in ["wandb", "all"] or "wandb" in self.report_to:
                wandb.log({
                    **loss_dict,
                    'learning_rate_main': lr_main,
                    'learning_rate_wan': lr_wan,
                    'learning_rate_vlm': lr_vlm,
                    'step_time': elapsed_time,
                    'epoch': self.epoch,
                    'global_step': self.global_step,
                }, step=self.global_step)

            # Log to TensorBoard - include all loss_dict metrics
            if ("tensorboard" in self.report_to or "all" in self.report_to) and self.tb_writer:
                for key, value in loss_dict.items():
                    self.tb_writer.add_scalar(f'train/{key}', value, self.global_step)
                self.tb_writer.add_scalar('train/learning_rate_main', lr_main, self.global_step)
                self.tb_writer.add_scalar('train/learning_rate_wan', lr_wan, self.global_step)
                self.tb_writer.add_scalar('train/learning_rate_vlm', lr_vlm, self.global_step)
                self.tb_writer.add_scalar('train/step_time', elapsed_time, self.global_step)
                self.tb_writer.add_scalar('train/epoch', self.epoch, self.global_step)


def create_model_and_optimizer(config: OmegaConf) -> tuple:
    """Create Ewam model and optimizer from config."""
    # The loss variant (plain vs reweighted) is selected inside the model
    # via `model.loss_reweight` — same class, same state dict, same inference path.

    # Create Ewam config
    model_config = EwamConfig(
        wan_checkpoint_path=config.model.wan.checkpoint_path,
        vae_path=config.model.wan.vae_path,
        wan_config_path=config.model.wan.config_path,
        vlm_checkpoint_path=config.model.vlm.checkpoint_path,
        video_precision=config.model.wan.precision,
        action_state_dim=config.common.state_dim,
        action_dim=config.common.action_dim,
        # Action Expert configuration
        action_expert_dim=config.model.action_expert.hidden_size,
        action_expert_ffn_dim_multiplier=config.model.action_expert.ffn_dim_multiplier,
        action_expert_norm_eps=config.model.action_expert.norm_eps,
        # Qwen3-VL Expert configuration (per-layer QKV projections)
        vlm_dim=config.model.qwen3_expert.vlm_dim,
        qwen3_expert_head_dim=config.model.qwen3_expert.head_dim,
        qwen3_expert_num_heads=config.model.qwen3_expert.num_heads,
        qwen3_expert_num_layers=config.model.qwen3_expert.num_layers,
        qwen3_expert_norm_eps=config.model.qwen3_expert.norm_eps,
        # Other settings
        global_downsample_rate=config.common.global_downsample_rate,
        video_action_freq_ratio=config.common.video_action_freq_ratio,
        num_video_frames=config.common.num_video_frames,
        video_height=config.common.video_height,
        video_width=config.common.video_width,
        batch_size=config.training.batch_size,
        video_loss_weight=config.model.loss_weights.video_loss_weight,
        action_loss_weight=config.model.loss_weights.action_loss_weight,
        loss_reweight=bool(getattr(config.model, "loss_reweight", False)),
        training_mode=config.model.training_mode,
        load_pretrained_backbones=getattr(config.model, 'load_pretrained_backbones', None),
        vlm_frozen=getattr(config.model.vlm, 'frozen', False),
    )

    # Create model
    model = Ewam(model_config)

    # Optimizer - parameter groups for separate learning rates
    base_lr = float(config.training.learning_rate)
    wan_lr = float(getattr(config.training, 'wan_learning_rate', base_lr))
    vlm_lr = float(getattr(config.training, 'vlm_learning_rate', base_lr))

    # Collect parameters by group
    wan_params = [p for p in model.video_model.wan_model.parameters() if p.requires_grad]
    vlm_params = [p for p in model.vlm_model.parameters() if p.requires_grad]
    all_trainable = [p for p in model.parameters() if p.requires_grad]

    # Get IDs for filtering
    wan_param_ids = {id(p) for p in wan_params}
    vlm_param_ids = {id(p) for p in vlm_params}

    # Other params are Action Expert + Qwen3 Module (projections)
    other_params = [p for p in all_trainable if id(p) not in wan_param_ids and id(p) not in vlm_param_ids]

    param_groups = []
    if len(other_params) > 0:
        param_groups.append({'params': other_params, 'lr': base_lr})
    if len(wan_params) > 0:
        param_groups.append({'params': wan_params, 'lr': wan_lr})
    if len(vlm_params) > 0:
        param_groups.append({'params': vlm_params, 'lr': vlm_lr})

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=config.training.weight_decay,
        betas=(0.9, 0.95)
    )

    # Scheduler
    scheduler = create_scheduler(optimizer, config)

    return model, optimizer, scheduler


def create_dataloaders(config: OmegaConf, rank: int, world_size: int) -> tuple:
    """Create train and validation dataloaders from config."""
    train_dataset = create_dataset(config, val=False)
    val_dataset = create_dataset(config, val=True)

    # Samplers
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    else:
        train_sampler = None
        val_sampler = None

    # Dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config.system.num_workers,
        pin_memory=config.system.pin_memory,
        collate_fn=collate_fn,
        drop_last=True,
    )

    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=config.system.num_workers,
            pin_memory=config.system.pin_memory,
            collate_fn=collate_fn,
            drop_last=False,
        )

    return train_dataloader, val_dataloader


def main():
    parser = argparse.ArgumentParser(description="Train EwamDirect (WAN + Action + VLM Direct MoT)")

    # Configuration file
    parser.add_argument("--config", type=str, required=True,
                       default="configs/robotwin_wan_vlm.yaml",
                       help="Path to configuration file")

    # System settings
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Override checkpoint directory")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max_steps")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (python / numpy / torch)")

    # Path overrides (launch scripts may pass these to override config values)
    parser.add_argument("--dataset_dir", type=str, default=None, help="Override dataset directory")
    parser.add_argument("--wan_checkpoint_path", type=str, default=None, help="Override WAN checkpoint path (vae_path and config_path derived from this)")
    parser.add_argument("--vlm_checkpoint_path", type=str, default=None, help="Override VLM checkpoint path")
    parser.add_argument("--finetune_checkpoint_path", type=str, default=None, help="Override finetune checkpoint path")
    parser.add_argument("--tensorboard_log_dir", type=str, default=None, help="Override tensorboard log directory")

    # Logging settings
    parser.add_argument("--report_to", type=str, default=None,
                       choices=["wandb", "tensorboard", "all", "none"],
                       help="Logging backends to use")
    parser.add_argument("--wandb_project", type=str, default=None, help="Override WandB project name")
    parser.add_argument("--run_name", type=str, default=None, help="Override run name")

    # Resume settings
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--reset_scheduler", action="store_true", help="Reset scheduler when resuming")

    # DeepSpeed settings
    parser.add_argument("--deepspeed", type=str, default=None, help="Path to DeepSpeed config file")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    if args.checkpoint_dir is not None:
        config.system.checkpoint_dir = args.checkpoint_dir
    if args.report_to is not None:
        config.logging.report_to = args.report_to
    if args.wandb_project is not None:
        config.logging.wandb_project = args.wandb_project
    if args.run_name is not None:
        config.logging.run_name = args.run_name
    if args.max_steps is not None:
        config.training.max_steps = args.max_steps
    if args.tensorboard_log_dir is not None:
        config.logging.tensorboard_log_dir = args.tensorboard_log_dir
    if args.dataset_dir is not None:
        config.dataset.dataset_dir = args.dataset_dir
    if args.wan_checkpoint_path is not None:
        config.model.wan.checkpoint_path = args.wan_checkpoint_path
        config.model.wan.vae_path = args.wan_checkpoint_path + "/Wan2.2_VAE.pth"
        config.model.wan.config_path = args.wan_checkpoint_path
    if args.vlm_checkpoint_path is not None:
        config.model.vlm.checkpoint_path = args.vlm_checkpoint_path
    if args.finetune_checkpoint_path is not None:
        config.finetune.checkpoint_path = args.finetune_checkpoint_path
    if args.resume:
        config.resume.checkpoint_path = args.resume

    # Decide backbone loading policy
    try:
        if (getattr(config.resume, 'checkpoint_path', None) or
            (hasattr(config, 'finetune') and getattr(config.finetune, 'checkpoint_path', None))):
            config.model.load_pretrained_backbones = False
    except Exception:
        pass

    # Extract dataset name from config file path for checkpoint organization
    config_filename = os.path.basename(args.config)
    dataset_name = os.path.splitext(config_filename)[0]

    # Update checkpoint directory to include dataset name
    base_checkpoint_dir = config.system.checkpoint_dir
    config.system.checkpoint_dir = os.path.join(base_checkpoint_dir, dataset_name)

    # Create the dataset directory
    os.makedirs(config.system.checkpoint_dir, exist_ok=True)

    # Initialize Accelerator
    accelerator_project_config = ProjectConfiguration(total_limit=20)
    accelerator = Accelerator(
        deepspeed_plugin=DeepSpeedPlugin(
            hf_ds_config=args.deepspeed
        ) if args.deepspeed is not None else None,
        gradient_accumulation_steps=config.training.get('gradient_accumulation_steps', 1),
        mixed_precision="bf16",
        log_with=config.logging.get('report_to', 'tensorboard'),
        project_dir=config.system.checkpoint_dir,
        project_config=accelerator_project_config,
    )

    rank = accelerator.process_index
    world_size = accelerator.num_processes
    if args.seed is not None:
        set_seed(args.seed + rank)
    setup_logging(rank, config.system.log_level if hasattr(config.system, 'log_level') else args.log_level)

    # Handle report_to settings
    report_to = config.logging.report_to
    if report_to == "all":
        report_to = ["wandb", "tensorboard"]
    elif report_to == "none":
        report_to = []
    elif isinstance(report_to, str):
        report_to = [report_to]

    # Create run name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = config.logging.get('run_name', None)
    if not run_name:
        run_name = f"ewam_{config.dataset.type}_bs{config.training.batch_size}_lr{config.training.learning_rate}"

    # Update checkpoint directory to include run name
    config.system.checkpoint_dir = os.path.join(config.system.checkpoint_dir, run_name)
    logger.info(f"Checkpoints will be saved to: {config.system.checkpoint_dir}")

    # Initialize TensorBoard writer
    tb_writer = None
    if rank == 0 and "tensorboard" in report_to:
        log_dir = Path(config.logging.tensorboard_log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        run_name = config.logging.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        tb_writer = SummaryWriter(log_dir / run_name)
        logger.info(f"TensorBoard logs will be saved to: {log_dir / run_name}")

    # Initialize WandB
    if rank == 0 and "wandb" in report_to:
        wandb.init(
            project=config.logging.wandb_project,
            config=OmegaConf.to_container(config, resolve=True),
            name=run_name,
        )

    try:
        # Create model and optimizer
        logger.info("Creating EwamDirect model and optimizer...")
        model, optimizer, scheduler = create_model_and_optimizer(config)

        # Optional: load finetune weights for partial init
        finetune_ckpt = getattr(config.finetune, 'checkpoint_path', None) if hasattr(config, 'finetune') else None
        if getattr(config.model, 'training_mode', 'finetune') == 'finetune' and finetune_ckpt:
            logger.info(f"Loading finetune weights from {finetune_ckpt} (partial)...")
            try:
                (model.module if hasattr(model, 'module') else model).load_pretrain_weights(finetune_ckpt)
                logger.info("Finetune weights loaded (partial).")
            except Exception as e:
                logger.error(f"Failed to load finetune weights: {e}")

        # Create dataloaders
        logger.info("Creating dataloaders...")
        train_dataloader, val_dataloader = create_dataloaders(config, rank, world_size)

        # Save dataset_stats.json to checkpoint directory
        # Allows eval scripts to automatically find normalization stats matching training.
        if rank == 0:
            try:
                train_dataset = train_dataloader.dataset
                if hasattr(train_dataset, 'stat') and train_dataset.stat is not None:
                    stats_save_path = os.path.join(config.system.checkpoint_dir, "dataset_stats.json")
                    os.makedirs(config.system.checkpoint_dir, exist_ok=True)
                    ewam_stats = _convert_stat_to_ewam_format(train_dataset.stat)
                    with open(stats_save_path, "w") as f:
                        json.dump(ewam_stats, f, indent=2)
                    logger.info(f"Saved dataset_stats.json to {stats_save_path}")
                else:
                    logger.warning("Dataset has no stat attribute, skipping dataset_stats.json save")
            except Exception as e:
                logger.warning(f"Failed to save dataset_stats.json: {e}")

        # Create custom saving hook to avoid NCCL timeout issues
        def save_model_hook(models, weights, output_dir):
            """Custom save hook to save model safely and avoid NCCL timeouts."""
            if accelerator.is_main_process:
                logger.info(f"Saving model to {output_dir}")
                for i, model_to_save in enumerate(models):
                    # Unwrap the model if it's wrapped by DDP/DeepSpeed
                    unwrapped_model = accelerator.unwrap_model(model_to_save)

                    # Save using torch.save instead of accelerator's default method
                    model_save_path = os.path.join(output_dir, f"pytorch_model_{i}.bin")
                    torch.save(unwrapped_model.state_dict(), model_save_path)
                    logger.info(f"Model {i} saved to {model_save_path}")

        # Register the custom save hook
        accelerator.register_save_state_pre_hook(save_model_hook)

        # Prepare with accelerator
        logger.info("Preparing model, optimizer, and dataloader with Accelerator...")
        model, optimizer, train_dataloader, scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, scheduler
        )

        # Create trainer
        trainer = EwamTrainer(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=accelerator.device,
            rank=rank,
            world_size=world_size,
            checkpoint_dir=config.system.checkpoint_dir,
            log_interval=config.system.log_interval,
            save_interval=config.system.save_interval,
            val_interval=config.system.val_interval,
            report_to=report_to,
            tb_writer=tb_writer,
            accelerator=accelerator,
            config=config,
        )

        # Start training
        trainer.train(
            max_steps=config.training.max_steps,
            resume_from=args.resume,
            reset_scheduler=args.reset_scheduler
        )

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        raise

    finally:
        # Clean up resources
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        if rank == 0 and "wandb" in report_to:
            wandb.finish()
        if tb_writer is not None:
            tb_writer.close()


if __name__ == "__main__":
    main()