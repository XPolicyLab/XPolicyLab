#!/usr/bin/env python3
# Training script for Motus

import os
import re
import sys
import argparse
import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import warnings

# Set CUDA memory management environment variables to avoid fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.distributed as dist
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import wandb
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin, ProjectConfiguration, set_seed
import yaml
from omegaconf import OmegaConf
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.motus import Motus, MotusConfig
from data.dataset import create_dataset, collate_fn
from utils.scheduler import create_scheduler
from sample import evaluate_model, log_evaluation_metrics

logger = logging.getLogger(__name__)


def suppress_noisy_loggers():
    """Keep third-party debug logs from flooding rank logs."""
    for logger_name in ["PIL", "PIL.Image", "PIL.PngImagePlugin"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def _worker_init_fn(worker_id: int):
    """Re-apply logger suppression inside dataloader workers."""
    suppress_noisy_loggers()


def _make_worker_init_fn(base_seed: int, rank: int, split_offset: int = 0):
    """Seed dataloader workers deterministically per rank and split."""

    def _seeded_worker_init_fn(worker_id: int):
        suppress_noisy_loggers()
        worker_seed = int(base_seed) + split_offset + rank * 1000 + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed % (2 ** 32))
        torch.manual_seed(worker_seed)

    return _seeded_worker_init_fn


def resolve_training_seed(args: argparse.Namespace, config: OmegaConf) -> Optional[int]:
    """Resolve the effective training seed from CLI first, then config."""
    if args.seed is not None:
        return int(args.seed)

    config_seed = getattr(config.training, 'seed', None) if hasattr(config, 'training') else None
    if config_seed is None:
        return None

    return int(config_seed)

def setup_logging(rank: int = 0, log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=f'[Rank {rank}] %(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    suppress_noisy_loggers()
    # Suppress specific distributed warnings that are noisy but harmless for our flow
    warnings.filterwarnings(
        "ignore",
        message=r"No device id is provided via `init_process_group` or `barrier`.*",
        category=UserWarning,
    )

def load_config(config_path: str) -> OmegaConf:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    config = OmegaConf.load(config_path)
    
    # Calculate derived parameters
    config.common.action_chunk_size = config.common.num_video_frames * config.common.video_action_freq_ratio
    
    # Validate dataset configuration
    dataset_config = {
        'dataset_type': config.dataset.type,
        'dataset_dir': config.dataset.dataset_dir if hasattr(config.dataset, 'dataset_dir') else None,
        'global_downsample_rate': config.common.global_downsample_rate,
        'video_action_freq_ratio': config.common.video_action_freq_ratio,
        'num_video_frames': config.common.num_video_frames
    }
    
    logger.info(f"Loaded config from {config_path}")
    logger.info(f"Dataset type: {config.dataset.type}")
    if hasattr(config, 'training_mode'):
        logger.info(f"Training mode: {config.training_mode}")
    logger.info(f"Action chunk size: {config.common.action_chunk_size}")
    logger.info(f"Video frames: {config.common.num_video_frames}")
    
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

class UniDiffuserTrainer:
    """Trainer class for Motus."""
    
    def __init__(
        self,
        model: Motus,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        device: str = "cuda",
        rank: int = 0,
        world_size: int = 1,
        checkpoint_dir: str = "./checkpoints_stage4",
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
        
        # Create checkpoint directory
        if rank == 0:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tracking variables
        self.global_step = 0
        self.epoch = 0
        
        logger.info(f"Motus Trainer initialized on rank {rank}/{world_size}")
        logger.info(f"Logging backends: {report_to}")

    def save_checkpoint(self, suffix: str = ""):
        """Save complete training state using accelerator."""
        checkpoint_dir = self.checkpoint_dir / f"checkpoint_step_{self.global_step}{suffix}"
        
        # Use accelerator to save complete training state
        # This saves model, optimizer, scheduler, dataloader, and RNG states
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
                "und_expert": model.get("und_expert", {}),
                "time_distribution": model.get("time_distribution", {}),
                "ema": model.get("ema", {}),
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

        # Load using accelerator (includes model, optimizer, scheduler states)
        self.accelerator.load_state(checkpoint_path)
        logger.info(f"Checkpoint loaded successfully from {checkpoint_path}")
        
        # Reset scheduler with new config if requested
        if reset_scheduler and self.config is not None and self.scheduler is not None:
            logger.info("Resetting scheduler to new configuration (not using checkpoint scheduler state)...")
            
            # Unwrap scheduler if it's wrapped by accelerator
            unwrapped_scheduler = self.scheduler
            if hasattr(self.scheduler, 'module'):
                unwrapped_scheduler = self.scheduler.module
            
            # Check if it's our custom LambdaLinearScheduler
            if hasattr(unwrapped_scheduler, 'warm_up_steps'):
                # Update scheduler parameters with new config
                unwrapped_scheduler.warm_up_steps = self.config.training.warmup_steps
                unwrapped_scheduler.cycle_length = self.config.training.cycle_length
                unwrapped_scheduler.f_max = self.config.training.f_max
                unwrapped_scheduler.f_min = self.config.training.f_min
                # Update base_lrs for all parameter groups
                unwrapped_scheduler.base_lrs = [group['lr'] for group in self.optimizer.param_groups]
                
                # Reset step_count to 0 so scheduler starts warmup from beginning
                unwrapped_scheduler.step_count = 0
                
                logger.info(f"Updated scheduler config: warmup={unwrapped_scheduler.warm_up_steps}, "
                          f"cycle_length={unwrapped_scheduler.cycle_length}, "
                          f"f_max={unwrapped_scheduler.f_max}, f_min={unwrapped_scheduler.f_min}")
                logger.info(f"Base learning rates: {[f'{lr:.2e}' for lr in unwrapped_scheduler.base_lrs]}")
                
                # Don't directly modify optimizer's lr! Let scheduler update it naturally on next step
                # Only log the target lr that scheduler will set
                initial_lrs = [base_lr * unwrapped_scheduler.f_max for base_lr in unwrapped_scheduler.base_lrs]
                logger.info(f"Reset scheduler step_count to 0 (will start warmup from next step)")
                logger.info(f"Target initial learning rates: {[f'{lr:.2e}' for lr in initial_lrs]}")
                logger.info(f"Learning rate will be updated by scheduler on first training step")
            
            # Log current learning rate (from checkpoint)
            current_lr = self.scheduler.get_last_lr()[0] if hasattr(self.scheduler, 'get_last_lr') else self.optimizer.param_groups[0]['lr']
            logger.info(f"Current learning rate after checkpoint load (will be overridden by scheduler): {current_lr:.2e}")
        elif self.scheduler is not None:
            # If not resetting scheduler, sync scheduler progress with global_step
            unwrapped_scheduler = self.scheduler
            if hasattr(self.scheduler, 'module'):
                unwrapped_scheduler = self.scheduler.module

            # Case 1: our custom LambdaLinearScheduler
            if hasattr(unwrapped_scheduler, 'step_count'):
                old_step_count = unwrapped_scheduler.step_count
                unwrapped_scheduler.step_count = self.global_step
                logger.info(f"Synchronized scheduler step_count: {old_step_count} -> {self.global_step}")

            # Case 2: diffusers_cosine wrapper with inner scheduler
            if hasattr(unwrapped_scheduler, 'inner') and hasattr(unwrapped_scheduler.inner, 'last_epoch'):
                try:
                    old_epoch = int(getattr(unwrapped_scheduler.inner, 'last_epoch', -1))
                except Exception:
                    old_epoch = -1
                # Align inner scheduler epoch with current global_step so schedule continues
                unwrapped_scheduler.inner.last_epoch = int(self.global_step)
                logger.info(f"Aligned diffusers scheduler last_epoch: {old_epoch} -> {self.global_step}")

            # Log current optimizer LR (authoritative)
            current_lr = self.optimizer.param_groups[0]['lr']
            logger.info(f"Current learning rate after checkpoint load (optimizer): {current_lr:.2e}")
    
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Single training step for UniDiffuser."""
        self.model.train()
        self.optimizer.zero_grad()
        
        first_frame = batch['first_frame'].to(self.device, dtype=self.dtype)          # [B, C, H, W]
        video_frames = batch['video_frames'].to(self.device, dtype=self.dtype)        # [B, num_video_frames, C, H, W]
        language_embeddings = batch['language_embedding']
        if language_embeddings is not None:
            language_embeddings = language_embeddings.to(self.device, dtype=self.dtype)
        state = batch.get('initial_state', None)
        if state is not None:
            state = state.to(self.device, dtype=self.dtype)      # [B, state_dim]
        actions = batch['action_sequence'].to(self.device, dtype=self.dtype)  # [B, action_chunk_size, action_dim]
        # Handle VLM inputs - it's a Dict[str, Tensor] from collate_fn
        vlm_inputs = batch['vlm_inputs']
        if vlm_inputs is not None:
            # Move all tensors in the VLM inputs dict to device
            vlm_inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in vlm_inputs.items()}
        
        # Forward pass through UniDiffuser
        # Handle DDP wrapper
        model = self.model.module if hasattr(self.model, 'module') else self.model
        loss_dict = model.training_step(
            first_frame=first_frame,
            video_frames=video_frames,
            state=state,
            actions=actions,
            language_embeddings=language_embeddings,  # For WAN cross attention
            vlm_inputs=vlm_inputs,  # Complete VLM inputs from dataset
            return_dict=True
        )
        
        total_loss = loss_dict['total_loss']
        
        # Backward pass (using accelerator if available)
        if hasattr(self, 'accelerator') and self.accelerator is not None:
            self.accelerator.backward(total_loss)
        else:
            total_loss.backward()
        
        # Gradient clipping
        grad_clip_norm = self.config.training.grad_clip_norm if hasattr(self.config.training, 'grad_clip_norm') else 1.0
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip_norm)
        
        # Optimizer step
        self.optimizer.step()
        
        if self.scheduler:
            self.scheduler.step()
        
        # Convert to float for logging
        metrics = {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
        
        return metrics
    
    def train(self, max_steps: int, resume_from: Optional[str] = None, val_interval: int = 500, reset_scheduler: Optional[bool] = None):
        """
        Main training loop.
        
        Args:
            max_steps: Maximum number of training steps
            resume_from: Path to checkpoint to resume from
            val_interval: Validation interval in steps
            reset_scheduler: If True, reset scheduler to new config. If None, use config.resume.reset_scheduler
        """
        # Load checkpoint if specified
        if resume_from:
            # Determine whether to reset scheduler
            if reset_scheduler is None:
                # Use config value if available, otherwise default to True
                if self.config is not None and hasattr(self.config, 'resume') and hasattr(self.config.resume, 'reset_scheduler'):
                    reset_scheduler = bool(self.config.resume.reset_scheduler)
                else:
                    reset_scheduler = True  # Default behavior
            
            self.load_checkpoint(resume_from, reset_scheduler=reset_scheduler)
        
        logger.info(f"Starting UniDiffuser training for {max_steps} steps")
        
        start_time = time.time()
        
        # Step-based training loop
        data_iter = iter(self.train_dataloader)
        epoch = 0
        
        while self.global_step < max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                # End of epoch, restart dataloader
                epoch += 1
                if hasattr(self.train_dataloader.sampler, 'set_epoch'):
                    self.train_dataloader.sampler.set_epoch(epoch)
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)
            
            if batch is None:  # Handle None batches
                continue
                
            step_start_time = time.time()

            # Training step
            metrics = self.train_step(batch)
            
            step_time = time.time() - step_start_time
            self.global_step += 1
            
            # Logging
            if self.global_step % self.log_interval == 0 and self.rank == 0:
                # Log optimizer LR as authoritative (may differ from inner scheduler get_last_lr)
                lrs = [g['lr'] for g in self.optimizer.param_groups]
                lr_main = lrs[0] if len(lrs) > 0 else 0.0
                lr_wan = lrs[1] if len(lrs) > 1 else lr_main
                
                log_str = (
                    f"Step {self.global_step}/{max_steps}, "
                    f"Loss: {metrics['total_loss']:.4f} "
                    f"(Video: {metrics['video_loss']:.4f}, Action: {metrics['action_loss']:.4f}), "
                    f"LR(main/wan): {lr_main:.2e}/{lr_wan:.2e}, Time: {step_time:.2f}s"
                )
                logger.info(log_str)
                
                # Log to WandB
                if "wandb" in self.report_to:
                    wandb.log({
                        **metrics,
                        'learning_rate_main': lr_main,
                        'learning_rate_wan': lr_wan,
                        'step_time': step_time,
                        'epoch': epoch,
                        'global_step': self.global_step,
                        })
                
                # Log to TensorBoard
                if self.tb_writer is not None:
                    for key, value in metrics.items():
                        self.tb_writer.add_scalar(f'train/{key}', value, self.global_step)
                    self.tb_writer.add_scalar('train/learning_rate_main', lr_main, self.global_step)
                    self.tb_writer.add_scalar('train/learning_rate_wan', lr_wan, self.global_step)
                    self.tb_writer.add_scalar('train/step_time', step_time, self.global_step)
                    self.tb_writer.add_scalar('train/epoch', epoch, self.global_step)

            # Validation: rank0-only local eval; then synchronize all processes
            if self.global_step % val_interval == 0 and self.val_dataloader is not None:
                if self.rank == 0:
                    val_metrics = evaluate_model(
                        self.model, self.val_dataloader, self.accelerator, self.config,
                        num_eval_batches=2
                    )
                    logger.info(f"Validation - Step {self.global_step}")
                    log_evaluation_metrics(val_metrics, self.tb_writer, self.accelerator, self.global_step)
                # Use explicit barrier with device_ids to avoid NCCL warnings
                if dist.is_available() and dist.is_initialized():
                    try:
                        dist.barrier(device_ids=[torch.cuda.current_device()])
                    except TypeError:
                        # Fallback for older PyTorch versions without device_ids argument
                        dist.barrier()
                
            # Save checkpoint
            if self.global_step % self.save_interval == 0:
                self.save_checkpoint()
        
        total_time = time.time() - start_time
        if self.rank == 0:
            logger.info(f"UniDiffuser training completed in {total_time:.2f}s ({self.global_step} steps)")
            self.save_checkpoint()

def create_model_and_optimizer(config: OmegaConf) -> tuple:
    """Create UniDiffuser model and optimizer from config."""
    # Create Motus config
    model_config = MotusConfig(
        wan_checkpoint_path=config.model.wan.checkpoint_path,
        vae_path=config.model.wan.vae_path,
        wan_config_path=config.model.wan.config_path,
        vlm_checkpoint_path=config.model.vlm.checkpoint_path,
        video_precision=config.model.wan.precision,
        action_state_dim=config.common.state_dim,
        action_dim=config.common.action_dim,
        # Action Expert configuration from config file
        action_expert_dim=config.model.action_expert.hidden_size,
        action_expert_ffn_dim_multiplier=config.model.action_expert.ffn_dim_multiplier,
        action_expert_norm_eps=config.model.action_expert.norm_eps,
        # Understanding Expert configuration from config file
        und_expert_hidden_size=config.model.und_expert.hidden_size,
        und_expert_ffn_dim_multiplier=config.model.und_expert.ffn_dim_multiplier,
        und_expert_norm_eps=config.model.und_expert.norm_eps,
        vlm_adapter_input_dim=config.model.und_expert.vlm.input_dim,
        vlm_adapter_projector_type=config.model.und_expert.vlm.projector_type,
        global_downsample_rate=config.common.global_downsample_rate,
        video_action_freq_ratio=config.common.video_action_freq_ratio,
        num_video_frames=config.common.num_video_frames,
        # Video dimensions from config
        video_height=config.common.video_height,
        video_width=config.common.video_width,
        batch_size=config.training.batch_size,
        video_loss_weight=config.model.loss_weights.video_loss_weight,
        action_loss_weight=config.model.loss_weights.action_loss_weight,
        training_mode=getattr(config, 'training_mode', 'finetune'),
        load_pretrained_backbones=getattr(config.model, 'load_pretrained_backbones', None),
    )
    
    # Create model (Accelerator will handle device placement and DDP)
    model = Motus(model_config)
    
    # Optimizer - parameter groups for separate WAN (video model) learning rate
    base_lr = float(config.training.learning_rate)
    wan_lr = float(getattr(config.training, 'wan_learning_rate', base_lr))

    # Collect WAN params explicitly (exclude VAE, we only train diffusion WAN)
    wan_params = [p for p in model.video_model.wan_model.parameters() if p.requires_grad]
    all_trainable = [p for p in model.parameters() if p.requires_grad]
    wan_param_ids = {id(p) for p in wan_params}
    other_params = [p for p in all_trainable if id(p) not in wan_param_ids]

    param_groups = []
    if len(other_params) > 0:
        param_groups.append({'params': other_params, 'lr': base_lr})
    if len(wan_params) > 0:
        param_groups.append({'params': wan_params, 'lr': wan_lr})

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=config.training.weight_decay,
        betas=(0.9, 0.95)
    )
    
    # Scheduler
    scheduler = create_scheduler(optimizer, config)
    
    return model, optimizer, scheduler

def create_dataloaders(config: OmegaConf, rank: int, world_size: int, seed: Optional[int] = None) -> tuple:
    """Create train and validation dataloaders from config."""
    train_dataset = create_dataset(config, val=False)
    val_dataset = create_dataset(config, val=True)

    sampler_seed = int(seed) if seed is not None else 0
    train_worker_init_fn = _worker_init_fn
    val_worker_init_fn = _worker_init_fn
    train_generator = None
    val_generator = None

    if seed is not None:
        train_worker_init_fn = _make_worker_init_fn(seed, rank, split_offset=0)
        val_worker_init_fn = _make_worker_init_fn(seed, rank, split_offset=10_000_000)
        train_generator = torch.Generator().manual_seed(seed)
        val_generator = torch.Generator().manual_seed(seed + 1)

    # Samplers
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            seed=sampler_seed,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            seed=sampler_seed,
        )
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
        worker_init_fn=train_worker_init_fn,
        generator=train_generator,
        collate_fn=collate_fn,
        drop_last=True,
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=config.system.num_workers,
        pin_memory=config.system.pin_memory,
        worker_init_fn=val_worker_init_fn,
        generator=val_generator,
        collate_fn=collate_fn,
        drop_last=False,
    )
    
    return train_dataloader, val_dataloader

def main():
    parser = argparse.ArgumentParser(description="Train Three-Modal UniDiffuser Model")
    
    # Configuration file
    parser.add_argument("--config", type=str, 
                       default="configs/aloha_agilex_2.yaml",
                       help="Path to configuration file")
    
    # System settings
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Override checkpoint directory")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    
    # Logging settings
    parser.add_argument("--report_to", type=str, default=None, 
                       choices=["wandb", "tensorboard", "all", "none"],
                       help="Logging backends to use")
    parser.add_argument("--wandb_project", type=str, default=None, help="Override WandB project name")
    parser.add_argument("--run_name", type=str, default=None, help="Override run name")
    
    # DeepSpeed settings
    parser.add_argument("--deepspeed", type=str, default=None, help="Path to DeepSpeed config file")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument("--seed", type=int, default=None, help="Training seed (overrides config.training.seed)")
    
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
    # Decide backbone loading policy:
    # If resuming or finetuning from a pretrain checkpoint, skip loading WAN/VLM pretrained weights.
    try:
        if (getattr(config.resume, 'checkpoint_path', None) or
            (hasattr(config, 'finetune') and getattr(config.finetune, 'checkpoint_path', None))):
            config.model.load_pretrained_backbones = False
    except Exception:
        pass
    
    # Extract dataset name from config file path for checkpoint organization
    config_filename = os.path.basename(args.config)  # e.g., "ac_one.yaml"
    bench_name = os.path.splitext(config_filename)[0]  # e.g., "ac_one"
    
    # Update checkpoint directory to include dataset name
    base_checkpoint_dir = config.system.checkpoint_dir
    config.system.checkpoint_dir = os.path.join(base_checkpoint_dir, bench_name)
    
    # Create the dataset directory if it doesn't exist
    os.makedirs(config.system.checkpoint_dir, exist_ok=True)
    
    # Initialize Accelerator with DeepSpeed (if provided)
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
    setup_logging(rank, args.log_level)

    seed = resolve_training_seed(args, config)
    if seed is not None:
        config.training.seed = seed
        set_seed(seed, device_specific=True)
        logger.info(f"Using training seed: {seed} (rank {rank} effective seed {seed + rank})")
    else:
        logger.info("Training seed not set; run will be non-deterministic")
    
    # Handle report_to settings - expand "all" to individual backends
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
        run_name = f"unidiffuser_{config.dataset.type}_bs{config.training.batch_size}_lr{config.training.learning_rate}"
    
    # Update checkpoint directory to include run name
    config.system.checkpoint_dir = os.path.join(config.system.checkpoint_dir, run_name)
    logger.info(f"Dataset: {bench_name}")
    logger.info(f"Checkpoints will be saved to: {config.system.checkpoint_dir}")
    
    # Initialize TensorBoard writer
    tb_writer = None
    if rank == 0 and "tensorboard" in report_to:
        tb_log_dir = os.path.join(config.system.checkpoint_dir, config.logging.tensorboard_log_dir)
        tb_writer = SummaryWriter(log_dir=tb_log_dir)
        logger.info(f"TensorBoard logs will be saved to: {tb_log_dir}")
        config_dict = OmegaConf.to_container(config, resolve=True)
        tb_writer.add_text('config', yaml.dump(config_dict))
    
    # Initialize WandB
    if rank == 0 and "wandb" in report_to:
        wandb.init(
            project=config.logging.wandb_project,
            config=OmegaConf.to_container(config, resolve=True),
            name=run_name,
        )
    
    try:
        # Create model and optimizer
        logger.info("Creating UniDiffuser model and optimizer...")
        model, optimizer, scheduler = create_model_and_optimizer(config)

        # Optional: load finetune weights for partial init
        finetune_ckpt = getattr(config.finetune, 'checkpoint_path', None) if hasattr(config, 'finetune') else None
        if getattr(config, 'training_mode', 'finetune') == 'finetune' and finetune_ckpt:
            logger.info(f"Loading finetune weights from {finetune_ckpt} (partial)...")
            try:
                (model.module if hasattr(model, 'module') else model).load_pretrain_weights(finetune_ckpt)
                logger.info("Finetune weights loaded (partial).")
            except Exception as e:
                logger.error(f"Failed to load finetune weights: {e}")
        
        # Create dataloaders
        logger.info("Creating dataloaders...")
        train_dataloader, val_dataloader = create_dataloaders(config, rank, world_size, seed=seed)
        
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
        
        # Prepare everything with accelerator (do not prepare val_dataloader to enable rank0-only local eval)
        logger.info("Preparing model, optimizer, and dataloaders with Accelerator...")
        model, optimizer, train_dataloader, scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, scheduler
        )
        
        # Create trainer
        trainer = UniDiffuserTrainer(
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
            resume_from=config.resume.checkpoint_path,
            val_interval=config.system.val_interval
        )
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        print(f"[CRITICAL ERROR] Training failed: {e}")
        print("Full traceback:")
        traceback.print_exc()
        raise
    finally:
        # Clean up resources
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        if rank == 0 and "wandb" in report_to:
            wandb.finish()
        if tb_writer is not None:
            tb_writer.close()

if __name__ == "__main__":
    main()