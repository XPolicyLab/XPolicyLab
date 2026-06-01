import os
import logging
from datetime import timedelta

from contextlib import nullcontext
from pathlib import Path
import hydra
import torch
import torch.distributed as dist

# TODO: fix bnb version
try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None

from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration

from ema_pytorch import EMA
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers.utils.versions import require_version

from galaxea_fm.data.base_lerobot_dataset import BaseLerobotDataset
from galaxea_fm.processors.base_processor import BaseProcessor
from galaxea_fm.models.base_policy import BasePolicy
from galaxea_fm.models.galaxea_zero.galaxea_zero_policy import GalaxeaZeroPolicy
from galaxea_fm.utils.get_scheduler import get_scheduler
from galaxea_fm.utils.logging_config import (
    setup_logging,
    log_allocated_gpu_memory,
    log_amp_config,
)
from galaxea_fm.utils.pytorch_utils import set_global_seed
from galaxea_fm.utils.dist import ResumableDistributedSampler
from galaxea_fm.utils.train_utils import MFUTracker, init_experiment_tracker, register_graceful_exit
from galaxea_fm.utils.normalizer import (
    load_dataset_stats_from_json, 
    save_dataset_stats_to_json, 
    search_dataset_stats_cache_json, 
)
from galaxea_fm.utils.config_resolvers import register_default_resolvers
from galaxea_fm.utils.train_utils import set_global_monitor, get_global_monitor
from galaxea_fm.utils.tqdm import tqdm
from galaxea_fm.utils.git_info import save_git_info, GitInfoError

register_default_resolvers()
logger = get_logger(__name__)
require_version("datasets==3.6.0", "To fix: uv pip install datasets==3.6.0")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def finetune(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    output_dir = Path(cfg.output_dir)

    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    project_config = ProjectConfiguration(project_dir=str(Path(cfg.output_dir)))
    init_process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=2))
    accelerator = Accelerator(
        mixed_precision="bf16" if cfg.model.enable_bf16_training else "no",
        project_config=project_config,
        kwargs_handlers=[init_process_group_kwargs],
        log_with=cfg.logger.type,
    )
    register_graceful_exit(accelerator)
    torch.cuda.set_device(device_id := accelerator.local_process_index)
    torch.cuda.empty_cache()

    # Pass is_main_process=True for setup_logging to enable distributed debugging using accelerate's logger
    setup_logging(log_level=logging.INFO, is_main_process=accelerator.is_main_process)
    logger.info(f"Output directory: {output_dir}")
    log_amp_config(logger, accelerator)
    init_experiment_tracker(cfg, accelerator, output_dir)
    set_global_monitor()
    worker_init_fn = set_global_seed(cfg.seed, get_worker_init_fn=True)  # Set seed BEFORE model creation for reproducibility

    # HACK: Select checkpoint functions based on format config. Legacy will be deprecated
    if cfg.get("load_legacy_checkpoint", True):
        from galaxea_fm.utils.load_pretrained_resumed import save_checkpoint
        from galaxea_fm.utils.load_pretrained_resumed_legacy import (
            load_pretrained_model,
            load_embedded_dataset_stats, 
            resume_checkpoint, 
        )
    else:
        from galaxea_fm.utils.load_pretrained_resumed import (
            load_pretrained_model,
            load_embedded_dataset_stats, 
            save_checkpoint, 
            resume_checkpoint, 
        )

    # Save git information at training start
    if accelerator.is_main_process:
        try:
            git_info_path = save_git_info(output_dir=output_dir)
            logger.info(f"Git info saved to: {git_info_path}")
        except GitInfoError as e:
            logger.warning(f"Could not save git info: {e}")

    model: BasePolicy = instantiate(cfg.model.model_arch)

    if cfg.model.model_weights_to_bf16:
        model = model.to(torch.bfloat16)

    use_ema = cfg.model.use_ema
    if use_ema:
        ema_model = EMA(
            model, 
            update_after_step=cfg.model.ema.update_after_step, 
            beta=cfg.model.ema.power,
        ).to(device_id) 

    if cfg.model.use_sync_bn and accelerator.num_processes > 1:
        logger.info("Use sync batch norm.")
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if cfg.model.use_torch_compile:  # model being compiled in the first batch which takes some time
        # torch._dynamo.config.suppress_errors = True
        model = torch.compile(model, mode="default")

    model = model.to(device_id)

    log_allocated_gpu_memory(logger, stage="loading model", device=0)

    train_dataset: BaseLerobotDataset = instantiate(cfg.data.dataset, is_training_set=True)
    eval_dataset: BaseLerobotDataset = instantiate(cfg.data.dataset, is_training_set=False)
    train_processor: BaseProcessor = instantiate(cfg.data.processor)
    eval_processor: BaseProcessor = instantiate(cfg.data.processor)
    train_dataset.set_processor(train_processor)
    eval_dataset.set_processor(eval_processor)

    train_sampler = ResumableDistributedSampler(
        train_dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=True,
        batch_size=cfg.model.batch_size,
    )
    eval_sampler = DistributedSampler(
        eval_dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=False,
    )
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=cfg.model.batch_size, 
        sampler=train_sampler,
        shuffle=False,
        num_workers=cfg.model.num_workers, 
        pin_memory=cfg.model.pin_memory, 
        persistent_workers=cfg.model.persistent_workers, 
        worker_init_fn=worker_init_fn, 
    )
    eval_dataloader = DataLoader(
        eval_dataset, 
        batch_size=cfg.batch_size_val, 
        sampler=eval_sampler, 
        shuffle=False, 
        num_workers=cfg.model.num_workers, 
        pin_memory=cfg.model.pin_memory, 
        persistent_workers=cfg.model.persistent_workers, 
        worker_init_fn=worker_init_fn, 
    )

    if cfg.model.max_epochs:
        assert not cfg.model.max_steps, "Cannot set both `max_epochs` and `max_steps`!"
        steps_per_epoch = len(train_dataloader) // cfg.model.grad_accumulation_steps
        max_steps = steps_per_epoch * cfg.model.max_epochs
    else:
        max_steps = cfg.model.max_steps
    
    # Determine whether MFU tracking is supported before wrapping with DDP
    use_mfu_tracker = isinstance(model, GalaxeaZeroPolicy)

    # Wrap model in PyTorch DDP Wrapper for Multi-GPU Training
    model = DDP(model, device_ids=[device_id], find_unused_parameters=cfg.model.find_unused_parameters, gradient_as_bucket_view=True)

    # Create Optimizer
    param_groups = model.module.get_optim_param_groups(cfg.model.learning_rate, cfg.model.weight_decay)
    # Convert OmegaConf objects to plain Python objects to avoid serialization issues
    betas = tuple(cfg.model.betas)
    if cfg.model.use_8bit_optimizer:
        assert bnb is not None, "bitsandbytes is not installed, cannot use 8bit optimizer"
        optimizer = bnb.optim.AdamW8bit(param_groups, betas=betas)
    else:
        optimizer = AdamW(param_groups, betas=betas)
    
    if cfg.model.lr_scheduler_type == "OneCycleLR":
        from torch.optim.lr_scheduler import OneCycleLR
        scheduler = OneCycleLR(
            optimizer=optimizer,
            max_lr=cfg.model.learning_rate,
            total_steps=max_steps,
            pct_start=cfg.model.pct_start,
            anneal_strategy=cfg.model.anneal_strategy,
            div_factor=cfg.model.div_factor,
            final_div_factor=cfg.model.final_div_factor,
        )
    else:
        scheduler = get_scheduler(
            name=cfg.model.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=cfg.model.warmup_steps,
            num_training_steps=max_steps,
        )

    # Resume training state
    if cfg.resume_ckpt:
        resume_dataloader = True
        step, epoch, batch_idx = resume_checkpoint(
            checkpoint_path=cfg.resume_ckpt,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ema_model=ema_model if use_ema else None,
            device_id=device_id,
        )
        dataset_stats = load_embedded_dataset_stats(cfg.resume_ckpt)
        logger.info(f"Resume training from step {step}, epoch {epoch}, batch_idx {batch_idx}")
    else:
        resume_dataloader = False
        step, epoch, batch_idx = 0, 0, 0

        if cfg.model.pretrained_ckpt:
            logger.info(f"Loading pretrained checkpoint from {cfg.model.pretrained_ckpt}")
            load_pretrained_model(cfg.model.pretrained_ckpt, model)
            if use_ema:                                                                                                                                                                                                                             
                ema_model.ema_model.load_state_dict(model.module.state_dict())                                                                                                                                                                      
                logger.info("Synced EMA model to pretrained weights") 
        else:
            logger.info(f"Train model from initialization")

        if cfg.model.pretrained_ckpt and cfg.model.use_pretrained_norm_stats:
            logger.info(f"Use pretrained dataset stats from {cfg.model.pretrained_ckpt}")
            dataset_stats = load_embedded_dataset_stats(cfg.model.pretrained_ckpt)
        else:
            logger.info("Calculate stats from dataset instead of loading from pretrained")
            if accelerator.is_main_process:
                exist_cache, cache_path = search_dataset_stats_cache_json(cfg.dataset_stats_cache_dir, cfg.data)
                if exist_cache:
                    logger.info(f"  Use dataset stats cache file {cache_path}")
                    dataset_stats = load_dataset_stats_from_json(cache_path)
                else:
                    logger.info("  No cached stats found, computing from dataset ...")
                    dataset_stats = train_dataset.get_dataset_stats(train_processor)
                    save_dataset_stats_to_json(dataset_stats, cache_path)
                    logger.info(f"  Saved dataset stats cache: {cache_path}")
            else:
                dataset_stats = None

            container = [dataset_stats]
            dist.broadcast_object_list(container, src=0)
            dataset_stats = container[0]
        
    train_processor.set_normalizer_from_stats(dataset_stats)
    eval_processor.set_normalizer_from_stats(dataset_stats)
    if accelerator.is_main_process:
        save_dataset_stats_to_json(dataset_stats, output_dir / "dataset_stats.json")

    # Initialize MFU Tracker
    mfu_tracker = None
    if accelerator.is_main_process:
        effective_batch_size = cfg.model.batch_size * cfg.model.grad_accumulation_steps * dist.get_world_size()
        if use_mfu_tracker:
            mfu_tracker = MFUTracker(
                model=model.module,
                batch_size=effective_batch_size,
                device_id=device_id,
                update_interval=cfg.logger.log_steps,
                world_size=dist.get_world_size(),
                dtype=torch.bfloat16 if cfg.model.enable_bf16_training else torch.float32,
            )
            mfu_tracker.reset(step)
        else:
            logger.info(f"Skipping MFU tracker as the policy is not an instance of class {GalaxeaZeroPolicy.__name__}.")

    accelerator.wait_for_everyone()
    # Train!
    training_done = False
    with tqdm.tqdm(initial=step, total=max_steps, leave=False, dynamic_ncols=True) as progress:
        while not training_done:
            train_sampler.set_epoch(epoch)
            if resume_dataloader:
                logger.info(f"Resume dataloader state from batch_idx {batch_idx} of epoch {epoch}")
                train_sampler.set_start_batch(batch_idx)
                resume_dataloader = False
            else:
                batch_idx = 0
                train_sampler.set_start_batch(0)

            data_iter = iter(train_dataloader)
            model.train()
            optimizer.zero_grad()
            while batch_idx < len(train_dataloader):
                batch = next(data_iter)
                # Turn off sync when is not optimizer step
                is_optimizer_step = (batch_idx + 1) % cfg.model.grad_accumulation_steps == 0
                sync_ctx = model.no_sync() if not is_optimizer_step else nullcontext()
                with sync_ctx:
                    with accelerator.autocast():
                        # AMP Best Practice: Keep input in FP32, let autocast handle the conversion
                        # No manual dtype conversion needed here - autocast will automatically
                        # cast operations to the appropriate precision
                        loss, loss_value_dict = model(batch)
                    # Normalize loss to account for gradient accumulation
                    normalized_loss = loss / cfg.model.grad_accumulation_steps
                    normalized_loss.backward()

                batch_idx += 1

                if is_optimizer_step:
                    # TODO : rename it into grad clip norm
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.model.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    progress.set_description(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")
                    progress.update()
                    progress.refresh()

                    if use_ema:
                        ema_model.update()

                    step += 1

                    # Log metrics on optimizer steps
                    if step % cfg.logger.log_steps == 0:
                        # Ensure values are plain Python numbers
                        log_dict = {k: (v.item() if hasattr(v, "item") else float(v)) for k, v in loss_value_dict.items()}
                        log_dict.update({
                            "lr/encoder": optimizer.param_groups[0]["lr"],
                            "lr/model": optimizer.param_groups[1]["lr"],
                            "grad_norm": grad_norm.item(),
                        })

                        # Add MFU metrics if tracker is available
                        if mfu_tracker is not None:
                            mfu_metrics = mfu_tracker.compute_metrics(step)
                            log_dict.update(mfu_metrics)
                        global_monitor = get_global_monitor()
                        if global_monitor is not None:
                            log_dict.update(global_monitor.get_metrics())

                        accelerator.log(log_dict, step=step)

                # Save checkpoint in the main process
                if step > 0 and (step % cfg.checkpointing_steps) == 0:
                    if accelerator.is_main_process:
                        logger.info(f"Saving model checkpoint for step {step} ...")
                        unwrapped_model = accelerator.unwrap_model(model)
                        checkpoint_path = output_dir / "checkpoints" / f"step_{step}"
                        save_checkpoint(
                            path=checkpoint_path,
                            step=step,
                            epoch=epoch,
                            batch_idx=batch_idx,
                            model=unwrapped_model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            ema_model=ema_model if use_ema else None,
                            dataset_stats=dataset_stats,
                            cfg=cfg,
                        )

                    # Block on main process checkpointing
                    accelerator.wait_for_everyone()

                # Stop training when max_steps is reached
                if step >= max_steps:
                    logger.info(f"Max step {max_steps} reached, stop training ...")
                    training_done = True
                    break

            epoch += 1

    if accelerator.is_main_process:
        logger.info(f"Saving model checkpoint for step {step} ...")
        unwrapped_model = accelerator.unwrap_model(model)
        checkpoint_path = output_dir / "checkpoints" / f"step_{step}"
        save_checkpoint(
            path=checkpoint_path,
            step=step,
            epoch=epoch,
            batch_idx=batch_idx,
            model=unwrapped_model,
            optimizer=optimizer,
            scheduler=scheduler,
            ema_model=ema_model if use_ema else None,
            dataset_stats=dataset_stats,
            cfg=cfg,
        )

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    finetune()
