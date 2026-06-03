"""
Distributed training entry for MemoryMatters execution module.

Design goals:
- Load config from source/config/execution_module_train.yaml.
- Heavy inline commentary for readability and auditability.
- Chatty terminal logging via cprint for step-by-step visibility.
"""

import sys, os
sys.path.append(os.getcwd())
from datetime import datetime
from pathlib import Path
from typing import Optional

# Silence HF progress globally before heavy imports.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import math
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from PIL import Image
from omegaconf import OmegaConf
from termcolor import cprint
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import wandb

try:
    from datasets import disable_progress_bar as hf_disable_progress_bar
    from datasets.utils.logging import set_verbosity_error as hf_set_verbosity_error
except Exception:  # pragma: no cover - optional dependency
    hf_disable_progress_bar = None
    hf_set_verbosity_error = None


from source.utils import BOLD, RESET_BOLD, RESET
from source.training.utils.classifier_metrics import ClassifierMetricsAggregator, ClassifierMetricsSync
from source.training.utils.cli_utils import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config_with_train_variant, parse_args
from source.training.utils.dataloader_builder import (
    # build_pretrain_dataloader,
    # build_rmbench_dataloader,
    build_rmbench_random_episode_dataloader,
)
from source.training.utils.debug_utils import log_debug_batch

from source.models.execution_module.memorymatters_executor import MemoryMattersExecutor
from source.training.utils.trainer_tools import (
    CheckpointManager,
    TrainerUtils,
    build_classifier_wandb_payload,
    build_param_lr_groups,
    build_scheduler,
    destroy_distributed,
    extract_scalar_metrics,
    log_step_metrics,
    rank0_print,
    setup_distributed,
    setup_seed,
)


def setup_wandb_run(cfg, rank: int):
    """
    Initialize wandb on rank-0 if enabled in config; returns a wandb run or None.
    """
    trainer_cfg = cfg.get("trainer", {})
    if not trainer_cfg.get("enable_wandb", False):
        return None
    if rank != 0:
        return None
    if wandb is None:
        cprint("[wandb] wandb is not installed; skipping wandb logging.", "yellow")
        return None

    run_name = trainer_cfg.get("wandb_run_name") or f"execution_module_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    entity = cfg.get("wandb_entity")
    # If the entity is placeholder/empty, let wandb use the currently logged-in account.
    if not entity or str(entity).strip() in {"your_wandb_entity", ""}:
        entity = None

    try:
        wandb_run = wandb.init(
            project=cfg.get("wandb_project", "default"),
            entity=entity,
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        cprint(
            f"[wandb] Initialized run -> project = {BOLD}{wandb_run.project}{RESET_BOLD}; name = {BOLD}{wandb_run.name}{RESET_BOLD}",
            "green",
        )
        return wandb_run
    except Exception as exc:  # pragma: no cover - defensive logging path
        cprint(f"[wandb] Init failed, disabling wandb logging: {exc}", "red")
        return None


def create_model(cfg, device: torch.device, resume_path: Optional[str] = None) -> torch.nn.Module:
    """Instantiate the execution module (CUDA required), optionally restoring from checkpoint."""
    if device.type != "cuda":
        raise RuntimeError("MemoryMattersExecutor requires CUDA.")
    model = MemoryMattersExecutor(cfg, device=device)
    model.to(device)

    if resume_path:
        ckpt_path = Path(resume_path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")
        try:
            payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        except TypeError:
            # PyTorch <2.6 fallback without weights_only arg
            payload = torch.load(ckpt_path, map_location=device)
        state_dict = payload.get("model_state_dict", payload)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        warn_parts = []
        if missing:
            warn_parts.append(f"missing={len(missing)}")
        if unexpected:
            warn_parts.append(f"unexpected={len(unexpected)}")
        suffix = f" (load_state_dict: {', '.join(warn_parts)})" if warn_parts else ""
        rank0_print(f"[model] Loaded checkpoint from {BOLD}{ckpt_path}{RESET_BOLD}{suffix}", "green" if not warn_parts else "yellow")
    else:
        rank0_print("[model] No resume provided; model weights are randomly initialized.", "yellow")
    return model


def _set_qwen_requires_grad(model: torch.nn.Module, requires_grad: bool) -> None:
    """Toggle Qwen submodule grads for warmup freeze/unfreeze."""
    target = model.module if hasattr(model, "module") else model
    if hasattr(target, "qwen_model"):
        for p in target.qwen_model.parameters():
            p.requires_grad = requires_grad
    else:
        raise AttributeError("Model does not have attribute 'qwen_model' to set requires_grad.")

def train_steps(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    rank: int,
    total_steps: int,
    grad_clip_norm: float = 0.0,
    global_step: int = 0,
    log_interval: int = 10,
    warmup_steps: int = 0,
    wandb_run=None,
    checkpoint_manager: Optional[CheckpointManager] = None,
    is_debug: bool = False,
    classifier_metrics_sync: Optional[ClassifierMetricsSync] = None,
    eval_action_interval: int = 0,
    eval_dataset=None,
) -> int:
    """
    Step-based training loop (no epoch semantics). Iterates dataloader as needed until total_steps are consumed.
    Returns updated global_step.
    is_debug adds verbose per-sample logging for order verification.
    """
    model.train()
    data_iter = iter(dataloader)
    upload_media = 0
    sampler_round = 0
    # is_qwen_frozen = warmup_steps > 0
    # if is_qwen_frozen:
    #     _set_qwen_requires_grad(model, False)
    #     rank0_print(f"[warmup] Freezing Qwen params for first {warmup_steps} steps", "yellow")
    is_rank0 = (not dist.is_initialized()) or dist.get_rank() == 0
    progress_bar = tqdm(
        total=total_steps,
        initial=0,
        disable=rank != 0,
        dynamic_ncols=True,
        desc="train",
    )
    for _ in range(total_steps):
        batch = None
        try:
            batch = next(data_iter)
        except StopIteration:
            pass
        need_reset = batch is None

        if need_reset:
            epoch_metrics = (
                classifier_metrics_sync.finalize_epoch(global_step) if classifier_metrics_sync is not None else None
            )
            window_metrics = epoch_metrics.get("window") if epoch_metrics else None
            aggregated_classifier = epoch_metrics.get("aggregated") if epoch_metrics else None
            should_log_classifier = (
                window_metrics is not None
                and (not dist.is_initialized() or dist.get_rank() == 0)
                and global_step % max(1, log_interval) == 0
            )
            if should_log_classifier and wandb_run is not None:
                merged_log = dict(window_metrics)
                if aggregated_classifier is not None:
                    merged_log.update(aggregated_classifier)
                wandb_payload = build_classifier_wandb_payload(merged_log)
                wandb_run.log(wandb_payload, step=global_step)

            rank0_print(f"[data] Resetting dataloader at step {BOLD}{global_step}{RESET_BOLD}", "yellow")
            data_iter, sampler_round = TrainerUtils.reset_dataloader(dataloader, sampler_round)
            try:
                batch = next(data_iter)
            except StopIteration:
                # If even after reset no data, abort loop.
                break

        if is_debug:
            log_debug_batch(global_step + 1, sampler_round, rank, batch)

        # Unfreeze Qwen after warmup
        # if is_qwen_frozen and global_step >= warmup_steps:
        #     _set_qwen_requires_grad(model, True)
        #     optimizer.zero_grad(set_to_none=True)
        #     is_qwen_frozen = False
        #     rank0_print("[warmup] Unfroze Qwen params post-warmup", "green")

        # --- Forward + loss ---
        outputs = model(batch)
        loss = outputs["loss"]

        # --- Backward + optimization ---
        optimizer.zero_grad()
        loss.backward()

        # --- Gradient norms (raw vs clipped) and stability gate ---
        raw_grad_norm = None
        clipped_grad_norm = None
        should_log_step = is_rank0 and (global_step % max(1, log_interval) == 0)
        do_wandb = wandb_run is not None

        if grad_clip_norm and grad_clip_norm > 0:
            # clip_grad_norm_ returns the total norm BEFORE clipping and applies clipping
            raw_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            # compute norm AFTER clipping only when we will log
            if should_log_step and do_wandb:
                sq_sum = torch.zeros((), device=device)
                for p in model.parameters():
                    if p.grad is not None:
                        sq_sum = sq_sum + torch.norm(p.grad, p=2) ** 2
                clipped_grad_norm = torch.sqrt(sq_sum)
        else:
            # no clipping requested: raw == clipped (when logging)
            sq_sum = torch.zeros((), device=device)
            for p in model.parameters():
                if p.grad is not None:
                    sq_sum = sq_sum + torch.norm(p.grad, p=2) ** 2
            raw_grad_norm = torch.sqrt(sq_sum)
            if should_log_step and do_wandb:
                clipped_grad_norm = raw_grad_norm

        # Skip this step if raw gradients are non-finite (inf/nan)
        has_finite_grads = True
        if raw_grad_norm is not None:
            has_finite_grads = bool(torch.isfinite(raw_grad_norm).item())
        else:
            has_finite_grads = all(
                p.grad is None or bool(torch.isfinite(p.grad).all().item())
                for p in model.parameters()
            )

        if not has_finite_grads:
            optimizer.zero_grad()
            if is_rank0:
                rank0_print(
                    f"[train] Skipping step due to non-finite gradients at {BOLD}{global_step}{RESET_BOLD}",
                    "red",
                )
                if wandb_run is not None:
                    wandb_run.log({"Grad/grad_nonfinite_skip": 1}, step=global_step)
            continue

        # Log both raw and clipped gradient norms when finite
        if should_log_step and wandb_run is not None:
            payload = {}
            if raw_grad_norm is not None and bool(torch.isfinite(raw_grad_norm).item()):
                payload["Grad/grad_norm_raw"] = raw_grad_norm.item()
            if clipped_grad_norm is not None and bool(torch.isfinite(clipped_grad_norm).item()):
                payload["Grad/grad_norm_clipped"] = clipped_grad_norm.item()
                # backward-compatible single key
                payload["Grad/grad_norm"] = clipped_grad_norm.item()
            
            if upload_media <= 0:
                media_images = []
                for i in range (0, len (batch), 8):
                    imgs = batch[i].get("image")
                    media_images.append(wandb.Image(imgs[0]))
                
                if media_images:
                    payload["Media/batch_first_last"] = media_images
                
                upload_media = 100 # reset
            
            upload_media -= 1
            if payload:
                wandb_run.log(payload, step=global_step)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        global_step += 1
        
        lr = optimizer.param_groups[0]["lr"]

        scalar_losses = extract_scalar_metrics(outputs)
        # Sync loss across ranks for logging (do not reuse for backward to avoid double scaling).
        loss_for_log = scalar_losses.get("loss")
        if loss_for_log is not None and dist.is_initialized():
            loss_tensor = torch.tensor(loss_for_log, device=device, dtype=torch.float32)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            loss_global_mean = (loss_tensor / dist.get_world_size()).item()
            scalar_losses["loss"] = loss_global_mean
        
        aggregated_classifier = None
        if classifier_metrics_sync is not None:
            classifier_metrics_sync.accumulate_batch(scalar_losses, batch_size=len(batch))
            aggregated_classifier = classifier_metrics_sync.maybe_sync(global_step, force=False)

        log_metrics = dict(scalar_losses)
        if aggregated_classifier is not None:
            log_metrics.update(aggregated_classifier)
            # Override per-rank classifier metrics with global aggregates for logging/W&B consistency.
            remap = {
                "classifier_accuracy": "classifier_accuracy_recent",
                "classifier_precision": "classifier_precision_recent",
                "classifier_recall": "classifier_recall_recent",
                "classifier_f1_score": "classifier_f1_score_recent",
                "classifier_prate": "classifier_prate_recent",
            }
            for base_key, global_key in remap.items():
                if global_key in aggregated_classifier:
                    log_metrics[base_key] = aggregated_classifier[global_key]
            # Attach count for console only; will be stripped before W&B.
            if aggregated_classifier.get("classifier_count_recent") is not None:
                log_metrics["classifier_count"] = aggregated_classifier["classifier_count_recent"]
            if aggregated_classifier.get("classifier_window_size") is not None:
                log_metrics["classifier_window_size"] = aggregated_classifier["classifier_window_size"]

        wandb_payload = build_classifier_wandb_payload(log_metrics)

        log_step_metrics(wandb_payload, lr, global_step, log_interval, wandb_run, console_only=log_metrics)
        
        if progress_bar is not None and rank == 0:
            progress_bar.update(1)

        if checkpoint_manager is not None:
            checkpoint_manager.maybe_save(model, optimizer, scheduler, global_step)

    if progress_bar is not None:
        progress_bar.close()

    # Ensure the last open epoch is aggregated at training end
    if classifier_metrics_sync is not None:
        final_metrics = classifier_metrics_sync.flush(global_step)
        final_window_metrics = final_metrics.get("window")
        final_aggregated = final_metrics.get("aggregated")
        should_log_final = (
            final_window_metrics is not None
            and (not dist.is_initialized() or dist.get_rank() == 0)
            and global_step % max(1, log_interval) == 0
        )
        if should_log_final:
            if wandb_run is not None and (not dist.is_initialized() or dist.get_rank() == 0):
                merged_final = dict(final_window_metrics)
                if final_aggregated is not None:
                    merged_final.update(final_aggregated)
                wandb_payload = build_classifier_wandb_payload(merged_final)
                wandb_run.log(wandb_payload, step=global_step)

    return global_step


def main() -> None:
    """
    End-to-end launcher:
    - load config from fixed path
    - init distributed
    - build data/model/optim/scheduler
    - run training loop
    """
    args = parse_args()
    # Load config from source/config/execution_module_train.yaml
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file {config_path} not found.")
        cfg = OmegaConf.load(config_path)
        config_name = config_path.stem
    else:
        cfg, config_path, config_name = load_config_with_train_variant()
    trainer_cfg = cfg.get("trainer", {})
    is_debug = bool(cfg.get("is_debug", False))

    # --- Setup seeds/distributed devices ---
    setup_seed(cfg.get("seed", 42))
    distributed, rank, world_size, device = setup_distributed()
    # Avoid shared-memory FD exhaustion with many workers
    mp.set_sharing_strategy("file_system")
    if rank == 0:
        cprint("[torch] set_sharing_strategy -> file_system", "cyan")

    # Reduce third-party verbosity on non-main ranks.
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if hf_disable_progress_bar is not None:
        hf_disable_progress_bar()
    if hf_set_verbosity_error is not None and rank != 0:
        hf_set_verbosity_error()

    rank0_print(f"[config] Config file: {BOLD}{config_path}{RESET}", "green")
    rank0_print(
        f"[dist] Distributed = {BOLD}{distributed}{RESET_BOLD}; world_size = {BOLD}{world_size}{RESET_BOLD}; device = {BOLD}{device}{RESET_BOLD}",
        "green",
    )

    # --- Data ---
    if is_debug:
        # Override for quick debug runs.
        trainer_cfg["batch_size"] = 25
        trainer_cfg["train_steps"] = 50
        trainer_cfg["log_interval"] = 10
        # trainer_cfg["eval_action_interval"] = 10
        rank0_print(f"[debug] is_debug = True -> forcing batch_size = {trainer_cfg['batch_size']}, train_steps = {trainer_cfg['train_steps']}, log_interval = {trainer_cfg['log_interval']}, eval_action_interval = {trainer_cfg['eval_action_interval']}", "yellow")

    dataloader_cfg = cfg.get("dataloader", {}) if hasattr(cfg, "get") else {}
    dataloader_type = dataloader_cfg.get("type", "base") if dataloader_cfg else "base"
    if dataloader_type == "rmbench_random_episode":
        builder = build_rmbench_random_episode_dataloader
    else:
        rank0_print(f"[data] Unsupported dataloader.type = {BOLD}{dataloader_type}{RESET_BOLD}", "red")
        raise ValueError(f"Unsupported dataloader.type = {dataloader_type}")

    memory_dataset, dataloader, dataset_source, dataset_breakdown = builder(
        cfg,
        rank=rank,
        world_size=world_size,
        norm_stats_path=dataloader_cfg.get("norm_stats_path", None),
    )
    train_steps_target = int(trainer_cfg.get("train_steps", len(dataloader)))
    if train_steps_target <= 0:
        raise ValueError("trainer.train_steps must be > 0 when using step-based training.")
    if rank == 0:
        dataset_size = len(memory_dataset) if hasattr(memory_dataset, "__len__") else "?"
        steps_per_epoch_est = len(dataloader) if hasattr(dataloader, "__len__") else "?"
        per_device_batch = trainer_cfg.get("batch_size", 2)
        effective_batch = per_device_batch * world_size
        est_epoch_steps = (
            steps_per_epoch_est
            if isinstance(steps_per_epoch_est, int) and steps_per_epoch_est > 0
            else (
                math.ceil(dataset_size / effective_batch)
                if isinstance(dataset_size, int) and isinstance(effective_batch, (int, float)) and effective_batch > 0
                else "?"
            )
        )
        est_total_epochs = (
            round(train_steps_target / est_epoch_steps, 2)
            if isinstance(est_epoch_steps, int) and est_epoch_steps > 0
            else "?"
        )
        cprint(
            f"[data] Source: {BOLD}{dataset_source or 'n/a'}{RESET_BOLD}",
            "green",
        )
        cprint(
            f"[data] samples = {BOLD}{dataset_size}{RESET_BOLD}; steps_per_epoch_est = {BOLD}{steps_per_epoch_est}{RESET_BOLD}",
            "green",
        )
        for source_name, embodiment_name, frames in dataset_breakdown:
            cprint(f"[data]   {source_name}/{embodiment_name}: {BOLD}{frames}{RESET_BOLD} frames", "cyan")
        cprint(
            f"[train-shape] per_device_batch = {BOLD}{per_device_batch}{RESET_BOLD}; effective_batch = {BOLD}{effective_batch}{RESET_BOLD}",
            "yellow",
        )
        cprint(
            f"[train-shape] est_epoch_steps = {BOLD}{est_epoch_steps}{RESET_BOLD}; est_total_epochs ~ {BOLD}{est_total_epochs}{RESET_BOLD}",
            "yellow",
        )

    # --- Model ---
    model = create_model(cfg, device, resume_path=trainer_cfg.get("resume"))
    # Enable verbose memory bank logging in debug mode (anchor changes / resets)
    # Note: Access memory_bank before DDP wrapping, as DDP wraps the model and changes attribute access
    if is_debug:
        target_model = model.module if hasattr(model, "module") else model
        if hasattr(target_model, "memory_bank"):
            target_model.memory_bank.debug = True
    TrainerUtils.freeze_backbones(model, trainer_cfg.get("freeze_modules", ""))
    TrainerUtils.print_trainable_parameters(model)
    if distributed:
        model = DDP(model, device_ids=[device] if device.type == "cuda" else None, find_unused_parameters=True)
        rank0_print("[ddp] Wrapped model with DDP", "green")

    # --- WandB / checkpoints ---
    wandb_run = None if is_debug else setup_wandb_run(cfg, rank)
    if not is_debug and trainer_cfg.get("enable_wandb", False) and wandb_run is None:
        rank0_print("[wandb] Disabled (either not installed or init failed).", "yellow")

    checkpoint_dir = trainer_cfg.get("checkpoint_dir", "")
    checkpoint_manager = None
    if not is_debug:
        checkpoint_manager = CheckpointManager(
            save_dir=PROJECT_ROOT / checkpoint_dir if checkpoint_dir else None,
            save_every_steps=int(trainer_cfg.get("save_every_steps", -1)),
            save_final=trainer_cfg.get("save_final", True),
            rank=rank,
        )

    # --- Optimizer + scheduler ---
    param_groups = build_param_lr_groups(model, cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=trainer_cfg.get("learning_rate", {}).get("base", 1e-4),
        weight_decay=trainer_cfg.get("weight_decay", 0.01),
    )

    warmup_ratio = trainer_cfg.get("warmup_ratio", 0.05)
    warmup_steps = int(train_steps_target * warmup_ratio)
    scheduler = build_scheduler(
        optimizer,
        total_steps=train_steps_target,
        warmup_steps=warmup_steps,
        scheduler_type=trainer_cfg.get("scheduler", "cosine"),
        min_lr_cfg=trainer_cfg.get("min_lr", {}),
    )

    grad_clip_norm = trainer_cfg.get("grad_clip_norm", 0.0)
    log_interval = trainer_cfg.get("log_interval", 10)
    classifier_metrics_window_size = int(trainer_cfg.get("classifier_metrics_window_size", 10))
    eval_action_interval = int(trainer_cfg.get("eval_action_interval", 0))
    is_rank0 = (not dist.is_initialized()) or dist.get_rank() == 0
    metrics_aggregator = ClassifierMetricsAggregator(window_size=classifier_metrics_window_size) if is_rank0 else None
    metrics_sync = ClassifierMetricsSync(
        log_interval=log_interval,
        device=device,
        is_distributed=dist.is_initialized(),
        is_rank0=is_rank0,
        aggregator=metrics_aggregator,
    )

    rank0_print(
        f"[train] total_steps = {BOLD}{train_steps_target}{RESET_BOLD}; "
        f"lr_base = {BOLD}{trainer_cfg.get('learning_rate', {}).get('base', 1e-4)}{RESET_BOLD}",
        "yellow",
    )
    rank0_print(
        f"[train] warmup = {BOLD}{warmup_steps}{RESET_BOLD} / {BOLD}{train_steps_target}{RESET_BOLD}; "
        f"clip = {BOLD}{grad_clip_norm}{RESET_BOLD}; log_interval = {BOLD}{log_interval}{RESET_BOLD}; "
        f"classifier_metrics_window_size = {BOLD}{classifier_metrics_window_size}{RESET_BOLD}",
        "yellow",
    )

    # --- Training loop ---
    global_step = 0
    if checkpoint_manager is not None and checkpoint_manager.enabled:
        # Save an early checkpoint after initialization to verify serialization path.
        checkpoint_manager._dump(model, optimizer, scheduler, global_step=0, tag="init")

    # Pass eval_dataset only if it's RandomEpisodeIterableDataset (has episode mapping)
    eval_dataset_for_eval = None
    if eval_action_interval > 0 and hasattr(memory_dataset, "episode_to_indices"):
        eval_dataset_for_eval = memory_dataset
    
    global_step = train_steps(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        rank=rank,
        total_steps=train_steps_target,
        grad_clip_norm=grad_clip_norm,
        global_step=global_step,
        log_interval=log_interval,
        warmup_steps=warmup_steps,
        wandb_run=wandb_run,
        checkpoint_manager=checkpoint_manager,
        is_debug=is_debug,
        classifier_metrics_sync=metrics_sync,
        eval_action_interval=eval_action_interval,
        eval_dataset=eval_dataset_for_eval,
    )

    # --- Finalization ---
    if checkpoint_manager is not None and checkpoint_manager.enabled and global_step > 0:
        # Proactive early checkpoint to surface any serialization issues.
        checkpoint_manager._dump(model, optimizer, scheduler, global_step=1, tag="sanity_step1")

    if checkpoint_manager is not None:
        checkpoint_manager.save_final(model, optimizer, scheduler, global_step=global_step)

    if wandb_run is not None:
        wandb_run.finish()

    destroy_distributed()
    rank0_print("Training finished.", "green")


if __name__ == "__main__":
    main()
