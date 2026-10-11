"""Base trainer.

Event-driven training loop that fires lifecycle events for callbacks.
Subclasses override _forward_step() for different training algorithms.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from leap.core.callback import Callback
from leap.core.config import instantiate
from leap.core.event import Event, EventBus
from leap.core.state import TrainingState
from leap.distributed.strategy import DistributedStrategy
from leap.distributed.utils import setup_logging

_TRAINER_STATE_FILE = "trainer_state.json"

logger = logging.getLogger(__name__)


def build_optimizer(cfg: Dict[str, Any], model: torch.nn.Module) -> Optimizer:
    """Build a default optimizer from config."""
    opt_type = cfg.get("type", "AdamW")
    opt_params = {k: v for k, v in cfg.items() if k != "type"}

    # Case-insensitive lookup: try exact name first, then search all optimizer names
    opt_cls = getattr(torch.optim, opt_type, None)
    if opt_cls is None:
        opt_type_lower = opt_type.lower()
        for name in dir(torch.optim):
            if name.lower() == opt_type_lower:
                opt_cls = getattr(torch.optim, name)
                break
    if opt_cls is None:
        raise ValueError(f"Unknown optimizer type: {opt_type}")

    if hasattr(model, "get_trainable_parameters"):
        params = model.get_trainable_parameters()
    else:
        params = [p for p in model.parameters() if p.requires_grad]
    return opt_cls(params, **opt_params)


def build_scheduler(cfg: Dict[str, Any], optimizer: Optimizer, num_training_steps: int) -> Any:
    """Build a learning rate scheduler from config."""
    if cfg is None:
        return None

    sched_type = cfg.get("type", "cosine")

    if sched_type == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps,
            eta_min=cfg.get("eta_min", 0.0),
        )
    elif sched_type == "cosine_with_warmup":
        import math
        from torch.optim.lr_scheduler import LambdaLR
        warmup_steps = cfg.get("warmup_steps", 0)
        warmup_ratio = cfg.get("warmup_ratio", 0.0)
        min_lr_ratio = cfg.get("min_lr_ratio", 0.0)
        if warmup_ratio > 0 and warmup_steps == 0:
            warmup_steps = int(num_training_steps * warmup_ratio)

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return current_step / max(1, warmup_steps)
            progress = (current_step - warmup_steps) / max(1, num_training_steps - warmup_steps)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

        return LambdaLR(optimizer, lr_lambda)
    elif sched_type == "linear":
        from transformers import get_linear_schedule_with_warmup
        warmup_steps = cfg.get("warmup_steps", 0)
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )
    elif sched_type == "constant_with_warmup":
        from transformers import get_constant_schedule_with_warmup
        warmup_steps = cfg.get("warmup_steps", 0)
        return get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
        )
    else:
        raise ValueError(f"Unknown scheduler type: {sched_type}")


class BaseTrainer:
    """Event-driven training loop with callback support.

    Subclasses override:
        _forward_step(batch) -> Dict[str, Tensor]: compute losses
        _build_train_dataloader() -> DataLoader: (optional) custom dataloader

    Args:
        cfg: Full experiment config (OmegaConf DictConfig).
    """

    def __init__(self, cfg: Any, optimizer: Optional[Optimizer] = None) -> None:
        self.cfg = cfg
        self.event_bus = EventBus()
        self.state = TrainingState()
        self.callbacks: List[Callback] = []
        self._resume_batch_idx: int = 0

        # Store config in state
        self.state.config = cfg

        # Setup distributed strategy
        training_cfg = cfg.get("training", {})
        grad_clip = training_cfg.get(
            "max_grad_norm",
            training_cfg.get("gradient_clip_max_norm", 1.0),
        )
        self.max_grad_norm = None if grad_clip is None else float(grad_clip)

        dist_cfg = cfg.get("distributed", {})
        self.strategy = DistributedStrategy(
            mixed_precision=dist_cfg.get("mixed_precision", "bf16"),
            deepspeed_config=dist_cfg.get("deepspeed_config", None),
            gradient_accumulation_steps=dist_cfg.get("gradient_accumulation_steps", 1),
            gradient_clipping=self.max_grad_norm,
        )

        # Configure logging: silence INFO on non-rank-0 processes
        setup_logging(self.strategy.process_index)

        # Build model
        self.strategy.print("[LEAP] Loading model...", flush=True)
        self.model = self._build_model()

        # Build optimizer
        opt_cfg = cfg.get("optimizer", {"type": "AdamW", "lr": 2e-5, "weight_decay": 0.01})
        self.optimizer = optimizer if optimizer is not None else self.build_optimizer(opt_cfg)

        # Build dataloader
        self.strategy.print("[LEAP] Loading dataset...", flush=True)
        self.train_dataloader = self._build_train_dataloader()
        self.strategy.wait_for_everyone()
        self.strategy.print("[LEAP] Dataset loaded.", flush=True)

        # Calculate total training steps (partially — num_batches updated after prepare)
        self.state.max_epochs = training_cfg.get("max_epochs", 1)
        self.state.max_steps = training_cfg.get("max_steps", 0)

        # Build scheduler — num_training_steps is the actual per-GPU optimizer steps.
        # We manage scheduler stepping manually (not via Accelerate) to avoid
        # AcceleratedScheduler's implicit num_processes multiplication.
        raw_num_batches = 0
        if self.train_dataloader is not None:
            try:
                raw_num_batches = len(self.train_dataloader)
            except TypeError:
                raw_num_batches = 0  # IterableDataset — no length
        num_processes = self.strategy.num_processes
        if num_processes > 1:
            self.state.num_batches = (raw_num_batches + num_processes - 1) // num_processes
        else:
            self.state.num_batches = raw_num_batches
        grad_accum = self.strategy.gradient_accumulation_steps
        total_steps = self.state.max_iters // grad_accum
        sched_cfg = cfg.get("scheduler", None)
        self.scheduler = build_scheduler(sched_cfg, self.optimizer, total_steps)
        if self.scheduler is not None:
            # Register so accelerator.save_state / load_state persist it even
            # though we didn't wrap it.
            self.strategy.accelerator.register_for_checkpointing(self.scheduler)

        self.strategy.print(
            f"[LEAP] Training started: max_steps={self.state.max_steps}, "
            f"max_epochs={self.state.max_epochs}, "
            f"num_batches={self.state.num_batches} (per GPU), "
            f"total_iters={total_steps}",
            flush=True,
        )

        # DataLoader.batch_size is None when a custom batch_sampler is used.
        # Accelerate cannot infer DeepSpeed's auto micro batch size in that case.
        if (
            self.train_dataloader is not None
            and self.train_dataloader.batch_size is None
            and getattr(self.strategy.accelerator.state, "deepspeed_plugin", None) is not None
        ):
            self.strategy.accelerator.state.deepspeed_plugin.deepspeed_config[
                "train_micro_batch_size_per_gpu"
            ] = int(cfg.get("data", {}).get("batch_size", 1))

        # Prepare with distributed strategy (scheduler excluded — we step it manually)
        self.model, self.optimizer, self.train_dataloader = self.strategy.prepare(
            self.model, self.optimizer, self.train_dataloader
        )

        # Verify num_batches after prepare — should match the pre-computed estimate.
        if self.train_dataloader is not None and hasattr(self.train_dataloader.dataset, "__len__"):
            self.state.num_batches = len(self.train_dataloader)
        else:
            self.state.num_batches = 0

        # Update state references
        self.state.model = self.model
        self.state.optimizer = self.optimizer
        self.state.scheduler = self.scheduler
        self.state.train_dataloader = self.train_dataloader

        # Setup work_dir
        work_dir = cfg.get("work_dir", "work_dirs/default")
        self.state.work_dir = work_dir
        if self.strategy.is_main_process:
            os.makedirs(work_dir, exist_ok=True)

        # Set trainer save function on state so callbacks (e.g. CheckpointCallback) can save
        self.state._trainer_save_fn = self.save_checkpoint

        # Register callbacks
        for cb_cfg in cfg.get("callbacks", []):
            cb = instantiate(cb_cfg)
            self.add_callback(cb)

        # Fire INIT event
        self._fire(Event.INIT)

    def _build_model(self) -> torch.nn.Module:
        """Build the model from config."""
        return instantiate(self.cfg.get("model"))

    def build_optimizer(self, cfg: Dict[str, Any]) -> Optimizer:
        """Build the optimizer from config.

        Subclasses can override this to implement model-specific parameter
        grouping. A prebuilt optimizer can also be passed to ``__init__``.
        """
        return build_optimizer(cfg, self.model)

    def _build_train_dataloader(self) -> Optional[DataLoader]:
        """Build the training dataloader from config.

        Auto-detects whether the dataset is Map-style or IterableDataset and
        configures the DataLoader accordingly.
        """
        data_cfg = self.cfg.get("data", {})
        train_cfg = data_cfg.get("train", None)
        if train_cfg is None:
            return None

        dataset = instantiate(train_cfg)
        batch_size = data_cfg.get("batch_size", 1)
        num_workers = data_cfg.get("num_workers", 4)

        collate_fn = None
        collate_cfg = data_cfg.get("collate_fn", None)
        if collate_cfg is not None:
            collate_fn = instantiate(collate_cfg)

        # Auto-detect collate_fn from dataset (e.g. PrePackedDataset)
        if collate_fn is None and hasattr(dataset, "collate_fn"):
            collate_fn = dataset.collate_fn

        # Map-style Dataset path
        logger.info(
            f"[DataLoader] DataLoader (map mode) | "
            f"dataset={type(dataset).__name__}, {len(dataset)} samples, "
            f"batch_size={batch_size}, num_workers={num_workers}, "
            f"steps/epoch={len(dataset) // batch_size}"
        )

        # Optional custom batch sampler (e.g. SourceAwareSampler).
        # When provided, it owns batch_size / shuffle / drop_last; the
        # DataLoader keyword args for those become unused.
        batch_sampler_cfg = data_cfg.get("batch_sampler", None)
        if batch_sampler_cfg is not None:
            from torch.utils.data import RandomSampler

            base_sampler = RandomSampler(dataset)
            sample_type_fn = getattr(dataset, "get_source", None)
            if sample_type_fn is None:
                raise ValueError(
                    "data.batch_sampler is configured but dataset "
                    f"{type(dataset).__name__} does not expose get_source(idx)."
                )
            batch_sampler = instantiate(
                batch_sampler_cfg,
                base_sampler=base_sampler,
                sample_type_fn=sample_type_fn,
            )
            return DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=torch.cuda.is_available(),
            )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )

    def add_callback(self, callback: Callback) -> None:
        """Add a callback and register its event handlers."""
        self.callbacks.append(callback)
        callback.register(self.event_bus)

    def _fire(self, event: Event) -> None:
        """Fire an event, passing the current state."""
        self.event_bus.fire(event, self.state)

    def _to_device(self, batch: Any) -> Any:
        """Move batch tensors to the accelerator device."""
        device = self.strategy.accelerator.device
        if isinstance(batch, torch.Tensor):
            return batch.to(device)
        if isinstance(batch, dict):
            return {k: self._to_device(v) for k, v in batch.items()}
        if isinstance(batch, (list, tuple)):
            return type(batch)(self._to_device(v) for v in batch)
        return batch

    def _forward_step(self, batch: Any) -> Dict[str, torch.Tensor]:
        """Compute forward pass and return loss dict.

        Must be overridden by subclasses.

        Args:
            batch: A batch of data from the dataloader.

        Returns:
            Dict with at least 'loss' key, plus optional component losses.
        """
        raise NotImplementedError("Subclasses must implement _forward_step()")

    def fit(self) -> None:
        """Main training loop with event-driven callbacks."""
        self._fire(Event.FIT_START)
        self.model.train()

        start_epoch = self.state.epoch

        # When max_steps is set, allow enough epochs to reach it.
        # Use a large upper bound so the loop is terminated by should_stop.
        # For IterableDatasets (num_batches=0), use a very large epoch bound
        # and rely on max_steps + should_stop to terminate.
        if self.state.max_steps > 0 and self.state.num_batches > 0:
            epochs_needed = (self.state.max_steps // self.state.num_batches) + 1
            end_epoch = max(start_epoch + epochs_needed, self.state.max_epochs)
        elif self.state.max_steps > 0 and self.state.num_batches == 0:
            # IterableDataset: epoch count is meaningless; loop until max_steps
            end_epoch = start_epoch + 1_000_000
        else:
            end_epoch = self.state.max_epochs

        for epoch in range(start_epoch, end_epoch):
            self.state.epoch = epoch
            self._fire(Event.EPOCH_START)

            self._train_epoch()

            self._fire(Event.EPOCH_END)

            if self.state.should_stop:
                break

        self._fire(Event.FIT_END)
        self.strategy.wait_for_everyone()

    def _train_epoch(self) -> None:
        """Run one training epoch."""
        # Forward set_epoch into the dataset and sampler so MixedDataset /
        # DistributedSampler reseeds at each epoch boundary. Datasets/samplers
        # without set_epoch (plain dataset reader, default RandomSampler)
        # are skipped via hasattr — safe for single-dataset / single-GPU yamls.
        epoch = self.state.epoch
        ds = getattr(self.train_dataloader, "dataset", None)
        if ds is not None and hasattr(ds, "set_epoch"):
            ds.set_epoch(epoch)
        sampler = getattr(self.train_dataloader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        batch_sampler = getattr(self.train_dataloader, "batch_sampler", None)
        batch_sampler = getattr(batch_sampler, "batch_sampler", batch_sampler)
        if batch_sampler is not None and hasattr(batch_sampler, "set_epoch"):
            batch_sampler.set_epoch(epoch)

        # When resuming mid-epoch, skip batches that were already processed.
        # _resume_batch_idx is set by load_checkpoint() and consumed once.
        skip_batches = getattr(self, "_resume_batch_idx", 0)
        if skip_batches > 0:
            logger.info(f"Skipping first {skip_batches} batches (already completed).")
            self._resume_batch_idx = 0  # consume — only skip once

        for batch_idx, batch in enumerate(self.train_dataloader):
            if batch_idx < skip_batches:
                continue

            # Move batch tensors to device (needed when DataLoader isn't Accelerate-wrapped)
            batch = self._to_device(batch)

            self.state.batch_idx = batch_idx
            self.state.batch = batch

            self._fire(Event.BATCH_START)

            # Forward
            self._fire(Event.BEFORE_FORWARD)
            with self.strategy.accelerator.accumulate(self.model):
                loss_dict = self._forward_step(batch)
                self.state.loss_dict = loss_dict

                self._fire(Event.AFTER_FORWARD)

                # Backward
                loss = loss_dict["loss"]
                self._fire(Event.BEFORE_BACKWARD)
                self.strategy.backward(loss)
                self._fire(Event.AFTER_BACKWARD)

                # Optimizer step
                self._fire(Event.BEFORE_OPTIMIZER_STEP)
                grad_norm = self.strategy.optimizer_step(
                    self.optimizer, max_grad_norm=self.max_grad_norm
                )
                self.state.grad_norm = grad_norm

                self.optimizer.zero_grad()
                self._fire(Event.AFTER_OPTIMIZER_STEP)

            # Step scheduler only on actual optimizer steps (after gradient sync)
            if self.scheduler is not None and self.strategy.accelerator.sync_gradients:
                self.scheduler.step()

            # Update LR in state
            self.state.lr = self.optimizer.param_groups[0]["lr"]
            self.state.global_step += 1

            self._fire(Event.BATCH_END)

            if self.state.should_stop:
                break

    def evaluate(self) -> Dict[str, float]:
        """Run evaluation loop."""
        self._fire(Event.EVAL_START)

        eval_dataloader = self.state.eval_dataloader
        if eval_dataloader is None:
            logger.warning("No eval dataloader configured, skipping evaluation.")
            self._fire(Event.EVAL_END)
            return {}

        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(eval_dataloader):
                self.state.batch_idx = batch_idx
                self.state.batch = batch
                self._fire(Event.EVAL_BATCH)

        self.model.train()
        self._fire(Event.EVAL_END)

        return self.state.extras.get("eval_metrics", {})

    def save_checkpoint(self, path: str) -> None:
        """Save a checkpoint (model/optimizer state + training progress)."""
        self._fire(Event.BEFORE_SAVE)
        self.strategy.save_state(path)
        # Save training progress so we can resume from the exact step
        if self.strategy.is_main_process:
            state_dict = {
                "epoch": self.state.epoch,
                "global_step": self.state.global_step,
                "batch_idx": self.state.batch_idx,
            }
            with open(os.path.join(path, _TRAINER_STATE_FILE), "w") as f:
                json.dump(state_dict, f, indent=2)
        self._fire(Event.AFTER_SAVE)

    def load_checkpoint(self, path: str) -> None:
        """Load a checkpoint for resuming (restores model/optimizer + training progress)."""
        self._fire(Event.BEFORE_LOAD)
        self.strategy.load_state(path)
        # Restore training progress
        state_path = os.path.join(path, _TRAINER_STATE_FILE)
        if os.path.isfile(state_path):
            with open(state_path, "r") as f:
                state_dict = json.load(f)
            self.state.epoch = state_dict.get("epoch", 0)
            self.state.global_step = state_dict.get("global_step", 0)
            self._resume_batch_idx = state_dict.get("batch_idx", 0)
            logger.info(
                f"Resumed training progress: epoch={self.state.epoch}, "
                f"global_step={self.state.global_step}"
            )
        else:
            logger.warning(
                f"No {_TRAINER_STATE_FILE} found in {path}. "
                "Model weights restored but training progress starts from 0."
            )
        self._fire(Event.AFTER_LOAD)
