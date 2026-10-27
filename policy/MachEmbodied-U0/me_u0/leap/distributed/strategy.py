"""Distributed training strategy wrapping accelerate + DeepSpeed.

Replaces MMEngine's DeepSpeedStrategy with a clean accelerate-based interface.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class DistributedStrategy:
    """Unified distributed training strategy using HuggingFace Accelerate.

    Wraps Accelerate + optional DeepSpeed for:
    - Mixed precision training
    - Distributed data parallel
    - DeepSpeed ZeRO stages 1/2/3
    - Gradient accumulation

    Args:
        mixed_precision: One of "no", "fp16", "bf16".
        deepspeed_config: Path to DeepSpeed JSON config, or None.
        gradient_accumulation_steps: Number of gradient accumulation steps.
        gradient_clipping: DeepSpeed internal gradient clipping value. For
            non-DeepSpeed runs, clipping is applied explicitly in optimizer_step.
        log_with: Logging integrations (e.g., "tensorboard", "wandb").
        project_dir: Directory for accelerate logs.
        kwargs: Additional kwargs passed to Accelerator.
    """

    def __init__(
        self,
        mixed_precision: str = "bf16",
        deepspeed_config: Optional[str] = None,
        gradient_accumulation_steps: int = 1,
        gradient_clipping: Optional[float] = None,
        log_with: Optional[str] = None,
        project_dir: Optional[str] = None,
        init_timeout_minutes: int = 60,
        **kwargs: Any,
    ) -> None:
        from datetime import timedelta

        from accelerate import Accelerator
        from accelerate.utils import DeepSpeedPlugin, InitProcessGroupKwargs

        ds_plugin = None
        if deepspeed_config:
            # Only enable DeepSpeed in distributed mode (multi-GPU).
            # Single-GPU doesn't need it and DeepSpeed requires dist init.
            import torch.distributed as dist
            if dist.is_initialized() or int(os.environ.get("WORLD_SIZE", "1")) > 1:
                ds_kwargs: Dict[str, Any] = {
                    "hf_ds_config": deepspeed_config,
                }
                if gradient_clipping is not None:
                    ds_kwargs["gradient_clipping"] = float(gradient_clipping)
                ds_plugin = DeepSpeedPlugin(
                    **ds_kwargs,
                )
            else:
                logger.info(
                    "Single-GPU detected, ignoring deepspeed_config. "
                    "DeepSpeed will be used automatically in multi-GPU mode."
                )

        # Extend NCCL watchdog timeout for multi-node training: large VLMs can
        # take >10min (the default) to load across all ranks, and one slow rank
        # otherwise triggers ncclRemoteError on the rest.
        init_pg_kwargs = InitProcessGroupKwargs(
            timeout=timedelta(minutes=init_timeout_minutes)
        )

        self.accelerator = Accelerator(
            mixed_precision=mixed_precision,
            deepspeed_plugin=ds_plugin,
            gradient_accumulation_steps=gradient_accumulation_steps,
            log_with=log_with,
            project_dir=project_dir,
            kwargs_handlers=[init_pg_kwargs],
            **kwargs,
        )
        self._prepared_model: Optional[nn.Module] = None

    def prepare(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        *dataloaders: DataLoader,
        scheduler: Optional[Any] = None,
    ) -> Tuple:
        """Prepare model, optimizer, and dataloaders for distributed training.

        Returns:
            Tuple of (model, optimizer, *dataloaders) or
            (model, optimizer, *dataloaders, scheduler) if scheduler provided.
        """
        items = [model, optimizer, *dataloaders]
        if scheduler is not None:
            items.append(scheduler)
        prepared = self.accelerator.prepare(*items)

        # Store reference to prepared model for clip_grad_norm
        self._prepared_model = prepared[0]

        if scheduler is not None:
            # Last item is the scheduler
            return prepared
        return prepared

    def backward(self, loss: torch.Tensor) -> None:
        """Run backward pass through accelerator."""
        self.accelerator.backward(loss)

    def optimizer_step(self, optimizer: Optimizer, max_grad_norm: Optional[float] = None) -> Optional[float]:
        """Execute optimizer step with optional gradient clipping.

        Args:
            optimizer: The optimizer.
            max_grad_norm: Max gradient norm for clipping. None = no clipping.

        Returns:
            The gradient norm if clipping was applied, else None.
        """
        grad_norm = None
        if max_grad_norm is not None and max_grad_norm > 0:
            grad_norm = self.clip_grad_norm(max_grad_norm)
        optimizer.step()
        return grad_norm

    def clip_grad_norm(self, max_norm: float) -> float:
        """Clip gradient norm across all parameters.

        Uses the model stored during prepare() instead of private accelerator APIs.
        """
        if self.accelerator.sync_gradients:
            model = self._prepared_model
            if model is not None:
                params = self.accelerator.unwrap_model(model).parameters()
                return self.accelerator.clip_grad_norm_(params, max_norm)
        return 0.0

    def unwrap_model(self, model: nn.Module) -> nn.Module:
        """Unwrap DDP/DeepSpeed wrapper to get the raw model."""
        return self.accelerator.unwrap_model(model)

    def save_state(self, output_dir: str) -> None:
        """Save accelerate state (model, optimizer, scheduler, rng)."""
        self.accelerator.save_state(output_dir)

    def load_state(self, input_dir: str) -> None:
        """Load accelerate state for resuming.

        Uses ``weights_only=False`` for scheduler / RNG state files that may
        contain non-tensor objects (e.g. OmegaConf configs).  For DeepSpeed,
        model + optimizer checkpoints are loaded by DeepSpeed's own engine
        and are unaffected by this setting.
        """
        self.accelerator.load_state(
            input_dir,
            load_kwargs={"weights_only": False},
        )

    def wait_for_everyone(self) -> None:
        """Synchronize all processes."""
        self.accelerator.wait_for_everyone()

    def gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """Gather tensor across all processes."""
        return self.accelerator.gather(tensor)

    @property
    def device(self) -> torch.device:
        return self.accelerator.device

    @property
    def is_main_process(self) -> bool:
        return self.accelerator.is_main_process

    @property
    def is_local_main_process(self) -> bool:
        return self.accelerator.is_local_main_process

    @property
    def num_processes(self) -> int:
        return self.accelerator.num_processes

    @property
    def process_index(self) -> int:
        return self.accelerator.process_index

    @property
    def gradient_accumulation_steps(self) -> int:
        return self.accelerator.gradient_accumulation_steps

    def print(self, *args: Any, **kwargs: Any) -> None:
        """Print only on the main process."""
        self.accelerator.print(*args, **kwargs)

    def log(self, values: Dict[str, float], step: Optional[int] = None) -> None:
        """Log values to configured backends."""
        self.accelerator.log(values, step=step)
