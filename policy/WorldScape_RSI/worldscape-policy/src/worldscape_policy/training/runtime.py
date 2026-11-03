"""Process and backend primitives for replicated native training."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel


@dataclass(frozen=True)
class NativeDistributedConfig:
    """Training distribution; independent from WAM image parallelism."""

    backend: str = "auto"
    process_group_backend: str | None = None
    process_group_timeout_seconds: int = 3600
    seed: int = 62
    deepspeed_config: dict[str, Any] | None = None
    device: str = "auto"


@dataclass(frozen=True)
class GradientHealth:
    """Global finite-gradient status and the first offending tensor."""

    finite: bool
    global_norm: float
    threshold_exceeded: bool = False
    first_nonfinite: str | None = None


class TrainingForwardAdapter(nn.Module):
    """Make the policy training path visible to DDP/DeepSpeed forward hooks."""

    def __init__(self, policy: nn.Module) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, **kwargs: Any) -> Any:
        return self.policy.training_forward(**kwargs)


class NativeTrainingRuntime:
    """Rank-aware single, DDP, or optional DeepSpeed ZeRO-2/3 runtime."""

    def __init__(
        self,
        config: NativeDistributedConfig | None = None,
    ) -> None:
        self.config = config or NativeDistributedConfig()
        requested = self.config.backend.lower()
        env_world = int(os.environ.get("WORLD_SIZE", "1"))
        self.backend = "ddp" if requested == "auto" and env_world > 1 else (
            "single" if requested == "auto" else requested
        )
        if self.backend not in {"single", "ddp", "deepspeed"}:
            raise ValueError("training backend must be auto, single, ddp, or deepspeed")
        self.owns_process_group = False
        needs_group = self.backend in {"ddp", "deepspeed"} and env_world > 1
        if needs_group and not dist.is_initialized():
            pg_backend = self.config.process_group_backend or (
                "nccl" if torch.cuda.is_available() else "gloo"
            )
            timeout_seconds = int(self.config.process_group_timeout_seconds)
            if timeout_seconds < 1:
                raise ValueError("process_group_timeout_seconds must be positive")
            try:
                dist.init_process_group(
                    backend=pg_backend,
                    init_method="env://",
                    timeout=timedelta(seconds=timeout_seconds),
                )
            except Exception as exc:
                raise RuntimeError(
                    "failed to initialize training process group from torchrun "
                    "RANK/WORLD_SIZE/LOCAL_RANK environment"
                ) from exc
            self.owns_process_group = True
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.local_rank = int(os.environ.get("LOCAL_RANK", str(self.rank)))
        use_cuda = self.config.device == "cuda" or (
            self.config.device == "auto" and torch.cuda.is_available()
        )
        if self.config.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("distributed training device must be auto, cpu, or cuda")
        if self.config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA training device selected but CUDA is unavailable")
        if use_cuda:
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
        else:
            self.device = torch.device("cpu")
        self.seed = int(self.config.seed) + self.rank
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        self.engine: Any | None = None
        self.forward_module: nn.Module | None = None
        self.policy: nn.Module | None = None

    @property
    def is_rank_zero(self) -> bool:
        return self.rank == 0

    def broadcast_from_rank_zero(self, tensor: torch.Tensor) -> torch.Tensor:
        """Broadcast a routing tensor so every rank executes identical branches."""

        if self.world_size > 1:
            if not dist.is_initialized():
                raise RuntimeError(
                    "distributed routing broadcast requires an initialized process group"
                )
            dist.broadcast(tensor, src=0)
        return tensor

    def setup(
        self, policy: nn.Module, optimizer: torch.optim.Optimizer
    ) -> tuple[nn.Module, torch.optim.Optimizer]:
        self.policy = policy
        policy.to(self.device)
        adapter = TrainingForwardAdapter(policy)
        if self.backend == "single":
            self.forward_module = adapter
        elif self.backend == "ddp":
            if self.world_size < 2:
                raise RuntimeError("DDP training requires torchrun WORLD_SIZE greater than one")
            kwargs: dict[str, Any] = {}
            if self.device.type == "cuda":
                kwargs = {"device_ids": [self.local_rank], "output_device": self.local_rank}
            self.forward_module = DistributedDataParallel(
                adapter, find_unused_parameters=True, **kwargs
            )
        else:
            try:
                import deepspeed
            except ImportError as exc:
                raise ImportError(
                    "DeepSpeed backend selected but deepspeed is not installed; "
                    "install worldscape-policy[train] or `pip install deepspeed`"
                ) from exc
            ds_config = dict(self.config.deepspeed_config or {})
            zero = dict(ds_config.get("zero_optimization", {}))
            stage = int(zero.get("stage", 2))
            if stage not in {2, 3}:
                raise ValueError(
                    "native DeepSpeed backend supports ZeRO stage 2 or 3"
                )
            zero["stage"] = stage
            ds_config["zero_optimization"] = zero
            ds_config.setdefault("train_micro_batch_size_per_gpu", 1)
            engine, optimizer, _, _ = deepspeed.initialize(
                model=adapter,
                optimizer=optimizer,
                config=ds_config,
                model_parameters=[p for p in policy.parameters() if p.requires_grad],
            )
            self.engine = engine
            self.forward_module = engine
        return self.forward_module, optimizer

    def forward(self, **kwargs: Any) -> Any:
        if self.forward_module is None:
            raise RuntimeError("training runtime has not been set up")
        return self.forward_module(**kwargs)

    def backward(self, loss: torch.Tensor) -> None:
        if self.engine is not None:
            self.engine.backward(loss)
        else:
            loss.backward()

    @property
    def manages_gradient_clipping(self) -> bool:
        """Whether the backend owns gradient partitioning and clipping."""
        return self.engine is not None

    def last_gradient_norm(self) -> float:
        if self.engine is None:
            return 0.0
        value = getattr(self.engine.optimizer, "_global_grad_norm", 0.0)
        if isinstance(value, torch.Tensor):
            return float(value.detach().float().cpu())
        return float(value)

    def inspect_gradients(
        self, *, max_norm_threshold: float | None = None
    ) -> GradientHealth:
        """Check gradients globally before an optimizer is allowed to update."""

        local_detail: str | None = None
        global_norm = 0.0
        if self.engine is not None:
            zero_optimizer = self.engine.optimizer
            check_overflow = getattr(zero_optimizer, "check_overflow", None)
            if callable(check_overflow):
                check_overflow(
                    partition_gradients=bool(
                        getattr(zero_optimizer, "partition_gradients", False)
                    )
                )
                finite = not bool(getattr(zero_optimizer, "overflow", False))
            else:
                finite, local_detail = self._inspect_parameter_gradients()
            if not finite and local_detail is None:
                local_detail = self._first_nonfinite_zero_gradient()
            if finite:
                scaled_norm = zero_optimizer.scaled_global_norm()
                if isinstance(scaled_norm, torch.Tensor):
                    scaled_norm = float(scaled_norm.detach().float().cpu())
                loss_scale = float(getattr(zero_optimizer, "loss_scale", 1.0))
                global_norm = float(scaled_norm) / loss_scale
        else:
            finite, local_detail = self._inspect_parameter_gradients()
            if finite:
                global_norm = self._global_parameter_gradient_norm()
            if self.world_size > 1:
                failed = torch.tensor(
                    int(not finite), dtype=torch.int32, device=self.device
                )
                dist.all_reduce(failed, op=dist.ReduceOp.MAX)
                finite = not bool(failed.item())
                norm = torch.tensor(
                    global_norm, dtype=torch.float64, device=self.device
                )
                dist.all_reduce(norm, op=dist.ReduceOp.MAX)
                global_norm = float(norm.item())

        threshold_exceeded = bool(
            finite
            and max_norm_threshold is not None
            and global_norm > max_norm_threshold
        )
        if finite:
            return GradientHealth(
                finite=True,
                global_norm=global_norm,
                threshold_exceeded=threshold_exceeded,
            )
        details: list[str | None] = [None] * self.world_size
        if self.world_size > 1:
            dist.all_gather_object(details, local_detail)
        else:
            details[0] = local_detail
        first = next((detail for detail in details if detail is not None), None)
        return GradientHealth(
            finite=False,
            global_norm=float("inf"),
            first_nonfinite=first or "non-finite gradient (parameter unknown)",
        )

    def skip_gradient_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Clear rejected gradients without updating model or optimizer state."""

        if self.engine is None:
            optimizer.zero_grad(set_to_none=True)
            return
        zero_optimizer = self.engine.optimizer
        if not hasattr(zero_optimizer, "overflow"):
            raise RuntimeError(
                "DeepSpeed optimizer cannot safely skip a non-finite gradient step"
            )
        previous_check = getattr(zero_optimizer, "check_grad_overflow", None)
        zero_optimizer.overflow = True
        if previous_check is not None:
            zero_optimizer.check_grad_overflow = False
        try:
            # DeepSpeed's overflow path clears partitioned gradients and advances
            # its micro-step bookkeeping without applying an optimizer update.
            self.engine.step()
        finally:
            if previous_check is not None:
                zero_optimizer.check_grad_overflow = previous_check

    def _inspect_parameter_gradients(self) -> tuple[bool, str | None]:
        if self.policy is None:
            raise RuntimeError("training runtime has not been set up")
        for name, parameter in self.policy.named_parameters():
            gradient = parameter.grad
            if gradient is not None and not bool(torch.isfinite(gradient).all()):
                return False, self._gradient_detail(name, gradient)
        return True, None

    def _global_parameter_gradient_norm(self) -> float:
        if self.policy is None:
            raise RuntimeError("training runtime has not been set up")
        squared_norm = torch.zeros((), dtype=torch.float64, device=self.device)
        for parameter in self.policy.parameters():
            if parameter.grad is not None:
                gradient = parameter.grad.detach().to(dtype=torch.float64)
                squared_norm.add_(gradient.square().sum())
        return float(squared_norm.sqrt().cpu())

    def _first_nonfinite_zero_gradient(self) -> str | None:
        if self.engine is None:
            return None
        zero_optimizer = self.engine.optimizer
        gradients_by_group = getattr(zero_optimizer, "averaged_gradients", {})
        parameters_by_group = getattr(zero_optimizer, "params_in_partition", ())
        parameter_names = (
            {
                id(parameter): name
                for name, parameter in self.policy.named_parameters()
            }
            if self.policy is not None
            else {}
        )
        items = (
            gradients_by_group.items()
            if isinstance(gradients_by_group, dict)
            else enumerate(gradients_by_group)
        )
        for group_index, gradients in items:
            if gradients is None:
                continue
            gradient_list = (
                [gradients] if isinstance(gradients, torch.Tensor) else gradients
            )
            parameters = (
                parameters_by_group[group_index]
                if group_index < len(parameters_by_group)
                else ()
            )
            for index, gradient in enumerate(gradient_list):
                if gradient is None or bool(torch.isfinite(gradient).all()):
                    continue
                name = (
                    parameter_names.get(id(parameters[index]))
                    if index < len(parameters)
                    else None
                )
                return self._gradient_detail(
                    name or f"zero_group_{group_index}[{index}]", gradient
                )
        return None

    def _gradient_detail(self, name: str, gradient: torch.Tensor) -> str:
        return (
            f"rank={self.rank} parameter={name} "
            f"shape={tuple(gradient.shape)} dtype={gradient.dtype}"
        )

    def save_deepspeed_checkpoint(self, save_dir: str, *, tag: str) -> None:
        if self.engine is None:
            raise RuntimeError("DeepSpeed checkpoint save requires an initialized engine")
        saved = self.engine.save_checkpoint(save_dir, tag=tag)
        if saved is False:
            raise RuntimeError("DeepSpeed engine checkpoint save failed")

    def load_deepspeed_checkpoint(self, load_dir: str, *, tag: str) -> None:
        if self.engine is None:
            raise RuntimeError("DeepSpeed checkpoint load requires an initialized engine")
        loaded = self.engine.load_checkpoint(
            load_dir,
            tag=tag,
            load_module_strict=True,
            load_optimizer_states=True,
            load_lr_scheduler_states=False,
        )
        if loaded is None or (isinstance(loaded, tuple) and loaded[0] is None):
            raise RuntimeError("DeepSpeed engine checkpoint load failed")

    def save_deepspeed_16bit_model(
        self,
        save_dir: str,
        *,
        save_filename: str,
    ) -> None:
        if self.engine is None:
            raise RuntimeError("DeepSpeed model export requires an initialized engine")
        exported = self.engine.save_16bit_model(
            save_dir,
            save_filename=save_filename,
        )
        if exported is False:
            raise RuntimeError(
                "DeepSpeed could not gather the portable 16-bit policy model; "
                "enable its official 16-bit model-save gathering option"
            )

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        if self.engine is not None:
            self.engine.step()
        else:
            optimizer.step()

    def all_reduce_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        if self.world_size == 1:
            return metrics
        names = sorted(metrics)
        values = torch.tensor(
            [metrics[name] for name in names], dtype=torch.float64, device=self.device
        )
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values /= self.world_size
        return {name: float(value) for name, value in zip(names, values.cpu().tolist())}

    def gather_rank_state(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        if self.world_size == 1:
            return [state]
        states: list[dict[str, Any] | None] = [None] * self.world_size
        dist.all_gather_object(states, state)
        return [value for value in states if value is not None]

    def state_dict(self) -> dict[str, Any]:
        if self.engine is None:
            return {}
        scaler = getattr(self.engine, "loss_scaler", None)
        return {
            "optimizer": self.engine.optimizer.state_dict(),
            "global_steps": int(getattr(self.engine, "global_steps", 0)),
            "skipped_steps": int(getattr(self.engine, "skipped_steps", 0)),
            "loss_scaler": (
                scaler.state_dict() if hasattr(scaler, "state_dict") else None
            ),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if self.engine is None:
            if state:
                raise ValueError("single/DDP runtime state must be empty")
            return
        expected = {"optimizer", "global_steps", "skipped_steps", "loss_scaler"}
        if set(state) != expected:
            raise ValueError("DeepSpeed runtime checkpoint state is incomplete")
        self.engine.optimizer.load_state_dict(state["optimizer"])
        self.engine.global_steps = int(state["global_steps"])
        self.engine.skipped_steps = int(state["skipped_steps"])
        scaler = getattr(self.engine, "loss_scaler", None)
        if state["loss_scaler"] is not None:
            if not hasattr(scaler, "load_state_dict"):
                raise ValueError("checkpoint has DeepSpeed scaler state but engine does not")
            scaler.load_state_dict(state["loss_scaler"])

    def barrier(self) -> None:
        if self.world_size > 1:
            dist.barrier()

    def close(self) -> None:
        if self.owns_process_group and dist.is_initialized():
            dist.destroy_process_group()
            self.owns_process_group = False


__all__ = [
    "NativeDistributedConfig",
    "NativeTrainingRuntime",
    "TrainingForwardAdapter",
]
