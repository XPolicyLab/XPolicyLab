from __future__ import annotations

import cProfile
import gc
import logging
import math
import os
import random
import shutil
import time
from collections import deque, defaultdict
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from pstats import SortKey
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from packaging import version
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset
from torchmetrics import Metric
from wandb.sdk.data_types.base_types.wb_value import WBValue

from .aliases import PathOrStr
from .checkpoint import Checkpointer, FullCheckpointer, build_sharded_checkpointer,save_state_dict, load_state_dict, load_fsdp_optim_state
from .config import (
    BlockType,
    CheckpointType,
    SchedulerUnits,
    ShardedCheckpointerType,
    SpeedMonitorConfig,
    TrainConfig, BatchDivisor,
)
from .data.iterable_dataset_mixture import IterableDatasetMixture
from .eval.inf_evaluator import InfDatasetEvaluator
from .exceptions import OLMoConfigurationError
from .model import Molmo
from .optim import Optimizer, Scheduler
from .torch_util import (
    barrier,
    gc_cuda,
    get_fs_local_rank,
    get_local_rank,
    get_global_rank,
    get_world_size,
    move_to_device,
    peak_gpu_memory,
    synchronize_flag,
    synchronize_value, get_local_world_size, )
from .util import upload

try:
    from megablocks.layers.moe import (
        batched_load_balancing_loss,
        clear_load_balancing_loss,
        get_load_balancing_loss,
    )
except ImportError:
    pass

__all__ = ["SpeedMonitor", "LRMonitor", "Trainer"]

log = logging.getLogger(__name__)


@dataclass
class BatchStatsMonitor:
    max_window_size: int = 20
    sync_nodes: bool = True
    _batch_stats: Deque[Dict[str, float]] = field(default_factory=lambda: deque([]))

    def log_batch(self, batch):
        input_ids = batch["input_ids"]
        non_masked = (input_ids >= 0).to(dtype=torch.float32)
        stats = {
            "batch/non_masked_tokens": non_masked.sum(-1).mean(),
            "batch/per_non_masked_tokens": non_masked.mean(),
            "batch/examples_truncated": non_masked[:, -1].mean()
        }
        if "loss_masks" in batch:
            mask = (batch["loss_masks"] > 0).to(dtype=torch.float32)
            stats["batch/loss_tokens"] = mask.sum(-1).mean()
            stats["batch/per_loss_tokens"] = mask.mean()

        self._batch_stats.append(stats)
        if len(self._batch_stats) > self.max_window_size:
            self._batch_stats.popleft()

    def reset(self) -> None:
        self._batch_stats.clear()

    def check(self, device):
        stats = defaultdict(list)
        for batch in self._batch_stats:
            for k, v in batch.items():
                stats[k].append(v)

        out = {}
        for k, v in stats.items():
            v = torch.stack(v).mean()
            if self.sync_nodes:
                v = v.to(device)
                dist.all_reduce(v)
                v.div_(get_world_size())
            out[k] = v.item()
        return out


@dataclass
class SpeedMonitor:
    cfg: SpeedMonitorConfig
    global_total_tokens: int = 0
    stats: Deque[Tuple[float, int, int]] = field(default_factory=lambda: deque([]))

    def batch_start(self, global_total_tokens: int, device_batch_num_tokens: int, device_batch_num_loss_tokens: int, record: bool = True) -> None:
        self.global_total_tokens = global_total_tokens
        if record:
            if len(self.stats) >= self.cfg.window_size:
                self.stats.popleft()
            self.stats.append((
                time.monotonic(),
                device_batch_num_tokens,
                device_batch_num_loss_tokens
            ))

    def reset(self) -> None:
        self.stats.clear()

    def check(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {"throughput/total_tokens": self.global_total_tokens}
        if self.stats:
            interval_seconds = time.monotonic() - self.stats[0][0]
            interval_batches = len(self.stats)
            interval_tokens = sum(x[1] for x in self.stats)
            interval_loss_tokens = sum(x[2] for x in self.stats)
            metrics["throughput/device/loss_tokens_per_second"] = interval_loss_tokens / interval_seconds
            metrics["throughput/device/tokens_per_second"] = interval_tokens / interval_seconds
            metrics["throughput/device/batches_per_second"] = interval_batches / interval_seconds
        return metrics


@dataclass
class LRMonitor:
    optim: torch.optim.Optimizer

    def check(self) -> Dict[str, float]:
        lrs = [group["lr"] for group in self.optim.param_groups]
        return {f"optim/learning_rate_group{idx}": lr for idx, lr in enumerate(lrs)}


def cross_entropy_loss(
    logits, labels, ignore_index: int = -100, reduction: str = "mean", compute_z_loss: bool = False, z_loss_scale: float = 1e-4,
):
    loss = F.cross_entropy(logits, labels, ignore_index=ignore_index, reduction=reduction)

    if not compute_z_loss:
        return loss, None
    z_squared = logits.logsumexp(-1).pow(2)
    if reduction == "mean":
        z_squared = (z_squared * (labels != ignore_index)).mean()
    elif reduction == "sum":
        z_squared = (z_squared * (labels != ignore_index)).sum()

    z_loss = z_loss_scale * z_squared

    return loss, z_loss


@dataclass
class DatasetMetrics:
    label: str
    eval_loader: DataLoader
    eval_metric: Union[Metric, Dict[str, Metric], List[Metric]]
    subset_num_batches: Optional[int] = None

    def reset_metrics(self) -> None:
        if isinstance(self.eval_metric, Metric):
            self.eval_metric.reset()
        else:
            for metric in self.eval_metric.values():
                metric.reset()

    def compute_metrics(self) -> Dict[str, float]:
        return {f"{self.label}/{k}": v.compute().item() for k, v in self.eval_metric.items()}

    def update_metrics(
        self,
        batch: Dict[str, Any],
        eval_out: Dict[str, torch.Tensor],
    ) -> None:
        total_weight = eval_out["total_weight"]
        self.eval_metric["Loss"].update(eval_out["total_loss"]/total_weight, total_weight)
        self.eval_metric["Accuracy"].update(eval_out["total_accuracy"]/total_weight, total_weight)
        self.eval_metric["ZLoss"].update(eval_out["total_zloss"]/total_weight, total_weight)


@dataclass
class Trainer:
    cfg: TrainConfig
    model: Molmo
    fsdp_model: FSDP
    optim: Optimizer
    scheduler: Scheduler
    train_loader: DataLoader
    device: torch.device
    evaluators: List[DatasetMetrics]
    inference_evaluators: List[InfDatasetEvaluator]
    epoch: Optional[int] = None
    global_step: int = 0

    global_train_examples_seen_this_epoch: int = 0
    """Tracks the global number of training examples seen in the current epoch for the purpose of restoring
    the data loader position on restarts."""

    global_train_tokens_seen: int = 0
    """Tracks the global total number of tokens trained on."""

    checkpoints: List[Path] = field(default_factory=list)
    unsharded_checkpoints: List[Path] = field(default_factory=list)
    ephemeral_checkpoints: List[Path] = field(default_factory=list)
    min_train_loss: float = float("inf")
    cur_train_loss: float = float("inf")
    _start_time: float = 0.0
    _gc_init_state: bool = True
    _inference_warmup: bool = True
    loss_fn: Callable[..., torch.Tensor] = field(default_factory=lambda: cross_entropy_loss)  # type: ignore
    last_sharded_checkpoint_step: Optional[int] = None
    last_unsharded_checkpoint_step: Optional[int] = None
    skip_next_model_load: bool = False
    _node_src: int = None
    _node_group: Any = None
    _node_group_ranks: Any = None

    def __post_init__(self):
        if self.cfg.fused_loss:
            import flash_attn
            from flash_attn.ops.triton.cross_entropy import (  # type: ignore
                cross_entropy_loss,
            )

            # The `ignored_index` parameter of `cross_entropy_loss` was changed to `ignore_index` in v2.5.8 with commit https://github.com/Dao-AILab/flash-attention/commit/ec6d22143b5d375e253b2ebfc563b26a43f43684
            ce_loss_use_ignore_index_param = version.parse(flash_attn.__version__) >= version.parse("2.5.8")

            def fused_loss_fn(
                logits, labels, ignore_index: int = -100, reduction: str = "mean", compute_z_loss: bool = False
            ):
                if ce_loss_use_ignore_index_param:
                    ignore_index_kwarg = {"ignore_index": ignore_index}
                else:
                    ignore_index_kwarg = {"ignored_index": ignore_index}

                loss, z_loss = cross_entropy_loss(
                    logits,
                    labels,
                    label_smoothing=0.0,
                    logit_scale=1.0,
                    lse_square_scale=self.cfg.softmax_auxiliary_loss_scale if self.cfg.softmax_auxiliary_loss else 0.0,
                    inplace_backward=False,
                    process_group=None,
                    **ignore_index_kwarg,
                )
                mask = labels != ignore_index

                if reduction == "mean":
                    loss = loss.sum() / mask.sum()
                elif reduction == "sum":
                    loss = loss.sum()
                else:
                    loss = loss

                if not compute_z_loss:
                    return loss, None

                if reduction == "mean":
                    z_loss = z_loss.sum() / mask.sum()
                elif reduction == "sum":
                    z_loss = z_loss.sum()
                else:
                    z_loss = z_loss

                return loss, z_loss

            self.loss_fn = fused_loss_fn


        if self.model.config.block_type == BlockType.moe:
            from .config import config_to_moe_args

            self.moe_args = config_to_moe_args(self.cfg.model)            

    @property
    def dataset(self) -> IterableDataset:
        return self.train_loader

    @property
    def tokens_per_batch(self) -> int:
        return self.cfg.global_train_batch_size * self.cfg.model.max_sequence_length

    @property
    def batches_per_epoch(self) -> int:
        return self.dataset.total_size // self.cfg.global_train_batch_size

    @property
    def max_epochs(self) -> int:
        if isinstance(self.cfg.max_duration, str) and self.cfg.max_duration.endswith("ep"):
            return int(self.cfg.max_duration[:-2].strip())
        else:
            return 1

    @property
    def max_steps(self) -> int:
        if isinstance(self.cfg.max_duration, int):
            return self.cfg.max_duration
        elif isinstance(self.cfg.max_duration, str):
            if self.cfg.max_duration.endswith("T"):
                # convert to float *first* to handle scientific notation
                max_tokens = int(float(self.cfg.max_duration[:-1].strip()))
                tokens_remaining = max(max_tokens - self.global_train_tokens_seen, 0)
                steps_remaining = tokens_remaining // self.tokens_per_batch
                return self.global_step + steps_remaining
            elif self.cfg.max_duration.endswith("ep"):
                max_epochs = int(self.cfg.max_duration[:-2].strip())
                return max_epochs * self.batches_per_epoch
            else:
                # convert to float *first* to handle scientific notation
                return int(float(self.cfg.max_duration))
        else:
            raise TypeError(f"expected int or str for 'max_duration', found {type(self.cfg.max_duration)}")

    @property
    def max_tokens(self) -> int:
        if isinstance(self.cfg.max_duration, int):
            return (
                self.global_train_tokens_seen
                + max(self.cfg.max_duration - self.global_step, 0) * self.tokens_per_batch
            )
        elif isinstance(self.cfg.max_duration, str):
            if self.cfg.max_duration.endswith("T"):
                # convert to float *first* to handle scientific notation
                return int(float(self.cfg.max_duration[:-1].strip()))
            elif self.cfg.max_duration.endswith("ep"):
                max_epochs = int(self.cfg.max_duration[:-2].strip())
                return max_epochs * self.batches_per_epoch * self.tokens_per_batch
            else:
                # convert to float *first* to handle scientific notation
                return (
                    self.global_train_tokens_seen
                    + max(int(float(self.cfg.max_duration)) - self.global_step, 0) * self.tokens_per_batch
                )
        else:
            raise TypeError(f"expected int or str for 'max_duration', found {type(self.cfg.max_duration)}")

    @property
    def scheduler_current(self) -> int:
        if self.cfg.scheduler.units == SchedulerUnits.steps:
            return self.global_step
        elif self.cfg.scheduler.units == SchedulerUnits.tokens:
            return self.global_train_tokens_seen
        else:
            raise NotImplementedError(self.cfg.scheduler.units)

    @property
    def scheduler_max(self) -> int:
        if self.cfg.scheduler.units == SchedulerUnits.steps:
            return self.max_steps
        elif self.cfg.scheduler.units == SchedulerUnits.tokens:
            return self.max_tokens
        else:
            raise NotImplementedError(self.cfg.scheduler.units)

    def trainer_state_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "global_train_examples_seen_this_epoch": self.global_train_examples_seen_this_epoch,
            "global_train_tokens_seen": self.global_train_tokens_seen,
            "world_size": get_world_size(),
            "checkpoints": self.checkpoints,
            "unsharded_checkpoints": self.unsharded_checkpoints,
            "ephemeral_checkpoints": self.ephemeral_checkpoints,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.random.get_rng_state(),
                "cuda": torch.cuda.get_rng_state(),
            },
        }

    def load_trainer_state_dict(self, state_dict: Dict[str, Any]) -> None:
        # Checkpoint paths.
        self.checkpoints = [
            path
            for path in state_dict["checkpoints"]
            if path.is_dir() and path.resolve().parent == Path(self.cfg.save_folder).resolve()
        ]
        self.unsharded_checkpoints = [
            path
            for path in state_dict["unsharded_checkpoints"]
            if path.is_dir() and path.resolve().parent == Path(self.cfg.save_folder).resolve()
        ]
        self.ephemeral_checkpoints = [
            path
            for path in state_dict.get("ephemeral_checkpoints", [])
            if path.is_dir() and path.resolve().parent == Path(self.cfg.save_folder).resolve()
        ]

        # Dataset / dataloader position.
        checkpoint_epoch = state_dict.get("epoch", 0)
        self.global_step = state_dict["global_step"]
        self.global_train_examples_seen_this_epoch = state_dict.get(
            "global_train_examples_seen_this_epoch",
            state_dict.get(  # for backwards compatibility
                "global_train_examples_seen",
                state_dict.get("global_data_step", self.global_step) * self.cfg.global_train_batch_size,
            ),
        )
        self.global_train_tokens_seen = state_dict.get(
            "global_train_tokens_seen",
            state_dict.get("global_data_step", self.global_step)  # for backwards compatibility
            * self.cfg.global_train_batch_size
            * self.cfg.model.max_sequence_length,
        )

        if not self.cfg.restore_dataloader:
            self.epoch = 0
            self.global_train_tokens_seen = 0
            self.global_train_examples_seen_this_epoch = 0
        elif self.epoch is None:
            self.epoch = checkpoint_epoch
        elif checkpoint_epoch != self.epoch:
            log.info(f"Starting new epoch (epoch = {self.epoch})")
            self.global_train_examples_seen_this_epoch = 0

        if self.cfg.fast_forward_batches:
            log.info(f"Fast-forwarding data loader by {self.cfg.fast_forward_batches:,d} steps")
            # Technically we don't "see" these batches that we fast-forward through, but we use
            # this variable to update the position of the dataset so we need to include them here.
            self.global_train_examples_seen_this_epoch += (
                self.cfg.fast_forward_batches * self.cfg.global_train_batch_size
            )
            # NOTE: on the other hand we don't add anything to 'self.global_train_tokens_seen' here because
            # that variable is meant to track the actual number of tokens trained on.

        if self.global_train_examples_seen_this_epoch > 0:
            # assert isinstance(self.dataset.dataset, IterableDatasetMixture)
            log.info(f"Data loader will start at instance index {self.global_train_examples_seen_this_epoch:,d}")
            self.dataset.dataset.start_index = self.global_train_examples_seen_this_epoch

        # Optionally keep LR/WD from checkpoint; otherwise reset per current config/scheduler.
        if getattr(self.cfg, "keep_lr_on_load", False):
            log.info("Keeping learning rate and weight decay from checkpoint (keep_lr_on_load=True)")
        else:
            # Reset learning rate and weight decay to the values from the config, not the checkpoint.
            log.info("Resetting learning rate...")
            if self.cfg.model.vision_backbone is not None:
                initial_lr_dict = {
                    "connector": self.cfg.optimizer.connector_learning_rate,
                    "vit": self.cfg.optimizer.vit_learning_rate,
                    "llm": self.cfg.optimizer.llm_learning_rate,
                }
                weight_decay_dict = {
                    "connector": self.cfg.optimizer.connector_weight_decay,
                    "vit": self.cfg.optimizer.vit_weight_decay,
                    "llm": self.cfg.optimizer.llm_weight_decay,
                }
                for group in self.optim.param_groups:
                    group_name = group["group_name"]
                    component_name = group_name.split("_")[0]
                    new_learning_rate = self.scheduler.get_lr(
                        initial_lr_dict[component_name],
                        self.scheduler_current,
                        self.scheduler_max,
                        group_name,
                    )
                    group["lr"] = new_learning_rate
                    if "weight_decay" in group and group["weight_decay"] > 0.0:
                        group["weight_decay"] = weight_decay_dict[component_name]
            else:
                new_learning_rate = self.scheduler.get_lr(
                    self.cfg.optimizer.learning_rate, self.scheduler_current, self.scheduler_max
                )
                for group in self.optim.param_groups:
                    group["lr"] = new_learning_rate
                    group["initial_lr"] = self.cfg.optimizer.learning_rate
                    if "weight_decay" in group and group["weight_decay"] > 0.0:
                        group["weight_decay"] = self.cfg.optimizer.weight_decay

        # RNG states.
        if "rng" in state_dict and state_dict.get("world_size", get_world_size()) == get_world_size():
            log.info("Restoring RNG states...")
            rng_state = state_dict["rng"]
            self.restore_rng_state(rng_state)
        else:
            log.warning(
                "Trainer will not restore RNG states since the RNG states in the checkpoint are missing or invalid. "
                "This typically happens when restoring from an unsharded checkpoint or a checkpoint that was saved "
                "with a different world size. If that's the case you can safely ignore this warning."
            )

    def restore_rng_state(self, rng_state: Dict[str, Any]) -> None:
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch"])
        torch.cuda.set_rng_state(rng_state["cuda"])

    def _save_checkpoint(
        self, checkpointer: Checkpointer, checkpoint_type: CheckpointType
    ) -> Tuple[PathOrStr, Optional[PathOrStr]]:
        if checkpoint_type == CheckpointType.sharded:
            suffix = ""
            current_checkpoints = self.checkpoints
            link_latest = get_fs_local_rank() == 0
            num_checkpoints_to_keep = self.cfg.save_num_checkpoints_to_keep
        elif checkpoint_type == CheckpointType.unsharded:
            suffix = "-unsharded"
            current_checkpoints = self.unsharded_checkpoints
            link_latest = get_global_rank() == 0
            num_checkpoints_to_keep = self.cfg.save_num_unsharded_checkpoints_to_keep
        elif checkpoint_type == CheckpointType.sharded_ephemeral:
            suffix = ""
            current_checkpoints = self.ephemeral_checkpoints
            link_latest = get_fs_local_rank() == 0
            num_checkpoints_to_keep = 1
        else:
            raise NotImplementedError(checkpoint_type)

        # Zero-gradients to avoid gathering them.
        # self.optim.zero_grad(set_to_none=True)

        checkpoint_dir = Path(self.cfg.save_folder) / f"step{self.global_step}{suffix}"
        remote_checkpoint_dir: Optional[str] = None
        if self.cfg.remote_save_folder is not None:
            remote_checkpoint_dir = f"{self.cfg.remote_save_folder.rstrip('/')}/{checkpoint_dir.name}"

        checkpointer.save_checkpoint(
            checkpoint_dir,
            self.fsdp_model,
            self.optim,
            self.trainer_state_dict(),
            upload_to=remote_checkpoint_dir,
        )

        # # Free cached GPU memory after checkpointing.
        # torch.cuda.empty_cache()
        
        if link_latest:
            # Link to 'latest'.
            latest_path = Path(self.cfg.save_folder) / f"latest{suffix}"
            latest_path.unlink(missing_ok=True)
            try:
                latest_path.symlink_to(checkpoint_dir.name, target_is_directory=True)
            except FileExistsError:
                # Same as above, caught when another (file-system) local rank 0 has already made the 'latest' symlink.
                # This can happen when nodes are saving to a common NFS drive but otherwise have distinct
                # file-systems.
                if latest_path.resolve().name != checkpoint_dir.name:
                    raise

        # Save multimodal dataset checkpoint
        if self.cfg.save_dataloader_state:
            data_ckpt_fname = checkpoint_dir / f"rank{get_global_rank()}_data.bin"
            self.dataset.save(data_ckpt_fname)

        # Remove old checkpoints.
        current_checkpoints.append(checkpoint_dir)
        if num_checkpoints_to_keep > 0:
            while len(current_checkpoints) > num_checkpoints_to_keep:
                self.remove_checkpoint(0, checkpoint_type)

        barrier()

        if remote_checkpoint_dir is not None:
            return remote_checkpoint_dir, checkpoint_dir
        else:
            return checkpoint_dir, None

    def save_sharded_checkpoint(self) -> Tuple[PathOrStr, Optional[PathOrStr]]:
        checkpointer = build_sharded_checkpointer(self.cfg)
        result = self._save_checkpoint(checkpointer, CheckpointType.sharded)
        self.last_sharded_checkpoint_step = self.global_step
        return result

    def save_ephemeral_checkpoint(self) -> Tuple[PathOrStr, Optional[PathOrStr]]:
        checkpointer = build_sharded_checkpointer(self.cfg)
        result = self._save_checkpoint(checkpointer, CheckpointType.sharded_ephemeral)
        self.last_sharded_checkpoint_step = self.global_step
        return result

    def _remove_sharded_checkpoint(self, idx: int, checkpoints: List[Path]):
        oldest_checkpoint = checkpoints.pop(idx)
        barrier()
        if get_fs_local_rank() == 0 and oldest_checkpoint.is_dir():
            shutil.rmtree(oldest_checkpoint, ignore_errors=True)
            latest_path = Path(self.cfg.save_folder) / "latest"
            if latest_path.resolve() == oldest_checkpoint.resolve():
                latest_path.unlink()
        barrier()

    def remove_sharded_checkpoint(self, idx: int = 0):
        self._remove_sharded_checkpoint(idx, self.checkpoints)

    def remove_ephemeral_checkpoint(self, idx: int = 0):
        self._remove_sharded_checkpoint(idx, self.ephemeral_checkpoints)

    def restore_sharded_checkpoint(
        self,
        load_path: PathOrStr,
        local_cache: Optional[PathOrStr] = None,
        *,
        load_optimizer_state: bool = True,
        load_trainer_state: bool = True,
        sharded_checkpointer: Optional[ShardedCheckpointerType] = None,
    ):
        # Zero-gradients to avoid gathering them.
        self.optim.zero_grad(set_to_none=True)
        checkpointer = build_sharded_checkpointer(self.cfg, name=sharded_checkpointer)
        trainer_state = checkpointer.restore_checkpoint(
            load_path,
            self.fsdp_model,
            self.optim,
            local_cache=local_cache,
            load_optimizer_state=load_optimizer_state,
        )
        if load_trainer_state:
            self.load_trainer_state_dict(trainer_state)
        barrier()

    def save_unsharded_checkpoint(self) -> Tuple[PathOrStr, Optional[PathOrStr]]:
        checkpointer = FullCheckpointer(self.cfg)
        result = self._save_checkpoint(checkpointer, CheckpointType.unsharded)
        self.last_unsharded_checkpoint_step = self.global_step
        return result

    def remove_unsharded_checkpoint(self, idx: int = 0):
        barrier()
        oldest_checkpoint = self.unsharded_checkpoints.pop(idx)
        if get_global_rank() == 0 and oldest_checkpoint.is_dir():
            shutil.rmtree(oldest_checkpoint, ignore_errors=True)
            latest_path = Path(self.cfg.save_folder) / "latest-unsharded"
            if latest_path.resolve() == oldest_checkpoint.resolve():
                latest_path.unlink()
        barrier()

    def restore_unsharded_checkpoint(
        self,
        load_path: PathOrStr,
        local_cache: Optional[PathOrStr] = None,
        *,
        load_optimizer_state: bool = True,
        load_trainer_state: bool = True,
    ):
        # Zero-gradients to avoid gathering them.
        self.optim.zero_grad(set_to_none=True)
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        from torch.distributed.fsdp.api import FullOptimStateDictConfig
        # Ensure variable exists for later reference across branches
        model_state = None

        # 若上一阶段已在 FSDP 之前加载过模型，则本阶段跳过模型权重加载
        if not getattr(self, "skip_next_model_load", False):
            # Load full model state on rank 0 only to reduce memory, then let FSDP handle distribution
            with FSDP.state_dict_type(
                self.fsdp_model,
                state_dict_type=StateDictType.FULL_STATE_DICT,
                state_dict_config=FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
                optim_state_dict_config=FullOptimStateDictConfig(rank0_only=True, offload_to_cpu=True),
            ):
                filtered_state = None
                target_keys = None
                if get_global_rank() == 0:
                    # 原始 ckpt
                    model_state = load_state_dict(load_path, "model.pt", local_cache=local_cache, map_location="cpu")  # type: ignore
                    # 仅加载与当前模型键重叠的部分，避免命名空间不一致导致 strict 报错
                    target_full = self.fsdp_model.state_dict()
                    target_keys = set(target_full.keys())
                    del target_full
                    filtered_state = {k: v for k, v in model_state.items() if k in target_keys}
                    missing_cnt = len([k for k in target_keys if k not in model_state])
                    unexpected_cnt = len([k for k in model_state.keys() if k not in target_keys])
                    log.info(
                        f"Filtered load: keep={len(filtered_state)} missing={missing_cnt} unexpected={unexpected_cnt}"
                    )
                # 加载重叠部分，放宽 strict
                self.fsdp_model.load_state_dict(filtered_state if filtered_state is not None else {}, strict=False)
        else:
            # 重置一次性跳过标志
            self.skip_next_model_load = False

        # Optimizer state: skip by default to reduce memory risk; load only if explicitly requested
        if load_optimizer_state:
            try:
                optim_state_dict_to_load = load_state_dict(
                    load_path, "optim.pt", local_cache=local_cache, map_location="cpu"
                )
                model_state_for_mapping = model_state if model_state is not None else load_state_dict(
                    load_path, "model.pt", local_cache=local_cache, map_location="cpu"
                )
                (
                    _,
                    og_keys_to_new,
                ) = self.fsdp_model._fsdp_wrapped_module._make_state_dict_compatible(model_state_for_mapping)  # type: ignore[attr-defined]
                optim_state_dict_to_load = FullCheckpointer(self.cfg)._make_optim_state_dict_compatible(  # type: ignore
                    optim_state_dict_to_load,
                    og_keys_to_new,
                )

                # 分本机 local_rank 轮流加载优化器状态，降低并发内存峰值
                gc.collect(); torch.cuda.empty_cache(); barrier()
                for turn in range(get_local_world_size()):
                    if turn == get_local_rank():
                        load_fsdp_optim_state(self.fsdp_model, self.optim, optim_state_dict_to_load)
                        gc.collect(); torch.cuda.empty_cache()
                    barrier()
                del optim_state_dict_to_load
            except Exception as e:
                logging.warning(f"Loading optimizer state failed, continuing without it: {e}")

        if load_trainer_state:
            try:
                trainer_state = load_state_dict(load_path, "train.pt", local_cache=local_cache)
            except FileNotFoundError:
                trainer_state = load_state_dict(load_path, "other.pt", local_cache=local_cache)
            self.load_trainer_state_dict(trainer_state)
        barrier()

    def save_checkpoint(
        self, checkpoint_type: CheckpointType = CheckpointType.sharded
    ) -> Tuple[PathOrStr, Optional[PathOrStr]]:
        result: Tuple[PathOrStr, Optional[PathOrStr]]
        if checkpoint_type == CheckpointType.sharded:
            result = self.save_sharded_checkpoint()
        elif checkpoint_type == CheckpointType.unsharded:
            result = self.save_unsharded_checkpoint()
        elif checkpoint_type == CheckpointType.sharded_ephemeral:
            result = self.save_ephemeral_checkpoint()
        elif checkpoint_type == CheckpointType.action_head:
            result = self.save_action_head_checkpoint() # default unsharded
        else:
            raise NotImplementedError(checkpoint_type)

        gc_cuda()
        return result

    def restore_checkpoint(
        self,
        load_path: PathOrStr,
        *,
        checkpoint_type: Optional[CheckpointType] = None,
        local_cache: Optional[PathOrStr] = None,
        load_optimizer_state: bool = True,
        load_trainer_state: bool = True,
        load_dataloader_state: bool = True,
        sharded_checkpointer: Optional[ShardedCheckpointerType] = None,
    ):
        if checkpoint_type == CheckpointType.unsharded or (
            checkpoint_type is None and str(load_path).rstrip("/").endswith("-unsharded")
        ):
            self.restore_unsharded_checkpoint(
                load_path,
                local_cache=local_cache,
                load_optimizer_state=load_optimizer_state,
                load_trainer_state=load_trainer_state,
            )
        elif checkpoint_type == CheckpointType.sharded or checkpoint_type is None:
            self.restore_sharded_checkpoint(
                load_path,
                local_cache=local_cache,
                load_optimizer_state=load_optimizer_state,
                load_trainer_state=load_trainer_state,
                sharded_checkpointer=sharded_checkpointer,
            )
        elif checkpoint_type is not None:
            raise NotImplementedError(checkpoint_type)

        if load_dataloader_state:
            # Restore multimodal dataset checkpoint
            logging.info("Loading dataloader state...")
            data_ckpt_fname = os.path.join(load_path, f"rank{get_global_rank()}_data.bin")
            self.dataset.restore(data_ckpt_fname)
            logging.info("Done")

        gc_cuda()

    def remove_checkpoint(self, idx: int = 0, checkpoint_type: CheckpointType = CheckpointType.sharded):
        if checkpoint_type == CheckpointType.sharded:
            self.remove_sharded_checkpoint(idx=idx)
        elif checkpoint_type == CheckpointType.unsharded:
            self.remove_unsharded_checkpoint(idx=idx)
        elif checkpoint_type == CheckpointType.sharded_ephemeral:
            self.remove_ephemeral_checkpoint(idx=idx)
        else:
            raise NotImplementedError(checkpoint_type)

    def move_to_device(self, batch, device):
        return move_to_device(batch, device)

    def get_labels(self, batch: Dict[str, Any]) -> torch.Tensor:
        # Labels are just input IDs shifted to the left (first item is ignored).
        labels, label_mask, attention_mask, instance_mask = (
            batch["input_ids"].clone(),
            batch.get("label_mask"),
            batch.get("attention_mask"),
            batch.get("instance_mask"),
        )
        # assert False, f"labels: {labels}, label_mask: {label_mask}, attention_mask: {attention_mask}, instance_mask: {instance_mask}"
        if label_mask is not None:
            labels.masked_fill_(~label_mask, -100)
        if attention_mask is not None:
            labels.masked_fill_(attention_mask == 0.0, -100)
        if instance_mask is not None:
            labels.masked_fill_(~instance_mask.unsqueeze(-1), value=-100)
        # assert False, f"labels: {labels[labels!=-100].max()}, {labels[labels!=-100].min()}"
        return labels[..., 1:].contiguous()

    def model_forward(
        self, batch: Dict[str, Any], loss_reduction: str = "mean", compute_z_loss: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        # shape: (batch_size, seq_len, vocab_size)
        with torch.autocast("cuda", enabled=True, dtype=self.cfg.autocast_precision):
            logits = self.fsdp_model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                attention_bias=batch.get("attention_bias"),
                response_mask=(batch["loss_masks"] > 0) if "loss_masks" in batch else None,
                images=batch.get("images"),
                image_masks=batch.get("image_masks"),
                image_input_idx=batch.get("image_input_idx"),
                subsegment_ids=batch.get("subsegment_ids"),
                position_ids=batch.get("position_ids"),
            ).logits
        if "labels" in batch:
            assert "loss_masks" in batch
            assert loss_reduction == "none"
            loss_masks = batch["loss_masks"] * (batch["loss_masks"] > 0)
            labels = batch["labels"].long()
            labels.masked_fill_(~(loss_masks > 0), -100)
            labels = labels.view(-1)
            logits_for_loss = logits.to(torch.float32).view(-1, logits.size(-1)) # for numerical stability
        else:
            logits_for_loss = logits[..., :-1, :].contiguous()
            # shape: (batch_size * seq_len, vocab_size)
            logits_for_loss = logits_for_loss.view(-1, logits_for_loss.size(-1))
            # shape: (batch_size, seq_len)
            labels = self.get_labels(batch)
            # shape: (batch_size * seq_len,)
            labels = labels.view(-1)
        ce_loss, z_loss = self.loss_fn(
            logits_for_loss, labels, ignore_index=-100, reduction=loss_reduction,
            compute_z_loss=compute_z_loss, z_loss_scale=self.cfg.softmax_auxiliary_loss_scale,
        )
        bs = batch["input_ids"].shape[0]
        if loss_reduction == "none":
            # Reshape (batch_size * seq_len,) -> (batch_size, seq_len)
            ce_loss = ce_loss.view(bs, -1)
            if z_loss is not None:
                z_loss = z_loss.view(bs, -1)

        accuracy = torch.argmax(logits_for_loss, dim=-1) == labels
        if "labels" in batch:
            ce_loss = ce_loss * loss_masks
            if z_loss is not None:
                z_loss = z_loss * loss_masks
            accuracy = accuracy.view(bs, -1)
            accuracy = accuracy * loss_masks
        else:
            accuracy = (accuracy * (labels >= 0))
            accuracy = accuracy.view(bs, -1)
        return accuracy, ce_loss, z_loss, logits

    def train_batch(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        # Split into micro-batches.
        micro_batches = self.split_batch(batch)
        has_labels = "labels" in batch

        if has_labels:
            loss_masks = batch["loss_masks"] * (batch["loss_masks"] > 0)
            if self.cfg.batch_divisor == BatchDivisor.global_batch:
                batch_size_in_tokens = loss_masks.sum()
                dist.all_reduce(batch_size_in_tokens)
                batch_size_in_tokens.div_(get_world_size())
            elif self.cfg.batch_divisor == BatchDivisor.device_batch:
                batch_size_in_tokens = loss_masks.sum()
            else:
                raise ValueError()
        else:
            batch_size_in_tokens = batch["input_ids"].numel()

        del batch  # in case this helps reduce memory

        ce_batch_loss = torch.tensor(0.0, device=self.device)
        batch_accuracy = torch.tensor(0.0, device=self.device)
        z_batch_loss = None if not self.cfg.softmax_auxiliary_loss else torch.tensor(0.0, device=self.device)
        lb_batch_loss = (
            None if self.model.config.block_type != BlockType.moe else torch.tensor(0.0, device=self.device)
        )
        moe_z_batch_loss = (
            None if not self.model.config.moe_zloss_weight else torch.tensor(0.0, device=self.device)
        )
        expert_assignments = (
            None
            if (
                (self.model.config.block_type != BlockType.moe)
                or (self.model.config.moe_log_expert_assignment is False)
            )
            else torch.zeros((self.model.config.n_layers, self.model.config.moe_num_experts))
        )
        for micro_batch in micro_batches:
            accuracy, ce_loss, z_loss, logits = self.model_forward(
                micro_batch, compute_z_loss=self.cfg.softmax_auxiliary_loss, loss_reduction="none" if has_labels else "sum"
            )
            if has_labels:
                accuracy = accuracy.sum()
                ce_loss = ce_loss.sum()
                if z_loss is not None:
                    z_loss = z_loss.sum()

            ce_loss = ce_loss / batch_size_in_tokens
            accuracy = accuracy / batch_size_in_tokens

            # In case this helps with memory utilization.
            del micro_batch

            # Update overall CE batch loss.
            ce_batch_loss += ce_loss.detach()
            batch_accuracy += accuracy.detach()

            # Get loss to optimize for.
            if self.cfg.softmax_auxiliary_loss:
                assert z_loss is not None
                assert z_batch_loss is not None
                z_loss = z_loss / batch_size_in_tokens

                loss = ce_loss + z_loss

                # Update overall Z batch loss.
                z_batch_loss += z_loss.detach()
            else:
                loss = ce_loss

            del logits

            if self.model.config.block_type == BlockType.moe:
                if self.model.config.moe_zloss_weight:
                    lb_loss, moe_z_loss = batched_load_balancing_loss(self.moe_args)
                    lb_loss = lb_loss / len(micro_batches)
                    moe_z_loss = moe_z_loss / len(micro_batches)
                elif self.model.config.moe_loss_weight:
                    lb_loss = batched_load_balancing_loss(self.moe_args) / len(micro_batches)
                if self.model.config.moe_log_expert_assignment:
                    if self.model.config.moe_zloss_weight:
                        tokens_per_expert, _, _ = zip(*get_load_balancing_loss())
                    else:
                        tokens_per_expert, _ = zip(*get_load_balancing_loss())
                    expert_assignments += torch.stack(tokens_per_expert, dim=0).cpu()
                clear_load_balancing_loss()
                if self.model.config.moe_loss_weight:
                    loss += lb_loss
                    lb_batch_loss += lb_loss.detach()
                if self.model.config.moe_zloss_weight:
                    loss += moe_z_loss
                    moe_z_batch_loss += moe_z_loss.detach()

            # Run backward pass.
        
            loss.backward()

        return ce_batch_loss, z_batch_loss, batch_accuracy, lb_batch_loss, moe_z_batch_loss, expert_assignments

    def train_step(self, batch: Dict[str, Any], reduce_global_loss: bool = True) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        # Record how many instances are going to be skipped (masked out).
        if (instance_mask := batch.get("instance_mask")) is not None:
            metrics["train/masked_instances_local_rank"] = (~instance_mask).sum().item()

        # Zero-gradients.
        self.optim.zero_grad(set_to_none=True)

        # Move tensors to the right device.
        batch = self.move_to_device(batch, self.device)

        # Run forward-backward pass.
        ce_batch_loss, z_batch_loss, batch_accuracy, lb_batch_loss, moe_z_batch_loss, expert_assignments = self.train_batch(batch)

        # Collect loss, potentially reducing over all ranks.
        if reduce_global_loss:
            dist.reduce(ce_batch_loss, 0)
            ce_batch_loss.div_(get_world_size())
            if z_batch_loss is not None:
                dist.reduce(z_batch_loss, 0)
                z_batch_loss.div_(get_world_size())
            if batch_accuracy is not None:
                dist.reduce(batch_accuracy, 0)
                batch_accuracy.div_(get_world_size())
            if lb_batch_loss is not None:
                dist.reduce(lb_batch_loss, 0)
                lb_batch_loss.div_(get_world_size())
            if moe_z_batch_loss is not None:
                dist.reduce(moe_z_batch_loss, 0)
                moe_z_batch_loss.div_(get_world_size())

        # Clip gradient norms and collect param/gradient/optim metrics.
        should_log_optim_metrics_this_step = self.should_log_optim_metrics_this_step()
        optim_metrics = self.optim.clip_grads_and_collect_metrics(
            self.global_step,
            collect_param_metrics=should_log_optim_metrics_this_step,
            # passing this process group here ensures metrics are reduced correctly when we're using
            # HYBRID sharding.
            process_group=self.fsdp_model.process_group,
            multi_modal=self.cfg.model.vision_backbone is not None,
        )
        # Adjust the learning rate.
        if self.cfg.model.vision_backbone is not None:
            initial_lr_dict = {
                "connector": self.cfg.optimizer.connector_learning_rate,
                "vit": self.cfg.optimizer.vit_learning_rate,
                "llm": self.cfg.optimizer.llm_learning_rate,
            }
            for group in self.optim.param_groups:
                group_name = group["group_name"]
                component_name = group_name.split("_")[0]
                group["lr"] = self.scheduler.get_lr(
                    initial_lr_dict[component_name],
                    self.scheduler_current,
                    self.scheduler_max,
                    group_name,
                )
                group["max_grad_norm"] = self.scheduler.get_max_grad_norm(
                    self.cfg.max_grad_norm, self.scheduler_current, self.scheduler_max
                )
                group["max_grad_norm_ratio"] = self.scheduler.get_max_grad_norm(
                    self.cfg.max_grad_norm_ratio, self.scheduler_current, self.scheduler_max
                )
        else:
            for group in self.optim.param_groups:
                # TODO (epwalsh): if we want to enable different LRs or gradient clipping settings per group
                # we should pass `group["initial_lr"]` or `group["initial_max_grad_norm"]` here instead of
                # the corresponding values from `self.cfg`.
                group["lr"] = self.scheduler.get_lr(
                    self.cfg.optimizer.learning_rate, self.scheduler_current, self.scheduler_max
                )
                group["max_grad_norm"] = self.scheduler.get_max_grad_norm(
                    self.cfg.max_grad_norm, self.scheduler_current, self.scheduler_max
                )
                group["max_grad_norm_ratio"] = self.scheduler.get_max_grad_norm(
                    self.cfg.max_grad_norm_ratio, self.scheduler_current, self.scheduler_max
                )

        # Optimizer step.
        self.optim.step()

        # Collect metrics and check for NaN loss.
        # NOTE: this involves a bunch of host-device syncs so we wait until the last moment to do this.
        if torch.isnan(ce_batch_loss):
            raise ValueError("nan loss encountered")
        if z_batch_loss is not None and torch.isnan(z_batch_loss):
            raise ValueError("nan loss encountered")
        for key, value in optim_metrics.items():
            metrics[f"optim/{key}"] = value.item()
        self.cur_train_loss = ce_batch_loss.item()
        self.min_train_loss = min(self.min_train_loss, self.cur_train_loss)
        metrics["train/CrossEntropyLoss"] = self.cur_train_loss
        metrics["train/Perplexity"] = math.exp(self.cur_train_loss)
        metrics["train/Accuracy"] = batch_accuracy.item()
        if z_batch_loss is not None:
            metrics["train/ZLoss"] = z_batch_loss.item()
        if lb_batch_loss is not None:
            metrics["train/LoadBalancingLoss"] = lb_batch_loss.item()
            # Log assignment metrics.
            if expert_assignments is not None:
                for layer_idx, expert_assignments_layer in enumerate(expert_assignments):
                    total_tokens = expert_assignments_layer.sum().item()
                    for expert_idx, expert_assignment in enumerate(expert_assignments_layer):
                        metrics[f"train/TokensPercentage/layer{layer_idx}/expert{expert_idx}"] = (
                            expert_assignment.item() / total_tokens
                        ) * 100
                        metrics[
                            f"train/TokensTotal/layer{layer_idx}/expert{expert_idx}"
                        ] = expert_assignment.item()
        if moe_z_batch_loss is not None:
            metrics["train/MoEZLoss"] = moe_z_batch_loss.item()

        # Maybe collect post-step optimizer-specific metrics.
        if should_log_optim_metrics_this_step:
            optim_metrics = self.optim.get_post_step_metrics(
                self.fsdp_model, process_group=self.fsdp_model.process_group
            )
            for key, value in optim_metrics.items():
                metrics[f"optim/{key}"] = value.item()

        return metrics

    def eval_batch(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast("cuda", enabled=True, dtype=self.cfg.autocast_precision):
            acc, ce_loss, z_loss, logits = self.model_forward(batch, loss_reduction="none", compute_z_loss=True)
        if "labels" in batch:
            loss_masks = batch["loss_masks"] * (batch["loss_masks"] > 0)
            batch_size_in_tokens = loss_masks.sum(-1)

            return dict(
                total_weight=batch_size_in_tokens.sum(),
                total_loss=ce_loss.sum(),
                total_accuracy=acc.sum(),
                total_zloss=z_loss.sum(),
                batch_loss=ce_loss.sum()/batch_size_in_tokens.sum(),
                batch_accuracy=acc.sum()/batch_size_in_tokens.sum(),
                batch_zloss=z_loss.sum()/batch_size_in_tokens.sum(),
                logits=logits
            )
        else:
            return dict(
                instance_loss=ce_loss.mean(-1),
                instance_aaccuracy=acc.mean(-1),
                batch_loss=ce_loss.mean(),
                batch_accuracy=acc.mean(),
                z_loss=z_loss.mean(),
                logits=logits
            )

    def eval_step(self, batch: Dict[str, Any], evaluator: DatasetMetrics) -> None:
        # Move tensors to the right device.
        batch = self.move_to_device(batch, self.device)

        # Run forward pass.
        with torch.no_grad():  # NOTE: 'torch.inference_mode()' doesn't work with 'torch.compile()'.
            eval_out = self.eval_batch(batch)

        # Update metrics.
        evaluator.update_metrics(
            batch, eval_out
        )  # batch includes all keys that the downstream evaluation needs

        barrier()

    def split_batch(self, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        microbatch_size = self.cfg.device_train_microbatch_size
        batch_size = batch["input_ids"].shape[0]
        if batch_size <= microbatch_size:
            return [batch]
        else:
            micro_batches = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    micro_batches[key] = value.split(microbatch_size, dim=0)
                elif isinstance(value, list):
                    micro_batches[key] = [
                        value[microbatch_size * i : microbatch_size * i + microbatch_size]
                        for i in range(math.ceil(batch_size / microbatch_size))
                    ]
                else:
                    raise ValueError(f"unexpected item in batch: '{key}={value}'")
            return [
                {key: value[i] for key, value in micro_batches.items()}  # type: ignore
                for i in range(len(micro_batches["input_ids"]))
            ]

    def system_metrics(self) -> Dict[str, float]:
        metrics = {}
        if self.global_step < 3 or self.global_step % 10 == 0:
            peak_gpu_mb = peak_gpu_memory()
            if peak_gpu_mb is not None:
                metrics["System/Peak GPU Memory (MB)"] = peak_gpu_mb
        return metrics

    def log_metrics_to_console(self, prefix: str, metrics: Dict[str, float]):
        def format_float(value: float) -> str:
            if value < 0.0001:
                return str(value)  # scientific notation
            elif value > 1000:
                return f"{int(value):,d}"
            elif value > 100:
                return f"{value:.1f}"
            elif value > 10:
                return f"{value:.2f}"
            elif value > 1:
                return f"{value:.3f}"
            else:
                return f"{value:.4f}"

        # log.info(
        print(
            f"{prefix}\n"
            + "\n".join(
                [
                    f"    {name}={format_float(value)}"
                    for name, value in metrics.items()
                    # there's too many optimizer metrics
                    # also skip non-float wandb.Metrics from inference evaluators
                    if (
                        isinstance(value, (int, float)) and (
                            name == "optim/total_grad_norm"
                            or (not name.startswith("optim/") and not name.startswith("batch/"))
                    ))
                ]
            )
        )

    def should_log_optim_metrics_this_step(self) -> bool:
        if self.cfg.wandb is None:
            # We only log optimizer-specific metrics to W&B, since there are usually too many metrics
            # to log to the console.
            return False
        optim_log_interval = self.cfg.optimizer.metrics_log_interval
        if optim_log_interval is None:
            optim_log_interval = self.cfg.wandb.log_interval
        else:
            optim_log_interval = max(optim_log_interval, self.cfg.wandb.log_interval)
        return self.global_step % optim_log_interval == 0

    def should_log_this_step(self) -> bool:
        if self.global_step % self.cfg.console_log_interval == 0:
            return True
        elif self.cfg.wandb is not None and self.global_step % self.cfg.wandb.log_interval == 0:
            return True
        else:
            return False

    def inference_eval(self) -> Dict[str, Union[float, WBValue]]:
        self.optim.zero_grad(set_to_none=True)
        self.fsdp_model.eval()
        all_metrics = {}
        for evaluator in self.inference_evaluators:
            log.info(f"Running evaluation for '{evaluator.label}'...")
            dataset_metrics = evaluator.evaluate_model(
                self.fsdp_model,
                device=self.device,
                autocast_precision=self.cfg.autocast_precision,
                is_distributed=True,
                inference_warmup=self._inference_warmup,
                pbar=False
            )
            self._inference_warmup = False
            self.log_metrics_to_console(f"{evaluator.label}", dataset_metrics)
            all_metrics.update({f"{evaluator.label}/{k}": v for k, v in dataset_metrics.items()})
        return all_metrics

    def eval(self) -> Dict[str, Any]:
        # Zero gradients and set model to 'eval' mode.
        self.optim.zero_grad(set_to_none=True)
        self.fsdp_model.eval()
        warmed_up = False
        torch.cuda.empty_cache()

        eval_metrics = {}
        for evaluator in self.evaluators:
            if not warmed_up:
                # The first batch can take a while as the iterator compiles/warms up, this
                # can cause the nodes to think they got de-synced since some of the nodes
                # might take much longer to get it and start the forward pass then others.
                # To avoid this, we manually sync the nodes for the first batch
                barrier()
                warmed_up = True

            log.info(f"Running evaluation for '{evaluator.label}'...")

            # Reset metrics.
            evaluator.reset_metrics()

            # Initialize data loader iterator.
            eval_batches = iter(evaluator.eval_loader)

            # Adjust how many batches to evaluate on.
            num_eval_batches = (
                evaluator.subset_num_batches
                if evaluator.subset_num_batches is not None
                else self.cfg.eval_subset_num_batches
            )
            if num_eval_batches > 0:
                if isinstance(evaluator.eval_loader, torch.utils.data.IterableDataset):
                    pass  # No defined length
                else:
                    num_eval_batches = min(num_eval_batches, len(evaluator.eval_loader))
                eval_batches = islice(eval_batches, num_eval_batches)

            # Run model over batches.
            for eval_step, eval_batch in enumerate(eval_batches):
                self.eval_step(eval_batch, evaluator)

                # Log to console.
                if eval_step + 1 == num_eval_batches or (eval_step + 1) % self.cfg.console_log_interval == 0:
                    log.info(f"[eval_step={eval_step + 1}/{num_eval_batches}]")

            if hasattr(evaluator.eval_loader, "reset"):
                evaluator.eval_loader.reset()  # Reset the loader to free RAM

            # Get final metrics.
            metrics = evaluator.compute_metrics()
            eval_metrics.update(metrics)
            self.log_metrics_to_console(f"{evaluator.label}", metrics)

            del eval_batches

        return eval_metrics

    def check_if_cancelled(self) -> Tuple[bool, int]:
        should_cancel = False
        cancel_reason: Optional[str] = None
        extra_steps = 0
        if get_global_rank() == 0:
            if self.cfg.time_limit is not None and time.time() - self._start_time >= self.cfg.time_limit:
                # First check if we've reached the training time limit.
                should_cancel = True
                cancel_reason = "time limit reached"
                extra_steps = self.cfg.extra_steps_after_cancel
            elif wandb.run is not None and (api_key := os.environ.get("WANDB_API_KEY")) is not None:
                # Finally, check if someone canceled the run from W&B by adding the 'cancel' / 'canceled' tag..
                # We won't see it in the run object. So we have to use the import/export API to check.
                from requests.exceptions import RequestException
                from wandb.errors import CommError

                try:
                    api = wandb.Api(api_key=api_key)
                    run = api.run(wandb.run.path)
                    for tag in run.tags or []:
                        if tag.lower() in {"cancel", "canceled", "cancelled"}:
                            should_cancel = True
                            cancel_reason = "Weights & Biases tag"
                            extra_steps = self.cfg.extra_steps_after_cancel
                            break
                except (RequestException, CommError):
                    log.info("Failed to check if W&B run is cancelled, continuing run.")

        run_canceled = synchronize_flag(should_cancel, self.device)
        if run_canceled:
            extra_steps = synchronize_value(extra_steps, self.device)
            if cancel_reason is None:
                if extra_steps > 0:
                    log.warning(f"Run canceled, stopping in {extra_steps} more steps...")
                else:
                    log.warning("Run canceled")
            else:
                if extra_steps > 0:
                    log.warning(f"Run canceled due to {cancel_reason}, stopping in {extra_steps} more steps...")
                else:
                    log.warning(f"Run canceled due to {cancel_reason}")

        return run_canceled, extra_steps

    def fit(self):
        log.info('fit start')
        if self.cfg.stop_after is not None:
            if self.cfg.stop_at is None:
                self.cfg.stop_at = self.global_step + self.cfg.stop_after
            else:
                self.cfg.stop_at = min(self.cfg.stop_at, self.global_step + self.cfg.stop_after)

        self._start_time = time.time()
        self._gc_init_state = gc.isenabled()  # cache if garbage collection is enabled, reset on close.

        # Disable automatic garbage collection, FSDP doesn't work well with it.
        if self.cfg.gen1_gc_interval is not None:
            gc.disable()

        if self.cfg.load_path is not None and self.global_step > 0 and self.cfg.eval_on_load:
            eval_metrics = self.eval()
            if wandb.run is not None:
                wandb.log(eval_metrics, step=self.global_step)

        # Set model to 'train' mode.
        self.fsdp_model.train()

        # Initialize monitors.
        assert self.cfg.device_train_batch_size is not None
        speed_monitor = SpeedMonitor(self.cfg.speed_monitor)
        lr_monitor = LRMonitor(self.optim)
        batch_monitor = BatchStatsMonitor()

        # Log system metrics at the start of training.
        sys_metrics = self.system_metrics()
        if sys_metrics:
            self.log_metrics_to_console("Pre-train system metrics", sys_metrics)
            if wandb.run is not None:
                wandb.log(sys_metrics, step=0)

        # Python Profiler stuff
        if self.cfg.python_profiling:
            python_profiler = cProfile.Profile()
        else:
            python_profiler = None

        # PyTorch Profiler stuff
        if self.cfg.torch_profiling and get_global_rank() == 0:
            from torch.profiler import schedule

            profiling_schedule = schedule(wait=1, warmup=5, active=3, repeat=1)

            def on_trace_ready(p):
                profiler_output_dir = Path(self.cfg.save_folder) / "profiler"
                profiler_output_dir.mkdir(exist_ok=True)

                output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=32)
                log.info(f"Profile by total GPU time at step {p.step_num}:\n{output}")
                output = p.key_averages().table(sort_by="self_cpu_time_total", row_limit=32)
                log.info(f"Profile by total CPU time at step {p.step_num}:\n{output}")

                p.export_chrome_trace(
                    str(trace_path := (profiler_output_dir / f"{p.step_num}.chrome_trace.json.gz"))
                )
                if self.cfg.remote_save_folder is not None:
                    upload_folder = f"{self.cfg.remote_save_folder.rstrip('/')}/profiler"
                    log.info(f"Tracing complete, uploading results to '{upload_folder}'...")
                    upload(trace_path, f"{upload_folder}/{trace_path.name}")

            from torch.profiler import ProfilerActivity

            torch_profiler = torch.profiler.profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=False,
                profile_memory=False,
                with_stack=True,
                schedule=profiling_schedule,
                on_trace_ready=on_trace_ready,
            )
            del profiling_schedule
        else:
            import contextlib

            torch_profiler = contextlib.nullcontext()

        # Train.
        first_batch: bool = True
        cancel_initiated: bool = False
        stop_at: Optional[int] = self.cfg.stop_at
        save_checkpoints: bool = True

        warmed_up = False

        # Cache the total dataset size once so we can report the data-epoch during training.
        # The VLA dataloader is an infinite stream (trainer `epoch` stays 0), so we derive the
        # epoch from (examples seen / dataset size) instead.
        try:
            self._train_dataset_size = len(self.train_loader.dataset)
        except Exception:
            self._train_dataset_size = 0
        if get_global_rank() == 0:
            log.info(f"Train dataset size (samples): {self._train_dataset_size:,d}")

        with torch_profiler as p:
            for epoch in range(self.epoch or 0, self.max_epochs):
                for batch in self.train_loader:
                    # print("*" * 50,'Trainer fit()')
                    # print(batch)
                    # print(type(batch),batch.keys())
                    if not warmed_up:
                        # The first batch can take a while as the iterator compiles/warms up, this
                        # can cause the nodes to think they got de-synced since some of the nodes
                        # might take much longer to get it and start the forward pass then others.
                        # To avoid this, we manually sync the nodes for the first batch
                        barrier()
                        warmed_up = True

                    # Bookkeeping.
                    # NOTE: To track the global batch size / number of tokens per batch we make the assumption that all
                    # batches see the same number of tokens, which should be the case for language model pre-training
                    # (at least when drop_last=True).
                    # Alternatively we'd have to use a distributed all reduce over seq_len here, but I don't want that
                    # overhead. So for now I'm putting these assertions here so if the assumption is violated it will
                    # fail loudly.
                    batch_size, seq_len = batch["input_ids"].shape
                    assert seq_len <= self.cfg.model.max_sequence_length
                    assert (
                        batch_size == (self.cfg.global_train_batch_size // get_world_size()),
                        f"batch size is {batch_size}, but bs={self.cfg.global_train_batch_size} among {get_local_world_size()} world size"
                    )
                    global_batch_size = batch_size * get_world_size()  # assumes batch size equal across ranks
                    self.global_step += 1
                    # print("Epoch:", epoch, "Step:", self.global_step, "Global Batch Size:", global_batch_size)
                    self.global_train_examples_seen_this_epoch += global_batch_size
                    self.global_train_tokens_seen += global_batch_size * seq_len
                    speed_monitor.batch_start(
                        self.global_train_tokens_seen,
                        batch_size * seq_len,  # num tokens in batch for this device
                        (batch["loss_masks"] > 0).sum(),
                        # We start monitoring speed after the first batch since the first
                        # batch might be an outlier due to compiling and other initialization overhead.
                        record=not first_batch,
                    )
                    batch_monitor.log_batch(batch)

                    should_log_this_step = self.should_log_this_step()

                    # Data-epoch: how many full passes over the dataset we've done so far.
                    data_epoch = (
                        self.global_train_examples_seen_this_epoch / self._train_dataset_size
                        if getattr(self, "_train_dataset_size", 0)
                        else 0.0
                    )

                    # Run train step on batch.
                    # print("Running train step...")
                    # s_time = time.time()
                    metrics = self.train_step(batch, reduce_global_loss=should_log_this_step)
                    # e_time = time.time()
                    # print("Train step done. Time taken:", e_time - s_time, "seconds")


                    # Maybe collect other metrics.
                    if should_log_this_step:
                        # Speed metrics.
                        metrics.update(speed_monitor.check())
                        # System metrics.
                        metrics.update(self.system_metrics())

                        # Learning rate metrics.
                        metrics.update(batch_monitor.check(self.device))

                        # Learning rate metrics.
                        metrics.update(lr_monitor.check())

                        # Data-epoch progress.
                        metrics["train/epoch"] = data_epoch

                    # Log metrics to console.
                    if self.global_step % self.cfg.console_log_interval == 0:
                        if get_global_rank() == 0:
                            self.log_metrics_to_console(
                                f"[step={self.global_step}/{self.max_steps} epoch={data_epoch:.3f}]", metrics
                            )
                        else:
                            log.info(f"[step={self.global_step}/{self.max_steps} epoch={data_epoch:.3f}]")

                    # Log metrics to W&B.
                    if (
                        wandb.run is not None
                        and self.cfg.wandb is not None
                        and self.global_step % self.cfg.wandb.log_interval == 0
                    ):
                        wandb.log(metrics, step=self.global_step)

                    # Check if/when run should be canceled.
                    if not cancel_initiated and self.global_step % self.cfg.canceled_check_interval == 0:
                        cancel_initiated, extra_steps = self.check_if_cancelled()
                        if cancel_initiated:
                            stop_at = (
                                self.global_step + extra_steps
                                if stop_at is None
                                else min(self.global_step + extra_steps, stop_at)
                            )

                    # Maybe save sharded checkpoint.
                    if save_checkpoints and (
                        cancel_initiated
                        or (
                            self.global_step % self.cfg.save_interval == 0
                            and self.cfg.save_num_checkpoints_to_keep != 0
                        )
                    ):
                        log.info("Saving checkpoint...")
                        checkpoint_path, _ = self.save_checkpoint(CheckpointType.sharded)
                        log.info(f"Checkpoint saved to {checkpoint_path}")

                        # Remove any ephemeral checkpoints.
                        while self.ephemeral_checkpoints:
                            self.remove_ephemeral_checkpoint()

                        # Reset speed monitor so that we don't count the time taken to save checkpoints.
                        speed_monitor.reset()

                        # If the run was just canceled this will be the final checkpoint.
                        if cancel_initiated:
                            save_checkpoints = False
                    elif (
                        self.cfg.save_interval_ephemeral is not None
                        and self.global_step % self.cfg.save_interval_ephemeral == 0
                    ):
                        log.info("Saving ephemeral checkpoint...")
                        checkpoint_path, _ = self.save_checkpoint(CheckpointType.sharded_ephemeral)
                        log.info(f"Checkpoint saved to {checkpoint_path}")

                        # Reset speed monitor so that we don't count the time taken to save checkpoints.
                        speed_monitor.reset()

                    # Maybe save unsharded checkpoint.
                    if (
                        save_checkpoints
                        and self.cfg.save_interval_unsharded is not None
                        and self.global_step % self.cfg.save_interval_unsharded == 0
                        and self.cfg.save_num_unsharded_checkpoints_to_keep != 0
                    ):
                        # self.fsdp_model.debug_module_hierarchy()###
                        # self.fsdp_model.set_size_module_is_root_false()
                        # self.fsdp_model.debug_module_hierarchy()###
                        log.info("Saving unsharded checkpoint...")
                        checkpoint_path, _ = self.save_checkpoint(CheckpointType.unsharded)
                        log.info(f"Unsharded checkpoint saved to {checkpoint_path}")

                        # Reset speed monitor so that we don't count the time taken to save checkpoints.
                        speed_monitor.reset()

                    # Maybe run evaluations.
                    last_step = stop_at and (self.global_step >= stop_at)
                    if not cancel_initiated and self.cfg.eval_interval > 0 and (
                        self.global_step % self.cfg.eval_interval == 0 or last_step):
                        eval_metrics = self.eval()

                        # Log metrics to W&B.
                        if wandb.run is not None:
                            wandb.log(eval_metrics, step=self.global_step)

                        # Reset speed monitor so that we don't count the time taken to run evaluations.
                        speed_monitor.reset()

                        # Reset model to 'train' mode.
                        self.fsdp_model.train()

                    if not cancel_initiated and (
                        self.inference_evaluators and
                        self.cfg.inf_eval_interval and
                        (self.global_step % self.cfg.inf_eval_interval == 0 or last_step)
                    ):
                        eval_metrics = self.inference_eval()

                        # Log metrics to W&B.
                        if wandb.run is not None:
                            wandb.log(eval_metrics, step=self.global_step)

                        # Reset speed monitor so that we don't count the time taken to run evaluations.
                        speed_monitor.reset()

                        # Reset model to 'train' mode.
                        self.fsdp_model.train()

                    # End of batch.
                    first_batch = False
                    if p is not None:
                        p.step()

                    if stop_at is not None and self.global_step >= stop_at:
                        break

                    # Run generation 1 garbage collection.
                    if self.cfg.gen1_gc_interval is not None and self.global_step % self.cfg.gen1_gc_interval == 0:
                        gc.collect(1)

                    # Python Profiler stuff
                    # We do this now, at the bottom of this loop, so we capture the work of getting the next batch.
                    if python_profiler is not None:
                        if self.global_step == 5:
                            python_profiler.enable()
                        elif self.global_step == 8:
                            python_profiler.disable()
                            python_profiler.print_stats(sort=SortKey.CUMULATIVE)
                            python_profiler = None
                else:
                    log.info("Training epoch complete")
                    self.epoch = epoch + 1
                    self.global_train_examples_seen_this_epoch = 0
                    if self.epoch < self.max_epochs:
                        self.dataset.reshuffle()
                    continue

                break

        # Save final checkpoint.
        if save_checkpoints:
            if (
                self.cfg.save_interval_unsharded is not None
                and self.last_unsharded_checkpoint_step != self.global_step
            ):
                log.info("Saving final unsharded model checkpoint...")
                checkpoint_path, _ = self.save_checkpoint(CheckpointType.unsharded)
                log.info(f"Unsharded checkpoint saved to {checkpoint_path}")
            elif (
                self.cfg.save_num_checkpoints_to_keep != 0
                and self.last_sharded_checkpoint_step != self.global_step
            ):
                log.info("Saving final checkpoint...")
                checkpoint_path, _ = self.save_checkpoint(CheckpointType.sharded)
                log.info(f"Checkpoint saved to {checkpoint_path}")

    def close(self, exit_code: int = 0) -> None:
        gc_cuda()

        if self._gc_init_state:
            gc.enable()
        else:
            gc.disable()
        if wandb.run is not None:
            wandb.finish(exit_code=exit_code, quiet=True)

    def __enter__(self) -> Trainer:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        del exc_val, exc_tb
        self.close(0 if exc_type is None else 1)


from .vla.affordvla import AffordVLA

class VLATrainer(Trainer):  
    """Inherit the VLA Trainer from Molmo Trainer"""  
      
    def __init__(  
        self,  
        cfg: TrainConfig,  
        action_loss_weight: float = 1.0, 
        **kwargs  
    ):  
        # 确保使用AffordVLA模型  
        if not isinstance(kwargs.get('model'), AffordVLA):  
            log.warning("Model is not AffordVLA, creating AffordVLA instance")  
            # kwargs['model'] = AffordVLA(cfg.model, action_dim=action_dim)  
          
        super().__init__(cfg, **kwargs)  
          
        self.action_loss_weight = action_loss_weight  
        self.action_criterion = torch.nn.L1Loss(reduction='mean')

        log.info(f"VLATrainer initialized with action_loss_weight={action_loss_weight}")  
    

    def save_action_head_checkpoint(self) -> Tuple[PathOrStr, Optional[PathOrStr]]:
        """
        Save only the action_head checkpoint in unsharded format.
        """
        # 确保所有进程在开始前同步
        barrier()
        # Zero-gradients to avoid gathering them.
        # self.optim.zero_grad(set_to_none=True)
        
        checkpoint_dir = Path(self.cfg.save_folder) / f"step{self.global_step}-action-head"
        remote_checkpoint_dir: Optional[str] = None
        if self.cfg.remote_save_folder is not None:
            remote_checkpoint_dir = f"{self.cfg.remote_save_folder.rstrip('/')}/{checkpoint_dir.name}"
        
        # Initialize action head checkpoints list if needed
        if not hasattr(self, 'action_head_checkpoints'):
            self.action_head_checkpoints: List[Path] = []
        self.action_head_checkpoints.append(checkpoint_dir)
        checkpointer = FullCheckpointer(self.cfg)
        try:
            with checkpointer._temporary_wd(checkpoint_dir) as temp_checkpoint_dir:
                # Use FSDP's FULL_STATE_DICT to get unsharded weights
                from torch.distributed.fsdp import FullStateDictConfig, StateDictType
                
                with FSDP.state_dict_type(
                    self.fsdp_model,
                    state_dict_type=StateDictType.FULL_STATE_DICT,
                    state_dict_config=FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
                ):
                    full_state_dict = self.fsdp_model.state_dict()
                    if get_global_rank() == 0:
                        # Get full model state dict
                        
                        
                        # Extract action_head related parameters
                        action_head_state_dict = {}
                        for key, value in full_state_dict.items():
                            if 'action_head' in key or 'proprio_projector' in key:
                                action_head_state_dict[key] = value
                        
                        # Save action head state dict using existing utility
                        save_state_dict(
                            temp_checkpoint_dir,
                            "action_head.pt",
                            action_head_state_dict,
                            upload_to=remote_checkpoint_dir,
                            save_overwrite=self.cfg.save_overwrite,
                            synchronize=False,
                        )
                        
                        # Save minimal metadata
                        metadata = {
                            "global_step": self.global_step,
                            "action_dim": getattr(self.cfg.model, 'action_dim', 7),
                            "action_head_type": self.cfg.model.action_head,
                        }
                        save_state_dict(
                            temp_checkpoint_dir,
                            "metadata.pt",
                            metadata,
                            upload_to=remote_checkpoint_dir,
                            save_overwrite=self.cfg.save_overwrite,
                            synchronize=False,
                        )
                        
                        # Create latest symlink
                        latest_path = Path(self.cfg.save_folder) / "latest-action-head"
                        latest_path.unlink(missing_ok=True)
                        try:
                            latest_path.symlink_to(checkpoint_dir.name, target_is_directory=True)
                        except FileExistsError:
                            if latest_path.resolve().name != checkpoint_dir.name:
                                raise
                
                # barrier()
                
        except FileExistsError:
            raise OLMoConfigurationError(
                f"Action head checkpoint for step {self.global_step} already exists, use --save-overwrite to overwrite it"
            )
        
        # Remove old action head checkpoints if needed
        if hasattr(self.cfg, 'save_num_action_head_checkpoints_to_keep') and self.cfg.save_num_action_head_checkpoints_to_keep > 0:
            while len(self.action_head_checkpoints) > self.cfg.save_num_action_head_checkpoints_to_keep:
                self._remove_action_head_checkpoint(0)
        
        barrier()
        
        if remote_checkpoint_dir is not None:
            return remote_checkpoint_dir, checkpoint_dir
        else:
            return checkpoint_dir, None

    def _remove_action_head_checkpoint(self, idx: int = 0):
        """Remove old action head checkpoints"""
        if not hasattr(self, 'action_head_checkpoints') or not self.action_head_checkpoints:
            return
            
        barrier()
        oldest_checkpoint = self.action_head_checkpoints.pop(idx)
        if get_global_rank() == 0 and oldest_checkpoint.is_dir():
            shutil.rmtree(oldest_checkpoint, ignore_errors=True)
            latest_path = Path(self.cfg.save_folder) / "latest-action-head"
            if latest_path.exists() and latest_path.resolve() == oldest_checkpoint.resolve():
                latest_path.unlink()
        barrier()

    def save_checkpoint(
        self, checkpoint_type: CheckpointType = CheckpointType.sharded
    ) -> Tuple[PathOrStr, Optional[PathOrStr]]:
        result: Tuple[PathOrStr, Optional[PathOrStr]]
        
        result = super().save_checkpoint(checkpoint_type)
        return result


    # (removed duplicate restore_unsharded_checkpoint definition)


    def model_forward(
        self, batch: Dict[str, Any], loss_reduction: str = "mean", compute_z_loss: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        # shape: (batch_size, seq_len, vocab_size)
        # Randomly mask state if enabled
        action_proprio = batch.get("proprio")
        if action_proprio is not None and hasattr(self.cfg, 'state_mask_prob') and self.cfg.state_mask_prob > 0.0:
            # Generate random mask: True means mask (set to 0), False means keep original
            mask = torch.rand(action_proprio.shape[0], device=action_proprio.device) < self.cfg.state_mask_prob
            if mask.any():
                # Clone to avoid modifying the original batch
                action_proprio = action_proprio.clone()
                # Mask entire state vector for samples where mask is True
                action_proprio[mask] = 0.0
        
        with torch.autocast("cuda", enabled=True, dtype=self.cfg.autocast_precision):
            # logits = self.fsdp_model(
            # print("Running model forward with actions...")
            # s_time = time.time()
            # outputs = self.fsdp_model.forward_with_actions(
            outputs = self.fsdp_model.forward(
                input_ids=batch["input_ids"],
                target_actions = batch.get("action"),  ##
                attention_mask=batch.get("attention_mask"),
                attention_bias=batch.get("attention_bias"),
                response_mask=(batch["loss_masks"] > 0) if "loss_masks" in batch else None,
                images=batch.get("images"),
                image_masks=batch.get("image_masks"),
                image_input_idx=batch.get("image_input_idx"),
                subsegment_ids=batch.get("subsegment_ids"),
                position_ids=batch.get("position_ids"),
                action_proprio=action_proprio,  ##
                proprio_token_idx = batch["proprio_token_idx"],  ##
                output_hidden_states=True if self.cfg.early_exit and self.cfg.model.action_head != 'flow_matching' else False,
                train_exit_random_layer = True if self.cfg.train_exit_random_layer else None,
                use_cache=True if self.cfg.model.action_head == 'flow_matching' else False
            )
            # ).logits
            # e_time = time.time()
            # print("Model forward done. Time taken:", e_time - s_time, "seconds")
            # logits = outputs["outputs"].logits
            logits = None


        if "labels" in batch:
            assert "loss_masks" in batch
            # assert loss_reduction == "none"
            loss_masks = batch["loss_masks"] * (batch["loss_masks"] > 0)
            labels = batch["labels"].long()
            labels.masked_fill_(~(loss_masks > 0), -100)
            labels = labels.view(-1)
            # logits_for_loss = logits.to(torch.float32).view(-1, logits.size(-1)) # for numerical stability
        else:
            # logits_for_loss = logits[..., :-1, :].contiguous()
            # shape: (batch_size * seq_len, vocab_size)
            # logits_for_loss = logits_for_loss.view(-1, logits_for_loss.size(-1))
            # shape: (batch_size, seq_len)
            labels = self.get_labels(batch)
            # shape: (batch_size * seq_len,)
            labels = labels.view(-1)
        # ce_loss, z_loss = self.loss_fn(
        #     logits_for_loss, labels, ignore_index=-100, reduction=loss_reduction,
        #     compute_z_loss=compute_z_loss, z_loss_scale=self.cfg.softmax_auxiliary_loss_scale,
        # )
        ce_loss, z_loss = 0,0 ##

        bs = batch["input_ids"].shape[0]
        if loss_reduction == "none":
            # Reshape (batch_size * seq_len,) -> (batch_size, seq_len)
            # ce_loss = ce_loss.view(bs, -1)
            # if z_loss is not None:
            #     z_loss = z_loss.view(bs, -1)
            pass

        # accuracy = torch.argmax(logits_for_loss, dim=-1) == labels
        accuracy = 0.1
        if "labels" in batch:
            ce_loss = ce_loss * loss_masks
            if z_loss is not None:
                z_loss = z_loss * loss_masks
            # accuracy = accuracy.view(bs, -1)
            # accuracy = accuracy * loss_masks
        else:
            # accuracy = (accuracy * (labels >= 0))
            # accuracy = accuracy.view(bs, -1)
            pass

        def get_action_loss(pred_acts, diff_pred, diff_target):
            if self.model.config.action_head == "l1_regression":
                if pred_acts is not None and 'action' in batch: 
                    # print("VLATrainer model_forward(), pred_acts shape:", pred_acts.shape)
                    assert batch["action"] is not None, "Target action data must be provided in batch"
                    # extract target actions from batch metadata
                    target_actions =  batch["action"]
                    assert target_actions is not None, "Target actions must be provided in batch metadata"
                    assert pred_acts.shape == target_actions.shape, f"Shape mismatch: pred_acts {pred_acts.shape}, target_actions {target_actions.shape}"
                    
                    # 仅在非pad位置计算L1损失
                    pad_mask = batch.get("action_pad_mask", None)
                    assert pad_mask is not None, "Action pad mask must be provided in batch"
                    valid_mask = (~pad_mask.bool()).to(device=pred_acts.device)
                    assert valid_mask.shape == target_actions.shape, f"Shape mismatch: valid_mask {valid_mask.shape}, target_actions {target_actions.shape}"
                    l1_elems = F.l1_loss(pred_acts, target_actions, reduction="none")
                    denom = valid_mask.sum().clamp_min(1)
                    action_loss = (l1_elems * valid_mask).sum() / denom

                    # else:
                        # action_loss = self.action_criterion(pred_acts, target_actions)  
            # elif self.model.config.action_head.startswith("diffusion"):
            else:
                # 仅在非pad位置计算MSE损失（若掩码形状匹配）
                pad_mask = batch.get("action_pad_mask", None)

                assert pad_mask is not None, "Action pad mask must be provided in batch"
                valid_mask = (~pad_mask.bool()).to(device=diff_pred.device)
                assert valid_mask.shape == diff_target.shape, f"Shape mismatch: valid_mask {valid_mask.shape}, diff_target {diff_target.shape}"
                mse_elems = F.mse_loss(diff_pred, diff_target, reduction="none")
                denom = valid_mask.sum().clamp_min(1)
                action_loss = (mse_elems * valid_mask).sum() / denom

                # else:
                    # action_loss = F.mse_loss(diff_pred, diff_target, reduction="mean")
            return action_loss

        # add action prediction loss
        action_loss = torch.tensor(0.0, device=self.device)  

        # check the type of outputs, dict or list
        if isinstance(outputs, dict):
            pred_acts = outputs['predicted_actions']
            diff_pred = outputs['diffusion_pred']
            diff_target = outputs['diffusion_target']
            action_loss += get_action_loss(pred_acts, diff_pred, diff_target)
        elif isinstance(outputs, list):
            num = len(outputs)
            for output in outputs:
                pred_acts = output['predicted_actions']
                diff_pred = output['diffusion_pred']
                diff_target = output['diffusion_target']
                action_loss += get_action_loss(pred_acts, diff_pred, diff_target)
            action_loss /= num
        
        ##
        return accuracy, ce_loss, z_loss, logits, action_loss
  

    def train_batch(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        # Split into micro-batches.
        micro_batches = self.split_batch(batch)
        has_labels = False # "labels" in batch
        if has_labels:
            loss_masks = batch["loss_masks"] * (batch["loss_masks"] > 0)
            if self.cfg.batch_divisor == BatchDivisor.global_batch:
                batch_size_in_tokens = loss_masks.sum()
                dist.all_reduce(batch_size_in_tokens)
                batch_size_in_tokens.div_(get_world_size())
            elif self.cfg.batch_divisor == BatchDivisor.device_batch:
                batch_size_in_tokens = loss_masks.sum()
            else:
                raise ValueError()
        else:
            batch_size_in_tokens = batch["input_ids"].numel()

        del batch  # in case this helps reduce memory

        ce_batch_loss = torch.tensor(0.0, device=self.device)
        batch_accuracy = torch.tensor(0.0, device=self.device)
        z_batch_loss = None if not self.cfg.softmax_auxiliary_loss else torch.tensor(0.0, device=self.device)
        lb_batch_loss = (
            None if self.model.config.block_type != BlockType.moe else torch.tensor(0.0, device=self.device)
        )
        moe_z_batch_loss = (
            None if not self.model.config.moe_zloss_weight else torch.tensor(0.0, device=self.device)
        )
        expert_assignments = (
            None
            if (
                (self.model.config.block_type != BlockType.moe)
                or (self.model.config.moe_log_expert_assignment is False)
            )
            else torch.zeros((self.model.config.n_layers, self.model.config.moe_num_experts))
        )
        for micro_batch in micro_batches:
            # print("Running model_forward for micro batch...")
            s_time = time.time()
            accuracy, ce_loss, z_loss, logits, action_loss = self.model_forward(
                micro_batch, compute_z_loss=self.cfg.softmax_auxiliary_loss, loss_reduction="none" if has_labels else "sum"
            )
            e_time = time.time()
            # print("Model forward done. Time taken:", e_time - s_time, "seconds")

            if has_labels:
                accuracy = accuracy.sum()
                ce_loss = ce_loss.sum()
                if z_loss is not None:
                    z_loss = z_loss.sum()

            ce_loss = ce_loss / batch_size_in_tokens
            accuracy = accuracy / batch_size_in_tokens

            # In case this helps with memory utilization.
            del micro_batch

            # Update overall CE batch loss.
            ce_batch_loss += 0.1
            batch_accuracy += 0.1
            
            # Get loss to optimize for.
            if self.cfg.softmax_auxiliary_loss:
                z_batch_loss += 0.1
            #     assert z_loss is not None
            #     assert z_batch_loss is not None
            #     z_loss = z_loss / batch_size_in_tokens

            #     loss = ce_loss + z_loss

            #     # Update overall Z batch loss.
            #     z_batch_loss += z_loss.detach()
            # else:
            #     loss = ce_loss

            loss = action_loss * self.action_loss_weight # replace loss with action loss

            del logits

            if self.model.config.block_type == BlockType.moe:
                if self.model.config.moe_zloss_weight:
                    lb_loss, moe_z_loss = batched_load_balancing_loss(self.moe_args)
                    lb_loss = lb_loss / len(micro_batches)
                    moe_z_loss = moe_z_loss / len(micro_batches)
                elif self.model.config.moe_loss_weight:
                    lb_loss = batched_load_balancing_loss(self.moe_args) / len(micro_batches)
                if self.model.config.moe_log_expert_assignment:
                    if self.model.config.moe_zloss_weight:
                        tokens_per_expert, _, _ = zip(*get_load_balancing_loss())
                    else:
                        tokens_per_expert, _ = zip(*get_load_balancing_loss())
                    expert_assignments += torch.stack(tokens_per_expert, dim=0).cpu()
                clear_load_balancing_loss()
                if self.model.config.moe_loss_weight:
                    loss += lb_loss
                    lb_batch_loss += lb_loss.detach()
                if self.model.config.moe_zloss_weight:
                    loss += moe_z_loss
                    moe_z_batch_loss += moe_z_loss.detach()

            # Run backward pass.
            loss.backward()

        return ce_batch_loss, z_batch_loss, batch_accuracy, lb_batch_loss, moe_z_batch_loss, expert_assignments,action_loss

    def train_step(self, batch: Dict[str, Any], reduce_global_loss: bool = True) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        # Record how many instances are going to be skipped (masked out).
        if (instance_mask := batch.get("instance_mask")) is not None:
            metrics["train/masked_instances_local_rank"] = (~instance_mask).sum().item()

        # Zero-gradients.
        self.optim.zero_grad(set_to_none=True)

        # Move tensors to the right device.
        batch = self.move_to_device(batch, self.device)

        # Run forward-backward pass.
        # print("Running train batch...")
        s_time = time.time()
        ce_batch_loss, z_batch_loss, batch_accuracy, lb_batch_loss, moe_z_batch_loss, expert_assignments,action_loss = self.train_batch(batch)
        e_time = time.time()
        # print("Train batch done. Time taken:", e_time - s_time, "seconds")

        # Collect loss, potentially reducing over all ranks.
        if reduce_global_loss:
            dist.reduce(ce_batch_loss, 0)
            ce_batch_loss.div_(get_world_size())
            if z_batch_loss is not None:
                dist.reduce(z_batch_loss, 0)
                z_batch_loss.div_(get_world_size())
            if batch_accuracy is not None:
                dist.reduce(batch_accuracy, 0)
                batch_accuracy.div_(get_world_size())
            if lb_batch_loss is not None:
                dist.reduce(lb_batch_loss, 0)
                lb_batch_loss.div_(get_world_size())
            if moe_z_batch_loss is not None:
                dist.reduce(moe_z_batch_loss, 0)
                moe_z_batch_loss.div_(get_world_size())
            if action_loss is not None:
                dist.reduce(action_loss, 0)
                action_loss.div_(get_world_size())

        # Clip gradient norms and collect param/gradient/optim metrics.
        should_log_optim_metrics_this_step = self.should_log_optim_metrics_this_step()
        optim_metrics = self.optim.clip_grads_and_collect_metrics(
            self.global_step,
            collect_param_metrics=should_log_optim_metrics_this_step,
            # passing this process group here ensures metrics are reduced correctly when we're using
            # HYBRID sharding.
            process_group=self.fsdp_model.process_group,
            multi_modal=self.cfg.model.vision_backbone is not None,
        )

        if self.cfg.model.vision_backbone is not None:
            initial_lr_dict = {
                "connector": self.cfg.optimizer.connector_learning_rate,
                "vit": self.cfg.optimizer.vit_learning_rate,
                "llm": self.cfg.optimizer.llm_learning_rate,
                "ah": self.cfg.optimizer.action_head_learning_rate,
            }
            for group in self.optim.param_groups:
                group_name = group["group_name"]
                component_name = group_name.split("_")[0]
                group["lr"] = self.scheduler.get_lr(
                    initial_lr_dict[component_name],
                    self.scheduler_current,
                    self.scheduler_max,
                    group_name,
                )
                group["max_grad_norm"] = self.scheduler.get_max_grad_norm(
                    self.cfg.max_grad_norm, self.scheduler_current, self.scheduler_max
                )
                group["max_grad_norm_ratio"] = self.scheduler.get_max_grad_norm(
                    self.cfg.max_grad_norm_ratio, self.scheduler_current, self.scheduler_max
                )
        else:
            for group in self.optim.param_groups:
                # TODO (epwalsh): if we want to enable different LRs or gradient clipping settings per group
                # we should pass `group["initial_lr"]` or `group["initial_max_grad_norm"]` here instead of
                # the corresponding values from `self.cfg`.
                group["lr"] = self.scheduler.get_lr(
                    self.cfg.optimizer.learning_rate, self.scheduler_current, self.scheduler_max
                )
                group["max_grad_norm"] = self.scheduler.get_max_grad_norm(
                    self.cfg.max_grad_norm, self.scheduler_current, self.scheduler_max
                )
                group["max_grad_norm_ratio"] = self.scheduler.get_max_grad_norm(
                    self.cfg.max_grad_norm_ratio, self.scheduler_current, self.scheduler_max
                )

        # Optimizer step.
        self.optim.step()

        # Collect metrics and check for NaN loss.
        # NOTE: this involves a bunch of host-device syncs so we wait until the last moment to do this.
        # if torch.isnan(ce_batch_loss):
        #     raise ValueError("nan loss encountered")
        if z_batch_loss is not None and torch.isnan(z_batch_loss):
            raise ValueError("nan loss encountered")
        if action_loss is not None and torch.isnan(action_loss):
            raise ValueError("nan action loss encountered")
        for key, value in optim_metrics.items():
            metrics[f"optim/{key}"] = value.item()
        self.cur_train_loss = ce_batch_loss.item()
        self.min_train_loss = min(self.min_train_loss, self.cur_train_loss)
        # metrics["train/CrossEntropyLoss"] = self.cur_train_loss
        # metrics["train/Perplexity"] = math.exp(self.cur_train_loss)
        # metrics["train/Accuracy"] = batch_accuracy.item()

        if self.model.config.action_head == "l1_regression":
            metrics["train/ActionL1Loss"] = action_loss.item() if action_loss is not None else 0.0
        # elif self.model.config.action_head == "diffusion":
        else:
            metrics["train/ActionNoiseL2Loss"] = action_loss.item() if action_loss is not None else 0.0
            
        # if z_batch_loss is not None:
        #     metrics["train/ZLoss"] = z_batch_loss.item()
        if lb_batch_loss is not None:
            metrics["train/LoadBalancingLoss"] = lb_batch_loss.item()
            # Log assignment metrics.
            if expert_assignments is not None:
                for layer_idx, expert_assignments_layer in enumerate(expert_assignments):
                    total_tokens = expert_assignments_layer.sum().item()
                    for expert_idx, expert_assignment in enumerate(expert_assignments_layer):
                        metrics[f"train/TokensPercentage/layer{layer_idx}/expert{expert_idx}"] = (
                            expert_assignment.item() / total_tokens
                        ) * 100
                        metrics[
                            f"train/TokensTotal/layer{layer_idx}/expert{expert_idx}"
                        ] = expert_assignment.item()
        if moe_z_batch_loss is not None:
            metrics["train/MoEZLoss"] = moe_z_batch_loss.item()

        # Maybe collect post-step optimizer-specific metrics.
        if should_log_optim_metrics_this_step:
            optim_metrics = self.optim.get_post_step_metrics(
                self.fsdp_model, process_group=self.fsdp_model.process_group
            )
            for key, value in optim_metrics.items():
                metrics[f"optim/{key}"] = value.item()

        return metrics

  
    def extract_target_actions(self, batch: Dict[str, Any]) -> Optional[torch.Tensor]:  
        """Extract the target action from the batch data"""  
        if 'metadata' not in batch:  
            return None  
          
        batch_size = len(batch['metadata'])  
        target_actions_list = []  
          
        for i in range(batch_size):  
            metadata = batch['metadata'][i]  
            if isinstance(metadata, dict) and 'action' in metadata:  
                action = metadata['action']  
                if isinstance(action, torch.Tensor):  
                    target_actions_list.append(action)  
                elif hasattr(action, '__iter__'):  
                    target_actions_list.append(torch.tensor(action, dtype=torch.float32))  
                else:  
                    return None  
            else:  
                return None  
          
        if target_actions_list:  
            target_actions = torch.stack(target_actions_list).to(batch['input_ids'].device)  
            # # Expand the sequence dimension if necessary
            # if target_actions.dim() == 2:  
            #     target_actions = target_actions.unsqueeze(1).expand(-1, 17, -1)  # 17是动作序列长度  
            return target_actions  
          
        return None  

    def fit(self):
        log.info('fit start')
        
        if self.cfg.stop_after is not None:
            if self.cfg.stop_at is None:
                self.cfg.stop_at = self.global_step + self.cfg.stop_after
            else:
                self.cfg.stop_at = min(self.cfg.stop_at, self.global_step + self.cfg.stop_after)

        self._start_time = time.time()
        self._gc_init_state = gc.isenabled()  # cache if garbage collection is enabled, reset on close.

        # Disable automatic garbage collection, FSDP doesn't work well with it.
        if self.cfg.gen1_gc_interval is not None:
            gc.disable()

        if self.cfg.load_path is not None and self.global_step > 0 and self.cfg.eval_on_load:
            eval_metrics = self.eval()
            if wandb.run is not None:
                wandb.log(eval_metrics, step=self.global_step)

        # Set model to 'train' mode.
        self.fsdp_model.train()

        # Initialize monitors.
        assert self.cfg.device_train_batch_size is not None
        speed_monitor = SpeedMonitor(self.cfg.speed_monitor)
        lr_monitor = LRMonitor(self.optim)
        batch_monitor = BatchStatsMonitor()

        # Log system metrics at the start of training.
        sys_metrics = self.system_metrics()
        if sys_metrics:
            self.log_metrics_to_console("Pre-train system metrics", sys_metrics)
            if wandb.run is not None:
                wandb.log(sys_metrics, step=0)

        # Python Profiler stuff
        if self.cfg.python_profiling:
            python_profiler = cProfile.Profile()
        else:
            python_profiler = None

        # PyTorch Profiler stuff
        if self.cfg.torch_profiling and get_global_rank() == 0:
            from torch.profiler import schedule

            profiling_schedule = schedule(wait=1, warmup=5, active=3, repeat=1)

            def on_trace_ready(p):
                profiler_output_dir = Path(self.cfg.save_folder) / "profiler"
                profiler_output_dir.mkdir(exist_ok=True)

                output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=32)
                log.info(f"Profile by total GPU time at step {p.step_num}:\n{output}")
                output = p.key_averages().table(sort_by="self_cpu_time_total", row_limit=32)
                log.info(f"Profile by total CPU time at step {p.step_num}:\n{output}")

                p.export_chrome_trace(
                    str(trace_path := (profiler_output_dir / f"{p.step_num}.chrome_trace.json.gz"))
                )
                if self.cfg.remote_save_folder is not None:
                    upload_folder = f"{self.cfg.remote_save_folder.rstrip('/')}/profiler"
                    log.info(f"Tracing complete, uploading results to '{upload_folder}'...")
                    upload(trace_path, f"{upload_folder}/{trace_path.name}")

            from torch.profiler import ProfilerActivity

            torch_profiler = torch.profiler.profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=False,
                profile_memory=False,
                with_stack=True,
                schedule=profiling_schedule,
                on_trace_ready=on_trace_ready,
            )
            del profiling_schedule
        else:
            import contextlib

            torch_profiler = contextlib.nullcontext()

        # Train.
        first_batch: bool = True
        cancel_initiated: bool = False
        stop_at: Optional[int] = self.cfg.stop_at
        save_checkpoints: bool = True

        warmed_up = False
        console_log_metrics = {}
        wandb_log_metrics = {}
        with torch_profiler as p:
            for epoch in range(self.epoch or 0, self.max_epochs):
                for batch in self.train_loader:
                    # print("*" * 50,'Trainer fit()')
                    # print(batch)
                    # print(type(batch),batch.keys())
                    # print("VLATrainer fit() epoch:", epoch, "batch size:", len(batch["input_ids"]))
                    # print('*'*13,'Before warmup...')
                    if not warmed_up:
                        # The first batch can take a while as the iterator compiles/warms up, this
                        # can cause the nodes to think they got de-synced since some of the nodes
                        # might take much longer to get it and start the forward pass then others.
                        # To avoid this, we manually sync the nodes for the first batch
                        barrier()
                        warmed_up = True
                    # print('*'*13,'After warmup...')
                    # Bookkeeping.
                    # NOTE: To track the global batch size / number of tokens per batch we make the assumption that all
                    # batches see the same number of tokens, which should be the case for language model pre-training
                    # (at least when drop_last=True).
                    # Alternatively we'd have to use a distributed all reduce over seq_len here, but I don't want that
                    # overhead. So for now I'm putting these assertions here so if the assumption is violated it will
                    # fail loudly.
                    batch_size, seq_len = batch["input_ids"].shape
                    assert seq_len <= self.cfg.model.max_sequence_length
                    assert (
                        batch_size == (self.cfg.global_train_batch_size // get_world_size()),
                        f"batch size is {batch_size}, but bs={self.cfg.global_train_batch_size} among {get_local_world_size()} world size"
                    )
                    global_batch_size = batch_size * get_world_size()  # assumes batch size equal across ranks
                    self.global_step += 1
                    # print("Epoch:", epoch, "Step:", self.global_step, "Global Batch Size:", global_batch_size)
                    self.global_train_examples_seen_this_epoch += global_batch_size
                    self.global_train_tokens_seen += global_batch_size * seq_len
                    speed_monitor.batch_start(
                        self.global_train_tokens_seen,
                        batch_size * seq_len,  # num tokens in batch for this device
                        (batch["loss_masks"] > 0).sum(),
                        # We start monitoring speed after the first batch since the first
                        # batch might be an outlier due to compiling and other initialization overhead.
                        record=not first_batch,
                    )
                    batch_monitor.log_batch(batch)

                    should_log_this_step = True # self.should_log_this_step()

                    # Run train step on batch.
                    # print("Running train step...")
                    metrics = self.train_step(batch, reduce_global_loss=should_log_this_step)

                    for k, v in metrics.items():
                        if k == 'optim/total_grad_norm':
                            if k not in console_log_metrics:
                                console_log_metrics[k] = 0
                            if k not in wandb_log_metrics:
                                wandb_log_metrics[k] = 0
                            console_log_metrics[k] += v
                            wandb_log_metrics[k] += v
                        if 'Loss' in k:
                            if k not in console_log_metrics:
                                console_log_metrics[k] = 0
                            if k not in wandb_log_metrics:
                                wandb_log_metrics[k] = 0
                            console_log_metrics[k] += v
                            wandb_log_metrics[k] += v

                    # Maybe collect other metrics.
                    if should_log_this_step:
                        # Speed metrics.
                        metrics.update(speed_monitor.check())
                        # System metrics.
                        metrics.update(self.system_metrics())

                        # Learning rate metrics.
                        metrics.update(batch_monitor.check(self.device))

                        # Learning rate metrics.
                        metrics.update(lr_monitor.check())

                    # Log metrics to console.
                    if self.global_step % self.cfg.console_log_interval == 0:
                        if get_global_rank() == 0:
                            for k, v in console_log_metrics.items():
                                metrics[k] = v/self.cfg.console_log_interval
                            console_log_metrics.clear()
                            self.log_metrics_to_console(f"[step={self.global_step}/{self.max_steps}]", metrics)
                        else:
                            log.info(f"[step={self.global_step}/{self.max_steps}]")

                    # Log metrics to W&B.
                    if (
                        wandb.run is not None
                        and self.cfg.wandb is not None
                        and self.global_step % self.cfg.wandb.log_interval == 0
                    ):
                        for k, v in wandb_log_metrics.items():
                            metrics[k] = v/self.cfg.wandb.log_interval
                        wandb_log_metrics.clear()
                        wandb.log(metrics, step=self.global_step)

                    # Check if/when run should be canceled.
                    if not cancel_initiated and self.global_step % self.cfg.canceled_check_interval == 0:
                        cancel_initiated, extra_steps = self.check_if_cancelled()
                        if cancel_initiated:
                            stop_at = (
                                self.global_step + extra_steps
                                if stop_at is None
                                else min(self.global_step + extra_steps, stop_at)
                            )

                    try:
                        # Maybe save sharded checkpoint.
                        if save_checkpoints and (
                            cancel_initiated
                            or (
                                self.global_step % self.cfg.save_interval == 0
                                and self.cfg.save_num_checkpoints_to_keep != 0
                            )
                        ):
                            log.info("Saving checkpoint...")
                            checkpoint_path, _ = self.save_checkpoint(CheckpointType.sharded)
                            log.info(f"Checkpoint saved to {checkpoint_path}")

                            # Remove any ephemeral checkpoints.
                            while self.ephemeral_checkpoints:
                                self.remove_ephemeral_checkpoint()

                            # Reset speed monitor so that we don't count the time taken to save checkpoints.
                            speed_monitor.reset()

                            # If the run was just canceled this will be the final checkpoint.
                            if cancel_initiated:
                                save_checkpoints = False
                        elif (
                            self.cfg.save_interval_ephemeral is not None
                            and self.global_step % self.cfg.save_interval_ephemeral == 0
                        ):
                            log.info("Saving ephemeral checkpoint...")
                            checkpoint_path, _ = self.save_checkpoint(CheckpointType.sharded_ephemeral)
                            log.info(f"Checkpoint saved to {checkpoint_path}")

                            # Reset speed monitor so that we don't count the time taken to save checkpoints.
                            speed_monitor.reset()

                        # Maybe save unsharded checkpoint.
                        if (
                            save_checkpoints
                            and self.cfg.save_interval_unsharded is not None
                            and self.global_step % self.cfg.save_interval_unsharded == 0
                            and self.cfg.save_num_unsharded_checkpoints_to_keep != 0
                        ):
                            # self.fsdp_model.debug_module_hierarchy()###
                            # self.fsdp_model.set_size_module_is_root_false()
                            # self.fsdp_model.debug_module_hierarchy()###
                            log.info("Saving unsharded checkpoint...")
                            checkpoint_path, _ = self.save_checkpoint(CheckpointType.unsharded)
                            log.info(f"Unsharded checkpoint saved to {checkpoint_path}")

                            # Reset speed monitor so that we don't count the time taken to save checkpoints.
                            speed_monitor.reset()
                        
                        # Save action head checkpoint.
                        if (
                            save_checkpoints
                            and self.cfg.save_interval_action_head is not None
                            and self.global_step % self.cfg.save_interval_action_head == 0
                            and self.cfg.save_num_action_head_checkpoints_to_keep != 0
                        ):
                            log.info("Saving action head checkpoint...")
                            action_head_checkpoint_path, _ = self.save_checkpoint(CheckpointType.action_head)
                            log.info(f"Action head checkpoint saved to {action_head_checkpoint_path}")

                            # Reset speed monitor so that we don't count the time taken to save checkpoints.
                            speed_monitor.reset()

                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        log.info(f"Saving checkpoint failed: {e}")
                    ##

                    # Maybe run evaluations.
                    last_step = stop_at and (self.global_step >= stop_at)
                    if not cancel_initiated and self.cfg.eval_interval > 0 and (
                        self.global_step % self.cfg.eval_interval == 0 or last_step):
                        eval_metrics = self.eval()

                        # Log metrics to W&B.
                        if wandb.run is not None:
                            wandb.log(eval_metrics, step=self.global_step)

                        # Reset speed monitor so that we don't count the time taken to run evaluations.
                        speed_monitor.reset()

                        # Reset model to 'train' mode.
                        self.fsdp_model.train()

                    if not cancel_initiated and (
                        self.inference_evaluators and
                        self.cfg.inf_eval_interval and
                        (self.global_step % self.cfg.inf_eval_interval == 0 or last_step)
                    ):
                        eval_metrics = self.inference_eval()

                        # Log metrics to W&B.
                        if wandb.run is not None:
                            wandb.log(eval_metrics, step=self.global_step)

                        # Reset speed monitor so that we don't count the time taken to run evaluations.
                        speed_monitor.reset()

                        # Reset model to 'train' mode.
                        self.fsdp_model.train()

                    # End of batch.
                    first_batch = False
                    if p is not None:
                        p.step()

                    if stop_at is not None and self.global_step >= stop_at:
                        break
                    # Run generation 1 garbage collection.
                    if self.cfg.gen1_gc_interval is not None and self.global_step % self.cfg.gen1_gc_interval == 0:
                        gc.collect(1)

                    # Python Profiler stuff
                    # We do this now, at the bottom of this loop, so we capture the work of getting the next batch.
                    if python_profiler is not None:
                        if self.global_step == 5:
                            python_profiler.enable()
                        elif self.global_step == 8:
                            python_profiler.disable()
                            python_profiler.print_stats(sort=SortKey.CUMULATIVE)
                            python_profiler = None
                else:
                    log.info("Training epoch complete")
                    self.epoch = epoch + 1
                    self.global_train_examples_seen_this_epoch = 0
                    if self.epoch < self.max_epochs:
                        self.dataset.reshuffle()
                    continue

                break

        # Save final checkpoint.
        if save_checkpoints:
            if (
                self.cfg.save_interval_unsharded is not None
                and self.last_unsharded_checkpoint_step != self.global_step
            ):
                log.info("Saving final unsharded model checkpoint...")
                checkpoint_path, _ = self.save_checkpoint(CheckpointType.unsharded)
                log.info(f"Unsharded checkpoint saved to {checkpoint_path}")
            elif (
                self.cfg.save_num_checkpoints_to_keep != 0
                and self.last_sharded_checkpoint_step != self.global_step
            ):
                log.info("Saving final checkpoint...")
                checkpoint_path, _ = self.save_checkpoint(CheckpointType.sharded)
                log.info(f"Checkpoint saved to {checkpoint_path}")

                
    def eval_batch(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast("cuda", enabled=True, dtype=self.cfg.autocast_precision):
            acc, ce_loss, z_loss, logits, action_loss = self.model_forward(batch, loss_reduction="none", compute_z_loss=True)
        if "labels" in batch:
            loss_masks = batch["loss_masks"] * (batch["loss_masks"] > 0)
            batch_size_in_tokens = loss_masks.sum(-1)

            return dict(
                total_weight=batch_size_in_tokens.sum(),
                total_loss=ce_loss.sum(),
                total_accuracy=acc.sum(),
                total_zloss=z_loss.sum(),
                batch_loss=ce_loss.sum()/batch_size_in_tokens.sum(),
                batch_accuracy=acc.sum()/batch_size_in_tokens.sum(),
                batch_zloss=z_loss.sum()/batch_size_in_tokens.sum(),
                logits=logits,
                action_loss=action_loss
            )
        else:
            return dict(
                instance_loss=ce_loss.mean(-1),
                instance_aaccuracy=acc.mean(-1),
                batch_loss=ce_loss.mean(),
                batch_accuracy=acc.mean(),
                z_loss=z_loss.mean(),
                logits=logits,
                action_loss=action_loss
            )

    # def eval_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:  
    #     """重写评估批次方法"""  
    #     self.fsdp_model.eval()  
          
    #     with torch.no_grad():  
    #         with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.cfg.precision == "amp_bf16"):  
    #             if hasattr(self.model, 'forward_with_actions'):  
    #                 outputs = self.fsdp_model.forward_with_actions(  
    #                     input_ids=batch["input_ids"],  
    #                     images=batch.get("images"),  
    #                     image_masks=batch.get("image_masks"),  
    #                     image_input_idx=batch.get("image_input_idx"),  
    #                 )  
    #             else:  
    #                 outputs = self.fsdp_model(  
    #                     input_ids=batch["input_ids"],  
    #                     images=batch.get("images"),  
    #                     image_masks=batch.get("image_masks"),  
    #                     image_input_idx=batch.get("image_input_idx"),  
    #                 )  
                  
    #             loss_dict = self.compute_vla_loss(batch, outputs)  
          
    #     return {  
    #         'loss': loss_dict['total_loss'].item(),  
    #         'language_loss': loss_dict['language_loss'].item(),  
    #         'action_loss': loss_dict['action_loss'].item(),  
    #     }  
  
    # def log_metrics(self, metrics: Dict[str, Any], step: int, prefix: str = "train"):  
    #     """记录VLA特定的指标"""  
    #     super().log_metrics(metrics, step, prefix)  
          
    #     # 记录动作损失相关指标  
    #     if 'action_loss' in metrics:  
    #         log.info(f"{prefix}/action_loss: {metrics['action_loss']:.6f}")  
    #     if 'language_loss' in metrics:  
    #         log.info(f"{prefix}/language_loss: {metrics['language_loss']:.6f}")

"""简化的DiT Action Trainer"""
import yaml
from .config import DiTActionConfig

log = logging.getLogger('train')
log.propagate = True

class SimpleDiTActionTrainer:
    """简化的DiT Action训练器"""
    
    def __init__(
        self,
        config: DiTActionConfig,
        model: torch.nn.Module,
        text_model: Optional[torch.nn.Module] ,
        vision_model: Optional[torch.nn.Module],
        train_dataloader: DataLoader,
        device: torch.device,
    ):
        self.config = config
        self.model = model
        self.text_model = text_model
        self.vision_model = vision_model
        self.train_dataloader = train_dataloader
        self.device = device
        
        # 训练状态
        self.global_step = 0
        self.epoch = 0
        
         # 确定训练精度
        self.train_dtype = torch.bfloat16 if config.precision == "amp_bf16" else torch.float32
        
        self.text_model = self.text_model.to(device, dtype=self.train_dtype)
        self.vision_model = self.vision_model.to(device, dtype=self.train_dtype)

        # 主模型也要设置正确的数据类型
        self.model = self.model.to(device, dtype=self.train_dtype)


        # 检查点管理
        self.checkpoints: List[Path] = []  # 保存所有检查点路径
        
        # 设置优化器
        param_groups = [
            {'params': self.model.parameters(), 'lr': config.learning_rate, 'weight_decay': config.weight_decay}
        ]
        if config.ft_vit:
            param_groups.append(
                {'params': self.vision_model.parameters(),'lr': config.learning_rate,'weight_decay': config.weight_decay}
            )
        # self.optimizer = torch.optim.AdamW(
        #     self.model.parameters(),
        #     lr=config.learning_rate,
        #     weight_decay=config.weight_decay
        # )
        self.optimizer = torch.optim.AdamW(param_groups)

        # 设置学习率调度器
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=config.warmup_steps
        )

        # 多阶段阶梯递减（step decay），如果你想在某些step后直接降低学习率
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #     self.optimizer,
        #     milestones=[10000, 20000],  # 在这些step时降低学习率
        #     gamma=0.1                   # 每次降低为原来的0.1倍
        # )

        # self.scheduler = torch.optim.lr_scheduler.LambdaLR(
        #     self.optimizer,
        #     lr_lambda=lambda step: 1.0 if step < 10000 else 0.1
        # )
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     self.optimizer,
        #     T_max=config.max_steps,   # 一个周期的步数，通常设为最大训练步数
        #     eta_min=1e-6              # 最低学习率，可根据需要调整
        # )
        #多阶段
        # self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        #     self.optimizer,
        #     start_factor=0.1,
        #     end_factor=1.0,
        #     total_iters=config.warmup_steps
        # )
        # self.decay_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #     self.optimizer,
        #     milestones=[10000,40000],  # 递减点
        #     gamma=0.1
        # )
        # self.scheduler = torch.optim.lr_scheduler.SequentialLR(
        #     self.optimizer,
        #     schedulers=[self.warmup_scheduler, self.decay_scheduler],
        #     milestones=[config.warmup_steps]
        # )

        # 损失函数
        # self.criterion = torch.nn.MSELoss()
        
        # 检查点保存目录
        self.save_dir = Path(config.save_folder)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        

        log.info(f"SimpleDiTActionTrainer initialized on device {device}")
        log.info(f"Training dtype: {self.train_dtype}")
        log.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def save_checkpoint(self, step: int) -> str:
        """保存检查点"""
        checkpoint_dir = self.save_dir / f"step_{step}"
        
        # 检查是否已存在且不允许覆盖
        if checkpoint_dir.exists() and not self.config.save_overwrite:
            raise FileExistsError(f"Checkpoint directory {checkpoint_dir} already exists. Use save_overwrite=True to overwrite.")
        
        # 保存模型状态
        if get_global_rank() == 0:
            # 创建检查点目录
            checkpoint_dir.mkdir(exist_ok=True)
            
            # 将新检查点添加到列表
            self.checkpoints.append(checkpoint_dir)

            if not self.config.use_fsdp:
                checkpoint_path = checkpoint_dir / "model.pt"
                # 如果需要微调vision model，则保存其状态
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'vision_model_state_dict': self.vision_model.state_dict() if self.config.ft_vit else None,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'global_step': self.global_step,
                    'epoch': self.epoch, # 这个值没有更新
                    'config': self.config.__dict__,
                }, checkpoint_path)

            else:
                # haven't debugged this part yet
                checkpoint_path = checkpoint_dir / "fsdp_model.pt"
                checkpoint = {
                    "model": self.fsdp_model.full_state_dict(),
                    "optimizer": self.fsdp_model.sharded_state_dict(),
                    'vision_model_state_dict': self.vision_model.state_dict() if self.config.ft_vit else None,
                    'scheduler': self.scheduler.state_dict(),
                    'global_step': self.global_step,
                    'config': self.config.__dict__,
                }
                torch.save(checkpoint, checkpoint_path)
            
            # 保存配置为YAML文件
            config_yaml_path = checkpoint_dir / "config.yaml"
            with open(config_yaml_path, 'w') as f:
                # 将配置对象转换为字典，然后序列化为YAML
                yaml.dump(self.config.__dict__, f, default_flow_style=False)
            
            log.info(f"Config saved to {config_yaml_path}")

            # 创建latest链接
            latest_path = self.save_dir / "latest"

            # 先删除旧的链接（如果存在）
            if latest_path.exists() or latest_path.is_symlink():
                try:
                    latest_path.unlink()
                except OSError as e:
                    log.warning(f"Failed to remove old latest link: {e}")
            
            # 创建新的符号链接
            try:
                latest_path.symlink_to(checkpoint_dir.name, target_is_directory=True)
                log.info(f"Created latest link: {latest_path} -> {checkpoint_dir.name}")
            except FileExistsError:
                # 如果链接已经存在，检查是否指向正确的目录
                try:
                    if latest_path.resolve().name == checkpoint_dir.name:
                        log.info(f"Latest link already points to correct checkpoint: {checkpoint_dir.name}")
                    else:
                        # 强制删除并重新创建
                        latest_path.unlink()
                        latest_path.symlink_to(checkpoint_dir.name, target_is_directory=True)
                        log.info(f"Updated latest link to: {checkpoint_dir.name}")
                except Exception as e:
                    log.warning(f"Failed to handle latest link: {e}")
            except Exception as e:
                log.warning(f"Failed to create latest link: {e}")
            
            log.info(f"Checkpoint saved to {checkpoint_path}")
        
        # 同步所有进程
        barrier()
        
        # 删除旧检查点
        self._remove_old_checkpoints()
        
        return str(checkpoint_dir)
    
    def _remove_old_checkpoints(self) -> None:
        """删除旧的检查点以保持指定的数量"""
        if self.config.save_num_checkpoints_to_keep <= 0:
            return  # -1 或 0 表示保留所有检查点
        
        while len(self.checkpoints) > self.config.save_num_checkpoints_to_keep:
            self._remove_checkpoint(0)  # 删除最旧的检查点
    
    def _remove_checkpoint(self, idx: int = 0) -> None:
        """删除指定索引的检查点"""
        if idx >= len(self.checkpoints):
            return
        
        oldest_checkpoint = self.checkpoints.pop(idx)
        
        # 同步所有进程
        # barrier()
        
        # 只有全局rank 0删除文件
        if get_global_rank() == 0 and oldest_checkpoint.is_dir():
            try:
                shutil.rmtree(oldest_checkpoint, ignore_errors=True)
                log.info(f"Removed old checkpoint: {oldest_checkpoint}")
                
                # 检查是否需要更新latest链接
                latest_path = self.save_dir / "latest"
                if latest_path.exists() and latest_path.resolve() == oldest_checkpoint.resolve():
                    latest_path.unlink()
                    log.info(f"Removed outdated latest link")
                    
            except Exception as e:
                log.warning(f"Failed to remove checkpoint {oldest_checkpoint}: {e}")
        
        # 再次同步
        # barrier()
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """加载检查点"""
        if not self.config.use_fsdp:
            if Path(checkpoint_path).is_dir():
                checkpoint_path = Path(checkpoint_path) / "model.pt"
            
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.global_step = checkpoint['global_step']
            self.epoch = checkpoint['epoch']

            if self.config.ft_vit:
                self.vision_model.load_state_dict(checkpoint['vision_model_state_dict'])
        else:
            if Path(checkpoint_path).is_dir():
                checkpoint_path = Path(checkpoint_path) / "fsdp_model.pt"
            # 读取检查点（rank0加载后广播）
            if dist.get_rank() == 0:
                checkpoint = torch.load(checkpoint_path)
            else:
                checkpoint = {}
            dist.broadcast_object_list([checkpoint], src=0)  # 广播到所有rank

            # 恢复模型和优化器状态
            self.model.load_state_dict(checkpoint["model"])
            optim_state = checkpoint["optimizer"]
            optim_state = FSDP.optim_state_dict_to_load(optim_state, self.model, self.optimizer)
            self.optimizer.load_state_dict(optim_state)

            # 恢复调度器状态（所有rank同步执行）
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            self.epoch  = checkpoint["epoch"]  # 手动设置epoch
        
            if self.config.ft_vit:
                self.vision_model.load_state_dict(checkpoint['vision_model_state_dict'])
        
        # 重建检查点列表（从已有的检查点目录扫描）
        self._rebuild_checkpoint_list()
        
        log.info(f"Checkpoint loaded from {checkpoint_path}")
        log.info(f"Resumed from step {self.global_step}, epoch {self.epoch}")
    
    def _rebuild_checkpoint_list(self) -> None:
        """从保存目录重建检查点列表"""
        self.checkpoints = []
        if not self.save_dir.exists():
            return
        
        # 扫描所有step_*目录
        for checkpoint_dir in self.save_dir.iterdir():
            if checkpoint_dir.is_dir() and checkpoint_dir.name.startswith("step_"):
                try:
                    step_num = int(checkpoint_dir.name.split("_")[1])
                    self.checkpoints.append(checkpoint_dir)
                except (ValueError, IndexError):
                    continue
        
        # 按步数排序
        self.checkpoints.sort(key=lambda x: int(x.name.split("_")[1]))
        
        log.info(f"Found {len(self.checkpoints)} existing checkpoints")
    
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """单步训练"""
        self.model.train()
        if self.config.ft_vit:
            self.vision_model.train()
        self.optimizer.zero_grad()
        
        # 移动数据到设备
        batch = move_to_device(batch, self.device)
        self.text_model = self.text_model.to(self.device)
        self.vision_model = self.vision_model.to(self.device)
        
        # 前向传播
        input_ids = batch["input_ids"].to(self.device)
        pixel_values = batch["pixel_values"].to(self.device, dtype=self.train_dtype)
        text_attention_mask = batch.get("text_attention_mask")
        if text_attention_mask is not None:
            text_attention_mask = text_attention_mask.to(self.device, ) #dtype=self.train_dtype

        # 准备目标动作，确保数据类型正确
        target_actions = batch["action"].to(self.device, dtype=self.train_dtype)
        proprio = batch.get("proprio")
        if proprio is not None:
            proprio = proprio.to(self.device, dtype=self.train_dtype)

        bs, seq_len = input_ids.shape
        pv_shape = pixel_values.shape
        C,H,W = pv_shape[-3],pv_shape[-2],pv_shape[-1]
        pixel_values = pixel_values.reshape(-1,C,H,W)

        # print(f"Start model forward...")
        # s_time = time.time()
        if self.config.text_model == 'SigLip':
            text_embeds = self.text_model(input_ids=batch["input_ids"],attention_mask=batch.get("text_attention_mask")).last_hidden_state.detach()  
        else:
            outputs = self.text_model(input_ids=batch["input_ids"],attention_mask=batch["text_attention_mask"],    
                                        return_dict=True,          # 返回 ModelOutput 对象
                                        output_hidden_states=True  # 要所有层的 hidden_states)
            )
            hidden_states = outputs.hidden_states          # Tuple[embedding, layer1, ..., layerN]
            last_hidden_state = hidden_states[-1]  # 获取最后一层的 hidden state
            text_embeds = last_hidden_state.detach()

        image_embeds = self.vision_model(pixel_values=batch["pixel_values"]).last_hidden_state
        if not self.config.ft_vit:
            image_embeds = image_embeds.detach()
        image_embeds = image_embeds.reshape(bs,-1,self.vision_model.config.hidden_size,) #


        if self.config.precision == "amp_bf16":
            with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                outputs = self.model(
                    # input_ids=batch["input_ids"],
                    # pixel_values=batch["pixel_values"],
                    text_embeds=text_embeds,
                    image_embeds=image_embeds,
                    target_actions=target_actions,
                    proprio=proprio,
                    text_attn_mask=text_attention_mask.to(torch.bool) if text_attention_mask is not None else None,
                    # text_attn_mask = None, # bug set to None
                )
        else:
            outputs = self.model(
                    text_embeds=text_embeds,
                    image_embeds=image_embeds,
                    target_actions=target_actions,
                    proprio=proprio,
                )
        # e_time = time.time()
        # print(f"Model forward done. Time taken: {e_time - s_time:.2f} seconds")
        # 计算损失
        diff_pred = outputs['diffusion_pred']
        diff_target = outputs['diffusion_target']
        action_loss = F.mse_loss(diff_pred, diff_target, reduction="mean")
        # loss = outputs["loss"]
        
        # 反向传播
        action_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            self.config.max_grad_norm
        )
        
        # 优化器步骤
        self.optimizer.step()
        self.scheduler.step()
        
        # 收集指标
        metrics = {
            "train/loss": action_loss.item(),
            "train/lr": self.scheduler.get_last_lr()[0],
            "train/step": self.global_step,
            "train/num_checkpoints": len(self.checkpoints),
        }
        
        return metrics
    
    def fit(self) -> None:
        """训练主循环"""
        # log.info("Starting training...")
        if get_global_rank() == 0:
            print("Starting training...")
        
        
        # if cfg.wandb is not None and (get_global_rank() == 0 or not cfg.wandb.rank_zero_only):
        #     wandb_dir = Path(cfg.save_folder) / "wandb"
        #     wandb_dir.mkdir(parents=True, exist_ok=True)
        #     wandb.init(
        #         dir=str(wandb_dir),
        #         project=cfg.wandb.project,
        #         entity=cfg.wandb.entity,
        #         group=cfg.wandb.group,
        #         name=cfg.wandb.name,
        #         tags=cfg.wandb.tags,
        #         config=cfg.asdict(exclude=["wandb"]),
        #     )

        # 初始化W&B
        wandb_dir = Path(self.config.save_folder) / "wandb"
        wandb_dir.mkdir(parents=True, exist_ok=True)
        if self.config.wandb_project and get_global_rank() == 0:
            wandb.init(
                dir=str(wandb_dir),
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                name=self.config.run_name,
                config=self.config.__dict__
            )
        
        config_yaml_path = Path(self.config.save_folder) / "config.yaml"
        with open(config_yaml_path, 'w') as f:
            # 将配置对象转换为字典，然后序列化为YAML
            yaml.dump(self.config.__dict__, f, default_flow_style=False)
        
        log.info(f"Config saved to {config_yaml_path}")

        start_time = time.time()
        running_loss = 0.0
        
        while self.global_step < self.config.max_steps:
            # print(f"Training step {self.global_step}/{self.config.max_steps}")
            for batch in self.train_dataloader:
                if self.global_step >= self.config.max_steps:
                    break
                
                # 训练步骤
                # print(f"Start train step...")
                # s_time = time.time()
                metrics = self.train_step(batch)
                # e_time = time.time()
                # print(f"Train step done. Time taken: {e_time - s_time:.2f} seconds")

                running_loss += metrics["train/loss"]
                
                self.global_step += 1
                
                # 记录指标
                if self.global_step % self.config.log_interval == 0:
                    avg_loss = running_loss / self.config.log_interval
                    running_loss = 0.0
                    
                    elapsed_time = time.time() - start_time
                    steps_per_sec = self.global_step / elapsed_time
                    
                    metrics.update({
                        "train/avg_loss": avg_loss,
                        "train/steps_per_sec": steps_per_sec,
                        "system/peak_memory_mb": peak_gpu_memory() or 0,
                    })
                    
                    if get_global_rank() == 0:
                        # log.info(
                        print(
                            f"Step {self.global_step}/{self.config.max_steps} | "
                            f"Loss: {avg_loss:.6f} | "
                            f"LR: {metrics['train/lr']:.2e} | "
                            f"Steps/sec: {steps_per_sec:.2f} | "
                            f"Checkpoints: {metrics['train/num_checkpoints']}"
                        )
                        
                        if self.config.wandb_project:
                            wandb.log(metrics, step=self.global_step)
                
                # 保存检查点
                if self.global_step % self.config.save_interval == 0:
                    try:
                        checkpoint_path = self.save_checkpoint(self.global_step)
                        if get_global_rank() == 0:
                            # log.info(f"Checkpoint saved to {checkpoint_path}")
                            print(f"Checkpoint saved to {checkpoint_path}")
                    except FileExistsError as e:
                        # log.warning(f"Checkpoint save failed: {e}")
                        print(f"Checkpoint save failed: {e}")
                    
                    # 清理GPU内存
                    gc.collect()
                    torch.cuda.empty_cache()
        
        # 保存最终检查点
        if get_global_rank() == 0:
            final_checkpoint = self.save_checkpoint(self.global_step)
            log.info(f"Final checkpoint saved to {final_checkpoint}")
            log.info("Training completed!")
            
            if self.config.wandb_project:
                wandb.finish()
    
    def cleanup_all_checkpoints(self) -> None:
        """清理所有检查点（用于调试或完全重置）"""
        if get_global_rank() == 0:
            for checkpoint_dir in self.checkpoints.copy():
                if checkpoint_dir.is_dir():
                    shutil.rmtree(checkpoint_dir, ignore_errors=True)
                    log.info(f"Removed checkpoint: {checkpoint_dir}")
            
            # 清理latest链接
            latest_path = self.save_dir / "latest"
            if latest_path.exists():
                latest_path.unlink()
                log.info("Removed latest link")
        
        self.checkpoints.clear()
        barrier()