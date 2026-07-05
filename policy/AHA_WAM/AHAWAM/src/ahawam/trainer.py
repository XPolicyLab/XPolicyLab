import gc
import json

import os
import re
from collections.abc import Iterable
from math import ceil
from pathlib import Path
import time

import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from omegaconf import DictConfig
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)
from torch.utils.data import DataLoader

from .utils.fs import ensure_dir
from .utils.logging_config import get_logger
from .utils.pytorch_utils import set_global_seed
from .utils.samplers import HistoryAwareResumableEpochSampler, ResumableEpochSampler

logger = get_logger(__name__)


class Wan22Trainer:
    def __init__(self, model, train_dataset, val_dataset=None, *, cfg: DictConfig):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = cfg
        self.output_dir = str(cfg.output_dir)
        self.learning_rate = float(cfg.learning_rate)
        self.weight_decay = float(cfg.weight_decay)
        self.batch_size = int(cfg.batch_size)
        self.num_workers = int(cfg.num_workers)
        self.num_epochs = int(cfg.num_epochs)
        max_steps = cfg.max_steps
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.log_every = int(cfg.log_every)
        self.save_every = int(cfg.save_every)
        self.eval_every = int(cfg.eval_every)
        self.eval_num_inference_steps = int(cfg.eval_num_inference_steps)
        self.gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
        self.max_grad_norm = float(cfg.max_grad_norm)
        self.seed = int(cfg.seed)
        self.history_aware_batching = bool(cfg.get("history_aware_batching", False))
        self.freeze_video_dit = bool(cfg.get("freeze_video_dit", False))

        self.resume = cfg.resume
        self.resume_global_batch_size = cfg.get("resume_global_batch_size", None)
        self.init_checkpoint = cfg.get("init_checkpoint", None)
        self.mixed_precision = str(cfg.mixed_precision).strip().lower()
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError(
                f"Unsupported mixed_precision: {cfg.mixed_precision}. "
                "Expected one of: ['no', 'fp16', 'bf16']."
            )
        self.wandb_enabled = bool(cfg.wandb.enabled)

        # DDP kwargs: applied only when accelerate uses MULTI_GPU (plain DDP).
        # Ignored by DeepSpeed configs which manage their own communication.
        ddp_kwargs = DistributedDataParallelKwargs(
            bucket_cap_mb=25,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            static_graph=True,
        )

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            step_scheduler_with_optimizer=False,
            kwargs_handlers=[ddp_kwargs],
        )

        ds_plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
        if ds_plugin is not None:
            zero_stage = ds_plugin.deepspeed_config.get(
                "zero_optimization", {}
            ).get("stage", "unknown")
        else:
            zero_stage = "n/a"

        logger.info(
            "Accelerate training: distributed_type=%s zero_stage=%s world_size=%d process_index=%d cfg_mixed_precision=%s accelerator_mixed_precision=%s grad_accum=%d grad_clip=%.4f",
            self.accelerator.distributed_type,
            zero_stage,
            self.accelerator.num_processes,
            self.accelerator.process_index,
            self.mixed_precision,
            self.accelerator.mixed_precision,
            self.gradient_accumulation_steps,
            self.max_grad_norm,
        )
        logger.info("using accelerator.device=%s", self.accelerator.device)
        worker_init_fn = set_global_seed(self.seed, get_worker_init_fn=True)
        self._assert_dataset_length_consistent(self.train_dataset, "train_dataset")
        if self.val_dataset is not None:
            self._assert_dataset_length_consistent(self.val_dataset, "val_dataset")

        if self.resume and self.init_checkpoint:
            raise ValueError(
                "`resume` and `init_checkpoint` are mutually exclusive; choose one."
            )

        # Freeze non-trainable modules before optimizer/deepspeed initialization.
        # This keeps DiT (+ optional additional trainable modules) trainable when ZeRO builds optimizer state.
        self._apply_dit_only_train_mode(
            self.model, freeze_video_dit=self.freeze_video_dit
        )

        # Load model weights BEFORE optimizer/accelerator.prepare() so that
        # DeepSpeed fp32 master copies are created from the correct weights.
        # Without this, ZeRO-1+bf16 creates fp32 masters from random/pretrained
        # weights, then the first optimizer.step() overwrites any post-prepare loading.
        self._load_weights_before_prepare()

        trainable_params = [p for p in self.model.dit.parameters() if p.requires_grad]
        extra_trainable_modules: list[nn.Module] = []
        get_extra_modules = getattr(
            self.model, "get_additional_trainable_modules", None
        )
        if callable(get_extra_modules):
            returned_modules = get_extra_modules()
            if isinstance(returned_modules, dict):
                extra_trainable_modules = list(returned_modules.values())
            elif isinstance(returned_modules, Iterable):
                extra_trainable_modules = list(returned_modules)
            elif returned_modules is not None:
                extra_trainable_modules = [returned_modules]
        else:
            proprio_encoder = getattr(self.model, "proprio_encoder", None)
            if proprio_encoder is not None:
                extra_trainable_modules.append(proprio_encoder)
        for module in extra_trainable_modules:
            trainable_params.extend(p for p in module.parameters() if p.requires_grad)
        if not trainable_params:
            raise ValueError(
                "No trainable parameters remain after applying freeze settings. "
                "Disable `freeze_video_dit` or enable another trainable module."
            )
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )

        self.train_loader = self._build_loader(
            self.train_dataset, worker_init_fn=worker_init_fn
        )
        total_train_steps = self._estimate_total_train_steps()
        self.max_steps = total_train_steps
        warmup_ratio = float(getattr(cfg, "warmup_ratio", 0.05))
        warmup_steps = int(total_train_steps * warmup_ratio)
        self.scheduler = self._build_scheduler(
            scheduler_type=cfg.lr_scheduler_type,
            total_train_steps=total_train_steps,
            warmup_steps=warmup_steps,
        )
        logger.info(
            "LR scheduler: type=%s class=%s total_train_steps=%d warmup_ratio=%.6g "
            "warmup_steps=%d learning_rate=%.6g",
            cfg.lr_scheduler_type,
            type(self.scheduler).__name__,
            total_train_steps,
            warmup_ratio,
            warmup_steps,
            self.learning_rate,
        )
        self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0

        self.checkpoint_root = os.path.join(self.output_dir, "checkpoints")
        self.weights_dir = os.path.join(self.checkpoint_root, "weights")
        self.state_dir = os.path.join(self.checkpoint_root, "state")
        self.eval_dir = os.path.join(self.output_dir, "eval")

        ensure_dir(self.output_dir)
        ensure_dir(self.checkpoint_root)
        ensure_dir(self.weights_dir)
        ensure_dir(self.state_dir)
        ensure_dir(self.eval_dir)

        self.model, self.optimizer, self.train_loader, self.scheduler = (
            self.accelerator.prepare(
                self.model, self.optimizer, self.train_loader, self.scheduler
            )
        )
        self.optimizer.zero_grad(set_to_none=True)
        self.wandb_run = None
        self._init_wandb()
        self._resume_or_load_checkpoint()
        self._profile_window = self._reset_profile_window()

        val_size = (
            len(self.val_dataset)
            if self.val_dataset is not None
            else len(self.train_dataset)
        )
        logger.info("Train/val dataset size: %d/%d", len(self.train_dataset), val_size)

    @staticmethod
    def _reset_profile_window():
        return {
            "steps": 0,
            "data_time": 0.0,
            "forward_time": 0.0,
            "backward_time": 0.0,
            "optimizer_time": 0.0,
        }

    def _update_profile_window(
        self,
        *,
        data_time: float,
        forward_time: float,
        backward_time: float,
        optimizer_time: float,
    ):
        self._profile_window["steps"] += 1
        self._profile_window["data_time"] += float(data_time)
        self._profile_window["forward_time"] += float(forward_time)
        self._profile_window["backward_time"] += float(backward_time)
        self._profile_window["optimizer_time"] += float(optimizer_time)

    def _consume_profile_window(self):
        steps = max(int(self._profile_window["steps"]), 1)
        profile = {
            "data_time": self._profile_window["data_time"] / steps,
            "forward_time": self._profile_window["forward_time"] / steps,
            "backward_time": self._profile_window["backward_time"] / steps,
            "optimizer_time": self._profile_window["optimizer_time"] / steps,
        }
        self._profile_window = self._reset_profile_window()
        return profile

    def _init_wandb(self):
        if not self.wandb_enabled or not self.accelerator.is_main_process:
            return
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb logging is enabled in config (`wandb.enabled=true`) but wandb is not installed."
            ) from e

        self.wandb_run = wandb.init(
            entity=self.cfg.wandb.workspace,
            project=self.cfg.wandb.project,
            name=self.cfg.wandb.name,
            group=None
            if self.cfg.wandb.group in (None, "null", "")
            else str(self.cfg.wandb.group),
            mode=self.cfg.wandb.mode,
            dir=self.output_dir,
        )
        logger.info(
            "Initialized wandb run: workspace=%s project=%s name=%s",
            self.cfg.wandb.workspace,
            self.cfg.wandb.project,
            self.cfg.wandb.name,
        )

    def _wandb_log(self, payload: dict):
        if self.wandb_run is None:
            return
        self.wandb_run.log(payload, step=self.global_step)

    def _finish_wandb(self):
        if self.wandb_run is None:
            return
        self.wandb_run.finish()
        self.wandb_run = None

    def _build_loader(self, dataset, worker_init_fn=None):
        sampler_cls = ResumableEpochSampler
        if self.history_aware_batching:
            sampler_cls = HistoryAwareResumableEpochSampler
            logger.info("Using history-aware train sampler.")
        self.train_sampler = sampler_cls(
            dataset=dataset,
            seed=self.seed,
            batch_size=self.batch_size,
            num_processes=self.accelerator.num_processes,
        )
        loader_kwargs = {
            "dataset": dataset,
            "batch_size": self.batch_size,
            "shuffle": False,
            "sampler": self.train_sampler,
            "num_workers": self.num_workers,
            "pin_memory": torch.cuda.is_available(),
            "worker_init_fn": worker_init_fn,
        }
        if self.num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 2
        return DataLoader(**loader_kwargs)

    def _assert_dataset_length_consistent(self, dataset, bench_name: str):
        if not hasattr(dataset, "__len__"):
            raise TypeError(
                f"`{bench_name}` must implement __len__ for rank consistency checks."
            )

        local_length = len(dataset)
        gathered_lengths = self.accelerator.gather(
            torch.tensor(
                [local_length], device=self.accelerator.device, dtype=torch.int64
            )
        ).reshape(-1)
        if torch.all(gathered_lengths == gathered_lengths[0]):
            return

        if self.accelerator.is_main_process:
            print(
                f"[dataset-check] {bench_name} length mismatch across ranks after initialization:"
            )
            for rank, rank_length in enumerate(gathered_lengths.cpu().tolist()):
                print(f"rank {rank}: {rank_length}")
        self.accelerator.wait_for_everyone()
        raise RuntimeError(
            f"{bench_name} length mismatch across ranks: {gathered_lengths.cpu().tolist()}"
        )

    def _estimate_total_train_steps(self) -> int:
        if self.max_steps is not None:
            return max(int(self.max_steps), 1)

        if not hasattr(self.train_dataset, "__len__"):
            raise TypeError(
                "`train_dataset` must implement __len__ when `max_steps` is None."
            )

        num_processes = max(int(self.accelerator.num_processes), 1)
        global_batch_size = max(self.batch_size * num_processes, 1)
        micro_steps_per_epoch = max(
            ceil(len(self.train_dataset) / global_batch_size), 1
        )
        opt_steps_per_epoch = max(
            ceil(micro_steps_per_epoch / self.gradient_accumulation_steps),
            1,
        )
        return max(opt_steps_per_epoch * self.num_epochs, 1)

    def _build_scheduler(
        self, scheduler_type, total_train_steps: int, warmup_steps: int = 0
    ):
        scheduler_type = str(scheduler_type).strip().lower()
        total_train_steps = max(int(total_train_steps), 1)
        warmup_steps = min(max(int(warmup_steps), 0), total_train_steps - 1)

        remaining_steps = max(total_train_steps - warmup_steps, 1)
        if scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=remaining_steps,
                eta_min=self.learning_rate * 0.01,
            )
        elif scheduler_type == "constant":
            main_scheduler = ConstantLR(
                self.optimizer, factor=1.0, total_iters=remaining_steps
            )
        else:
            raise ValueError(
                f"Unsupported lr_scheduler_type: {scheduler_type}. "
                "Expected one of: ['cosine', 'constant']."
            )

        if warmup_steps <= 0:
            return main_scheduler

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0 / warmup_steps,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )

    def _estimate_eta(self):
        elapsed = max(time.perf_counter() - self.run_start_time, 1e-6)
        done_steps = max(self.global_step - self.run_start_step, 1)
        steps_per_sec = done_steps / elapsed
        max_steps = self.max_steps if self.max_steps is not None else self.global_step
        remaining_steps = max(max_steps - self.global_step, 0)
        eta_seconds = int(remaining_steps / max(steps_per_sec, 1e-9))
        eta_h, eta_rem = divmod(eta_seconds, 3600)
        eta_m, eta_s = divmod(eta_rem, 60)
        return f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}", steps_per_sec

    def _load_weights_before_prepare(self):
        """Load model weights BEFORE accelerator.prepare() to ensure DeepSpeed
        fp32 master copies are initialized from the correct checkpoint weights.

        Handles three cases:
        1. init_checkpoint (.pt) — fresh training start from pretrained weights
        2. resume (.pt) — weight-only resume, progress aligned after prepare
        3. resume (directory, DDP format) — cross-backend resume; extract weights
           here, skip incompatible optimizer state
        """
        self._resume_weights_preloaded = False
        self._resume_state_dir_is_ddp = False

        if self.init_checkpoint:
            init_path = Path(str(self.init_checkpoint))
            if init_path.is_dir():
                raise ValueError(
                    f"`init_checkpoint` must point to a weight file (.pt), "
                    f"got directory: {self.init_checkpoint}"
                )
            if not init_path.exists():
                raise FileNotFoundError(
                    f"Init checkpoint not found: {self.init_checkpoint}"
                )
            logger.info(
                "Loading initialization weights BEFORE prepare: %s",
                self.init_checkpoint,
            )
            self.model.load_checkpoint(str(init_path), optimizer=None)
            return

        if not self.resume:
            return

        resume_path = Path(str(self.resume))

        # Case: .pt file — load weights before prepare
        if not resume_path.is_dir():
            if not resume_path.exists():
                raise FileNotFoundError(
                    f"Resume checkpoint not found: {self.resume}"
                )
            logger.info(
                "Loading .pt weights BEFORE prepare (ZeRO-safe): %s",
                self.resume,
            )
            self.model.load_checkpoint(str(resume_path), optimizer=None)
            self._resume_weights_preloaded = True
            return

        # Case: state directory — check if DDP format (incompatible with DeepSpeed)
        if self._is_ddp_state_dir(resume_path):
            logger.warning(
                "State directory %s was saved under DDP but current config uses "
                "DeepSpeed. Loading model weights only (before prepare) and "
                "discarding incompatible optimizer state.",
                self.resume,
            )
            self._load_model_weights_from_ddp_state(resume_path)
            self._resume_weights_preloaded = True
            self._resume_state_dir_is_ddp = True

    @staticmethod
    def _is_ddp_state_dir(state_dir: Path) -> bool:
        """Detect whether a state directory was saved under plain DDP (not DeepSpeed).

        DDP state dirs contain model.safetensors or pytorch_model.bin at top level.
        DeepSpeed state dirs contain a global_step* subdirectory or
        zero_pp_rank* / mp_rank* files.
        """
        # DeepSpeed indicators
        for item in state_dir.iterdir():
            if item.is_dir() and item.name.startswith("global_step"):
                return False
            if item.name.startswith("zero_pp_rank") or item.name.startswith("mp_rank"):
                return False
        # DDP indicators
        ddp_markers = ["model.safetensors", "pytorch_model.bin", "pytorch_model_0.bin"]
        for marker in ddp_markers:
            if (state_dir / marker).exists():
                return True
        return False

    def _load_model_weights_from_ddp_state(self, state_dir: Path):
        """Extract and load model weights from a DDP-format state directory."""
        from safetensors.torch import load_file as safetensors_load

        model_path = state_dir / "model.safetensors"
        if model_path.exists():
            state_dict = safetensors_load(str(model_path), device="cpu")
        else:
            pt_path = state_dir / "pytorch_model.bin"
            if not pt_path.exists():
                pt_path = state_dir / "pytorch_model_0.bin"
            if not pt_path.exists():
                raise FileNotFoundError(
                    f"Cannot find model weights in DDP state dir: {state_dir}. "
                    f"Expected model.safetensors or pytorch_model.bin"
                )
            state_dict = torch.load(str(pt_path), map_location="cpu", weights_only=True)

        incompatible = self.model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            logger.warning(
                "DDP state load: %d missing keys (likely frozen modules): %s...",
                len(incompatible.missing_keys),
                incompatible.missing_keys[:5],
            )
        if incompatible.unexpected_keys:
            logger.warning(
                "DDP state load: %d unexpected keys: %s...",
                len(incompatible.unexpected_keys),
                incompatible.unexpected_keys[:5],
            )

    def _resume_or_load_checkpoint(self):
        """Post-prepare checkpoint resume. Handles:
        - State directory resume (DeepSpeed-compatible format)
        - Progress alignment for weights pre-loaded before prepare
        """
        if self.init_checkpoint:
            # init_checkpoint = fresh start, no progress to restore
            return

        resume = self.resume
        if not resume:
            return

        resume_path = Path(str(resume))

        # State directory resume
        if resume_path.is_dir():
            if self._resume_state_dir_is_ddp:
                # Weights already loaded before prepare; just align progress
                logger.info(
                    "DDP→DeepSpeed cross-backend resume: weights loaded before "
                    "prepare, aligning progress from trainer_state.json."
                )
                self._align_progress_from_state_dir(resume_path)
                return
            # Normal DeepSpeed-compatible state resume
            logger.info("Resuming full training state from directory: %s", resume)
            self.load_training_state(str(resume_path))
            return

        # .pt weight-only resume — weights already loaded before prepare
        if self._resume_weights_preloaded:
            parsed_step = self._parse_step_from_checkpoint_name(resume_path)
            if parsed_step is None:
                logger.warning(
                    "Loaded .pt weights only; optimizer/scheduler/step were not "
                    "restored because no `step_XXXXXX` tag was found in name."
                )
                return
            self._align_weight_only_resume_progress(parsed_step)
            logger.info(
                "Loaded .pt weights (before prepare) and aligned progress: "
                "step=%d epoch=%d batch_in_epoch=%d lr=%.6e. "
                "Optimizer state was not restored.",
                self.global_step,
                self.epoch,
                self.batch_in_epoch,
                self.optimizer.param_groups[0]["lr"],
            )

    def _align_progress_from_state_dir(self, state_dir: Path):
        """Restore step/epoch/batch progress from trainer_state.json without
        loading optimizer state (used for cross-backend resume)."""
        state_file = state_dir / "trainer_state.json"
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            step = int(payload["global_step"])
            self._align_weight_only_resume_progress(step)
            logger.info(
                "Aligned progress from state dir: step=%d epoch=%d batch_in_epoch=%d",
                self.global_step,
                self.epoch,
                self.batch_in_epoch,
            )
        else:
            match = re.search(r"step[_-](\d+)$", str(state_dir).rstrip("/"))
            if match:
                self._align_weight_only_resume_progress(int(match.group(1)))
            else:
                logger.warning(
                    "No trainer_state.json found in %s and no step tag in path; "
                    "starting from step 0.",
                    state_dir,
                )

    @staticmethod
    def _parse_step_from_checkpoint_name(path: Path) -> int | None:
        match = re.search(r"(?:^|[^A-Za-z0-9])step[_-](\d+)(?:\D|$)", path.stem)
        if match is None:
            return None
        return int(match.group(1))

    def _micro_steps_per_epoch(self) -> int:
        if not hasattr(self.train_dataset, "__len__"):
            raise TypeError(
                "`train_dataset` must implement __len__ to align a weight-only resume."
            )
        num_processes = max(int(self.accelerator.num_processes), 1)
        global_batch_size = max(self.batch_size * num_processes, 1)
        return max(ceil(len(self.train_dataset) / global_batch_size), 1)

    def _align_weight_only_resume_progress(self, step: int):
        step = max(int(step), 0)
        current_num_procs = int(self.accelerator.num_processes)
        current_micro_bs = self.batch_size * current_num_procs
        current_global_bs = current_micro_bs * self.gradient_accumulation_steps

        resume_global_bs = self.resume_global_batch_size
        if resume_global_bs is not None:
            resume_global_bs = int(resume_global_bs)
        else:
            resume_global_bs = current_global_bs

        consumed_samples = step * resume_global_bs
        consumed_micro_batches = consumed_samples // current_micro_bs

        max_consumed_micro_batches = int(self.max_steps) * self.gradient_accumulation_steps
        consumed_micro_batches = min(consumed_micro_batches, max_consumed_micro_batches)
        self.global_step = min(
            consumed_micro_batches // self.gradient_accumulation_steps,
            int(self.max_steps),
        )
        remainder_micro_batches = consumed_micro_batches % self.gradient_accumulation_steps
        micro_steps_per_epoch = self._micro_steps_per_epoch()
        self.epoch = consumed_micro_batches // micro_steps_per_epoch
        self.batch_in_epoch = consumed_micro_batches % micro_steps_per_epoch
        self.train_sampler.set_epoch_offset(self.epoch)
        self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)
        self._set_scheduler_to_step(self.global_step)

        logger.info(
            "Weight-only resume progress: parsed_step=%d clamped_step=%d "
            "micro_steps_per_epoch=%d resume_global_bs=%d current_global_bs=%d "
            "current_num_procs=%d grad_accum=%d",
            step,
            self.global_step,
            micro_steps_per_epoch,
            resume_global_bs,
            current_global_bs,
            current_num_procs,
            self.gradient_accumulation_steps,
        )
        if remainder_micro_batches != 0:
            logger.warning(
                "Weight-only resume landed inside an accumulation window: "
                "consumed_micro_batches=%d grad_accum=%d remainder=%d. "
                "Dataloader offset is exact, but partial accumulated gradients "
                "cannot be reconstructed from a weights-only checkpoint.",
                consumed_micro_batches,
                self.gradient_accumulation_steps,
                remainder_micro_batches,
            )

    def _set_scheduler_to_step(self, step: int):
        step = max(int(step), 0)
        raw_scheduler = getattr(self.scheduler, "scheduler", self.scheduler)
        for _ in range(step):
            raw_scheduler.step()
        lr = float(self.optimizer.param_groups[0]["lr"])
        logger.info(
            "Advanced LR scheduler to step=%d; lr=%.6g", step, lr
        )

    def _set_dit_only_train_mode(self):
        # Match DiffSynth's freeze_except("dit"): only DiT stays trainable/in-train-mode.
        if self.freeze_video_dit:
            logger.info(
                "Setting DiT to train mode, freezing video DiT and other model components."
            )
        else:
            logger.info("Setting DiT to train mode and freezing other model components.")
        model = self.accelerator.unwrap_model(self.model)
        self._apply_dit_only_train_mode(
            model, freeze_video_dit=self.freeze_video_dit
        )

    @staticmethod
    def _apply_dit_only_train_mode(model, *, freeze_video_dit: bool = False):
        model.eval()
        model.requires_grad_(False)
        model.dit.train()
        model.dit.requires_grad_(True)
        if freeze_video_dit:
            video_dit = Wan22Trainer._get_video_dit(model)
            if video_dit is None:
                raise ValueError(
                    "`freeze_video_dit=true` requires a model with a video DiT expert."
                )
            video_dit.eval()
            video_dit.requires_grad_(False)
        extra_trainable_modules: list[nn.Module] = []
        get_extra_modules = getattr(model, "get_additional_trainable_modules", None)
        if callable(get_extra_modules):
            returned_modules = get_extra_modules()
            if isinstance(returned_modules, dict):
                extra_trainable_modules = list(returned_modules.values())
            elif isinstance(returned_modules, Iterable):
                extra_trainable_modules = list(returned_modules)
            elif returned_modules is not None:
                extra_trainable_modules = [returned_modules]
        else:
            proprio_encoder = getattr(model, "proprio_encoder", None)
            if proprio_encoder is not None:
                extra_trainable_modules.append(proprio_encoder)
        for module in extra_trainable_modules:
            module.train()
            module.requires_grad_(True)

    @staticmethod
    def _get_video_dit(model):
        dit = getattr(model, "dit", None)
        mixtures = getattr(dit, "mixtures", None)
        if mixtures is not None and "video" in mixtures:
            return mixtures["video"]
        return getattr(model, "video_expert", dit)

    @staticmethod
    def _to_batched_eval_sample(sample):
        video = sample["video"]
        prompt = sample["prompt"]
        action = sample.get("action", None)
        proprio = sample.get("proprio", None)
        context = sample.get("context", None)
        context_mask = sample.get("context_mask", None)

        if not isinstance(video, torch.Tensor):
            raise TypeError(
                f"Expected tensor video for evaluation, got {type(video)}. "
                "Evaluation now expects `video` with shape [3,T,H,W] or [B,3,T,H,W]."
            )
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if video.ndim != 5:
            raise ValueError(
                f"Expected video shape [3,T,H,W] or [B,3,T,H,W], got {tuple(video.shape)}"
            )
        num_video_frames = video.shape[2]
        if num_video_frames <= 1:
            raise ValueError(
                f"`sample['video']` must have at least 2 frames for action evaluation, got {num_video_frames}"
            )

        if isinstance(prompt, str):
            prompt = [prompt]
        elif isinstance(prompt, tuple):
            prompt = list(prompt)
        elif not isinstance(prompt, list):
            raise TypeError(f"Expected prompt type str/list[str], got {type(prompt)}")
        if len(prompt) != video.shape[0]:
            raise ValueError(
                f"Prompt batch mismatch: len(prompt)={len(prompt)} vs video batch={video.shape[0]}"
            )

        action_horizon = None
        action = None
        if "action" in sample:
            action = sample["action"]
            if not isinstance(action, torch.Tensor):
                raise TypeError(
                    f"`sample['action']` must be a torch.Tensor, got {type(action)}"
                )
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3:
                raise ValueError(
                    f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}"
                )
            if action.shape[1] % (num_video_frames - 1) != 0:
                raise ValueError(
                    f"`sample['action']` temporal dimension must be divisible by video frames-1={num_video_frames - 1}, got {action.shape[1]}"
                )
            action_horizon = int(action.shape[1])

        proprio = None
        if "proprio" in sample:
            proprio = sample["proprio"]
            if not isinstance(proprio, torch.Tensor):
                raise TypeError(
                    f"`sample['proprio']` must be a torch.Tensor, got {type(proprio)}"
                )
            if proprio.ndim == 2:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 3:
                raise ValueError(
                    f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}"
                )

        if context is not None or context_mask is not None:
            if context is None or context_mask is None:
                raise ValueError(
                    "`context` and `context_mask` must both exist in eval sample."
                )
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )

        optional_tensors = {}
        for key in ("action_offset", "chunk_obs_images", "chunk_obs_images_no_offset"):
            if key not in sample or sample[key] is None:
                continue
            value = sample[key]
            if key == "action_offset":
                value = torch.as_tensor(value, dtype=torch.long)
                if value.ndim == 0:
                    value = value.unsqueeze(0)
                if value.ndim != 1 or int(value.shape[0]) != int(video.shape[0]):
                    raise ValueError(
                        "`action_offset` must be scalar or [B], "
                        f"got shape {tuple(value.shape)} for batch={video.shape[0]}."
                    )
            else:
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"`sample['{key}']` must be a torch.Tensor, got {type(value)}")
                if value.ndim == 4:
                    value = value.unsqueeze(0)
                if value.ndim != 5 or int(value.shape[0]) != int(video.shape[0]):
                    raise ValueError(
                        f"`{key}` must be [N,3,H,W] or [B,N,3,H,W], "
                        f"got shape {tuple(value.shape)} for batch={video.shape[0]}."
                    )
            optional_tensors[key] = value

        video_history = None
        video_history_valid_len = None
        video_history_frame_indices = None
        video_current_frame_index = sample.get("video_current_frame_index")
        if "video_history" in sample:
            video_history = sample["video_history"]
            if not isinstance(video_history, torch.Tensor):
                raise TypeError(
                    "`sample['video_history']` must be a torch.Tensor, "
                    f"got {type(video_history)}"
                )
            if video_history.ndim == 4:
                video_history = video_history.unsqueeze(0)
            if video_history.ndim != 5:
                raise ValueError(
                    "`sample['video_history']` must be [B,C,N,H,W], "
                    f"got shape {tuple(video_history.shape)}"
                )
            if video_history.shape[0] != video.shape[0]:
                raise ValueError(
                    "`video_history` batch mismatch: "
                    f"{video_history.shape[0]} vs video batch {video.shape[0]}"
                )
            video_history_valid_len = sample.get("video_history_valid_len")
            if video_history_valid_len is None:
                raise ValueError(
                    "`sample['video_history_valid_len']` is required with `video_history`."
                )
            video_history_valid_len = torch.as_tensor(
                video_history_valid_len, dtype=torch.long
            )
            if video_history_valid_len.ndim == 0:
                video_history_valid_len = video_history_valid_len.unsqueeze(0)
            if (
                video_history_valid_len.ndim != 1
                or video_history_valid_len.shape[0] != video.shape[0]
            ):
                raise ValueError(
                    "`video_history_valid_len` must be [B], "
                    f"got shape {tuple(video_history_valid_len.shape)}"
                )
            video_history_frame_indices = sample.get("video_history_frame_indices")
            if video_history_frame_indices is None:
                raise ValueError(
                    "`sample['video_history_frame_indices']` is required with "
                    "`video_history`."
                )
            video_history_frame_indices = torch.as_tensor(
                video_history_frame_indices, dtype=torch.long
            )
            if video_history_frame_indices.ndim == 1:
                video_history_frame_indices = video_history_frame_indices.unsqueeze(0)
            if (
                video_history_frame_indices.ndim != 2
                or video_history_frame_indices.shape[0] != video.shape[0]
                or video_history_frame_indices.shape[1] != video_history.shape[2]
            ):
                raise ValueError(
                    "`video_history_frame_indices` must be [B,N], "
                    f"got shape {tuple(video_history_frame_indices.shape)}"
                )

        batched_sample = {
            "video": video,
            "prompt": prompt,
            "action": action,
            "proprio": proprio,
            "context": context,
            "context_mask": context_mask,
            "action_horizon": action_horizon,
            "video_history": video_history,
            "video_history_valid_len": video_history_valid_len,
            "video_history_frame_indices": video_history_frame_indices,
            "video_current_frame_index": video_current_frame_index,
        }
        batched_sample.update(optional_tensors)
        return batched_sample

    @torch.no_grad()
    def evaluate(self):
        if self.val_dataset is None:
            return None

        model = self.accelerator.unwrap_model(self.model)
        was_dit_training = model.dit.training
        model.eval()

        # eval_index = (self.global_step + self.accelerator.process_index) % len(self.val_dataset)
        rng = torch.Generator(device="cpu").manual_seed(
            self.global_step + self.accelerator.process_index
        )
        eval_index = torch.randint(0, len(self.val_dataset), (1,), generator=rng).item()
        sample = self._to_batched_eval_sample(self.val_dataset[eval_index])
        sample["_eval_dataset"] = self.val_dataset

        if not hasattr(model, "evaluate_validation"):
            raise AttributeError(
                f"{type(model).__name__} must implement `evaluate_validation(...)` for trainer eval."
            )
        with self.accelerator.autocast():
            metrics = model.evaluate_validation(
                sample,
                eval_num_inference_steps=self.eval_num_inference_steps,
                eval_dir=self.eval_dir,
                global_step=self.global_step,
                process_index=self.accelerator.process_index,
            )

        result: dict[str, float | str] = {}
        scalar_items = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        for key, value in scalar_items.items():
            metric_tensor = torch.tensor(
                float(value), device=self.accelerator.device, dtype=torch.float32
            ).reshape(1)
            result[key] = float(
                self.accelerator.gather_for_metrics(metric_tensor).mean().item()
            )
        for key, value in metrics.items():
            if key not in scalar_items:
                result[key] = value

        if was_dit_training:
            self._set_dit_only_train_mode()
        return result

    def _save_weights_checkpoint(self, step_tag: str):
        model = self.accelerator.unwrap_model(self.model)
        ckpt_path = os.path.join(self.weights_dir, f"{step_tag}.pt")
        model.save_checkpoint(ckpt_path, optimizer=None, step=self.global_step)
        return ckpt_path

    def _save_trainer_state(self, state_path: str):
        state_file = os.path.join(state_path, "trainer_state.json")
        payload = {
            "global_step": int(self.global_step),
            "epoch": int(self.epoch),
            "batch_in_epoch": int(self.batch_in_epoch),
        }
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    def save_checkpoint(self):
        step_tag = f"step_{self.global_step:06d}"

        self.accelerator.wait_for_everyone()
        ckpt_path = None
        if self.accelerator.is_main_process:
            ckpt_path = self._save_weights_checkpoint(step_tag=step_tag)
        self.accelerator.wait_for_everyone()

        state_path = os.path.join(self.state_dir, step_tag)
        ensure_dir(state_path)
        self.accelerator.save_state(output_dir=state_path)
        if self.accelerator.is_main_process:
            self._save_trainer_state(state_path)
        self.accelerator.wait_for_everyone()

        return {"weights_path": ckpt_path, "state_path": state_path}

    def load_training_state(self, state_dir: str):
        self.accelerator.load_state(input_dir=state_dir)
        state_file = Path(state_dir) / "trainer_state.json"
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.global_step = int(payload["global_step"])

            if "epoch" in payload and "batch_in_epoch" in payload:
                self.epoch = int(payload["epoch"])
                self.batch_in_epoch = int(payload["batch_in_epoch"])
                self.train_sampler.set_epoch_offset(self.epoch)
                self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)
                logger.info(
                    "Restored dataloader progress: epoch=%d batch_in_epoch=%d sample_offset=%d",
                    self.epoch,
                    self.batch_in_epoch,
                    self.batch_in_epoch
                    * self.batch_size
                    * self.accelerator.num_processes,
                )
            else:
                self.epoch = 0
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                logger.warning(
                    "State file does not contain `epoch`/`batch_in_epoch`; "
                    "optimizer/scheduler were restored, but dataloader progress resume is skipped."
                )
            self.accelerator.wait_for_everyone()
            return

        match = re.search(r"step[_-](\d+)$", str(state_dir).rstrip("/"))
        if match:
            self.global_step = int(match.group(1))
        else:
            self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.train_sampler.clear_resume_batch_offset()
        self.accelerator.wait_for_everyone()
        logger.info(
            "Loaded accelerate training state from %s at step=%d",
            state_dir,
            self.global_step,
        )
        logger.warning(
            "State file `%s` is missing; dataloader progress resume is skipped.",
            state_file,
        )

    def train(self):
        self._set_dit_only_train_mode()

        # Enable TF32 for float32 matmuls and optimal cudnn kernels.
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

        if self.max_steps is None:
            raise ValueError(
                "`max_steps` must be set before entering the while-step training loop."
            )

        logger.info("Starting training with max_steps=%d.", self.max_steps)

        # Disable automatic GC to prevent mid-backward collection stalls.
        # With gradient checkpointing, backward re-materializes activations causing
        # massive temporary allocations that trigger GC traversal of millions of objects.
        gc.disable()
        logger.info("Disabled automatic GC to avoid bwd stalls with gradient checkpointing.")

        data_iter = iter(self.train_loader)
        self.run_start_step = self.global_step
        self.run_start_time = time.perf_counter()
        last_step_end_time = self.run_start_time

        while self.global_step < self.max_steps:
            try:
                sample = next(data_iter)
                data_ready_time = time.perf_counter()
                self.batch_in_epoch += 1
            except StopIteration:
                self.epoch += 1
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                data_iter = iter(self.train_loader)
                data_ready_time = time.perf_counter()
                last_step_end_time = data_ready_time
                continue

            with self.accelerator.accumulate(self.model):
                train_model = (
                    self.model
                    if hasattr(self.model, "training_loss")
                    else self.accelerator.unwrap_model(self.model)
                )
                data_time = data_ready_time - last_step_end_time
                forward_start_time = time.perf_counter()

                with self.accelerator.autocast():
                    loss, loss_dict = train_model.training_loss(sample)
                forward_end_time = time.perf_counter()
                self.accelerator.backward(loss)
                backward_end_time = time.perf_counter()

                if self.accelerator.sync_gradients:
                    grad_norm = self.accelerator.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self.optimizer.step()
                    optimizer_end_time = time.perf_counter()
                    if not self.accelerator.optimizer_step_was_skipped:
                        self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1

                    # Manual GC every step: promptly releases Python references to
                    # CUDA tensors, preventing allocator fragmentation/defrag stalls.
                    gc.collect()
                    current_lr = float(self.optimizer.param_groups[0]["lr"])
                    optimizer_time = optimizer_end_time - backward_end_time
                    should_log = (
                        self.log_every > 0 and self.global_step % self.log_every == 0
                    )
                else:
                    optimizer_time = 0.0
                    should_log = False

                self._update_profile_window(
                    data_time=data_time,
                    forward_time=forward_end_time - forward_start_time,
                    backward_time=backward_end_time - forward_end_time,
                    optimizer_time=optimizer_time,
                )
                last_step_end_time = time.perf_counter()

                if self.accelerator.sync_gradients:
                    if should_log:
                        global_loss = float(
                            self.accelerator.gather(loss.detach().float().reshape(1))
                            .mean()
                            .item()
                        )
                        global_loss_metrics = {}
                        for key, value in loss_dict.items():
                            metric_tensor = torch.tensor(
                                float(value), device=loss.device, dtype=torch.float32
                            ).reshape(1)
                            global_loss_metrics[key] = float(
                                self.accelerator.gather(metric_tensor).mean().item()
                            )
                        grad_norm_tensor = torch.as_tensor(
                            grad_norm, device=loss.device, dtype=torch.float32
                        )
                        global_grad_norm = float(
                            self.accelerator.gather(grad_norm_tensor).mean().item()
                        )
                        profile_metrics = self._consume_profile_window()
                        if self.accelerator.is_main_process:
                            eta_str, steps_per_sec = self._estimate_eta()
                            description = "[train] epoch=%d step=%d/%d loss=%.4f " % (
                                self.epoch,
                                self.global_step,
                                self.max_steps,
                                global_loss,
                            )
                            if global_loss_metrics:
                                detail_str = " ".join(
                                    [
                                        f"{k}={v:.4f}"
                                        for k, v in sorted(global_loss_metrics.items())
                                    ]
                                )
                                description += detail_str + " "
                            description += "grad_norm=%.4f " % global_grad_norm
                            description += (
                                "lr=%.2e speed=%.2f step/s, %.2f samples/s eta=%s"
                                % (
                                    current_lr,
                                    steps_per_sec,
                                    steps_per_sec
                                    * self.batch_size
                                    * self.accelerator.num_processes,
                                    eta_str,
                                )
                            )
                            description += (
                                " timing(data=%.3fs fwd=%.3fs bwd=%.3fs opt=%.3fs)"
                                % (
                                    profile_metrics["data_time"],
                                    profile_metrics["forward_time"],
                                    profile_metrics["backward_time"],
                                    profile_metrics["optimizer_time"],
                                )
                            )
                            lambda_state = getattr(self, "_lambda_state", None)
                            if lambda_state:
                                description += " anneal(" + " ".join(
                                    f"{k}={v:.3f}" for k, v in lambda_state.items()
                                ) + ")"
                            logger.info(description)

                            wandb_payload = {
                                "train/loss": global_loss,
                                "train/grad_norm": global_grad_norm,
                                "train/lr": current_lr,
                                "performance/steps_per_sec": steps_per_sec,
                                "performance/samples_per_sec": steps_per_sec
                                * self.batch_size
                                * self.accelerator.num_processes,
                                "performance/data_time_s": profile_metrics["data_time"],
                                "performance/forward_time_s": profile_metrics[
                                    "forward_time"
                                ],
                                "performance/backward_time_s": profile_metrics[
                                    "backward_time"
                                ],
                                "performance/optimizer_time_s": profile_metrics[
                                    "optimizer_time"
                                ],
                            }
                            for key, value in global_loss_metrics.items():
                                wandb_payload[f"train/{key}"] = value
                            lambda_state = getattr(self, "_lambda_state", None)
                            if lambda_state:
                                for k, v in lambda_state.items():
                                    wandb_payload[f"lambda/{k}"] = v
                            self._wandb_log(wandb_payload)

                    if (
                        self.eval_every > 0
                        and self.val_dataset is not None
                        and self.global_step % self.eval_every == 0
                    ):
                        metrics = self.evaluate()
                        self.accelerator.wait_for_everyone()
                        if metrics is not None and self.accelerator.is_main_process:
                            description = "[eval] step=%d val_loss=%.4f" % (
                                self.global_step,
                                metrics["val_loss"],
                            )
                            if "psnr_rd" in metrics and "ssim_rd" in metrics:
                                description += " infer_psnr=%.4f infer_ssim=%.4f" % (
                                    metrics["psnr_rd"],
                                    metrics["ssim_rd"],
                                )
                            if "action_l2" in metrics:
                                description += " action_l2=%.4f" % metrics["action_l2"]
                            if "action_l1" in metrics:
                                description += " action_l1=%.4f" % metrics["action_l1"]
                            if "history_len" in metrics:
                                description += " history_len=%.4f" % metrics[
                                    "history_len"
                                ]
                            for key in (
                                "val_loss_teacher_rollout_vs_gt",
                                "val_loss_action_1step",
                                "val_loss_action_2step",
                                "val_loss_action_4step",
                            ):
                                if key in metrics:
                                    description += f" {key}=%.4f" % metrics[key]
                            logger.info(description)
                            eval_payload = {
                                "eval/val_loss": float(metrics["val_loss"]),
                            }
                            for key, value in metrics.items():
                                if key == "val_loss":
                                    continue
                                if isinstance(value, (int, float)):
                                    eval_payload[f"eval/{key}"] = float(value)
                            self._wandb_log(eval_payload)

                    if self.save_every > 0 and self.global_step % self.save_every == 0:
                        ckpt_info = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[ckpt] step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )

                    if self.global_step >= self.max_steps:
                        ckpt_info = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[done] max_steps reached step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )
                        return

                    last_step_end_time = time.perf_counter()

        ckpt_info = self.save_checkpoint()
        if self.accelerator.is_main_process:
            logger.info(
                "[done] training finished step=%d weights=%s state=%s",
                self.global_step,
                ckpt_info["weights_path"],
                ckpt_info["state_path"],
            )
