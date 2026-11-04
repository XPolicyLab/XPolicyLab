import logging
import json
import inspect
import math
import os
import re
import gc
from math import ceil
from pathlib import Path
import time

import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import DictConfig
from PIL import Image
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LambdaLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from flexwam.utils.fs import ensure_dir
from flexwam.utils.logging_config import get_logger, setup_logging
from flexwam.utils.pytorch_utils import set_global_seed
from flexwam.utils.samplers import ResumableEpochSampler
from flexwam.utils.camera_metrics import configure_camera_metrics
from flexwam.utils.video_io import save_mp4
from flexwam.utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim
from flexwam.training.optimization import TrainingStep, configure_deepspeed_clipping

logger = get_logger(__name__)


class FlexWAMTrainer:
    def __init__(self, model, train_dataset, val_dataset=None, *, cfg: DictConfig):
        self.model = model
        if "data" in cfg and "train" in cfg.data:
            configure_camera_metrics(model, cfg.data.train)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = cfg
        self.output_dir = str(cfg.output_dir)
        self.learning_rate = float(cfg.learning_rate)
        self.critic_lr_cfg = self._parse_critic_lr(cfg)
        self.weight_decay = float(cfg.weight_decay)
        self.batch_size = int(cfg.batch_size)
        self.num_workers = int(cfg.num_workers)
        self.prefetch_factor = int(cfg.get("prefetch_factor", 1))
        if self.num_workers < 0:
            raise ValueError(f"`num_workers` must be non-negative, got {self.num_workers}.")
        if self.prefetch_factor < 1:
            raise ValueError(f"`prefetch_factor` must be >= 1, got {self.prefetch_factor}.")
        self.num_epochs = int(cfg.num_epochs)
        self.training_stages = self._parse_training_stages(cfg)
        max_steps = cfg.max_steps
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.log_every = int(cfg.log_every)
        self.save_every = int(cfg.save_every)
        self.save_weights = bool(cfg.get("save_weights", True))
        self.save_training_state = bool(cfg.get("save_training_state", True))
        self.eval_every = int(cfg.eval_every)
        self.eval_num_inference_steps = int(cfg.eval_num_inference_steps)
        self.gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
        self.max_grad_norm = float(cfg.max_grad_norm)
        self.seed = int(cfg.seed)

        self.resume = cfg.resume
        self.mixed_precision = str(cfg.mixed_precision).strip().lower()
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError(
                f"Unsupported mixed_precision: {cfg.mixed_precision}. "
                "Expected one of: ['no', 'fp16', 'bf16']."
            )
        self.wandb_enabled = bool(cfg.wandb.enabled)
        self.wandb_init_timeout = float(cfg.wandb.get("init_timeout", 30))

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            step_scheduler_with_optimizer=False,
        )
        configure_deepspeed_clipping(self.accelerator, self.max_grad_norm)
        ds_plugin = self.accelerator.state.deepspeed_plugin

        logger.info(
            "Accelerate training: distributed_type=%s zero_stage=%s world_size=%d process_index=%d cfg_mixed_precision=%s accelerator_mixed_precision=%s grad_accum=%d grad_clip=%.4f",
            self.accelerator.distributed_type,
            ds_plugin.deepspeed_config.get("zero_optimization", {}).get("stage", "unknown") if ds_plugin else "none",
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

        # Freeze non-trainable modules before optimizer/deepspeed initialization.
        # This keeps DiT (+ optional state_proj) as trainable when ZeRO builds optimizer state.
        self._apply_dit_only_train_mode(self.model)
        trainable_params = list(self.model.dit.parameters())
        state_proj = getattr(self.model, "state_proj", None)
        if state_proj is not None:
            trainable_params.extend(list(state_proj.parameters()))
        context_adapter = getattr(self.model, "context_adapter", None)
        if context_adapter is not None:
            trainable_params.extend(list(context_adapter.parameters()))
        action_camera_router = getattr(self.model, "action_camera_router", None)
        if action_camera_router is not None and getattr(self.model, "camera_router_enabled", True):
            trainable_params.extend(list(action_camera_router.parameters()))
        action_camera_critic = getattr(self.model, "action_camera_critic", None)
        critic_enabled = getattr(self.model, "camera_critic_enabled", True)
        critic_params = (
            list(action_camera_critic.parameters())
            if action_camera_critic is not None and critic_enabled
            else []
        )
        self._critic_param_group = None
        self._critic_optimizer_group = None
        param_groups = [{"params": trainable_params, "lr": self.learning_rate}]
        if critic_params:
            # Always isolate critic parameters, even without a separate LR:
            # ZeRO merges each group into FP32 partitions before AdamW.step.
            critic_lr = self.learning_rate if self.critic_lr_cfg is None else float(self.critic_lr_cfg["start"])
            param_groups.append({"params": critic_params, "lr": critic_lr})
            self._critic_optimizer_group = 1
            if self.critic_lr_cfg is not None:
                self._critic_param_group = 1
        self.optimizer = torch.optim.AdamW(
            param_groups, weight_decay=self.weight_decay, betas=(0.9, 0.95),
        )

        self.train_loader = self._build_loader(self.train_dataset, worker_init_fn=worker_init_fn)
        total_train_steps = self._estimate_total_train_steps()
        self.max_steps = total_train_steps
        warmup_steps = int(total_train_steps * 0.05)
        self.scheduler = self._build_scheduler(
            scheduler_type=cfg.lr_scheduler_type,
            total_train_steps=total_train_steps,
            warmup_steps=warmup_steps,
        )
        self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self._last_applied_stage = None
        self._last_train_metrics: dict[str, float] = {}

        self.checkpoint_root = os.path.join(self.output_dir, "checkpoints")
        self.weights_dir = os.path.join(self.checkpoint_root, "weights")
        self.state_dir = os.path.join(self.checkpoint_root, "state")
        self.eval_dir = os.path.join(self.output_dir, "eval")

        ensure_dir(self.output_dir)
        ensure_dir(self.checkpoint_root)
        ensure_dir(self.weights_dir)
        ensure_dir(self.state_dir)
        ensure_dir(self.eval_dir)

        self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.scheduler
        )
        self.optimizer.zero_grad(set_to_none=True)
        self.wandb_run = None
        self._init_wandb()
        self.accelerator.wait_for_everyone()
        self._resume_or_load_checkpoint()
        self.training_step = TrainingStep(
            self.accelerator, self.model, self.optimizer,
            max_grad_norm=self.max_grad_norm,
            critic_group_index=self._critic_optimizer_group,
        )
        if self.training_step.engine is not None:
            logger.info("DeepSpeed effective gradient_clipping=%.4f; one update per accumulation window.",
                        self.training_step.engine.gradient_clipping())

        val_size = len(self.val_dataset) if self.val_dataset is not None else len(self.train_dataset)
        logger.info("Train/val dataset size: %d/%d", len(self.train_dataset), val_size)
        self._apply_training_stage()

    @staticmethod
    def _stage_unit(stages) -> str:
        unit = str(stages.get("unit", stages.get("schedule", "steps"))).strip().lower()
        if unit in {"step", "steps", "step_ratio", "ratio"}:
            return "steps"
        if unit in {"epoch", "epochs"}:
            raise ValueError(
                "Epoch-based training_stages was removed. Use unit='steps' with stage*_ratio."
            )
        raise ValueError(f"training_stages.unit must be 'steps', got {unit!r}.")

    @staticmethod
    def _stage_fractions(stages) -> list[float]:
        ratio_keys = (
            "stage1_ratio",
            "stage2_ratio",
            "stage3_ratio",
            "stage4_ratio",
            "stage5_ratio",
        )
        epoch_keys = (
            "stage1_epochs",
            "stage2_epochs",
            "stage3_epochs",
            "stage4_epochs",
            "stage5_epochs",
        )
        has_ratio = any(stages.get(key) is not None for key in ratio_keys)
        has_epochs = any(stages.get(key) is not None for key in epoch_keys)
        if has_epochs and not has_ratio:
            raise ValueError(
                "Epoch-based training_stages (stage*_epochs) was removed. "
                "Set stage1_ratio..stage5_ratio with unit='steps'."
            )
        if not has_ratio:
            raise ValueError("training_stages requires stage1_ratio..stage5_ratio.")
        weights = [float(stages.get(key, 0.0) or 0.0) for key in ratio_keys]
        if min(weights) < 0:
            raise ValueError(f"training_stages weights must be non-negative, got {weights}.")
        total = float(sum(weights))
        if total <= 0:
            raise ValueError("training_stages weights must sum to a positive value.")
        if abs(total - 1.0) > 1e-3:
            raise ValueError(
                "training_stages ratios must sum to 1.0: "
                f"sum={total}."
            )
        return [weight / total for weight in weights]

    @staticmethod
    def _stage_step_bounds(fracs: list[float], max_steps: int) -> list[int]:
        max_steps = max(int(max_steps), 1)
        bounds = [0]
        acc = 0.0
        for idx, frac in enumerate(fracs):
            acc += float(frac)
            if idx == len(fracs) - 1:
                bounds.append(max_steps)
            else:
                bounds.append(int(round(acc * max_steps)))
        for idx in range(1, len(bounds)):
            bounds[idx] = min(max(bounds[idx], bounds[idx - 1]), max_steps)
        bounds[-1] = max_steps
        return bounds

    def _stage4_linear_progress(self) -> float:
        if self.training_stages is None:
            return 1.0
        max_steps = max(int(self.max_steps or 1), 1)
        bounds = self._stage_step_bounds(list(self.training_stages["fracs"]), max_steps)
        start, end = bounds[3], bounds[4]
        duration = end - start
        if duration <= 1:
            return 1.0
        offset = min(max(int(self.global_step) - start, 0), duration - 1)
        return offset / float(duration - 1)

    def _parse_training_stages(self, cfg: DictConfig):
        stages = cfg.get("training_stages", None)
        if stages is None or not bool(stages.get("enabled", False)):
            return None
        return {
            "unit": self._stage_unit(stages),
            "fracs": self._stage_fractions(stages),
        }

    def _current_training_stage_info(self) -> dict:
        if self.training_stages is None:
            return {
                "stage": None,
                "stage3_epoch": max(int(self.epoch), 0),
                "stage3_epochs": max(int(self.num_epochs), 1),
                "stage4_epoch": 0,
                "stage4_epochs": 1,
                "stage4_progress": None,
            }
        return self._current_step_training_stage_info()

    def _current_step_training_stage_info(self) -> dict:
        fracs = list(self.training_stages["fracs"])
        max_steps = max(int(self.max_steps or 1), 1)
        bounds = self._stage_step_bounds(fracs, max_steps)
        step = min(max(int(self.global_step), 0), max_steps)
        n_stages = len(fracs)
        stage = 1
        for idx in range(n_stages):
            start, end = bounds[idx], bounds[idx + 1]
            if start >= end:
                continue
            stage = idx + 1
            if step < end:
                break
        starts = {idx + 1: bounds[idx] for idx in range(n_stages) if bounds[idx + 1] > bounds[idx]}
        if stage not in starts:
            stage = max(starts) if starts else 1
        stage3_start, stage3_end = bounds[2], bounds[3]
        stage4_start, stage4_end = bounds[3], bounds[4]
        stage3_steps = max(stage3_end - stage3_start, 0)
        stage4_steps = max(stage4_end - stage4_start, 0)
        stage3_epoch = min(max(step - stage3_start, 0), max(stage3_steps - 1, 0)) if stage == 3 else 0
        stage4_epoch = min(max(step - stage4_start, 0), max(stage4_steps - 1, 0)) if stage == 4 else 0
        stage4_progress = self._stage4_linear_progress() if stage == 4 else 0.0
        return {
            "stage": stage,
            "stage3_epoch": stage3_epoch,
            "stage3_epochs": max(stage3_steps, 1),
            "stage4_epoch": stage4_epoch,
            "stage4_epochs": max(stage4_steps, 1),
            "stage4_progress": stage4_progress,
        }

    def _set_model_training_progress(self, model) -> dict:
        stage_info = self._current_training_stage_info()
        if hasattr(model, "set_train_progress"):
            model.set_train_progress(
                self.global_step,
                self.max_steps,
                epoch=self.epoch,
                stage=stage_info["stage"],
                stage3_epoch=stage_info["stage3_epoch"],
                stage3_epochs=stage_info["stage3_epochs"],
                stage4_epoch=stage_info["stage4_epoch"],
                stage4_epochs=stage_info["stage4_epochs"],
                stage4_progress=stage_info.get("stage4_progress"),
            )
        return stage_info

    def _apply_training_stage(self) -> None:
        model = self.accelerator.unwrap_model(self.model)
        stage_info = self._set_model_training_progress(model)
        if stage_info["stage"] != self._last_applied_stage:
            unit = "disabled" if self.training_stages is None else "steps"
            logger.info(
                "Training stage: stage=%s unit=%s epoch=%d/%d step=%s/%s.",
                stage_info["stage"],
                unit,
                self.epoch,
                self.num_epochs,
                self.global_step,
                self.max_steps,
            )
            self._last_applied_stage = stage_info["stage"]

    def _backward_main_loss(self, loss) -> None:
        unwrapped = self.accelerator.unwrap_model(self.model)
        has_deferred_cf = getattr(unwrapped, "_deferred_pixel_cf_ctx", None) is not None
        self.training_step.backward(loss, final_backward=not has_deferred_cf)

    def _backward_deferred_pixel_cf(self, loss, loss_dict: dict) -> tuple:
        unwrapped = self.accelerator.unwrap_model(self.model)
        if not hasattr(unwrapped, "run_deferred_pixel_critic_cf"):
            return loss, loss_dict
        with self.accelerator.autocast():
            cf_loss, cf_metrics = unwrapped.run_deferred_pixel_critic_cf()
        if cf_loss is None:
            # Do not clear critic gradients from earlier microbatches. The
            # optimizer hook skips it only if the whole window has no CF.
            return loss, loss_dict
        self.training_step.backward(cf_loss, critic=True)
        router_cf_lambda = float(getattr(unwrapped, "router_cf_lambda", 1.0))
        for key, value in cf_metrics.items():
            if key == "loss_router_cf":
                loss_dict[key] = router_cf_lambda * value
            else:
                loss_dict[key] = value
        return loss.detach() + cf_loss.detach(), loss_dict

    def _init_wandb(self):
        if not self.wandb_enabled or not self.accelerator.is_main_process:
            return
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb logging is enabled in config (`wandb.enabled=true`) but wandb is not installed."
            ) from e

        workspace = self.cfg.wandb.workspace
        if workspace is not None:
            workspace = str(workspace).strip()
        if workspace is None or workspace.lower() in {"", "null", "none"}:
            workspace = None

        try:
            self.wandb_run = wandb.init(
                entity=workspace,
                project=self.cfg.wandb.project,
                name=self.cfg.wandb.name,
                group=None if self.cfg.wandb.group in (None, "null", "") else str(self.cfg.wandb.group),
                mode=self.cfg.wandb.mode,
                dir=self.output_dir,
                settings=wandb.Settings(init_timeout=self.wandb_init_timeout),
            )
        except Exception as e:
            logger.warning(
                "wandb init failed; disabling wandb for this run. "
                "workspace=%s project=%s name=%s error=%s",
                workspace,
                self.cfg.wandb.project,
                self.cfg.wandb.name,
                e,
            )
            self.wandb_run = None
            return
        logger.info(
            "Initialized wandb run: workspace=%s project=%s name=%s",
            workspace,
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
        self.train_sampler = ResumableEpochSampler(
            dataset=dataset,
            seed=self.seed,
            batch_size=self.batch_size,
            num_processes=self.accelerator.num_processes,
        )
        loader_kwargs = {
            "batch_size": self.batch_size,
            "shuffle": False,
            "sampler": self.train_sampler,
            "num_workers": self.num_workers,
            "pin_memory": torch.cuda.is_available(),
            "worker_init_fn": worker_init_fn,
        }
        if self.num_workers > 0:
            loader_kwargs["prefetch_factor"] = self.prefetch_factor
            loader_kwargs["persistent_workers"] = True
        logger.info(
            "Building train DataLoader: batch_size=%d num_workers=%d prefetch_factor=%s pin_memory=%s persistent_workers=%s",
            self.batch_size,
            self.num_workers,
            loader_kwargs.get("prefetch_factor"),
            loader_kwargs["pin_memory"],
            loader_kwargs.get("persistent_workers", False),
        )
        return DataLoader(dataset, **loader_kwargs)

    def _assert_dataset_length_consistent(self, dataset, dataset_name: str):
        if not hasattr(dataset, "__len__"):
            raise TypeError(f"`{dataset_name}` must implement __len__ for rank consistency checks.")

        local_length = len(dataset)
        gathered_lengths = self.accelerator.gather(
            torch.tensor([local_length], device=self.accelerator.device, dtype=torch.int64)
        ).reshape(-1)
        if torch.all(gathered_lengths == gathered_lengths[0]):
            return

        if self.accelerator.is_main_process:
            print(f"[dataset-check] {dataset_name} length mismatch across ranks after initialization:")
            for rank, rank_length in enumerate(gathered_lengths.cpu().tolist()):
                print(f"rank {rank}: {rank_length}")
        self.accelerator.wait_for_everyone()
        raise RuntimeError(
            f"{dataset_name} length mismatch across ranks: {gathered_lengths.cpu().tolist()}"
        )

    def _estimate_total_train_steps(self) -> int:
        if self.max_steps is not None:
            return max(int(self.max_steps), 1)

        if not hasattr(self.train_dataset, "__len__"):
            raise TypeError("`train_dataset` must implement __len__ when `max_steps` is None.")

        num_processes = max(int(self.accelerator.num_processes), 1)
        global_batch_size = max(self.batch_size * num_processes, 1)
        micro_steps_per_epoch = max(ceil(len(self.train_dataset) / global_batch_size), 1)
        opt_steps_per_epoch = max(
            ceil(micro_steps_per_epoch / self.gradient_accumulation_steps),
            1,
        )
        return max(opt_steps_per_epoch * self.num_epochs, 1)

    @staticmethod
    def _parse_critic_lr(cfg: DictConfig):
        model_cfg = cfg.get("model", None)
        router_cf = None if model_cfg is None else model_cfg.get("router_cf", None)
        start = None if router_cf is None else router_cf.get("critic_lr", None)
        if start is None:
            return None
        end = router_cf.get("critic_lr_end", start)
        start_lr = float(start)
        end_lr = float(end)
        if start_lr <= 0.0 or end_lr < 0.0:
            raise ValueError(
                f"`router_cf.critic_lr` must be > 0 and `critic_lr_end` >= 0, got {start_lr}, {end_lr}."
            )
        return {"start": start_lr, "end": end_lr}

    def _critic_schedule_start_step(self) -> int:
        if self.training_stages is None:
            return 0
        max_steps = max(int(self.max_steps or 1), 1)
        bounds = self._stage_step_bounds(list(self.training_stages["fracs"]), max_steps)
        return int(bounds[2])

    def _build_scheduler(self, scheduler_type, total_train_steps: int, warmup_steps: int = 0):
        scheduler_type = str(scheduler_type).strip().lower()
        total_train_steps = max(int(total_train_steps), 1)
        warmup_steps = min(max(int(warmup_steps), 0), total_train_steps - 1)
        remaining_steps = max(total_train_steps - warmup_steps, 1)

        if self._critic_param_group is None:
            if scheduler_type == "cosine":
                main_scheduler = CosineAnnealingLR(
                    self.optimizer,
                    T_max=remaining_steps,
                    eta_min=self.learning_rate * 0.01,
                )
            elif scheduler_type == "constant":
                main_scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=remaining_steps)
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

        eta_ratio = 0.01 if scheduler_type == "cosine" else 1.0
        if scheduler_type not in {"cosine", "constant"}:
            raise ValueError(
                f"Unsupported lr_scheduler_type: {scheduler_type}. "
                "Expected one of: ['cosine', 'constant']."
            )
        critic_start = self._critic_schedule_start_step()
        critic_end_ratio = float(self.critic_lr_cfg["end"]) / float(self.critic_lr_cfg["start"])
        logger.info(
            "Critic LR is separate: %.2e cosine to %.2e from step %d (stage 3 start).",
            float(self.critic_lr_cfg["start"]),
            float(self.critic_lr_cfg["end"]),
            critic_start,
        )

        def main_lambda(last_epoch: int) -> float:
            if scheduler_type == "constant":
                if warmup_steps <= 0 or last_epoch >= warmup_steps:
                    return 1.0
                if last_epoch <= 0:
                    return 1.0 / float(warmup_steps)
                start_factor = 1.0 / float(warmup_steps)
                return start_factor + (1.0 - start_factor) * (last_epoch / float(warmup_steps))
            if warmup_steps > 0 and last_epoch < warmup_steps:
                if last_epoch <= 0:
                    return 1.0 / float(warmup_steps)
                start_factor = 1.0 / float(warmup_steps)
                return start_factor + (1.0 - start_factor) * (last_epoch / float(warmup_steps))
            t = min(max(last_epoch - warmup_steps, 0), remaining_steps)
            return eta_ratio + (1.0 - eta_ratio) * (1.0 + math.cos(math.pi * t / float(remaining_steps))) / 2.0

        def critic_lambda(last_epoch: int) -> float:
            duration = max(total_train_steps - critic_start - 1, 1)
            if last_epoch < critic_start:
                return 1.0
            t = min(max(last_epoch - critic_start, 0), duration) / float(duration)
            return critic_end_ratio + (1.0 - critic_end_ratio) * (1.0 + math.cos(math.pi * t)) / 2.0

        return LambdaLR(self.optimizer, lr_lambda=[main_lambda, critic_lambda])

    def _estimate_eta(self):
        elapsed = max(time.perf_counter() - self.run_start_time, 1e-6)
        done_steps = max(self.global_step - self.run_start_step, 1)
        steps_per_sec = done_steps / elapsed
        remaining_steps = max(self.max_steps - self.global_step, 0)
        eta_seconds = int(remaining_steps / max(steps_per_sec, 1e-9))
        eta_h, eta_rem = divmod(eta_seconds, 3600)
        eta_m, eta_s = divmod(eta_rem, 60)
        return f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}", steps_per_sec

    def _resume_or_load_checkpoint(self):
        resume = self.resume
        if not resume:
            return
        resume_path = Path(str(resume))
        migrate_state = bool(self.cfg.get("migrate_state_conditioning", False))
        if resume_path.is_dir():
            if migrate_state:
                raise ValueError("State migration requires a .pt weight checkpoint and a fresh optimizer, not full training state.")
            logger.info("Resuming full training state from directory: %s", resume)
            self.load_training_state(str(resume_path))
            return
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        logger.info("Loading weight checkpoint only: %s", resume)
        self.accelerator.unwrap_model(self.model).load_checkpoint(
            str(resume_path), optimizer=None, migrate_state_conditioning=migrate_state,
        )
        logger.warning("Loaded .pt weights only; optimizer/scheduler/step were not restored under ZeRO2.")

    def _set_dit_only_train_mode(self):
        # Match DiffSynth's freeze_except("dit"): only DiT stays trainable/in-train-mode.
        logger.info("Setting DiT to train mode and freezing other model components.")
        model = self.accelerator.unwrap_model(self.model)
        self._apply_dit_only_train_mode(model)

    @staticmethod
    def _apply_dit_only_train_mode(model):
        model.eval()
        model.requires_grad_(False)
        model.dit.train()
        model.dit.requires_grad_(True)
        state_proj = getattr(model, "state_proj", None)
        if state_proj is not None:
            state_proj.train()
            state_proj.requires_grad_(True)
        context_adapter = getattr(model, "context_adapter", None)
        if context_adapter is not None:
            context_adapter.train()
            context_adapter.requires_grad_(True)
        action_camera_router = getattr(model, "action_camera_router", None)
        if action_camera_router is not None and getattr(model, "camera_router_enabled", True):
            action_camera_router.train()
            action_camera_router.requires_grad_(True)
        action_camera_critic = getattr(model, "action_camera_critic", None)
        if action_camera_critic is not None and getattr(model, "camera_critic_enabled", True):
            action_camera_critic.train()
            action_camera_critic.requires_grad_(True)

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
                "Evaluation now expects `video` with shape [3,T,H,W], [B,3,T,H,W], "
                "[T,V,3,H,W], or [B,T,V,3,H,W]."
            )
        if video.ndim == 4:
            video = video.unsqueeze(0)
        elif video.ndim == 5 and video.shape[2] == 3:
            video = video.unsqueeze(0)
        if video.ndim not in (5, 6):
            raise ValueError(
                "Expected video shape [3,T,H,W], [B,3,T,H,W], [T,V,3,H,W], "
                f"or [B,T,V,3,H,W], got {tuple(video.shape)}"
            )
        num_video_frames = video.shape[1] if video.ndim == 6 else video.shape[2]
        if num_video_frames <= 1:
            raise ValueError(f"`sample['video']` must have at least 2 frames for action evaluation, got {num_video_frames}")

        if isinstance(prompt, str):
            prompt = [prompt]
        elif isinstance(prompt, tuple):
            prompt = list(prompt)
        elif not isinstance(prompt, list):
            raise TypeError(f"Expected prompt type str/list[str], got {type(prompt)}")
        if len(prompt) != video.shape[0]:
            raise ValueError(f"Prompt batch mismatch: len(prompt)={len(prompt)} vs video batch={video.shape[0]}")

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
                raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
            if action.shape[1] % (num_video_frames - 1) != 0:
                raise ValueError(f"`sample['action']` temporal dimension must be divisible by video frames-1={num_video_frames - 1}, got {action.shape[1]}")
            action_horizon = int(action.shape[1])

        proprio = None
        if "proprio" in sample:
            proprio = sample["proprio"]
            if not isinstance(proprio, torch.Tensor):
                raise TypeError(f"`sample['proprio']` must be a torch.Tensor, got {type(proprio)}")
            if proprio.ndim == 2:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")

        if context is not None or context_mask is not None:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must both exist in eval sample.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )

        return {
            "video": video,
            "prompt": prompt,
            "action": action,
            "proprio": proprio,
            "context": context,
            "context_mask": context_mask,
            "action_horizon": action_horizon,
        }

    @staticmethod
    def _pil_video_to_eval_tensor(frames):
        if len(frames) > 0 and isinstance(frames[0], list):
            return torch.stack([pil_frames_to_video_tensor(view_frames) for view_frames in frames], dim=0)
        return pil_frames_to_video_tensor(frames)

    @staticmethod
    def _video_metric_mean(metric_fn, pred: torch.Tensor, target: torch.Tensor) -> float:
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}")
        if pred.ndim == 4:
            return metric_fn(pred=pred, target=target)
        if pred.ndim == 5:
            values = [metric_fn(pred=pred[v], target=target[v]) for v in range(pred.shape[0])]
            return float(sum(values) / len(values))
        raise ValueError(f"Expected eval video tensor [3,T,H,W] or [V,3,T,H,W], got {tuple(pred.shape)}")

    @staticmethod
    def _flatten_eval_views(video: torch.Tensor) -> torch.Tensor:
        if video.ndim == 4:
            return video
        if video.ndim == 5:
            return torch.cat([video[v] for v in range(video.shape[0])], dim=3)
        raise ValueError(f"Expected [3,T,H,W] or [V,3,T,H,W], got {tuple(video.shape)}")

    def _release_cached_cuda(self) -> None:
        model = self.accelerator.unwrap_model(self.model)
        if getattr(model, "_deferred_pixel_cf_ctx", None) is not None:
            model._deferred_pixel_cf_ctx = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def evaluate(self):
        if self.val_dataset is None:
            return None

        self.optimizer.zero_grad(set_to_none=True)
        self._release_cached_cuda()

        model = self.accelerator.unwrap_model(self.model)
        was_dit_training = model.dit.training
        model.eval()

        # eval_index = (self.global_step + self.accelerator.process_index) % len(self.val_dataset)
        rng = torch.Generator(device="cpu").manual_seed(self.global_step + self.accelerator.process_index)
        eval_index = torch.randint(0, len(self.val_dataset), (1,), generator=rng).item()
        sample = self._to_batched_eval_sample(self.val_dataset[eval_index])

        # 1. training loss
        with self.accelerator.autocast():
            val_loss, _ = model.training_loss(sample)
            val_loss = val_loss.float().item()
        self._release_cached_cuda()

        prompt = sample["prompt"][0]
        video0 = sample["video"][0] # Tensor [3,T,H,W] or [T,V,3,H,W] in (-1, 1)
        action = sample["action"][0] if "action" in sample and sample["action"] is not None else None
        proprio = sample["proprio"][0, 0] if "proprio" in sample and sample["proprio"] is not None else None # from [1, T, d] to [d]
        if video0.ndim == 4:
            input_image = video0[:, 0].unsqueeze(0)
            _, num_frames, _, _ = video0.shape
            gt_video_tensor = ((video0.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()
            gt_video_batch = video0.unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
        elif video0.ndim == 5:
            input_image = video0[0].contiguous()
            num_frames = video0.shape[0]
            gt_video_tensor = (
                (video0.detach().float().cpu().permute(1, 2, 0, 3, 4).clamp(-1.0, 1.0) + 1.0) * 0.5
            ).contiguous()
            gt_video_batch = video0.permute(1, 2, 0, 3, 4).unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
        else:
            raise ValueError(f"Expected eval video [3,T,H,W] or [T,V,3,H,W], got {tuple(video0.shape)}")

        # 2. inference and video saving
        infer_kwargs = {
            "input_image": input_image,
            "num_frames": num_frames,
            "action": action,
            "action_horizon": sample['action_horizon'],
            "proprio": proprio,
            "text_cfg_scale": 1.0,
            "action_cfg_scale": 1.0,
            "num_inference_steps": self.eval_num_inference_steps,
            "seed": 42,
            "tiled": False,
        }
        if sample["context"] is not None:
            infer_kwargs["prompt"] = None
            infer_kwargs["context"] = sample["context"][0]
            infer_kwargs["context_mask"] = sample["context_mask"][0]
        else:
            infer_kwargs["prompt"] = prompt

        pred = model.infer(
            **infer_kwargs,
        )

        pred_video = pred["video"]
        pred_action = pred.get("action", None)

        # 3. inference metrics against GT video
        pred_video_tensor = self._pil_video_to_eval_tensor(pred_video)

        assert pred_video_tensor.shape == gt_video_tensor.shape, (
            "Eval infer prediction/GT shape mismatch: "
            f"pred={tuple(pred_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_rollout_vs_gt = self._video_metric_mean(video_psnr, pred=pred_video_tensor, target=gt_video_tensor)
        ssim_rollout_vs_gt = self._video_metric_mean(video_ssim, pred=pred_video_tensor, target=gt_video_tensor)

        action_l1 = None
        action_l2 = None
        if action is not None and pred_action is not None:
            if sample["proprio"] is None:
                raise ValueError("Eval sample must contain `proprio` for action denormalization.")
            proprio = sample["proprio"].detach().to(device="cpu", dtype=torch.float32)

            processor = self.val_dataset.lerobot_dataset.processor

            denorm_actions = {}
            action_meta = processor.shape_meta["action"]
            state_meta = processor.shape_meta["state"]
            for action_name, raw_action in (("pred", pred_action), ("gt", action)):
                if not isinstance(raw_action, torch.Tensor):
                    raise TypeError(f"{action_name} action must be a torch.Tensor, got {type(raw_action)}")
                if raw_action.ndim == 2:
                    action_btd = raw_action.unsqueeze(0)
                elif raw_action.ndim == 3 and raw_action.shape[0] == 1:
                    action_btd = raw_action
                else:
                    raise ValueError(
                        f"{action_name} action must have shape [T, D] or [1, T, D], got {tuple(raw_action.shape)}"
                    )
                action_btd = action_btd.detach().to(device="cpu", dtype=torch.float32)

                batch = {
                    "action": action_btd,
                    "state": proprio,
                }
                batch = processor.action_state_merger.backward(batch)
                batch = processor.normalizer.backward(batch)
                merged_batch = {
                    "action": {meta["key"]: batch["action"][meta["key"]].squeeze(0) for meta in action_meta},
                    "state": {meta["key"]: batch["state"][meta["key"]].squeeze(0) for meta in state_meta},
                }
                merged_batch = processor.action_state_merger.forward(merged_batch)
                denorm_action = merged_batch["action"].unsqueeze(0)
                if denorm_action.ndim != 3 or denorm_action.shape[0] != 1:
                    raise ValueError(
                        f"Denormalized {action_name} action must have shape [1, T, D], got {tuple(denorm_action.shape)}"
                    )
                denorm_actions[action_name] = denorm_action

            pred_action_denorm = denorm_actions["pred"]
            gt_action_denorm = denorm_actions["gt"]

            if pred_action_denorm.shape != gt_action_denorm.shape:
                raise ValueError(
                    "Predicted action/GT action shape mismatch after denormalization: "
                    f"pred={tuple(pred_action_denorm.shape)} vs gt={tuple(gt_action_denorm.shape)}"
                )
            action_diff = pred_action_denorm - gt_action_denorm
            action_l1 = action_diff.abs().mean().item()
            action_l2 = action_diff.pow(2).mean().item()

        # 4. VAE reconstruction metrics against GT video
        vae_latents = model._encode_video_latents(gt_video_batch, tiled=False)
        vae_recon_video = model._decode_latents(vae_latents, tiled=False)
        vae_video_tensor = self._pil_video_to_eval_tensor(vae_recon_video)

        assert vae_video_tensor.shape == gt_video_tensor.shape, (
            "Eval VAE reconstruction/GT shape mismatch: "
            f"vae={tuple(vae_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_decode_vs_gt = self._video_metric_mean(video_psnr, pred=vae_video_tensor, target=gt_video_tensor)
        ssim_decode_vs_gt = self._video_metric_mean(video_ssim, pred=vae_video_tensor, target=gt_video_tensor)

        psnr_rollout_vs_decode = self._video_metric_mean(video_psnr, pred=pred_video_tensor, target=vae_video_tensor)
        ssim_rollout_vs_decode = self._video_metric_mean(video_ssim, pred=pred_video_tensor, target=vae_video_tensor)

        stitched_video_tensor = torch.cat(
            [
                self._flatten_eval_views(pred_video_tensor),
                self._flatten_eval_views(vae_video_tensor),
                self._flatten_eval_views(gt_video_tensor),
            ],
            dim=2,
        ).contiguous()
        stitched_frames = []
        for t in range(stitched_video_tensor.shape[1]):
            frame = (stitched_video_tensor[:, t].permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
            stitched_frames.append(Image.fromarray(frame))

        video_path = os.path.join(
            self.eval_dir,
            f"step_{self.global_step:06d}_rank_{self.accelerator.process_index:03d}.mp4",
        )
        save_mp4(stitched_frames, video_path, fps=8)

        local_metrics = torch.tensor(
            [
                float(val_loss),
                float(psnr_rollout_vs_gt),
                float(ssim_rollout_vs_gt),
                float(psnr_rollout_vs_decode),
                float(ssim_rollout_vs_decode),
                float(psnr_decode_vs_gt),
                float(ssim_decode_vs_gt),
                float(action_l2) if action_l2 is not None else -1.0,
                float(action_l1) if action_l1 is not None else -1.0,
            ],
            device=self.accelerator.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        gathered_metrics = self.accelerator.gather_for_metrics(local_metrics)
        mean_metrics = gathered_metrics[:, :7].mean(dim=0)
        action_l2_mean = gathered_metrics[:, 7].mean().item() if action_l2 is not None else None
        action_l1_mean = gathered_metrics[:, 8].mean().item() if action_l1 is not None else None

        if was_dit_training:
            self._set_dit_only_train_mode()

        result = {
            "val_loss": float(mean_metrics[0].item()),
            "psnr_rg": float(mean_metrics[1].item()),
            "ssim_rg": float(mean_metrics[2].item()),
            "psnr_rd": float(mean_metrics[3].item()),
            "ssim_rd": float(mean_metrics[4].item()),
            "psnr_dg": float(mean_metrics[5].item()),
            "ssim_dg": float(mean_metrics[6].item()),
            "video_path": video_path,
        }
        if action_l2_mean is not None:
            result["action_l2"] = float(action_l2_mean)
        if action_l1_mean is not None:
            result["action_l1"] = float(action_l1_mean)
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
            "state_conditioning": self.accelerator.unwrap_model(self.model).state_conditioning_signature(),
        }
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    def save_checkpoint(self):
        step_tag = f"step_{self.global_step:06d}"

        self.accelerator.wait_for_everyone()
        ckpt_path = None
        if self.save_weights and self.accelerator.is_main_process:
            ckpt_path = self._save_weights_checkpoint(step_tag=step_tag)
        self.accelerator.wait_for_everyone()

        state_path = None
        if self.save_training_state:
            state_path = os.path.join(self.state_dir, step_tag)
            ensure_dir(state_path)
            self.accelerator.save_state(output_dir=state_path)
            if self.accelerator.is_main_process:
                self._save_trainer_state(state_path)
            self.accelerator.wait_for_everyone()

        return {"weights_path": ckpt_path, "state_path": state_path}

    def load_training_state(self, state_dir: str):
        state_file = Path(state_dir) / "trainer_state.json"
        payload = {}
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
        signature = self.accelerator.unwrap_model(self.model).state_conditioning_signature()
        saved_signature = payload.get("state_conditioning")
        if saved_signature is None:
            if signature["mode"] != "separate" or signature["drop_prob"] or any(signature[role]["enabled"] for role in ("video", "router", "critic")):
                raise ValueError("Legacy full training state has no optional state paths; migrate a .pt checkpoint instead.")
        elif {"mode": "separate", **saved_signature} != signature:
            raise ValueError("State conditioning config differs from full training checkpoint; use weight-only migration for architecture changes.")
        self.accelerator.load_state(input_dir=state_dir)
        if state_file.exists():
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
                    self.batch_in_epoch * self.batch_size * self.accelerator.num_processes,
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
        logger.info("Loaded accelerate training state from %s at step=%d", state_dir, self.global_step)
        logger.warning(
            "State file `%s` is missing; dataloader progress resume is skipped.",
            state_file,
        )

    def train(self):
        self._set_dit_only_train_mode()

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        if self.max_steps is None:
            raise ValueError("`max_steps` must be set before entering the while-step training loop.")

        logger.info("Starting training with max_steps=%d.", self.max_steps)
        data_iter = iter(self.train_loader)
        self.run_start_step = self.global_step
        self.run_start_time = time.perf_counter()

        while self.global_step < self.max_steps:
            try:
                sample = next(data_iter)
                self.batch_in_epoch += 1
            except StopIteration:
                self.epoch += 1
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                data_iter = iter(self.train_loader)
                continue

            with self.accelerator.accumulate(self.model):
                train_model = self.model if hasattr(self.model, "training_loss") else self.accelerator.unwrap_model(self.model)
                self._apply_training_stage()

                with self.accelerator.autocast():
                    loss, loss_dict = train_model.training_loss(sample)
                self._backward_main_loss(loss)
                loss, loss_dict = self._backward_deferred_pixel_cf(loss, loss_dict)

                if self.accelerator.sync_gradients:
                    grad_norm = self.training_step.step()
                    if not self.accelerator.optimizer_step_was_skipped:
                        self.scheduler.step()
                    self.global_step += 1
                    global_loss = float(
                        self.accelerator.gather(loss.detach().float().reshape(1)).mean().item()
                    )
                    global_loss_metrics = {}
                    for key, value in loss_dict.items():
                        metric_tensor = torch.tensor(float(value), device=loss.device, dtype=torch.float32).reshape(1)
                        global_loss_metrics[key] = float(
                            self.accelerator.gather(metric_tensor).mean().item()
                        )
                    grad_norm_tensor = torch.tensor(
                        float(grad_norm), device=loss.device, dtype=torch.float32
                    )
                    global_grad_norm = float(self.accelerator.gather(grad_norm_tensor).mean().item())

                    current_lr = float(self.optimizer.param_groups[0]["lr"])
                    critic_lr = None
                    if self._critic_param_group is not None and len(self.optimizer.param_groups) > self._critic_param_group:
                        critic_lr = float(self.optimizer.param_groups[self._critic_param_group]["lr"])
                    logged_metrics = dict(self._last_train_metrics)
                    logged_metrics.update(global_loss_metrics)
                    self._last_train_metrics = logged_metrics

                    if self.log_every > 0 and self.global_step % self.log_every == 0 and self.accelerator.is_main_process:
                        eta_str, steps_per_sec = self._estimate_eta()
                        description = "[train] epoch=%d step=%d/%d loss=%.4f " % (
                            self.epoch,
                            self.global_step,
                            self.max_steps,
                            global_loss,
                        )
                        if logged_metrics:
                            detail_str = " ".join([f"{k}={v:.4f}" for k, v in sorted(logged_metrics.items())])
                            description += detail_str + " "
                        description += "lr=%.2e " % current_lr
                        if critic_lr is not None:
                            description += "critic_lr=%.2e " % critic_lr
                        description += "speed=%.2f step/s, %.2f samples/s eta=%s" % (
                            steps_per_sec,
                            steps_per_sec * self.batch_size * self.accelerator.num_processes,
                            eta_str,
                        )
                        logger.info(description)

                        wandb_payload = {
                            "train/loss": global_loss,
                            "train/grad_norm": global_grad_norm,
                            "train/lr": current_lr,
                            "performance/steps_per_sec": steps_per_sec,
                            "performance/samples_per_sec": steps_per_sec * self.batch_size * self.accelerator.num_processes,
                        }
                        if critic_lr is not None:
                            wandb_payload["train/critic_lr"] = critic_lr
                        for key, value in logged_metrics.items():
                            wandb_payload[f"train/{key}"] = value
                        self._wandb_log(wandb_payload)

                    if (
                        self.eval_every > 0
                        and self.val_dataset is not None
                        and self.global_step % self.eval_every == 0
                    ):
                        metrics = self.evaluate()
                        self.accelerator.wait_for_everyone()
                        if metrics is not None and self.accelerator.is_main_process:
                            description = "[eval] step=%d val_loss=%.4f infer_psnr=%.4f infer_ssim=%.4f" % (
                                self.global_step,
                                metrics["val_loss"],
                                metrics["psnr_rd"],
                                metrics["ssim_rd"],
                            )
                            if "action_l2" in metrics:
                                description += " action_l2=%.4f" % metrics["action_l2"]
                            if "action_l1" in metrics:
                                description += " action_l1=%.4f" % metrics["action_l1"]
                            logger.info(description)
                            eval_payload = {
                                "eval/val_loss": float(metrics["val_loss"]),
                                "eval/psnr_rg": float(metrics["psnr_rg"]),
                                "eval/ssim_rg": float(metrics["ssim_rg"]),
                                "eval/psnr_rd": float(metrics["psnr_rd"]),
                                "eval/ssim_rd": float(metrics["ssim_rd"]),
                                "eval/psnr_dg": float(metrics["psnr_dg"]),
                                "eval/ssim_dg": float(metrics["ssim_dg"]),
                            }
                            if "action_l2" in metrics:
                                eval_payload["eval/action_l2"] = float(metrics["action_l2"])
                            if "action_l1" in metrics:
                                eval_payload["eval/action_l1"] = float(metrics["action_l1"])
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

        ckpt_info = self.save_checkpoint()
        if self.accelerator.is_main_process:
            logger.info(
                "[done] training finished step=%d weights=%s state=%s",
                self.global_step,
                ckpt_info["weights_path"],
                ckpt_info["state_path"],
            )
