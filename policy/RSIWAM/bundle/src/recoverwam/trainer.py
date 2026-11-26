# See LICENSE and THIRD_PARTY_NOTICES.md for retained source notices.
import logging
import json
import inspect
import math
import os
import re
from math import ceil
from pathlib import Path
import time
import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LambdaLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from .utils.fs import ensure_dir
from .utils.logging_config import get_logger, setup_logging
from .utils.pytorch_utils import set_global_seed
from .utils.samplers import ResumableEpochSampler
from .utils.training_profiler import TrainingProfiler
from .utils.video_io import save_mp4
from .utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim

logger = get_logger(__name__)


class JointTrainer:
    supported_modes = {("joint", "full")}

    @staticmethod
    def _validate_preflight_config(cfg):
        from recoverwam.preflight import validate_preflight_config

        return validate_preflight_config(cfg)

    def __init__(
        self,
        model,
        train_dataset,
        val_dataset=None,
        *,
        cfg: DictConfig,
        evaluator=None,
        profiler: TrainingProfiler | None = None,
    ):
        if (
            cfg.get("train_mode", "joint"),
            cfg.get("checkpoint_mode", "full"),
        ) not in self.supported_modes:
            raise ValueError(
                f"Unsupported trainer mode/checkpoint pair: {cfg.train_mode}/{cfg.checkpoint_mode}"
            )
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.evaluator = evaluator
        self.cfg = cfg
        self.preflight = self._validate_preflight_config(cfg)
        self.preflight_contract = None
        self.preflight_last_loss = None
        self.preflight_last_loss_metrics = None
        self.preflight_finalized = False
        self.output_dir = str(cfg.output_dir)
        self.profiler = profiler or TrainingProfiler.disabled(Path(self.output_dir) / "profiling")
        self.learning_rate = float(cfg.learning_rate)
        self.lr_min_ratio = float(cfg.get("lr_min_ratio", 0.01))
        if not 0.0 <= self.lr_min_ratio < 1.0:
            raise ValueError(f"`lr_min_ratio` must be in [0, 1), got {self.lr_min_ratio}")
        self.warmup_steps = int(cfg.warmup_steps)
        if self.warmup_steps < 0:
            raise ValueError(f"`warmup_steps` must be non-negative, got {self.warmup_steps}.")
        self.weight_decay = float(cfg.weight_decay)
        self.batch_size = int(cfg.batch_size)
        self.num_workers = int(cfg.num_workers)
        self.num_epochs = int(cfg.num_epochs)
        max_steps = cfg.max_steps
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.log_every = int(cfg.log_every)
        self.save_every = int(cfg.save_every)
        self.save_final_checkpoint = bool(cfg.get("save_final_checkpoint", True))
        if self.preflight is not None and (not self.save_final_checkpoint):
            raise ValueError("Preflight runs require `save_final_checkpoint=true`.")
        self.eval_every = int(cfg.eval_every)
        self.eval_num_inference_steps = int(cfg.eval_num_inference_steps)
        self.gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
        legacy_train_wan = bool(cfg.get("train_wan", True))
        self.train_mode = (
            str(cfg.get("train_mode", "joint" if legacy_train_wan else "action_only"))
            .strip()
            .lower()
        )
        self.train_wan = self.train_mode != "action_only"
        self.video_lora_enabled = bool(getattr(self.model, "video_lora_enabled", False))
        model_cfg = cfg.get("model")
        video_lora_cfg = None if model_cfg is None else model_cfg.get("video_lora")
        self.video_lora_learning_rate = self.learning_rate
        alternating_cfg = cfg.get("alternating_training")
        self.alternating_video_steps = 0
        self.alternating_action_steps = 0
        self.video_learning_rate = self.learning_rate
        self.action_learning_rate = self.learning_rate
        joint_lrs_cfg = cfg.get("joint_learning_rates")
        self.joint_learning_rate_groups = joint_lrs_cfg is not None
        if self.joint_learning_rate_groups:
            self.video_learning_rate = float(joint_lrs_cfg.get("video"))
            self.action_learning_rate = float(joint_lrs_cfg.get("action"))
            if self.video_learning_rate <= 0 or self.action_learning_rate <= 0:
                raise ValueError("`joint_learning_rates.video/action` must be positive.")
        self.optimizer_betas = self._normalize_optimizer_betas(
            cfg.get("optimizer_betas", [0.9, 0.95])
        )
        self.checkpoint_mode = str(cfg.get("checkpoint_mode", "full")).strip().lower()
        self.frozen_video_checkpoint = cfg.get("frozen_video_checkpoint")
        self.max_grad_norm = float(cfg.max_grad_norm)
        self.seed = int(cfg.seed)
        self.resume = cfg.get("resume")
        self.resume_ckpt = cfg.get("resume_ckpt")
        self.lr_continuation = cfg.get("lr_continuation")
        self.lr_tail = cfg.get("lr_tail")
        if self.lr_tail and self.lr_continuation:
            raise ValueError("lr_tail and lr_continuation are mutually exclusive")
        if self.lr_continuation and (
            not self.resume or not Path(str(self.resume)).is_dir() or self.resume_ckpt
        ):
            raise ValueError("lr_continuation requires a full training-state directory")
        self.resume_ckpt_load_video = bool(cfg.get("resume_ckpt_load_video", True))
        if self.resume and self.resume_ckpt:
            raise ValueError(
                "`resume` and `resume_ckpt` are mutually exclusive; configure only one."
            )
        self.mixed_precision = str(cfg.mixed_precision).strip().lower()
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError(
                f"Unsupported mixed_precision: {cfg.mixed_precision}. Expected one of: ['no', 'fp16', 'bf16']."
            )
        self.wandb_enabled = bool(cfg.wandb.enabled)
        with self.profiler.measure_startup("accelerator_init_ms", cuda=True):
            self.accelerator = Accelerator(
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                mixed_precision=self.mixed_precision,
                step_scheduler_with_optimizer=False,
            )
        logger.info(
            "Accelerate training: distributed_type=%s zero_stage=%s world_size=%d process_index=%d cfg_mixed_precision=%s accelerator_mixed_precision=%s grad_accum=%d grad_clip=%.4f train_mode=%s",
            self.accelerator.distributed_type,
            self._deepspeed_zero_stage(),
            self.accelerator.num_processes,
            self.accelerator.process_index,
            self.mixed_precision,
            self.accelerator.mixed_precision,
            self.gradient_accumulation_steps,
            self.max_grad_norm,
            self.train_mode,
        )
        self.profiler.rank = int(self.accelerator.process_index)
        self.profiler.local_rank = int(self.accelerator.local_process_index)
        self.profiler.world_size = int(self.accelerator.num_processes)
        logger.info("using accelerator.device=%s", self.accelerator.device)
        worker_init_fn = set_global_seed(self.seed, get_worker_init_fn=True)
        self._assert_dataset_length_consistent(self.train_dataset, "train_dataset")
        if self.val_dataset is not None:
            self._assert_dataset_length_consistent(self.val_dataset, "val_dataset")
        with self.profiler.measure_startup("parameter_checkpoint_load_ms", cuda=True):
            self._load_parameter_checkpoint()
        self._apply_train_mode(self.model)
        trainable_params = self._collect_trainable_params(self.model)
        if self.preflight is not None:
            from recoverwam.preflight import trainable_parameter_manifest

            self.preflight_trainable_manifest = trainable_parameter_manifest(self.model)
            logger.info(
                "Preflight trainable parameters: tensors=%d parameters=%d",
                self.preflight_trainable_manifest["num_tensors"],
                self.preflight_trainable_manifest["num_parameters"],
            )
            for parameter_name in self.preflight_trainable_manifest["names"]:
                logger.info("Preflight trainable: %s", parameter_name)
        with self.profiler.measure_startup("optimizer_init_ms", cuda=True):
            optimizer_parameters = trainable_params
            if self._uses_video_action_lr_groups():
                (video_params, action_params) = self._joint_parameter_groups(self.model)
                optimizer_parameters = [
                    {"params": video_params, "lr": self.video_learning_rate},
                    {"params": action_params, "lr": self.action_learning_rate},
                ]
            self.optimizer = torch.optim.AdamW(
                optimizer_parameters,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=self.optimizer_betas,
            )
        with self.profiler.measure_startup("dataloader_init_ms"):
            self.train_loader = self._build_loader(
                self.train_dataset, worker_init_fn=worker_init_fn
            )
        with self.profiler.measure_startup("scheduler_init_ms"):
            total_train_steps = self._estimate_total_train_steps()
            self.max_steps = total_train_steps
            self.scheduler = self._build_scheduler(
                scheduler_type=cfg.lr_scheduler_type,
                total_train_steps=total_train_steps,
                warmup_steps=self.warmup_steps,
            )
        self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.last_saved_step = None
        self.last_checkpoint_info = None
        self.checkpoint_root = os.path.join(self.output_dir, "checkpoints")
        self.weights_dir = os.path.join(self.checkpoint_root, "weights")
        self.state_dir = os.path.join(self.checkpoint_root, "state")
        self.eval_dir = os.path.join(self.output_dir, "eval")
        ensure_dir(self.output_dir)
        ensure_dir(self.checkpoint_root)
        ensure_dir(self.weights_dir)
        ensure_dir(self.state_dir)
        ensure_dir(self.eval_dir)
        with self.profiler.measure_startup("accelerator_prepare_ms", cuda=True):
            (self.model, self.optimizer, self.train_loader, self.scheduler) = (
                self.accelerator.prepare(
                    self.model, self.optimizer, self.train_loader, self.scheduler
                )
            )
        self.optimizer.zero_grad(set_to_none=True)
        self.wandb_run = None
        self._init_wandb()
        with self.profiler.measure_startup("resume_training_state_ms", cuda=True):
            self._resume_or_load_checkpoint()
        if self.lr_continuation:
            from .utils.lr_continuation import validate_continuation_state

            restored_lrs = validate_continuation_state(
                self.scheduler, self.optimizer, self.global_step, self.lr_continuation
            )
            logger.info("LR continuation restored: step=%d lrs=%s", self.global_step, restored_lrs)
        if self.lr_tail and self.resume:
            from .utils.lr_continuation import validate_joint_state

            validate_joint_state(
                self.scheduler, self.optimizer, self.global_step, cfg.warmup_steps, self.lr_tail
            )
        val_size = (
            len(self.val_dataset) if self.val_dataset is not None else len(self.train_dataset)
        )
        logger.info("Train/val dataset size: %d/%d", len(self.train_dataset), val_size)

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
            group=None if self.cfg.wandb.group in (None, "null", "") else str(self.cfg.wandb.group),
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
        self.train_sampler = ResumableEpochSampler(
            dataset=dataset,
            seed=self.seed,
            batch_size=self.batch_size,
            num_processes=self.accelerator.num_processes,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=worker_init_fn,
        )

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
            print(
                f"[dataset-check] {dataset_name} length mismatch across ranks after initialization:"
            )
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
        micro_steps_per_epoch = self._micro_steps_per_epoch()
        opt_steps_per_epoch = max(ceil(micro_steps_per_epoch / self.gradient_accumulation_steps), 1)
        return max(opt_steps_per_epoch * self.num_epochs, 1)

    def _micro_steps_per_epoch(self) -> int:
        num_processes = max(int(self.accelerator.num_processes), 1)
        global_batch_size = max(self.batch_size * num_processes, 1)
        return max(ceil(len(self.train_dataset) / global_batch_size), 1)

    def _build_scheduler(self, scheduler_type, total_train_steps: int, warmup_steps: int = 0):
        scheduler_type = str(scheduler_type).strip().lower()
        total_train_steps = max(int(total_train_steps), 1)
        tail = getattr(self, "lr_tail", None)
        if tail:
            from .utils.lr_continuation import joint_multiplier

            if scheduler_type != "cosine" or not self._uses_video_action_lr_groups():
                raise ValueError("lr_tail requires grouped Video/Action cosine training")
            if total_train_steps > int(tail["end_step"]):
                raise ValueError("max_steps exceeds the joint LR schedule")
            if self.lr_min_ratio != float(tail["start_ratio"]):
                raise ValueError("lr_min_ratio must match lr_tail.start_ratio")
            joint_multiplier(0, warmup_steps, tail)

            def joint_lr(step):
                return joint_multiplier(step, warmup_steps, tail)

            return LambdaLR(self.optimizer, lr_lambda=[joint_lr] * len(self.optimizer.param_groups))
        warmup_steps = min(max(int(warmup_steps), 0), total_train_steps - 1)
        continuation = getattr(self, "lr_continuation", None)
        if continuation:
            from .utils.lr_continuation import continuation_multiplier

            if scheduler_type != "cosine" or not self._uses_video_action_lr_groups():
                raise ValueError("LR continuation requires grouped Video/Action cosine training")
            if not int(continuation["start_step"]) < total_train_steps <= int(continuation["end_step"]):
                raise ValueError("max_steps must be inside the LR continuation interval")

            def continued_multiplier(step):
                return continuation_multiplier(step, continuation)

            return LambdaLR(
                self.optimizer,
                lr_lambda=[continued_multiplier] * len(self.optimizer.param_groups),
            )
        if self._uses_video_action_lr_groups() or bool(False):
            if scheduler_type not in {"cosine", "constant"}:
                raise ValueError(
                    f"Unsupported lr_scheduler_type: {scheduler_type}. Expected one of: ['cosine', 'constant']."
                )

            def lr_multiplier(step: int) -> float:
                step = max(int(step), 0)
                if warmup_steps > 0 and step < warmup_steps:
                    return float(step + 1) / float(warmup_steps)
                if scheduler_type == "constant":
                    return 1.0
                cosine_steps = max(total_train_steps - warmup_steps, 1)
                progress = min(max((step - warmup_steps) / cosine_steps, 0.0), 1.0)
                floor = float(getattr(self, "lr_min_ratio", 0.01))
                return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

            return LambdaLR(
                self.optimizer, lr_lambda=[lr_multiplier] * len(self.optimizer.param_groups)
            )
        remaining_steps = max(total_train_steps - warmup_steps, 1)
        if scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=remaining_steps,
                eta_min=self.learning_rate * float(getattr(self, "lr_min_ratio", 0.01)),
            )
        elif scheduler_type == "constant":
            main_scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=remaining_steps)
        else:
            raise ValueError(
                f"Unsupported lr_scheduler_type: {scheduler_type}. Expected one of: ['cosine', 'constant']."
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
            self.optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_steps]
        )

    def _estimate_eta(self):
        elapsed = max(time.perf_counter() - self.run_start_time, 1e-06)
        done_steps = max(self.global_step - self.run_start_step, 1)
        steps_per_sec = done_steps / elapsed
        remaining_steps = max(self.max_steps - self.global_step, 0)
        eta_seconds = int(remaining_steps / max(steps_per_sec, 1e-09))
        (eta_h, eta_rem) = divmod(eta_seconds, 3600)
        (eta_m, eta_s) = divmod(eta_rem, 60)
        return (f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}", steps_per_sec)

    def _resume_or_load_checkpoint(self):
        if self.resume_ckpt:
            logger.info(
                "Using parameter-only initialization: global_step=%d learning_rate=%.3e warmup_steps=%d; optimizer and scheduler were created from the loaded parameters",
                self.global_step,
                self.learning_rate,
                self.warmup_steps,
            )
            return
        resume = self.resume
        if not resume:
            return
        resume_path = Path(str(resume))
        if resume_path.is_dir():
            logger.info("Resuming full training state from directory: %s", resume)
            self.load_training_state(str(resume_path))
            return
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        logger.info("Loading weight checkpoint only: %s", resume)
        self.accelerator.unwrap_model(self.model).load_checkpoint(str(resume_path), optimizer=None)
        logger.warning(
            "Loaded .pt weights only; optimizer/scheduler/step were not restored under ZeRO2."
        )

    def _load_parameter_checkpoint(self):
        if not self.resume_ckpt:
            return
        checkpoint_path = Path(str(self.resume_ckpt))
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Parameter checkpoint not found: {self.resume_ckpt}")
        logger.info(
            "Initializing model parameters before optimizer/DeepSpeed setup from checkpoint: %s",
            self.resume_ckpt,
        )
        load_kwargs = {}
        if not self.resume_ckpt_load_video:
            if self.checkpoint_mode != "action_only" or not self.frozen_video_checkpoint:
                raise ValueError(
                    "resume_ckpt_load_video=false requires checkpoint_mode=action_only and frozen_video_checkpoint."
                )
            load_kwargs = {
                "video_checkpoint_override": self.frozen_video_checkpoint,
                "load_video_from_action_checkpoint": False,
            }
        self.model.load_checkpoint(str(checkpoint_path), optimizer=None, **load_kwargs)
        logger.info("Model parameter initialization complete; optimizer state will start fresh.")

    def _set_dit_only_train_mode(self):
        logger.info("Setting training mode to %s.", self.train_mode)
        model = self.accelerator.unwrap_model(self.model)
        self._apply_train_mode(model)

    def _apply_train_mode(self, model):
        model.eval().requires_grad_(False)
        set_train_mode = getattr(model, "set_train_mode", None)
        set_train_wan = getattr(model, "set_train_wan", None)
        if callable(set_train_mode):
            set_train_mode("joint")
        elif callable(set_train_wan):
            set_train_wan("joint" != "action_only")
        else:
            setattr(model, "train_wan", "joint" != "action_only")
        model.dit.train().requires_grad_(True)
        proprio_encoder = getattr(model, "proprio_encoder", None)
        if proprio_encoder is not None:
            proprio_encoder.train().requires_grad_(True)

    def _collect_trainable_params(self, model):
        params = list(model.dit.parameters())
        proprio_encoder = getattr(model, "proprio_encoder", None)
        if proprio_encoder is not None:
            params.extend(proprio_encoder.parameters())
        params = list(dict.fromkeys(params))
        if not params:
            raise ValueError("No trainable parameters selected for optimizer.")
        if not all((parameter.requires_grad for parameter in params)):
            raise ValueError(
                f"Selected optimizer parameters include frozen tensors in train_mode={'joint'}."
            )
        num_params = sum((p.numel() for p in params))
        logger.info(
            "Selected %.3fB trainable parameters for optimizer (train_mode=%s).",
            num_params / 1000000000.0,
            self.train_mode,
        )
        return params

    def _uses_video_action_lr_groups(self) -> bool:
        return "joint" == "alternating" or bool(getattr(self, "joint_learning_rate_groups", False))

    def _joint_parameter_groups(self, model):
        if not self._uses_video_action_lr_groups():
            raise ValueError(
                "Video/action parameter groups require train_mode=alternating or `joint_learning_rates`."
            )
        video_expert = getattr(model, "video_expert", None)
        if video_expert is None:
            raise ValueError("train_mode=alternating requires model.video_expert.")
        video_params = list(dict.fromkeys(video_expert.parameters()))
        action_params = [
            parameter
            for expert in self._get_action_expert_modules(model)
            for parameter in expert.parameters()
        ]
        proprio_encoder = getattr(model, "proprio_encoder", None)
        if proprio_encoder is not None:
            action_params.extend(proprio_encoder.parameters())
        action_params = list(dict.fromkeys(action_params))
        overlap = {id(parameter) for parameter in video_params} & {
            id(parameter) for parameter in action_params
        }
        if overlap:
            raise ValueError("Joint optimizer parameter groups overlap.")
        selected = {id(parameter) for parameter in video_params + action_params}
        expected = {id(parameter) for parameter in self._collect_trainable_params(model)}
        if selected != expected:
            raise ValueError("Joint optimizer groups do not cover every trainable parameter.")
        logger.info(
            "Video/action optimizer groups (%s): video=%.3fB lr=%.3e action=%.3fB lr=%.3e",
            "joint",
            sum((parameter.numel() for parameter in video_params)) / 1000000000.0,
            self.video_learning_rate,
            sum((parameter.numel() for parameter in action_params)) / 1000000000.0,
            self.action_learning_rate,
        )
        return (video_params, action_params)

    @staticmethod
    def _normalize_optimizer_betas(values):
        try:
            betas = tuple((float(value) for value in values))
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"`optimizer_betas` must contain two numeric values in [0,1), got {values}"
            ) from error
        if len(betas) != 2 or not all((0.0 <= value < 1.0 for value in betas)):
            raise ValueError(f"`optimizer_betas` must contain two values in [0,1), got {betas}")
        return betas

    @staticmethod
    def _get_action_expert_modules(model):
        """Every Action expert that training must update.

        A timestep-routed model can own more than one
        expert; ``model.action_expert`` only points at the currently selected
        one, so parameter collection, ``requires_grad`` and checkpointing all
        go through this list instead.
        """
        modules_fn = getattr(model, "action_expert_modules", None)
        if callable(modules_fn):
            modules = list(modules_fn().values())
            if not modules:
                raise ValueError("`action_expert_modules()` returned no experts.")
            return modules
        return [JointTrainer._get_action_expert(model)]

    @staticmethod
    def _get_action_expert(model):
        action_expert = getattr(model, "action_expert", None)
        if action_expert is not None:
            return action_expert
        dit = getattr(model, "dit", None)
        mixtures = getattr(dit, "mixtures", None)
        if mixtures is not None and "action" in mixtures:
            return mixtures["action"]
        raise ValueError(
            "`train_mode=action_only` requires a model with `action_expert` or `dit.mixtures['action']`."
        )

    def _evaluate_and_log(self):
        metrics = self.evaluate()
        self.accelerator.wait_for_everyone()
        if metrics is not None and self.accelerator.is_main_process:
            numeric_metrics = {
                key: float(value)
                for (key, value) in metrics.items()
                if isinstance(value, (int, float))
            }
            artifact_metrics = {
                key: value for (key, value) in metrics.items() if isinstance(value, str)
            }
            description = f"[eval] step={self.global_step}"
            if numeric_metrics:
                description += " " + " ".join(
                    (f"{key}={value:.6g}" for (key, value) in sorted(numeric_metrics.items()))
                )
            if artifact_metrics:
                description += " artifacts=" + ",".join(sorted(artifact_metrics.values()))
            logger.info(description)
            self._wandb_log({f"eval/{key}": value for (key, value) in numeric_metrics.items()})

    @torch.no_grad()
    def evaluate(self):
        if self.evaluator is not None and getattr(self.evaluator, "distributed", False):
            if self._deepspeed_zero_stage() >= 3:
                raise ValueError("distributed evaluators require DeepSpeed ZeRO stage < 3")
            model = self.accelerator.unwrap_model(self.model)
            modes = [(module, module.training) for module in model.modules()]
            model.eval()
            try:
                with self.accelerator.autocast():
                    local = self.evaluator.evaluate_local(
                        model,
                        rank=self.accelerator.process_index,
                        world_size=self.accelerator.num_processes,
                    )
            finally:
                for module, training in modes:
                    module.training = training
            summed = self.accelerator.reduce(
                local.to(device=self.accelerator.device, dtype=torch.float64), reduction="sum"
            )
            metrics = None
            if self.accelerator.is_main_process:
                metrics = self.evaluator.summarize(summed, self.global_step)
            self.accelerator.wait_for_everyone()
            return metrics
        return {}

    def _deepspeed_zero_stage(self) -> int:
        plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
        if plugin is None:
            return 0
        config = getattr(plugin, "deepspeed_config", {}) or {}
        return int(config.get("zero_optimization", {}).get("stage", 0))

    @staticmethod
    def _clone_state_dict_to_cpu(state_dict):
        return {
            key: value.detach().to(device="cpu").contiguous().clone()
            for (key, value) in state_dict.items()
        }

    def _save_weights_checkpoint(self, step_tag: str, *, state_dict_overrides=None):
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
        model = self.accelerator.unwrap_model(self.model)
        state_dict_overrides = None
        ckpt_path = None
        if self.accelerator.is_main_process:
            ckpt_path = self._save_weights_checkpoint(
                step_tag=step_tag, state_dict_overrides=state_dict_overrides
            )
        self.accelerator.wait_for_everyone()
        state_path = os.path.join(self.state_dir, step_tag)
        ensure_dir(state_path)
        self.accelerator.save_state(output_dir=state_path)
        if self.accelerator.is_main_process:
            self._save_trainer_state(state_path)
        self.accelerator.wait_for_everyone()
        return {"weights_path": ckpt_path, "state_path": state_path}

    def _save_once(self, step):
        step = int(step)
        if self.last_saved_step == step:
            return self.last_checkpoint_info
        timings_ms = {}
        with self.profiler.measure(timings_ms, "checkpoint_ms", cuda=True):
            checkpoint_info = self.save_checkpoint()
        if "checkpoint_ms" in timings_ms:
            self.profiler.record_event(
                "checkpoint_ms", value_ms=timings_ms["checkpoint_ms"], step=step
            )
        self.last_saved_step = step
        self.last_checkpoint_info = checkpoint_info
        return checkpoint_info

    def _finalize_training_profile(self, *, status: str) -> None:
        if not self.profiler.enabled:
            return
        self.profiler.record_event(
            "run_elapsed_ms", value_ms=self.profiler.elapsed_ms(), step=self.global_step
        )
        rank_path = self.profiler.write_rank_report(status=status, final_step=self.global_step)
        logger.info("[profile] rank=%d report=%s", self.profiler.rank, rank_path)
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            summary_path = self.profiler.write_global_report()
            logger.info("[profile] summary=%s", summary_path)
        self.accelerator.wait_for_everyone()

    def _reset_preflight_memory_stats(self):
        if self.preflight is not None and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.accelerator.device)

    def _finalize_preflight(self, checkpoint_info):
        if self.preflight is None or self.preflight_finalized:
            return
        if self.global_step != self.preflight["expected_steps"]:
            raise RuntimeError(
                f"Preflight optimizer-step mismatch: expected {self.preflight['expected_steps']}, got {self.global_step}"
            )
        from recoverwam.preflight import write_json_atomic

        model = self.accelerator.unwrap_model(self.model)
        self.accelerator.wait_for_everyone()
        weights_path = os.path.join(self.weights_dir, f"step_{self.global_step:06d}.pt")
        if self.accelerator.is_main_process:
            model.load_checkpoint(weights_path)
        self.accelerator.wait_for_everyone()
        if torch.cuda.is_available():
            peak_allocated = torch.cuda.max_memory_allocated(self.accelerator.device)
            peak_reserved = torch.cuda.max_memory_reserved(self.accelerator.device)
            total_memory = torch.cuda.get_device_properties(self.accelerator.device).total_memory
        else:
            peak_allocated = peak_reserved = total_memory = 0
        local_memory = torch.tensor(
            [[peak_allocated, peak_reserved, total_memory]],
            device=self.accelerator.device,
            dtype=torch.int64,
        )
        gathered_memory = self.accelerator.gather(local_memory).reshape(-1, 3)
        memory_by_rank = [
            {
                "rank": rank,
                "peak_allocated_bytes": int(values[0]),
                "peak_reserved_bytes": int(values[1]),
                "total_device_bytes": int(values[2]),
            }
            for (rank, values) in enumerate(gathered_memory.cpu().tolist())
        ]
        allowed_fraction = 1.0 - self.preflight["memory_headroom_fraction"]
        headroom_pass = all(
            (
                item["total_device_bytes"] == 0
                or item["peak_reserved_bytes"] <= item["total_device_bytes"] * allowed_fraction
                for item in memory_by_rank
            )
        )
        if self.accelerator.is_main_process:
            payload = {
                "stage": self.preflight["stage"],
                "optimizer_steps": int(self.global_step),
                "world_size": int(self.accelerator.num_processes),
                "micro_batch_size": int(self.batch_size),
                "global_batch_size": int(
                    self.batch_size
                    * self.accelerator.num_processes
                    * self.gradient_accumulation_steps
                ),
                "gradient_accumulation_steps": int(self.gradient_accumulation_steps),
                "observed_contract": self.preflight_contract,
                "trainable": self.preflight_trainable_manifest,
                "final_loss": self.preflight_last_loss,
                "final_loss_metrics": self.preflight_last_loss_metrics,
                "memory_by_rank": memory_by_rank,
                "memory_headroom_fraction": self.preflight["memory_headroom_fraction"],
                "headroom_pass": bool(headroom_pass),
                "checkpoint_round_trip": True,
                "weights_path": weights_path,
                "state_path": checkpoint_info["state_path"],
                "formal_training_started": False,
            }
            write_json_atomic(payload, self.preflight["output_file"])
            logger.info("Wrote preflight report: %s", self.preflight["output_file"])
        self.preflight_finalized = True

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
                micro_steps_per_epoch = self._micro_steps_per_epoch()
                if self.batch_in_epoch >= micro_steps_per_epoch:
                    (completed_epochs, self.batch_in_epoch) = divmod(
                        self.batch_in_epoch, micro_steps_per_epoch
                    )
                    self.epoch += completed_epochs
                    logger.info(
                        "Normalized end-of-epoch resume boundary: epoch=%d batch_in_epoch=%d micro_steps_per_epoch=%d",
                        self.epoch,
                        self.batch_in_epoch,
                        micro_steps_per_epoch,
                    )
                self.train_sampler.set_epoch_offset(self.epoch)
                self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)
                logger.info(
                    "Restored dataloader progress: epoch=%d batch_in_epoch=%d sample_offset=%d",
                    self.epoch,
                    self.batch_in_epoch,
                    self.batch_in_epoch
                    * self.batch_size
                    * self.accelerator.num_processes
                    * self.gradient_accumulation_steps,
                )
            else:
                self.epoch = 0
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                logger.warning(
                    "State file does not contain `epoch`/`batch_in_epoch`; optimizer/scheduler were restored, but dataloader progress resume is skipped."
                )
            self.accelerator.wait_for_everyone()
            return
        match = re.search("step[_-](\\d+)$", str(state_dir).rstrip("/"))
        if match:
            self.global_step = int(match.group(1))
        else:
            self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.train_sampler.clear_resume_batch_offset()
        self.accelerator.wait_for_everyone()
        logger.info(
            "Loaded accelerate training state from %s at step=%d", state_dir, self.global_step
        )
        logger.warning(
            "State file `%s` is missing; dataloader progress resume is skipped.", state_file
        )

    def train(self):
        self._set_dit_only_train_mode()
        self._reset_preflight_memory_stats()
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        if self.max_steps is None:
            raise ValueError(
                "`max_steps` must be set before entering the while-step training loop."
            )
        logger.info("Starting training with max_steps=%d.", self.max_steps)
        if hasattr(self.train_dataset, "set_epoch"):
            self.train_dataset.set_epoch(self.epoch)
        if (
            bool(self.cfg.get("eval_at_start", False))
            and self.eval_every > 0
            and (self.val_dataset is not None or self.evaluator is not None)
        ):
            self._evaluate_and_log()
        data_iter = iter(self.train_loader)
        self.run_start_step = self.global_step
        self.run_start_time = time.perf_counter()
        profile_step_timings: dict[str, float] = {}
        profile_step_wall_started = None
        profile_first_batch_recorded = False
        profile_first_optimizer_step_recorded = False
        while self.global_step < self.max_steps:
            profile_collecting = self.profiler.wants_step_sample(self.global_step)
            profile_micro_timings: dict[str, float] = {}
            profile_candidate_wall_start = (
                time.perf_counter()
                if profile_collecting and profile_step_wall_started is None
                else None
            )
            try:
                with self.profiler.measure(
                    profile_micro_timings, "data_wait_ms", active=profile_collecting
                ):
                    sample = next(data_iter)
                self.batch_in_epoch += 1
            except StopIteration:
                self.epoch += 1
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                if hasattr(self.train_dataset, "set_epoch"):
                    self.train_dataset.set_epoch(self.epoch)
                data_iter = iter(self.train_loader)
                continue
            if profile_collecting:
                if profile_step_wall_started is None:
                    profile_step_wall_started = profile_candidate_wall_start
                for key, value in profile_micro_timings.items():
                    profile_step_timings[key] = profile_step_timings.get(key, 0.0) + value
                if not profile_first_batch_recorded:
                    self.profiler.record_event(
                        "time_to_first_batch_ms",
                        value_ms=self.profiler.elapsed_ms(),
                        step=self.global_step,
                    )
                    profile_first_batch_recorded = True
            with self.accelerator.accumulate(self.model):
                train_model = (
                    self.model
                    if hasattr(self.model, "training_loss")
                    else self.accelerator.unwrap_model(self.model)
                )
                training_stage = None
                with self.profiler.measure(
                    profile_step_timings, "forward_ms", cuda=True, active=profile_collecting
                ):
                    with self.accelerator.autocast():
                        if training_stage is None:
                            (loss, loss_dict) = train_model.training_loss(sample)
                        else:
                            (loss, loss_dict) = train_model.training_loss(
                                sample, training_stage=training_stage
                            )
                    if not bool(torch.isfinite(loss).all().item()):
                        raise FloatingPointError(
                            f"Non-finite training loss at step {self.global_step}: {loss}"
                        )
                    if self.preflight is not None:
                        from recoverwam.preflight import validate_batch_contract

                        if self.preflight_contract is None:
                            self.preflight_contract = validate_batch_contract(
                                sample,
                                self.preflight["stage"],
                                video_size=self.preflight["video_size"],
                                **self.preflight["contract_kwargs"],
                            )
                        self.preflight_last_loss = float(loss.detach().float().item())
                        self.preflight_last_loss_metrics = {
                            str(key): float(value) for (key, value) in loss_dict.items()
                        }
                with self.profiler.measure(
                    profile_step_timings, "backward_ms", cuda=True, active=profile_collecting
                ):
                    self.accelerator.backward(loss)
                    if self.preflight is not None:
                        from recoverwam.preflight import assert_finite_gradients

                        assert_finite_gradients(self.model)
                if self.accelerator.sync_gradients:
                    with self.profiler.measure(
                        profile_step_timings, "optimizer_ms", cuda=True, active=profile_collecting
                    ):
                        grad_norm = self.accelerator.clip_grad_norm_(
                            self.model.parameters(), self.max_grad_norm
                        )
                        self.optimizer.step()
                        if not self.accelerator.optimizer_step_was_skipped:
                            self.scheduler.step()
                        self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    with self.profiler.measure(
                        profile_step_timings,
                        "metric_reduce_ms",
                        cuda=True,
                        active=profile_collecting,
                    ):
                        global_loss = float(
                            self.accelerator.gather(loss.detach().float().reshape(1)).mean().item()
                        )
                        global_loss_metrics = {}
                        for key, value in loss_dict.items():
                            metric_tensor = torch.tensor(
                                float(value), device=loss.device, dtype=torch.float32
                            ).reshape(1)
                            global_loss_metrics[key] = float(
                                self.accelerator.gather(metric_tensor).mean().item()
                            )
                        global_grad_norm = None
                        if grad_norm is not None:
                            grad_norm_tensor = torch.as_tensor(
                                grad_norm, device=loss.device, dtype=torch.float32
                            ).reshape(1)
                            global_grad_norm = float(
                                self.accelerator.gather(grad_norm_tensor).mean().item()
                            )
                    if profile_collecting and profile_step_wall_started is not None:
                        profile_step_timings["total_step_ms"] = (
                            time.perf_counter() - profile_step_wall_started
                        ) * 1000.0
                        self.profiler.record_step(
                            step=self.global_step, epoch=self.epoch, timings_ms=profile_step_timings
                        )
                        if not profile_first_optimizer_step_recorded:
                            self.profiler.record_event(
                                "time_to_first_optimizer_step_ms",
                                value_ms=self.profiler.elapsed_ms(),
                                step=self.global_step,
                            )
                            profile_first_optimizer_step_recorded = True
                        if (
                            self.profiler.log_every > 0
                            and len(self.profiler.step_samples) % self.profiler.log_every == 0
                        ):
                            local_total = self.profiler.step_samples[-1].get("total_step_ms", 0.0)
                            logger.info(
                                "[profile] rank=%d collected=%d/%d latest_step_ms=%.2f",
                                self.profiler.rank,
                                len(self.profiler.step_samples),
                                self.profiler.collect_steps,
                                local_total,
                            )
                        profile_step_timings = {}
                        profile_step_wall_started = None
                    current_lr = float(self.optimizer.param_groups[0]["lr"])
                    current_action_lr = (
                        float(self.optimizer.param_groups[1]["lr"])
                        if self._uses_video_action_lr_groups()
                        else current_lr
                    )
                    if (
                        self.log_every > 0
                        and self.global_step % self.log_every == 0
                        and self.accelerator.is_main_process
                    ):
                        (eta_str, steps_per_sec) = self._estimate_eta()
                        description = "[train] epoch=%d step=%d/%d loss=%.4f " % (
                            self.epoch,
                            self.global_step,
                            self.max_steps,
                            global_loss,
                        )
                        if training_stage is not None:
                            description += f"stage={training_stage} "
                        if global_loss_metrics:
                            detail_str = " ".join(
                                [f"{k}={v:.4f}" for (k, v) in sorted(global_loss_metrics.items())]
                            )
                            description += detail_str + " "
                        if global_grad_norm is not None:
                            description += "grad_norm=%.4f " % global_grad_norm
                        description += "lr=%.2e " % current_lr
                        if self._uses_video_action_lr_groups():
                            description += "action_lr=%.2e " % current_action_lr
                        description += "speed=%.2f step/s, %.2f samples/s eta=%s" % (
                            steps_per_sec,
                            steps_per_sec * self.batch_size * self.accelerator.num_processes,
                            eta_str,
                        )
                        logger.info(description)
                        wandb_payload = {
                            "train/loss": global_loss,
                            "train/lr": current_lr,
                            "performance/steps_per_sec": steps_per_sec,
                            "performance/samples_per_sec": steps_per_sec
                            * self.batch_size
                            * self.accelerator.num_processes,
                        }
                        if self._uses_video_action_lr_groups():
                            wandb_payload["train/video_lr"] = current_lr
                            wandb_payload["train/action_lr"] = current_action_lr
                        if global_grad_norm is not None:
                            wandb_payload["train/grad_norm"] = global_grad_norm
                        for key, value in global_loss_metrics.items():
                            wandb_payload[f"train/{key}"] = value
                        self._wandb_log(wandb_payload)
                    if (
                        self.eval_every > 0
                        and (self.val_dataset is not None or self.evaluator is not None)
                        and (self.global_step % self.eval_every == 0)
                    ):
                        self._evaluate_and_log()
                    if self.save_every > 0 and self.global_step % self.save_every == 0:
                        ckpt_info = self._save_once(self.global_step)
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[ckpt] step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )
                    if self.global_step >= self.max_steps:
                        if self.save_final_checkpoint:
                            ckpt_info = self._save_once(self.global_step)
                            self._finalize_preflight(ckpt_info)
                            if self.accelerator.is_main_process:
                                logger.info(
                                    "[done] max_steps reached step=%d weights=%s state=%s",
                                    self.global_step,
                                    ckpt_info["weights_path"],
                                    ckpt_info["state_path"],
                                )
                        elif self.accelerator.is_main_process:
                            logger.info(
                                "[done] max_steps reached step=%d final checkpoint disabled",
                                self.global_step,
                            )
                        self._finalize_training_profile(status="completed")
                        return
        if self.save_final_checkpoint:
            ckpt_info = self._save_once(self.global_step)
            self._finalize_preflight(ckpt_info)
            if self.accelerator.is_main_process:
                logger.info(
                    "[done] training finished step=%d weights=%s state=%s",
                    self.global_step,
                    ckpt_info["weights_path"],
                    ckpt_info["state_path"],
                )
        elif self.accelerator.is_main_process:
            logger.info(
                "[done] training finished step=%d final checkpoint disabled", self.global_step
            )
        self._finalize_training_profile(status="completed")
