import gc
import json
import os
import re
from math import ceil
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist
from accelerate import Accelerator
from omegaconf import DictConfig
from PIL import Image
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from .cache import CacheRuntimeBindings
from .training.checkpoint import load_wam_checkpoint, save_wam_checkpoint
from .training.distributed import deepspeed_zero_stage
from .training.metrics import gather_mean_scalars
from .training.module import WAMTrainingModule
from .utils.fs import ensure_dir
from .utils.logging_config import get_logger
from .utils.pytorch_utils import set_global_seed
from .utils.samplers import EpisodeShardSampler, ResumableEpochSampler
from .utils.video_io import save_mp4
from .utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim

logger = get_logger(__name__)


class WAMTrainer:
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset=None,
        *,
        cfg: DictConfig,
        accelerator: Accelerator | None = None,
        cache_runtime: CacheRuntimeBindings | None = None,
    ):
        self.model = WAMTrainingModule(model)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cache_runtime = cache_runtime or CacheRuntimeBindings.disabled()
        self.cfg = cfg
        self.output_dir = str(cfg.output_dir)
        self.learning_rate = float(cfg.learning_rate)
        configured_min_learning_rate = cfg.get("min_learning_rate", None)
        self.min_learning_rate_is_explicit = configured_min_learning_rate is not None
        self.min_learning_rate = float(
            configured_min_learning_rate
            if self.min_learning_rate_is_explicit
            else self.learning_rate * 0.01
        )
        if not 0.0 < self.min_learning_rate <= self.learning_rate:
            raise ValueError(
                "min_learning_rate must satisfy 0 < min_learning_rate <= learning_rate, "
                f"got {self.min_learning_rate} and {self.learning_rate}."
            )
        self.vlm_learning_rate = float(
            cfg.get("vlm_learning_rate", min(self.learning_rate, 5.0e-5))
        )
        self.weight_decay = float(cfg.weight_decay)
        self.batch_size = int(cfg.batch_size)
        self.num_workers = int(cfg.num_workers)
        self.pin_memory = bool(cfg.get("pin_memory", torch.cuda.is_available()))
        self.persistent_workers = bool(cfg.get("persistent_workers", True))
        prefetch_factor = cfg.get("prefetch_factor", 2)
        self.prefetch_factor = None if prefetch_factor is None else int(prefetch_factor)
        self.num_epochs = int(cfg.num_epochs)
        max_steps = cfg.max_steps
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.log_every = int(cfg.log_every)
        self.save_every = int(cfg.save_every)
        self.eval_every = int(cfg.eval_every)
        self.eval_num_inference_steps = int(cfg.eval_num_inference_steps)
        self.eval_loss_samples_per_rank = max(
            1, int(cfg.get("eval_loss_samples_per_rank", 2))
        )
        eval_max_videos = cfg.get("eval_max_videos", None)
        self.eval_max_videos = None if eval_max_videos in (None, "", "null") else int(eval_max_videos)
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

        self.accelerator = accelerator or Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            step_scheduler_with_optimizer=False,
        )
        
        deepspeed_plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
        deepspeed_config = getattr(deepspeed_plugin, "deepspeed_config", {}) or {}
        zero_stage = deepspeed_config.get("zero_optimization", {}).get("stage", "disabled")
        logger.info(
            "Accelerate training: distributed_type=%s zero_stage=%s world_size=%d process_index=%d cfg_mixed_precision=%s accelerator_mixed_precision=%s micro_batch=%d grad_accum=%d effective_global_batch=%d grad_clip=%.4f",
            self.accelerator.distributed_type,
            zero_stage,
            self.accelerator.num_processes,
            self.accelerator.process_index,
            self.mixed_precision,
            self.accelerator.mixed_precision,
            self.batch_size,
            self.gradient_accumulation_steps,
            self.batch_size * self.accelerator.num_processes * self.gradient_accumulation_steps,
            self.max_grad_norm,
        )
        logger.debug("Using accelerator.device=%s", self.accelerator.device)
        worker_init_fn = set_global_seed(self.seed, get_worker_init_fn=True)
        self._assert_dataset_length_consistent(self.train_dataset, "train_dataset")
        if self.val_dataset is not None:
            self._assert_dataset_length_consistent(self.val_dataset, "val_dataset")

        # Freeze non-trainable modules before optimizer/deepspeed initializes optimizer state.
        self._apply_trainable_modules_mode(model)
        self._weight_resume_loaded = self._load_weight_checkpoint_before_prepare(model)
        if hasattr(model, "trainable_parameters"):
            trainable_params = list(model.trainable_parameters())
        else:
            trainable_params = list(model.mot.parameters())
            proprio_encoder = getattr(model, "proprio_encoder", None)
            if proprio_encoder is not None:
                trainable_params.extend(list(proprio_encoder.parameters()))
        self.grad_clip_params = trainable_params
        vlm_param_ids = {
            id(param)
            for name, param in model.named_parameters()
            if param.requires_grad and name.startswith("understanding.vlm.model.")
        }
        vlm_trainable_params = [
            param for param in trainable_params if id(param) in vlm_param_ids
        ]
        main_trainable_params = [
            param for param in trainable_params if id(param) not in vlm_param_ids
        ]
        optimizer_param_groups = [
            {
                "params": main_trainable_params,
                "lr": self.learning_rate,
                "name": "wam",
            }
        ]
        if vlm_trainable_params:
            optimizer_param_groups.append(
                {
                    "params": vlm_trainable_params,
                    "lr": self.vlm_learning_rate,
                    "name": "vlm",
                }
            )
        self.optimizer = torch.optim.AdamW(
            optimizer_param_groups,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )
        logger.info(
            "Optimizer peak learning rates: wam=%.3e vlm=%s",
            self.learning_rate,
            (
                f"{self.vlm_learning_rate:.3e}"
                if vlm_trainable_params
                else "disabled"
            ),
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

        self.checkpoint_root = os.path.join(self.output_dir, "checkpoints")
        self.weights_dir = os.path.join(self.checkpoint_root, "weights")
        self.state_dir = os.path.join(self.checkpoint_root, "state")
        self.eval_dir = os.path.join(self.output_dir, "eval")
        self.max_weight_checkpoints = max(1, int(cfg.get("max_weight_checkpoints", 5)))

        ensure_dir(self.output_dir)
        ensure_dir(self.checkpoint_root)
        ensure_dir(self.weights_dir)
        ensure_dir(self.state_dir)
        ensure_dir(self.eval_dir)

        self._log_trainable_parameter_summary(model)

        placement = self.cache_runtime.device_placement("train")
        prepare_args = (
            self.model,
            self.optimizer,
            self.train_loader,
            self.scheduler,
        )
        if placement.input_builder_owns_h2d or bool(cfg.model.get("understanding", {}).get("defer_image_transfer", False)):
            is_deepspeed = getattr(
                self.accelerator.distributed_type, "name", ""
            ) == "DEEPSPEED"
            if is_deepspeed:
                previous_device_placement = self.accelerator.device_placement
                self.accelerator.device_placement = False
                try:
                    prepared = self.accelerator.prepare(*prepare_args)
                finally:
                    self.accelerator.device_placement = previous_device_placement
            else:
                prepared = self.accelerator.prepare(
                    *prepare_args,
                    device_placement=[True, True, False, True],
                )
        else:
            prepared = self.accelerator.prepare(*prepare_args)
        self.model, self.optimizer, self.train_loader, self.scheduler = prepared
        self._unwrap_core_model().set_runtime_device(self.accelerator.device)
        self.optimizer.zero_grad(set_to_none=True)
        self.wandb_run = None
        self._init_wandb()
        self._resume_or_load_checkpoint()

        val_size = len(self.val_dataset) if self.val_dataset is not None else len(self.train_dataset)
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
        logger.debug(
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

    def _release_runtime_memory(self) -> None:
        try:
            import pyarrow as pa

            pa.default_memory_pool().release_unused()
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def _build_loader(self, dataset, worker_init_fn=None):
        dataset = self.cache_runtime.wrap_dataset("train", dataset)
        preserve_order = bool(getattr(dataset, "prefer_sequential_indices", False))
        if preserve_order:
            logger.info("DataLoader preserves grouped indices for random decoded-segment sampling.")
        locality_cfg = self.cfg.get("cache_locality_sampling", None)
        locality_enabled = bool(
            locality_cfg is not None and locality_cfg.get("enabled", False)
        )
        if locality_enabled:
            if preserve_order:
                raise ValueError(
                    "cache_locality_sampling cannot be combined with a Dataset "
                    "that requires sequential sampler order."
                )
            episode_ranges = getattr(dataset, "sample_episode_ranges", None)
            if episode_ranges is None:
                raise TypeError(
                    "cache_locality_sampling requires Dataset.sample_episode_ranges."
                )
            if self.cache_runtime.binding("train").enabled:
                shard_ranges = self.cache_runtime.contiguous_shard_sample_ranges("train")
            else:
                # No latent cache: keep the batch-local episode layout but treat
                # the whole dataset as a single shard, so segments are episodes.
                shard_ranges = ((0, len(dataset)),)
                logger.info(
                    "Cache-locality sampler running without a latent cache; "
                    "using episode-only locality."
                )
            self.train_sampler = EpisodeShardSampler(
                dataset=dataset,
                seed=self.seed,
                batch_size=self.batch_size,
                num_processes=self.accelerator.num_processes,
                episode_ranges=episode_ranges,
                shard_ranges=shard_ranges,
                episodes_per_batch=int(locality_cfg.get("episodes_per_batch", 4)),
            )
            logger.info(
                "Cache-locality sampler enabled: episodes_per_rank_batch=%d "
                "samples_per_episode=%d structured_samples=%d/%d (%.2f%%) "
                "tail_samples=%d cache_shards=%d",
                self.train_sampler.episodes_per_batch,
                self.train_sampler.samples_per_episode,
                self.train_sampler.structured_sample_count,
                len(dataset),
                100.0 * self.train_sampler.structured_sample_count / len(dataset),
                self.train_sampler.tail_sample_count,
                len(shard_ranges),
            )
        else:
            self.train_sampler = ResumableEpochSampler(
                dataset=dataset,
                seed=self.seed,
                batch_size=self.batch_size,
                num_processes=self.accelerator.num_processes,
                shuffle=not preserve_order,
            )
        kwargs = {}
        if self.num_workers > 0:
            kwargs["persistent_workers"] = self.persistent_workers
            if self.prefetch_factor is not None:
                kwargs["prefetch_factor"] = self.prefetch_factor
        collate_fn = self.cache_runtime.wrap_collate(
            "train", getattr(dataset, "collate_fn", None)
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn,
            **kwargs,
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

    def _build_scheduler(self, scheduler_type, total_train_steps: int, warmup_steps: int = 0):
        scheduler_type = str(scheduler_type).strip().lower()
        total_train_steps = max(int(total_train_steps), 1)
        warmup_steps = min(max(int(warmup_steps), 0), total_train_steps - 1)

        remaining_steps = max(total_train_steps - warmup_steps, 1)
        if scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=remaining_steps,
                eta_min=self.min_learning_rate,
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
            start_factor=(
                self.min_learning_rate / self.learning_rate
                if self.min_learning_rate_is_explicit
                else 1.0 / warmup_steps
            ),
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )
    
    def _estimate_eta(self):
        start_step, start_time = self.speed_history[0]
        end_step, end_time = self.speed_history[-1]
        elapsed = max(end_time - start_time, 1e-6)
        done_steps = max(end_step - start_step, 1)
        steps_per_sec = done_steps / elapsed
        remaining_steps = max(self.max_steps - self.global_step, 0)
        eta_seconds = int(remaining_steps / max(steps_per_sec, 1e-9))
        eta_h, eta_rem = divmod(eta_seconds, 3600)
        eta_m, eta_s = divmod(eta_rem, 60)
        return f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}", steps_per_sec

    @staticmethod
    def _step_number_from_state_dir(path: Path) -> int:
        match = re.search(r"step[_-](\d+)$", path.name)
        return int(match.group(1)) if match else -1

    def _resolve_training_state_dir(self, resume_path: Path) -> Path | None:
        if not resume_path.is_dir():
            return None
        if (resume_path / "trainer_state.json").is_file():
            return resume_path

        candidate_roots = [
            resume_path / "checkpoints" / "state",
            resume_path / "state",
            resume_path,
        ]
        seen_roots = set()
        for state_root in candidate_roots:
            if state_root in seen_roots or not state_root.is_dir():
                continue
            seen_roots.add(state_root)
            latest_dir = state_root / "latest"
            if latest_dir.is_dir():
                return latest_dir

            step_dirs = [p for p in state_root.iterdir() if p.is_dir() and self._step_number_from_state_dir(p) >= 0]
            if step_dirs:
                return sorted(step_dirs, key=self._step_number_from_state_dir)[-1]
        return None

    def _resume_or_load_checkpoint(self):
        resume = self.resume
        if not resume:
            return
        if self._weight_resume_loaded:
            return

        resume_text = str(resume).strip()
        resume_path = Path(self.output_dir) if resume_text.lower() in {"latest", "auto"} else Path(resume_text)
        if resume_path.is_dir():
            state_dir = self._resolve_training_state_dir(resume_path)
            if state_dir is None:
                raise FileNotFoundError(
                    f"Resume directory does not contain a training state: {resume_path}. "
                    "Expected `checkpoints/state/latest`, `state/latest`, or a `step_*` state directory."
                )
            logger.info("Resuming full training state from directory: %s", state_dir)
            self.load_training_state(str(state_dir))
            return
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        logger.info("Loading weight checkpoint only: %s", resume_path)
        load_wam_checkpoint(self._unwrap_core_model(), str(resume_path))
        logger.warning("Loaded .pt weights only; optimizer/scheduler/step were not restored.")

    def _load_weight_checkpoint_before_prepare(self, model) -> bool:
        resume = self.resume
        if not resume:
            return False
        resume_text = str(resume).strip()
        if resume_text.lower() in {"latest", "auto"}:
            return False
        resume_path = Path(resume_text)
        if not resume_path.is_file():
            return False
        logger.info("Loading weight checkpoint before distributed parameter sharding: %s", resume_path)
        load_wam_checkpoint(model, str(resume_path))
        logger.warning("Loaded .pt weights only; optimizer/scheduler/step were not restored.")
        return True

    def _unwrap_core_model(self):
        training_module = self.accelerator.unwrap_model(self.model)
        if not isinstance(training_module, WAMTrainingModule):
            raise TypeError(
                "Expected the distributed model to unwrap to WAMTrainingModule, "
                f"got {type(training_module).__name__}."
            )
        return training_module.wam

    def _set_trainable_modules_mode(self):
        logger.info("Enabling configured trainable modules and freezing the remaining model components.")
        self.model.train()
        model = self._unwrap_core_model()
        self._apply_trainable_modules_mode(model)

    @staticmethod
    def _apply_trainable_modules_mode(model):
        if hasattr(model, "set_trainable_modules"):
            model.set_trainable_modules()
            return
        model.eval()
        model.requires_grad_(False)
        model.mot.train()
        model.mot.requires_grad_(True)
        proprio_encoder = getattr(model, "proprio_encoder", None)
        if proprio_encoder is not None:
            proprio_encoder.train()
            proprio_encoder.requires_grad_(True)

    @staticmethod
    def _format_param_count(num_params: int) -> str:
        num_params = int(num_params)
        if num_params >= 1_000_000_000:
            return f"{num_params / 1_000_000_000:.3f}B ({num_params:,})"
        if num_params >= 1_000_000:
            return f"{num_params / 1_000_000:.3f}M ({num_params:,})"
        if num_params >= 1_000:
            return f"{num_params / 1_000:.3f}K ({num_params:,})"
        return f"{num_params:,}"

    @staticmethod
    def _trainable_param_group_name(name: str) -> str:
        if name.startswith("video_memory_registers."):
            return "video_memory_registers"
        if name.startswith("proprio_encoder."):
            return "proprio_encoder"
        if name.startswith("action_proprio_encoder."):
            return "action_proprio_encoder"
        if name.startswith("memory_proprio_token_encoder."):
            return "memory_proprio_token_encoder"
        if name.startswith("understanding.vlm.model."):
            return "understanding.vlm"
        if name.startswith("understanding."):
            parts = name.split(".")
            return ".".join(parts[:2]) if len(parts) >= 2 else "understanding"
        if name.startswith("mot.mixtures.video."):
            return "video_expert"
        if name.startswith("mot.mixtures.action."):
            return "action_expert"
        return name.split(".", 1)[0] if "." in name else name

    def _log_trainable_parameter_summary(self, model) -> None:
        if not self.accelerator.is_main_process:
            return
        groups: dict[str, int] = {}
        total = 0
        seen: set[int] = set()
        trainable_tensors = 0
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            param_id = id(param)
            if param_id in seen:
                continue
            seen.add(param_id)
            numel = int(param.numel())
            total += numel
            trainable_tensors += 1
            group_name = WAMTrainer._trainable_param_group_name(str(name))
            groups[group_name] = groups.get(group_name, 0) + numel

        logger.info(
            "Trainable parameter summary: total=%s tensors=%d groups=%d",
            WAMTrainer._format_param_count(total),
            trainable_tensors,
            len(groups),
        )
        for group_name, numel in sorted(groups.items(), key=lambda item: item[1], reverse=True):
            logger.info(
                "  trainable %-36s %s",
                group_name + ":",
                WAMTrainer._format_param_count(numel),
            )

    @staticmethod
    def _eval_video_num_frames(video: torch.Tensor) -> int:
        if video.ndim == 4:
            return int(video.shape[1])
        if video.ndim == 5:
            return int(video.shape[2])
        if video.ndim == 6:
            return int(video.shape[3])
        raise ValueError(f"Unsupported eval video shape: {tuple(video.shape)}")

    @staticmethod
    def _eval_input_image_from_video(video: torch.Tensor) -> torch.Tensor:
        if video.ndim == 4:
            return video[:, 0].unsqueeze(0).contiguous()
        if video.ndim == 5:
            return video[:, :, 0].unsqueeze(0).contiguous()
        raise ValueError(f"Unsupported unbatched eval video shape: {tuple(video.shape)}")

    @staticmethod
    def _flatten_eval_video_for_metrics(video: torch.Tensor) -> torch.Tensor:
        if video.ndim == 4:
            return video
        if video.ndim == 5:
            # [N,C,T,H,W] -> [C,T,H,N*W] for saving/metrics against decoded latent-horizontal video.
            return torch.cat([video[idx] for idx in range(int(video.shape[0]))], dim=-1).contiguous()
        raise ValueError(f"Unsupported unbatched eval video shape: {tuple(video.shape)}")

    @staticmethod
    def _to_batched_eval_sample(sample):
        video = sample["video"]
        prompt = sample["prompt"]
        action = sample.get("action", None)
        proprio = sample.get("proprio", None)
        context = sample.get("context", None)
        context_mask = sample.get("context_mask", None)
        image_is_pad = sample.get("image_is_pad", None)
        action_is_pad = sample.get("action_is_pad", None)
        action_mask = sample.get("action_mask", None)
        action_dim_loss_weight = sample.get("action_dim_loss_weight", None)
        action_dim_is_pad = sample.get("action_dim_is_pad", None)
        stats_group = sample.get("stats_group", None)
        vlm_current_images = sample.get("vlm_current_images", None)
        if vlm_current_images is not None and vlm_current_images.ndim == 4:
            vlm_current_images = vlm_current_images.unsqueeze(0)
        memory_keys = (
            "memory_video_anchor",
            "memory_video_anchor_is_pad",
            "memory_video_recent",
            "memory_video_recent_is_pad",
            "memory_video_anchor_proprio",
            "memory_video_anchor_proprio_is_pad",
            "memory_video_recent_proprio",
            "memory_video_recent_proprio_is_pad",
        )
        memory_video_keys = {
            "memory_video_anchor",
            "memory_video_recent",
        }
        memory_proprio_keys = {
            "memory_video_anchor_proprio",
            "memory_video_recent_proprio",
        }
        memory_inputs = {}
        for key in memory_keys:
            value = sample.get(key, None)
            if value is None:
                continue
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"`sample['{key}']` must be a torch.Tensor, got {type(value)}")
            if key in memory_video_keys:
                if value.ndim == 4:
                    value = value.unsqueeze(0)
                elif value.ndim == 5:
                    value = value.unsqueeze(0)
                if value.ndim not in (5, 6):
                    raise ValueError(
                        f"`{key}` must be [C,T,H,W], [N,C,T,H,W], [B,C,T,H,W], "
                        f"or [B,N,C,T,H,W], got {tuple(value.shape)}"
                    )
            elif key in memory_proprio_keys:
                if value.ndim == 2:
                    value = value.unsqueeze(0)
                if value.ndim != 3:
                    raise ValueError(
                        f"`{key}` must be [T,D] or [B,T,D], got {tuple(value.shape)}"
                    )
            else:
                if value.ndim == 1:
                    value = value.unsqueeze(0)
                if value.ndim != 2:
                    raise ValueError(f"`{key}` must be [T] or [B,T], got {tuple(value.shape)}")
            memory_inputs[key] = value
        if not memory_inputs:
            memory_inputs = None

        if not isinstance(video, torch.Tensor):
            raise TypeError(
                f"Expected tensor video for evaluation, got {type(video)}. "
                "Evaluation expects `video` with shape [C,T,H,W], [N,C,T,H,W], "
                "[B,C,T,H,W], or [B,N,C,T,H,W]."
            )
        if video.ndim == 4:
            video = video.unsqueeze(0)
        elif video.ndim == 5:
            video = video.unsqueeze(0)
        if video.ndim not in (5, 6):
            raise ValueError(
                "Expected video shape [C,T,H,W], [N,C,T,H,W], [B,C,T,H,W], "
                f"or [B,N,C,T,H,W], got {tuple(video.shape)}"
            )
        num_video_frames = WAMTrainer._eval_video_num_frames(video)
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

        if image_is_pad is not None:
            if not isinstance(image_is_pad, torch.Tensor):
                raise TypeError(
                    f"`sample[image_is_pad]` must be a torch.Tensor, got {type(image_is_pad)}"
                )
            if image_is_pad.ndim == 1:
                image_is_pad = image_is_pad.unsqueeze(0)
            if image_is_pad.ndim != 2:
                raise ValueError(
                    f"`image_is_pad` must have shape [T] or [B,T], got {tuple(image_is_pad.shape)}"
                )

        if action_is_pad is not None:
            if not isinstance(action_is_pad, torch.Tensor):
                raise TypeError(f"`sample[action_is_pad]` must be a torch.Tensor, got {type(action_is_pad)}")
            if action_is_pad.ndim == 1:
                action_is_pad = action_is_pad.unsqueeze(0)
            if action_is_pad.ndim != 2:
                raise ValueError(f"`action_is_pad` must have shape [T] or [B,T], got {tuple(action_is_pad.shape)}")

        if action_mask is not None:
            if not isinstance(action_mask, torch.Tensor):
                raise TypeError(
                    f"`sample[action_mask]` must be a torch.Tensor, got {type(action_mask)}"
                )
            if action_mask.ndim == 2:
                action_mask = action_mask.unsqueeze(0)
            if action is None or action_mask.shape != action.shape:
                raise ValueError(
                    "`action_mask` must match action [B,T,D], got "
                    f"{tuple(action_mask.shape)}"
                )

        if action_dim_loss_weight is not None:
            if not isinstance(action_dim_loss_weight, torch.Tensor):
                raise TypeError(
                    "`sample[action_dim_loss_weight]` must be a torch.Tensor, "
                    f"got {type(action_dim_loss_weight)}"
                )
            if action_dim_loss_weight.ndim == 1:
                action_dim_loss_weight = action_dim_loss_weight.unsqueeze(0)

        if action_dim_is_pad is not None:
            if not isinstance(action_dim_is_pad, torch.Tensor):
                raise TypeError(f"`sample[action_dim_is_pad]` must be a torch.Tensor, got {type(action_dim_is_pad)}")
            if action_dim_is_pad.ndim == 1:
                action_dim_is_pad = action_dim_is_pad.unsqueeze(0)
            if action_dim_is_pad.ndim != 2:
                raise ValueError(f"`action_dim_is_pad` must have shape [D] or [B,D], got {tuple(action_dim_is_pad.shape)}")

        if isinstance(stats_group, str):
            stats_group = [stats_group]
        elif isinstance(stats_group, tuple):
            stats_group = list(stats_group)
        elif stats_group is not None and not isinstance(stats_group, list):
            stats_group = [str(stats_group)]

        batched = {
            "video": video,
            "prompt": prompt,
            "action": action,
            "proprio": proprio,
            "context": context,
            "context_mask": context_mask,
            "action_horizon": action_horizon,
            "image_is_pad": image_is_pad,
            "action_is_pad": action_is_pad,
            "action_mask": action_mask,
            "action_dim_loss_weight": action_dim_loss_weight,
            "action_dim_is_pad": action_dim_is_pad,
            "memory_inputs": memory_inputs,
            "stats_group": stats_group,
            "vlm_current_images": vlm_current_images,
            "video_layout": sample.get("video_layout", None),
            "video_view_names": sample.get("video_view_names", None),
            "video_canvas_layout": sample.get("video_canvas_layout", None),
            "video_canvas_view_names": sample.get("video_canvas_view_names", None),
        }
        if memory_inputs is not None:
            # Frame loss reads the same anchor/recent tensors from the top
            # level; the sampler consumes the nested copy.
            batched.update(memory_inputs)
        return batched

    @staticmethod
    def _unbatch_eval_tensor_dict(values: dict | None) -> dict | None:
        if values is None:
            return None
        out = {}
        for key, value in values.items():
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                out[key] = value[0]
            else:
                out[key] = value
        return out

    @staticmethod
    def _action_error_metrics(
        pred_action: torch.Tensor,
        gt_action: torch.Tensor,
        *,
        action_mask: torch.Tensor | None = None,
        action_is_pad: torch.Tensor | None = None,
        action_dim_is_pad: torch.Tensor | None = None,
    ) -> tuple[float | None, float | None]:
        if pred_action.ndim == 2:
            pred_action = pred_action.unsqueeze(0)
        if gt_action.ndim == 2:
            gt_action = gt_action.unsqueeze(0)
        if pred_action.shape != gt_action.shape:
            raise ValueError(
                "Predicted action/GT action shape mismatch after denormalization: "
                f"pred={tuple(pred_action.shape)} vs gt={tuple(gt_action.shape)}"
            )

        diff = pred_action.detach().to(device="cpu", dtype=torch.float32) - gt_action.detach().to(device="cpu", dtype=torch.float32)
        valid = torch.ones_like(diff, dtype=torch.bool)

        if action_mask is not None:
            exact = action_mask.detach().to(device=diff.device, dtype=torch.bool)
            if exact.ndim == 2:
                exact = exact.unsqueeze(0)
            if exact.shape != diff.shape:
                raise ValueError(
                    f"action_mask shape {tuple(exact.shape)} does not match action {tuple(diff.shape)}"
                )
            valid &= exact

        if action_is_pad is not None:
            time_pad = action_is_pad.detach().to(device=diff.device, dtype=torch.bool)
            if time_pad.ndim == 1:
                time_pad = time_pad.unsqueeze(0)
            if time_pad.shape != diff.shape[:2]:
                raise ValueError(f"action_is_pad shape {tuple(time_pad.shape)} does not match action {tuple(diff.shape)}")
            valid &= ~time_pad.unsqueeze(-1)

        if action_dim_is_pad is not None:
            dim_pad = action_dim_is_pad.detach().to(device=diff.device, dtype=torch.bool)
            if dim_pad.ndim == 1:
                dim_pad = dim_pad.unsqueeze(0)
            if dim_pad.shape != (diff.shape[0], diff.shape[2]):
                raise ValueError(f"action_dim_is_pad shape {tuple(dim_pad.shape)} does not match action {tuple(diff.shape)}")
            valid &= ~dim_pad.unsqueeze(1)

        selected = diff[valid]
        if selected.numel() == 0:
            return None, None
        return selected.abs().mean().item(), selected.pow(2).mean().item()

    @torch.no_grad()
    def evaluate(self):
        if self.val_dataset is None:
            return None

        model = self._unwrap_core_model()
        was_mot_training = model.mot.training
        self.model.eval()

        # Use a fixed, stratified panel of independent frame/action windows.
        world_size = int(self.accelerator.num_processes)
        rank = int(self.accelerator.process_index)
        global_panel_size = world_size * self.eval_loss_samples_per_rank
        local_loss_rows: list[list[float]] = []
        sample = None
        for local_slot in range(self.eval_loss_samples_per_rank):
            global_slot = rank * self.eval_loss_samples_per_rank + local_slot
            eval_index = min(
                len(self.val_dataset) - 1,
                int((global_slot + 0.5) * len(self.val_dataset) / global_panel_size),
            )
            eval_seed = self.seed + 100_000 + global_slot
            fork_devices = (
                [int(self.accelerator.device.index)]
                if self.accelerator.device.type == "cuda"
                and self.accelerator.device.index is not None
                else []
            )
            with torch.random.fork_rng(devices=fork_devices):
                torch.manual_seed(eval_seed)
                if self.accelerator.device.type == "cuda":
                    torch.cuda.manual_seed(eval_seed)
                eval_sample = self._to_batched_eval_sample(
                    self.val_dataset[eval_index]
                )
                with self.accelerator.autocast():
                    loss_value, loss_dict = self.model(eval_sample, global_step=self.global_step)
            local_loss_rows.append(
                [
                    float(loss_value.float().item()),
                    float(loss_dict.get("loss_video", float("nan"))),
                    float(loss_dict.get("loss_action", float("nan"))),
                ]
            )
            if sample is None:
                sample = eval_sample

        if sample is None:
            raise RuntimeError("Validation panel produced no samples.")
        panel_metrics = torch.tensor(
            local_loss_rows,
            device=self.accelerator.device,
            dtype=torch.float32,
        )
        gathered_panel_metrics = self.accelerator.gather_for_metrics(panel_metrics)
        panel_mean = gathered_panel_metrics.mean(dim=0)
        val_loss = float(panel_mean[0].item())
        val_loss_video = float(panel_mean[1].item())
        val_loss_action = float(panel_mean[2].item())
        
        prompt = sample["prompt"][0]
        video0 = sample["video"][0]  # [C,T,H,W] or [N,C,T,H,W] in (-1, 1)
        action = (
            sample["action"][0]
            if sample.get("action") is not None
            else None
        )
        proprio_for_denorm = sample.get("proprio")
        proprio = (
            proprio_for_denorm[0, 0]
            if proprio_for_denorm is not None
            else None
        )
        action_is_pad_for_metrics = sample.get("action_is_pad")
        action_mask_for_metrics = sample.get("action_mask")
        video_layout = sample.get("video_layout")
        video_view_names = sample.get("video_view_names")
        vlm_view_names = sample.get("video_canvas_view_names")
        if isinstance(video_layout, (list, tuple)):
            video_layout = video_layout[0] if video_layout else None
        if isinstance(video_view_names, (list, tuple)):
            video_view_names = video_view_names[0] if video_view_names else None
        if isinstance(vlm_view_names, (list, tuple)):
            vlm_view_names = vlm_view_names[0] if vlm_view_names else None

        input_image = self._eval_input_image_from_video(video0)
        num_frames = self._eval_video_num_frames(video0)
        video0_for_metrics = self._flatten_eval_video_for_metrics(video0)

        # FastWAM-style one-window joint video/action diffusion.
        infer_kwargs = {
            "input_image": input_image,
            "num_frames": num_frames,
            "action": action,
            "action_horizon": sample["action_horizon"],
            "proprio": proprio,
            "num_inference_steps": self.eval_num_inference_steps,
            "seed": 42,
            "tiled": False,
            "understanding_prompt": prompt,
            "memory_inputs": self._unbatch_eval_tensor_dict(
                sample.get("memory_inputs")
            ),
            "vlm_current_images": (
                sample["vlm_current_images"][0]
                if sample.get("vlm_current_images") is not None
                else None
            ),
            "vlm_current_view_is_pad": (
                sample["vlm_current_view_is_pad"][0]
                if sample.get("vlm_current_view_is_pad") is not None else None
            ),
            "vlm_view_names": vlm_view_names,
            "video_layout": video_layout,
            "video_view_names": video_view_names,
        }
        if infer_kwargs["memory_inputs"] is None:
            raise ValueError(
                "Frame evaluation requires one anchor and one recent frame."
            )
        if sample["context"] is not None:
            infer_kwargs["prompt"] = None
            infer_kwargs["context"] = sample["context"][0]
            infer_kwargs["context_mask"] = sample["context_mask"][0]
        else:
            infer_kwargs["prompt"] = prompt

        pred = self.model(
            operation="infer_frame_joint_window",
            **infer_kwargs,
        )
        
        pred_video = pred["video"]
        pred_action = pred.get("action", None)

        # 3. inference metrics against GT video
        pred_video_tensor = pil_frames_to_video_tensor(pred_video)
        gt_video_tensor = ((video0_for_metrics.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()

        assert pred_video_tensor.shape == gt_video_tensor.shape, (
            "Eval infer prediction/GT shape mismatch: "
            f"pred={tuple(pred_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_rollout_vs_gt = video_psnr(pred=pred_video_tensor, target=gt_video_tensor)
        ssim_rollout_vs_gt = video_ssim(pred=pred_video_tensor, target=gt_video_tensor)

        action_l1 = None
        action_l2 = None
        if action is not None and pred_action is not None:
            if proprio_for_denorm is None:
                raise ValueError("Eval sample must contain `proprio` for action denormalization.")
            proprio = proprio_for_denorm.detach().to(device="cpu", dtype=torch.float32)
            
            lerobot_dataset = getattr(self.val_dataset, "lerobot_dataset", None)
            processor = getattr(lerobot_dataset, "processor", None)
            if processor is not None:
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
                action_l1, action_l2 = self._action_error_metrics(
                    pred_action_denorm,
                    gt_action_denorm,
                    action_mask=action_mask_for_metrics,
                    action_is_pad=action_is_pad_for_metrics,
                    action_dim_is_pad=sample.get("action_dim_is_pad"),
                )
            else:
                denormalize_action = getattr(self.val_dataset, "denormalize_action", None)
                if callable(denormalize_action):
                    pred_action_denorm = denormalize_action(
                        pred_action,
                        stats_group=sample.get("stats_group"),
                        dim_is_pad=sample.get("action_dim_is_pad"),
                    )
                    gt_action_denorm = denormalize_action(
                        action,
                        stats_group=sample.get("stats_group"),
                        dim_is_pad=sample.get("action_dim_is_pad"),
                    )
                    action_l1, action_l2 = self._action_error_metrics(
                        pred_action_denorm,
                        gt_action_denorm,
                        action_mask=action_mask_for_metrics,
                        action_is_pad=action_is_pad_for_metrics,
                        action_dim_is_pad=sample.get("action_dim_is_pad"),
                    )

        # 4. VAE reconstruction metrics against GT video
        gt_video_batch = video0.unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
        vae_latents = self.model(
            operation="encode_video_latents",
            video_tensor=gt_video_batch,
            tiled=False,
            video_layout=video_layout,
            video_view_names=video_view_names,
        )
        vae_recon_video = self.model(
            operation="decode_latents",
            latents=vae_latents,
            tiled=False,
            video_layout=video_layout,
            video_view_names=video_view_names,
        )
        vae_video_tensor = pil_frames_to_video_tensor(vae_recon_video)

        assert vae_video_tensor.shape == gt_video_tensor.shape, (
            "Eval VAE reconstruction/GT shape mismatch: "
            f"vae={tuple(vae_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_decode_vs_gt = video_psnr(pred=vae_video_tensor, target=gt_video_tensor)
        ssim_decode_vs_gt = video_ssim(pred=vae_video_tensor, target=gt_video_tensor)

        psnr_rollout_vs_decode = video_psnr(pred=pred_video_tensor, target=vae_video_tensor)
        ssim_rollout_vs_decode = video_ssim(pred=pred_video_tensor, target=vae_video_tensor)

        video_path = ""
        should_save_video = self.eval_max_videos is None or self.accelerator.process_index < self.eval_max_videos
        if should_save_video:
            stitched_video_tensor = torch.cat(
                [pred_video_tensor, vae_video_tensor, gt_video_tensor],
                dim=3,
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
                float(val_loss_video),
                float(val_loss_action),
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
        mean_metrics = gathered_metrics[:, :9].mean(dim=0)
        action_l2_values = gathered_metrics[:, 9]
        action_l1_values = gathered_metrics[:, 10]
        action_l2_valid = action_l2_values >= 0.0
        action_l1_valid = action_l1_values >= 0.0
        action_l2_mean = action_l2_values[action_l2_valid].mean().item() if bool(action_l2_valid.any().item()) else None
        action_l1_mean = action_l1_values[action_l1_valid].mean().item() if bool(action_l1_valid.any().item()) else None

        if was_mot_training:
            self._set_trainable_modules_mode()

        result = {
            "val_loss": float(mean_metrics[0].item()),
            "loss_video": float(mean_metrics[1].item()),
            "loss_action": float(mean_metrics[2].item()),
            "psnr_rg": float(mean_metrics[3].item()),
            "ssim_rg": float(mean_metrics[4].item()),
            "psnr_rd": float(mean_metrics[5].item()),
            "ssim_rd": float(mean_metrics[6].item()),
            "psnr_dg": float(mean_metrics[7].item()),
            "ssim_dg": float(mean_metrics[8].item()),
            "video_path": video_path,
        }
        if action_l2_mean is not None:
            result["action_l2"] = float(action_l2_mean)
        if action_l1_mean is not None:
            result["action_l1"] = float(action_l1_mean)
        return result

    def _read_weights_manifest(self) -> dict:
        manifest_path = Path(self.weights_dir) / "weights_manifest.json"
        default = {"version": 1, "max_slots": int(self.max_weight_checkpoints), "slots": []}
        if not manifest_path.is_file():
            return default
        try:
            with manifest_path.open("r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception as exc:
            logger.warning("Could not read weight checkpoint manifest %s: %s", manifest_path, exc)
            return default
        if not isinstance(manifest, dict):
            return default
        if not isinstance(manifest.get("slots"), list):
            manifest["slots"] = []
        manifest["version"] = int(manifest.get("version", 1))
        manifest["max_slots"] = int(self.max_weight_checkpoints)
        return manifest

    def _write_weights_manifest(self, manifest: dict) -> None:
        manifest_path = Path(self.weights_dir) / "weights_manifest.json"
        tmp_path = Path(self.weights_dir) / "weights_manifest.tmp.json"
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=True, indent=2)
            f.write("\n")
        os.replace(tmp_path, manifest_path)

    def _uses_deepspeed_zero3(self) -> bool:
        return deepspeed_zero_stage(self.accelerator) == 3

    def _save_weights_checkpoint_rank0(self, model, step_tag: str):
        max_slots = int(self.max_weight_checkpoints)
        manifest = self._read_weights_manifest()
        existing_slots: list[dict] = []
        occupied_slots: set[int] = set()
        for item in manifest.get("slots", []):
            try:
                item_slot = int(item.get("slot", -1))
                item_step = int(item.get("step", -1))
            except (TypeError, ValueError):
                continue
            if 0 <= item_slot < max_slots:
                existing_slots.append({**item, "slot": item_slot, "step": item_step})
                occupied_slots.add(item_slot)

        free_slots = [idx for idx in range(max_slots) if idx not in occupied_slots]
        if free_slots:
            slot_index = free_slots[0]
        elif existing_slots:
            slot_index = min(existing_slots, key=lambda item: int(item.get("step", -1)))["slot"]
        else:
            slot_index = 0

        ckpt_name = f"slot_{slot_index}.pt"
        ckpt_path = os.path.join(self.weights_dir, ckpt_name)

        # Write directly into the selected slot to avoid an extra full-size temp file.
        # The manifest preserves the real training step for each slot.
        save_wam_checkpoint(model, ckpt_path, step=self.global_step)

        entry = {
            "slot": int(slot_index),
            "step": int(self.global_step),
            "step_tag": str(step_tag),
            "path": ckpt_name,
        }
        slots = [
            item
            for item in existing_slots
            if int(item.get("slot", -1)) != slot_index
        ]
        slots.append(entry)
        slots = sorted(slots, key=lambda item: int(item.get("step", -1)))[-max_slots:]
        self._write_weights_manifest(
            {
                "version": 1,
                "max_slots": max_slots,
                "latest": entry,
                "slots": slots,
            }
        )
        logger.info("Saved weight checkpoint %s as %s", step_tag, ckpt_path)
        return ckpt_path

    def _save_weights_checkpoint(self, step_tag: str):
        model = self._unwrap_core_model()
        if self._uses_deepspeed_zero3():
            try:
                import deepspeed
            except ImportError as exc:
                raise ImportError("DeepSpeed is required to export ZeRO-3 sharded weights.") from exc

            if hasattr(model, "trainable_parameters"):
                params = list(model.trainable_parameters())
            else:
                params = [p for p in model.parameters() if p.requires_grad]
            with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                if self.accelerator.is_main_process:
                    return self._save_weights_checkpoint_rank0(model, step_tag)
            return None

        if not self.accelerator.is_main_process:
            return None
        return self._save_weights_checkpoint_rank0(model, step_tag)

    def _save_trainer_state(self, state_path: str, step_tag: str, weights_path: str | None):
        state_file = os.path.join(state_path, "trainer_state.json")
        payload = {
            "global_step": int(self.global_step),
            "epoch": int(self.epoch),
            "batch_in_epoch": int(self.batch_in_epoch),
            "step_tag": str(step_tag),
        }
        if weights_path is not None:
            payload["weights_path"] = str(weights_path)
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    def save_checkpoint(self):
        step_tag = f"step_{self.global_step:06d}"

        self.accelerator.wait_for_everyone()
        ckpt_path = self._save_weights_checkpoint(step_tag=step_tag)
        self.accelerator.wait_for_everyone()

        if not bool(self.cfg.get("save_training_state", True)):
            return {"weights_path": ckpt_path, "state_path": None}

        latest_state_path = os.path.join(self.state_dir, "latest")
        if self.accelerator.is_main_process:
            ensure_dir(latest_state_path)
        self.accelerator.wait_for_everyone()

        # Keep one optimizer state snapshot; weight checkpoints rotate separately.
        self.accelerator.save_state(output_dir=latest_state_path)
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            self._save_trainer_state(latest_state_path, step_tag=step_tag, weights_path=ckpt_path)
        self.accelerator.wait_for_everyone()

        return {"weights_path": ckpt_path, "state_path": latest_state_path}

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

    def _shutdown_distributed(self) -> None:
        self.cache_runtime.close()
        self._finish_wandb()
        self.accelerator.end_training()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

    def _finish_training(self) -> None:
        if bool(self.cfg.get("save_on_train_end", True)):
            ckpt_info = self.save_checkpoint()
            if self.accelerator.is_main_process:
                logger.info(
                    "[done] training finished step=%d weights=%s state=%s",
                    self.global_step,
                    ckpt_info["weights_path"],
                    ckpt_info["state_path"],
                )
        elif self.accelerator.is_main_process:
            logger.info("[done] training finished step=%d", self.global_step)
        self._shutdown_distributed()

    def train(self):
        self._set_trainable_modules_mode()

        if self.max_steps is None:
            raise ValueError("`max_steps` must be set before entering the while-step training loop.")

        logger.info("Starting training with max_steps=%d.", self.max_steps)
        # Keep map-style sampling aligned when resuming in the middle of an
        # epoch.  Mix datasets publish this value through shared memory, so
        # persistent workers observe it before their first resumed fetch.
        self.train_sampler.set_epoch_offset(self.epoch)
        if hasattr(self.train_dataset, "set_epoch"):
            self.train_dataset.set_epoch(self.epoch)
        data_iter = iter(self.train_loader)
        self.speed_history = [(self.global_step, time.perf_counter())]

        while self.global_step < self.max_steps:
            try:
                sample = next(data_iter)
                self.batch_in_epoch += 1
            except StopIteration:
                self.epoch += 1
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                self.train_sampler.set_epoch_offset(self.epoch)
                if hasattr(self.train_dataset, "set_epoch"):
                    self.train_dataset.set_epoch(self.epoch)
                data_iter = iter(self.train_loader)
                continue

            with self.accelerator.accumulate(self.model):
                with self.accelerator.autocast():
                    loss, loss_dict = self.model(sample, global_step=self.global_step)

                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    grad_norm = self.accelerator.clip_grad_norm_(self.grad_clip_params, self.max_grad_norm)
                    self.optimizer.step()
                    if not self.accelerator.optimizer_step_was_skipped:
                        self.scheduler.step()
                        core = self._unwrap_core_model()
                        if getattr(core, "pica_alignment", None) is not None:
                            core.pica_alignment.update_ema(core.mot)
                    self.optimizer.zero_grad(set_to_none=True)

                    self.global_step += 1
                    self.speed_history.append(
                        (self.global_step, time.perf_counter())
                    )
                    if len(self.speed_history) > 11:
                        del self.speed_history[:-11]
                    should_log = self.log_every > 0 and self.global_step % self.log_every == 0
                    global_loss = None
                    global_loss_metrics = {}
                    global_grad_norm = None
                    if should_log:
                        global_metrics = gather_mean_scalars(
                            self.accelerator,
                            {
                                "loss": loss.detach(),
                                "grad_norm": grad_norm,
                                **loss_dict,
                            },
                            device=loss.device,
                        )
                        global_loss = global_metrics.pop("loss")
                        global_grad_norm = global_metrics.pop("grad_norm")
                        global_loss_metrics = global_metrics

                    current_lr = float(self.optimizer.param_groups[0]["lr"])
                    current_vlm_lr = next(
                        (
                            float(group["lr"])
                            for group in self.optimizer.param_groups
                            if group.get("name") == "vlm"
                        ),
                        None,
                    )

                    if should_log and self.accelerator.is_main_process:
                        eta_str, steps_per_sec = self._estimate_eta()
                        effective_global_batch_size = (
                            self.batch_size
                            * self.accelerator.num_processes
                            * self.gradient_accumulation_steps
                        )
                        samples_per_sec = steps_per_sec * effective_global_batch_size
                        description = "[train] epoch=%d step=%d/%d loss=%.4f " % (
                            self.epoch,
                            self.global_step,
                            self.max_steps,
                            global_loss,
                        )
                        if global_loss_metrics:
                            detail_str = " ".join([f"{k}={v:.4f}" for k, v in sorted(global_loss_metrics.items())])
                            description += detail_str + " "
                        description += "lr=%.2e speed=%.2f optimizer_step/s, %.2f samples/s eta=%s" % (
                            current_lr,
                            steps_per_sec,
                            samples_per_sec,
                            eta_str,
                        )
                        if current_vlm_lr is not None:
                            description += " vlm_lr=%.2e" % current_vlm_lr
                        logger.info(description)

                        wandb_payload = {
                            "train/loss": global_loss,
                            "train/grad_norm": global_grad_norm,
                            "train/lr": current_lr,
                            "performance/steps_per_sec": steps_per_sec,
                            "performance/optimizer_steps_per_sec": steps_per_sec,
                            "performance/samples_per_sec": samples_per_sec,
                            "performance/effective_global_batch_size": effective_global_batch_size,
                        }
                        if current_vlm_lr is not None:
                            wandb_payload["train/vlm_lr"] = current_vlm_lr
                        for key, value in global_loss_metrics.items():
                            wandb_payload[f"train/{key}"] = value
                        self._wandb_log(wandb_payload)

                    if (
                        self.eval_every > 0
                        and self.val_dataset is not None
                        and self.global_step % self.eval_every == 0
                    ):
                        metrics = self.evaluate()
                        self._release_runtime_memory()
                        self.accelerator.wait_for_everyone()
                        if metrics is not None and self.accelerator.is_main_process:
                            description = "[eval] step=%d val_loss=%.4f loss_action=%.4f loss_video=%.4f" % (
                                self.global_step,
                                metrics["val_loss"],
                                metrics["loss_action"],
                                metrics["loss_video"],
                            )
                            if "psnr_rg" in metrics and "ssim_rg" in metrics:
                                description += " infer_psnr=%.4f infer_ssim=%.4f" % (
                                    metrics["psnr_rg"],
                                    metrics["ssim_rg"],
                                )
                            if "action_l2" in metrics:
                                description += " action_l2=%.6g" % metrics["action_l2"]
                            if "action_l1" in metrics:
                                description += " action_l1=%.6g" % metrics["action_l1"]
                            if metrics.get("video_path"):
                                description += " video=%s" % metrics["video_path"]
                            logger.info(description)
                            eval_payload = {
                                "eval/val_loss": float(metrics["val_loss"]),
                                "eval/loss_action": float(metrics["loss_action"]),
                                "eval/loss_video": float(metrics["loss_video"]),
                            }
                            for key in (
                                "psnr_rg",
                                "ssim_rg",
                                "psnr_rd",
                                "ssim_rd",
                                "psnr_dg",
                                "ssim_dg",
                            ):
                                if key in metrics:
                                    eval_payload[f"eval/{key}"] = float(metrics[key])
                            if "action_l2" in metrics:
                                eval_payload["eval/action_l2"] = float(metrics["action_l2"])
                            if "action_l1" in metrics:
                                eval_payload["eval/action_l1"] = float(metrics["action_l1"])
                            self._wandb_log(eval_payload)
                        self.speed_history = [
                            (self.global_step, time.perf_counter())
                        ]

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
                        self._finish_training()
                        return

        self._finish_training()
        
