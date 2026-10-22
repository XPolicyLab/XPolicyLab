import gc
import hashlib
import json
import os
import random
import re
from contextlib import contextmanager
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
from torch.utils.data._utils.collate import default_collate

from .cache import CacheRuntimeBindings
from .training.checkpoint import (
    joint96_checkpoint_metadata, joint96_file_sha256, joint96_json_sha256,
    joint96_resume_recipe, load_wam_checkpoint, save_wam_checkpoint,
    validate_joint96_resume_directory, load_joint96_initialization,
)
from .training.distributed import deepspeed_zero_stage
from .training.metrics import gather_mean_scalars
from .training.module import WAMTrainingModule
from .training.multiview_recipe import (
    CappedWarmupCosineLR,
    MULTIVIEW_RESUME_SCHEMA,
    multiview_enabled,
)
from .utils.fs import ensure_dir
from .utils.logging_config import get_logger
from .utils.pytorch_utils import set_global_seed
from .utils.training_timing import TrainingTimer, time_methods
from .utils.samplers import EpisodeShardSampler, ResumableEpochSampler
from .utils.video_io import save_mp4
from .utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim

logger = get_logger(__name__)

_JOINT96_PANEL_LOSS_KEYS = (
    "val_loss", "loss_video", "loss_action", "loss_vae", "loss_stvae",
    "loss_vae_raw", "loss_stvae_raw",
)


@contextmanager
def _preserve_eval_rng_and_modes(module):
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    torch_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    modes = [(child, child.training) for child in module.modules()]
    try:
        module.eval()
        with torch.no_grad():
            yield
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        for child, training in modes:
            child.training = training


def _joint96_panel(panel, dataset_length):
    if not panel:
        raise ValueError("local_joint96 requires a nonempty fixed held-out validation_panel")
    result, seen = [], set()
    for entry in panel:
        index, source, seed = int(entry["index"]), str(entry["source"]), int(entry["seed"])
        if not 0 <= index < dataset_length or not source or index in seen:
            raise ValueError("Invalid or duplicated local_joint96 validation panel entry")
        seen.add(index)
        row = {"index": index, "source": source, "seed": seed}
        for key in ("episode_id", "start", "embodiment_id"):
            if key in entry:
                row[key] = int(entry[key])
        result.append(row)
    return result


def _joint96_summarize_panel(sources, totals):
    # Columns are sample count and sums, never means-of-rank-means.
    keys = _JOINT96_PANEL_LOSS_KEYS
    per_source = {}
    for source, values in zip(sources, totals.tolist(), strict=True):
        count = int(values[0])
        if count <= 0:
            raise RuntimeError(f"No validation samples for {source}")
        per_source[source] = {"count": count, **dict(zip(keys, (v / count for v in values[1:]), strict=True))}
    sums = totals.sum(dim=0).tolist()
    return {"count": int(sums[0]), **dict(zip(keys, (v / sums[0] for v in sums[1:]), strict=True)), "sources": per_source}


class _LocalRobotEpochSampler(ResumableEpochSampler):
    """Global ordered indices; Accelerate shards once, dataset owns epoch mixing."""

    def __init__(self, dataset, seed, batch_size, num_processes):
        super().__init__(dataset, seed, batch_size, num_processes, shuffle=False)
        global_batch = self.batch_size * self.num_processes
        self.usable = len(dataset) // global_batch * global_batch
        if not self.usable:
            raise ValueError("Local Robot dataset has less than one complete global batch")

    def __iter__(self):
        start = self.resume_batch_offset * self.batch_size * self.num_processes
        return iter(range(start, self.usable))

    def __len__(self):
        return self.usable


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
        from .training.soft_prompt import wrap_soft_prompt_datasets
        train_dataset, val_dataset = wrap_soft_prompt_datasets(model, train_dataset, val_dataset, cfg)
        self.model = WAMTrainingModule(model)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cache_runtime = cache_runtime or CacheRuntimeBindings.disabled()
        self.cfg = cfg
        self.local_joint96 = bool(cfg.get("local_robot_joint96", {}).get("enabled", False))
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
        if cfg.get("local_robot_softprompt", {}).get("enabled", False):
            registry_sha = self.train_dataset.embodiment_registry.sha256
            fingerprint = torch.tensor(list(bytes.fromhex(registry_sha)), dtype=torch.uint8,
                                       device=self.accelerator.device)[None]
            gathered = self.accelerator.gather(fingerprint).reshape(-1, 32)
            if not torch.equal(gathered, gathered[:1].expand_as(gathered)):
                raise ValueError("Soft prompt registries differ across ranks")
        deepspeed_config = getattr(deepspeed_plugin, "deepspeed_config", {}) or {}
        if self.local_joint96:
            self._validate_local_joint96_setup(deepspeed_config)
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
        warmup_steps = (
            int(cfg.warmup_steps) if self.local_joint96 else int(total_train_steps * 0.05)
        )
        if self.local_joint96 and not 0 <= warmup_steps < total_train_steps:
            raise ValueError("local_joint96 warmup_steps must be in [0, max_steps)")
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
        if self.local_joint96:
            self._validate_local_joint96_engine()
        self._unwrap_core_model().set_runtime_device(self.accelerator.device)
        self.optimizer.zero_grad(set_to_none=True)
        self.wandb_run = None
        self._init_wandb()
        if self.local_joint96:
            self._local_joint96_contract = self._build_local_joint96_resume_contract()
        self._resume_or_load_checkpoint()

        val_size = len(self.val_dataset) if self.val_dataset is not None else len(self.train_dataset)
        logger.info("Train/val dataset size: %d/%d", len(self.train_dataset), val_size)

    def _validate_local_joint96_setup(self, deepspeed_config):
        local = self.cfg.local_robot_joint96
        if self.batch_size <= 0 or self.batch_size % 4:
            raise ValueError("Local Robot batch_size must be divisible by the actual block size 4")
        if self.gradient_accumulation_steps != 1:
            raise ValueError("local_joint96 uses one Robot Joint batch per optimizer update; GAS must be 1")
        if self.max_grad_norm != 1.0:
            raise ValueError("local_joint96 requires actual gradient clipping at 1.0")
        if self.resume is not None:
            validate_joint96_resume_directory(self.resume, cfg=self.cfg, world_size=int(self.accelerator.num_processes))
        if self.save_every <= 0 or self.eval_every <= 0 or self.eval_every % self.save_every:
            raise ValueError(
                "local_joint96 requires positive save/eval intervals, with eval_every a multiple of save_every"
            )
        stop = local.get("stop_after_step")
        if stop is not None and (int(stop) <= 0 or int(stop) > int(self.cfg.max_steps) or int(stop) % self.save_every):
            raise ValueError("stop_after_step must be a positive absolute checkpoint boundary <= max_steps")
        if int(self.accelerator.num_processes) != int(local.expected_world_size):
            raise ValueError("local_joint96 actual world size differs from expected_world_size")
        if self.val_dataset is None or self.val_dataset is self.train_dataset:
            raise ValueError("local_joint96 requires a separate, fixed held-out validation dataset")
        if not bool(self.cfg.get("save_training_state", False)):
            raise ValueError("local_joint96 requires both weights and complete training state")
        if bool(local.get("require_deepspeed", True)) and not deepspeed_config:
            raise ValueError("local_joint96 requires its dedicated DeepSpeed configuration")
        if deepspeed_config:
            if deepspeed_config.get("zero_optimization", {}).get("stage") != 1:
                raise ValueError("local_joint96 requires ZeRO-1")
            if deepspeed_config.get("gradient_clipping") != 1.0:
                raise ValueError("DeepSpeed gradient_clipping must explicitly be 1.0, not absent/auto")
            if deepspeed_config.get("gradient_accumulation_steps") != 1:
                raise ValueError("DeepSpeed gradient_accumulation_steps must explicitly be 1")

    def _validate_local_joint96_engine(self, expected_step=0):
        if not bool(self.cfg.local_robot_joint96.get("require_deepspeed", True)):
            return
        engine = self.model
        required = ("gradient_clipping", "gradient_accumulation_steps", "train_micro_batch_size_per_gpu")
        if any(not callable(getattr(engine, name, None)) for name in required):
            raise RuntimeError("Prepared local_joint96 model is not a verifiable DeepSpeed engine")
        if float(engine.gradient_clipping()) != 1.0 or float(getattr(engine.optimizer, "clip_grad", -1)) != 1.0:
            raise RuntimeError("DeepSpeed engine/ZeRO optimizer does not actually clip gradients at 1.0")
        if int(engine.gradient_accumulation_steps()) != 1 or int(engine.train_micro_batch_size_per_gpu()) != self.batch_size:
            raise RuntimeError("DeepSpeed effective batch/GAS differs from local_joint96 recipe")
        if not hasattr(engine, "global_steps") or not hasattr(engine, "skipped_steps"):
            raise RuntimeError("DeepSpeed engine is missing optimizer update counters")
        if int(engine.global_steps) != expected_step or int(engine.skipped_steps):
            raise RuntimeError(f"local_joint96 engine must have {expected_step} absolute updates and zero skips")
        if expected_step:
            base_optimizer = getattr(engine.optimizer, "optimizer", None)
            states = getattr(base_optimizer, "state", {})
            if not states or any("step" not in state or int(state["step"]) != expected_step for state in states.values()):
                raise RuntimeError("Restored AdamW partition steps differ from the absolute engine update count")
        logger.info("[local_joint96] actual ZeRO-1 optimizer clip_grad=1.0 GAS=1 micro_batch=%d", self.batch_size)

    def _local_joint96_update_snapshot(self):
        if hasattr(self.model, "global_steps"):
            return int(self.model.global_steps), int(self.model.skipped_steps)
        return None

    def _validate_local_joint96_update(self, before):
        if self.accelerator.optimizer_step_was_skipped:
            raise RuntimeError("local_joint96 optimizer update was skipped; step/scheduler must not advance")
        if before is not None:
            after = self._local_joint96_update_snapshot()
            if after != (before[0] + 1, before[1]):
                raise RuntimeError(f"Expected exactly one actual optimizer update: before={before}, after={after}")

    def _validate_local_joint96_grad_norm(self, grad_norm):
        norm = torch.as_tensor(float("nan") if grad_norm is None else grad_norm,
                               device=self.accelerator.device, dtype=torch.float64).detach()
        nonfinite = self.accelerator.reduce((~torch.isfinite(norm).all()).to(torch.int32), reduction="sum")
        if int(nonfinite.item()):
            raise RuntimeError(
                "Non-finite or missing local_joint96 grad_norm on at least one rank; "
                "run validation failed. DeepSpeed may already have updated parameters during backward; "
                "this is not a pre-update guard."
            )

    def _local_joint96_log_suffix(self, grad_norm):
        snapshot = self._local_joint96_update_snapshot()
        if snapshot is None:
            raise RuntimeError("Cannot log actual local_joint96 optimizer updates without engine counters")
        return f" grad_norm={float(grad_norm):.6f} optimizer_updates={snapshot[0] - snapshot[1]}"

    def _build_local_joint96_resume_contract(self):
        package = self.train_dataset.package
        manifest_hash = joint96_file_sha256(package.root / "manifest.json")
        norm_hash = joint96_file_sha256(package.root / "norm.json")
        marker = json.loads((package.root / "_NORM_COMPLETE").read_text())
        if (manifest_hash != package.fingerprint or self.val_dataset.package.fingerprint != manifest_hash
                or marker["package_sha256"] != manifest_hash or marker["norm_sha256"] != norm_hash):
            raise ValueError("Joint96 package/norm identity changed during construction")
        model = self._unwrap_core_model()
        contract = {
            "schema": "phoenix.local_joint96.resume.v1",
            "recipe": joint96_resume_recipe(self.cfg),
            "world_size": int(self.accelerator.num_processes),
            "package_sha256": manifest_hash, "norm_sha256": norm_hash,
            "joint96_metadata": joint96_checkpoint_metadata(model),
            "trainable_parameters": [[n, list(p.shape), str(p.dtype)] for n, p in model.named_parameters() if p.requires_grad],
            "batches_per_epoch": self.train_sampler.usable // (self.batch_size * self.accelerator.num_processes),
            "validation_panel_sha256": joint96_json_sha256(self.val_dataset.fixed_validation_panel),
        }
        if multiview_enabled(self.cfg):
            sampling = getattr(self.train_dataset, "sampling_contract", None)
            if not isinstance(sampling, dict) or not sampling:
                raise ValueError("Multiview reader must expose its immutable sampling_contract")
            if sampling.get("plan_sha256") != self.cfg.local_robot_multiview.plan_sha256:
                raise ValueError("Multiview reader plan identity differs from the pinned configuration")
            # Canonical serialization also prevents later reader mutation from
            # changing the identity attached to already-consumed batches.
            contract["sampling_contract"] = json.loads(json.dumps(sampling, allow_nan=False))
            contract["schema"] = MULTIVIEW_RESUME_SCHEMA
            contract["run_budget"] = {"max_steps": int(self.cfg.max_steps)}
        return contract

    def _local_joint96_iterator(self):
        self.train_dataset.set_epoch(self.epoch)
        if hasattr(self.train_loader, "set_epoch"):
            self.train_loader.set_epoch(self.epoch)
        # The reader's augmentation is stateless by sample/epoch. Worker seeds
        # must not consume the global RNG used for model noise after a restart.
        self._local_loader_generator.manual_seed(self.seed + self.epoch)
        return iter(self.train_loader)

    def _write_local_joint96_trace(self, sample, loss, grad_norm):
        if not self.cfg.local_robot_joint96.get("integration_trace", False):
            return
        def plain(value):
            if torch.is_tensor(value):
                return value.detach().cpu().tolist()
            if isinstance(value, (tuple, list)):
                return [plain(v) for v in value]
            return value
        keys = ("sample_id", "physical_episode_uid", "source_id", "profile_id", "episode_id",
                "start", "task_index", "language_variant", "color_params", "camera_is_pad")
        if "main_aug_mode" in sample:
            keys += ("main_aug_eligible", "main_aug_mode", "main_noise_type", "main_noise_params", "augmentation_camera_mask")
        identity = {key: plain(sample[key]) for key in keys}
        snapshot = self._local_joint96_update_snapshot()
        record = dict(step=self.global_step, optimizer_updates=snapshot[0] - snapshot[1],
                      engine_skipped_steps=snapshot[1], epoch=self.epoch, batch_in_epoch=self.batch_in_epoch,
                      sample=identity, sample_sha256=joint96_json_sha256(identity),
                      loss=float(loss.detach()), grad_norm=float(grad_norm),
                      learning_rates=[float(g["lr"]) for g in self.optimizer.param_groups],
                      scheduler_last_epoch=int(self.scheduler.state_dict()["last_epoch"]))
        record["cuda_memory"] = None
        device = self.accelerator.device
        if device.type == "cuda" and torch.cuda.is_initialized():
            free, total = torch.cuda.mem_get_info(device)
            # Keep lifetime peaks, including model construction; do not reset stats.
            record["cuda_memory"] = dict(
                unit="bytes", device=str(device),
                max_memory_allocated=int(torch.cuda.max_memory_allocated(device)),
                max_memory_reserved=int(torch.cuda.max_memory_reserved(device)),
                memory_allocated=int(torch.cuda.memory_allocated(device)),
                free=int(free), total=int(total),
            )
        output = Path(self.output_dir) / "integration_trace"
        output.mkdir(parents=True, exist_ok=True)
        with (output / f"rank_{self.accelerator.process_index:05d}.jsonl").open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")

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
        preserve_order = self.local_joint96 or bool(getattr(dataset, "prefer_sequential_indices", False))
        if preserve_order:
            logger.info("DataLoader preserves grouped indices for random decoded-segment sampling.")
        locality_cfg = self.cfg.get("cache_locality_sampling", None)
        locality_enabled = bool(
            locality_cfg is not None and locality_cfg.get("enabled", False)
        )
        if self.local_joint96:
            if locality_enabled:
                raise ValueError("Local Robot mixture indices must not be reshuffled by a cache sampler")
            mixture = getattr(dataset, "mixture", None)
            if mixture is not None and (
                int(mixture.training_batch_size_per_rank) != self.batch_size
                or int(mixture.bucket_wave_world_size) != int(self.accelerator.num_processes)
            ):
                raise ValueError("Local mixture and Accelerator batch/world geometry differ")
            self.train_sampler = _LocalRobotEpochSampler(
                dataset, self.seed, self.batch_size, self.accelerator.num_processes
            )
        elif locality_enabled:
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
        if self.local_joint96:
            self._local_loader_generator = torch.Generator().manual_seed(self.seed)
            kwargs["generator"] = self._local_loader_generator
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
            drop_last=self.local_joint96,
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
        if multiview_enabled(self.cfg):
            if str(scheduler_type).strip().lower() != "cosine":
                raise ValueError("Multiview budget extension requires capped cosine")
            return CappedWarmupCosineLR(
                self.optimizer,
                schedule_steps=int(self.cfg.local_robot_multiview.schedule_steps),
                warmup_steps=warmup_steps,
                min_lr=self.min_learning_rate,
            )
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
        if self.local_joint96:
            self.load_training_state(str(resume))
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
        prompt_cfg = self.cfg.get("local_robot_softprompt", {})
        if prompt_cfg.get("enabled", False) and prompt_cfg.get("init_checkpoint"):
            from .training.checkpoint import load_joint96_soft_prompt_initialization
            receipt = load_joint96_soft_prompt_initialization(
                model, prompt_cfg.init_checkpoint, expected_step=prompt_cfg.expected_init_step,
                frame_ablation=self.cfg.data.get("frame_ablation", None),
            )
            if self.accelerator.is_main_process:
                output = Path(self.output_dir)
                output.mkdir(parents=True, exist_ok=True)
                (output / "initialization.json").write_text(json.dumps(receipt, indent=2) + "\n")
            logger.info("[soft prompt] weights-only migration from step=%s; new optimizer and step=0",
                        receipt["pretrain_step"])
            return True
        if bool(self.cfg.get("joint96_posttrain", {}).get("enabled", False)):
            if self.resume:
                raise ValueError("Joint96 posttraining initialization is weights-only; resume must be null")
            loader = load_joint96_initialization
            if self.cfg.joint96_posttrain.get("robocasa", {}).get("enabled", False):
                from .training.robocasa_posttrain import load_robocasa_delta80_slot18_initialization
                from .datasets.robocasa_delta80_slot18 import CONTRACT
                if self.cfg.joint96_posttrain.robocasa.frame_contract != CONTRACT["version"]:
                    raise ValueError("This release supports RoboCasa Delta80 slot18 only")
                loader = load_robocasa_delta80_slot18_initialization
            receipt = loader(
                model, self.cfg.init_checkpoint,
                expected_step=self.cfg.joint96_posttrain.expected_pretrain_step,
            )
            if self.accelerator.is_main_process:
                output = Path(self.output_dir)
                output.mkdir(parents=True, exist_ok=True)
                (output / "initialization.json").write_text(json.dumps(receipt, indent=2) + "\n")
            logger.info("[joint96 posttrain] inherited step=%s, fresh optimizer/scheduler and step=0",
                        receipt["pretrain_step"])
            return True
        resume = self.resume
        if not resume:
            return False
        if self.local_joint96:
            if Path(str(resume)).is_file():
                raise ValueError("Local Joint96 weights-only policy initialization is forbidden")
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
        if "embodiment_id" in sample:
            ids = sample["embodiment_id"]
            if not isinstance(ids, torch.Tensor) or ids.dtype != torch.long or ids.ndim not in (0, 1):
                raise ValueError("Eval embodiment_id must be scalar/[B] int64")
            ids = ids.reshape(1) if ids.ndim == 0 else ids
            if ids.shape != (video.shape[0],):
                raise ValueError("Eval embodiment_id batch mismatch")
            batched["embodiment_id"] = ids
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
    def _evaluate_local_joint96(self):
        dataset = self.val_dataset
        panel = getattr(dataset, "fixed_validation_panel", getattr(dataset, "validation_panel", None))
        if panel is None and hasattr(dataset, "eval_steps"):
            seed = int(self.cfg.local_robot_joint96.validation_seed)
            panel = [{"index": index, "source": dataset.datasets[entry[0]].profile["id"], "seed": seed + index}
                     for index, entry in enumerate(dataset.eval_steps)]
        panel = _joint96_panel(panel, len(dataset))
        package_sha256 = getattr(getattr(dataset, "package", None), "fingerprint", None)
        panel_json = json.dumps({"package_sha256": package_sha256, "panel": panel}, sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(panel_json).digest()
        fingerprints = self.accelerator.gather(
            torch.tensor(list(digest), device=self.accelerator.device, dtype=torch.uint8)[None]
        ).reshape(-1, len(digest))
        if not torch.equal(fingerprints, fingerprints[:1].expand_as(fingerprints)):
            raise RuntimeError("local_joint96 validation panel differs across ranks")
        sources = sorted({entry["source"] for entry in panel})
        source_index = {name: index for index, name in enumerate(sources)}
        totals = torch.zeros((len(sources), 1 + len(_JOINT96_PANEL_LOSS_KEYS)), device=self.accelerator.device, dtype=torch.float64)
        registry = getattr(dataset, "embodiment_registry", None)
        embodiment_totals = None
        if registry is not None:
            embodiment_ids = sorted({entry["embodiment_id"] for entry in panel})
            embodiment_index = {value: index for index, value in enumerate(embodiment_ids)}
            embodiment_names = [registry.content["embodiments"][value] for value in embodiment_ids]
            embodiment_totals = totals.new_zeros((len(embodiment_ids), totals.shape[1]))
        rank, world = int(self.accelerator.process_index), int(self.accelerator.num_processes)
        collate = getattr(dataset, "collate_fn", None) or default_collate
        # All ranks execute equal forward counts, including on short final rounds.
        # Padded forwards participate in model collectives but contribute no metrics.
        with _preserve_eval_rng_and_modes(self.model):
            for start in range(0, len(panel), world):
                slot = start + rank
                entry = panel[min(slot, len(panel) - 1)]
                seed = entry["seed"]
                random.seed(seed)
                np.random.seed(seed % (2**32))
                torch.manual_seed(seed)
                sample = collate([dataset[entry["index"]]])
                with self.accelerator.autocast():
                    loss, metrics = self.model(sample, global_step=self.global_step)
                # Joint96 loss must expose local per-sample means, not a value
                # already averaged across ranks with different source identities.
                # Preserve training metric names: contributions and raw terms
                # are separate, with no second application of the 0.5 weight.
                values = [loss, *(metrics[key] for key in _JOINT96_PANEL_LOSS_KEYS[1:])]
                values = torch.stack([
                    torch.as_tensor(value, device=totals.device, dtype=torch.float64).detach().reshape(())
                    for value in values
                ])
                if slot < len(panel):
                    row = source_index[entry["source"]]
                    totals[row, 0] += 1
                    totals[row, 1:] += values
                    if embodiment_totals is not None:
                        erow = embodiment_index[entry["embodiment_id"]]
                        embodiment_totals[erow, 0] += 1
                        embodiment_totals[erow, 1:] += values
        totals = self.accelerator.reduce(totals, reduction="sum")
        if not bool(torch.isfinite(totals).all()):
            raise RuntimeError("Non-finite loss in local_joint96 held-out panel")
        if int(totals[:, 0].sum()) != len(panel):
            raise RuntimeError("local_joint96 validation count mismatch")
        result = _joint96_summarize_panel(sources, totals)
        if embodiment_totals is not None:
            embodiment_totals = self.accelerator.reduce(embodiment_totals, reduction="sum")
            if not bool(torch.isfinite(embodiment_totals).all()) or int(embodiment_totals[:, 0].sum()) != len(panel):
                raise RuntimeError("Soft prompt validation aggregation failed")
            result["embodiments"] = _joint96_summarize_panel(embodiment_names, embodiment_totals)["sources"]
            result["embodiment_registry_sha256"] = registry.sha256
        result.update({"step": int(self.global_step), "optimizer_updates": int(self.global_step), "panel_sha256": digest.hex(), "package_sha256": package_sha256})
        if self.accelerator.is_main_process:
            output = Path(self.eval_dir) / f"step_{self.global_step:06d}"
            output.mkdir(parents=True, exist_ok=True)
            temporary = output / "loss_panel.json.tmp"
            temporary.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            temporary.replace(output / "loss_panel.json")
        self.accelerator.wait_for_everyone()
        return result

    def _local_joint96_periodic(self, checkpoint=None):
        if self.global_step % self.save_every:
            return
        checkpoint = self.save_checkpoint() if checkpoint is None else checkpoint
        self.accelerator.wait_for_everyone()
        if self.global_step % self.eval_every:
            if self.accelerator.is_main_process:
                logger.info(
                    "[local_joint96] checkpoint without eval: step=%d weights=%s state=%s",
                    self.global_step, checkpoint["weights_path"], checkpoint["state_path"],
                )
            self.speed_history = [(self.global_step, time.perf_counter())]
            return
        metrics = self.evaluate()
        if self.accelerator.is_main_process:
            logger.info("[local_joint96] saved before eval: step=%d weights=%s state=%s metrics=%s", self.global_step, checkpoint["weights_path"], checkpoint["state_path"], metrics)
            payload = {f"eval/{key}": value for key, value in metrics.items() if isinstance(value, (int, float))}
            for source, values in metrics["sources"].items():
                payload.update({f"eval/source/{source}/{key}": value for key, value in values.items()})
            for name, values in metrics.get("embodiments", {}).items():
                payload.update({f"eval/embodiment/{name}/{key}": value for key, value in values.items()})
            self._wandb_log(payload)
        self._release_runtime_memory()
        self.accelerator.wait_for_everyone()
        self.speed_history = [(self.global_step, time.perf_counter())]

    @torch.no_grad()
    def evaluate(self):
        if self.local_joint96:
            return self._evaluate_local_joint96()
        if bool(self.cfg.get("joint96_posttrain", {}).get("enabled", False)):
            # Loss panel only: the original pixel-canvas rollout path is not
            # compatible with masked Joint96 camera slots.
            return self._evaluate_local_joint96()
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
            **({"embodiment_ids": sample["embodiment_id"][:1]} if "embodiment_id" in sample else {}),
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
        if self.local_joint96:
            path = os.path.join(self.weights_dir, f"{step_tag}.pt")
            save_wam_checkpoint(model, path, step=self.global_step)
            return path
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
        if self.local_joint96:
            payload.update({"optimizer_updates": int(self.global_step), "world_size": int(self.accelerator.num_processes), "recipe": "local_robot_joint96"})
            payload["joint96_metadata"] = joint96_checkpoint_metadata(self._unwrap_core_model())
            payload["resume_contract"] = self._local_joint96_contract
            history = getattr(self, "_resume_mask_change_history", None)
            if history:
                payload["resume_mask_change_history"] = history
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

        latest_state_path = os.path.join(self.state_dir, step_tag if self.local_joint96 else "latest")
        if self.accelerator.is_main_process:
            ensure_dir(latest_state_path)
        self.accelerator.wait_for_everyone()

        # Keep one optimizer state snapshot; weight checkpoints rotate separately.
        self.accelerator.save_state(output_dir=latest_state_path)
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            self._save_trainer_state(latest_state_path, step_tag=step_tag, weights_path=ckpt_path)
        self.accelerator.wait_for_everyone()

        if self.local_joint96:
            self._complete_local_joint96_checkpoint(latest_state_path, ckpt_path)
            self._last_local_joint96_checkpoint = int(self.global_step)

        return {"weights_path": ckpt_path, "state_path": latest_state_path}

    def _complete_local_joint96_checkpoint(self, state_path, weights_path):
        error = None
        if self.accelerator.is_main_process:
            try:
                root = Path(state_path)
                world = int(self.accelerator.num_processes)
                required = [root / "trainer_state.json", root / "scheduler.bin", Path(weights_path)]
                required.extend(root / f"random_states_{rank}.pkl" for rank in range(world))
                if bool(self.cfg.local_robot_joint96.get("require_deepspeed", True)):
                    optimizers = list(root.rglob("*optim_states.pt"))
                    models = list(root.rglob("*model_states.pt"))
                    if len(optimizers) != world or not models:
                        raise RuntimeError(f"Incomplete ZeRO-1 state: {len(optimizers)} optimizer shards, expected {world}")
                    required.extend([*optimizers, *models])
                missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
                if missing:
                    raise RuntimeError(f"Incomplete checkpoint files: {missing}")
                payload = {"step": int(self.global_step), "optimizer_updates": int(self.global_step), "world_size": world, "weights_path": str(weights_path), "files": [str(path.relative_to(root)) for path in required if path.is_relative_to(root)]}
                state_payload = json.loads((root / "trainer_state.json").read_text())
                payload["resume_contract_sha256"] = joint96_json_sha256(state_payload["resume_contract"])
                payload["trainer_state_sha256"] = joint96_file_sha256(root / "trainer_state.json")
                payload["file_sizes"] = {name: (root / name).stat().st_size for name in payload["files"]}
                temporary = root / "COMPLETE.json.tmp"
                temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
                temporary.replace(root / "COMPLETE.json")
            except Exception as exc:
                error = str(exc)
                logger.error("local_joint96 checkpoint incomplete: %s", error)
        failed = self.accelerator.reduce(torch.tensor(int(error is not None), device=self.accelerator.device), reduction="sum")
        if int(failed.item()):
            raise RuntimeError(f"local_joint96 checkpoint incomplete; eval not started. {error or 'See rank 0 log.'}")
        self.accelerator.wait_for_everyone()

    def load_training_state(self, state_dir: str):
        if self.local_joint96:
            self._load_local_joint96_state(state_dir)
            return
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

    def _load_local_joint96_state(self, state_dir):
        error, checked = None, None
        try:
            checked = validate_joint96_resume_directory(state_dir, cfg=self.cfg,
                world_size=int(self.accelerator.num_processes), expected_contract=self._local_joint96_contract)
        except Exception as exc:
            error = str(exc)
        failed = self.accelerator.reduce(torch.tensor(int(error is not None), device=self.accelerator.device), reduction="sum")
        if int(failed.item()):
            raise ValueError(f"Joint96 resume validation rejected on at least one rank: {error}")
        root, payload = checked
        load_options = dict(load_module_strict=True, load_optimizer_states=True, load_lr_scheduler_states=True)
        if not self.cfg.local_robot_joint96.get("require_deepspeed", True):
            load_options = {}
        self.accelerator.load_state(input_dir=str(root), **load_options)
        error = None
        try:
            self.global_step = int(payload["global_step"])
            self._validate_local_joint96_engine(expected_step=self.global_step)
            if int(self.scheduler.state_dict()["last_epoch"]) != self.global_step:
                raise ValueError("Restored scheduler step differs from absolute optimizer updates")
            lrs = self.scheduler.get_last_lr()
            if [float(g["lr"]) for g in self.optimizer.param_groups] != [float(lr) for lr in lrs]:
                raise ValueError("Restored optimizer LR differs from scheduler")
            self.epoch, self.batch_in_epoch = int(payload["epoch"]), int(payload["batch_in_epoch"])
            self.train_sampler.set_epoch_offset(self.epoch)
            self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)
            self.train_dataset.set_epoch(self.epoch)
            # Accelerate catches RNG restore exceptions. Exact resume cannot.
            rng = torch.load(root / f"random_states_{self.accelerator.process_index}.pkl", map_location="cpu", weights_only=False)
            random.setstate(rng["random_state"])
            np.random.set_state(rng["numpy_random_seed"])
            torch.set_rng_state(rng["torch_manual_seed"])
            if torch.cuda.is_initialized():
                states = rng["torch_cuda_manual_seed"]
                if len(states) != torch.cuda.device_count():
                    raise ValueError("CUDA RNG device geometry changed")
                torch.cuda.set_rng_state_all(states)
            self.accelerator.step = int(rng["step"])
        except Exception as exc:
            error = str(exc)
        failed = self.accelerator.reduce(torch.tensor(int(error is not None), device=self.accelerator.device), reduction="sum")
        if int(failed.item()):
            raise RuntimeError(f"Joint96 loaded state failed counter/scheduler/RNG checks on at least one rank: {error}")
        self._resume_mask_change_history = list(payload.get("resume_mask_change_history", []))
        transition = payload.get("applied_resume_mask_change")
        if transition is not None:
            self._resume_mask_change_history.append(transition)
            logger.info("[local_joint96 resume mask change] %s", json.dumps(transition, sort_keys=True))
        self._local_joint96_resume_checkpoint = {"state_path": str(root), "weights_path": payload.get("weights_path")}
        logger.info("[local_joint96 resume] step=%d epoch=%d consumed_batches=%d source=%s",
                    self.global_step, self.epoch, self.batch_in_epoch, root)

    def _shutdown_distributed(self) -> None:
        self.cache_runtime.close()
        self._finish_wandb()
        self.accelerator.end_training()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

    def _finish_training(self) -> None:
        already_saved = self.local_joint96 and getattr(self, "_last_local_joint96_checkpoint", None) == self.global_step
        if bool(self.cfg.get("save_on_train_end", True)) and not already_saved:
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

        stop_step = self.max_steps
        if self.local_joint96:
            stop_step = int(self.cfg.local_robot_joint96.get("stop_after_step") or self.max_steps)
            if stop_step <= self.global_step:
                raise ValueError("stop_after_step must be after the restored step; max_steps stays unchanged")
            checkpoint = getattr(self, "_local_joint96_resume_checkpoint", None)
            if checkpoint is not None:
                # Replay the fixed panel only at evaluation boundaries; a
                # save-only checkpoint resumes without an extra evaluation.
                self._local_joint96_periodic(checkpoint=checkpoint)

        logger.info("Starting training with max_steps=%d.", self.max_steps)
        # Keep map-style sampling aligned when resuming in the middle of an
        # epoch.  Mix datasets publish this value through shared memory, so
        # persistent workers observe it before their first resumed fetch.
        self.train_sampler.set_epoch_offset(self.epoch)
        if hasattr(self.train_dataset, "set_epoch"):
            self.train_dataset.set_epoch(self.epoch)
        data_iter = self._local_joint96_iterator() if self.local_joint96 else iter(self.train_loader)
        self.speed_history = [(self.global_step, time.perf_counter())]
        timer = TrainingTimer.from_environment(
            self.output_dir, rank=self.accelerator.process_index,
            world_size=self.accelerator.num_processes, device=self.accelerator.device,
        )
        if timer.steps and (not self.local_joint96 or self.gradient_accumulation_steps != 1):
            raise ValueError("Bounded training timing currently requires local_joint96 GAS=1")
        if timer.steps:
            logger.info("[timing] per-rank steps [%d,%d), output=%s; no extra collectives",
                        timer.start_step, timer.start_step + timer.steps, timer.output_dir)

        while self.global_step < stop_step:
            timer.begin(self.global_step)
            try:
                with timer.section("data_next"):
                    sample = next(data_iter)
                self.batch_in_epoch += 1
                if self.cfg.data.train.get("_target_") == "wam.datasets.local_robot_main_view_aug.LocalRobot80MainAugDataset":
                    from wam.datasets.local_robot_main_view_aug import count_batch
                    if not hasattr(self, "_main_aug_counts"):
                        from collections import Counter
                        self._main_aug_sources = sorted({p["source"] for p in self.train_dataset.package.manifest["profiles"].values()})
                        self._main_aug_counts = Counter()
                    self._main_aug_counts.update(count_batch(sample, self._main_aug_sources))
            except StopIteration:
                timer.cancel()
                self.epoch += 1
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                self.train_sampler.set_epoch_offset(self.epoch)
                if hasattr(self.train_dataset, "set_epoch"):
                    self.train_dataset.set_epoch(self.epoch)
                data_iter = self._local_joint96_iterator() if self.local_joint96 else iter(self.train_loader)
                continue

            with self.accelerator.accumulate(self.model):
                with timer.section("forward"), self.accelerator.autocast():
                    loss, loss_dict = self.model(sample, global_step=self.global_step)

                if self.local_joint96:
                    with timer.section("loss_finite_collective"):
                        nonfinite = self.accelerator.reduce((~torch.isfinite(loss.detach())).to(torch.int32), reduction="sum")
                        if int(nonfinite.item()):
                            raise RuntimeError("Non-finite local_joint96 loss; refusing optimizer update")
                update_before = self._local_joint96_update_snapshot() if self.local_joint96 else None
                engine = self.model if hasattr(self.model, "global_steps") else None
                with timer.section("backward_and_engine_step"), time_methods([
                    (engine, "backward", "deepspeed_backward"),
                    (engine, "step", "deepspeed_step"),
                    (getattr(engine, "optimizer", None), "step", "zero_optimizer_step"),
                ]):
                    self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    with timer.section("grad_norm_and_guard"):
                        grad_norm = self.accelerator.clip_grad_norm_(self.grad_clip_params, self.max_grad_norm)
                        if self.local_joint96:
                            self._validate_local_joint96_grad_norm(grad_norm)
                    with timer.section("optimizer_wrapper_scheduler"):
                        self.optimizer.step()
                        if self.local_joint96:
                            self._validate_local_joint96_update(update_before)
                        if not self.accelerator.optimizer_step_was_skipped:
                            self.scheduler.step()
                        self.optimizer.zero_grad(set_to_none=True)

                    self.global_step += 1
                    if self.local_joint96:
                        self._write_local_joint96_trace(sample, loss, grad_norm)
                    self.speed_history.append(
                        (self.global_step, time.perf_counter())
                    )
                    if len(self.speed_history) > 11:
                        del self.speed_history[:-11]
                    should_log = self.log_every > 0 and self.global_step % self.log_every == 0
                    global_loss = None
                    global_loss_metrics = {}
                    global_grad_norm = None
                    global_main_aug_metrics = {}
                    if should_log:
                        with timer.section("metrics_collective"):
                            global_metrics = gather_mean_scalars(
                                self.accelerator,
                                {
                                    "loss": loss.detach(),
                                    "grad_norm": grad_norm,
                                    **loss_dict,
                                    **getattr(self, "_main_aug_counts", {}),
                                },
                                device=loss.device,
                            )
                        global_loss = global_metrics.pop("loss")
                        global_grad_norm = global_metrics.pop("grad_norm")
                        for key in list(global_metrics):
                            if key.startswith("main_aug/"):
                                global_main_aug_metrics[key] = global_metrics.pop(key) * self.accelerator.num_processes
                        if global_main_aug_metrics:
                            groups = ["all", "color_off", "color_on", *self._main_aug_sources]
                            for group in groups:
                                prefix = f"main_aug/{group}/"
                                total = global_main_aug_metrics[prefix + "total"]
                                eligible = global_main_aug_metrics[prefix + "eligible"]
                                for mode in ("mask", "noise"):
                                    count = global_main_aug_metrics[prefix + mode]
                                    global_main_aug_metrics[prefix + mode + "_global_rate"] = count / max(total, 1)
                                    global_main_aug_metrics[prefix + mode + "_eligible_rate"] = count / max(eligible, 1)
                            self._main_aug_counts.clear()
                        global_loss_metrics = global_metrics
                    timing_record = timer.finish(sample=sample)
                    if timing_record is not None and self.accelerator.is_main_process:
                        logger.info("[timing] completed_step=%d local_wall=%.3fs counts=%s",
                                    self.global_step, timing_record["iteration_wall_ms"] / 1000,
                                    timing_record["counts"])

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
                        if self.local_joint96:
                            description += self._local_joint96_log_suffix(global_grad_norm)
                        logger.info(description)
                        if global_main_aug_metrics:
                            logger.info("[main_aug] %s", json.dumps(global_main_aug_metrics, sort_keys=True))

                        wandb_payload = {
                            "train/loss": global_loss,
                            "train/grad_norm": global_grad_norm,
                            "train/lr": current_lr,
                            "performance/steps_per_sec": steps_per_sec,
                            "performance/optimizer_steps_per_sec": steps_per_sec,
                            "performance/samples_per_sec": samples_per_sec,
                            "performance/effective_global_batch_size": effective_global_batch_size,
                        }
                        if self.local_joint96:
                            wandb_payload["train/optimizer_updates"] = int(self.global_step)
                        if current_vlm_lr is not None:
                            wandb_payload["train/vlm_lr"] = current_vlm_lr
                        for key, value in global_loss_metrics.items():
                            wandb_payload[f"train/{key}"] = value
                        wandb_payload.update(global_main_aug_metrics)
                        self._wandb_log(wandb_payload)

                    if self.local_joint96:
                        self._local_joint96_periodic()

                    if (
                        not self.local_joint96
                        and self.eval_every > 0
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

                    if not self.local_joint96 and self.save_every > 0 and self.global_step % self.save_every == 0:
                        ckpt_info = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[ckpt] step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )

                    if self.global_step >= stop_step:
                        self._finish_training()
                        return

        self._finish_training()
        
