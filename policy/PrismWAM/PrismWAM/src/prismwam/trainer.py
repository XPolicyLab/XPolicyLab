import json
import math
import os
import random
import re
import datetime
from contextlib import nullcontext
from math import ceil
from pathlib import Path
import time
import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    LambdaLR,
    LinearLR,
    SequentialLR,
)
from torch.utils.data import DataLoader, default_collate
from .utils.fs import ensure_dir
from .utils.logging_config import get_logger
from .utils.pytorch_utils import set_global_seed
from .utils.samplers import (
    ResumableEpochSampler,
    TaskBalancedEpochSampler,
)
from .utils.wandb_utils import (
    resolve_train_wandb_resume,
    resolve_train_wandb_run_id,
    save_wandb_run_id,
)
from .validation_sampling import (
    assert_robotwin_validation_dataset,
    build_robotwin_episode_validation_manifest,
    build_robodojo_episode_validation_manifest,
    is_robotwin_episode_dataset,
    robotwin_validation_sample_seed,
)

logger = get_logger(__name__)


def training_wandb_metric_key(metric_name: str) -> str:
    """Preserve explicitly namespaced metrics; scope ordinary losses to train/."""
    name = str(metric_name)
    return name if name.startswith(("latent/", "performance/")) else f"train/{name}"


def evaluation_objective_metric_key(metric_name: str) -> str | None:
    """Return the W&B key for metrics produced by validation training_loss."""
    name = str(metric_name)
    if name.startswith(("loss_", "conditioning/")):
        return f"eval/{name}"
    return None


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
        self.pin_memory = bool(cfg.get("pin_memory", torch.cuda.is_available()))
        self.persistent_workers = bool(
            cfg.get("persistent_workers", self.num_workers > 0)
        )
        prefetch_factor = cfg.get("prefetch_factor", None)
        self.prefetch_factor = None if prefetch_factor is None else int(prefetch_factor)
        self.num_epochs = int(cfg.num_epochs)
        max_steps = cfg.max_steps
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.log_every = int(cfg.log_every)
        self.save_every = int(cfg.save_every)
        self.state_save_every = int(cfg.get("state_save_every", self.save_every))
        self.save_at_end = bool(cfg.get("save_at_end", True))
        self.eval_every = int(cfg.eval_every)
        self.eval_samples_per_process = int(cfg.get("eval_samples_per_process", 1))
        if self.eval_samples_per_process <= 0:
            raise ValueError("eval_samples_per_process must be positive.")
        eval_time_budget = cfg.get("eval_time_budget_seconds", None)
        self.eval_time_budget_seconds = (
            None if eval_time_budget is None else float(eval_time_budget)
        )
        self.eval_seed = int(cfg.get("eval_seed", cfg.seed))
        requested_eval_sampling_mode = (
            str(cfg.get("eval_sampling_mode", "auto")).strip().lower()
        )
        supported_eval_sampling_modes = {
            "auto",
            "rank_stratified",
            "robotwin_episode_random",
            "robodojo_episode_random",
        }
        if requested_eval_sampling_mode not in supported_eval_sampling_modes:
            raise ValueError(
                f"Unsupported eval_sampling_mode={requested_eval_sampling_mode!r}; expected one of {sorted(supported_eval_sampling_modes)}."
            )
        self.eval_sampling_mode = requested_eval_sampling_mode
        if self.eval_sampling_mode == "auto":
            self.eval_sampling_mode = (
                "robotwin_episode_random"
                if self.val_dataset is not None
                and is_robotwin_episode_dataset(self.val_dataset)
                else "rank_stratified"
            )
        self.robotwin_validation_manifest = None
        if self.eval_sampling_mode == "robotwin_episode_random":
            if self.val_dataset is None or not is_robotwin_episode_dataset(
                self.val_dataset
            ):
                raise TypeError(
                    "robotwin_episode_random sampling requires a raw RoboTwin episode-indexed validation dataset."
                )
            assert_robotwin_validation_dataset(self.val_dataset)
            self.robotwin_validation_manifest = (
                build_robotwin_episode_validation_manifest(
                    self.val_dataset,
                    seed=self.eval_seed,
                    expected_tasks=int(cfg.get("eval_robotwin_expected_tasks", 50)),
                    expected_episodes_per_task=int(
                        cfg.get("eval_robotwin_episodes_per_task", 10)
                    ),
                )
            )
            logger.info(
                "RoboTwin validation contract: tasks=%d episodes_per_task=%d samples=%d seed=%d manifest_sha256=%s",
                self.robotwin_validation_manifest["task_count"],
                self.robotwin_validation_manifest["episodes_per_task"],
                self.robotwin_validation_manifest["sample_count"],
                self.robotwin_validation_manifest["seed"],
                self.robotwin_validation_manifest["manifest_sha256"],
            )
        if self.eval_sampling_mode == "robodojo_episode_random":
            from prismwam.robodojo_split import validate_validation_pair

            validate_validation_pair(self.train_dataset, self.val_dataset)
            self.robotwin_validation_manifest = (
                build_robodojo_episode_validation_manifest(
                    self.val_dataset, seed=self.eval_seed
                )
            )
            logger.info(
                "RoboDojo validation: samples=%d seed=%d manifest_sha256=%s",
                self.robotwin_validation_manifest["sample_count"],
                self.eval_seed,
                self.robotwin_validation_manifest["manifest_sha256"],
            )
        if (
            self.eval_time_budget_seconds is not None
            and self.eval_time_budget_seconds <= 0.0
        ):
            raise ValueError("eval_time_budget_seconds must be positive or null.")
        self.gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
        self.sampling_mode = str(cfg.get("sampling_mode", "epoch")).strip().lower()
        if self.sampling_mode not in {"epoch", "task_balanced"}:
            raise ValueError(
                f"Unsupported sampling_mode={self.sampling_mode!r}; expected 'epoch' or 'task_balanced'."
            )
        self.max_grad_norm = float(cfg.max_grad_norm)
        self.seed = int(cfg.seed)
        self.resume = cfg.resume
        self.mixed_precision = str(cfg.mixed_precision).strip().lower()
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError(
                f"Unsupported mixed_precision: {cfg.mixed_precision}. Expected one of: ['no', 'fp16', 'bf16']."
            )
        self.wandb_enabled = bool(cfg.wandb.enabled)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            step_scheduler_with_optimizer=False,
        )
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage = (
            "disabled"
            if deepspeed_plugin is None
            else deepspeed_plugin.deepspeed_config.get("zero_optimization", {}).get(
                "stage", "unknown"
            )
        )
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
        self._apply_train_mode(self.model)
        optimizer_parameters = [p for p in self.model.parameters() if p.requires_grad]
        if not optimizer_parameters:
            raise RuntimeError(
                "No trainable parameters were selected for the optimizer."
            )
        self.optimizer = torch.optim.AdamW(
            optimizer_parameters,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )
        self.train_loader = self._build_loader(
            self.train_dataset, worker_init_fn=worker_init_fn
        )
        total_train_steps = self._estimate_total_train_steps()
        self.max_steps = total_train_steps
        warmup_steps = int(total_train_steps * 0.05)
        if callable(getattr(self.model, "build_lr_lambdas", None)):
            lr_lambdas = self.model.build_lr_lambdas(
                param_groups=self.optimizer.param_groups,
                total_steps=total_train_steps,
                warmup_steps=warmup_steps,
            )
            self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambdas)
        else:
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
        ensure_dir(self.output_dir)
        ensure_dir(self.checkpoint_root)
        ensure_dir(self.weights_dir)
        ensure_dir(self.state_dir)
        ensure_dir(self.eval_dir)
        if self.robotwin_validation_manifest is not None:
            manifest_path = (
                Path(self.eval_dir) / f"validation_samples_seed{self.eval_seed}.json"
            )
            if self.accelerator.is_main_process:
                if manifest_path.exists():
                    recorded_manifest = json.loads(
                        manifest_path.read_text(encoding="utf-8")
                    )
                    if recorded_manifest != self.robotwin_validation_manifest:
                        raise RuntimeError(
                            f"Recorded RoboTwin validation manifest differs from the active dataset: {manifest_path}."
                        )
                else:
                    temporary = manifest_path.with_suffix(
                        manifest_path.suffix + f".tmp.{os.getpid()}"
                    )
                    temporary.write_text(
                        json.dumps(
                            self.robotwin_validation_manifest, indent=2, sort_keys=True
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    os.replace(temporary, manifest_path)
            self.accelerator.wait_for_everyone()
        (self.model, self.optimizer, self.train_loader, self.scheduler) = (
            self.accelerator.prepare(
                self.model, self.optimizer, self.train_loader, self.scheduler
            )
        )
        self.optimizer.zero_grad(set_to_none=True)
        self.wandb_run = None
        self._init_wandb()
        self._resume_or_load_checkpoint()
        val_size = (
            len(self.val_dataset)
            if self.val_dataset is not None
            else len(self.train_dataset)
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
        run_id = resolve_train_wandb_run_id(self.output_dir, self.cfg)
        self.wandb_run = wandb.init(
            entity=self.cfg.wandb.workspace,
            project=self.cfg.wandb.project,
            name=self.cfg.wandb.name,
            mode=self.cfg.wandb.mode,
            dir=self.output_dir,
            id=run_id,
            resume=resolve_train_wandb_resume(self.output_dir, self.cfg),
            config=OmegaConf.to_container(self.cfg, resolve=True),
        )
        save_wandb_run_id(self.output_dir, self.cfg, self.wandb_run)
        logger.info(
            "Initialized wandb run: workspace=%s project=%s name=%s run_id=%s",
            self.cfg.wandb.workspace,
            self.cfg.wandb.project,
            self.cfg.wandb.name,
            self.wandb_run.id,
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
        if self.sampling_mode == "task_balanced":
            group_builder = getattr(dataset, "task_index_groups", None)
            if not callable(group_builder):
                raise TypeError(
                    "sampling_mode=task_balanced requires dataset.task_index_groups()."
                )
            task_to_indices = group_builder()
            self.train_sampler = TaskBalancedEpochSampler(
                dataset=dataset,
                task_to_indices=task_to_indices,
                seed=self.seed,
                batch_size=self.batch_size,
                num_processes=self.accelerator.num_processes,
                gradient_accumulation_steps=self.gradient_accumulation_steps,
            )
            logger.info(
                "Using task-balanced sampler: tasks=%d global_batch=%d layout=%s",
                len(task_to_indices),
                self.train_sampler.global_batch_size,
                self.train_sampler.LAYOUT_VERSION,
            )
        else:
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
            "pin_memory": self.pin_memory,
            "worker_init_fn": worker_init_fn,
            "collate_fn": default_collate,
        }
        # Keep loader construction independent of the restored model/diffusion RNG.
        loader_kwargs["generator"] = torch.Generator().manual_seed(
            self.seed + self.accelerator.process_index
        )
        if self.num_workers > 0:
            loader_kwargs["persistent_workers"] = self.persistent_workers
            if self.prefetch_factor is not None:
                loader_kwargs["prefetch_factor"] = self.prefetch_factor
        return DataLoader(dataset, **loader_kwargs)

    def _assert_dataset_length_consistent(self, dataset, dataset_name: str):
        if not hasattr(dataset, "__len__"):
            raise TypeError(
                f"`{dataset_name}` must implement __len__ for rank consistency checks."
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
            raise TypeError(
                "`train_dataset` must implement __len__ when `max_steps` is None."
            )
        num_processes = max(int(self.accelerator.num_processes), 1)
        global_batch_size = max(self.batch_size * num_processes, 1)
        micro_steps_per_epoch = max(
            ceil(len(self.train_dataset) / global_batch_size), 1
        )
        opt_steps_per_epoch = max(
            ceil(micro_steps_per_epoch / self.gradient_accumulation_steps), 1
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
                self.optimizer, T_max=remaining_steps, eta_min=self.learning_rate * 0.01
            )
        elif scheduler_type == "constant":
            main_scheduler = ConstantLR(
                self.optimizer, factor=1.0, total_iters=remaining_steps
            )
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
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )

    def _estimate_progress(self):
        elapsed = max(time.perf_counter() - self.run_start_time, 1e-06)
        done_steps = max(self.global_step - self.run_start_step, 1)
        seconds_per_step = elapsed / done_steps
        steps_per_sec = 1.0 / max(seconds_per_step, 1e-09)
        remaining_steps = max(self.max_steps - self.global_step, 0)
        eta_seconds = int(seconds_per_step * remaining_steps)
        elapsed_seconds = int(seconds_per_step * done_steps)
        eta_str = str(datetime.timedelta(seconds=eta_seconds))
        elapsed_str = str(datetime.timedelta(seconds=elapsed_seconds))
        return (eta_str, elapsed_str, steps_per_sec, eta_seconds, elapsed_seconds)

    def _resume_or_load_checkpoint(self):
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
        self.accelerator.unwrap_model(self.model).load_checkpoint(
            str(resume_path), optimizer=None
        )
        logger.warning(
            "Loaded .pt weights only; optimizer/scheduler/step were not restored under ZeRO2."
        )

    def _set_dit_only_train_mode(self):
        model = self.accelerator.unwrap_model(self.model)
        self._apply_train_mode(model)

    def _apply_train_mode(self, model):
        self._apply_full_dit_train_mode(model)

    @staticmethod
    def _apply_full_dit_train_mode(model):
        model.eval()
        model.requires_grad_(False)
        model.dit.train()
        model.dit.requires_grad_(True)
        configure_trainability = getattr(model, "configure_full_dit_trainability", None)
        if callable(configure_trainability):
            configure_trainability()
        proprio_encoder = getattr(model, "proprio_encoder", None)
        if proprio_encoder is not None:
            proprio_encoder.train()
            proprio_encoder.requires_grad_(True)

    @staticmethod
    def _fast_validation_indices(
        *,
        dataset_size: int,
        process_index: int,
        num_processes: int,
        samples_per_process: int,
    ) -> list[int]:
        """Choose a fixed, rank-stratified subset for comparable validation curves."""
        dataset_size = int(dataset_size)
        process_index = int(process_index)
        num_processes = int(num_processes)
        samples_per_process = int(samples_per_process)
        if dataset_size <= 0:
            raise ValueError("Loss-only validation requires a non-empty dataset.")
        if num_processes <= 0 or samples_per_process <= 0:
            raise ValueError("Validation process and sample counts must be positive.")
        if process_index < 0 or process_index >= num_processes:
            raise ValueError(
                f"Invalid validation process index {process_index}/{num_processes}."
            )
        global_slots = num_processes * samples_per_process
        first_slot = process_index * samples_per_process
        return [
            min((first_slot + offset) * dataset_size // global_slots, dataset_size - 1)
            for offset in range(samples_per_process)
        ]

    @torch.no_grad()
    def _evaluate_loss_only(self):
        model = self.accelerator.unwrap_model(self.model)
        num_processes = int(getattr(self.accelerator, "num_processes", 1))
        sampling_mode = str(getattr(self, "eval_sampling_mode", "rank_stratified"))
        if sampling_mode in ("robotwin_episode_random", "robodojo_episode_random"):
            manifest = getattr(self, "robotwin_validation_manifest", None)
            if manifest is None:
                raise RuntimeError("RoboTwin validation manifest was not initialized.")
            all_selections = list(manifest["samples"])
            local_selections = all_selections[
                int(self.accelerator.process_index) :: num_processes
            ]
            eval_selections = [
                (int(row["dataset_index"]), int(row["selection_id"]))
                for row in local_selections
            ]
        else:
            eval_indices = self._fast_validation_indices(
                dataset_size=len(self.val_dataset),
                process_index=self.accelerator.process_index,
                num_processes=num_processes,
                samples_per_process=self.eval_samples_per_process,
            )
            eval_selections = [(int(index), None) for index in eval_indices]
        if not eval_selections:
            raise ValueError(
                "Loss-only validation assigned no samples to this process."
            )
        python_rng_state = random.getstate()
        numpy_rng_state = np.random.get_state()
        cuda_devices = []
        self.accelerator.wait_for_everyone()
        if self.accelerator.device.type == "cuda":
            cuda_devices = [torch.cuda.current_device()]
            torch.cuda.synchronize(self.accelerator.device)
        started = time.perf_counter()
        try:
            validation_seed = int(getattr(self, "eval_seed", 42)) + int(
                self.accelerator.process_index
            )
            metric_sums: dict[str, torch.Tensor] = {}
            validation_context = getattr(model, "loss_validation_context", nullcontext)
            with torch.random.fork_rng(devices=cuda_devices), validation_context():
                random.seed(validation_seed)
                np.random.seed(validation_seed % 2**32)
                torch.manual_seed(validation_seed)
                if cuda_devices:
                    torch.cuda.manual_seed(validation_seed)
                for eval_index, selection_id in eval_selections:
                    if selection_id is not None:
                        sample_seed = robotwin_validation_sample_seed(
                            int(getattr(self, "eval_seed", 42)), selection_id
                        )
                        random.seed(sample_seed)
                        np.random.seed(sample_seed % 2**32)
                        torch.manual_seed(sample_seed)
                        if cuda_devices:
                            torch.cuda.manual_seed(sample_seed)
                    sample = default_collate([self.val_dataset[eval_index]])
                    with torch.enable_grad(), self.accelerator.autocast():
                        (val_loss, loss_dict) = model.training_loss(sample)
                    values = {"val_loss": val_loss.detach().float()}
                    values.update(
                        {
                            str(name): val_loss.new_tensor(float(value)).float()
                            for (name, value) in loss_dict.items()
                        }
                    )
                    for name, value in values.items():
                        metric_sums[name] = metric_sums.get(name, 0.0) + value
            metric_names = sorted(metric_sums)
            if "val_loss" in metric_names:
                metric_names.remove("val_loss")
                metric_names.insert(0, "val_loss")
            if self.accelerator.device.type == "cuda":
                torch.cuda.synchronize(self.accelerator.device)
            local_duration = time.perf_counter() - started
            reference = metric_sums["val_loss"]
            local_metrics = torch.stack(
                [metric_sums[name] for name in metric_names]
                + [
                    reference.new_tensor(len(eval_selections)),
                    reference.new_tensor(local_duration),
                ]
            ).unsqueeze(0)
            gathered_metrics = self.accelerator.gather_for_metrics(local_metrics)
            sample_counts = gathered_metrics[:, -2]
            global_samples = int(sample_counts.sum().item())
            result = {
                name: float((gathered_metrics[:, index].sum() / global_samples).item())
                for (index, name) in enumerate(metric_names)
            }
            result.update(
                {
                    "duration_seconds": float(gathered_metrics[:, -1].max().item()),
                    "samples_per_process": int(sample_counts.max().item()),
                    "samples_per_process_min": int(sample_counts.min().item()),
                    "global_samples": global_samples,
                }
            )
        finally:
            random.setstate(python_rng_state)
            np.random.set_state(numpy_rng_state)
        non_finite_metrics = sorted(
            (
                name
                for (name, value) in result.items()
                if isinstance(value, float) and (not math.isfinite(value))
            )
        )
        if non_finite_metrics:
            raise FloatingPointError(
                f"Loss-only validation produced non-finite metrics after distributed aggregation: {non_finite_metrics}."
            )
        if (
            self.eval_time_budget_seconds is not None
            and result["duration_seconds"] > self.eval_time_budget_seconds
        ):
            raise RuntimeError(
                f"Loss-only validation exceeded its wall-time budget: {result['duration_seconds']:.3f}s > {self.eval_time_budget_seconds:.3f}s."
            )
        return result

    def evaluate(self):
        if self.val_dataset is None:
            return None
        return self._evaluate_loss_only()

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
            "sampler": self.train_sampler.state_metadata(),
        }
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    def save_checkpoint(self, *, save_training_state: bool = True):
        step_tag = f"step_{self.global_step:06d}"
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            ckpt_path = self._save_weights_checkpoint(step_tag=step_tag)
        else:
            ckpt_path = None
        self.accelerator.wait_for_everyone()
        state_path = None
        if save_training_state:
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
            self.train_sampler.validate_resume_metadata(payload.get("sampler"))
            if "epoch" in payload and "batch_in_epoch" in payload:
                self.epoch = int(payload["epoch"])
                self.batch_in_epoch = int(payload["batch_in_epoch"])
                self.train_sampler.set_epoch(self.epoch)
                set_loader_epoch = getattr(self.train_loader, "set_epoch", None)
                if callable(set_loader_epoch):
                    set_loader_epoch(self.epoch)
                self.train_sampler.set_epoch_offset(0)
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
        if self.max_steps is None:
            raise ValueError(
                "`max_steps` must be set before entering the while-step training loop."
            )
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
                self.train_sampler.set_epoch(self.epoch)
                self.train_sampler.clear_resume_batch_offset()
                data_iter = iter(self.train_loader)
                continue
            with self.accelerator.accumulate(self.model):
                with self.accelerator.autocast():
                    # Enter the distributed wrapper so DDP/ZeRO forward hooks run.
                    loss, loss_dict = self.model(sample)
                if not bool(torch.isfinite(loss.detach()).all().item()):
                    raise FloatingPointError(
                        f"Non-finite training loss before backward at optimizer_step={self.global_step}: loss={loss.detach().float().item()} metrics={loss_dict}."
                    )
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    grad_norm = self.accelerator.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    if grad_norm is not None:
                        grad_norm_tensor = torch.as_tensor(
                            grad_norm,
                            device=self.accelerator.device,
                            dtype=torch.float32,
                        )
                        if not bool(torch.isfinite(grad_norm_tensor).all().item()):
                            raise FloatingPointError(
                                f"Non-finite gradient norm before optimizer step {self.global_step}: grad_norm={grad_norm_tensor.detach().cpu().tolist()}"
                            )
                    self.optimizer.step()
                    optimizer_step_was_skipped = bool(
                        self.accelerator.optimizer_step_was_skipped
                    )
                    if not optimizer_step_was_skipped:
                        self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
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
                        gathered_metric = self.accelerator.gather(metric_tensor)
                        global_loss_metrics[key] = float(
                            (
                                gathered_metric.max()
                                if str(key).startswith("performance/")
                                else gathered_metric.mean()
                            ).item()
                        )
                    global_grad_norm = None
                    if grad_norm is not None:
                        if isinstance(grad_norm, torch.Tensor):
                            grad_norm_tensor = (
                                grad_norm.detach()
                                .to(device=loss.device, dtype=torch.float32)
                                .reshape(1)
                            )
                        else:
                            grad_norm_tensor = torch.tensor(
                                [float(grad_norm)],
                                device=loss.device,
                                dtype=torch.float32,
                            )
                        global_grad_norm = float(
                            self.accelerator.gather(grad_norm_tensor).mean().item()
                        )
                    current_lr = float(self.optimizer.param_groups[0]["lr"])
                    group_lrs = {
                        str(group.get("name", f"group_{index}")): float(group["lr"])
                        for (index, group) in enumerate(self.optimizer.param_groups)
                    }
                    if (
                        self.log_every > 0
                        and self.global_step % self.log_every == 0
                        and self.accelerator.is_main_process
                    ):
                        (
                            eta_str,
                            elapsed_str,
                            steps_per_sec,
                            eta_seconds,
                            elapsed_seconds,
                        ) = self._estimate_progress()
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
                                    for (k, v) in sorted(global_loss_metrics.items())
                                ]
                            )
                            description += detail_str + " "
                        description += (
                            "lr=%.2e speed=%.2f step/s, %.2f samples/s elapsed=%s eta=%s"
                            % (
                                current_lr,
                                steps_per_sec,
                                steps_per_sec
                                * self.batch_size
                                * self.accelerator.num_processes,
                                elapsed_str,
                                eta_str,
                            )
                        )
                        logger.info(description)
                        wandb_payload = {
                            "train/loss": global_loss,
                            "train/lr": current_lr,
                            "train/eta": eta_seconds / 3600.0,
                            "train/elapsed": elapsed_seconds / 3600.0,
                            "performance/steps_per_sec": steps_per_sec,
                            "performance/samples_per_sec": steps_per_sec
                            * self.batch_size
                            * self.accelerator.num_processes,
                        }
                        for group_name, group_lr in group_lrs.items():
                            wandb_payload[f"train/lr_{group_name}"] = group_lr
                        if global_grad_norm is not None:
                            wandb_payload["train/grad_norm"] = global_grad_norm
                        for key, value in global_loss_metrics.items():
                            wandb_payload[training_wandb_metric_key(key)] = value
                        self._wandb_log(wandb_payload)
                    if (
                        self.eval_every > 0
                        and self.val_dataset is not None
                        and (self.global_step % self.eval_every == 0)
                    ):
                        metrics = self.evaluate()
                        self.accelerator.wait_for_everyone()
                        if metrics is not None and self.accelerator.is_main_process:
                            description = "[eval] step=%d val_loss=%.4f" % (
                                self.global_step,
                                metrics["val_loss"],
                            )
                            if "duration_seconds" in metrics:
                                description += " duration=%.3fs samples=%d" % (
                                    metrics["duration_seconds"],
                                    metrics["global_samples"],
                                )
                            for metric_name in sorted(metrics):
                                if (
                                    evaluation_objective_metric_key(metric_name)
                                    is not None
                                ):
                                    description += " %s=%.4f" % (
                                        metric_name,
                                        metrics[metric_name],
                                    )
                            logger.info(description)
                            eval_payload = {"eval/val_loss": float(metrics["val_loss"])}
                            if "duration_seconds" in metrics:
                                eval_payload["eval/duration_seconds"] = float(
                                    metrics["duration_seconds"]
                                )
                                eval_payload["eval/global_samples"] = int(
                                    metrics["global_samples"]
                                )
                            for metric_name, metric_value in metrics.items():
                                metric_key = evaluation_objective_metric_key(
                                    metric_name
                                )
                                if metric_key is not None:
                                    eval_payload[metric_key] = float(metric_value)
                            self._wandb_log(eval_payload)
                    if self.save_every > 0 and self.global_step % self.save_every == 0:
                        save_training_state = (
                            self.state_save_every > 0
                            and self.global_step % self.state_save_every == 0
                        )
                        ckpt_info = self.save_checkpoint(
                            save_training_state=save_training_state
                        )
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[ckpt] step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )
                    if self.global_step >= self.max_steps:
                        already_saved = (
                            self.save_every > 0
                            and self.global_step % self.save_every == 0
                        )
                        if self.save_at_end and (not already_saved):
                            ckpt_info = self.save_checkpoint()
                        elif not already_saved:
                            ckpt_info = {"weights_path": None, "state_path": None}
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[done] max_steps reached step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )
                        return
        if self.save_at_end:
            ckpt_info = self.save_checkpoint()
        else:
            ckpt_info = {"weights_path": None, "state_path": None}
        if self.accelerator.is_main_process:
            logger.info(
                "[done] training finished step=%d weights=%s state=%s",
                self.global_step,
                ckpt_info["weights_path"],
                ckpt_info["state_path"],
            )
