import logging
import json
import inspect
import os
import re
import shutil
from math import ceil
from pathlib import Path
import time
import warnings

import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from .utils.fs import ensure_dir
from .utils.logging_config import get_logger
from .utils.pytorch_utils import set_global_seed
from .utils.samplers import ResumableEpochSampler, WeightedResumableEpochSampler
from .utils.video_io import save_mp4
from .utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim

logger = get_logger(__name__)


class Wan22Trainer:
    def __init__(self, model, train_dataset, val_dataset=None, *, cfg: DictConfig):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = cfg
        # Propagate composite_layout from cfg.model to the model instance so
        # build_inputs (in flexpi.py / flexpi_xwam.py) can dispatch the per-cam
        # compose function. Defaulting to the legacy asymmetric T preserves all
        # existing behavior;
        # a missing key OR an explicit null both fall back to the default. Set
        # BEFORE accelerator.prepare so the attribute lives on the underlying
        # module, which `self` resolves to inside build_inputs.
        _cl_raw = getattr(cfg.model, "composite_layout", None)
        _composite_layout = str(_cl_raw) if _cl_raw is not None else "tshape_robotwin_384x320"
        setattr(self.model, "composite_layout", _composite_layout)
        self.output_dir = str(cfg.output_dir)
        self.learning_rate = float(cfg.learning_rate)
        self.weight_decay = float(cfg.weight_decay)
        self.batch_size = int(cfg.batch_size)
        self.num_workers = int(cfg.num_workers)
        self.num_epochs = float(cfg.num_epochs)   # fractional epochs allowed (e.g. 0.5)
        max_steps = cfg.max_steps
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.log_every = int(cfg.log_every)
        self.save_every = int(cfg.save_every)
        self.eval_every = int(cfg.eval_every)
        self.eval_num_inference_steps = int(cfg.eval_num_inference_steps)
        # When False, intermediate eval skips the expensive main video rollout
        # (model.infer) + VAE-recon viz + mp4, keeping val-loss. Lean eval for
        # FlexPi's no-test-time-video deployment. Default True = full eval.
        self.eval_video = bool(getattr(cfg, "eval_video", True))
        self.gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
        self.max_grad_norm = float(cfg.max_grad_norm)
        self.seed = int(cfg.seed)

        self.resume = cfg.resume
        # Weights-only warm-init from a pretrained ckpt. Distinct semantics from
        # `resume` — see configs/train.yaml. Defaults preserve all existing
        # callers (key absent → None).
        self.pretrained_ckpt = getattr(cfg, "pretrained_ckpt", None)
        self.pretrained_ckpt_strict_shape = bool(
            getattr(cfg, "pretrained_ckpt_strict_shape", True)
        )
        if self.resume and self.pretrained_ckpt:
            raise ValueError(
                "`cfg.resume` and `cfg.pretrained_ckpt` are mutually exclusive: "
                "`resume` restores full training state to continue the SAME run; "
                "`pretrained_ckpt` warm-inits a NEW run from external weights. "
                f"Got resume={self.resume!r}, pretrained_ckpt={self.pretrained_ckpt!r}."
            )
        self.mixed_precision = str(cfg.mixed_precision).strip().lower()
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError(
                f"Unsupported mixed_precision: {cfg.mixed_precision}. "
                "Expected one of: ['no', 'fp16', 'bf16']."
            )
        self.wandb_enabled = bool(cfg.wandb.enabled)

        _accel_kwargs = dict(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            step_scheduler_with_optimizer=False,
        )
        self.accelerator = Accelerator(**_accel_kwargs)

        ds_plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
        zero_stage = (
            ds_plugin.deepspeed_config.get("zero_optimization", {}).get("stage", "unknown")
            if ds_plugin is not None else "N/A"
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

        # Freeze non-trainable modules (VAE, text encoder) before optimizer/deepspeed initialization.
        # _apply_dit_only_train_mode unfreezes dit + any model-specific trainable layers.
        self._apply_dit_only_train_mode(self.model)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = self._build_optimizer(trainable_params)

        self.train_loader = self._build_loader(self.train_dataset, worker_init_fn=worker_init_fn)
        total_train_steps = self._estimate_total_train_steps()
        self.max_steps = total_train_steps
        warmup_steps = int(total_train_steps * 0.05)
        self.scheduler = self._build_scheduler(
            total_train_steps=total_train_steps,
            warmup_steps=warmup_steps,
        )
        # Peak base LRs the schedule anchors to, captured before accelerator.prepare()
        # and any checkpoint restore. Reused by _reanchor_lr_schedule() to rebuild the
        # cosine at the original peak when resuming into an extended horizon.
        self._scheduler_base_lrs = [
            float(g.get("initial_lr", g["lr"])) for g in self.optimizer.param_groups
        ]
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

        self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.scheduler
        )
        self.optimizer.zero_grad(set_to_none=True)

        self.wandb_run = None
        self._init_wandb()
        self._resume_or_load_checkpoint()

        val_size = len(self.val_dataset) if self.val_dataset is not None else len(self.train_dataset)
        logger.info("Train/val dataset size: %d/%d", len(self.train_dataset), val_size)

    def _build_optimizer(self, trainable_params):
        adam_betas = tuple(getattr(self.cfg, "adam_betas", [0.9, 0.95]))
        logger.info(
            "Using AdamW optimizer: lr=%.2e wd=%.2e betas=%s",
            self.learning_rate, self.weight_decay, adam_betas,
        )
        return torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=adam_betas,
        )

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
        # Weighted multi-dataset frame sampling: when the dataset exposes
        # per-frame `dataset_weights`, draw frames with-replacement at the target
        # ratio (size-agnostic) instead of one uniform pass. Epoch length is the
        # dataset's `samples_per_epoch`. Single-dataset / unweighted falls back
        # to the uniform-permutation sampler.
        dataset_weights = getattr(dataset, "dataset_weights", None)
        if dataset_weights is not None:
            self.train_sampler = WeightedResumableEpochSampler(
                per_dataset_num_frames=dataset.per_dataset_num_frames,
                weights=dataset_weights,
                samples_per_epoch=dataset.samples_per_epoch,
                seed=self.seed,
                batch_size=self.batch_size,
                num_processes=self.accelerator.num_processes,
            )
        else:
            self.train_sampler = ResumableEpochSampler(
                dataset=dataset,
                seed=self.seed,
                batch_size=self.batch_size,
                num_processes=self.accelerator.num_processes,
            )
        # persistent_workers + prefetch_factor: free latency-hiding for the
        # map-style path. PyTorch defaults are persistent_workers=False (workers
        # respawn every epoch, ~15 s setup at YAM scale) and prefetch_factor=2
        # (only ~1.3 batches ahead per worker). The iterable-dataset branch
        # above pins prefetch_factor=2 deliberately; the map-style branch
        # benefits from a deeper buffer because each sample is expensive.
        kwargs = dict(
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=worker_init_fn,
        )
        if self.num_workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = 4
        return DataLoader(dataset, **kwargs)

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

        # Epoch length = the sampler's per-epoch sample count. For the weighted
        # sampler this is `samples_per_epoch` (decoupled from dataset size); for
        # the uniform sampler / no sampler it equals len(train_dataset).
        sampler = getattr(self, "train_sampler", None)
        if sampler is not None:
            epoch_len = len(sampler)
        elif hasattr(self.train_dataset, "__len__"):
            epoch_len = len(self.train_dataset)
        else:
            raise TypeError("`train_dataset` must implement __len__ when `max_steps` is None.")

        num_processes = max(int(self.accelerator.num_processes), 1)
        global_batch_size = max(self.batch_size * num_processes, 1)
        micro_steps_per_epoch = max(ceil(epoch_len / global_batch_size), 1)
        opt_steps_per_epoch = max(
            ceil(micro_steps_per_epoch / self.gradient_accumulation_steps),
            1,
        )
        return max(ceil(opt_steps_per_epoch * self.num_epochs), 1)

    def _build_scheduler(self, total_train_steps: int, warmup_steps: int = 0):
        """Linear warmup into cosine decay to 1% of peak LR."""
        total_train_steps = max(int(total_train_steps), 1)
        warmup_steps = min(max(int(warmup_steps), 0), total_train_steps - 1)

        remaining_steps = max(total_train_steps - warmup_steps, 1)
        main_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=remaining_steps,
            eta_min=self.learning_rate * 0.01,
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
        remaining_steps = max(self.max_steps - self.global_step, 0)
        eta_seconds = int(remaining_steps / max(steps_per_sec, 1e-9))
        eta_h, eta_rem = divmod(eta_seconds, 3600)
        eta_m, eta_s = divmod(eta_rem, 60)
        return f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}", steps_per_sec

    def _resume_or_load_checkpoint(self):
        resume = self.resume
        if resume:
            resume_path = Path(str(resume))
            if resume_path.is_dir():
                logger.info("Resuming full training state from directory: %s", resume)
                self.load_training_state(str(resume_path))
                if getattr(self.cfg, "resume_reanchor_lr_schedule", False):
                    self._reanchor_lr_schedule()
                return
            if not resume_path.exists():
                raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
            logger.info("Loading weight checkpoint only: %s", resume)
            self.accelerator.unwrap_model(self.model).load_checkpoint(str(resume_path), optimizer=None)
            logger.warning("Loaded .pt weights only; optimizer/scheduler/step were not restored under ZeRO2.")
            self._check_pretrain_norm_mode_compat(resume_path)
            return

        if self.pretrained_ckpt:
            pt_path = self._resolve_pretrained_ckpt_path(self.pretrained_ckpt)
            logger.info(
                "Loading pretrained weights (warm-init, fresh optimizer/scheduler/step): %s",
                pt_path,
            )
            self.accelerator.unwrap_model(self.model).load_checkpoint(
                str(pt_path),
                optimizer=None,
                strict_shape=self.pretrained_ckpt_strict_shape,
            )
            self._check_pretrain_norm_mode_compat(pt_path)

    @staticmethod
    def _resolve_pretrained_ckpt_path(raw: str) -> Path:
        """Resolve `cfg.pretrained_ckpt` to a concrete `.pt` file.

        Accepts:
          * A `.pt` file path — returned as-is after existence check.
          * A `checkpoints/state/step_NNNNNN/` directory — auto-resolved to the
            sibling `checkpoints/weights/step_NNNNNN.pt`.
          * A run_dir (anything containing `checkpoints/weights/step_*.pt`) —
            the highest-numbered step is selected.

        Raises FileNotFoundError with the candidate paths tried so failures
        are unambiguous.
        """
        p = Path(str(raw)).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"pretrained_ckpt path does not exist: {p}")
        if p.is_file():
            if p.suffix != ".pt":
                logger.warning(
                    "pretrained_ckpt %s is a file but lacks `.pt` extension; "
                    "loading anyway via model.load_checkpoint.", p,
                )
            return p
        # Directory cases.
        # Case 1: state/step_NNNNNN/ → sibling weights/step_NNNNNN.pt
        step_match = re.match(r"^step[_-](\d+)$", p.name)
        if step_match and p.parent.name == "state":
            run_ckpt_root = p.parent.parent  # checkpoints/
            candidate = run_ckpt_root / "weights" / f"{p.name}.pt"
            if candidate.exists():
                return candidate
            raise FileNotFoundError(
                f"pretrained_ckpt resolved from state dir {p} expected sibling "
                f"weights file at {candidate}, which is missing."
            )
        # Case 2: run_dir → checkpoints/weights/step_*.pt (latest)
        weights_dir = p / "checkpoints" / "weights"
        if weights_dir.is_dir():
            pts = sorted(
                weights_dir.glob("step_*.pt"),
                key=lambda fp: int(re.search(r"step[_-](\d+)", fp.stem).group(1)),
            )
            if pts:
                return pts[-1]
            raise FileNotFoundError(
                f"pretrained_ckpt resolved run_dir {p} but no step_*.pt under {weights_dir}."
            )
        # Case 3: bare checkpoints/weights/ directory
        if p.name == "weights" and (p / "..").resolve().name == "checkpoints":
            pts = sorted(
                p.glob("step_*.pt"),
                key=lambda fp: int(re.search(r"step[_-](\d+)", fp.stem).group(1)),
            )
            if pts:
                return pts[-1]
        raise FileNotFoundError(
            f"pretrained_ckpt {p} is a directory but does not match any "
            "expected layout: .pt file | state/step_NNNNNN/ | run_dir with "
            "checkpoints/weights/step_*.pt."
        )

    def _check_pretrain_norm_mode_compat(self, resume_path: Path) -> None:
        """Warn if the pretrain run's norm mode looks incompatible with the current config.

        Expects <run_dir>/checkpoints/weights/<ckpt>.pt so the run dir is three levels up.
        Prefers the run's config.yaml snapshot (`data.train.processor.norm_default_mode`
        — ground truth). Only without one falls back to inferring from
        dataset_stats.json keys; that guess always reads as z-score (stats files carry
        every stat family regardless of the mode used), so it mislabels q01/q99 and
        min/max pretrains.
        """
        run_dir = resume_path.parent.parent.parent
        try:
            finetune_mode = str(self.cfg.data.train.processor.norm_default_mode)
        except Exception:
            logger.info("Could not read data.train.processor.norm_default_mode; skipping compat check.")
            return
        # Preferred: the pretrain run's config snapshot records the actual mode.
        pretrain_cfg_path = run_dir / "config.yaml"
        if pretrain_cfg_path.exists():
            try:
                pretrain_mode = OmegaConf.select(
                    OmegaConf.load(pretrain_cfg_path),
                    "data.train.processor.norm_default_mode",
                )
            except Exception as e:
                logger.warning("Failed to read %s for norm-mode compat: %s", pretrain_cfg_path, e)
                pretrain_mode = None
            if pretrain_mode is not None:
                if str(pretrain_mode) == finetune_mode:
                    logger.info(
                        "Pretrain (%s) and finetune norm modes match: %s.",
                        pretrain_cfg_path,
                        finetune_mode,
                    )
                else:
                    logger.error(
                        "NORM MODE MISMATCH: finetune `norm_default_mode`=%s but pretrain run %s "
                        "used %s (config snapshot). Cross-embodiment transfer will likely break — "
                        "unify modes and re-pad pretrain stats.",
                        finetune_mode, run_dir, pretrain_mode,
                    )
                return
        # Fallback (no config snapshot): guess from dataset_stats.json keys.
        pretrain_stats_path = run_dir / "dataset_stats.json"
        if not pretrain_stats_path.exists():
            logger.info(
                "No pretrain config.yaml/dataset_stats.json under %s; skipping norm-mode compat check.",
                run_dir,
            )
            return
        try:
            with open(pretrain_stats_path, "r") as f:
                stats = json.load(f)
            action_stats = stats.get("action", {})
            if not action_stats:
                return
            first_key = next(iter(action_stats))
            keys = set(action_stats[first_key].keys())
        except Exception as e:
            logger.warning("Failed to inspect %s for norm-mode compat: %s", pretrain_stats_path, e)
            return
        inferred = "z-score" if {"global_mean", "global_std"}.issubset(keys) else "min/max-or-q01q99"
        if finetune_mode == "z-score" and inferred == "z-score":
            logger.info("Pretrain stats (%s) and finetune norm mode match: z-score.", pretrain_stats_path)
        elif finetune_mode != inferred:
            logger.error(
                "NORM MODE MISMATCH: finetune `norm_default_mode`=%s but pretrain stats at %s look like %s. "
                "Cross-embodiment transfer will likely break — unify modes and re-pad pretrain stats.",
                finetune_mode, pretrain_stats_path, inferred,
            )

    def _reanchor_lr_schedule(self):
        """Rebuild the LR schedule for the *current* horizon and fast-forward it to
        the restored global_step.

        accelerator.load_state() restores the saved scheduler verbatim — its T_max /
        last_epoch reflect the horizon of the original run. When resuming into a longer
        schedule (e.g. a 20-epoch checkpoint continued at num_epochs=40), the restored
        cosine is already at its floor, so training would proceed at eta_min. This
        rebuilds the schedule for self.max_steps (which already reflects the active
        num_epochs/max_steps) and advances it to global_step so the LR lands at the
        right point on the new curve. Opt-in via cfg.resume_reanchor_lr_schedule;
        a no-op for same-horizon resumes (fast-forward reproduces the restored LR).
        """
        # load_state left the restored (decayed) LR in the param groups; re-anchor the
        # base LRs at the captured peak so the rebuilt cosine starts from the peak.
        for group, base_lr in zip(self.optimizer.param_groups, self._scheduler_base_lrs):
            group["initial_lr"] = base_lr
            group["lr"] = base_lr
        warmup_steps = int(self.max_steps * 0.05)
        fresh = self._build_scheduler(
            total_train_steps=self.max_steps,
            warmup_steps=warmup_steps,
        )
        # Advance the fresh schedule to the resumed step. step() count tracks optimizer
        # steps, matching how global_step is incremented in the training loop.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # silence "step() before optimizer.step()"
            for _ in range(int(self.global_step)):
                fresh.step()
        self.scheduler.load_state_dict(fresh.state_dict())
        logger.info(
            "Re-anchored LR schedule: horizon=%d steps, warmup=%d, resumed at step=%d, lr=%.3e",
            self.max_steps,
            warmup_steps,
            self.global_step,
            self.optimizer.param_groups[0]["lr"],
        )

    def _set_dit_only_train_mode(self):
        # Match DiffSynth's freeze_except("dit"): only DiT stays trainable/in-train-mode.
        logger.info("Setting DiT to train mode and freezing other model components.")
        model = self.accelerator.unwrap_model(self.model)
        self._apply_dit_only_train_mode(model)

    @staticmethod
    def _apply_dit_only_train_mode(model):
        # Frozen: large pretrained modules declared in model.FROZEN_MODULES.
        # Trainable: everything else (dit, proprio_encoder, _state_proj, dino layers, etc.)
        frozen = getattr(model, "FROZEN_MODULES", {"vae", "text_encoder"})
        model.eval()
        model.requires_grad_(False)
        unfrozen_names = []
        for name, child in model.named_children():
            if name in frozen:
                continue
            child.train()
            child.requires_grad_(True)
            unfrozen_names.append(name)
        # Bare root-level parameters (nn.Parameters set directly on the model)
        # are NOT child modules, so the named_children() loop above never
        # re-enables them — they would stay frozen at init. They are never
        # pretrained-frozen tensors, so unfreeze them explicitly.
        n_root_params = sum(1 for _ in model.named_parameters(recurse=False))
        if n_root_params:
            for _, p in model.named_parameters(recurse=False):
                p.requires_grad_(True)
            logger.info("Unfrozen %d bare root-level parameter(s).", n_root_params)
        # Sanity check: every nn.Module child with parameters should be either
        # frozen (in FROZEN_MODULES) or unfrozen (trainable). If a child has
        # parameters but is neither, it's likely a new layer that someone forgot
        # to add to FROZEN_MODULES — warn loudly so it doesn't silently freeze.
        for name, child in model.named_children():
            if name in frozen:
                continue
            n_params = sum(1 for _ in child.parameters())
            n_grad = sum(1 for p in child.parameters() if p.requires_grad)
            if n_params > 0 and n_grad == 0:
                logger.warning(
                    "Module '%s' has %d parameters but none are trainable. "
                    "If this is a pretrained encoder, add it to FROZEN_MODULES. "
                    "Otherwise this is a bug — the layer will never learn.",
                    name, n_params,
                )
        logger.info("Unfrozen modules: %s | Frozen: %s", unfrozen_names, sorted(frozen))

    @staticmethod
    def _to_batched_eval_sample(sample, model=None):
        # RobotVideoDataset returns per_cam only; compose the composite
        # here so the rest of the eval path (VAE decode, inference rollout)
        # sees the same sample['video'] shape it always did.
        per_cam_out = None
        if "per_cam" in sample and "video" not in sample:
            from flexpi.per_cam_compose import compose_from_per_cam
            per_cam_batched = {}
            for k, v in sample["per_cam"].items():
                if v.ndim == 4:
                    v = v.unsqueeze(0)  # add batch dim
                per_cam_batched[k] = v
            sample = dict(sample)
            sample["video"] = compose_from_per_cam(
                per_cam_batched, **model._layout_kwargs(),
            ).squeeze(0)
            per_cam_out = per_cam_batched

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
            raise ValueError(f"Expected video shape [3,T,H,W] or [B,3,T,H,W], got {tuple(video.shape)}")
        num_video_frames = video.shape[2]
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

        out = {
            "video": video,
            "prompt": prompt,
            "action": action,
            "proprio": proprio,
            "context": context,
            "context_mask": context_mask,
            "action_horizon": action_horizon,
        }

        # Pass through camera_intrinsics for FlexPi3D val loss. Shape out of
        # RobotVideoDataset is [num_cams, 3, 3]; unsqueeze to match the [B,
        # num_cams, 3, 3] the training collation produces. Only include the key
        # if we actually have a tensor — FlexPi3D.build_inputs raises a clear
        # KeyError when the key is absent, which we want to preserve.
        camera_intrinsics = sample.get("camera_intrinsics", None)
        if isinstance(camera_intrinsics, torch.Tensor):
            if camera_intrinsics.ndim == 3:
                camera_intrinsics = camera_intrinsics.unsqueeze(0)
            out["camera_intrinsics"] = camera_intrinsics

        # Pass through cross-embodiment fields (action_dim_is_pad, proprio_dim_is_pad).
        # DataLoader collate handles these in training; eval builds the batch
        # manually here so we must add the batch dim explicitly.
        action_dim_is_pad = sample.get("action_dim_is_pad", None)
        if isinstance(action_dim_is_pad, torch.Tensor):
            if action_dim_is_pad.ndim == 1:
                action_dim_is_pad = action_dim_is_pad.unsqueeze(0)
            out["action_dim_is_pad"] = action_dim_is_pad

        proprio_dim_is_pad = sample.get("proprio_dim_is_pad", None)
        if isinstance(proprio_dim_is_pad, torch.Tensor):
            if proprio_dim_is_pad.ndim == 1:
                proprio_dim_is_pad = proprio_dim_is_pad.unsqueeze(0)
            out["proprio_dim_is_pad"] = proprio_dim_is_pad

        # Forward the per_cam dict so val build_inputs uses the per_cam path
        # (matches training behavior on RobotVideoDataset).
        if per_cam_out is not None:
            out["per_cam"] = per_cam_out

        # Forward per_cam_depth for FlexPi3D val loss. Each tensor arrives as
        # [T, H, W] from RobotVideoDataset; add batch dim to match
        # what the train-time collate produces ([B, T, H, W]).
        per_cam_depth = sample.get("per_cam_depth", None)
        if isinstance(per_cam_depth, dict):
            pcd_batched = {}
            for k, v in per_cam_depth.items():
                if v.ndim == 3:
                    v = v.unsqueeze(0)
                pcd_batched[k] = v
            out["per_cam_depth"] = pcd_batched

        return out

    def _run_eval_and_log(self):
        metrics = self.evaluate()
        self.accelerator.wait_for_everyone()
        if metrics is None or not self.accelerator.is_main_process:
            return
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
        # FlexPiLatentGoal val-vis caveat: when `infer_joint_video=False`, the
        # mask gates DINO ↔ V_rem off, so V_rem self-denoises uncoupled from the
        # goal. The video PSNR/SSIM under this mode reflects the parent video DiT
        # only, NOT joint goal-video learning. We append a suffix to the wandb
        # keys so this is unmistakable when comparing across runs.
        underlying_model = self.accelerator.unwrap_model(self.model)
        is_latent_goal = type(underlying_model).__name__ == "FlexPiLatentGoal"
        video_suffix = ""
        if is_latent_goal:
            joint_v = bool(getattr(underlying_model, "infer_joint_video", False))
            video_suffix = "_video_joint_goal" if joint_v else "_video_decoupled_goal"
        eval_payload = {
            "eval/val_loss": float(metrics["val_loss"]),
            f"eval/psnr_rg{video_suffix}": float(metrics["psnr_rg"]),
            f"eval/ssim_rg{video_suffix}": float(metrics["ssim_rg"]),
            f"eval/psnr_rd{video_suffix}": float(metrics["psnr_rd"]),
            f"eval/ssim_rd{video_suffix}": float(metrics["ssim_rd"]),
            f"eval/psnr_dg{video_suffix}": float(metrics["psnr_dg"]),
            f"eval/ssim_dg{video_suffix}": float(metrics["ssim_dg"]),
        }
        if "action_l2" in metrics:
            eval_payload["eval/action_l2"] = float(metrics["action_l2"])
        if "action_l1" in metrics:
            eval_payload["eval/action_l1"] = float(metrics["action_l1"])
        if metrics.get("video_path") and os.path.isfile(metrics["video_path"]):
            import wandb
            has_dino = "dino_mse" in metrics
            has_pointmap = bool(metrics.get("has_pointmap"))
            row_descs = ["pred_rgb / vae_recon / gt_rgb"]
            if has_dino:
                row_descs.append("pred_dino_pca / gt_dino_pca / gt_dino_norm")
            if has_pointmap:
                row_descs.append("pred_pt_xyz / vae_recon_pt_xyz / gt_pt_xyz")
                row_descs.append("pred_pt_depth / vae_recon_pt_depth / gt_pt_depth")
                row_descs.append("pred_pt_vae_pca / gt_pt_vae_pca / gt_pt_vae_norm")
            row_descs.append("pred_vae_pca / gt_vae_pca / gt_vae_norm")
            caption = f"step {self.global_step} | " + " | ".join(
                f"row{i}: {d}" for i, d in enumerate(row_descs, 1)
            )
            eval_payload["eval/video"] = wandb.Video(
                metrics["video_path"], caption=caption,
            )
        if "dino_mse" in metrics:
            eval_payload["eval/dino_mse"] = float(metrics["dino_mse"])
        self._wandb_log(eval_payload)

    @torch.no_grad()
    def evaluate(self):
        if self.val_dataset is None:
            return None

        model = self.accelerator.unwrap_model(self.model)
        was_dit_training = model.dit.training
        model.eval()

        # eval_index = (self.global_step + self.accelerator.process_index) % len(self.val_dataset)
        rng = torch.Generator(device="cpu").manual_seed(self.global_step + self.accelerator.process_index)
        eval_index = torch.randint(0, len(self.val_dataset), (1,), generator=rng).item()
        sample = self._to_batched_eval_sample(self.val_dataset[eval_index], model=model)

        # 1. training loss
        with self.accelerator.autocast():
            val_loss, _ = model.training_loss(sample)
            val_loss = val_loss.float().item()

        prompt = sample["prompt"][0]
        video0 = sample["video"][0] # Tensor [3, T, H, W] in (-1, 1)
        action = sample["action"][0] if "action" in sample and sample["action"] is not None else None
        input_image = video0[:, 0].unsqueeze(0)
        _, num_frames, _, _ = video0.shape

        # Proprio: for FlexTF (per-chunk state), pass full [T, D] proprio;
        # for base models, pass single [D] state at t=0.
        if "proprio" in sample and sample["proprio"] is not None:
            if getattr(model, "use_per_chunk_state", False):
                proprio = sample["proprio"][0]  # [T, D] — per-timestep
            else:
                proprio = sample["proprio"][0, 0]  # [D] — first timestep only
        else:
            proprio = None

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
        # FlexPi3D-specific: forward per_cam_depth + camera_intrinsics + per_cam
        # when present in the sample. FlexPi3D.infer (override) accepts these;
        # base FlexPi.infer ignores extras only via fixed signature, so we
        # gate by presence in the sample dict.
        if "per_cam_depth" in sample:
            infer_kwargs["per_cam_depth"] = sample["per_cam_depth"]
        if "camera_intrinsics" in sample:
            infer_kwargs["camera_intrinsics"] = sample["camera_intrinsics"]
        if "per_cam" in sample:
            infer_kwargs["per_cam"] = sample["per_cam"]
        if sample["context"] is not None:
            infer_kwargs["prompt"] = None
            infer_kwargs["context"] = sample["context"][0]
            infer_kwargs["context_mask"] = sample["context_mask"][0]
        else:
            infer_kwargs["prompt"] = prompt

        # FlexTF val vis: pass full GT video for correct clean conditioning
        if hasattr(model, "chunk_size"):
            infer_kwargs["gt_video"] = video0

        # FlexPi3D needs per_cam_depth at inference; per_cam is also forwarded
        # when present so the per-cam path matches what build_inputs used for
        # the val loss above. Both are only attached to the eval sample by
        # _to_batched_eval_sample when the val dataset emits them, so the
        # conditional pass-through is safe across model variants.
        if "per_cam" in sample and isinstance(sample["per_cam"], dict):
            infer_kwargs["per_cam"] = sample["per_cam"]
        if "per_cam_depth" in sample and isinstance(sample["per_cam_depth"], dict):
            infer_kwargs["per_cam_depth"] = sample["per_cam_depth"]
        if "camera_intrinsics" in sample and isinstance(sample["camera_intrinsics"], torch.Tensor):
            infer_kwargs["camera_intrinsics"] = sample["camera_intrinsics"]

        # Main video rollout — skipped in lean eval (`eval_video=False`).
        pred = model.infer(**infer_kwargs) if self.eval_video else None

        if pred is not None:
            pred_video = pred["video"]
            pred_action = pred.get("action", None)
            pred_video_latents = pred.get("video_latents", None)  # [1, 48, F, H', W'] or None
            pred_dino_latents = pred.get("dino_latents", None)    # [1, 768, F_dino, 294, 1] or None
            # FlexPiLatentGoal returns only a subset (the goal frames). When present,
            # we use these indices to slice GT to match the prediction count for viz.
            pred_dino_frame_indices = pred.get("dino_frame_indices", None)
            # Number of leading dino tiles that are anchors (clean obs DINO) rather
            # than denoised predictions. Used to label the viz row.
            pred_dino_anchor_count = int(pred.get("dino_anchor_count", 0) or 0)
            pred_pointmap_latents = pred.get("pointmap_latents", None)  # [1, 384, F_pt, 294, 1] or None
            # FlexPiXWAM-only: dict[cam_name -> [1, T_raw, H, W, 1]] inverse depth in [0, 1].
            # No other variant sets this key, so any val-vis branching on it is XWAM-specific.
            pred_depth_pred = pred.get("depth_pred", None)

            # 3. inference metrics against GT video
            pred_video_tensor = pil_frames_to_video_tensor(pred_video)
            gt_video_tensor = ((video0.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()

            assert pred_video_tensor.shape == gt_video_tensor.shape, (
                "Eval infer prediction/GT shape mismatch: "
                f"pred={tuple(pred_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
            )

            psnr_rollout_vs_gt = video_psnr(pred=pred_video_tensor, target=gt_video_tensor)
            ssim_rollout_vs_gt = video_ssim(pred=pred_video_tensor, target=gt_video_tensor)
        else:
            # Lean eval: no main rollout. Defaults so the per-regime sweep and
            # the lean return below stay well-defined.
            pred_video = pred_action = pred_video_latents = pred_dino_latents = None
            pred_dino_frame_indices = None
            pred_dino_anchor_count = 0
            pred_pointmap_latents = pred_depth_pred = None
            pred_video_tensor = None
            gt_video_tensor = ((video0.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()
            psnr_rollout_vs_gt = ssim_rollout_vs_gt = float("nan")

        action_l1 = None
        action_l2 = None
        gt_action_denorm = None
        if action is not None:
            if sample["proprio"] is None:
                raise ValueError("Eval sample must contain `proprio` for action denormalization.")
            denorm_proprio = sample["proprio"].detach().to(device="cpu", dtype=torch.float32)

            processor = self.val_dataset.lerobot_dataset.processor
            action_meta = processor.shape_meta["action"]
            state_meta = processor.shape_meta["state"]

            def _denorm(raw_action, name="action"):
                if not isinstance(raw_action, torch.Tensor):
                    raise TypeError(f"{name} must be a torch.Tensor, got {type(raw_action)}")
                if raw_action.ndim == 2:
                    action_btd = raw_action.unsqueeze(0)
                elif raw_action.ndim == 3 and raw_action.shape[0] == 1:
                    action_btd = raw_action
                else:
                    raise ValueError(
                        f"{name} must have shape [T, D] or [1, T, D], got {tuple(raw_action.shape)}"
                    )
                action_btd = action_btd.detach().to(device="cpu", dtype=torch.float32)
                batch = {"action": action_btd, "state": denorm_proprio}
                batch = processor.action_state_merger.backward(batch)
                batch = processor.normalizer.backward(batch)
                merged_batch = {
                    "action": {meta["key"]: batch["action"][meta["key"]].squeeze(0) for meta in action_meta},
                    "state": {meta["key"]: batch["state"][meta["key"]].squeeze(0) for meta in state_meta},
                }
                merged_batch = processor.action_state_merger.forward(merged_batch)
                denorm = merged_batch["action"].unsqueeze(0)
                if denorm.ndim != 3 or denorm.shape[0] != 1:
                    raise ValueError(
                        f"Denormalized {name} must have shape [1, T, D], got {tuple(denorm.shape)}"
                    )
                return denorm

            gt_action_denorm = _denorm(action, "gt action")

            # Main action metrics only when the main rollout ran (eval_video).
            if pred_action is not None:
                pred_action_denorm = _denorm(pred_action, "pred action")
                if pred_action_denorm.shape != gt_action_denorm.shape:
                    raise ValueError(
                        "Predicted action/GT action shape mismatch after denormalization: "
                        f"pred={tuple(pred_action_denorm.shape)} vs gt={tuple(gt_action_denorm.shape)}"
                    )
                action_diff = pred_action_denorm - gt_action_denorm
                action_l1 = action_diff.abs().mean().item()
                action_l2 = action_diff.pow(2).mean().item()


        # ── Lean eval (eval_video=False): skip the VAE-recon viz + stitched mp4.
        # val-loss and the action metrics are already computed above, so gather
        # + return here. The eval_video=True path below is unchanged.
        # Video-decode metrics default to NaN (wandb simply gaps them). Both
        # gathers are collective and run on every rank (eval_video is global).
        if not self.eval_video:
            psnr_rollout_vs_decode = ssim_rollout_vs_decode = float("nan")
            psnr_decode_vs_gt = ssim_decode_vs_gt = float("nan")
            if was_dit_training:
                self._set_dit_only_train_mode()
            local_metrics = torch.tensor(
                [
                    float(val_loss),
                    float(psnr_rollout_vs_gt), float(ssim_rollout_vs_gt),
                    float(psnr_rollout_vs_decode), float(ssim_rollout_vs_decode),
                    float(psnr_decode_vs_gt), float(ssim_decode_vs_gt),
                    float(action_l2) if action_l2 is not None else -1.0,
                    float(action_l1) if action_l1 is not None else -1.0,
                ],
                device=self.accelerator.device, dtype=torch.float32,
            ).unsqueeze(0)
            gathered_metrics = self.accelerator.gather_for_metrics(local_metrics)
            mean_metrics = gathered_metrics[:, :7].mean(dim=0)
            action_l2_mean = gathered_metrics[:, 7].mean().item() if action_l2 is not None else None
            action_l1_mean = gathered_metrics[:, 8].mean().item() if action_l1 is not None else None
            result = {
                "val_loss": float(mean_metrics[0].item()),
                "psnr_rg": float(mean_metrics[1].item()),
                "ssim_rg": float(mean_metrics[2].item()),
                "psnr_rd": float(mean_metrics[3].item()),
                "ssim_rd": float(mean_metrics[4].item()),
                "psnr_dg": float(mean_metrics[5].item()),
                "ssim_dg": float(mean_metrics[6].item()),
                "video_path": None,
            }
            if action_l2_mean is not None:
                result["action_l2"] = float(action_l2_mean)
            if action_l1_mean is not None:
                result["action_l1"] = float(action_l1_mean)
            return result

        # 4. VAE reconstruction metrics against GT video
        gt_video_batch = video0.unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
        vae_latents = model._encode_video_latents(gt_video_batch, tiled=False)
        vae_recon_video = model._decode_latents(vae_latents, tiled=False)
        vae_video_tensor = pil_frames_to_video_tensor(vae_recon_video)

        assert vae_video_tensor.shape == gt_video_tensor.shape, (
            "Eval VAE reconstruction/GT shape mismatch: "
            f"vae={tuple(vae_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_decode_vs_gt = video_psnr(pred=vae_video_tensor, target=gt_video_tensor)
        ssim_decode_vs_gt = video_ssim(pred=vae_video_tensor, target=gt_video_tensor)

        psnr_rollout_vs_decode = video_psnr(pred=pred_video_tensor, target=vae_video_tensor)
        ssim_rollout_vs_decode = video_ssim(pred=pred_video_tensor, target=vae_video_tensor)

        stitched_video_tensor = torch.cat(
            [pred_video_tensor, vae_video_tensor],
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
        # DINO + VAE latent visualization — combine with video into single stitched output
        dino_mse = None
        drew_pointmap_row = False  # True iff the pointmap rows were drawn
        try:
            from flexpi.vis import build_vae_row, build_dino_row

            num_raw_frames = stitched_video_tensor.shape[1]  # e.g., 33

            # --- Video row as numpy ---
            video_row_np = np.stack([
                (stitched_video_tensor[:, t].permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
                for t in range(num_raw_frames)
            ])  # [num_raw_frames, composite_h, 2*composite_w, 3]

            # Panel width = the RGB video row's OWN native width (2·composite_w),
            # so the video renders at TRUE resolution — no downscale. Each aux row
            # (VAE / DINO / pointmap / inverse-depth) is conformed to this width by
            # _row_to_panel_w below, so the vertical np.concatenate(rows, axis=1)
            # still stacks cleanly. Across configs the panel now scales with the
            # real composite (RoboTwin 2·320=640, wide-T 2·512=1024, near-sq-T
            # 2·448=896), so the viz reflects RELATIVE resolution instead of
            # forcing every layout to one fixed width.
            _PANEL_W = int(video_row_np.shape[2])

            rows = [video_row_np]

            # --- DINO row (from inference output) ---
            if pred_dino_latents is not None:
                # Get GT DINO features by encoding GT video / per_cam. Input
                # source precedence matches build_inputs: per_cam → composite slice.
                _dino_per_cam = None
                if "per_cam" in sample:
                    _dino_per_cam = {
                        k: v.to(device=model.device, dtype=model.torch_dtype, non_blocking=True)
                        for k, v in sample["per_cam"].items()
                    }
                gt_video_for_dino = None
                if _dino_per_cam is None:
                    gt_video_for_dino = sample["video"].to(device=model.device, dtype=model.torch_dtype)
                with torch.no_grad():
                    gt_dino = model.dino_encoder.encode_video(
                        video=gt_video_for_dino, concat_mode="tshape_robotwin_384x320_uniform", **model._dino_encode_kwargs(),
                        temporal_stride=model.dino_temporal_stride,
                        per_cam=_dino_per_cam,
                    ).cpu()  # [1, 768, F_dino, 294, 1]

                # pred_dino_latents: [1, 768, F_dino, 294, 1]
                # build_dino_row expects gt=[C, F, N, 1] and pred=[B, F*N, C]
                gt_dino_0 = gt_dino[0]  # [768, F_dino, 294, 1]
                # If the model only predicted a subset of DINO frames (FlexPiLatentGoal
                # with `dino_goal_frame_indices` != all F_dino positions), slice GT to
                # the same indices so pred and gt have aligned frame counts for viz.
                if pred_dino_frame_indices is not None and len(pred_dino_frame_indices) != gt_dino_0.shape[1]:
                    idx = torch.tensor(pred_dino_frame_indices, dtype=torch.long)
                    gt_dino_0 = gt_dino_0.index_select(1, idx)  # [768, G_frames, 294, 1]
                pred_dino_0 = pred_dino_latents[0]  # [768, F_dino, 294, 1]
                F_dino, N_patches = gt_dino_0.shape[1], gt_dino_0.shape[2]
                dino_dim = gt_dino_0.shape[0]
                # Reshape pred to [1, F_dino*294, 768]
                pred_dino_for_vis = pred_dino_0.squeeze(-1).permute(1, 2, 0).reshape(1, -1, dino_dim)

                cam_patches = [tuple(p) for p in model.dino_cam_patches]
                dino_row = build_dino_row(
                    gt_features=gt_dino_0,
                    pred_features=pred_dino_for_vis,
                    cam_patch_sizes=cam_patches,
                    anchor_count=pred_dino_anchor_count,
                    share_pca_basis=bool(getattr(self.cfg, "share_pca_basis", True)),
                    layout=getattr(model, "_layout", None),
                    pixel_unshuffle=getattr(model, "_dino_pixel_unshuffle", 0),
                )  # [F_dino, H_total, 2*W_total, 3]

                # DINO MSE — exclude the obs frame (its prediction is trivial).
                # For FlexPiLatent (pred_dino_frame_indices is None) the model predicts
                # all F_dino frames including frame 0 (obs), so skip the first slot.
                # For FlexPiLatentGoal, only skip if frame 0 is in the predicted subset.
                gt_flat = gt_dino_0.squeeze(-1).permute(1, 2, 0).reshape(-1, dino_dim)  # [F*N, 768]
                pred_flat = pred_dino_0.squeeze(-1).permute(1, 2, 0).reshape(-1, dino_dim)
                first_is_obs = (
                    pred_dino_frame_indices is None
                    or (len(pred_dino_frame_indices) > 0 and pred_dino_frame_indices[0] == 0)
                )
                if first_is_obs and F_dino > 1:
                    diff = (pred_flat[N_patches:].float() - gt_flat[N_patches:].float()).pow(2)
                    dino_mse = float(diff.mean().item())
                elif not first_is_obs and F_dino >= 1:
                    diff = (pred_flat.float() - gt_flat.float()).pow(2)
                    dino_mse = float(diff.mean().item())
                else:
                    dino_mse = 0.0

                # Debug: pred vs gt feature scale. If pred_std ≪ gt_std the model
                # is producing under-magnitude features (early training) — the PCA
                # viz can still look colorful due to per-frame min-max stretching.
                with torch.no_grad():
                    pred_std = float(pred_dino_0.float().std().item())
                    gt_std = float(gt_dino_0.float().std().item())
                logger.info(
                    "dino_mse=%.4f pred_std=%.4f gt_std=%.4f std_ratio=%.3f",
                    dino_mse, pred_std, gt_std, pred_std / max(gt_std, 1e-8),
                )

                # Expand DINO to raw frame count
                num_dino_frames = dino_row.shape[0]
                if num_dino_frames < num_raw_frames:
                    dino_indices = np.clip(
                        np.arange(num_raw_frames) * num_dino_frames // num_raw_frames,
                        0, num_dino_frames - 1,
                    )
                    dino_row = dino_row[dino_indices]
                rows.append(dino_row[:num_raw_frames])

            # --- Pointmap row (from FlexPi3D joint inference output) ---
            if pred_pointmap_latents is not None and hasattr(model, "pointmap_encoder"):
                drew_pointmap_row = True
                K = model._resolve_camera_intrinsics()

                if "per_cam_depth" not in sample:
                    raise KeyError(
                        "Val sample missing 'per_cam_depth'. FlexPi3D requires "
                        "depth in the sample dict — use configs/data/robotwin.yaml."
                    )
                _pt_per_cam_depth = {
                    k: v.to(device=model.device, non_blocking=True)
                    for k, v in sample["per_cam_depth"].items()
                }

                # Pred and VAE-recon both live in the same XYZ-composite
                # space. Map [-1,1] → [0,1] and tile as RGB.
                from flexpi.vis import (
                    build_pointmap_depth_row_vae,
                    build_pointmap_row_vae,
                )

                with torch.no_grad():
                    gt_pt_composite = model.pointmap_encoder.encode_composite(
                        per_cam_depth=_pt_per_cam_depth,
                        camera_intrinsics=K, concat_mode="tshape_robotwin_384x320_uniform", **model._layout_kwargs(),
                    )  # [1, 3, F_pt, 384, 320] in [-1, 1]
                    gt_pt_latents = model._encode_video_latents(gt_pt_composite, tiled=False)
                    recon_pt_frames = model._decode_latents(gt_pt_latents, tiled=False)
                    pred_pt_latents_for_decode = pred_pointmap_latents.to(
                        device=model.device, dtype=model.torch_dtype,
                    )
                    pred_pt_frames = model._decode_latents(
                        pred_pt_latents_for_decode, tiled=False,
                    )

                recon_pt_tensor = pil_frames_to_video_tensor(recon_pt_frames)  # [3, F, 384, 320]
                pred_pt_tensor = pil_frames_to_video_tensor(pred_pt_frames)    # [3, F, 384, 320]

                pt_row = build_pointmap_row_vae(
                    pred=pred_pt_tensor, vae_recon=recon_pt_tensor,
                )  # [F, 384, 640, 3] uint8

                # Project Z back to metric depth and render a second row.
                # pt_min/pt_max are [1, 3, 1, 1]; channel 2 is Z.
                pt_min = model.pointmap_encoder.pt_min
                pt_max = model.pointmap_encoder.pt_max
                z_min = float(pt_min[0, 2, 0, 0].item())
                z_max = float(pt_max[0, 2, 0, 0].item())
                depth_vis_mode = getattr(model, "pointmap_depth_vis_mode", "turbo")
                pt_depth_row = build_pointmap_depth_row_vae(
                    pred=pred_pt_tensor, vae_recon=recon_pt_tensor,
                    z_min=z_min, z_max=z_max, vis_mode=depth_vis_mode,
                )  # [F, 384, 640, 3] uint8

                num_pt_frames = pt_row.shape[0]
                if num_pt_frames < num_raw_frames:
                    pt_indices = np.clip(
                        np.arange(num_raw_frames) * num_pt_frames // num_raw_frames,
                        0, num_pt_frames - 1,
                    )
                    pt_row = pt_row[pt_indices]
                rows.append(pt_row[:num_raw_frames])

                # Depth-projection row: pred | recon.
                if num_pt_frames < num_raw_frames:
                    pt_depth_row = pt_depth_row[pt_indices]
                rows.append(pt_depth_row[:num_raw_frames])

                # VAE latent feature vis for the pointmap stream (mirror of the
                # RGB-video VAE row, appended later).
                pt_vae_row = build_vae_row(
                    gt_pt_latents[0].detach().cpu(),
                    pred_pointmap_latents[0].detach().cpu(),
                    target_hw=model._layout.composite_hw,
                )  # [F_lat, H_total, 2*W_total, 3]
                num_pt_vae_frames = pt_vae_row.shape[0]
                if num_pt_vae_frames < num_raw_frames:
                    pt_vae_indices = np.clip(
                        np.arange(num_raw_frames) * num_pt_vae_frames // num_raw_frames,
                        0, num_pt_vae_frames - 1,
                    )
                    pt_vae_row = pt_vae_row[pt_vae_indices]
                rows.append(pt_vae_row[:num_raw_frames])

            # --- Inverse-depth row (FlexPi-XWAM only) ---
            # Gated on `pred_depth_pred is not None`, which is set ONLY by
            # FlexPiXWAM.infer_joint. All other variants leave the key absent,
            # so this branch has zero effect on FlexPi/FlexPiJoint/FlexPi3D/
            # FlexPiLatent paths.
            if pred_depth_pred is not None and "per_cam_depth" in sample:
                from flexpi.vis import build_inverse_depth_row
                depth_max = float(getattr(model, "depth_max_meters", 2.0))
                # GT: uint16 mm → normalized inverse depth in [0, 1].
                gt_inv_per_cam = {}
                for k, v in sample["per_cam_depth"].items():
                    m = v[0].float() / 1000.0  # [T, H, W] meters
                    m = m.clamp(min=1.0 / 10.0, max=depth_max)
                    gt_inv_per_cam[k] = ((depth_max / m) / 10.0).clamp(0.0, 1.0).cpu()
                # Pred: [1, T_raw, H, W, 1] → [T_raw, H, W].
                pred_inv_per_cam = {
                    k: v[0].squeeze(-1).float().cpu()
                    for k, v in pred_depth_pred.items()
                }
                depth_row = build_inverse_depth_row(
                    pred_inv_per_cam=pred_inv_per_cam,
                    gt_inv_per_cam=gt_inv_per_cam,
                    vis_mode=getattr(model, "depth_vis_mode", "turbo"),
                )  # [F, 384, 640, 3] uint8
                num_depth_frames = depth_row.shape[0]
                if num_depth_frames < num_raw_frames:
                    depth_indices = np.clip(
                        np.arange(num_raw_frames) * num_depth_frames // num_raw_frames,
                        0, num_depth_frames - 1,
                    )
                    depth_row = depth_row[depth_indices]
                rows.append(depth_row[:num_raw_frames])

            # --- VAE latent row (from inference output) ---
            gt_vae = vae_latents[0].detach().cpu()  # [48, F_latent, H', W']
            pred_vae = pred_video_latents[0] if pred_video_latents is not None else None
            vae_row = build_vae_row(
                gt_vae,
                pred_vae,
                share_pca_basis=bool(getattr(self.cfg, "share_pca_basis", True)),
                target_hw=model._layout.composite_hw,
            )  # [F_latent, H_total, 2*W_total, 3]
            num_vae_frames = vae_row.shape[0]
            if num_vae_frames < num_raw_frames:
                vae_indices = np.clip(
                    np.arange(num_raw_frames) * num_vae_frames // num_raw_frames,
                    0, num_vae_frames - 1,
                )
                vae_row = vae_row[vae_indices]
            rows.append(vae_row[:num_raw_frames])

            # Harmonize every row to _PANEL_W (the video row's native width, set
            # above) before stacking. Aux builders emit rows at mixed widths —
            # some scale with composite_w (VAE), some are fixed 384×640 (DINO /
            # pointmap / inverse-depth). Resize each to _PANEL_W (preserving its
            # own aspect; heights may differ, that's the stacked axis) so
            # np.concatenate(axis=1) doesn't raise on a width mismatch. No-op for
            # the video row itself, and for every row under RoboTwin (2·320 = 640).
            def _row_to_panel_w(arr):
                if arr.shape[2] == _PANEL_W:
                    return arr
                _h = max(int(round(arr.shape[1] * _PANEL_W / arr.shape[2])), 1)
                _out = np.empty((arr.shape[0], _h, _PANEL_W, 3), dtype=arr.dtype)
                for _i in range(arr.shape[0]):
                    _out[_i] = np.array(
                        Image.fromarray(arr[_i]).resize((_PANEL_W, _h), Image.BILINEAR)
                    )
                return _out
            rows = [_row_to_panel_w(r) for r in rows]
            combined = np.concatenate(rows, axis=1)
            stitched_frames = [Image.fromarray(combined[t]) for t in range(num_raw_frames)]
        except Exception as e:
            import traceback
            logger.warning("DINO/Pointmap/VAE visualization failed: %s\n%s", e, traceback.format_exc())

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
        if dino_mse is not None:
            result["dino_mse"] = float(dino_mse)
        if drew_pointmap_row:
            result["has_pointmap"] = True
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
        state_path = os.path.join(self.state_dir, step_tag)

        # A checkpoint save can fail mid-write when the (shared) filesystem is
        # full — DeepSpeed's save_state then raises a torch "inline_container"
        # error and, unguarded, kills a run that does NOT auto-requeue on crash.
        # Guard each write: on failure, drop the partial checkpoint (a truncated
        # state dir is a resume landmine — the launcher resumes from the latest
        # step_*) and keep training, so the next save can succeed once space
        # frees. The wait_for_everyone() barriers stay unconditional (never gated
        # on save_ok) so a save failure can't desync ranks; on a shared FS all
        # ranks fail the same save together.
        save_ok = True

        self.accelerator.wait_for_everyone()
        ckpt_path = None
        if self.accelerator.is_main_process:
            try:
                ckpt_path = self._save_weights_checkpoint(step_tag=step_tag)
            except Exception as e:  # noqa: BLE001 — any I/O failure must not kill the run
                save_ok = False
                logger.warning("[ckpt] weights save failed step=%d: %s: %s",
                               self.global_step, type(e).__name__, e)
        self.accelerator.wait_for_everyone()

        try:
            ensure_dir(state_path)
            self.accelerator.save_state(output_dir=state_path)
            if self.accelerator.is_main_process:
                self._save_trainer_state(state_path)
        except Exception as e:  # noqa: BLE001 — any I/O failure must not kill the run
            save_ok = False
            logger.warning("[ckpt] state save failed step=%d: %s: %s",
                           self.global_step, type(e).__name__, e)
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            if save_ok:
                # Full-state dirs hold sharded optimizer + grads + weights (tens of
                # GB each); only the latest is ever needed for resume. Weights .pt
                # files (~one model copy each) accumulate too. Cap retention of both
                # to the most recent few (keep_last_n_states / keep_last_n_weights;
                # <=0 keeps all).
                self._prune_old_states(keep=int(getattr(self.cfg, "keep_last_n_states", 3)))
                self._prune_old_weights(keep=int(getattr(self.cfg, "keep_last_n_weights", 3)))
            else:
                logger.warning("[ckpt] discarding partial checkpoint step=%d and continuing training",
                               self.global_step)
                for p in (state_path, os.path.join(self.weights_dir, f"{step_tag}.pt")):
                    if os.path.isdir(p):
                        shutil.rmtree(p, ignore_errors=True)
                    elif os.path.exists(p):
                        try:
                            os.remove(p)
                        except OSError:
                            pass
        self.accelerator.wait_for_everyone()

        if not save_ok:
            return {"weights_path": None, "state_path": None}
        return {"weights_path": ckpt_path, "state_path": state_path}

    def _prune_old_states(self, keep: int):
        """Delete all but the `keep` most recent state/step_* dirs (main proc only)."""
        if keep is None or keep <= 0:
            return
        entries = [
            (int(m.group(1)), e.path)
            for e in os.scandir(self.state_dir)
            if e.is_dir() and (m := re.fullmatch(r"step_(\d+)", e.name))
        ]
        for _step, path in sorted(entries)[:-keep]:
            shutil.rmtree(path, ignore_errors=True)

    def _prune_old_weights(self, keep: int):
        """Delete all but the `keep` most recent weights/step_*.pt files (main proc only)."""
        if keep is None or keep <= 0:
            return
        entries = [
            (int(m.group(1)), e.path)
            for e in os.scandir(self.weights_dir)
            if e.is_file() and (m := re.fullmatch(r"step_(\d+)\.pt", e.name))
        ]
        for _step, path in sorted(entries)[:-keep]:
            try:
                os.remove(path)
            except OSError:
                pass

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
                if self.train_sampler is not None:
                    self.train_sampler.set_epoch_offset(self.epoch)
                    self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)
                    logger.info(
                        "Restored dataloader progress: epoch=%d batch_in_epoch=%d sample_offset=%d",
                        self.epoch,
                        self.batch_in_epoch,
                        self.batch_in_epoch * self.batch_size * self.accelerator.num_processes,
                    )
                else:
                    logger.warning(
                        "Restored epoch=%d but dataloader has no resume hook — "
                        "samples will start from beginning of epoch %d.",
                        self.epoch, self.epoch,
                    )
            else:
                self.epoch = 0
                self.batch_in_epoch = 0
                if self.train_sampler is not None:
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
        if self.train_sampler is not None:
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

        if (
            self.eval_every > 0
            and self.val_dataset is not None
            and self.global_step == 0
        ):
            self._run_eval_and_log()

        # DLTIMING harness — env-gated. Measures wall time spent in
        # `next(data_iter)` (i.e., waiting for a worker to deliver a batch).
        # Emits when wait > FLEXPI_DL_TIMING_THRESHOLD_MS. No-op otherwise.
        _dl_timing = os.environ.get("FLEXPI_DL_TIMING") == "1"
        _dl_threshold_ms = float(os.environ.get("FLEXPI_DL_TIMING_THRESHOLD_MS", "2000"))
        while self.global_step < self.max_steps:
            try:
                _dl_t_fetch = time.perf_counter() if _dl_timing else None
                sample = next(data_iter)
                if _dl_t_fetch is not None:
                    _dl_wait_ms = int((time.perf_counter() - _dl_t_fetch) * 1000)
                    if (_dl_wait_ms > _dl_threshold_ms
                            or os.environ.get("FLEXPI_DL_TIMING_VERBOSE") == "1"):
                        print(
                            f"[DLTIMING] stage=batch_wait step={self.global_step} "
                            f"rank={self.accelerator.process_index} "
                            f"batch_wait_ms={_dl_wait_ms}",
                            flush=True,
                        )
                self.batch_in_epoch += 1
            except StopIteration:
                self.epoch += 1
                self.batch_in_epoch = 0
                if self.train_sampler is not None:
                    self.train_sampler.clear_resume_batch_offset()
                data_iter = iter(self.train_loader)
                continue

            with self.accelerator.accumulate(self.model):
                train_model = self.model if hasattr(self.model, "training_loss") else self.accelerator.unwrap_model(self.model)

                with self.accelerator.autocast():
                    loss, loss_dict = train_model.training_loss(sample)
                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    if not self.accelerator.optimizer_step_was_skipped:
                        self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
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
                    grad_norm_tensor = torch.tensor(grad_norm, device=loss.device, dtype=torch.float32)
                    global_grad_norm = float(self.accelerator.gather(grad_norm_tensor).mean().item())

                    current_lr = float(self.optimizer.param_groups[0]["lr"])

                    if self.log_every > 0 and self.global_step % self.log_every == 0 and self.accelerator.is_main_process:
                        eta_str, steps_per_sec = self._estimate_eta()
                        # Instantaneous step/s = wall time since the *previous*
                        # log message divided by the number of steps in between.
                        # Reflects current throughput, unlike `steps_per_sec`
                        # (running mean since training start, which gets dragged
                        # down by checkpoint saves + evals on short smoke runs
                        # with low save_every). For long production runs with
                        # save_every >> log_every, the two converge.
                        now = time.perf_counter()
                        last_t = getattr(self, "_last_log_time", None)
                        last_s = getattr(self, "_last_log_step", None)
                        if last_t is not None and last_s is not None and self.global_step > last_s:
                            inst_steps_per_sec = (self.global_step - last_s) / max(now - last_t, 1e-9)
                        else:
                            inst_steps_per_sec = steps_per_sec
                        self._last_log_time = now
                        self._last_log_step = self.global_step
                        description = "[train] epoch=%d step=%d/%d loss=%.4f " % (
                            self.epoch,
                            self.global_step,
                            self.max_steps,
                            global_loss,
                        )
                        if global_loss_metrics:
                            detail_str = " ".join([
                                f"{k}={v:.4f}"
                                for k, v in sorted(global_loss_metrics.items())
                            ])
                            description += detail_str + " "
                        # samples_per_sec = actual data throughput = steps/s × per-rank-batch × ranks × accum.
                        # `steps_per_sec` counts OPT steps (increments only on sync_gradients), so when
                        # gradient_accumulation_steps > 1 each opt step has already pushed `accum` micro-batches
                        # through the GPU — multiply by accum to report the true sample rate.
                        _samples_mul = (
                            self.batch_size
                            * self.accelerator.num_processes
                            * self.gradient_accumulation_steps
                        )
                        samples_per_sec = steps_per_sec * _samples_mul
                        inst_samples_per_sec = inst_steps_per_sec * _samples_mul
                        description += "lr=%.2e speed=%.2f step/s (inst=%.2f) %.2f samples/s eta=%s" % (
                            current_lr,
                            steps_per_sec,
                            inst_steps_per_sec,
                            samples_per_sec,
                            eta_str,
                        )
                        logger.info(description)

                        wandb_payload = {
                            "train/loss": global_loss,
                            "train/grad_norm": global_grad_norm,
                            "train/lr": current_lr,
                            "performance/steps_per_sec": steps_per_sec,
                            "performance/instantaneous_steps_per_sec": inst_steps_per_sec,
                            "performance/samples_per_sec": samples_per_sec,
                            "performance/instantaneous_samples_per_sec": inst_samples_per_sec,
                        }
                        for key, value in global_loss_metrics.items():
                            wandb_payload[f"train/{key}"] = value
                        self._wandb_log(wandb_payload)

                    if (
                        self.eval_every > 0
                        and self.val_dataset is not None
                        and self.global_step % self.eval_every == 0
                    ):
                        self._run_eval_and_log()

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
