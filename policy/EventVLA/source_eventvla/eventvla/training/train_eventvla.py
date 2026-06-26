# Copyright 2025 eventvla community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# Implemented by [Jinhui YE / HKUST University] in [2025].


"""
EventVLA’s trainer is built directly on native PyTorch + Accelerate + DeepSpeed, keeping the loop explicit and easy to hack.
Conventions:
1. Store runtime state in dicts where possible (simplifies data info, procesing info, config, etc).  
2. Use multiple dataloaders to adapt heterogeneous data types / task mixtures.  
3. Put each training strategy in its own `trainer_*.py` file (avoid large if‑else chains).  
"""

# Standard Library
import argparse
import json
import os
from pathlib import Path
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import re

# Third-Party Libraries
import torch
import torch.distributed as dist
import wandb
import yaml
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed, GradientAccumulationPlugin
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler
import sys
sys.path.insert(0, os.getcwd())
# Local Modules
from eventvla.training.trainer_utils.trainer_tools import normalize_dotlist_args
from eventvla.model.framework import build_framework
from eventvla.model.memory_ablation import resolve_and_apply_memory_ablation_profile
from eventvla.training.trainer_utils.trainer_tools import TrainerUtils
from eventvla.training.trainer_utils.trainer_tools import build_param_lr_groups
from eventvla.training.trainer_utils.config_tracker import wrap_config, AccessTrackedConfig

accelerator = None

# Sane Defaults
# os.environ['RANK'] = '0'
# os.environ['LOCAL_RANK'] = '0'
# os.environ['WORLD_SIZE'] = "1"
# os.environ["MASTER_ADDR"] = "127.0.0.1"
# os.environ["MASTER_PORT"] = "27383"
# os.environ["TRITON_CACHE_DIR"]="~/.triton"
# os.environ["WANDB_MODE"]="disabled"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Initialize Overwatch =>> Wraps `logging.Logger`
from accelerate.logging import get_logger

logger = get_logger(__name__)


def load_fast_tokenizer():
    fast_tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)
    return fast_tokenizer


def setup_directories(cfg) -> Path:
    """create output directory and save config"""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        # create output directory and checkpoint directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

        # # save config
        # OmegaConf.save(cfg, output_dir / "config.yaml")
        # with open(output_dir / "config.yaml", "r") as f_yaml, open(output_dir / "config.json", "w") as f_json:
        #     yaml_cfg = yaml.safe_load(f_yaml)
        #     json.dump(yaml_cfg, f_json, indent=2)

    return output_dir


def build_model(cfg) -> torch.nn.Module:
    """build model framework"""
    logger.info(f"Loading Base VLM `{cfg.framework.qwenvl.base_vlm}` from ID/Path")
    model = build_framework(cfg)

    return model


# here changes need to 📦 encapsulate Dataloader
from eventvla.dataloader import build_dataloader


def prepare_data(cfg, accelerator, output_dir) -> Tuple[DataLoader, DataLoader]:
    """prepare training data"""
    # VLA data loader
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)

    # This dataloader already uses a rank-aware batch sampler. Keep it as-is and
    # avoid relying on Accelerate dataloader sharding semantics for sequence memory.
    accelerator.dataloader_config.dispatch_batches = False
    dist.barrier()

    return vla_train_dataloader


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """set optimizer and scheduler"""
    # initialize optimizer
    debug_mode = bool(getattr(cfg, "debug_mode", False))
    if debug_mode:
        base_lr = cfg.trainer.learning_rate.get("base", 1e-4)
        train_only_modules = str(getattr(cfg.trainer, "train_only_modules", "action_model.model.fc2"))
        train_paths = [p.strip() for p in train_only_modules.split(",") if p.strip()]
        used_params = set()
        param_groups = []
        for path in train_paths:
            module = model
            try:
                for attr in path.split("."):
                    module = getattr(module, attr)
            except AttributeError as exc:
                raise ValueError(f"Invalid trainer.train_only_modules path: `{path}`") from exc

            group_params = []
            for p in module.parameters():
                if id(p) in used_params:
                    continue
                used_params.add(id(p))
                group_params.append(p)
            if group_params:
                group_lr = cfg.trainer.learning_rate.get(path, base_lr)
                param_groups.append({"params": group_params, "lr": group_lr, "name": path})
        if len(param_groups) == 0:
            raise ValueError(
                "debug_mode=True but no parameters were collected for trainer.train_only_modules. "
                "Please check trainer.train_only_modules path."
            )
    else:
        param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
    )

    # print optimizer group info
    if dist.is_initialized() and dist.get_rank() == 0:
        for i, group in enumerate(optimizer.param_groups):
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    # initialize learning rate scheduler
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=cfg.trainer.scheduler_specific_kwargs,  # minimum learning rate
    )

    return optimizer, lr_scheduler


class VLATrainer(TrainerUtils):
    def __init__(self, cfg, model, vla_train_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        # training status tracking
        self.completed_steps = 0
        self.total_batch_size = self._calculate_total_batch_size()
    
    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        # load pretrained weights
        self._init_checkpointing() # TODO merge with load pretrained weights

        # 根据  resume 调整 lr_scheduler
        self._adjust_lr_scheduler_for_resume()

        # freeze parameters
        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)
        debug_mode = bool(getattr(self.config, "debug_mode", False))
        if debug_mode:
            train_only_modules = (
                self.config.trainer.train_only_modules
                if (self.config and hasattr(self.config.trainer, "train_only_modules"))
                else "action_model.model.fc2"
            )
            self._freeze_all_except_selected(train_only_modules)
            if self.accelerator.is_main_process:
                logger.info(f"debug_mode=True: train only `{train_only_modules}`")

        self.has_trainable_params = any(p.requires_grad for p in self.model.parameters())
        if (not self.has_trainable_params) and self.accelerator.is_main_process:
            logger.warning("No trainable parameters found: backward/optimizer/scheduler steps will be skipped.")

        #  print model trainable parameters:
        self.print_trainable_parameters(self.model)

        # Fix: Manually set DeepSpeed runtime batch settings because DataLoader uses a custom sampler.
        if self.accelerator.state.deepspeed_plugin is not None:
            ds_cfg = self.accelerator.state.deepspeed_plugin.deepspeed_config
            ds_cfg["train_micro_batch_size_per_gpu"] = int(self.config.datasets.vla_data.per_device_batch_size)
            ds_cfg["gradient_accumulation_steps"] = int(self.config.trainer.gradient_accumulation_steps)

        # Keep the rank-aware sequential dataloader unwrapped. If it enters
        # `accelerator.prepare(...)`, Accelerate may shard it again and break the
        # per-rank temporal stream required by sequence-aware memory.
        self.model, self.optimizer = self.setup_distributed_training(
            self.accelerator,  # must be the first param
            self.model,
            self.optimizer,
        )

        self._init_wandb()

    def _freeze_all_except_selected(self, train_only_modules: str) -> None:
        if not isinstance(train_only_modules, str):
            return
        module_paths = [p.strip() for p in train_only_modules.split(",") if p.strip()]
        if len(module_paths) == 0:
            return

        for p in self.model.parameters():
            p.requires_grad = False

        enabled = []
        for path in module_paths:
            module = self.model
            try:
                for attr in path.split("."):
                    module = getattr(module, attr)
                for p in module.parameters():
                    p.requires_grad = True
                enabled.append(path)
            except AttributeError:
                logger.warning(f"train_only_modules path does not exist: {path}")

        if len(enabled) == 0:
            raise ValueError(
                "debug_mode=True but trainer.train_only_modules did not match any module path; "
                "no trainable parameters remain."
            )

        if self.accelerator.is_main_process:
            logger.info(f"Debug mode: only these modules are trainable: {enabled}")


    def _adjust_lr_scheduler_for_resume(self):
        """根据已完成的步数调整学习率调度器状态"""
        if self.completed_steps > 0:
            logger.info(f"Adjusting LR scheduler for resume from step {self.completed_steps}")
            
            # 方法1: 直接模拟已完成的步数（适用于大多数调度器）
            for _ in range(self.completed_steps):
                self.lr_scheduler.step()
            
            # 或者方法2: 对于某些调度器，可以直接设置最后步数
            # if hasattr(self.lr_scheduler, '_step_count'):
            #     self.lr_scheduler._step_count = self.completed_steps
            
            logger.info(f"LR scheduler adjusted to step {self.completed_steps}, current LR: {self.lr_scheduler.get_last_lr()}")

    def _calculate_total_batch_size(self):
        """calculate global batch size"""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_wandb(self):
        """initialize Weights & Biases"""
        if self.accelerator.is_main_process:
            wandb.init(
                name=self.config.run_id,
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group="vla-train",
            )

    def _init_checkpointing(self):
        """Initialize checkpoint directory and handle checkpoint loading."""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # 获取预训练检查点和是否恢复训练的标志
        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)
        self.resume_from_checkpoint = pretrained_checkpoint
        # TODO retinking resume and load from pretrained_checkpoint
        if is_resume:
            # 恢复训练状态
            resume_from_checkpoint, self.completed_steps = self._get_latest_checkpoint(self.checkpoint_dir)
            
            if resume_from_checkpoint:
                self.resume_from_checkpoint = resume_from_checkpoint
                self.model = self.load_pretrained_backbones(self.model, self.resume_from_checkpoint, reload_modules=None)
                logger.info(f"Resuming training from checkpoint: {self.resume_from_checkpoint}, steps: {self.completed_steps}")
                return None
            else:
                logger.warning(f"No valid checkpoint found in {self.checkpoint_dir}. Starting training from scratch.")
                self.completed_steps = 0

        # 加载预训练权重
        if pretrained_checkpoint:
            reload_modules = getattr(self.config.trainer, "reload_modules", None)
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)
            try:
                self.completed_steps = int(
                    re.search(r"steps_(\d+)_(?:pytorch_model\.pt|model\.safetensors)$", pretrained_checkpoint).group(1)
                )
            except AttributeError:
                logger.warning(f"Could not parse steps from pretrained checkpoint: {pretrained_checkpoint}")
                self.completed_steps = 0
            self.resume_from_checkpoint = pretrained_checkpoint
            logger.info(f"Loaded pretrained checkpoint: {pretrained_checkpoint}, steps: {self.completed_steps}")
        else:
            logger.info("No pretrained checkpoint provided. Starting training from scratch.")
            self.completed_steps = 0
    

    def _load_checkpoint(self, checkpoint_path):
        """load checkpoint"""
        self.accelerator.load_state(checkpoint_path)
        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")

    def _save_checkpoint(self):
        """save current training state"""

        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")

            checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")
            # Free space for the incoming checkpoint before materializing the model state.
            self.prune_old_checkpoints(self.checkpoint_dir, incoming_checkpoints=1)

            # save model state
            checkpoint_file = None
            try:
                state_dict = self.accelerator.get_state_dict(self.model)
                if save_format == "safetensors":
                    from safetensors.torch import save_file

                    checkpoint_file = checkpoint_path + "_model.safetensors"
                    save_file(state_dict, checkpoint_file)
                elif save_format == "pt":
                    checkpoint_file = checkpoint_path + "_pytorch_model.pt"
                    torch.save(state_dict, checkpoint_file)
                else:
                    raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")
            except Exception:
                if checkpoint_file is not None and os.path.exists(checkpoint_file):
                    try:
                        os.remove(checkpoint_file)
                    except OSError as cleanup_error:
                        logger.warning(f"Failed to remove incomplete checkpoint {checkpoint_file}: {cleanup_error}")
                raise

            # save training metadata
            summary_data = {
                "steps": self.completed_steps,
            }
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}")
            # ✅ Save accessed configuration only
            if isinstance(self.config, AccessTrackedConfig):
                logger.info("📊 Saving accessed configuration...")
                output_dir = Path(self.config.output_dir)
                # self.config.save_accessed_config(
                #     output_dir / "config.json", 
                #     use_original_values=False
                # )
                self.config.save_accessed_config(
                    output_dir / "config.yaml", 
                    use_original_values=False 
                )
                logger.info("✅ Configuration files saved")

        self.accelerator.wait_for_everyone()

    def _log_metrics(self, metrics):
        """record training metrics"""
        if self.completed_steps % self.config.trainer.logging_frequency == 0:
            if dist.get_rank() == 0:
                # add learning rate 
                metrics["learning_rate"] = self.lr_scheduler.get_last_lr()[0] # see lr group in yaml.trainer.learning_rate

                # add epoch info
                metrics["epoch"] = round(self.completed_steps / len(self.vla_train_dataloader), 2)

                # record to W&B
                wandb.log(metrics, step=self.completed_steps)
                # debug output
                logger.info(f"Step {self.completed_steps}, Loss: {metrics})")

    def _create_data_iterators(self):
        """create data iterators"""
        self.vla_iter = iter(self.vla_train_dataloader)
        # self.vlm_iter = iter(self.vlm_train_dataloader)

    def _get_next_batch(self):
        """get next batch (automatically handle data loop)"""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)

        return batch_vla

    def train(self):
        """execute training loop"""
        # print training config
        self._log_training_config()

        # prepare data iterators
        self._create_data_iterators()

        # create progress bar
        import os
        disable_tqdm = (not self.accelerator.is_local_main_process) or (os.environ.get("TQDM_DISABLE", "False").lower() == "true")
        progress_bar = tqdm(
            range(self.config.trainer.max_train_steps), disable=disable_tqdm
        )

        # main training loop
        while self.completed_steps < self.config.trainer.max_train_steps:
            # get data batch
            t_start_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            t_end_data = time.perf_counter()

            # execute training step
            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            t_end_model = time.perf_counter()

            # update progress
            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.completed_steps += 1
            
            if self.accelerator.is_local_main_process and not disable_tqdm:
                if self.completed_steps % self.config.trainer.logging_frequency == 0:
                    progress_bar.set_postfix(
                            {
                                "data_times": f"{t_end_data - t_start_data:.3f}",
                                "model_times": f"{t_end_model - t_start_model:.3f}",
                            }
                        )

            # evaluate model
            if self.completed_steps % self.config.trainer.eval_interval == 0:
                step_metrics = self.eval_action_model(step_metrics)

            # record metrics
            step_metrics["data_time"] = t_end_data - t_start_data
            step_metrics["model_time"] = t_end_model - t_start_model
            self._log_metrics(step_metrics)

            # save checkpoint
            if self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                self._save_checkpoint()

            # check termination condition
            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        # training end processing
        self._finalize_training()

        # execute evaluation step

    def eval_action_model(self, step_metrics: dict = None) -> float:
        """
        Evaluate the model on the given dataset using the specified metric function.

        :param eval_dataset: List of evaluation samples, each containing 'image', 'instruction', and 'action'.
        :param metric_fn: Function to compute the distance between predicted and ground truth actions.
        :return: Average metric score across the evaluation dataset.
        """

        examples = self._get_next_batch()
        score = 0.0
        num_samples = len(examples)
        actions = [example["action"] for example in examples]  # label
        # Predict actions using the model
        model_for_eval = self.model.module if hasattr(self.model, "module") else self.model
        output_dict = model_for_eval.predict_action(
            examples=examples,
            use_ddim=True,
            num_ddim_steps=20,
            isolated_memory_bank=True,
        )

        if self.accelerator.is_main_process:
            normalized_actions = output_dict["normalized_actions"]  # B, T, D
            actions = np.array(actions)  # convert actions to numpy.ndarray
            # B, Chunk, dim = actions.shape
            num_pots = np.prod(actions.shape)
            # Compute the metric score
            score = TrainerUtils.euclidean_distance(normalized_actions, actions)
            average_score = score / num_pots
            step_metrics["mse_score"] = average_score

        del examples
        dist.barrier()  # ensure all processes are synchronized
        return step_metrics

    def _log_training_config(self):
        """record training config"""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.config.trainer.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")
            memory_cfg = getattr(self.config.framework, "memory_buffer", {}) or {}
            injection_cfg = memory_cfg.get("qwen_memory_injection", {}) if hasattr(memory_cfg, "get") else {}
            keyframe_image_cfg = self.config.datasets.vla_data.get("keyframe_image_memory", {})
            temporal_image_cfg = self.config.datasets.vla_data.get("temporal", {}).get("image", {})
            memory_ablation_mode = getattr(
                self.config.framework,
                "memory_ablation_mode",
                injection_cfg.get("mode", "pure_image_keyframe_memory"),
            )
            logger.info(f"  Memory ablation mode = {memory_ablation_mode}")
            logger.info(f"  Resolved profile = {memory_ablation_mode}")
            logger.info(f"  Keyframe image input mode = {injection_cfg.get('mode', 'pure_image_keyframe_memory')}")
            logger.info(f"  Keyframe image memory enabled = {keyframe_image_cfg.get('enabled', False)}")
            logger.info(
                f"  Model-side memory buffer enabled = {memory_cfg.get('enable', False)}"
            )
            logger.info(
                "  Temporal image absolute_indices = %s",
                list(temporal_image_cfg.get("absolute_indices", [])),
            )
            logger.info(
                "  Temporal image delta_indices = %s",
                list(temporal_image_cfg.get("delta_indices", [0])),
            )
            logger.info(f"  Max keyframe images = {injection_cfg.get('max_keyframe_images', keyframe_image_cfg.get('max_keyframes', 0))}")

    @staticmethod
    def _cfg_get_value(container, key: str, default=None):
        if container is None:
            return default
        if hasattr(container, "get"):
            return container.get(key, default)
        return getattr(container, key, default)

    @staticmethod
    def _cfg_get_bool(container, key: str, default: bool = False) -> bool:
        value = VLATrainer._cfg_get_value(container, key, default)
        if isinstance(value, str):
            return value.strip().lower() not in {"", "0", "false", "no", "off", "none", "null"}
        return bool(value)

    @staticmethod
    def _cfg_get_int(container, key: str, default: int = 0) -> int:
        value = VLATrainer._cfg_get_value(container, key, default)
        if value is None:
            return int(default)
        return int(value)

    @staticmethod
    def _to_python_scalar(value):
        if hasattr(value, "item"):
            value = value.item()
        return value

    @staticmethod
    def _to_int_list(values) -> list[int]:
        if values is None:
            return []
        if hasattr(values, "detach"):
            values = values.detach().cpu().view(-1).tolist()
        elif isinstance(values, np.ndarray):
            values = values.reshape(-1).tolist()
        elif not isinstance(values, (list, tuple)):
            values = [values]
        return [int(VLATrainer._to_python_scalar(value)) for value in values]

    def _log_keyframe_sample_inputs(self, batch_vla, output_dict: dict) -> None:
        if not isinstance(batch_vla, list) or len(batch_vla) == 0:
            return
        if not self._cfg_get_bool(self.config.trainer, "log_keyframe_samples", True):
            return

        interval = max(1, self._cfg_get_int(self.config.trainer, "log_keyframe_samples_interval", 1))
        if int(self.completed_steps) % interval != 0:
            return

        rank = dist.get_rank() if dist.is_initialized() else 0
        log_all_ranks = self._cfg_get_bool(self.config.trainer, "log_keyframe_samples_all_ranks", False)
        if (not log_all_ranks) and rank != 0:
            return

        pred_offsets = self._to_int_list(output_dict.get("pred_event_offset", []))
        pred_triggers = output_dict.get("should_trigger_event", output_dict.get("predicted_is_keyframe", None))
        if pred_triggers is not None:
            if hasattr(pred_triggers, "detach"):
                pred_triggers = pred_triggers.detach().cpu().view(-1).tolist()
            elif isinstance(pred_triggers, np.ndarray):
                pred_triggers = pred_triggers.reshape(-1).tolist()
            elif not isinstance(pred_triggers, (list, tuple)):
                pred_triggers = [pred_triggers]
            pred_triggers = [bool(self._to_python_scalar(value)) for value in pred_triggers]
        else:
            pred_triggers = []

        for sample_idx, example in enumerate(batch_vla):
            timestep = example.get("timestep", None)
            sample_timestep = None if timestep is None else int(self._to_python_scalar(timestep))
            actual_memory_source = example.get("keyframe_input_memory_source", "dataloader")
            dataloader_memory_steps = self._to_int_list(example.get("keyframe_input_dataloader_steps", []))
            runtime_memory_steps = self._to_int_list(example.get("keyframe_input_runtime_steps", []))
            input_keyframe_steps = self._to_int_list(
                example.get("keyframe_input_steps", example.get("memory_keyframe_steps", []))
            )

            chunk_gt_steps = self._to_int_list(example.get("chunk_keyframe_exact_steps", []))
            if chunk_gt_steps:
                gt_keyframe = chunk_gt_steps[0]
            else:
                teacher_commit_timestep = int(
                    self._to_python_scalar(example.get("teacher_commit_timestep", -1))
                )
                teacher_should_commit = bool(
                    self._to_python_scalar(example.get("teacher_should_commit", False))
                )
                gt_keyframe = teacher_commit_timestep if teacher_should_commit and teacher_commit_timestep >= 0 else -1

            pred_kf = -1
            if (
                sample_timestep is not None
                and sample_idx < len(pred_offsets)
                and int(pred_offsets[sample_idx]) >= 0
                and (sample_idx >= len(pred_triggers) or pred_triggers[sample_idx])
            ):
                pred_kf = int(sample_timestep) + int(pred_offsets[sample_idx])
            logger.info(
                "[keyframe_input_sample] sample_timestep=%s actual_memory_source=%s "
                "dataloader_memory_steps=%s runtime_memory_steps=%s input_keyframe_steps=%s "
                "pred_kf=%d gt_keyframe=%d",
                sample_timestep,
                actual_memory_source,
                dataloader_memory_steps,
                runtime_memory_steps,
                input_keyframe_steps,
                int(pred_kf),
                int(gt_keyframe),
            )

    @staticmethod
    def _add_keyframe_image_batch_metrics(metrics: dict, batch_vla) -> None:
        if not isinstance(batch_vla, list) or len(batch_vla) == 0:
            return

        anchor_counts = []
        memory_counts = []
        future_step_violations = 0
        for example in batch_vla:
            images = example.get("image", [])
            memory_images = example.get("memory_keyframe_images", [])
            memory_steps = example.get("memory_keyframe_steps", [])
            timestep = example.get("timestep", None)
            if hasattr(timestep, "item"):
                timestep = timestep.item()

            anchor_counts.append(len(images) if images is not None else 0)
            memory_counts.append(len(memory_images) if memory_images is not None else 0)
            if timestep is not None and memory_steps is not None:
                for step in memory_steps:
                    if hasattr(step, "item"):
                        step = step.item()
                    if int(step) > int(timestep):
                        future_step_violations += 1

        if anchor_counts:
            metrics["num_anchor_images"] = float(np.mean(anchor_counts))
        if memory_counts:
            metrics["num_memory_keyframe_images"] = float(np.mean(memory_counts))
            metrics["max_memory_keyframe_images"] = float(np.max(memory_counts))
        metrics["memory_keyframe_future_step_violations"] = float(future_step_violations)

    def _attach_predict_exact_fetches(self, batch_vla):
        if not isinstance(batch_vla, list) or len(batch_vla) == 0:
            return batch_vla
        model_for_fetch = self.model.module if hasattr(self.model, "module") else self.model
        collect_requests = getattr(model_for_fetch, "collect_due_predict_exact_fetch_requests", None)
        if not callable(collect_requests):
            return batch_vla

        for example in batch_vla:
            if isinstance(example, dict):
                example.pop("runtime_memory_exact_fetches", None)
                example.pop("predict_exact_fetch_request_count", None)
                example.pop("predict_exact_fetch_success", None)
                example.pop("predict_exact_fetch_missing", None)

        requests = collect_requests(batch_vla)
        if not requests:
            return batch_vla

        dataset = getattr(self.vla_train_dataloader, "dataset", None)
        fetch_image = getattr(dataset, "get_memory_image_at_step", None)
        fetched_items = []
        missing_count = 0
        if not callable(fetch_image):
            missing_count = len(requests)
        else:
            for request in requests:
                try:
                    fetched = fetch_image(request)
                except Exception as exc:
                    missing_count += 1
                    logger.warning(
                        "Failed to fetch predict exact memory image "
                        f"request={request.get('request_id', request)}: {exc}"
                    )
                    continue
                if fetched is None:
                    missing_count += 1
                    continue
                fetched_items.append(fetched)

        batch_vla[0]["runtime_memory_exact_fetches"] = fetched_items
        batch_vla[0]["predict_exact_fetch_request_count"] = int(len(requests))
        batch_vla[0]["predict_exact_fetch_success"] = int(len(fetched_items))
        batch_vla[0]["predict_exact_fetch_missing"] = int(missing_count)
        return batch_vla

    def _train_step(self, batch_vla, batch_vlm=None):
        """execute single training step"""
        # Initialize counters for reset stats if they don't exist
        if not hasattr(self, "reset_stats_counter"):
            self.reset_stats_counter = 0
            self.total_resets_in_window = 0
            self.total_episodes_in_window = 0
            self._slot_episode_id = None
            self._slot_timestep = None
            self._slot_history = None

        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()

            # If sequence-aware batching is enabled, reset memory for slots that switched to new episodes.
            if isinstance(batch_vla, list) and len(batch_vla) > 0 and "is_new_episode" in batch_vla[0]:
                rank = dist.get_rank() if dist.is_initialized() else 0
                batch_size = len(batch_vla)
                reset_mask = torch.tensor(
                    [bool(example.get("is_new_episode", False)) for example in batch_vla],
                    dtype=torch.bool,
                )

                # Collect episode_id / timestep for detailed logging
                episode_ids = []
                timesteps = []
                for example in batch_vla:
                    ep_id = example.get("episode_id", None)
                    if hasattr(ep_id, "item"):
                        ep_id = ep_id.item()
                    episode_ids.append(ep_id)
                    ts = example.get("timestep", None)
                    if hasattr(ts, "item"):
                        ts = ts.item()
                    timesteps.append(ts)

                # Initialize per-slot tracking if needed
                if self._slot_episode_id is None or len(self._slot_episode_id) != batch_size:
                    self._slot_episode_id = [None] * batch_size
                    self._slot_timestep = [None] * batch_size
                    from collections import deque
                    self._slot_history = [deque(maxlen=10) for _ in range(batch_size)]

                # Update stats
                num_new_episodes = reset_mask.sum().item()
                self.total_episodes_in_window += num_new_episodes
                if num_new_episodes > 0:
                    self.total_resets_in_window += 1

                model_for_reset = self.model.module if hasattr(self.model, "module") else self.model
                if hasattr(model_for_reset, "reset_memory_by_mask"):
                    model_for_reset.reset_memory_by_mask(reset_mask, episode_ids=episode_ids)

                # Update per-slot tracking
                for idx in range(batch_size):
                    self._slot_episode_id[idx] = episode_ids[idx]
                    self._slot_timestep[idx] = timesteps[idx]
                    self._slot_history[idx].append((episode_ids[idx], timesteps[idx]))

            # VLA task forward propagation
            model_for_step = self.model.module if hasattr(self.model, "module") else self.model
            if hasattr(model_for_step, "set_keyframe_schedule_state"):
                model_for_step.set_keyframe_schedule_state(
                    completed_steps=self.completed_steps,
                    max_train_steps=self.config.trainer.max_train_steps,
                )
            batch_vla = self._attach_predict_exact_fetches(batch_vla)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(batch_vla)

                action_loss = output_dict["action_loss"]
                total_loss = output_dict.get("total_loss", action_loss)
            self._log_keyframe_sample_inputs(batch_vla, output_dict)

            if self.has_trainable_params:
                # VLA backward propagation
                self.accelerator.backward(total_loss)

                # gradient clipping
                if self.config.trainer.gradient_clipping is not None:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)

                # optimizer step
                self.optimizer.step()
                self.lr_scheduler.step()

        metrics = {
            "action_dit_loss": action_loss.item(),
            "total_loss": total_loss.item(),
        }

        keyframe_loss = output_dict.get("keyframe_loss", None)
        if keyframe_loss is not None:
            metrics["keyframe_loss"] = keyframe_loss.item()
            metrics["keyframe_loss_supervised"] = 1.0
        else:
            metrics["keyframe_loss"] = 0.0
            metrics["keyframe_loss_supervised"] = 0.0

        for metric_name in [
            "chunk_keyframe_accuracy",
            "chunk_keyframe_pred_rate",
            "chunk_keyframe_target_rate",
            "chunk_keyframe_recall",
            "chunk_keyframe_precision",
            "event_commit_accuracy",
            "event_commit_pred_rate",
            "event_commit_target_rate",
            "event_commit_recall",
            "event_commit_precision",
            "event_offset_mae",
            "keyframe_accuracy",
            "keyframe_pred_rate",
            "keyframe_target_rate",
            "keyframe_recall",
            "keyframe_precision",
            "keyframe_memory_rate",
            "keyframe_memory_teacher_prob",
            "keyframe_memory_teacher_usage",
            "keyframe_memory_predict_usage",
            "keyframe_memory_schedule_progress",
            "keyframe_input_teacher_prob",
            "keyframe_input_teacher_usage",
            "keyframe_input_predict_usage",
            "keyframe_input_schedule_progress",
            "keyframe_input_memory_count",
            "runtime_keyframe_bank_count",
            "runtime_pending_keyframe_count",
            "runtime_memory_exact_fetch_consumed",
            "runtime_memory_exact_fetch_dropped",
            "predict_exact_pending_registered",
            "keyframe_head_enabled",
            "keyframe_annotation_rate",
        ]:
            metric_value = output_dict.get(metric_name, None)
            if metric_value is not None:
                metrics[metric_name] = metric_value.item()

        self._add_keyframe_image_batch_metrics(metrics, batch_vla)
        if isinstance(batch_vla, list) and len(batch_vla) > 0:
            for metric_name in [
                "predict_exact_fetch_request_count",
                "predict_exact_fetch_success",
                "predict_exact_fetch_missing",
            ]:
                if metric_name in batch_vla[0]:
                    metrics[metric_name] = float(batch_vla[0][metric_name])

        if not self.has_trainable_params:
            metrics["all_params_frozen"] = 1.0
        return metrics

    def _finalize_training(self):
        """training end processing"""
        # save final model
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            state_dict = self.accelerator.get_state_dict(self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, os.path.join(final_checkpoint, "model.safetensors"))
            elif save_format == "pt":
                torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")


        # close W&B
        if self.accelerator.is_main_process:
            wandb.finish()

        self.accelerator.wait_for_everyone()


def main(cfg, accelerator) -> None:
    logger.info("VLA Training :: Warming Up")

    #  Wrap config to enable access tracking
    cfg = wrap_config(cfg)
    logger.info("✅ Configuration wrapped for access tracking")

    # create output directory and save config
    output_dir = setup_directories(cfg=cfg)
    # build model
    vla = build_framework(cfg)
    # prepare data
    vla_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)

    # set optimizer and scheduler
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    # create trainer
    # Run VLA Training
    trainer = VLATrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )

    # execute training preparation
    trainer.prepare_training()
    # execute training
    trainer.train()

    # And... we're done!
    logger.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/RoboTwin-Mem/train_files/eventvla_robotwin_mem.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    # Load YAML config & Convert CLI overrides to dotlist config
    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)  # Normalize CLI args to dotlist format
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)
    if OmegaConf.select(cfg, "framework.memory_ablation_mode", default=None) is not None:
        profile = resolve_and_apply_memory_ablation_profile(cfg)
        logger.info(
            "Resolved framework.memory_ablation_mode=%s to keyframe_image_mode=%s, model_memory_buffer.enable=%s, keyframe_image_memory.enabled=%s.",
            profile.name,
            profile.qwen_memory_injection_mode,
            profile.enable_memory_buffer,
            profile.enable_keyframe_image_memory,
        )

    # Build accelerator after CLI overrides are merged so runtime args
    # (e.g., --trainer.gradient_accumulation_steps) are truly effective.
    deepspeed_plugin = DeepSpeedPlugin()
    grad_acc_steps = int(getattr(cfg.trainer, "gradient_accumulation_steps", 1))
    grad_acc_plugin = GradientAccumulationPlugin(
        num_steps=grad_acc_steps,
        sync_each_batch=True,  # avoid no_sync path, incompatible with DeepSpeed ZeRO-2
    )
    accelerator = Accelerator(
        deepspeed_plugin=deepspeed_plugin,
        gradient_accumulation_plugin=grad_acc_plugin,
    )
    accelerator.print(accelerator.state)

    # if cfg.is_debug:
    if cfg.is_debug and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy
        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    main(cfg, accelerator)
