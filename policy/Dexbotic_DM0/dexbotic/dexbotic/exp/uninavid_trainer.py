"""UniNaVid Trainer: length-grouped sampling, per-module LR scaling, projector checkpoint save."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import Sampler
from transformers import TrainingArguments
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names
from transformers.trainer_utils import has_length
from transformers.utils import is_sagemaker_mp_enabled

from dexbotic.exp.trainer import DexboticTrainer


def _split_to_even_chunks(indices, lengths, num_chunks):
    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks
    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")
    return chunks


def _length_grouped_sample_indices(lengths, batch_size, world_size, generator=None):
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [
        indices[i : i + megabatch_size].tolist()
        for i in range(0, len(lengths), megabatch_size)
    ]
    megabatches = [
        sorted(megabatch, key=lambda i: lengths[i], reverse=True)
        for megabatch in megabatches
    ]
    megabatches = [
        _split_to_even_chunks(megabatch, lengths, world_size)
        for megabatch in megabatches
    ]
    return [i for megabatch in megabatches for batch in megabatch for i in batch]


def _modality_length_grouped_sample_indices(
    lengths, batch_size, world_size, generator=None
):
    assert all(l != 0 for l in lengths), "Should not have zero length."
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    assert len(mm_indices) > 0, "Should have at least one multimodal sample."

    mm_shuffle = [
        mm_indices[i]
        for i in _length_grouped_sample_indices(
            mm_lengths, batch_size, world_size, generator=None
        )
    ]
    megabatch_size = world_size * batch_size
    mm_megabatches = [
        mm_shuffle[i : i + megabatch_size]
        for i in range(0, len(mm_shuffle), megabatch_size)
    ]

    last_mm = mm_megabatches[-1]
    additional_batch = last_mm
    megabatches = mm_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) >= megabatch_size:
        megabatches = [additional_batch[:megabatch_size]] + megabatches
        additional_batch = additional_batch[megabatch_size:]

    if len(additional_batch) > 0:
        megabatches.append(additional_batch)

    return [i for megabatch in megabatches for i in megabatch]


class LengthGroupedSampler(Sampler):
    """Samples indices grouped by approximate sequence length (multimodal or generic)."""

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")
        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = _modality_length_grouped_sample_indices(
                self.lengths, self.batch_size, self.world_size, generator=self.generator
            )
        else:
            indices = _length_grouped_sample_indices(
                self.lengths, self.batch_size, self.world_size, generator=self.generator
            )
        return iter(indices)


@dataclass
class UniNaVidTrainingArguments(TrainingArguments):
    """Extends HF TrainingArguments with UniNaVid-specific flags."""

    group_by_modality_length: bool = field(default=False)
    lr_multi: Optional[str] = field(default=None)
    tune_mm_mlp_adapter: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)


class DexboticUniNaVidTrainer(DexboticTrainer):
    """Extends DexboticTrainer with:

    - Optional modality-length-grouped sampler.
    - Per-module LR scaling via ``lr_multi`` (``"key1:scale1\\key2:scale2"``).
    - Projector-only checkpoint when ``tune_mm_mlp_adapter`` is set.
    """
    def _link_exp_config(self) -> UniNaVidTrainingArguments:
        # NOTE: A full override is required here (rather than calling super()) because
        # DexboticTrainer._link_exp_config() instantiates TrainingArguments directly,
        # whereas we need UniNaVidTrainingArguments.  We replicate the base field
        # mapping and add UniNaVid-specific fields (group_by_modality_length, lr_multi,
        # tune_mm_mlp_adapter, freeze_mm_mlp_adapter) plus
        # warmup_ratio which the base currently does not forward.
        tc = self.exp_config.trainer_config
        oc = self.exp_config.optimizer_config

        linked_args: Dict[str, Any] = {
            # ---- fields mirrored from DexboticTrainer._link_exp_config ----
            "output_dir": tc.output_dir,
            "num_train_epochs": tc.num_train_epochs,
            "max_steps": tc.num_train_steps,
            "per_device_train_batch_size": tc.per_device_train_batch_size,
            "gradient_accumulation_steps": tc.gradient_accumulation_steps,
            "save_strategy": tc.save_strategy,
            "save_steps": tc.save_steps,
            "save_total_limit": tc.save_total_limit,
            "save_only_model": tc.save_only_model,
            "logging_steps": tc.logging_steps,
            "gradient_checkpointing": tc.gradient_checkpointing,
            "dataloader_num_workers": tc.dataloader_num_workers,
            "bf16": tc.bf16,
            "tf32": tc.tf32,
            "lr_scheduler_type": tc.lr_scheduler_type,
            "lr_scheduler_kwargs": tc.lr_scheduler_kwargs,
            "run_name": tc.run_name,
            "remove_unused_columns": False,
            "deepspeed": tc.deepspeed,
            "learning_rate": oc.base_lr,
            "adam_beta1": oc.adam_beta1,
            "adam_beta2": oc.adam_beta2,
            "warmup_steps": oc.warmup_steps,
            "warmup_ratio": oc.warmup_ratio,
            "weight_decay": oc.weight_decay,
            # ---- UniNaVid-specific fields ----
            "group_by_modality_length": tc.group_by_modality_length,
            "lr_multi": tc.lr_multi,
            "tune_mm_mlp_adapter": tc.tune_mm_mlp_adapter,
            "freeze_mm_mlp_adapter": tc.freeze_mm_mlp_adapter,
        }
        self.added_args = tc.added_args_dict()
        return UniNaVidTrainingArguments(**linked_args)

    def _get_train_sampler(self) -> Optional[Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None
        if getattr(self.args, "group_by_modality_length", False):
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size,
                lengths=self.train_dataset.modality_lengths,
                group_by_modality=True,
            )
        return super()._get_train_sampler()

    def create_optimizer(self) -> torch.optim.Optimizer:
        # SageMaker MP delegates entirely to base.
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        if self.optimizer is not None:
            return self.optimizer

        opt_model = self.model
        decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]

        lr_multi_str: Optional[str] = getattr(self.args, "lr_multi", None)
        if lr_multi_str is not None:
            # Parse "module_key:lr_scale\module_key2:lr_scale2" into a dict.
            lr_multi_dict: Dict[str, float] = {}
            for entry in lr_multi_str.split("\\"):
                key, val = entry.split(":")
                lr_multi_dict[key] = float(val)

            # Default groups (parameters not matched by any key).
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (
                            n in decay_parameters
                            and p.requires_grad
                            and not any(k in n for k in lr_multi_dict)
                        )
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (
                            n not in decay_parameters
                            and p.requires_grad
                            and not any(k in n for k in lr_multi_dict)
                        )
                    ],
                    "weight_decay": 0.0,
                },
            ]
            # Per-module scaled LR groups.
            for key, scale in lr_multi_dict.items():
                key_decay = [
                    p
                    for n, p in opt_model.named_parameters()
                    if n in decay_parameters and p.requires_grad and key in n
                ]
                key_no_decay = [
                    p
                    for n, p in opt_model.named_parameters()
                    if n not in decay_parameters and p.requires_grad and key in n
                ]
                if key_decay:
                    optimizer_grouped_parameters.append(
                        {
                            "params": key_decay,
                            "lr": self.args.learning_rate * scale,
                            "weight_decay": self.args.weight_decay,
                        }
                    )
                if key_no_decay:
                    optimizer_grouped_parameters.append(
                        {
                            "params": key_no_decay,
                            "lr": self.args.learning_rate * scale,
                            "weight_decay": 0.0,
                        }
                    )
        else:
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if n in decay_parameters and p.requires_grad
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if n not in decay_parameters and p.requires_grad
                    ],
                    "weight_decay": 0.0,
                },
            ]

        optimizer_cls, optimizer_kwargs = DexboticTrainer.get_optimizer_cls_and_kwargs(
            self.args
        )
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        if optimizer_cls.__name__ == "Adam8bit":
            import bitsandbytes

            manager = bitsandbytes.optim.GlobalOptimManager.get_instance()
            skipped = 0
            for module in opt_model.modules():
                if isinstance(module, nn.Embedding):
                    skipped += sum(
                        {p.data_ptr(): p.numel() for p in module.parameters()}.values()
                    )
                    logger.info(f"skipped {module}: {skipped / 2**20}M params")
                    manager.register_module_override(
                        module, "weight", {"optim_bits": 32}
                    )
                    logger.debug("bitsandbytes: will optimize {} in fp32", module)
            logger.info(f"skipped: {skipped / 2**20}M params")

        return self.optimizer
