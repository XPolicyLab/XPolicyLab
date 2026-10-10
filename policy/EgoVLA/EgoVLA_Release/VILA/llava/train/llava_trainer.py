# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# This file is modified from https://github.com/haotian-liu/LLaVA/


import json
import os
import random
import time
from typing import Dict, List, Optional
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data import DataLoader, Dataset

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import ConcatDataset, Dataset, DistributedSampler, RandomSampler, Sampler
from transformers import PreTrainedModel, Trainer
from transformers.modeling_utils import unwrap_model
from transformers.trainer import ALL_LAYERNORM_LAYERS  # ShardedDDPOption,
from transformers.trainer import get_parameter_names, has_length, is_sagemaker_mp_enabled, logger

from llava.train.sequence_parallel import get_pg_manager
from llava.trl.trainer import DPOTrainer

from transformers.trainer_utils import (
    EvalLoopOutput,
    EvalPrediction,
    denumpify_detensorize,
    has_length,
)
from transformers.integrations.deepspeed import deepspeed_init

from transformers.trainer_pt_utils import (
    DistributedTensorGatherer,
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_dataloader_sampler,
    get_model_param_count,
    get_module_class_from_name,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
    remove_dummy_checkpoint,
)
from transformers.utils import (
    ADAPTER_CONFIG_NAME,
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    PushInProgress,
    can_return_loss,
    find_labels,
    is_accelerate_available,
    is_apex_available,
    is_bitsandbytes_available,
    is_datasets_available,
    is_in_notebook,
    is_ipex_available,
    is_peft_available,
    is_safetensors_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_compile_available,
    is_torch_neuroncore_available,
    is_torch_npu_available,
    is_torch_tpu_available,
    logging,
    strtobool,
)

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met


from packaging import version

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

import numpy as np

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, "no ignore status")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

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


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [
        lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)
    ]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class VILADistributedSampler(DistributedSampler):
    """This class is implemented by Jason Lu."""

    def __init__(
        self,
        dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        batch_size=None,
        # NOTE: this is the total size but not per-worker
        sample_len_list=None,
        force_accumulation=True,
        sp_degree: int = 1,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval" " [0, {}]".format(rank, num_replicas - 1)
            )
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = True  # always True
        self.sp_degree = max(1, sp_degree)
        self.bs_divisible_by_sp = batch_size % self.sp_degree == 0

        # Consider sequence parallelism
        if self.sp_degree > 1:  # Sequence Parallelism is enabled
            PROCESS_GROUP_MANAGER = get_pg_manager()
            self.dp_rank = PROCESS_GROUP_MANAGER.dp_rank
            self.dp_num_replicas = num_replicas // sp_degree
            self.corresponding_ranks = list(range(self.dp_rank * self.sp_degree, (self.dp_rank + 1) * self.sp_degree))
        else:
            self.dp_rank = rank
            self.dp_num_replicas = num_replicas

        self.batch_size = batch_size
        self.global_batch_size = batch_size * self.dp_num_replicas

        # NOTE: org_ is without drop last
        self.org_sample_len_list = self.per_replica_samples = sample_len_list
        assert sum(sample_len_list) == len(self.dataset)

        if self.drop_last:  # type: ignore[arg-type]
            self.per_replica_samples = [
                sample_len
                // (self.num_replicas * self.batch_size // self.sp_degree)
                * self.batch_size
                // self.sp_degree
                for sample_len in self.per_replica_samples
            ]
            self.num_samples = sum(self.per_replica_samples)
        else:
            raise NotImplementedError

        self.total_size = self.num_samples * self.num_replicas
        self.total_samples = [samples * self.num_replicas for samples in self.per_replica_samples]

        self.shuffle = shuffle
        self.seed = seed

        # whether to force accumulate
        self.force_accumulation = force_accumulation

    def __len__(self) -> int:
        return self.num_samples * self.sp_degree

    def __iter__(self):

        indices = list(range(len(self.dataset)))

        # 1. split the full indices first (note: without drop last at this moment)
        indices_list = []
        for i in range(len(self.org_sample_len_list)):
            indices_list.append(
                indices[sum(self.org_sample_len_list[:i]) : sum(self.org_sample_len_list[:i]) + self.total_samples[i]]
            )

        assert sum([len(indices) for indices in indices_list]) == self.total_size, (
            sum([len(indices) for indices in indices_list]),
            self.total_size,
        )

        if (
            self.sp_degree > 1 and self.bs_divisible_by_sp
        ):  # Sequence Parallelism is enabled, to ensure the same behavior as data parallelism
            dp_indices_dict = {}  # {rank: indices_list}
            all_indices_dict = {}  # {rank: all_indices}

            for i in self.corresponding_ranks:
                dp_indices_list = []
                for idx, indices in enumerate(indices_list):
                    dp_indices_list.append(
                        indices[i * self.per_replica_samples[idx] : (i + 1) * self.per_replica_samples[idx]]
                    )

                random.seed(self.seed + self.epoch)
                for indice in range(len(dp_indices_list)):
                    random.shuffle(dp_indices_list[indice])

                dp_indices_dict[i] = dp_indices_list.copy()

            for rank, dp_indices_list in dp_indices_dict.items():
                dp_indices_list = sorted(dp_indices_list, key=lambda x: -len(x))
                dp_all_indices = [-1] * self.num_samples
                indices_available = list(range(self.num_samples))

                for indice in dp_indices_list:

                    original_indices = range(len(indice))
                    transformed_indices = [idx * len(indices_available) // len(indice) for idx in original_indices]

                    mapped_indices = [indices_available[idx] for idx in transformed_indices]
                    # update indices_available
                    for idx in reversed(transformed_indices):
                        del indices_available[idx]
                    for i, idx in enumerate(mapped_indices):
                        dp_all_indices[idx] = indice[i]

                all_indices_dict[rank] = dp_all_indices

            # Interleaving Merge
            merged_indices = []
            interleaved_indices = []
            for item_idx in range(len(all_indices_dict[self.corresponding_ranks[0]])):
                for rank in self.corresponding_ranks:
                    interleaved_indices.append(all_indices_dict[rank][item_idx])
            merged_indices.append(interleaved_indices)

            all_indices = merged_indices[0]
        else:
            # let's first do subsample
            for idx, indices in enumerate(indices_list):
                indices_list[idx] = indices[
                    self.rank * self.per_replica_samples[idx] : (self.rank + 1) * self.per_replica_samples[idx]
                ]

            random.seed(self.seed + self.epoch)
            for indice in range(len(indices_list)):
                random.shuffle(indices_list[indice])

            indices_list = sorted(indices_list, key=lambda x: -len(x))
            all_indices = [-1] * self.num_samples
            indices_available = list(range(self.num_samples))

            for indice in indices_list:

                original_indices = range(len(indice))
                transformed_indices = [idx * len(indices_available) // len(indice) for idx in original_indices]

                mapped_indices = [indices_available[idx] for idx in transformed_indices]
                # update indices_available
                for idx in reversed(transformed_indices):
                    del indices_available[idx]
                for i, idx in enumerate(mapped_indices):
                    all_indices[idx] = indice[i]
        assert -1 not in all_indices
        return iter(all_indices)


class LongVILADistributedSampler(VILADistributedSampler):
    """This class is implemented by Yukang Chen."""

    def __iter__(self):
        def batch_shuffle(indices):
            batch_indices = list(range(indices[0] // self.batch_size, indices[-1] // self.batch_size + 1))
            random.shuffle(batch_indices)
            indices_shuffled = [
                batch_indices[i // self.batch_size] * self.batch_size + index % self.batch_size
                for i, index in enumerate(indices)
            ]
            return indices_shuffled

        indices = list(range(len(self.dataset)))

        # 1. split the full indices first (note: without drop last at this moment)
        indices_list = []
        for i in range(len(self.org_sample_len_list)):
            indices_list.append(
                indices[sum(self.org_sample_len_list[:i]) : sum(self.org_sample_len_list[:i]) + self.total_samples[i]]
            )

        assert sum([len(indices) for indices in indices_list]) == self.total_size, (
            sum([len(indices) for indices in indices_list]),
            self.total_size,
        )

        if self.sp_degree > 1:  # Sequence Parallelism is enabled, to ensure the same behavior as data parallelism
            dp_indices_dict = {}  # {rank: indices_list}
            all_indices_dict = {}  # {rank: all_indices}

            for i in self.corresponding_ranks:
                dp_indices_list = []
                for idx, indices in enumerate(indices_list):
                    dp_indices_list.append(
                        indices[i * self.per_replica_samples[idx] : (i + 1) * self.per_replica_samples[idx]]
                    )

                random.seed(self.seed + self.epoch)
                for indice in range(len(dp_indices_list)):
                    batch_shuffle(dp_indices_list[indice])

                dp_indices_dict[i] = dp_indices_list.copy()

            for rank, dp_indices_list in dp_indices_dict.items():
                dp_indices_list = sorted(dp_indices_list, key=lambda x: -len(x))
                dp_all_indices = [-1] * self.num_samples
                indices_available = list(range(self.num_samples))

                for indice in dp_indices_list:

                    original_indices = range(len(indice))
                    transformed_indices = [idx * len(indices_available) // len(indice) for idx in original_indices]

                    mapped_indices = [indices_available[idx] for idx in transformed_indices]
                    # update indices_available
                    for idx in reversed(transformed_indices):
                        del indices_available[idx]
                    for i, idx in enumerate(mapped_indices):
                        dp_all_indices[idx] = indice[i]

                all_indices_dict[rank] = dp_all_indices

            # Interleaving Merge
            merged_indices = []
            interleaved_indices = []
            for item_idx in range(len(all_indices_dict[self.corresponding_ranks[0]])):
                for rank in self.corresponding_ranks:
                    interleaved_indices.append(all_indices_dict[rank][item_idx])
            merged_indices.append(interleaved_indices)

            all_indices = merged_indices[0]
        else:
            # let's first do subsample
            for idx, indices in enumerate(indices_list):
                indices_list[idx] = indices[
                    self.rank * self.per_replica_samples[idx] : (self.rank + 1) * self.per_replica_samples[idx]
                ]

            random.seed(self.seed + self.epoch)
            for indice in range(len(indices_list)):
                batch_shuffle(indices_list[indice])

            indices_list = sorted(indices_list, key=lambda x: -len(x))
            all_indices = [-1] * self.num_samples
            indices_available = list(range(self.num_samples))
            for indice in indices_list:
                original_indices = range(len(indice))
                transformed_indices = [idx * len(indices_available) // len(indice) for idx in original_indices]
                mapped_indices = [indices_available[idx] for idx in transformed_indices]
                # update indices_available
                for idx in reversed(transformed_indices):
                    del indices_available[idx]
                for i, idx in enumerate(mapped_indices):
                    all_indices[idx] = indice[i]
        assert -1 not in all_indices
        return iter(all_indices)


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

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
            indices = get_modality_length_grouped_indices(
                self.lengths, self.batch_size, self.world_size, generator=self.generator
            )
        else:
            indices = get_length_grouped_indices(
                self.lengths, self.batch_size, self.world_size, generator=self.generator
            )
        return iter(indices)


class VILADPOTrainer(DPOTrainer):
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        # Always using Jason's sampler.
        sample_len_list = self.args.sample_lens
        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed
        num_replicas = self.args.world_size
        rank = self.args.process_index
        return VILADistributedSampler(
            self.train_dataset,
            num_replicas=num_replicas,
            rank=rank,
            seed=seed,
            batch_size=self.args.train_batch_size,
            sample_len_list=sample_len_list,
            sp_degree=self.args.seq_parallel_size,
        )

    def _get_eval_sampler(self, eval_dataset: Dataset) -> Optional[torch.utils.data.Sampler]:
        if self.eval_dataset is None or not has_length(self.eval_dataset):
            return None

        # Always using Jason's sampler.
        sample_len_list = self.args.eval_sample_lens
        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed
        return VILADistributedSampler(
            eval_dataset,
            num_replicas=self.args.world_size,
            rank=self.args.process_index,
            seed=seed,
            batch_size=self.args.eval_batch_size,
            sample_len_list=sample_len_list,
        )

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()
        # if self.sharded_ddp == ShardedDDPOption.SIMPLE:
        #     return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            if 0:  # self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
                if optimizer_cls.__name__ == "Adam8bit":
                    import bitsandbytes

                    manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                    skipped = 0
                    for module in opt_model.modules():
                        if isinstance(module, nn.Embedding):
                            skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                            logger.info(f"skipped {module}: {skipped/2**20}M params")
                            manager.register_module_override(module, "weight", {"optim_bits": 32})
                            logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                    logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def save_model(self, output_dir: Optional[str], _internal_call: bool):
        ## save tuned model separately
        if self.is_deepspeed_enabled:
            state_dict = self.accelerator.get_state_dict(self.deepspeed)
        else:
            # TODO(ligeng): fix save_model for multi-node training on large models (e.g., Llama-70b)
            state_dict = self.model.state_dict()

        if self.args.should_save:
            return self.model.save_pretrained(output_dir, state_dict=state_dict)


class LLaVATrainer(Trainer):
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        # Always using Jason's sampler.
        sample_len_list = self.args.sample_lens
        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed
        num_replicas = self.args.world_size
        rank = self.args.process_index
        longvila_sampler = self.args.longvila_sampler
        sampler = LongVILADistributedSampler if longvila_sampler else VILADistributedSampler

        # # Consider sequence parallelism
        # sp_degree = self.args.seq_parallel_size
        # if sp_degree > 1:  # Sequence Parallelism is enabled
        #     num_replicas = num_replicas // sp_degree
        #     PROCESS_GROUP_MANAGER = get_pg_manager()
        #     rank = PROCESS_GROUP_MANAGER.dp_rank
        #     # rank = dist.get_rank() // sp_degree

        return sampler(
            self.train_dataset,
            num_replicas=num_replicas,
            rank=rank,
            seed=seed,
            batch_size=self.args.train_batch_size,
            sample_len_list=sample_len_list,
            sp_degree=self.args.seq_parallel_size,
        )

        # if self.args.group_by_modality_length:
        #     if not isinstance(self.train_dataset, ConcatDataset):
        #         lengths = self.train_dataset.modality_lengths
        #     else:
        #         lengths = []
        #         for d in self.train_dataset.datasets:
        #             lengths += d.modality_lengths
        #     return LengthGroupedSampler(
        #         self.args.train_batch_size,
        #         world_size=self.args.world_size * self.args.gradient_accumulation_steps,
        #         lengths=lengths,
        #         group_by_modality=True,
        #     )
        # else:
        #     return super()._get_train_sampler()

    def _get_eval_sampler(self, eval_dataset: Dataset) -> Optional[torch.utils.data.Sampler]:
        if self.eval_dataset is None or not has_length(self.eval_dataset):
            return None

        # Always using Jason's sampler.
        sample_len_list = self.args.eval_sample_lens
        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed
        return VILADistributedSampler(
            eval_dataset,
            num_replicas=self.args.world_size,
            rank=self.args.process_index,
            seed=seed,
            batch_size=self.args.eval_batch_size,
            sample_len_list=sample_len_list,
        )

    def _inner_training_loop(self, batch_size: Optional[int] = None, *args, **kwargs):
        # NOTE(zhijianl): In the latest transformers, if the batch size in the training arguments differs from
        # the one in the training state, the batch size from the state is used by default. This can be
        # problematic when resuming with different batch sizes or gradient accumulation steps. To prevent this,
        # we enforce using the batch size specified in the training arguments.
        batch_size = self.args.train_batch_size
        return super()._inner_training_loop(batch_size, *args, **kwargs)

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()
        # if self.sharded_ddp == ShardedDDPOption.SIMPLE:
        #     return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            elif self.args.hand_decoder_lr is not None:
                decoder_parameters = [name for name, _ in opt_model.named_parameters() if "traj_decoder" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and n not in decoder_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and n not in decoder_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and n in decoder_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.hand_decoder_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and n in decoder_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.hand_decoder_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            if 0:  # self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
                if optimizer_cls.__name__ == "Adam8bit":
                    import bitsandbytes

                    manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                    skipped = 0
                    for module in opt_model.modules():
                        if isinstance(module, nn.Embedding):
                            skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                            logger.info(f"skipped {module}: {skipped/2**20}M params")
                            manager.register_module_override(module, "weight", {"optim_bits": 32})
                            logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                    logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def save_model(self, output_dir: Optional[str], _internal_call: bool):
        ## save tuned model separately
        if self.is_deepspeed_enabled:
            state_dict = self.accelerator.get_state_dict(self.deepspeed)
        else:
            # TODO(ligeng): fix save_model for multi-node training on large models (e.g., Llama-70b)
            state_dict = self.model.state_dict()

        if self.args.lora_enable:
            non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(self.model.named_parameters())
            os.makedirs(output_dir, exist_ok=True)
            torch.save(
                non_lora_state_dict,
                os.path.join(output_dir, "non_lora_trainables.bin"),
            )

        if self.args.should_save:
            return self.model.save_pretrained(output_dir, state_dict=state_dict)

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)
        if self.args.include_num_input_tokens_seen:
            logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)

        if self.args.debug_e2e and self.control.should_training_stop:

            # Only save log history if the current process is rank 0
            if dist.get_rank() == 0:
                with open(f"{self.args.output_dir}/log_history.json", "w") as f:
                    json.dump(self.state.log_history, f, indent=4)

        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        inputs["inference"] = True

        logging_loss_keys = self.args.logging_loss_keys

        # has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
        has_labels = False if len(self.label_names) == 0 else True
        # print(self.label_names, has_labels)
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = tuple(loss for loss in (inputs.get(name) for name in self.label_names) if loss is not None)
            # labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            labels = nested_detach(labels)
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                raw_outputs = smp_forward_only(model, inputs)
                if has_labels or loss_without_labels:
                    assert isinstance(raw_outputs, dict)
                    loss_mb = raw_outputs["loss"]
                        # logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
                    logged_losses_mb  = tuple(raw_outputs[k] for k in logging_loss_keys if k in raw_outputs)
                        # logged_loss_mb = tuple(
                        #   v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"] and "loss" in k
                        # )
                        # logged_loss_keys = tuple(
                        #   k for k in raw_outputs.keys() if k not in ignore_keys + ["loss"] and "loss" in k
                        # )
                        # logged_l
                    # else:
                    #     # assert False
                    #     loss_mb = raw_outputs[0]
                    #     logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logged_losses = smp_nested_concat(logged_losses_mb)
                else:
                    assert False
                    loss = None
                    # if isinstance(raw_outputs, dict):
                    #     logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
                    # else:
                    logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)
            else:
                if has_labels or loss_without_labels:
                    # print(inputs.keys())
                    with self.compute_loss_context_manager():
                        
                        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                    loss = loss.mean().detach()
                    # print(loss)
                    logits = outputs
                    assert isinstance(outputs, dict)
                    #     logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                    # else:
                    #     logits = outputs[1:]
                    logged_losses  = tuple(outputs[k] for k in logging_loss_keys if k in outputs)
                    # logged_losses = smp_nested_concat(logged_losses_mb)
                else:
                    assert False
                    loss = None
                    with self.compute_loss_context_manager():
                        outputs = model(**inputs)
                    # if isinstance(outputs, dict):
                    #     logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                    # else:
                    logits = outputs
                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        logged_losses = nested_detach(logged_losses)

        return (loss, logged_losses)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        # prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only
        
        prediction_loss_only = (len(args.logging_loss_keys) == 0)
        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        logged_losses_host = None
        labels_host = None
        inputs_host = None

        split_losses_host = None

        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_logged_losses = None
        all_labels = None
        all_inputs = None

        # Will be useful when we have an iterable dataset so don't know its length.
        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logged_losses = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            main_input_name = getattr(self.model, "main_input_name", "input_ids")
            inputs_decode = self._prepare_input(inputs[main_input_name]) if args.include_inputs_for_metrics else None

            if is_torch_tpu_available():
                xm.mark_step()

            # Update containers on host
            if loss is not None:
                losses = self.gather_function((loss.repeat(batch_size)))
                losses_host = losses if losses_host is None else nested_concat(losses_host, losses, padding_index=-100)


            # if labels is not None:
            #     labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
                inputs_decode = self.gather_function((inputs_decode))
                inputs_host = (
                    inputs_decode
                    if inputs_host is None
                    else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                )

            if logged_losses is not None:
                logged_losses = self.accelerator.pad_across_processes(logged_losses, dim=1, pad_index=-100)
                # if self.preprocess_logits_for_metrics is not None:
                #     logits = self.preprocess_logits_for_metrics(logits, labels)
                logged_losses = self.gather_function((logged_losses))
                logged_losses_host = logged_losses if logged_losses_host is None else nested_concat(
                  logged_losses_host, logged_losses, padding_index=-100
                )

            # if labels is not None:
            #     labels = self.gather_function((labels))
            #     labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if logged_losses_host is not None:
                    logged_losses = nested_numpify(logged_losses_host)
                    all_logged_losses = logged_losses if all_logged_losses is None else nested_concat(
                      all_logged_losses, logged_losses, padding_index=-100)
                if inputs_host is not None:
                    inputs_decode = nested_numpify(inputs_host)
                    all_inputs = (
                        inputs_decode
                        if all_inputs is None
                        else nested_concat(all_inputs, inputs_decode, padding_index=-100)
                    )
                # if labels_host is not None:
                #     labels = nested_numpify(labels_host)
                #     all_labels = (
                #         labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                #     )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, inputs_host, labels_host = None, None, None, None

        # After all calls to `.gather_function`, reset to `gather_for_metrics`:
        self.gather_function = self.accelerator.gather_for_metrics
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if logged_losses_host is not None:
            logged_losses = nested_numpify(logged_losses_host)
            all_logged_losses = logged_losses if all_logged_losses is None else nested_concat(
              all_logged_losses, logged_losses, padding_index=-100
            )
        if inputs_host is not None:
            inputs_decode = nested_numpify(inputs_host)
            all_inputs = (
                inputs_decode if all_inputs is None else nested_concat(all_inputs, inputs_decode, padding_index=-100)
            )
        # if labels_host is not None:
        #     labels = nested_numpify(labels_host)
        #     all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # # Metrics!
        # if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
        #     if args.include_inputs_for_metrics:
        #         metrics = self.compute_metrics(
        #             EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs)
        #         )
        #     else:
        #         metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        # else:
        metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        
        if all_logged_losses is not None:
            for k, v in zip(args.logging_loss_keys, all_logged_losses):
              metrics[f"{metric_key_prefix}_{k}"] = v.mean().item()
        
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=None, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

