# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FLUX block sharding and activation-checkpoint setup; no additional trainer superclass."""

from __future__ import annotations
import logging
import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl, checkpoint_wrapper
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.nn.parallel import DistributedDataParallel
from awomo.awomo.training.distributed.config import FSDP2Config
from awomo.awomo.training.distributed.mesh_utils import create_device_mesh, get_fsdp_dp_mesh

def setup_fsdp(self, fsdp_cfg):
    self._activation_checkpointing_block_count = 0
    self.__dict__['model'] = self.model.to(device=self.device)
    self.device_mesh = None
    use_fsdp = fsdp_cfg is not None and fsdp_cfg.get('enabled', False)
    if use_fsdp and dist.get_world_size() > 1:
        setup_fsdp2(self, fsdp_cfg)
    elif dist.is_initialized() and dist.get_world_size() > 1:
        self.__dict__['model'] = DistributedDataParallel(self.model, device_ids=[self.local_rank], gradient_as_bucket_view=True)

def setup_fsdp2(self, fsdp_cfg):
    assert dist.is_initialized() and dist.get_world_size() > 1
    self._broadcast_module_state(self.model.proprio_encoder)
    reduce_dtype_by_name = {'bf16': torch.bfloat16, 'fp32': torch.float32}
    reduce_dtype_name = fsdp_cfg.get('reduce_dtype', 'bf16')
    assert reduce_dtype_name in reduce_dtype_by_name
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=reduce_dtype_by_name[reduce_dtype_name], output_dtype=torch.bfloat16, cast_forward_inputs=False)
    dp_size = int(fsdp_cfg.dp_size)
    dp_replicate_size = int(fsdp_cfg.dp_replicate_size)
    tp_size = int(fsdp_cfg.tp_size)
    cp_size = int(fsdp_cfg.cp_size)
    pp_size = int(fsdp_cfg.pp_size)
    if dp_size * tp_size * cp_size * pp_size != self.world_size:
        raise ValueError(f'FSDP topology mismatch: dp_size={dp_size} * tp_size={tp_size} * cp_size={cp_size} * pp_size={pp_size} != world_size={self.world_size}')
    if tp_size != 1 or cp_size != 1 or pp_size != 1:
        raise ValueError('Action Patch currently requires fsdp.tp_size=1, cp_size=1, and pp_size=1')
    fsdp2_config = FSDP2Config(mp_policy=mp_policy, backend='nccl')
    (device_mesh, _) = create_device_mesh(fsdp2_config, dp_size=dp_size, dp_replicate_size=dp_replicate_size, tp_size=tp_size, cp_size=cp_size, pp_size=pp_size, ep_size=1, world_size=self.world_size)
    dp_mesh = get_fsdp_dp_mesh(device_mesh)
    self.device_mesh = device_mesh
    transformer = self.model.transformer
    double_blocks = transformer.transformer_blocks
    single_blocks = transformer.single_transformer_blocks
    if fsdp_cfg.get('activation_checkpointing', False):
        double_count = int(fsdp_cfg.get('activation_checkpointing_double_blocks', 0))
        single_count = int(fsdp_cfg.get('activation_checkpointing_single_blocks', len(single_blocks)))
        if not 0 <= double_count <= len(double_blocks):
            raise ValueError('fsdp.activation_checkpointing_double_blocks is out of range')
        if not 0 <= single_count <= len(single_blocks):
            raise ValueError('fsdp.activation_checkpointing_single_blocks is out of range')
    else:
        double_count = single_count = 0
    checkpoint_and_shard_blocks(self, double_blocks, double_count, mp_policy, dp_mesh)
    checkpoint_and_shard_blocks(self, single_blocks, single_count, mp_policy, dp_mesh)
    fully_shard(transformer, mesh=dp_mesh, mp_policy=mp_policy, reshard_after_forward=False)
    fully_shard(self.model, mesh=dp_mesh, mp_policy=mp_policy, reshard_after_forward=False)
    dp_shard_size = dp_size // dp_replicate_size
    logging.info(f'[INFO] HSDP applied to FLUX.2 Action-as-Patch; dp_replicate={dp_replicate_size}, dp_shard={dp_shard_size}, activation checkpointing: {self._activation_checkpointing_block_count} blocks')

def checkpoint_and_shard_blocks(self, blocks, count, mp_policy, dp_mesh):
    count = min(count, len(blocks))
    for index in range(count):
        blocks[index] = checkpoint_wrapper(blocks[index], checkpoint_impl=CheckpointImpl.NO_REENTRANT, preserve_rng_state=False)
        fully_shard(blocks[index], mesh=dp_mesh, mp_policy=mp_policy)
    self._activation_checkpointing_block_count += count
