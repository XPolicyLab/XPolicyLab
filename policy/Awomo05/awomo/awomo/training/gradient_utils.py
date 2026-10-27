# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import math
import re
from typing import Iterable

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Partial, Replicate

from awomo.awomo.training.model_utils.utils import set_is_first_microbatch, set_is_optim_step

_TE_EXPERT_PARAM_PATTERN = re.compile(r"(^|\.)mlp\.experts\.(gate_up_linear|down_linear)\.(weight|bias)\d+")


@torch.no_grad()
def count_tail_padding(labels, ignore_label=-100):
    flipped = labels.flip(dims=[1])
    tail_mask = flipped == ignore_label

    prod_mask = torch.cumprod(tail_mask.int(), dim=1)

    return prod_mask.view(-1).sum().item()


@torch.no_grad()
def _clip_grad_norm_impl(
    parameters: torch.Tensor | Iterable[torch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
    pp_mesh: DeviceMesh | None = None,
) -> torch.Tensor:
    if torch.cuda.is_available():
        target_device = torch.device(f"cuda:{torch.cuda.current_device()}")
    else:
        target_device = torch.device("cpu")

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        parameters = list(parameters)

    sharding_groups = {}

    for p in parameters:
        if p.grad is None:
            continue

        if isinstance(p, DTensor):
            mesh_id = id(p.device_mesh)
            placements_tuple = tuple(str(placement) for placement in p.placements)
            key = (mesh_id, placements_tuple)
        else:
            key = ("regular", "regular")

        if key not in sharding_groups:
            sharding_groups[key] = []
        sharding_groups[key].append(p)

    is_inf = math.isinf(norm_type)
    group_norms = []
    for group_params in sharding_groups.values():
        first = group_params[0]
        is_dtensor = isinstance(first, DTensor)
        has_partial = is_dtensor and any(isinstance(pl, Partial) for pl in first.placements)

        local_val = torch.zeros((), dtype=torch.float32, device=target_device)
        for p in group_params:
            g = p.grad
            if isinstance(g, DTensor):
                g = g.full_tensor() if has_partial else g.to_local()
            g = g.detach().float()
            if is_inf:
                local_val = torch.maximum(local_val, g.abs().max())
            else:
                local_val = local_val + g.abs().pow(norm_type).sum()

        if is_dtensor and not has_partial:
            mesh = first.device_mesh
            op = torch.distributed.ReduceOp.MAX if is_inf else torch.distributed.ReduceOp.SUM
            for dim_idx, pl in enumerate(first.placements):
                if isinstance(pl, Replicate):
                    continue
                torch.distributed.all_reduce(local_val, op=op, group=mesh.get_group(mesh_dim=dim_idx))

        group_norms.append(local_val if is_inf else local_val.pow(1.0 / norm_type))

    if len(group_norms) == 0:
        total_norm = torch.tensor(0.0, device=target_device)
    elif len(group_norms) == 1:
        total_norm = group_norms[0]
    elif is_inf:
        total_norm = torch.stack(group_norms).max()
    else:
        total_norm = torch.zeros((), dtype=torch.float32, device=target_device)
        for gn in group_norms:
            total_norm = total_norm + gn.pow(norm_type)
        total_norm = total_norm.pow(1.0 / norm_type)

    total_norm = total_norm.float().to(target_device)
    if pp_mesh is not None:
        if math.isinf(norm_type):
            torch.distributed.all_reduce(total_norm, op=torch.distributed.ReduceOp.MAX, group=pp_mesh.get_group())
        else:
            total_norm = total_norm**norm_type
            torch.distributed.all_reduce(total_norm, op=torch.distributed.ReduceOp.SUM, group=pp_mesh.get_group())
            total_norm = total_norm ** (1.0 / norm_type)

    for group_params in sharding_groups.values():
        torch.nn.utils.clip_grads_with_norm_(group_params, max_norm, total_norm, foreach)

    return total_norm


@torch.no_grad()
def clip_grad_norm(
    max_grad_norm: float | None,
    model_parts: list[torch.nn.Module],
    *,
    norm_type: float = 2.0,
    pp_enabled: bool = False,
    device_mesh: DeviceMesh | None = None,
    pp_axis_name: str | None = None,
    foreach: bool = True,
):
    if max_grad_norm is None:
        return 0.0

    parameters = [p for m in model_parts for p in m.parameters() if p.requires_grad]

    pp_mesh = None
    if pp_enabled:
        assert pp_axis_name is not None, "pp_axis_name must be provided when pp_enabled is True"
        pp_mesh = device_mesh[pp_axis_name] if device_mesh is not None else None

    grad_norm = _clip_grad_norm_impl(
        parameters=parameters,
        max_norm=max_grad_norm,
        norm_type=norm_type,
        error_if_nonfinite=False,
        foreach=foreach,
        pp_mesh=pp_mesh,
    )

    if isinstance(grad_norm, torch.Tensor):
        grad_norm = grad_norm.item() if grad_norm.numel() == 1 else grad_norm
        if hasattr(grad_norm, "full_tensor"):
            grad_norm = grad_norm.full_tensor()

    return grad_norm


def prepare_for_grad_accumulation(model_parts: list[torch.nn.Module], pp_enabled: bool = False):
    set_is_optim_step(False)
    set_is_first_microbatch(True)
    if pp_enabled:
        return

    for mp in model_parts:
        if hasattr(mp, "prepare_for_grad_accumulation"):
            mp.prepare_for_grad_accumulation(pp_enabled=pp_enabled)


def prepare_after_first_microbatch():
    set_is_first_microbatch(False)


def prepare_for_final_backward(model_parts: list[torch.nn.Module], pp_enabled: bool = False):
    set_is_optim_step(True)
    if pp_enabled:
        return

    for mp in model_parts:
        if hasattr(mp, "prepare_for_final_backward"):
            mp.prepare_for_final_backward(pp_enabled=pp_enabled)


@torch.no_grad()
def scale_grads_and_clip_grad_norm(
    max_grad_norm: float | None,
    model_parts: list[torch.nn.Module],
    *,
    norm_type: float = 2.0,
    pp_enabled: bool = False,
    device_mesh: DeviceMesh | None = None,
    moe_mesh: DeviceMesh | None = None,
    ep_axis_name: str | None = None,
    pp_axis_name: str | None = None,
    foreach: bool = True,
    num_label_tokens: int | None = None,
    dp_group_size: int | None = None,
):

    pp_divisor: float | None = None
    if pp_enabled and num_label_tokens is not None and dp_group_size is not None:
        if dp_group_size != 0:
            candidate = num_label_tokens / dp_group_size
            pp_divisor = float(candidate) if candidate != 0 else None

    ep_ratio: float | None = None
    if moe_mesh is not None and dp_group_size is not None:
        ep_shard_size = moe_mesh["ep_shard"].size() if "ep_shard" in moe_mesh.mesh_dim_names else 1
        if ep_shard_size > 0:
            ep_ratio = float(dp_group_size) / float(ep_shard_size)

    if pp_divisor is not None or ep_ratio is not None:
        for mp in model_parts:
            for name, p in mp.named_parameters():
                if p.grad is None:
                    continue
                if pp_divisor is not None:
                    p.grad.div_(pp_divisor)
                if ep_ratio is not None:
                    is_ep_sharded_dtensor = (
                        isinstance(p, DTensor)
                        and isinstance(p.grad, DTensor)
                        and ep_axis_name
                        and ep_axis_name in p.device_mesh.mesh_dim_names
                    )
                    is_expert_param = (
                        isinstance(p, torch.Tensor)
                        and isinstance(p.grad, torch.Tensor)
                        and _TE_EXPERT_PARAM_PATTERN.search(name) is not None
                    )
                    if is_ep_sharded_dtensor or is_expert_param:
                        p.grad.div_(ep_ratio)

    return clip_grad_norm(
        max_grad_norm,
        model_parts,
        norm_type=norm_type,
        pp_enabled=pp_enabled,
        device_mesh=device_mesh,
        pp_axis_name=pp_axis_name,
        foreach=foreach,
    )


def move_to_device(model, device):
    for v in model.buffers():
        v.data = v.data.to(device)
    model.to(device)
    gc.collect()
    torch.cuda.empty_cache()


class ScopedModuleOffloading:
    def __init__(self, model, enabled=False):
        self.model = model
        self.enabled = enabled

    def __enter__(self):
        if self.enabled:
            move_to_device(self.model, "cuda")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled:
            move_to_device(self.model, "cpu")
        return False
