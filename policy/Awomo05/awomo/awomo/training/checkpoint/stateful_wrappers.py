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

from functools import partial
from typing import Any, Optional

import torch
from awomo.awomo.training.shared.import_utils import safe_import_te

HAS_TE, transformer_engine = safe_import_te()

if HAS_TE:
    import transformer_engine.pytorch.module.base as te_base
    import transformer_engine.pytorch.ops.op as te_ops

    _original_set_extra_state = te_base.TransformerEngineBaseModule.set_extra_state
    _original_op_set_extra_state = te_ops.BasicOperation.set_extra_state

    def _safe_set_extra_state(self, state):
        if state is not None and "EXTRA_STATE" in str(type(state)):
            return
        return _original_set_extra_state(self, state)

    def _safe_op_set_extra_state(self, state):
        if state is not None and "EXTRA_STATE" in str(type(state)):
            return
        return _original_op_set_extra_state(self, state)

    te_base.TransformerEngineBaseModule.set_extra_state = _safe_set_extra_state
    te_ops.BasicOperation.set_extra_state = _safe_op_set_extra_state

from awomo.awomo.training.checkpoint.utils import (
    get_lm_head_weight_and_name,
    has_local_tied_lm_head,
    is_tied_word_embeddings,
    materialize_missing_tied_lm_head,
)
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)

_PREFIX = "model."


def _is_quantized_module(module: torch.nn.Module) -> bool:
    return getattr(module, "quant_state", None) is not None


def _has_quantized_params(model: torch.nn.Module) -> bool:
    return any(map(_is_quantized_module, model.modules()))


def _has_expert_parallelism(model: torch.nn.Module) -> bool:
    return any(getattr(m, "ep_size", 1) > 1 for m in model.modules())


def _is_megatron_fsdp_optimizer(optimizer: torch.optim.Optimizer) -> bool:
    try:
        from torch.distributed._tensor import DTensor
    except ImportError:
        DTensor = None

    for group in optimizer.param_groups:
        for p in group["params"]:
            if getattr(p, "_megatron_fsdp_model", None) is not None:
                return True
            if DTensor is not None and isinstance(p.data, DTensor):
                return True
    return False


def _get_peft_state_dict(model: torch.nn.Module) -> dict[str, Any]:
    state_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            name = name.replace("_checkpoint_wrapped_module.", "")
            param = param.full_tensor() if hasattr(param, "full_tensor") else param
            state_dict[name] = param.detach().cpu()
    return state_dict


def _set_peft_state_dict(model: torch.nn.Module, state_dict: dict[str, Any]) -> None:
    from torch.distributed.tensor import DTensor, Replicate

    param_dict = {name.replace("_checkpoint_wrapped_module.", ""): param for name, param in model.named_parameters()}
    loaded, skipped = 0, 0

    for name, saved_tensor in state_dict.items():
        if name not in param_dict:
            skipped += 1
            continue

        param = param_dict[name]
        if not param.requires_grad:
            skipped += 1
            continue

        if isinstance(param.data, DTensor):
            full_t = saved_tensor.to(param.data.to_local().device)
            full_dt = DTensor.from_local(
                full_t, device_mesh=param.data.device_mesh, placements=[Replicate()] * param.data.device_mesh.ndim
            )
            local_shard = full_dt.redistribute(placements=param.data.placements).to_local()
            param.data.to_local().copy_(local_shard)
        else:
            param.data.copy_(saved_tensor.to(param.data.device))
        loaded += 1

    if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        import logging

        logging.getLogger(__name__).info(f"_set_peft_state_dict: loaded {loaded} params, skipped {skipped} keys")


def _drop_outer_prefix(sd: dict[str, Any], prefix: str = _PREFIX) -> None:
    for k in list(sd.keys()):
        if k.startswith(prefix):
            sd[k[len(prefix) :]] = sd.pop(k)


def _add_outer_prefix(sd: dict[str, Any], prefix: str = _PREFIX, skip_keys: list[str] = []) -> None:
    for k in list(sd.keys()):
        if not k.startswith(prefix) and k not in skip_keys:
            sd[prefix + k] = sd.pop(k)


def _rename_dora_keys_to_hf(sd: dict[str, Any]) -> None:
    for k in list(sd.keys()):
        if k.endswith(".lora_magnitude"):
            sd[k[: -len(".lora_magnitude")] + ".lora_magnitude_vector"] = sd.pop(k)


def _rename_dora_keys_from_hf(sd: dict[str, Any]) -> None:
    for k in list(sd.keys()):
        if k.endswith(".lora_magnitude_vector.default.weight"):
            sd[k[: -len(".lora_magnitude_vector.default.weight")] + ".lora_magnitude"] = sd.pop(k)
        elif k.endswith(".lora_magnitude_vector"):
            sd[k[: -len(".lora_magnitude_vector")] + ".lora_magnitude"] = sd.pop(k)


def _get_lm_head_weight_and_name(model: torch.nn.Module) -> Optional[tuple[torch.Tensor, str]]:
    return get_lm_head_weight_and_name(model)


# modified from pytorch tutorial https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html
class ModelState:
    """Helper class for tracking model state in distributed checkpointing."""

    def __init__(
        self,
        model: torch.nn.Module | list[torch.nn.Module],
        is_peft: bool = False,
        is_init_step: bool = False,
        skip_task_head_prefixes: list[str] | None = None,
    ):
        self.model = [model] if isinstance(model, torch.nn.Module) else model
        self.uses_tied_lm_head = is_tied_word_embeddings(self.model[0])
        self.has_local_tied_lm_head = has_local_tied_lm_head(self.model[0])

        if self.uses_tied_lm_head:
            _, lm_head_param_name = _get_lm_head_weight_and_name(self.model[0])
            self.lm_head_param_name = lm_head_param_name
        self.is_peft = is_peft
        self.is_init_step = is_init_step
        self.skip_task_head_prefixes = skip_task_head_prefixes or []

    def state_dict(self) -> dict[str, Any]:
        if self.is_init_step:
            return self._get_base_model_state_dict()

        if self.is_peft and (_has_expert_parallelism(self.model[0]) or _has_quantized_params(self.model[0])):
            model_state_dict = {k: v for sd in map(_get_peft_state_dict, self.model) for k, v in sd.items()}
        else:
            options = None
            if self.is_peft:
                options = StateDictOptions(full_state_dict=True, cpu_offload=True, ignore_frozen_params=True)

            func = partial(get_model_state_dict, options=options)
            model_state_dict = {k: v for sd in map(func, self.model) for k, v in sd.items()}

        if self.is_peft:
            model_state_dict = {k: v for k, v in model_state_dict.items() if "lora_" in k}

        if self.has_local_tied_lm_head:
            model_state_dict.pop(self.lm_head_param_name, None)

        if self.is_peft:
            _add_outer_prefix(model_state_dict, "base_model.model.")
            _rename_dora_keys_to_hf(model_state_dict)

        return model_state_dict

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True) -> None:
        if self.is_init_step:
            self._set_base_model_state_dict(state_dict)
            return

        options = StateDictOptions(strict=strict)
        if self.is_peft:
            _drop_outer_prefix(state_dict, "base_model.model.")
            _rename_dora_keys_from_hf(state_dict)
            if _has_expert_parallelism(self.model[0]):
                for model_part in self.model:
                    _set_peft_state_dict(model_part, state_dict)
                return
            options = StateDictOptions(strict=False, broadcast_from_rank0=True, full_state_dict=True)

        if self.uses_tied_lm_head and not self.is_peft:
            materialize_missing_tied_lm_head(
                state_dict,
                self.model[0],
                allow_current_lm_head_fallback=True,
            )

        for model_part in self.model:
            set_model_state_dict(model_part, state_dict, options=options)

    def _get_base_model_state_dict(self) -> dict[str, Any]:
        model_state_dict = {k: v for sd in map(get_model_state_dict, self.model) for k, v in sd.items()}

        if self.has_local_tied_lm_head:
            model_state_dict.pop(self.lm_head_param_name, None)

        if self.is_peft:
            keys_to_remove = [k for k in model_state_dict.keys() if "lora" in k]
            for k in keys_to_remove:
                model_state_dict.pop(k)

        if self.skip_task_head_prefixes:
            keys_to_remove = [
                k
                for k in model_state_dict.keys()
                if any(k.startswith(prefix) for prefix in self.skip_task_head_prefixes)
            ]
            for k in keys_to_remove:
                model_state_dict.pop(k)

        return model_state_dict

    def _set_base_model_state_dict(self, state_dict: dict[str, Any]) -> None:
        func = partial(set_model_state_dict, model_state_dict=state_dict, options=StateDictOptions(strict=False))
        list(map(func, self.model))


class OptimizerState:
    """Helper class for tracking optimizer state in distributed checkpointing."""

    def __init__(
        self,
        model: torch.nn.Module | list[torch.nn.Module],
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        is_peft: bool = False,
    ):
        self.model = [model] if isinstance(model, torch.nn.Module) else model
        self.optimizer = [optimizer] if isinstance(optimizer, torch.optim.Optimizer) else optimizer
        self.scheduler = [scheduler] if isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler) else scheduler
        self.is_peft = is_peft

    def state_dict(self) -> dict[str, Any]:
        if self.is_peft and (_has_expert_parallelism(self.model[0]) or _has_quantized_params(self.model[0])):
            optimizer_state_dict = self.optimizer[0].state_dict()
        elif _is_megatron_fsdp_optimizer(self.optimizer[0]):
            optimizer_state_dict = self.optimizer[0].state_dict()
        else:
            func = partial(
                get_optimizer_state_dict,
                options=StateDictOptions(flatten_optimizer_state_dict=True),
            )
            optimizer_state_dict = {k: v for sd in map(func, self.model, self.optimizer) for k, v in sd.items()}

        state_dict = {
            "optim": optimizer_state_dict,
        }
        if self.scheduler is not None:
            state_dict["sched"] = self.scheduler[0].state_dict()

        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if self.is_peft and (_has_expert_parallelism(self.model[0]) or _has_quantized_params(self.model[0])):
            self.optimizer[0].load_state_dict(state_dict["optim"])
        elif _is_megatron_fsdp_optimizer(self.optimizer[0]):
            self.optimizer[0].load_state_dict(state_dict["optim"])
        else:
            func = partial(
                set_optimizer_state_dict,
                optim_state_dict=state_dict["optim"],
                options=StateDictOptions(flatten_optimizer_state_dict=True),
            )
            list(map(func, self.model, self.optimizer))

        if "sched" in state_dict and self.scheduler is not None:
            list(map(lambda x: x.load_state_dict(state_dict["sched"]), self.scheduler))
