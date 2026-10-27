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
import glob
import logging
import os
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.distributed.checkpoint as dcp
import yaml

try:
    from huggingface_hub.constants import HF_HUB_CACHE
except ImportError:
    HF_HUB_CACHE = None

from awomo.awomo.training.checkpoint._backports.consolidate_hf_safetensors import (
    consolidate_safetensors_files_on_every_rank,
)
from awomo.awomo.training.checkpoint._backports.filesystem import FileSystemReader, SerializationFormat
from awomo.awomo.training.checkpoint._backports.hf_storage import (
    _HuggingFaceStorageReader,
    _HuggingFaceStorageWriter,
    _maybe_rename_index_for_diffusers,
    get_fqn_to_file_index_mapping,
)
from awomo.awomo.training.checkpoint.addons import ConsolidatedHFAddon, PeftAddon
from awomo.awomo.training.checkpoint.conversion_mapping import (
    get_combined_key_mapping,
    requires_tensor_merging,
)
from awomo.awomo.training.checkpoint.stateful_wrappers import ModelState, OptimizerState
from awomo.awomo.training.checkpoint.utils import (
    get_tied_lm_head_source_names,
    is_tied_word_embeddings,
    materialize_missing_tied_lm_head,
)
from packaging.version import parse
from safetensors.torch import load_file, save_file
from torch import nn
from torch.distributed.checkpoint._nested_dict import flatten_state_dict, set_element
from torch.distributed.checkpoint.state_dict import _init_optim_state
from torch.distributed.device_mesh import DeviceMesh

if TYPE_CHECKING:
    pass
    pass


def _ckpt_debug(event: str, **fields: Any) -> None:
    if not os.environ.get("CKPT_DEBUG"):
        return
    rank = "na"
    world_size = "na"
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    payload = {
        "event": event,
        "rank": rank,
        "world_size": world_size,
        "local_rank": os.environ.get("LOCAL_RANK", "na"),
        "pid": os.getpid(),
        "host": socket.gethostname(),
    }
    payload.update(fields)
    msg = " ".join(f"{key}={value}" for key, value in payload.items())
    print(f"[CKPT TRACE] {msg}", flush=True, file=sys.stderr)


def _is_geq_torch_2_9() -> bool:
    return parse(torch.__version__).base_version >= "2.9.0"


def _is_safetensors_checkpoint(path: str) -> bool:
    if os.path.isfile(path):
        return path.endswith(".safetensors")
    if not os.path.isdir(path):
        return False
    if os.path.isfile(os.path.join(path, "model.safetensors.index.json")):
        return True
    return len(glob.glob(os.path.join(path, "*.safetensors"))) > 0


def _is_bin_checkpoint(path: str) -> bool:
    if os.path.isfile(path):
        return path.endswith(".bin")
    if not os.path.isdir(path):
        return False
    if os.path.isfile(os.path.join(path, "pytorch_model.bin.index.json")):
        return True
    if os.path.isfile(os.path.join(path, "pytorch_model.bin")):
        return True
    return len(glob.glob(os.path.join(path, "*.bin"))) > 0


def _summarize_state_dict_key_diff(
    expected_keys: set[str],
    loaded_keys: set[str],
    *,
    limit: int = 10,
) -> dict[str, Any]:
    missing = sorted(expected_keys - loaded_keys)
    unexpected = sorted(loaded_keys - expected_keys)
    return {
        "missing_count": len(missing),
        "unexpected_count": len(unexpected),
        "missing_examples": missing[:limit],
        "unexpected_examples": unexpected[:limit],
    }


def _get_checkpoint_metadata_keys(
    path: str,
    storage_reader: Optional[_HuggingFaceStorageReader] = None,
) -> set[str]:
    reader = storage_reader if storage_reader is not None else FileSystemReader(path)
    metadata = reader.read_metadata()
    return set(metadata.state_dict_metadata.keys())


if _is_geq_torch_2_9():
    from torch.distributed.checkpoint.staging import DefaultStager
    from torch.distributed.checkpoint.state_dict_saver import AsyncCheckpointerType


@dataclass
class _AsyncSaveContext:
    """Internal container for async checkpointing state."""

    stager: Any | None
    process_group: Any | None
    future: Any | None
    staging_active: bool = False


@dataclass
class CheckpointingConfig:
    """Configuration for checkpointing."""

    enabled: bool
    checkpoint_dir: str | Path
    model_save_format: str
    model_cache_dir: str | Path
    model_repo_id: str
    save_consolidated: bool
    is_peft: bool
    model_state_dict_keys: list[str] = (
        None
    )
    is_async: bool = False
    dequantize_base_checkpoint: bool | None = None
    original_model_root_dir: str | None = None
    skip_task_head_prefixes_for_base_model: list[str] | None = (
        None
    )
    single_rank_consolidation: bool = False
    staging_dir: str | None = None
    v4_compatible: bool = False
    diffusers_compatible: bool = False
    best_metric_key: str = "default"

    def __post_init__(self):
        formats = [v.value for v in SerializationFormat]
        assert (
            self.model_save_format in formats
        ), f"Unsupported model save format: {self.model_save_format}. Supported formats: {formats}"
        self.model_save_format = SerializationFormat[self.model_save_format.upper()]
        if self.save_consolidated or False:
            if not self.v4_compatible:
                logging.warning(
                    "save_consolidated=True but v4_compatible=False; "
                    "checkpoint assets may be not compatible with transformers v4; "
                    "[experimental] set --checkpoint.v4_compatible=True to enable"
                )
            else:
                logging.warning("[experimental] v4_compatible=True enables transformers v4 compatibility")

        if self.is_async and not _is_geq_torch_2_9():
            logging.error("Async mode is only supported for torch >= 2.9.0, disabling async mode")
            self.is_async = False


class Checkpointer:
    """High-level checkpoint manager built on torch.distributed.checkpoint (DCP)."""

    def __init__(
        self,
        config: CheckpointingConfig,
        dp_rank: int,
        tp_rank: int,
        pp_rank: int,
        moe_mesh: Optional[DeviceMesh] = None,
        model_process_group: Optional[Any] = None,
        optim_process_group: Optional[Any] = None,
        create_process_groups: bool = True,
    ) -> None:
        self.config = config
        self.moe_mesh = moe_mesh
        self.dp_rank = dp_rank
        self.tp_rank = tp_rank
        self.pp_rank = pp_rank

        self._model_ctx = _AsyncSaveContext(
            stager=None, process_group=model_process_group, future=None, staging_active=False
        )
        self._optim_ctx = _AsyncSaveContext(
            stager=None, process_group=optim_process_group, future=None, staging_active=False
        )
        if create_process_groups and torch.distributed.is_initialized():
            self._model_ctx.process_group = torch.distributed.new_group(backend="gloo")
            self._optim_ctx.process_group = torch.distributed.new_group(backend="gloo")
        if self.config.is_async:
            self._model_ctx.stager = DefaultStager()
            self._optim_ctx.stager = DefaultStager()

        self._addons = []
        if False:
            self._addons.append(ConsolidatedHFAddon())
        if self.config.is_peft:
            self._addons.append(PeftAddon())

    @torch.no_grad()
    def save_model(
        self,
        model: nn.Module,
        weights_path: str,
        peft_config: Optional["PeftConfig"] = None,
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
    ) -> None:
        save_start = time.monotonic()
        _ckpt_debug(
            "save_model.enter",
            weights_path=weights_path,
            is_peft=self.config.is_peft,
            is_async=self.config.is_async,
            model_save_format=self.config.model_save_format,
            save_consolidated=self.config.save_consolidated,
            single_rank_consolidation=self.config.single_rank_consolidation,
        )
        model_dir = os.path.join(weights_path, "model")
        consolidated_dir = (
            os.path.join(model_dir, "consolidated") if self._should_write_consolidated_safetensors() else None
        )
        hf_metadata_dir = os.path.join(model_dir, ".hf_metadata") if self._should_write_hf_metadata() else None
        _ckpt_debug(
            "save_model.ensure_dirs.start",
            model_dir=model_dir,
            consolidated_dir=consolidated_dir,
            hf_metadata_dir=hf_metadata_dir,
        )
        _ensure_dirs(model_dir, consolidated_dir, hf_metadata_dir)
        _ckpt_debug("save_model.ensure_dirs.done")

        consolidate_on_all_ranks = (
            self._should_write_consolidated_safetensors()
            and not self.config.is_async
            and not self.config.single_rank_consolidation
        )
        _ckpt_debug(
            "save_model.consolidation_flags",
            should_write_hf_metadata=self._should_write_hf_metadata(),
            should_write_consolidated=self._should_write_consolidated_safetensors(),
            consolidate_on_all_ranks=consolidate_on_all_ranks,
            addon_count=len(self._addons),
        )

        state_dict_start = time.monotonic()
        _ckpt_debug("save_model.state_dict.start")
        model_state = ModelState(model, self.config.is_peft)
        state_dict = model_state.state_dict()
        _ckpt_debug(
            "save_model.state_dict.done",
            seconds=f"{time.monotonic() - state_dict_start:.3f}",
            key_count=len(state_dict),
            sample_keys="|".join(list(state_dict.keys())[:3]),
        )

        adapt_start = time.monotonic()
        _ckpt_debug(
            "save_model.adapt_to_hf.start",
            model_part_type=type(model_state.model[0]).__name__,
            has_state_dict_adapter=hasattr(model_state.model[0], "state_dict_adapter"),
        )
        state_dict = _maybe_adapt_state_dict_to_hf(
            model_state.model[0],
            state_dict,
            quantization=False,
            device_mesh=self.moe_mesh,
            v4_compatible=self.config.v4_compatible,
        )
        _ckpt_debug(
            "save_model.adapt_to_hf.done",
            seconds=f"{time.monotonic() - adapt_start:.3f}",
            key_count=len(state_dict),
        )
        index_start = time.monotonic()
        _ckpt_debug("save_model.index.start")
        fqn_to_file_index_mapping = self._maybe_build_consolidated_index(model_state, state_dict)
        _ckpt_debug(
            "save_model.index.done",
            seconds=f"{time.monotonic() - index_start:.3f}",
            mapping_count=len(fqn_to_file_index_mapping) if fqn_to_file_index_mapping else 0,
        )

        for addon in self._addons:
            addon_start = time.monotonic()
            _ckpt_debug("save_model.pre_save.start", addon=type(addon).__name__)
            addon.pre_save(
                model_state=model_state,
                model_path=model_dir,
                consolidated_path=consolidated_dir,
                hf_metadata_dir=hf_metadata_dir,
                tokenizer=tokenizer,
                peft_config=peft_config,
                fqn_to_file_index_mapping=fqn_to_file_index_mapping,
                original_model_path=self._get_original_model_path(model_state),
                v4_compatible=self.config.v4_compatible,
                gloo_process_group=self._model_ctx.process_group,
            )
            _ckpt_debug(
                "save_model.pre_save.done",
                addon=type(addon).__name__,
                seconds=f"{time.monotonic() - addon_start:.3f}",
            )

        storage_start = time.monotonic()
        _ckpt_debug("save_model.storage_writer.start")
        storage_writer = self._get_storage_writer(
            consolidated_dir, fqn_to_file_index_mapping, model_dir, consolidate_on_all_ranks
        )
        _ckpt_debug(
            "save_model.storage_writer.done",
            seconds=f"{time.monotonic() - storage_start:.3f}",
            storage_writer_type=type(storage_writer).__name__ if storage_writer else "None",
        )
        do_save_start = time.monotonic()
        _ckpt_debug("save_model.do_save.start", model_dir=model_dir)
        self._model_ctx.future = self._do_save(state_dict, model_dir, storage_writer)
        _ckpt_debug(
            "save_model.do_save.done",
            seconds=f"{time.monotonic() - do_save_start:.3f}",
            future_type=type(self._model_ctx.future).__name__ if self._model_ctx.future else "None",
        )

        for addon in self._addons:
            addon_start = time.monotonic()
            _ckpt_debug("save_model.post_save.start", addon=type(addon).__name__)
            addon.post_save(
                consolidated_path=consolidated_dir,
                hf_metadata_path=hf_metadata_dir,
                gloo_process_group=self._model_ctx.process_group,
            )
            _ckpt_debug(
                "save_model.post_save.done",
                addon=type(addon).__name__,
                seconds=f"{time.monotonic() - addon_start:.3f}",
            )

        if consolidate_on_all_ranks:
            consolidate_start = time.monotonic()
            _ckpt_debug("save_model.consolidate.start", input_dir=model_dir, output_dir=consolidated_dir)
            consolidate_safetensors_files_on_every_rank(
                input_dir=model_dir,
                output_dir=consolidated_dir,
                fqn_to_index_mapping=fqn_to_file_index_mapping,
                num_threads=5,
                use_staging=self.config.staging_dir is not None,
                staging_dir=self.config.staging_dir,
            )
            _ckpt_debug(
                "save_model.consolidate.done",
                seconds=f"{time.monotonic() - consolidate_start:.3f}",
            )
            if self.config.diffusers_compatible:
                if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                    _maybe_rename_index_for_diffusers(consolidated_dir)
        _ckpt_debug("save_model.exit", seconds=f"{time.monotonic() - save_start:.3f}")

    @torch.no_grad()
    def save_optimizer(
        self, optimizer: torch.optim.Optimizer, model: nn.Module, weights_path: str, scheduler: Optional[Any] = None
    ) -> None:
        optimizer_path = os.path.join(weights_path, "optim")
        _ensure_dirs(optimizer_path)
        optimizer_state = OptimizerState(model, optimizer, scheduler, is_peft=self.config.is_peft)
        state_dict = optimizer_state.state_dict()
        self._optim_ctx.future = self._do_save(state_dict, optimizer_path)

    def load_optimizer(
        self, optimizer: torch.optim.Optimizer, model: nn.Module, weights_path: str, scheduler: Optional[Any] = None
    ) -> None:
        optimizer_state = OptimizerState(model, optimizer, scheduler, is_peft=self.config.is_peft)
        _init_optim_state(optimizer)
        state_dict = optimizer_state.state_dict()
        optimizer_path = os.path.join(weights_path, "optim")
        checkpoint_metadata = FileSystemReader(optimizer_path).read_metadata()
        flat_state_dict, flat_mappings = flatten_state_dict(state_dict)

        def ignore_optimizer_key(key: str) -> bool:
            return scheduler is None and key.startswith("optim.param_groups.") and key.endswith(".initial_lr")

        ignored_target_values = {key: flat_state_dict[key] for key in flat_state_dict if ignore_optimizer_key(key)}
        for key in ignored_target_values:
            path = flat_mappings[key]
            container = state_dict
            for path_element in path[:-1]:
                container = container[path_element]
            del container[path[-1]]

        optimizer_keys = {key for key in flat_state_dict if key.startswith("optim.") and not ignore_optimizer_key(key)}
        checkpoint_optimizer_keys = {
            key
            for key in checkpoint_metadata.state_dict_metadata
            if key.startswith("optim.") and not ignore_optimizer_key(key)
        }
        optimizer_state_keys = {key for key in optimizer_keys if key.startswith("optim.state.")}
        checkpoint_optimizer_state_keys = {key for key in checkpoint_optimizer_keys if key.startswith("optim.state.")}
        optimizer_metadata_keys = optimizer_keys - optimizer_state_keys
        checkpoint_optimizer_metadata_keys = checkpoint_optimizer_keys - checkpoint_optimizer_state_keys
        missing_keys = optimizer_metadata_keys - checkpoint_optimizer_metadata_keys
        unexpected_keys = (checkpoint_optimizer_metadata_keys - optimizer_metadata_keys) | (
            checkpoint_optimizer_state_keys - optimizer_state_keys
        )
        if missing_keys or unexpected_keys:
            raise ValueError(
                "optimizer checkpoint key mismatch: "
                f"missing={sorted(missing_keys)[:10]} ({len(missing_keys)} total), "
                f"extra={sorted(unexpected_keys)[:10]} ({len(unexpected_keys)} total)"
            )

        optimizer_state_groups: dict[str, set[str]] = {}
        checkpoint_optimizer_state_groups: dict[str, set[str]] = {}
        for key in optimizer_state_keys:
            optimizer_state_groups.setdefault(key.rsplit(".", maxsplit=1)[0], set()).add(key)
        for key in checkpoint_optimizer_state_keys:
            checkpoint_optimizer_state_groups.setdefault(key.rsplit(".", maxsplit=1)[0], set()).add(key)
        incomplete_state_groups = {
            prefix: optimizer_state_groups[prefix] - checkpoint_keys
            for prefix, checkpoint_keys in checkpoint_optimizer_state_groups.items()
            if checkpoint_keys != optimizer_state_groups[prefix]
        }
        if incomplete_state_groups:
            missing_state_keys = set().union(*incomplete_state_groups.values())
            raise ValueError(
                "optimizer checkpoint has incomplete parameter state: "
                f"missing={sorted(missing_state_keys)[:10]} ({len(missing_state_keys)} total)"
            )

        missing_target_state_values = {
            key: flat_state_dict[key] for key in optimizer_state_keys - checkpoint_optimizer_state_keys
        }
        for key in missing_target_state_values:
            path = flat_mappings[key]
            container = state_dict
            for path_element in path[:-1]:
                container = container[path_element]
            del container[path[-1]]

        for key in checkpoint_optimizer_keys:
            tensor = flat_state_dict[key]
            if not isinstance(tensor, torch.Tensor):
                continue
            checkpoint_dtype = checkpoint_metadata.state_dict_metadata[key].properties.dtype
            if checkpoint_dtype != tensor.dtype:
                raise ValueError(
                    f"optimizer checkpoint dtype mismatch for {key}: "
                    f"checkpoint={checkpoint_dtype}, optimizer={tensor.dtype}"
                )
        self._do_load(state_dict, optimizer_path)
        for key, value in missing_target_state_values.items():
            set_element(state_dict, flat_mappings[key], value)
        for key, value in ignored_target_values.items():
            set_element(state_dict, flat_mappings[key], value)
        optimizer_state.load_state_dict(state_dict)

        checkpoint_state_prefixes = {key.rsplit(".", maxsplit=1)[0] for key in checkpoint_optimizer_state_keys}
        parameters_with_checkpoint_state = {
            parameter
            for name, parameter in model.named_parameters()
            if f"optim.state.{name.replace('_checkpoint_wrapped_module.', '')}" in checkpoint_state_prefixes
        }
        native_optimizer_state = optimizer.state_dict()
        for native_group, parameter_group in zip(
            native_optimizer_state["param_groups"],
            optimizer.param_groups,
            strict=True,
        ):
            for parameter_id, parameter in zip(
                native_group["params"],
                parameter_group["params"],
                strict=True,
            ):
                if f"optim.state.{parameter_id}" in checkpoint_state_prefixes:
                    parameters_with_checkpoint_state.add(parameter)
        for _, parameter in model.named_parameters():
            if parameter not in parameters_with_checkpoint_state:
                optimizer.state.pop(parameter, None)

    @torch.no_grad()
    def load_model(
        self,
        model: nn.Module,
        model_path: str,
        is_init_step: bool = False,
        use_checkpoint_id: bool = True,
        key_mapping: Optional[dict[str, str]] = None,
    ) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path {model_path} does not exist")
        model_state = ModelState(
            model,
            is_peft=self.config.is_peft,
            is_init_step=is_init_step,
            skip_task_head_prefixes=getattr(self.config, "skip_task_head_prefixes_for_base_model", None),
        )

        model_type = getattr(getattr(model_state.model[0], "config", None), "model_type", None)
        has_state_dict_adapter = hasattr(model_state.model[0], "state_dict_adapter")

        if is_init_step and model_type and requires_tensor_merging(model_type) and not has_state_dict_adapter:
            converted_state_dict = _convert_checkpoint_with_transformers(model_state.model[0], model_path, key_mapping)
            if converted_state_dict:
                materialized_tied_lm_head = materialize_missing_tied_lm_head(
                    converted_state_dict,
                    model_state.model[0],
                    allow_current_lm_head_fallback=False,
                )
                if materialized_tied_lm_head:
                    logging.info(
                        "Materialized missing tied lm_head.weight from embedding weights for %s during init load.",
                        type(model_state.model[0]).__name__,
                    )
                _load_full_state_dict_into_model(model_state.model, converted_state_dict)
                return

        is_safetensors = _is_safetensors_checkpoint(model_path)
        if (
            is_init_step
            and len(model_state.model) == 1
            and (_is_bin_checkpoint(model_path) or (is_safetensors and not _is_custom_model(model_state.model[0])))
        ):
            t0 = time.monotonic()
            weights_only = not _is_remote_code_model(model_state.model[0])
            state_dict_from_disk = _load_hf_checkpoint_preserving_dtype(model_path, weights_only=weights_only)
            t_disk = time.monotonic()
            if state_dict_from_disk is not None:
                state_dict_from_disk = _maybe_adapt_state_dict_from_hf(
                    model_state.model[0], state_dict_from_disk, moe_mesh=self.moe_mesh
                )
            else:
                state_dict_from_disk = {}

            if key_mapping and state_dict_from_disk:
                state_dict_from_disk = _apply_key_mapping(state_dict_from_disk, key_mapping)

            materialized_tied_lm_head = materialize_missing_tied_lm_head(
                state_dict_from_disk,
                model_state.model[0],
                allow_current_lm_head_fallback=False,
            )
            if materialized_tied_lm_head:
                logging.info(
                    "Materialized missing tied lm_head.weight from embedding weights for %s during init load.",
                    type(model_state.model[0]).__name__,
                )

            total_bytes = sum(
                t.nelement() * t.element_size() for t in state_dict_from_disk.values() if isinstance(t, torch.Tensor)
            )
            _load_full_state_dict_into_model(model_state.model, state_dict_from_disk)
            t_end = time.monotonic()

            disk_s = t_disk - t0
            dist_s = t_end - t_disk
            total_s = t_end - t0
            gb = total_bytes / (1 << 30)
            logging.info(
                f"load_model: {gb:.2f} GB loaded in {total_s:.2f}s "
                f"({gb / total_s:.2f} GB/s overall | "
                f"disk read {disk_s:.2f}s, distribute {dist_s:.2f}s)"
            )
            del state_dict_from_disk
            gc.collect()
            return

        state_dict = model_state.state_dict()
        expected_keys = set(state_dict.keys())
        reader_key_mapping = None if has_state_dict_adapter else key_mapping
        storage_reader = self._get_storage_reader(model_path, reader_key_mapping, is_init_step=is_init_step)

        state_dict = _maybe_adapt_state_dict_to_hf(
            model_state.model[0],
            state_dict,
            quantization=self.config.dequantize_base_checkpoint,
            device_mesh=self.moe_mesh,
        )

        compat_tied_lm_head_source_key: str | None = None
        lm_head_param_name = getattr(model_state, "lm_head_param_name", None)
        should_try_tied_lm_head_compat = (
            getattr(model_state, "uses_tied_lm_head", False)
            and not getattr(model_state, "has_local_tied_lm_head", False)
            and isinstance(lm_head_param_name, str)
            and lm_head_param_name in state_dict
        )
        if should_try_tied_lm_head_compat:
            checkpoint_metadata_keys = _get_checkpoint_metadata_keys(model_path, storage_reader)
            if lm_head_param_name not in checkpoint_metadata_keys:
                for source_name in get_tied_lm_head_source_names(model_state.model[0], lm_head_param_name):
                    if source_name not in checkpoint_metadata_keys or source_name in state_dict:
                        continue
                    compat_tied_lm_head_source_key = source_name
                    state_dict[source_name] = state_dict.pop(lm_head_param_name)
                    logging.warning(
                        "Checkpoint %s is missing %s. Loading tied source %s into lm_head "
                        "(HF tied-embedding checkpoints omit lm_head, and pre-fix DCP "
                        "checkpoints with PP also omit it).",
                        model_path,
                        lm_head_param_name,
                        source_name,
                    )
                    break
                if compat_tied_lm_head_source_key is None:
                    logging.warning(
                        "Checkpoint %s is missing %s and no tied source key was found. "
                        "Keeping the current lm_head initialization for compatibility.",
                        model_path,
                        lm_head_param_name,
                    )
                    state_dict.pop(lm_head_param_name, None)

        state_dict = self._do_load(state_dict, model_path, storage_reader, is_init_step=is_init_step)

        if compat_tied_lm_head_source_key is not None and isinstance(lm_head_param_name, str):
            state_dict[lm_head_param_name] = state_dict.pop(compat_tied_lm_head_source_key)

        state_dict = _maybe_adapt_state_dict_from_hf(model_state.model[0], state_dict, moe_mesh=self.moe_mesh)
        key_diff = _summarize_state_dict_key_diff(expected_keys, set(state_dict.keys()))
        if key_diff["missing_count"] or key_diff["unexpected_count"]:
            logging.warning(
                "Checkpoint key mismatch for %s: missing=%d unexpected=%d "
                "(missing examples=%s, unexpected examples=%s)",
                type(model_state.model[0]).__name__,
                key_diff["missing_count"],
                key_diff["unexpected_count"],
                key_diff["missing_examples"],
                key_diff["unexpected_examples"],
            )
        model_state.load_state_dict(state_dict, strict=not (len(model_state.model) > 1 or has_state_dict_adapter))

        del state_dict
        gc.collect()

    @staticmethod
    def initialize_model_weights(
        model: torch.nn.Module, device: torch.device, peft_init_method: str | None = None
    ) -> None:
        has_meta_params = any(p.device.type == "meta" for p in model.parameters())
        if has_meta_params:
            to_empty_parameters_only(model, device=device)

        for module in model.modules():
            for key in list(module._buffers):
                buf = module._buffers[key]
                if buf is not None and buf.device.type == "meta":
                    module._buffers[key] = torch.empty_like(buf, device=device)


        try:
            model_class = model.config.architectures[0]
        except Exception:
            model_class = ""
        is_nemotron_v2 = model_class == "NemotronHForCausalLM" and not getattr(model.config, "n_routed_experts", None)
        is_nemotron_v3_hf = (
            model_class == "NemotronHForCausalLM"
            and getattr(model.config, "n_routed_experts", None)
            and hasattr(model, "backbone")
        )
        has_padding_idx = any(
            isinstance(mod, nn.Embedding)
            and type(mod.weight).__name__ == "DTensor"
            and getattr(mod, "padding_idx", None) is not None
            for mod in model.modules()
        )
        owns_weight_load = bool(getattr(model, "_skip_init_weights_on_load", False))
        skip_initialize_weights = (
            model_class
            in [
                "Gemma3ForConditionalGeneration",
                "Gemma3ForCausalLM",
            ]
            or is_nemotron_v2
            or is_nemotron_v3_hf
            or has_padding_idx
            or owns_weight_load
        )
        if not skip_initialize_weights:
            for _, module in model.named_modules():
                if hasattr(module, "_is_hf_initialized"):
                    module._is_hf_initialized = False

            if hasattr(model, "initialize_weights"):
                model.initialize_weights()
            else:
                logging.warning(
                    "Warning: Model does not have initialize_weights method."
                    " Requires custom initialization to be implemented."
                )

        if peft_init_method is not None:
            _init_peft_adapters(model, peft_init_method)

    def load_base_model(
        self,
        model: torch.nn.Module,
        device: torch.device,
        root_dir: str,
        model_name: str | None,
        load_base_model: bool = True,
    ) -> None:
        model_type = getattr(getattr(model, "config", None), "model_type", None)

        if load_base_model:
            assert model_name is not None, "model_name is required when loading base model"
            model_key_mapping = getattr(model, "_checkpoint_conversion_mapping", None)
            key_mapping = get_combined_key_mapping(model_type, model_key_mapping)
            if model_type == "nemotron_h" and hasattr(model, "backbone"):
                key_mapping = None
            self.load_model(
                model,
                model_path=(
                    model_name if os.path.exists(model_name) else get_safetensors_index_path(root_dir, model_name)
                ),
                is_init_step=True,
                key_mapping=key_mapping,
            )

        _reinit_non_persistent_buffers(model, device, model_type=model_type)

        is_tied_lm_head = is_tied_word_embeddings(model)
        self.config.original_model_root_dir = root_dir
        if hasattr(model, "tie_weights") and is_tied_lm_head:
            try:
                model.tie_weights()
            except AttributeError:
                pass

    def maybe_wait_for_staging(self) -> None:
        if self._model_ctx.staging_active and self._model_ctx.future is not None:
            self._model_ctx.future.staging_completion.result()
            self._model_ctx.staging_active = False
        if self._optim_ctx.staging_active and self._optim_ctx.future is not None:
            self._optim_ctx.future.staging_completion.result()
            self._optim_ctx.staging_active = False

    def async_wait(self) -> None:
        if self._model_ctx.future is not None:
            self._model_ctx.future.upload_completion.result()
            self._model_ctx.future = None
        if self._optim_ctx.future is not None:
            self._optim_ctx.future.upload_completion.result()
            self._optim_ctx.future = None

    def save_on_dp_ranks(self, state: Any, state_name: str, path: str) -> None:
        import shutil

        _ckpt_debug(
            "save_on_dp_ranks.enter",
            state_name=state_name,
            path=path,
            dp_rank=self.dp_rank,
            tp_rank=self.tp_rank,
            pp_rank=self.pp_rank,
        )
        state_dir = os.path.join(path, state_name)
        _ensure_dirs(state_dir)
        _ckpt_debug("save_on_dp_ranks.ensure_dirs.done", state_dir=state_dir)
        if self.tp_rank == 0 and self.pp_rank == 0:
            filename = f"{state_name}_dp_rank_{self.dp_rank}.pt"
            local_tmp = f"/tmp/ckpt_tmp_{state_name}_{self.dp_rank}"
            os.makedirs(local_tmp, exist_ok=True)
            local_file = os.path.join(local_tmp, filename)
            _ckpt_debug("save_on_dp_ranks.local_write.start", local_file=local_file)
            torch.save(state.state_dict(), local_file)
            _ckpt_debug(
                "save_on_dp_ranks.local_write.done",
                local_file=local_file,
                local_size=os.path.getsize(local_file),
            )
            shared_file = os.path.join(state_dir, filename)
            _ckpt_debug("save_on_dp_ranks.copy_to_shared.start", shared_file=shared_file)
            shutil.copy2(local_file, shared_file)
            _ckpt_debug(
                "save_on_dp_ranks.copy_to_shared.done",
                shared_file=shared_file,
                shared_size=os.path.getsize(shared_file),
            )
            os.remove(local_file)
            os.rmdir(local_tmp)
        else:
            _ckpt_debug("save_on_dp_ranks.skip_write_non_owner")
        _ckpt_debug("save_on_dp_ranks.exit", state_name=state_name)

    def load_on_dp_ranks(self, state: Any, state_name: str, path: str) -> None:
        state_dir = os.path.join(path, state_name)
        state.load_state_dict(
            torch.load(os.path.join(state_dir, f"{state_name}_dp_rank_{self.dp_rank}.pt"), weights_only=False)
        )

    def save_distributed_state(self, state: Any, state_name: str, path: str) -> None:
        state_dir = os.path.join(path, state_name)
        _ensure_dirs(state_dir)
        state_dict = state.state_dict()
        planner = dcp.DefaultSavePlanner(enable_plan_caching=True)
        process_group = self._model_ctx.process_group
        dcp.save(state_dict, checkpoint_id=state_dir, planner=planner, process_group=process_group)

    def load_distributed_state(self, state: Any, state_name: str, path: str) -> None:
        state_dir = os.path.join(path, state_name)
        state_dict = state.state_dict()
        process_group = self._model_ctx.process_group
        dcp.load(state_dict, checkpoint_id=state_dir, process_group=process_group)
        state.load_state_dict(state_dict)

    def close(self) -> None:
        self.maybe_wait_for_staging()
        self.async_wait()
        if self._model_ctx.stager is not None:
            self._model_ctx.stager.close()
        if self._optim_ctx.stager is not None:
            self._optim_ctx.stager.close()

    def _do_load(
        self,
        state_dict: dict[str, torch.Tensor],
        path: str,
        storage_reader: Optional[_HuggingFaceStorageReader] = None,
        is_init_step: bool = False,
    ) -> dict[str, torch.Tensor]:
        is_model = True if "/model" in path else False
        if self.config.is_peft and is_model and (not is_init_step):
            state_dict = load_file(os.path.join(path, "adapter_model.safetensors"))
        else:
            dcp.load(state_dict, checkpoint_id=path, storage_reader=storage_reader)
        return state_dict

    def _do_save(
        self, state_dict: dict[str, torch.Tensor], path: str, storage_writer: Optional[_HuggingFaceStorageWriter] = None
    ) -> Optional["AsyncSaveResponse"]:
        is_model = True if "/model" in path else False
        if self.config.is_peft and is_model:
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                save_file(state_dict, os.path.join(path, "adapter_model.safetensors"))
            return

        _ckpt_debug(
            "do_save.enter",
            path=path,
            is_model=is_model,
            is_async=self.config.is_async,
            is_peft=self.config.is_peft,
            storage_writer_type=type(storage_writer).__name__ if storage_writer else "None",
            state_key_count=len(state_dict),
            sample_keys="|".join(list(state_dict.keys())[:3]),
        )
        ret = None
        _ckpt_debug("do_save.planner.start")
        if storage_writer is None:
            enable_plan_caching = True
        else:
            enable_plan_caching = False
        planner = dcp.DefaultSavePlanner(enable_plan_caching=enable_plan_caching)
        _ckpt_debug("do_save.planner.done", planner_type=type(planner).__name__)
        if self.config.is_async:
            ctx = self._model_ctx if is_model else self._optim_ctx
            _ckpt_debug(
                "do_save.async_save.start",
                process_group_set=ctx.process_group is not None,
                stager_type=type(ctx.stager).__name__ if ctx.stager else "None",
            )
            ret = dcp.async_save(
                state_dict,
                checkpoint_id=path,
                storage_writer=storage_writer,
                process_group=ctx.process_group,
                async_stager=ctx.stager,
                async_checkpointer_type=AsyncCheckpointerType.PROCESS,
                planner=planner,
            )
            ctx.staging_active = True
            _ckpt_debug("do_save.async_save.submitted", response_type=type(ret).__name__ if ret else "None")
        else:
            ctx = self._model_ctx if is_model else self._optim_ctx
            _ckpt_debug(
                "do_save.dcp_save.start",
                path=path,
                process_group_set=ctx.process_group is not None,
                storage_writer_type=type(storage_writer).__name__ if storage_writer else "None",
            )
            dcp.save(
                state_dict,
                checkpoint_id=path,
                storage_writer=storage_writer,
                planner=planner,
                process_group=ctx.process_group,
            )
            _ckpt_debug("do_save.dcp_save.done", path=path)
        _ckpt_debug("do_save.exit", path=path)
        return ret

    def _should_write_consolidated_safetensors(self) -> bool:
        return self.config.save_consolidated and self._should_write_hf_metadata()

    def _should_write_hf_metadata(self) -> bool:
        return self.config.model_save_format == SerializationFormat.SAFETENSORS and not self.config.is_peft

    def _maybe_build_consolidated_index(
        self, model_state: ModelState, state_dict: dict[str, torch.Tensor]
    ) -> Optional[dict[str, int]]:
        if not self._should_write_hf_metadata():
            return None
        model = model_state.model[0]
        index_path = get_safetensors_index_path(
            self.config.model_cache_dir,
            self.config.model_repo_id,
        )
        if index_path:
            fqn_to_file_index_mapping = get_fqn_to_file_index_mapping(
                index_path, getattr(model, "_checkpoint_conversion_mapping", None)
            )
            model_part = model_state.model[0]
            config = getattr(model_part, "config", None)
            model_type = getattr(config, "model_type", None)
            pre_shard_hf_state_dict_keys = (
                getattr(model, "_pre_shard_hf_state_dict_keys", None) or self.config.model_state_dict_keys
            )
            if pre_shard_hf_state_dict_keys is None:
                pre_shard_hf_state_dict_keys = list(state_dict.keys())
            if model_type and requires_tensor_merging(model_type) and not hasattr(model_part, "state_dict_adapter"):
                num_shards = max(fqn_to_file_index_mapping.values()) if fqn_to_file_index_mapping else 1
                fqn_to_file_index_mapping = _equally_divide_layers(num_shards, pre_shard_hf_state_dict_keys)
            else:
                keys_to_remove = list(set(fqn_to_file_index_mapping.keys()) - set(pre_shard_hf_state_dict_keys))
                if getattr(model_state, "has_local_tied_lm_head", False):
                    keys_to_remove.append(model_state.lm_head_param_name)
                for key in keys_to_remove:
                    fqn_to_file_index_mapping.pop(key, None)
        else:
            pre_shard_hf_state_dict_keys = (
                getattr(model, "_pre_shard_hf_state_dict_keys", None) or self.config.model_state_dict_keys
            )
            if pre_shard_hf_state_dict_keys:
                fqn_to_file_index_mapping = {k: 1 for k in pre_shard_hf_state_dict_keys}
            else:
                fqn_to_file_index_mapping = {k: 1 for k in state_dict.keys()}

        default_index = max(fqn_to_file_index_mapping.values()) if fqn_to_file_index_mapping else 1

        for fqn in list(state_dict.keys()):
            fqn_to_file_index_mapping[fqn] = fqn_to_file_index_mapping.get(fqn, default_index)
        return fqn_to_file_index_mapping

    def _get_storage_writer(
        self,
        consolidated_output_path: Optional[str],
        fqn_to_index_mapping: Optional[dict[str, int]],
        model_path: str,
        consolidate_on_all_ranks: bool = False,
    ) -> Optional[_HuggingFaceStorageWriter]:
        if self.config.model_save_format == SerializationFormat.SAFETENSORS:
            return _HuggingFaceStorageWriter(
                path=model_path,
                save_sharded=True,
                consolidated_output_path=consolidated_output_path if not consolidate_on_all_ranks else None,
                fqn_to_index_mapping=fqn_to_index_mapping,
                staging_dir=self.config.staging_dir,
                diffusers_compatible=self.config.diffusers_compatible,
            )

    def _get_storage_reader(
        self, model_path: str, key_mapping: Optional[dict[str, str]], is_init_step: bool = False
    ) -> Optional[_HuggingFaceStorageReader]:
        if self.config.model_save_format == SerializationFormat.SAFETENSORS or is_init_step:
            if os.path.isfile(os.path.join(model_path, ".metadata")):
                return None

            consolidated_path = os.path.join(model_path, "consolidated")
            reader_path = consolidated_path if os.path.isdir(consolidated_path) else model_path

            if key_mapping is None and not is_init_step:
                try:
                    from torch.distributed.checkpoint.hf_storage import HuggingFaceStorageReader as _UpstreamHFReader

                    return _UpstreamHFReader(path=model_path)
                except ImportError:
                    pass
            return _HuggingFaceStorageReader(path=reader_path, key_mapping=key_mapping)

    def _get_original_model_path(self, model_state: ModelState) -> str | None:
        if not hasattr(model_state.model[0], "name_or_path") and not hasattr(
            getattr(model_state.model[0], "config", None), "name_or_path"
        ):
            return None

        pretrained_model_name_or_path = getattr(model_state.model[0], "name_or_path", None) or getattr(
            getattr(model_state.model[0], "config", None), "name_or_path", None
        )
        if not pretrained_model_name_or_path:
            return None

        if os.path.isdir(pretrained_model_name_or_path):
            return pretrained_model_name_or_path

        cache_dir = getattr(self.config, "original_model_root_dir", None) or HF_HUB_CACHE
        return get_safetensors_index_path(cache_dir, pretrained_model_name_or_path)


def get_safetensors_index_path(cache_dir: str | Path | None, repo_id: str | None) -> str | None:
    if repo_id is None:
        return None

    if os.path.exists(repo_id):
        return repo_id

    cache_dir = cache_dir or HF_HUB_CACHE
    if cache_dir is None:
        raise ValueError("Hugging Face cache directory is not set (cache_dir=None).")
    repo_dir = f"models--{repo_id.replace('/', '--')}"
    snapshots_root = Path(cache_dir) / repo_dir / "snapshots"

    pattern = snapshots_root / "*" / "model.safetensors.index.json"
    matches = glob.glob(str(pattern))
    if matches:
        return str(Path(matches[0]).parent)

    snapshot_dirs = [p for p in glob.glob(str(snapshots_root / "*")) if Path(p).is_dir()]
    if snapshot_dirs:
        try:
            return snapshot_dirs[0]
        except IndexError:
            raise FileNotFoundError(f"No snapshot directories found in {snapshots_root}")


def to_empty_parameters_only(
    model: nn.Module, *, device: torch.device, recurse: bool = True, dtype: torch.dtype | None = None
) -> nn.Module:
    return _apply(model, lambda t: torch.empty_like(t, device=device, dtype=dtype), recurse=recurse)


def save_config(config: dict[str, Any], weights_path: str) -> None:
    with open(os.path.join(weights_path, "config.yaml"), "w") as f:
        yaml.dump(config, f, sort_keys=False, default_flow_style=False)


def _ensure_dirs(*dirs: Optional[str]) -> None:
    for d in dirs:
        if d:
            os.makedirs(d, exist_ok=True)


def _init_peft_adapters(model: nn.Module, peft_init_method: str) -> None:
    for module in model.modules():
        if hasattr(module, "init_lora_weights"):
            try:
                module.init_lora_weights(peft_init_method)
            except Exception as e:
                logging.warning(f"Failed to initialize weights for PEFT adapter `{module.__class__.__name__}`: {e}")


_MODELS_REQUIRING_BUFFER_REINIT: frozenset[str] = frozenset(
    {
        "gemma3",
        "nemotron-nas",
    }
)


def _reinit_non_persistent_buffers(model: nn.Module, device: torch.device, model_type: str | None = None) -> None:
    if model_type not in _MODELS_REQUIRING_BUFFER_REINIT:
        return

    for name, module in model.named_modules():
        if hasattr(module, "rope_init_fn") and hasattr(module, "inv_freq") and hasattr(module, "rope_kwargs"):
            try:
                inv_freq, _ = module.rope_init_fn(module.config, device, **module.rope_kwargs)
                module.inv_freq = inv_freq
                if hasattr(module, "original_inv_freq"):
                    module.original_inv_freq = inv_freq.clone()
                logging.debug(f"Reinitialized RoPE inv_freq for {name} on device {device}")
            except Exception as e:
                logging.warning(f"Failed to reinitialize RoPE inv_freq for {name}: {e}")

        elif hasattr(module, "layer_types") and hasattr(module, "rope_type") and hasattr(module, "config"):
            rope_config = getattr(module, "config", None)
            rope_parameters = getattr(rope_config, "rope_parameters", None)
            if rope_parameters is None:
                continue
            for layer_type in getattr(module, "layer_types", []):
                inv_freq_attr = f"{layer_type}_inv_freq"
                if not hasattr(module, inv_freq_attr):
                    continue
                try:
                    rope_init_fn = getattr(module, "compute_default_rope_parameters", None)
                    if rope_init_fn is None:
                        continue
                    rope_type = module.rope_type.get(layer_type, "default")
                    if rope_type != "default":
                        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

                        rope_init_fn = ROPE_INIT_FUNCTIONS[rope_type]
                    curr_inv_freq, curr_attention_scaling = rope_init_fn(rope_config, device, layer_type=layer_type)
                    setattr(module, inv_freq_attr, curr_inv_freq)
                    orig_attr = f"{layer_type}_original_inv_freq"
                    if hasattr(module, orig_attr):
                        setattr(module, orig_attr, curr_inv_freq.clone())
                    setattr(module, f"{layer_type}_attention_scaling", curr_attention_scaling)
                    logging.debug(f"Reinitialized RoPE {inv_freq_attr} for {name} on device {device}")
                except Exception as e:
                    logging.warning(f"Failed to reinitialize RoPE {inv_freq_attr} for {name}: {e}")

        if hasattr(module, "scalar_embed_scale") and "embed_scale" in getattr(module, "_buffers", {}):
            try:
                module.embed_scale = torch.tensor(module.scalar_embed_scale, device=device)
                logging.debug(f"Reinitialized embed_scale={module.scalar_embed_scale} for {name} on device {device}")
            except Exception as e:
                logging.warning(f"Failed to reinitialize embed_scale for {name}: {e}")

        if hasattr(module, "num_positions") and "position_ids" in getattr(module, "_buffers", {}):
            try:
                module.position_ids = torch.arange(module.num_positions, device=device).expand((1, -1))
                logging.debug(f"Reinitialized position_ids (num_positions={module.num_positions}) for {name}")
            except Exception as e:
                logging.warning(f"Failed to reinitialize position_ids for {name}: {e}")


def _apply(module, fn, recurse=True) -> nn.Module:
    from torch.utils._python_dispatch import is_traceable_wrapper_subclass

    if recurse:
        for child in module.children():
            _apply(child, fn, recurse=recurse)

    def compute_should_use_set_data(tensor, tensor_applied):
        if torch._has_compatible_shallow_copy_type(tensor, tensor_applied):
            return not torch.__future__.get_overwrite_module_params_on_conversion()
        else:
            return False

    should_use_swap_tensors = torch.__future__.get_swap_module_params_on_conversion()
    for key, param in module._parameters.items():
        if param is None:
            continue
        with torch.no_grad():
            param_applied = fn(param)
        p_should_use_set_data = compute_should_use_set_data(param, param_applied)

        p_should_use_swap_tensors = should_use_swap_tensors or is_traceable_wrapper_subclass(param_applied)

        param_grad = param.grad
        if p_should_use_swap_tensors:
            try:
                if param_grad is not None:
                    param.grad = None
                param_applied = torch.nn.Parameter(param_applied, requires_grad=param.requires_grad)
                torch.utils.swap_tensors(param, param_applied)
            except Exception as e:
                if param_grad is not None:
                    param.grad = param_grad
                raise RuntimeError(f"_apply(): Couldn't swap {module._get_name()}.{key}") from e
            out_param = param
        elif p_should_use_set_data:
            param.data = param_applied
            out_param = param
        else:
            assert isinstance(param, torch.nn.Parameter)
            assert param.is_leaf
            out_param = torch.nn.Parameter(param_applied, param.requires_grad)
            module._parameters[key] = out_param

        if param_grad is not None:
            with torch.no_grad():
                grad_applied = fn(param_grad)
            g_should_use_set_data = compute_should_use_set_data(param_grad, grad_applied)
            if p_should_use_swap_tensors:
                grad_applied.requires_grad_(param_grad.requires_grad)
                try:
                    torch.utils.swap_tensors(param_grad, grad_applied)
                except Exception as e:
                    raise RuntimeError(f"_apply(): Couldn't swap {module._get_name()}.{key}.grad") from e
                out_param.grad = param_grad
            elif g_should_use_set_data:
                assert out_param.grad is not None
                out_param.grad.data = grad_applied
            else:
                assert param_grad.is_leaf
                out_param.grad = grad_applied.requires_grad_(param_grad.requires_grad)

    return module


def _apply_key_mapping(
    state_dict: dict[str, torch.Tensor],
    key_mapping: dict[str, str],
) -> dict[str, torch.Tensor]:
    from awomo.awomo.training.checkpoint._backports.hf_storage import (
        _get_key_renaming_mapping,
    )

    return {_get_key_renaming_mapping(k, key_mapping): v for k, v in state_dict.items()}


def _load_full_state_dict_into_model(
    model_parts: list[nn.Module],
    state_dict: dict[str, torch.Tensor],
) -> None:
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        _CHECKPOINT_PREFIX,
    )
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

    for model in model_parts:
        for name, module in model.named_modules():
            if type(module).get_extra_state is not nn.Module.get_extra_state:
                key = f"{name}._extra_state" if name else "_extra_state"
                key = key.replace(_CHECKPOINT_PREFIX, "")
                if key not in state_dict:
                    state_dict[key] = torch.tensor([], dtype=torch.uint8)

    options = StateDictOptions(
        strict=False,
        full_state_dict=True,
    )

    for part in model_parts:
        set_model_state_dict(part, model_state_dict=state_dict, options=options)


def _convert_checkpoint_with_transformers(
    model: nn.Module,
    model_path: str,
    key_mapping: Optional[dict[str, str]] = None,
) -> Optional[dict[str, torch.Tensor]]:
    try:
        from copy import deepcopy

        from safetensors import safe_open
        from transformers.conversion_mapping import get_model_conversion_mapping
        from transformers.core_model_loading import (
            WeightConverter,
            WeightRenaming,
            dot_natural_key,
            rename_source_key,
        )
    except ImportError:
        logging.warning(
            "transformers library with conversion_mapping not available. "
            "Cannot use transformers' WeightConverter for tensor merging."
        )
        return None

    try:
        weight_mapping = get_model_conversion_mapping(model, key_mapping=key_mapping, add_legacy=True)
        if not weight_mapping:
            logging.warning(
                f"No conversion mapping found for model type {getattr(model.config, 'model_type', 'unknown')}"
            )
            return None

        safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))
        if not safetensors_files:
            logging.warning(f"No safetensors files found in {model_path}")
            return None

        checkpoint_state_dict = {}
        for sf_path in safetensors_files:
            with safe_open(sf_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    checkpoint_state_dict[key] = f.get_tensor(key)

        renamings = [entry for entry in weight_mapping if isinstance(entry, WeightRenaming)]
        converters = [entry for entry in weight_mapping if isinstance(entry, WeightConverter)]
        pattern_to_converter = {k: converter for converter in converters for k in converter.source_patterns}

        converted_state_dict = {}
        param_name_to_mapping: dict[str, WeightRenaming | WeightConverter] = {}

        sorted_items = sorted(checkpoint_state_dict.items(), key=lambda kv: dot_natural_key(kv[0]))

        n_converter_keys = 0
        n_rename_keys = 0
        for original_key, tensor in sorted_items:
            renamed_key, source_pattern = rename_source_key(original_key, renamings, converters)

            if source_pattern is not None:
                n_converter_keys += 1
                new_converter = deepcopy(pattern_to_converter[source_pattern])
                mapping = param_name_to_mapping.setdefault(renamed_key, new_converter)
                mapping.add_tensor(renamed_key, original_key, source_pattern, tensor)
            else:
                n_rename_keys += 1
                mapping = param_name_to_mapping.setdefault(renamed_key, WeightRenaming(original_key, renamed_key))
                mapping.add_tensor(renamed_key, original_key, original_key, tensor)

        logging.debug(
            "[convert_ckpt] {} keys matched converters, {} keys simple rename, {} total mappings".format(
                n_converter_keys, n_rename_keys, len(param_name_to_mapping)
            )
        )

        for first_param_name, mapping in param_name_to_mapping.items():
            result = mapping.convert(first_param_name, model=model, config=model.config)
            if isinstance(result, tuple):
                realized_value = result[0]
            elif isinstance(result, dict):
                realized_value = result
            else:
                raise TypeError(
                    "Expected convert() to return dict or (dict, errors) tuple, got {}".format(type(result))
                )
            for target_name, param in realized_value.items():
                param = param[0] if isinstance(param, list) else param
                converted_state_dict[target_name] = param
            if callable(getattr(mapping, "reset", None)):
                mapping.reset()
        logging.debug("Converted {} keys using transformers conversion mapping".format(len(converted_state_dict)))
        return converted_state_dict

    except Exception as e:
        logging.warning("Failed to convert checkpoint with transformers: {}".format(e))
        return None


def _maybe_adapt_state_dict_to_hf(
    model_part: nn.Module, state_dict: dict[str, torch.Tensor], quantization: bool = False, **kwargs
) -> dict[str, torch.Tensor]:
    adapter = getattr(model_part, "state_dict_adapter", None)
    if adapter:
        return adapter.to_hf(state_dict, exclude_key_regex=r".*_extra_state.*", quantization=quantization, **kwargs)
    return state_dict


def _equally_divide_layers(num_shards: int, keys: list[str]) -> dict[str, int]:
    if num_shards <= 0:
        raise ValueError(f"num_shards must be > 0, got {num_shards}")

    num_layers = len(keys)
    if num_layers == 0:
        return {}

    layers_per_shard, remainder = divmod(num_layers, num_shards)
    fqn_to_index_mapping: dict[str, int] = {}
    start = 0
    for shard_index in range(1, num_shards + 1):
        extra = 1 if shard_index <= remainder else 0
        end = start + layers_per_shard + extra
        for key in keys[start:end]:
            fqn_to_index_mapping[key] = shard_index
        start = end
    return fqn_to_index_mapping


def _model_has_dtensors(module: nn.Module) -> bool:
    return any(type(p).__name__ == "DTensor" for p in module.parameters())


def _is_custom_model(module: nn.Module) -> bool:
    _MIXIN_MODULE = "awomo.awomo.training.model_utils."
    return any(
        (c.__module__ or "").startswith("awomo.components.models.")
        and not (c.__module__ or "").startswith(_MIXIN_MODULE)
        for c in type(module).__mro__
    )


def _is_remote_code_model(module: nn.Module) -> bool:
    return any("transformers_modules" in (c.__module__ or "") for c in type(module).__mro__)


def _load_hf_checkpoint_preserving_dtype(
    model_path: str, weights_only: bool = True
) -> Optional[dict[str, torch.Tensor]]:

    if _is_bin_checkpoint(model_path):
        return _load_hf_bin_checkpoint(model_path, weights_only=weights_only)
    elif _is_safetensors_checkpoint(model_path):
        return _load_hf_safetensors_checkpoint(model_path)
    return None


def _load_hf_safetensors_checkpoint(model_path: str) -> Optional[dict[str, torch.Tensor]]:
    from safetensors import safe_open

    out: dict[str, torch.Tensor] = {}
    if os.path.isfile(model_path):
        return dict(load_file(model_path))
    index_file = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.isfile(index_file):
        import json

        with open(index_file) as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        for key, filename in weight_map.items():
            sf_path = os.path.join(model_path, filename)
            if not os.path.isfile(sf_path):
                continue
            with safe_open(sf_path, framework="pt", device="cpu") as f:
                if key in f.keys():
                    out[key] = f.get_tensor(key)
    else:
        for sf_path in glob.glob(os.path.join(model_path, "*.safetensors")):
            with safe_open(sf_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    out[key] = f.get_tensor(key)
    return out if out else None


def _load_hf_bin_checkpoint(model_path: str, weights_only: bool = True) -> Optional[dict[str, torch.Tensor]]:
    if not _is_bin_checkpoint(model_path):
        return None

    load_kwargs = dict(map_location="cpu", weights_only=weights_only)

    if os.path.isfile(model_path):
        return torch.load(model_path, **load_kwargs)

    index_file = os.path.join(model_path, "pytorch_model.bin.index.json")
    if os.path.isfile(index_file):
        import json

        with open(index_file) as f:
            index = json.load(f)
        weight_map: dict[str, str] = index.get("weight_map", {})
        out: dict[str, torch.Tensor] = {}
        loaded_files: set[str] = set()
        for key, filename in weight_map.items():
            if filename in loaded_files:
                continue
            bin_path = os.path.join(model_path, filename)
            if not os.path.isfile(bin_path):
                continue
            shard = torch.load(bin_path, **load_kwargs)
            out.update(shard)
            loaded_files.add(filename)
        return out if out else None

    single = os.path.join(model_path, "pytorch_model.bin")
    if os.path.isfile(single):
        return torch.load(single, **load_kwargs)

    out = {}
    for bin_path in sorted(glob.glob(os.path.join(model_path, "*.bin"))):
        shard = torch.load(bin_path, **load_kwargs)
        out.update(shard)
    return out if out else None


def _maybe_adapt_state_dict_from_hf(
    model_part: nn.Module, state_dict: dict[str, torch.Tensor], moe_mesh: Optional[DeviceMesh] = None
) -> dict[str, torch.Tensor]:
    adapter = getattr(model_part, "state_dict_adapter", None)
    if adapter:
        ep_mesh_dims = [dim for dim in moe_mesh.mesh_dim_names if dim != "pp"] if moe_mesh is not None else []
        ep_mesh = moe_mesh[tuple(ep_mesh_dims)] if ep_mesh_dims else moe_mesh
        return adapter.from_hf(state_dict, device_mesh=ep_mesh)
    return state_dict
