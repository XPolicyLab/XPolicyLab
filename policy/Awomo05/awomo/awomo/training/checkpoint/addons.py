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

import glob
import json
import os
import shutil
from typing import TYPE_CHECKING, Protocol

import torch
from torch import nn

from awomo.awomo.training.checkpoint.stateful_wrappers import ModelState

if TYPE_CHECKING:
    pass


class CheckpointAddon(Protocol):
    """Optional hooks that run around backend IO (used for PEFT and consolidated HF metadata)."""

    def pre_save(self, **kwargs) -> None: ...

    def post_save(self, **kwargs) -> None: ...


class ConsolidatedHFAddon:
    """Addon that writes consolidated Hugging Face metadata alongside sharded weights."""

    def pre_save(self, **kwargs) -> None:
        model_state = kwargs["model_state"]
        hf_metadata_dir = kwargs["hf_metadata_dir"]
        fqn_to_file_index_mapping = kwargs["fqn_to_file_index_mapping"]
        tokenizer = kwargs.get("tokenizer", None)
        model_part = model_state.model[0]
        original_model_path = kwargs["original_model_path"]

        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            _maybe_save_custom_model_code(original_model_path, hf_metadata_dir, model_part=model_part)
            if hasattr(model_part, "config"):
                v4_compatible = kwargs.get("v4_compatible", False)
                config_name = "config.json"
                if v4_compatible and _config_exists(original_model_path, config_name):
                    _save_original_config_json(original_model_path, hf_metadata_dir, config_name)
                    config_name = "config.v5.json"

                _maybe_strip_quantization_config(model_part)
                with open(os.path.join(hf_metadata_dir, config_name), "w") as f:
                    if hasattr(model_part.config, "to_json_string"):
                        f.write(model_part.config.to_json_string(use_diff=False))
                    else:
                        json.dump(dict(model_part.config), f, indent=2, default=str)

            if getattr(model_part, "generation_config", None) is not None:
                config_name = "generation_config.json"
                if v4_compatible and _config_exists(original_model_path, config_name):
                    _save_original_config_json(original_model_path, hf_metadata_dir, config_name)
                    config_name = "generation_config.v5.json"
                with open(os.path.join(hf_metadata_dir, config_name), "w") as f:
                    f.write(model_part.generation_config.to_json_string())

            if tokenizer is not None:
                tokenizer.save_pretrained(hf_metadata_dir)

            with open(os.path.join(hf_metadata_dir, "fqn_to_file_index_mapping.json"), "w") as f:
                json.dump(fqn_to_file_index_mapping, f, indent=2, sort_keys=True)
        if torch.distributed.is_initialized():
            torch.distributed.barrier(group=kwargs.get("gloo_process_group"))

    def post_save(self, **kwargs) -> None:
        consolidated_path = kwargs["consolidated_path"]
        hf_metadata_path = kwargs["hf_metadata_path"]
        if not consolidated_path:
            return

        if (not torch.distributed.is_initialized()) or (torch.distributed.get_rank() == 0):
            for item_name in os.listdir(hf_metadata_path):
                if item_name == "fqn_to_file_index_mapping.json":
                    continue
                src_path = os.path.join(hf_metadata_path, item_name)
                dst_path = os.path.join(consolidated_path, item_name)
                shutil.move(src_path, dst_path)
            shutil.rmtree(hf_metadata_path, ignore_errors=True)
        if torch.distributed.is_initialized():
            torch.distributed.barrier(group=kwargs.get("gloo_process_group"))


class PeftAddon:
    """Addon that writes PEFT-specific metadata and tokenizer alongside adapter weights."""

    def pre_save(self, **kwargs) -> None:
        model_path = kwargs["model_path"]
        tokenizer = kwargs.get("tokenizer", None)
        model_state = kwargs["model_state"]
        peft_config = kwargs["peft_config"]
        original_model_path = kwargs["original_model_path"]
        v4_compatible = kwargs.get("v4_compatible", False)
        hf_peft_config = _get_hf_peft_config(peft_config, model_state, v4_compatible=v4_compatible)
        automodel_peft_metadata = _get_automodel_peft_metadata(peft_config)
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            model_part = model_state.model[0] if model_state is not None else None
            _maybe_save_custom_model_code(original_model_path, model_path, model_part=model_part)
            if tokenizer is not None:
                tokenizer.save_pretrained(model_path)
            with open(os.path.join(model_path, "adapter_config.json"), "w") as f:
                json.dump(hf_peft_config, f, indent=2, sort_keys=True)
            with open(os.path.join(model_path, "automodel_peft_config.json"), "w") as f:
                json.dump(automodel_peft_metadata, f, indent=2, sort_keys=True)
        if torch.distributed.is_initialized():
            torch.distributed.barrier(group=kwargs.get("gloo_process_group"))

    def post_save(self, **kwargs) -> None:
        pass


def _get_hf_peft_config(peft_config: "PeftConfig", model_state: ModelState, v4_compatible: bool = False) -> dict:
    MODEL_TYPE_TO_PEFT_TASK_TYPE = {
        "SequenceClassification": "SEQ_CLS",
        "Seq2SeqLM": "SEQ_2_SEQ_LM",
        "CausalLM": "CAUSAL_LM",
        "TokenClassification": "TOKEN_CLS",
        "QuestionAnswering": "QUESTION_ANS",
        "FeatureExtraction": "FEATURE_EXTRACTION",
    }
    model_part = model_state.model[0]
    target_modules = _extract_target_modules(model_part, v4_compatible=v4_compatible)
    target_parameters = _extract_target_parameters(model_part, v4_compatible=v4_compatible)
    try:
        arch_name = model_part.config.architectures[0]
        parts = arch_name.split("For")
        model_task = parts[-1] if len(parts) > 1 else "FeatureExtraction"
    except (AttributeError, IndexError, TypeError):
        model_task = "N/A"

    try:
        name_or_path = model_part.config.name_or_path
    except (AttributeError, TypeError):
        name_or_path = "N/A"

    try:
        task_type = MODEL_TYPE_TO_PEFT_TASK_TYPE[model_task]
    except KeyError:
        task_type = "CAUSAL_LM"

    config = {
        "task_type": task_type,
        "peft_type": "LORA",
        "r": peft_config.dim,
        "lora_alpha": peft_config.alpha,
        "use_dora": peft_config.use_dora,
        "target_modules": target_modules,
        "bias": "none",
        "base_model_name_or_path": name_or_path,
    }
    if target_parameters:
        config["target_parameters"] = target_parameters
    return config


def _get_automodel_peft_metadata(peft_config: "PeftConfig") -> dict:
    PEFT_KEYS = {"dim", "alpha"}
    result = {}
    for k, v in peft_config.to_dict().items():
        if k in PEFT_KEYS:
            continue
        if isinstance(v, torch.dtype):
            v = str(v)
        result[k] = v
    return result


def _is_qwen3_moe(model: nn.Module) -> bool:
    adapter = getattr(model, "state_dict_adapter", None)
    if adapter is None:
        return False
    raise NotImplementedError("Only FLUX checkpoints without a model-specific state_dict_adapter are supported")


def _extract_target_parameters(model: nn.Module, v4_compatible: bool = False) -> list[str]:
    if v4_compatible:
        return []
    if _is_qwen3_moe(model):
        return ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]
    return []


def _extract_target_modules(model: nn.Module, v4_compatible: bool = False) -> list[str]:
    _COMBINED_TO_SPLIT = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    _MOE_LORA_SUFFIXES = ("lora_gate_and_up_A", "lora_gate_and_up_B", "lora_down_A", "lora_down_B")

    final_target_modules = set()
    for name, _ in model.named_modules():
        if "lora" in name.lower():
            target_name = name.rsplit(".", 1)[0]
            if target_name.startswith("_orig_mod."):
                target_name = target_name[len("_orig_mod.") :]
            target_name = target_name.replace("_checkpoint_wrapped_module.", "")

            last_component = target_name.rsplit(".", 1)[-1]
            if last_component in _COMBINED_TO_SPLIT:
                parent = target_name.rsplit(".", 1)[0] if "." in target_name else ""
                for split_name in _COMBINED_TO_SPLIT[last_component]:
                    expanded = f"{parent}.{split_name}" if parent else split_name
                    final_target_modules.add(expanded)
            else:
                final_target_modules.add(target_name)

    if getattr(model, "state_dict_adapter", None) is not None:
        raise NotImplementedError("The released checkpoint exporter supports dense FLUX models only")
    return sorted(final_target_modules)


def _maybe_strip_quantization_config(model_part: nn.Module) -> None:
    config = getattr(model_part, "config", None)
    if config is None or not hasattr(config, "quantization_config"):
        return

    _QUANTIZED_DTYPES = frozenset({torch.uint8, torch.int8})
    if any(p.dtype in _QUANTIZED_DTYPES for p in model_part.parameters()):
        return

    delattr(config, "quantization_config")


def _config_exists(original_model_path: str, config_name: str) -> bool:
    if original_model_path is None or not os.path.isdir(original_model_path):
        return False
    src = os.path.join(original_model_path, config_name)
    return os.path.isfile(src)


def _save_original_config_json(original_model_path: str, hf_metadata_dir: str, config_name: str) -> None:
    src = os.path.join(original_model_path, config_name)
    if not os.path.isfile(src):
        return
    with open(src) as f:
        cfg = json.load(f)
    cfg.pop("quantization_config", None)
    dst = os.path.join(hf_metadata_dir, config_name)
    with open(dst, "w") as f:
        json.dump(cfg, f, indent=2)


def _maybe_save_custom_model_code(
    original_model_path: str | None,
    hf_metadata_dir: str,
    model_part: nn.Module | None = None,
) -> None:
    copied: set[str] = set()

    def _copy_py_tree(src_dir: str) -> None:
        for src_path in glob.glob(os.path.join(src_dir, "**", "*.py"), recursive=True):
            if os.path.basename(src_path) == "__init__.py":
                continue
            rel_path = os.path.relpath(src_path, src_dir)
            dst_path = os.path.join(hf_metadata_dir, rel_path)
            if dst_path in copied:
                continue
            os.makedirs(os.path.dirname(dst_path) or hf_metadata_dir, exist_ok=True)
            shutil.copy2(src_path, dst_path)
            copied.add(dst_path)

    if original_model_path is not None:
        if os.path.isfile(original_model_path):
            dst_path = os.path.join(hf_metadata_dir, os.path.basename(original_model_path))
            os.makedirs(hf_metadata_dir, exist_ok=True)
            shutil.copy2(original_model_path, dst_path)
            copied.add(dst_path)
        elif os.path.isdir(original_model_path):
            _copy_py_tree(original_model_path)

    if model_part is not None and not copied:
        custom_dirs: set[str] = set()
        for cls in _iter_custom_code_classes(model_part):
            try:
                import inspect

                src_file = inspect.getfile(cls)
            except (TypeError, OSError):
                continue
            module_name = getattr(cls, "__module__", "") or ""
            if not module_name.startswith("transformers_modules."):
                continue
            custom_dirs.add(os.path.dirname(src_file))
        for src_dir in custom_dirs:
            _copy_py_tree(src_dir)


def _iter_custom_code_classes(model_part: nn.Module):
    seen: set[type] = set()
    custom_pkg = ""
    for base in type(model_part).__mro__:
        mod = getattr(base, "__module__", "") or ""
        if mod.startswith("transformers_modules."):
            if base not in seen:
                seen.add(base)
                yield base
            if not custom_pkg:
                custom_pkg = ".".join(mod.split(".")[:-1])

    config = getattr(model_part, "config", None)
    auto_map = getattr(config, "auto_map", None) if config is not None else None
    if not isinstance(auto_map, dict) or not custom_pkg:
        return
    import importlib

    for value in auto_map.values():
        candidates = value if isinstance(value, (list, tuple)) else [value]
        for ref in candidates:
            if not isinstance(ref, str) or "." not in ref:
                continue
            module_path, class_name = ref.rsplit(".", 1)
            full_module = f"{custom_pkg}.{module_path}"
            try:
                mod = importlib.import_module(full_module)
                target = getattr(mod, class_name, None)
            except Exception:
                continue
            if isinstance(target, type) and target not in seen:
                seen.add(target)
                yield target
