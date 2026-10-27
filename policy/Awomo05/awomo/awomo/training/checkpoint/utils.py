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

import os
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
from transformers.modeling_utils import _get_resolved_checkpoint_files, load_state_dict


def resolve_trust_remote_code(pretrained_model_name_or_path):
    if not pretrained_model_name_or_path:
        return False
    return not os.path.isdir(pretrained_model_name_or_path) and pretrained_model_name_or_path.startswith("nvidia/")


def is_tied_word_embeddings(model: nn.Module) -> bool:
    non_tied_lm_head_models = {
        "Qwen3OmniMoeThinkerForConditionalGeneration",
    }
    model_class_name = type(model).__name__
    for m in non_tied_lm_head_models:
        if m in model_class_name:
            return False
    config = getattr(model, "config", None)
    text_config = getattr(config, "get_text_config", lambda: None)()
    return bool(getattr(text_config, "tie_word_embeddings", getattr(config, "tie_word_embeddings", False)))


def _normalize_param_name(name: str) -> str:
    return name.replace("_orig_mod.", "")


def get_lm_head_weight_and_name(model: nn.Module) -> tuple[torch.Tensor | None, str | None]:
    for name, param in model.named_parameters(remove_duplicate=False):
        normalized_name = _normalize_param_name(name)
        if "lm_head" in normalized_name and normalized_name.endswith(".weight"):
            return param, normalized_name
    return None, None


def get_input_embeddings_weight_and_name(model: nn.Module) -> tuple[torch.Tensor | None, str | None]:
    get_input_embeddings = getattr(model, "get_input_embeddings", None)
    if callable(get_input_embeddings):
        try:
            input_embeddings = get_input_embeddings()
        except Exception:
            input_embeddings = None
        if input_embeddings is not None and hasattr(input_embeddings, "weight"):
            for name, param in model.named_parameters(remove_duplicate=False):
                if param is input_embeddings.weight:
                    return param, _normalize_param_name(name)

    candidate_suffixes = (
        "embed_tokens.weight",
        "language_model.embed_tokens.weight",
        "model.language_model.embed_tokens.weight",
    )
    for name, param in model.named_parameters(remove_duplicate=False):
        normalized_name = _normalize_param_name(name)
        if normalized_name.endswith(candidate_suffixes):
            return param, normalized_name
    return None, None


def get_tied_lm_head_source_names(model: nn.Module, lm_head_param_name: str | None = None) -> list[str]:
    candidate_source_names: list[str] = []
    tied_keys = getattr(model, "_tied_weights_keys", None)
    if isinstance(tied_keys, dict):
        for target_name, source_name in tied_keys.items():
            if not isinstance(target_name, str) or not isinstance(source_name, str):
                continue
            if (
                lm_head_param_name is None
                or target_name == lm_head_param_name
                or target_name.endswith("lm_head.weight")
            ):
                candidate_source_names.append(source_name)

    _, input_embeddings_param_name = get_input_embeddings_weight_and_name(model)
    if input_embeddings_param_name is not None:
        candidate_source_names.append(input_embeddings_param_name)

    candidate_source_names.extend(
        [
            "model.language_model.embed_tokens.weight",
            "language_model.embed_tokens.weight",
            "model.embed_tokens.weight",
            "embed_tokens.weight",
        ]
    )

    seen_source_names: set[str] = set()
    deduped_source_names: list[str] = []
    for source_name in candidate_source_names:
        if source_name in seen_source_names:
            continue
        seen_source_names.add(source_name)
        deduped_source_names.append(source_name)
    return deduped_source_names


def has_local_tied_lm_head(model: nn.Module) -> bool:
    if not is_tied_word_embeddings(model):
        return False
    lm_head_weight, _ = get_lm_head_weight_and_name(model)
    input_embeddings_weight, _ = get_input_embeddings_weight_and_name(model)
    return lm_head_weight is not None and input_embeddings_weight is not None


def materialize_missing_tied_lm_head(
    state_dict: dict[str, Any],
    model: nn.Module,
    *,
    allow_current_lm_head_fallback: bool = False,
) -> bool:
    if not is_tied_word_embeddings(model):
        return False

    lm_head_weight, lm_head_param_name = get_lm_head_weight_and_name(model)
    if lm_head_weight is None or lm_head_param_name is None or lm_head_param_name in state_dict:
        return False

    for source_name in get_tied_lm_head_source_names(model, lm_head_param_name):
        tensor = state_dict.get(source_name)
        if isinstance(tensor, torch.Tensor):
            state_dict[lm_head_param_name] = tensor.detach()
            return True

    if allow_current_lm_head_fallback:
        state_dict[lm_head_param_name] = lm_head_weight.detach()
        return True

    return False


def _get_checkpoint_tensor_dtypes(
    pretrained_model_name_or_path: str,
    hf_config: Any,
    load_kwargs: Mapping[str, object] | None = None,
) -> dict[str, torch.dtype]:
    load_kwargs = dict(load_kwargs or {})

    provided_state_dict = load_kwargs.get("state_dict")
    if isinstance(provided_state_dict, Mapping):
        return {name: tensor.dtype for name, tensor in provided_state_dict.items() if isinstance(tensor, torch.Tensor)}

    if load_kwargs.get("gguf_file") is not None:
        return {}

    trust_remote_code = load_kwargs.get(
        "trust_remote_code",
        resolve_trust_remote_code(pretrained_model_name_or_path),
    )
    download_kwargs = {
        key: load_kwargs[key]
        for key in (
            "cache_dir",
            "force_download",
            "proxies",
            "local_files_only",
            "token",
            "revision",
            "subfolder",
            "commit_hash",
        )
        if key in load_kwargs
    }
    checkpoint_files, _ = _get_resolved_checkpoint_files(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        variant=load_kwargs.get("variant"),
        gguf_file=load_kwargs.get("gguf_file"),
        use_safetensors=load_kwargs.get("use_safetensors"),
        user_agent={"file_type": "model", "framework": "pytorch"},
        is_remote_code=bool(trust_remote_code),
        transformers_explicit_filename=getattr(hf_config, "transformers_weights", None),
        download_kwargs=download_kwargs,
    )
    if not checkpoint_files:
        return {}

    checkpoint_dtypes: dict[str, torch.dtype] = {}
    weights_only = bool(load_kwargs.get("weights_only", True))
    for checkpoint_file in checkpoint_files:
        state_dict = load_state_dict(checkpoint_file, map_location="meta", weights_only=weights_only)
        checkpoint_dtypes.update(
            {name: tensor.dtype for name, tensor in state_dict.items() if isinstance(tensor, torch.Tensor)}
        )
    return checkpoint_dtypes
