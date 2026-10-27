# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Checkpoint conversion mappings for loading HuggingFace checkpoints."""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    pass


_TRANSFORMERS_AVAILABLE = False
try:
    from transformers.conversion_mapping import (
        get_checkpoint_conversion_mapping as _transformers_get_checkpoint_conversion_mapping,
    )
    from transformers.conversion_mapping import (
        get_model_conversion_mapping as _transformers_get_model_conversion_mapping,
    )
    from transformers.core_model_loading import WeightConverter, WeightRenaming

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    WeightConverter = None
    WeightRenaming = None


MODELS_REQUIRING_TENSOR_MERGING = {
    "mixtral",
    "minimax",
    "phimoe",
    "qwen2_moe",
    "qwen3_moe",
    "deepseek_v2",
    "deepseek_v3",
    "jamba",
    "olmoe",
    "lfm2_moe",
    "dots1",
    "ernie4_5_moe",
    "glm4_moe",
    "glm4v_moe",
    "longcat_flash",
    "qwen3_omni_moe",
    "qwen3_next",
    "qwen3_vl_moe",
    "hunyuan_v1_moe",
    "flex_olmo",
}


def requires_tensor_merging(model_type: str) -> bool:
    return model_type in MODELS_REQUIRING_TENSOR_MERGING


def get_checkpoint_conversion_mapping(model_type: str) -> Optional[list]:
    if not _TRANSFORMERS_AVAILABLE:
        return None
    return _transformers_get_checkpoint_conversion_mapping(model_type)


def get_model_conversion_mapping(
    model: "nn.Module",
    key_mapping: Optional[dict[str, str]] = None,
    hf_quantizer: Optional[object] = None,
    add_legacy: bool = True,
) -> list:
    if not _TRANSFORMERS_AVAILABLE:
        return []
    return _transformers_get_model_conversion_mapping(
        model,
        key_mapping=key_mapping,
        hf_quantizer=hf_quantizer,
        add_legacy=add_legacy,
    )


_VLM_KEY_MAPPINGS: dict[str, dict[str, str]] = {
    "gemma3": {
        r"^language_model\.model\.": "model.language_model.",
        r"^vision_tower\.": "model.vision_tower.",
        r"^multi_modal_projector\.": "model.multi_modal_projector.",
        r"^language_model\.lm_head\.": "lm_head.",
    },
}


def get_combined_key_mapping(
    model_type: str,
    model_key_mapping: Optional[dict[str, str]] = None,
) -> Optional[dict[str, str]]:
    if model_type in _VLM_KEY_MAPPINGS:
        return dict(_VLM_KEY_MAPPINGS[model_type])

    result = {}

    if model_key_mapping:
        result.update(model_key_mapping)

    if _TRANSFORMERS_AVAILABLE:
        conversions = get_checkpoint_conversion_mapping(model_type)
        if conversions:
            for conv in conversions:
                if WeightRenaming is not None and isinstance(conv, WeightRenaming):
                    sources = getattr(conv, "source_patterns", None)
                    targets = getattr(conv, "target_patterns", None)
                    if sources and targets:
                        if isinstance(sources, str):
                            sources = [sources]
                        if isinstance(targets, str):
                            targets = [targets]
                        for source, target in zip(sources, targets):
                            if source not in result:
                                result[source] = target

    return result if result else None


def is_transformers_conversion_available() -> bool:
    return _TRANSFORMERS_AVAILABLE
