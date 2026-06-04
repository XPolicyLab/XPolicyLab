# Project RoboOrchard
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

import os
from typing import List, Optional

import torch
from transformers import AutoConfig, PretrainedConfig, PreTrainedModel

from robo_orchard_lab.models.monodream.language_model.llava_llama import (
    LlavaLlamaModel,
)
from robo_orchard_lab.models.monodream.multimodal_encoder.mm_utils import (
    get_model_name_from_path,
)


def load_navigation_vlm(
    model_path: str,
    model_base: Optional[str] = None,
    devices: Optional[List[int]] = None,
    **kwargs,
) -> PreTrainedModel:
    model_name = get_model_name_from_path(model_path)
    model_path = os.path.expanduser(model_path)
    if os.path.exists(os.path.join(model_path, "model")):
        model_path = os.path.join(model_path, "model")

    if devices is not None:
        assert "max_memory" not in kwargs, (
            "`max_memory` should not be set when `devices` is set"
        )
        kwargs.update(
            max_memory={
                device: torch.cuda.get_device_properties(device).total_memory
                for device in devices
            }
        )

    return load_pretrained_model(
        model_path, model_name, model_base, device_map="cuda:0", **kwargs
    )[1]


def load_pretrained_model(
    model_path,
    device_map="auto",
    device="cuda",
    **kwargs,
):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs["device_map"] = {"": device}

    kwargs["torch_dtype"] = torch.float16
    # kwargs["torch_dtype"] = torch.bfloat16

    config = AutoConfig.from_pretrained(model_path)
    config.resume_path = model_path
    prepare_config_for_eval(config, kwargs)
    model = LlavaLlamaModel(config=config, low_cpu_mem_usage=True, **kwargs)
    tokenizer = model.tokenizer

    model.eval()
    image_processor = None

    model.resize_token_embeddings(len(tokenizer))
    vision_tower = model.get_vision_tower()
    vision_tower.to(device=device, dtype=torch.float16)
    # vision_tower.to(device=device, dtype=torch.bfloat16)
    mm_projector = model.get_mm_projector()
    mm_projector.to(device=device, dtype=torch.float16)
    # mm_projector.to(device=device, dtype=torch.bfloat16)
    image_processor = vision_tower.image_processor

    if hasattr(model.llm.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len


def prepare_config_for_eval(config: PretrainedConfig, kwargs: dict):
    try:
        # compatible with deprecated config convention
        if getattr(config, "vision_tower_cfg", None) is None:
            config.vision_tower_cfg = config.mm_vision_tower
    except AttributeError:
        raise ValueError(
            f"Invalid configuration!"
            f"Cannot find vision_tower in config:\n{config}"
        )

    config.model_dtype = kwargs.pop("torch_dtype").__str__()
