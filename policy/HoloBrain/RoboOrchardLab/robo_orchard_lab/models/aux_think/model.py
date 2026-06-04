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
from transformers import PreTrainedModel

from robo_orchard_lab.models.mixin import (
    ClassType_co,
    ModelMixin,
    TorchModuleCfg,
)

__all__ = ["AuxThink", "AuxThinkConfig"]


def load_navigation_vlm(
    model_path: str,
    model_base: Optional[str] = None,
    devices: Optional[List[int]] = None,
    **kwargs,
) -> PreTrainedModel:
    from llava.conversation import (
        auto_set_conversation_mode,
    )
    from llava.mm_utils import (
        get_model_name_from_path,
    )
    from llava.model.builder import (
        load_pretrained_model,
    )

    auto_set_conversation_mode(model_path)

    model_name = get_model_name_from_path(model_path)
    model_path = os.path.expanduser(model_path)
    if os.path.exists(os.path.join(model_path, "model")):
        model_path = os.path.join(model_path, "model")

    # Set `max_memory` to constrain which GPUs to use
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

    return load_pretrained_model(model_path, model_name, model_base, **kwargs)[
        1
    ]


class AuxThink(ModelMixin):
    cfg: "AuxThinkConfig"  # for type hint

    def __init__(self, cfg: "AuxThinkConfig"):
        super().__init__(cfg)
        self.cfg = cfg
        self.model = None

    def load_from_llava_model(self, model_path):
        self.model = load_navigation_vlm(model_path)

    def generate_content(self, prompt):
        return self.model.generate_content(prompt)

    def forward(self, inputs, is_training: bool = False):
        if is_training:
            return self.model(inputs)
        else:
            return self.model.generate_content(inputs)

    def save_pretrained(self, directory: str):
        self.model.save_pretrained(directory)

    @classmethod
    def load_model(cls, directory: str):
        cfg = AuxThinkConfig()
        instance = cls(cfg)
        llm_path = os.path.join(directory, "llm")

        if os.path.isdir(llm_path):
            instance.load_from_llava_model(directory)
        else:
            instance = ModelMixin.load_model(directory)

        return instance


class AuxThinkConfig(TorchModuleCfg[AuxThink]):
    class_type: ClassType_co[AuxThink] = AuxThink
