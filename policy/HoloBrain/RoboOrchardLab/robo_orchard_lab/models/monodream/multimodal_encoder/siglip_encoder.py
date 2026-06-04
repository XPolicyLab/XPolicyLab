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

# This file was originally copied from the [VILA] repository:
# https://github.com/NVlabs/VILA
# Modifications have been made to fit the needs of this project.

from transformers import PretrainedConfig, SiglipImageProcessor

from robo_orchard_lab.models.monodream.multimodal_encoder.siglip import (
    SiglipVisionModel,
)
from robo_orchard_lab.models.monodream.multimodal_encoder.vision_encoder import (  # noqa: E501
    VisionTower,
)


class SiglipVisionTowerWrapper(VisionTower):
    def __init__(
        self, model_name_or_path: str, config: PretrainedConfig
    ) -> None:
        super().__init__(model_name_or_path, config)
        self.vision_tower = SiglipVisionModel.from_pretrained(
            model_name_or_path,
            attn_implementation="flash_attention_2",
            torch_dtype=eval(config.model_dtype),
        )
        self.image_processor = SiglipImageProcessor.from_pretrained(
            model_name_or_path
        )
        self.is_loaded = True
