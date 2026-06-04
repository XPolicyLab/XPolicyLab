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

from transformers import AutoConfig, PretrainedConfig, PreTrainedModel

from robo_orchard_lab.models.monodream.multimodal_encoder.siglip_encoder import (  # noqa: E501
    SiglipVisionTowerWrapper,
)


def build_vision_tower(
    model_name_or_path: str, config: PretrainedConfig
) -> PreTrainedModel:
    ## skip vision tower instantiation
    if model_name_or_path is None:
        return None

    vision_tower_arch = None
    if config.resume_path:
        assert os.path.exists(model_name_or_path), (
            f"Resume vision tower path {model_name_or_path} does not exist!"
        )
        vision_tower_cfg = AutoConfig.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        vision_tower_arch = vision_tower_cfg.architectures[0].lower()
    vision_tower_name = (
        vision_tower_arch
        if vision_tower_arch is not None
        else model_name_or_path
    )

    use_s2 = getattr(config, "s2", False)
    use_dynamic_s2 = getattr(config, "dynamic_s2", False)

    if "siglip" in vision_tower_name:
        vision_tower = SiglipVisionTowerWrapper(model_name_or_path, config)
    else:
        raise ValueError(f"Unknown vision tower: {model_name_or_path}")

    config.mm_hidden_size = (
        vision_tower.config.hidden_size
        if not (use_s2 or use_dynamic_s2)
        else vision_tower.hidden_size
    )
    return vision_tower
