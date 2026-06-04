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

from transformers import PretrainedConfig, PreTrainedModel

from robo_orchard_lab.models.monodream.multimodal_projector.base_projector import (  # noqa: E501
    MultimodalProjector,
    MultimodalProjectorConfig,
)


def build_mm_projector(
    model_type_or_path: str, config: PretrainedConfig
) -> PreTrainedModel:
    if model_type_or_path is None:
        return None

    ## load from pretrained model
    if config.resume_path:
        assert os.path.exists(model_type_or_path), (
            f"Resume mm projector path {model_type_or_path} does not exist!"
        )
        return MultimodalProjector.from_pretrained(
            model_type_or_path, config, torch_dtype=eval(config.model_dtype)
        )
    ## build from scratch
    else:
        mm_projector_cfg = MultimodalProjectorConfig(model_type_or_path)
        mm_projector = MultimodalProjector(mm_projector_cfg, config).to(
            eval(config.model_dtype)
        )
        return mm_projector
