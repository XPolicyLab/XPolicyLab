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
from typing import Any, Dict, Optional

from robo_orchard_core.utils.config import (
    load_config_class,
)
from transformers import (
    PretrainedConfig,
    Qwen2ForCausalLM,
    SiglipImageProcessor,
)
from transformers.models.siglip.configuration_siglip import SiglipVisionConfig

from robo_orchard_lab.models.mixin import (
    ClassType_co,
    ModelMixin,
    TorchModuleCfg,
)
from robo_orchard_lab.models.monodream.builder import load_navigation_vlm
from robo_orchard_lab.models.monodream.language_model import (
    LlavaLlamaConfig,
    LlavaLlamaModel,
    init_tokenizer,
)
from robo_orchard_lab.models.monodream.multimodal_encoder import (
    BasicImageEncoder,
)
from robo_orchard_lab.models.monodream.multimodal_encoder.vision_encoder import (  # noqa: E501
    SiglipVisionTower,
)
from robo_orchard_lab.models.monodream.multimodal_projector import (
    MultimodalProjector,
    MultimodalProjectorConfig,
)

__all__ = ["MonoDream", "MonoDreamConfig"]


class MonoDream(ModelMixin):
    cfg: "MonoDreamConfig"  # for type hint

    def __init__(self, cfg: "MonoDreamConfig"):
        super().__init__(cfg)
        self.cfg = cfg

        self.model_config = LlavaLlamaConfig(**cfg.llava_llama_config)
        self.model = LlavaLlamaModel(
            config=self.model_config, preload=False
        ).cuda()

        # Initialize multimodal config
        self.projector_config = MultimodalProjectorConfig(
            **cfg.projector_config
        )
        self.vision_config = SiglipVisionConfig(**cfg.vision_config)
        self.llm_cfg = PretrainedConfig(**cfg.llm_config)

        # Initialize multimodal components
        self.model.mm_projector = MultimodalProjector(
            self.projector_config, self.model_config
        ).cuda()
        self.model.vision_tower = SiglipVisionTower(
            self.vision_config, self.model_config
        ).cuda()
        self.model.llm = Qwen2ForCausalLM._from_config(self.llm_cfg).cuda()

        self.model.encoders = {}
        self.model.encoders["image"] = BasicImageEncoder(
            parent=self.model
        ).cuda()

    def init_components(self, directory: str):
        if directory.startswith("hf://"):
            directory = directory.replace("hf://", "")
        self.model.vision_tower.image_processor = (
            SiglipImageProcessor.from_pretrained(directory)
        )
        self.model.tokenizer = init_tokenizer(directory, self.model_config)

        if self.cfg.model_dtype == "torch.float16":
            self.model = self.model.half()

    def generate_content(self, prompt: list):
        return self.model.generate_content(prompt)

    def forward(self, inputs, is_training: bool = False):
        if is_training:
            return self.model(inputs)
        else:
            return self.model.generate_content(inputs)

    def save_pretrained(self, directory: str):
        self.model.save_pretrained(directory)

    def load_from_decrete_model(self, model_path: str):
        self.model = load_navigation_vlm(model_path)

    @classmethod
    def load_model(cls, directory: str, use_decrete: bool = False):
        if use_decrete:
            config_file = os.path.join(directory, "model.config.json")
            with open(config_file, "r") as f:
                cfg: MonoDreamConfig = load_config_class(f.read())
            instance = cls(cfg)
            instance.load_from_decrete_model(directory)
        else:
            instance = ModelMixin.load_model(directory)

        return instance


class MonoDreamConfig(TorchModuleCfg[MonoDream]):
    class_type: ClassType_co[MonoDream] = MonoDream

    model_dtype: str = "torch.float16"

    llava_llama_config: Optional[Dict[str, Any]] = None
    projector_config: Optional[Dict[str, Any]] = None
    vision_config: Optional[Dict[str, Any]] = None
    llm_config: Optional[Dict[str, Any]] = None
