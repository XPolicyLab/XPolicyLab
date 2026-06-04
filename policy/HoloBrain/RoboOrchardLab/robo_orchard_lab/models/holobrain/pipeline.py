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

from typing import Any

from robo_orchard_core.utils.config import (
    ClassType_co,
    ConfigInstanceOf,
)

from robo_orchard_lab.models.holobrain.processor import (
    HoloBrainProcessor,
    HoloBrainProcessorCfg,
    MultiArmManipulationInput,
    MultiArmManipulationOutput,
)
from robo_orchard_lab.models.mixin import TorchModelMixin
from robo_orchard_lab.pipeline.inference.basic import (
    InferencePipeline,
    InferencePipelineCfg,
)


class HoloBrainInferencePipeline(InferencePipeline):
    cfg: "HoloBrainInferencePipelineCfg"
    processor: HoloBrainProcessor

    def __init__(
        self,
        cfg: "HoloBrainInferencePipelineCfg",
        model: TorchModelMixin | None = None,
    ):
        super().__init__(cfg=cfg, model=model)

    def __call__(
        self, data: MultiArmManipulationInput
    ) -> MultiArmManipulationOutput:
        return super().__call__(data)

    def _model_forward(self, data: dict) -> Any:
        return super()._model_forward(data)


class HoloBrainInferencePipelineCfg(InferencePipelineCfg):
    class_type: ClassType_co[HoloBrainInferencePipeline] = (
        HoloBrainInferencePipeline
    )
    """Class type for the pipeline."""
    processor: ConfigInstanceOf[HoloBrainProcessorCfg] | None = None
    """Processor configuration for the pipeline."""
