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

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import fsspec
from PIL import Image as PILImage

from robo_orchard_lab.processing.io_processor.base import (
    ClassType_co,
    ModelIOProcessor,
    ModelIOProcessorCfg,
)
from robo_orchard_lab.utils.build import DelayInitDictType, build

__all__ = ["ProgressModelProcessor", "ProgressModelProcessorCfg"]


@dataclass
class ProgressModelInput:
    """Data structure for inputs to the ProgressModel navigation model.

    Each input contains:
        - a list of image file paths
    """

    image_paths: List[str]
    """List of image file paths representing the observation sequence."""


@dataclass
class ProgressModelOutput:
    """Output structure for the ProgressModel model."""

    text: str
    """Model-generated text output."""


class PathListToData:
    """Convert a list of image paths into input format."""

    def __init__(self, load_image: bool = True):
        self.load_image = load_image

    def __call__(self, data: ProgressModelInput) -> dict:
        input_data = {}
        if self.load_image:
            # Load all images as PIL.Image or Image
            images = []
            for p in data.image_paths:
                if isinstance(p, PILImage.Image):
                    img = p
                else:
                    with fsspec.open(p, "rb") as f:
                        img = PILImage.open(f).convert("RGB")
                images.append(img)
        else:
            images = data.image_paths

        input_data["images"] = images
        return input_data


class ProgressModelProcessor(ModelIOProcessor):
    """Processor for the ProgressModel navigation model.

    Converts image paths into a multimodal list input format,
    and extracts the generated text from model outputs.
    """

    cfg: "ProgressModelProcessorCfg"

    def __init__(self, cfg: "ProgressModelProcessorCfg"):
        super().__init__(cfg)
        self.pathlist_to_data = PathListToData(load_image=self.cfg.load_image)
        self.transforms = (
            [build(transform) for transform in self.cfg.transforms]
            if self.cfg.transforms is not None
            else []
        )
        self.prompt_template_list = [
            (
                "Assume you are a robot designed for navigation. "
                "You are provided with captured image sequences: "
            ),
            (
                ". Based on this image sequence, please describe "
                "the navigation trajectory of the robot."
            ),
        ]

    def pre_process(self, data: Union[ProgressModelInput, Dict]):
        """Convert ProgressModelInput to a model-ready format."""
        if isinstance(data, ProgressModelInput):
            data = self.pathlist_to_data(data)

        for ts_i in self.transforms:
            data = ts_i(data)

        prompt = (
            self.prompt_template_list[0]
            + "".join(["<image>"] * len(data["images"]))
            + self.prompt_template_list[1]
        )

        input_list = [prompt]
        input_list.extend(data["images"])
        return input_list

    def post_process(self, model_outputs, _) -> ProgressModelOutput:
        """Extract model-generated text output."""
        text = model_outputs
        return ProgressModelOutput(text=text)


class ProgressModelProcessorCfg(ModelIOProcessorCfg[ProgressModelProcessor]):
    """Configuration for ProgressModelProcessor."""

    class_type: ClassType_co[ProgressModelProcessor] = ProgressModelProcessor
    load_image: bool = True
    transforms: Optional[List[DelayInitDictType]] = None
