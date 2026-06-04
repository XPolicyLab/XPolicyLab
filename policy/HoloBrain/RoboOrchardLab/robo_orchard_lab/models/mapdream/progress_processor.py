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

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Union

import fsspec
from PIL import Image as PILImage

from robo_orchard_lab.inference.processor import (
    ClassType_co,
    ProcessorMixin,
    ProcessorMixinCfg,
)

__all__ = [
    "ProgressModelInput",
    "ProgressModelOutput",
    "ProgressModelProcessor",
    "ProgressModelProcessorCfg",
]


class ImageListToData:
    """Convert image paths and instructions into the input format."""

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
        input_data["instruction"] = data.instruction
        return input_data


@dataclass
class ProgressModelInput:
    """Inputs to the Janus-based ProgressModel.

    `image_paths` elements can be either file paths or base64 strings.
    """

    image_paths: List[str]
    instruction: str


@dataclass
class ProgressModelOutput:
    """Output structure for the ProgressModel model."""

    image: PILImage.Image


class ProgressModelProcessor(ProcessorMixin):
    """Prepare inputs to match janus_inference()'s `construct_input` usage."""

    cfg: "ProgressModelProcessorCfg"

    def __init__(self, cfg: "ProgressModelProcessorCfg"):
        super().__init__(cfg)
        self.image_list_to_data = ImageListToData()
        self.prompt_template_list = [
            "Given",
            " the historical observations: ",
            " the current image: <image>, and the instruction: ",
            " as input, the task is to predict a 448×448 BEV navigation map.",
        ]

    def pre_process(self, data: Union[ProgressModelInput, Dict]):

        # if isinstance(data, ProgressModelInput):
        data = self.image_list_to_data(data)

        if len(data["images"]) == 1:
            prompt = (
                self.prompt_template_list[0]
                + self.prompt_template_list[2]
                + data["instruction"]
                + self.prompt_template_list[3]
            )
        else:
            prompt = (
                self.prompt_template_list[0]
                + self.prompt_template_list[1]
                + "".join(["<image>"] * (len(data["images"]) - 1))
                + ","
                + self.prompt_template_list[2]
                + data["instruction"]
                + self.prompt_template_list[3]
            )

        input_list = {"prompt": prompt, "images": data["images"]}
        return input_list

    def post_process(self, model_outputs, _) -> ProgressModelOutput:
        # model_outputs is the base64 string returned by
        # ProgressModel.forward().
        image = model_outputs
        return ProgressModelOutput(image=image)


class ProgressModelProcessorCfg(ProcessorMixinCfg[ProgressModelProcessor]):
    class_type: ClassType_co[ProgressModelProcessor] = ProgressModelProcessor
    load_image: bool = True
