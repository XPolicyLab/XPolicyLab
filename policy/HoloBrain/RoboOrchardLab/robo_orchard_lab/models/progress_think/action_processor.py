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

__all__ = ["ActionModelProcessor", "ActionModelProcessorCfg"]


@dataclass
class ActionModelInput:
    """Data structure for inputs to the ProgressThink's navigation model.

    Each input contains:
        - a list of image file paths
        - a text instruction
        - a text partial instruction
    """

    image_paths: List[str]
    """List of image file paths representing the observation sequence."""

    instruction: str
    """Text instruction describing the navigation goal."""

    partial_instruction: str
    """Text instruction describing the completed navigation goal."""


@dataclass
class ActionModelOutput:
    """Output structure for the action model."""

    text: str
    """Model-generated text output."""


class PathListToData:
    """Convert image paths and instructions into the input format."""

    def __init__(self, load_image: bool = True):
        self.load_image = load_image

    def __call__(self, data: ActionModelInput) -> dict:
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
        input_data["partial_instruction"] = data.partial_instruction
        return input_data


class ActionModelProcessor(ModelIOProcessor):
    """Processor for the MonoDream navigation model.

    Converts image paths + instruction into a multimodal list input format,
    and extracts the generated text from model outputs.
    """

    cfg: "ActionModelProcessorCfg"

    def __init__(self, cfg: "ActionModelProcessorCfg"):
        super().__init__(cfg)
        self.pathlist_to_data = PathListToData(load_image=self.cfg.load_image)
        self.transforms = (
            [build(transform) for transform in self.cfg.transforms]
            if self.cfg.transforms is not None
            else []
        )
        self.prompt_template_list = [
            "Imagine you are a robot programmed for navigation tasks.",
            "You have been given a video of historical observations:",
            "and current observation:<image>",
            "Your full assigned task instruction is:",
            "You have probably finished the historical subtask instruction: ",
            (
                "Analyze this series of images to decide your next move, "
                "which could involve "
                "turning left or right by a specific degree, "
                "moving forward a certain distance, "
                "or stop if the task is completed."
            ),
        ]

    def pre_process(self, data: Union[ActionModelInput, Dict]):
        """Convert ActionModelInput into model-ready multimodal list format."""
        if isinstance(data, ActionModelInput):
            data = self.pathlist_to_data(data)

        for ts_i in self.transforms:
            data = ts_i(data)

        if len(data["images"]) == 1:
            prompt = (
                self.prompt_template_list[0]
                + self.prompt_template_list[2]
                + self.prompt_template_list[3]
                + data["instruction"]
                + self.prompt_template_list[5]
            )
        else:
            if len(data["partial_instruction"]) < 1000:
                prompt = (
                    self.prompt_template_list[0]
                    + "\n"
                    + self.prompt_template_list[1]
                    + "\n"
                    + "".join(["<image>"] * (len(data["images"]) - 1))
                    + self.prompt_template_list[2]
                    + "\n"
                    + self.prompt_template_list[3]
                    + "\n"
                    + data["instruction"]
                    + "\n"
                    + self.prompt_template_list[4]
                    + "\n"
                    + data["partial_instruction"]
                    + "\n"
                    + self.prompt_template_list[5]
                )
            else:
                prompt = (
                    self.prompt_template_list[0]
                    + "\n"
                    + self.prompt_template_list[1]
                    + "\n"
                    + "".join(["<image>"] * (len(data["images"]) - 1))
                    + self.prompt_template_list[2]
                    + "\n"
                    + self.prompt_template_list[3]
                    + "\n"
                    + data["instruction"]
                    + "\n"
                    + self.prompt_template_list[5]
                )

        input_list = [prompt]
        input_list.extend(data["images"])
        # print(input_list)
        # input("stop")
        return input_list

    def post_process(self, model_outputs, _) -> ActionModelOutput:
        """Extract model-generated text output."""
        text = model_outputs
        return ActionModelOutput(text=text)


class ActionModelProcessorCfg(ModelIOProcessorCfg[ActionModelProcessor]):
    """Configuration forActionModelProcessor."""

    class_type: ClassType_co[ActionModelProcessor] = ActionModelProcessor
    load_image: bool = True
    transforms: Optional[List[DelayInitDictType]] = None
