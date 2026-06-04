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

from robo_orchard_core.utils.config import (
    load_config_class,
)

from robo_orchard_lab.models.mixin import ClassType_co
from robo_orchard_lab.models.monodream import MonoDream, MonoDreamConfig

__all__ = ["ActionModel", "ActionModelConfig"]


class ActionModel(MonoDream):
    cfg: "ActionModelConfig"  # for type hint

    def __init__(self, cfg: "ActionModelConfig"):
        super().__init__(cfg)
        self.cfg = cfg

    @classmethod
    def load_model(cls, directory: str, use_decrete: bool = True):

        config_file = os.path.join(directory, "model.config.json")

        with open(config_file, "r") as f:
            cfg: ActionModelConfig = load_config_class(f.read())

        instance = cls(cfg)
        instance.load_from_decrete_model(directory)

        return instance


class ActionModelConfig(MonoDreamConfig):
    class_type: ClassType_co[ActionModel] = ActionModel
