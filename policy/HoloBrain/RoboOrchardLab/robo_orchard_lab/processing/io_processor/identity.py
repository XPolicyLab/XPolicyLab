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

from robo_orchard_lab.processing.io_processor.base import (
    ClassType_co,
    ModelIOProcessor,
    ModelIOProcessorCfg,
)

__all__ = [
    "IdentityIOProcessor",
    "IdentityIOProcessorCfg",
]


class IdentityIOProcessor(ModelIOProcessor):
    """A processor that performs no operations.

    This processor serves as a pass-through component, returning the data it
    receives without modification. It is useful as a default processor or as a
    placeholder in pipelines where no pre-processing or post-processing is
    required. It preserves the standard processor contract unchanged:
    ``pre_process`` passes through one sample at a time, while
    ``post_process`` returns the usually-batched model outputs unchanged.
    """

    cfg: "IdentityIOProcessorCfg"

    def __init__(self, cfg: "IdentityIOProcessorCfg"):
        """Initialize the identity processor."""
        super().__init__(cfg)

    def pre_process(self, data):
        """Return the input data without modification.

        Args:
            data: The raw input data. In the default inference flow this is
                typically one sample before collation.

        Returns:
            The same input data, unchanged.
        """
        return data

    def post_process(self, model_outputs, _):
        """Return the model outputs without modification.

        Args:
            model_outputs: The raw output from the model's forward pass.
                In the default inference flow this is typically batched.
            _: Unused model input placeholder kept for interface
                compatibility. In the default inference flow it is usually the
                collated batch.

        Returns:
            The same model outputs, unchanged.
        """
        return model_outputs


class IdentityIOProcessorCfg(ModelIOProcessorCfg[IdentityIOProcessor]):
    """Configuration for :class:`IdentityIOProcessor`."""

    class_type: ClassType_co[IdentityIOProcessor] = IdentityIOProcessor
