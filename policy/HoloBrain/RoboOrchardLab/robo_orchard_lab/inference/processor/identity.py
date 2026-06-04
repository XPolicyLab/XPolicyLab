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

from typing_extensions import deprecated

from robo_orchard_lab.inference.processor.mixin import (
    ProcessorMixin,
    ProcessorMixinCfg,
)
from robo_orchard_lab.processing.io_processor.identity import (
    IdentityIOProcessor,
    IdentityIOProcessorCfg,
)

__all__ = [
    "IdentityProcessor",
    "IdentityProcessorCfg",
]


@deprecated(
    "Use `robo_orchard_lab.processing.io_processor.identity."
    "IdentityIOProcessor` instead.",
    category=None,
)
class IdentityProcessor(IdentityIOProcessor):
    """Backward-compatible facade for the historical identity processor.

    This deprecated class preserves the old ``IdentityProcessor`` import path
    while delegating behavior to
    :class:`robo_orchard_lab.processing.io_processor.identity.IdentityIOProcessor`.
    It remains a pass-through processor for pipelines that require a processor
    object but no transformation.
    """

    __add__ = ProcessorMixin.__add__


@deprecated(
    "Use `robo_orchard_lab.processing.io_processor.identity."
    "IdentityIOProcessorCfg` instead.",
    category=None,
)
class IdentityProcessorCfg(IdentityIOProcessorCfg):
    """Backward-compatible config for :class:`IdentityProcessor`.

    This deprecated config preserves the legacy serialized config path for the
    identity processor while reusing the canonical implementation.
    """

    class_type: type[IdentityProcessor] = IdentityProcessor
    __add__ = ProcessorMixinCfg.__add__
