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

"""Public processor families.

`EnvelopeIOProcessor` and related envelope helpers are the preferred surface
for new code. `ModelIOProcessor` and `ComposedIOProcessor` remain exported as
legacy-compatible inputs that are automatically resolved into the envelope
runtime where supported.

`__all__` intentionally stays focused on the most common processor families
and compose helpers. Legacy `ModelIOProcessor` family names remain available
through deprecated package-level re-exports so explicit historical imports do
not break during the migration window.
"""

from __future__ import annotations
import warnings
from typing import TYPE_CHECKING, Any

from . import (
    base as _base_module,
    compose as _compose_module,
)
from .base import (
    ClassType_co,
    ModelIOProcessorCfgType_co,
    ModelIOProcessorType_co,
)
from .compose_envelope import (
    ComposedEnvelopeIOProcessor,
    ComposedEnvelopeIOProcessorCfg,
    ProcessorContextStack,
    compose_envelope,
    compose_envelope_cfg,
)
from .envelope import (
    EnvelopeIOProcessor,
    EnvelopeIOProcessorCfg,
    EnvelopeIOProcessorCfgType_co,
    EnvelopeIOProcessorType_co,
    PipelineEnvelope,
    normalize_pipeline_envelope,
)
from .identity import IdentityIOProcessor, IdentityIOProcessorCfg

if TYPE_CHECKING:
    from .base import ModelIOProcessor, ModelIOProcessorCfg
    from .compose import ComposedIOProcessor, ComposedIOProcessorCfg

__all__ = [
    "EnvelopeIOProcessor",
    "EnvelopeIOProcessorCfg",
    "PipelineEnvelope",
    "ComposedEnvelopeIOProcessor",
    "ComposedEnvelopeIOProcessorCfg",
    "ProcessorContextStack",
    "compose_envelope",
    "compose_envelope_cfg",
    "IdentityIOProcessor",
    "IdentityIOProcessorCfg",
]

_DEPRECATED_COMPAT_EXPORTS = {
    "ModelIOProcessor": _base_module.ModelIOProcessor,
    "ModelIOProcessorCfg": _base_module.ModelIOProcessorCfg,
    "ComposedIOProcessor": _compose_module.ComposedIOProcessor,
    "ComposedIOProcessorCfg": _compose_module.ComposedIOProcessorCfg,
}


def __getattr__(name: str) -> Any:
    """Lazily resolve deprecated compatibility re-exports.

    Args:
        name (str): Attribute requested from this package module.

    Returns:
        Any: Resolved legacy symbol from its defining submodule.

    Raises:
        AttributeError: If ``name`` is not a supported package export.
    """
    if name in _DEPRECATED_COMPAT_EXPORTS:
        warnings.warn(
            f"`{__name__}.{name}` is a deprecated compatibility re-export. "
            f"Import `{name}` from `{__name__}.base` or "
            f"`{__name__}.compose` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _DEPRECATED_COMPAT_EXPORTS[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return the public module surface including deprecated compat names."""
    return sorted(set(globals()) | _DEPRECATED_COMPAT_EXPORTS)
