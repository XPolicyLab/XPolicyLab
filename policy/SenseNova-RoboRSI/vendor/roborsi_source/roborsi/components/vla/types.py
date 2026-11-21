# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Provider-independent VLA configuration, capabilities and predictions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class VLAConfig:
    """Select a provider factory with finite JSON configuration parameters."""

    factory: str
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VLASpec:
    """Describe formats, actual capabilities and known provider identity."""

    name: str
    observation_format: str
    action_format: str
    stateful: bool = False
    supports_reset: bool = False
    supports_timeout: bool = False
    identity: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VLAPrediction:
    """Keep the provider's native actions and factual inference metadata."""

    actions: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)


class VLA(Protocol):
    """Synchronous inference component whose caller controls its lifecycle."""

    @property
    def spec(self) -> VLASpec:
        """Return this provider's formats and capabilities."""
        ...

    def predict(
        self,
        observation: Mapping[str, Any],
        *,
        instruction: str | None = None,
        options: Mapping[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> VLAPrediction:
        """Infer actions without executing them or modifying the observation."""
        ...

    def reset(self) -> None:
        """Reset an owned model session if this capability is supported."""
        ...

    def close(self) -> None:
        """Release owned resources without shutting down external services."""
        ...
