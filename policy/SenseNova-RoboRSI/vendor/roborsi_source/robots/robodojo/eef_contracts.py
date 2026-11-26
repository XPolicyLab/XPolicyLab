# Copyright 2026 The HEART Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Shared native action bounds and failure types, without Agent dependencies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .eef_types import JOINT_CHANNELS, ActionSpace


class CapabilityFailure(Exception):
    """Model-side failure: repair exhaustion, no tool call, give_up, or budget."""


class InfrastructureFailure(BaseException):
    """Provider-side failure: missing key, auth, misconfiguration, exhausted retries.

    Derived from ``BaseException`` on purpose. RoboDojo's layout loop in
    ``src/eval_client/main.py`` wraps each episode in ``except Exception`` and
    advances to the next layout, which would silently fold provider outages into
    the success rate. Every adapter site that must observe one catches it by
    name, so the only handler this bypasses is a blanket ``except Exception``.
    """

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        retry_after_s: float | None = None,
        status: int | None = None,
        key_unusable: bool = False,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.retry_after_s = retry_after_s
        self.status = status
        # Set when the provider refused the credential rather than the request:
        # the key is not one this account can serve the model with, now or in an
        # hour. Waiting cannot fix it; another key can.
        self.key_unusable = key_unusable


@dataclass(frozen=True)
class RoboDojoActionSpec:
    """The semantics RoboDojo exposes to the joint agent policy."""

    labels: tuple[str, ...]
    low: np.ndarray
    high: np.ndarray
    control_hz: float
    docs: str
    max_step: tuple[float | None, ...] | None = None

    def __post_init__(self) -> None:
        width = len(self.labels)
        channel_width = ActionSpace(JOINT_CHANNELS).width
        if width != channel_width:
            raise ValueError(
                f"action spec has {width} labels but the joint channels carry "
                f"{channel_width}; labels name channels by position"
            )
        if self.low.shape != (width,) or self.high.shape != (width,):
            raise ValueError("action bounds must be flat and match labels")
        if not np.all(np.isfinite(self.low)) or not np.all(np.isfinite(self.high)):
            raise ValueError("action bounds must be finite")
        if np.any(self.low > self.high):
            raise ValueError("action lower bounds must not exceed upper bounds")
        if not np.isfinite(self.control_hz) or self.control_hz <= 0:
            raise ValueError("control_hz must be finite and > 0")
        if self.max_step is None:
            return
        if len(self.max_step) != width:
            raise ValueError("max_step must be one entry per action dimension")
        for entry in self.max_step:
            if entry is not None and (not np.isfinite(entry) or entry <= 0):
                raise ValueError("max_step entries must be finite and > 0 or None")
