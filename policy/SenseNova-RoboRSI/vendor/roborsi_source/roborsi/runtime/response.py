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

"""Policy decisions made on a complete model response before tool dispatch."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pydantic_ai.messages import ModelRequest, ModelResponse


@dataclass(frozen=True)
class ResponseDecision:
    """Replace a response, or reject it with feedback and request another turn.

    A replacement must preserve any provider reasoning parts needed for replay.
    ``feedback`` bypasses SDK tool execution and validation for this response;
    it must contain a result for every retained call. It consumes one model
    completion, but no tool execution. Raise ``PlannerFailure`` to stop instead.
    """

    response: ModelResponse
    feedback: ModelRequest | None = None


ResponseProcessor = Callable[[ModelResponse], ResponseDecision]
