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

"""Immutable request evidence for user-owned context cache policies.

Fingerprints describe SDK inputs, not provider tokenization or cache residency.
No message bodies, credentials or model clients are retained in these records.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from datetime import date, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

from pydantic_ai.messages import (
    CachePoint,
    ModelRequest,
    ModelResponse,
    ToolReturnPart,
    UserPromptPart,
)

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage
    from pydantic_ai.models import ModelRequestContext

    from roborsi.runtime.context_engine import CacheBreakpoint


def _canonical(value: Any) -> Any:
    if isinstance(value, type):
        return {"class": f"{value.__module__}.{value.__qualname__}"}
    if isinstance(value, bytes):
        return {"bytes_sha256": hashlib.sha256(value).hexdigest()}
    if isinstance(value, Enum):
        return _canonical(value.value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "type": type(value).__qualname__,
            **{
                field.name: _canonical(getattr(value, field.name))
                for field in fields(value)
                if field.name not in {"timestamp", "run_id", "conversation_id"}
                and not (
                    isinstance(value, ModelRequest) and field.name == "instructions"
                )
                and not (isinstance(value, ModelResponse) and field.name == "usage")
            },
        }
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("cache fingerprint requires string mapping keys")
        return {key: _canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value if not isinstance(item, CachePoint)]
    if isinstance(value, (set, frozenset)):
        return sorted(
            (_canonical(item) for item in value),
            key=lambda item: json.dumps(item, sort_keys=True),
        )
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError("unsupported cache fingerprint value")


def _digest(value: Any) -> str | None:
    try:
        encoded = json.dumps(
            _canonical(value), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class CacheUsage:
    """Reported counters; None means unavailable, not zero."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class CacheBoundary:
    """A selected boundary in a request handed to the model adapter."""

    position: CacheBreakpoint
    content_id: str | None
    prefix_digest: str | None
    tool_call_id: str | None = None


@dataclass(frozen=True, slots=True)
class CacheRequest:
    """One logical model call, including any transport retries.

    Completed means a response was received, not that a cache write succeeded.
    Adapter dispatch is recorded even when no HTTP request reached the provider.
    Static instruction markers are separate from the message boundaries.
    """

    sequence: int
    boundaries: tuple[CacheBoundary, ...]
    configuration_digest: str | None
    static_prefix: bool
    status: Literal["pending", "completed", "failed", "cancelled"] = "pending"
    usage: CacheUsage | None = None


@dataclass(frozen=True, slots=True)
class CacheMatch:
    """Locate an old boundary without assuming its old array index is valid."""

    boundary: CacheBoundary
    position: CacheBreakpoint | None
    prefix_unchanged: bool | None
    configuration_unchanged: bool | None

    @property
    def reusable_prefix(self) -> bool:
        """Whether known SDK inputs match; does not guarantee a cache hit."""
        return (
            self.position is not None
            and self.prefix_unchanged is True
            and self.configuration_unchanged is True
        )


@dataclass(frozen=True, slots=True)
class CacheState:
    """Previous call evidence plus current configuration for a context callback.

    Call ``resolve`` after modifying history to obtain positions in that history.
    Returning a position in ContextResult is still the user's explicit choice.
    """

    previous: CacheRequest | None
    configuration_digest: str | None

    def resolve(self, messages: list[ModelMessage]) -> tuple[CacheMatch, ...]:
        """Match by tool-call identity or unique user content, then check prefixes."""
        if self.previous is None or not self.previous.boundaries:
            return ()
        candidates = _boundaries(messages)
        configuration_unchanged = _same(
            self.previous.configuration_digest, self.configuration_digest
        )
        matches = []
        for old in self.previous.boundaries:
            found = [
                item
                for item in candidates
                if old.content_id is not None and old.content_id == item.content_id
            ]
            # Repeated identical user blocks are ambiguous; never guess an index.
            current = found[0] if len(found) == 1 else None
            matches.append(
                CacheMatch(
                    old,
                    current.position if current else None,
                    _same(old.prefix_digest, current.prefix_digest)
                    if current
                    else False,
                    configuration_unchanged,
                )
            )
        return tuple(matches)


def _same(first: str | None, second: str | None) -> bool | None:
    return first == second if first is not None and second is not None else None


def _boundaries(messages: list[ModelMessage]) -> tuple[CacheBoundary, ...]:
    from roborsi.runtime.context_engine import CacheBreakpoint

    output = []
    prefix = hashlib.sha256()
    known = True

    def extend(value: Any) -> None:
        nonlocal known
        digest = _digest(value)
        if digest is None:
            known = False
        else:
            prefix.update(digest.encode("ascii"))

    for mi, message in enumerate(messages):
        extend(replace(message, parts=[]))
        for pi, part in enumerate(message.parts):
            if isinstance(part, UserPromptPart):
                extend(replace(part, content=[]))
                items = (
                    [part.content] if isinstance(part.content, str) else part.content
                )
                index = -1
                for item in items:
                    if isinstance(item, CachePoint):
                        continue
                    index += 1
                    digest = _digest(item)
                    extend(item)
                    output.append(
                        CacheBoundary(
                            CacheBreakpoint(mi, pi, index),
                            f"user:{digest}" if digest else None,
                            prefix.hexdigest() if known else None,
                        )
                    )
                extend("end-user-content")
            else:
                extend(part)
                if isinstance(part, ToolReturnPart):
                    output.append(
                        CacheBoundary(
                            CacheBreakpoint(mi, pi),
                            f"tool:{part.tool_call_id}",
                            prefix.hexdigest() if known else None,
                            part.tool_call_id,
                        )
                    )
        extend("end-message")
    return tuple(output)


def configuration_digest(request: ModelRequestContext) -> str | None:
    """Include model, instructions, tools and settings; unsupported values stay unknown."""
    from roborsi.llm.retry import RetryLoggingModel

    model = request.model
    if isinstance(model, RetryLoggingModel):
        model = model.wrapped
    settings = {**(model.settings or {}), **(request.model_settings or {})}
    settings.pop("roborsi_cache_tool_call_ids", None)
    return _digest(
        {
            "system": model.system,
            "model": model.model_name,
            "base_url": str(model.base_url),
            "profile": dict(model.profile),
            "settings": settings,
            "parameters": request.model_request_parameters,
        }
    )


def capture_request(request: ModelRequestContext, sequence: int) -> CacheRequest:
    """Record the final selected SDK boundaries immediately before model dispatch."""
    from roborsi.runtime.context_engine import CacheBreakpoint

    settings = {**(request.model.settings or {}), **(request.model_settings or {})}
    tool_ids = set(settings.get("roborsi_cache_tool_call_ids", ()))
    positions = set()
    for mi, message in enumerate(request.messages):
        for pi, part in enumerate(message.parts):
            if isinstance(part, UserPromptPart) and isinstance(part.content, list):
                index = -1
                for item in part.content:
                    if isinstance(item, CachePoint):
                        positions.add(CacheBreakpoint(mi, pi, index))
                    else:
                        index += 1
            if isinstance(part, ToolReturnPart) and part.tool_call_id in tool_ids:
                positions.add(CacheBreakpoint(mi, pi))
    return CacheRequest(
        sequence=sequence,
        boundaries=(
            tuple(
                boundary
                for boundary in _boundaries(request.messages)
                if boundary.position in positions
            )
            if positions
            else ()
        ),
        configuration_digest=configuration_digest(request),
        static_prefix=bool(settings.get("roborsi_cache_static_prefix")),
    )
