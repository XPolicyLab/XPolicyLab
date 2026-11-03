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

"""Finite JSON and credential field checks shared by public configuration APIs."""

from __future__ import annotations

import json
from typing import Any

SECRET_KEYS = {"api_key", "authorization", "password", "secret", "access_token"}


def json_value(value: Any) -> str:
    """Serialize finite JSON values without implicit key/type conversions."""
    try:
        result = json.dumps(value, allow_nan=False, sort_keys=True)
        if json.loads(result) != value:
            raise ValueError("configuration must use JSON object keys and values")
        return result
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Agent config must contain finite JSON-compatible values"
        ) from exc


def reject_secrets(value: Any) -> None:
    """Require named credential fields to come from the environment."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower() in SECRET_KEYS and item is not None:
                raise ValueError(
                    f"{key} must come from the environment, not Agent config"
                )
            reject_secrets(item)
    elif isinstance(value, list):
        for item in value:
            reject_secrets(item)
