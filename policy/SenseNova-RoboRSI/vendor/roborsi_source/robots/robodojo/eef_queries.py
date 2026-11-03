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

"""RoboDojo-owned EEF queries; no action, observation or model invocation."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from jsonschema import Draft202012Validator


def query(env: Any, method: str, arguments: dict) -> dict:
    """Validate a remote planning request before accessing the live robot."""
    if method == "eef_spec":
        if arguments:
            raise ValueError("eef_spec does not accept arguments")
        from .eef_native import action_spec

        spec = asdict(action_spec(env))
        spec["low"] = spec["low"].tolist()
        spec["high"] = spec["high"].tolist()
        return spec
    if method != "plan_eef":
        raise ValueError(f"unsupported RoboDojo query: {method}")
    Draft202012Validator(
        {
            "type": "object",
            "required": ["arm", "target_pose"],
            "properties": {
                "arm": {"enum": ["left", "right"]},
                "target_pose": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 7,
                    "maxItems": 7,
                },
            },
            "additionalProperties": False,
        }
    ).validate(arguments)
    json.dumps(arguments, allow_nan=False)
    from .eef_native import _as_numpy, native_planner

    result = native_planner(env)(
        arm=arguments["arm"], target_pose=_as_numpy(arguments["target_pose"])
    )
    # Only these two fields are consumed by the source motion controller.
    reply = {"status": result.get("status", "Fail")}
    if result.get("position") is not None:
        reply["position"] = _as_numpy(result["position"]).tolist()
    json.dumps(reply, allow_nan=False)
    return reply
