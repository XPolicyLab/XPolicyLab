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

"""Compose preset layers and typed overrides without importing Agent code."""

from __future__ import annotations

import copy
import json
import os
from collections.abc import Sequence
from pathlib import Path

from roborsi.utils.config_values import SECRET_KEYS, json_value, reject_secrets

FIELDS = {
    "schema_version",
    "factory",
    "source_root",
    "code",
    "agent",
    "runtime",
    "prompt_file",
    "skills",
    "resources",
    "parameters",
    "dependencies",
}


def _segments(path: str) -> list[str]:
    if path.startswith("/"):
        import re

        if re.search(r"~(?![01])", path):
            raise ValueError("invalid override path escape")
        parts = [p.replace("~1", "/").replace("~0", "~") for p in path[1:].split("/")]
    else:
        parts = path.split(".")
    if not all(parts) or parts[0] not in FIELDS:
        raise ValueError("override must select a known Agent config field")
    if any(p.lower() in SECRET_KEYS for p in parts):
        raise ValueError("credentials must come from the environment, not overrides")
    return parts


def normalize_overrides(values: Sequence[str]) -> tuple[str, ...]:
    """Validate PATH=JSON values and give equivalent spellings one identity."""
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError("overrides must be a sequence of PATH=JSON strings")
    result = []
    for item in values:
        if not isinstance(item, str):
            raise ValueError("overrides must contain PATH=JSON strings")
        path, sep, raw = item.partition("=")
        if not sep:
            raise ValueError("override must be PATH=JSON")
        parts = _segments(path)
        try:
            value = json.loads(raw)
        except ValueError as exc:
            raise ValueError(
                "override value must be JSON; quote string values"
            ) from exc
        encoded = json_value(value)
        reject_secrets(value)
        pointer = "/" + "/".join(p.replace("~", "~0").replace("/", "~1") for p in parts)
        result.append(pointer + "=" + encoded)
    return tuple(result)


def _set(data: dict, override: str) -> None:
    path, _, raw = override.partition("=")
    parts = _segments(path)
    current = data
    for index, key in enumerate(parts):
        last = index == len(parts) - 1
        if isinstance(current, dict):
            if last:
                current[key] = json.loads(raw)
            else:
                current = current.setdefault(key, {})
        elif isinstance(current, list) and key.isdecimal() and int(key) < len(current):
            if last:
                current[int(key)] = json.loads(raw)
            else:
                current = current[int(key)]
        else:
            raise ValueError(f"override cannot traverse or index {path}")


def _merge(base: dict, patch: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in patch.items():
        result[key] = (
            _merge(result[key], value)
            if isinstance(result.get(key), dict) and isinstance(value, dict)
            else copy.deepcopy(value)
        )
    return result


def _rebase(data: dict, origin: Path, target: Path) -> dict:
    result = copy.deepcopy(data)
    if origin == target:
        return result

    def path(value):
        if not isinstance(value, str) or not value.strip() or Path(value).is_absolute():
            return value
        return os.path.relpath(os.path.abspath(origin / value), target)

    def paths(container, name):
        if isinstance(container, dict) and isinstance(container.get(name), list):
            container[name] = [path(p) for p in container[name]]

    for name in ("source_root", "prompt_file"):
        if name in result:
            result[name] = path(result[name])
    for name in ("code", "skills"):
        paths(result, name)
    if isinstance(result.get("resources"), dict):
        result["resources"] = {k: path(v) for k, v in result["resources"].items()}
    agent = result.get("agent")
    if isinstance(agent, dict) and "working_dir" in agent:
        agent["working_dir"] = path(agent["working_dir"])
    runtime = result.get("runtime")
    paths(runtime, "skill_paths")
    if isinstance(runtime, dict) and isinstance(runtime.get("subagents"), dict):
        for subagent in runtime["subagents"].values():
            paths(subagent, "skills")
            paths(subagent, "skill_paths")
    return result


def compose(
    path: Path, overrides: Sequence[str] = ()
) -> tuple[dict, tuple[Path, ...], tuple[str, ...]]:
    """Merge parents left to right, then the leaf, then ordered CLI overrides.

    Mappings merge recursively; lists/scalars replace. File paths keep the
    declaring file's directory. Every parent document remains a pinned input.
    """
    import yaml

    normalized = normalize_overrides(overrides)
    documents = {}

    def read(selected: Path, stack: tuple[Path, ...]) -> dict:
        selected = Path(os.path.abspath(selected))
        if any(p.is_symlink() for p in (selected, *selected.parents)):
            raise ValueError("Agent config cannot contain symlinks")
        if selected in stack:
            raise ValueError("Agent config inheritance cycle")
        if len(stack) >= 32:
            raise ValueError("Agent config inheritance exceeds 32 layers")
        if selected in documents:
            return documents[selected]
        try:
            data = yaml.safe_load(selected.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ValueError(f"invalid Agent config: {selected}") from exc
        if not isinstance(data, dict) or set(data) - FIELDS - {"extends"}:
            raise ValueError("Agent config must be a mapping with known fields")
        if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
            raise ValueError("Agent config requires schema_version: 1")
        json_value(data)
        reject_secrets(data)
        parents = data.pop("extends", [])
        if isinstance(parents, str):
            parents = [parents]
        if not isinstance(parents, list) or any(
            not isinstance(p, str) or not p.strip() for p in parents
        ):
            raise ValueError("extends must be a path or list of paths")
        merged = {}
        for parent in parents:
            merged = _merge(merged, read(selected.parent / parent, (*stack, selected)))
        merged = _merge(merged, _rebase(data, selected.parent, path.parent))
        documents[selected] = merged
        return merged

    data = copy.deepcopy(read(path, ()))
    for override in normalized:
        _set(data, override)
    return data, tuple(sorted(documents)), normalized
