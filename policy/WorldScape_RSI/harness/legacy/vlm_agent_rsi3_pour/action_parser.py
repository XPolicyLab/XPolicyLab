"""Parse complete VLM responses without turning malformed objects into actions."""

from __future__ import annotations

import json
import math
import re
from typing import Any


class ActionParseError(ValueError):
    """The response cannot be interpreted as an unambiguous action."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ActionParseError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ActionParseError("nonfinite JSON number")


_DECODER = json.JSONDecoder(
    object_pairs_hook=_unique_object, parse_constant=_reject_constant,
)


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _check_finite_tree(value: Any) -> None:
    if isinstance(value, dict):
        for child in value.values():
            _check_finite_tree(child)
    elif isinstance(value, list):
        for child in value:
            _check_finite_tree(child)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if not _is_finite_number(value):
            raise ActionParseError("nonfinite JSON number")


ANSWER_KEY_RE = re.compile(r'"(commands|action)"\s*:')


def _candidate(text: str) -> tuple[str, int]:
    """Select the outer JSON object to parse.

    A reply may carry several top-level containers (a thinking object
    followed by the answer, or the answer followed by a note): the object
    holding the answer keys wins, otherwise the last object. Containers are
    never re-tried after a nested failure.
    """
    if not isinstance(text, str) or not text.strip():
        raise ActionParseError("empty model response")
    if len(text) > 200_000:
        raise ActionParseError("model response exceeds parser size limit")

    spans: list[tuple[int, int]] = []
    start = None
    stack: list[str] = []
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"' and start is not None:
            in_string = True
        elif char in "{[":
            if start is None:
                start = index
            stack.append(char)
        elif char in "}]" and start is not None:
            if not stack or (stack.pop(), char) not in (("{", "}"), ("[", "]")):
                raise ActionParseError("mismatched JSON container delimiters")
            if not stack:
                spans.append((start, index + 1))
                start = None
    if not spans:
        if start is None:
            raise ActionParseError("no JSON object in model response")
        raise ActionParseError("incomplete outer JSON container")
    objects = [(first, last) for first, last in spans if text[first] == "{"]
    if not objects:
        raise ActionParseError("model response must be an object, not an array")
    chosen = objects[0]
    if len(spans) > 1:
        answers = [(first, last) for first, last in objects if ANSWER_KEY_RE.search(text[first:last])]
        chosen = answers[0] if answers else objects[-1]
    return text[chosen[0]:chosen[1]], chosen[0]


def _normalize_numeric_tokens(candidate: str, source_offset: int) -> tuple[str, list[dict]]:
    """Use JSON5 only for leading-plus numbers; strict JSON checks the grammar.

    Strings are consumed by the standard JSON decoder, so their contents are
    never rewritten. Comments, trailing commas, bare keys and truncated JSON
    remain errors even when a response also contains a recoverable number.
    """
    try:
        import json5
    except ImportError as exc:
        raise ActionParseError("numeric JSON recovery requires the json5 dependency") from exc

    pieces = []
    repairs = []
    index = 0
    separators = " \t\r\n{}[]:,\""
    while index < len(candidate):
        char = candidate[index]
        if char == '"':
            _, end = _DECODER.raw_decode(candidate, index)
            pieces.append(candidate[index:end])
            index = end
            continue
        if char in separators:
            pieces.append(char)
            index += 1
            continue

        end = index + 1
        while end < len(candidate) and candidate[end] not in separators:
            end += 1
        token = candidate[index:end]
        replacement = token
        if token not in ("true", "false", "null"):
            try:
                value = _DECODER.decode(token)
            except json.JSONDecodeError:
                if not token.startswith("+"):
                    raise ActionParseError("only numeric leading-plus recovery is supported")
                # The remainder must already be a standard JSON number; do not
                # widen recovery to hexadecimal, decimal shorthand or repair.
                plain_value = _DECODER.decode(token[1:])
                if not _is_finite_number(plain_value):
                    raise ActionParseError("recovery accepts only finite JSON numbers")
                try:
                    value = json5.loads(token, allow_duplicate_keys=False)
                except ValueError as exc:
                    raise ActionParseError("invalid JSON token") from exc
                if not _is_finite_number(value):
                    raise ActionParseError("recovery accepts only finite JSON5 numbers")
                replacement = json.dumps(value, allow_nan=False)
                repairs.append({
                    "kind": "numeric_leading_plus",
                    "offset": source_offset + index,
                    "original": token,
                    "normalized": replacement,
                })
            if not _is_finite_number(value):
                raise ActionParseError("invalid or nonfinite JSON number")
        pieces.append(replacement)
        index = end
    if not repairs:
        raise ActionParseError("invalid JSON syntax; no numeric recovery applies")
    return "".join(pieces), repairs


def _parse_object(text: str) -> tuple[dict, dict]:
    candidate, offset = _candidate(text)
    metadata = _metadata()
    try:
        parsed = _DECODER.decode(candidate)
    except json.JSONDecodeError:
        normalized, repairs = _normalize_numeric_tokens(candidate, offset)
        parsed = _DECODER.decode(normalized)
        metadata.update(
            parser="json5_numeric", recovered=True, repair_details=repairs,
            repairs=[
                f"numeric leading plus at offset {item['offset']}: {item['original']} -> {item['normalized']}"
                for item in repairs
            ],
        )
    _check_finite_tree(parsed)
    return parsed, metadata


def _metadata() -> dict:
    return {"parser": "json", "recovered": False, "repairs": [], "repair_details": [], "error": None}


# A turn is answered with a list of command strings (see commands.py) plus
# free-text fields.
MAIN_TEXT_FIELDS = ("scene", "progress", "memory", "plan")
MAIN_MAX_COMMANDS = 16


def _coerce_commands(value: Any) -> list[str] | None:
    """Command strings from the shapes models actually send: a list of strings, one string with several
    lines, or a list of {"command": ...} objects. ``None`` when the value is none of those."""
    if isinstance(value, str):
        parts = [part.strip() for part in re.split(r"[\n;]+", value) if part.strip()]
        return parts or None
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        return None
    commands: list[str] = []
    for item in value:
        if isinstance(item, dict):
            item = item.get("command") or item.get("cmd") or item.get("text")
        if not isinstance(item, str) or not item.strip():
            return None
        commands.append(item.strip())
    return commands


def _validate_main(parsed: dict, allow_flow=False) -> None:
    if allow_flow and isinstance(parsed.get("flow"), dict):
        parsed["commands"] = []
        for field in MAIN_TEXT_FIELDS:
            if parsed.get(field) is not None and not isinstance(parsed[field], str):
                parsed[field] = json.dumps(parsed[field], ensure_ascii=False)
        return
    commands = _coerce_commands(parsed.get("commands"))
    import os
    attack_control_only = (commands == [] and os.environ.get('RDA_ATTACK_MODE') == '1'
                           and os.environ.get('RDA_STAGE') == 'experiment'
                           and any(k in parsed for k in ('attack_checkpoint', 'early_abort')))
    if (not commands and not attack_control_only) or len(commands or []) > MAIN_MAX_COMMANDS:
        raise ActionParseError(f"commands must be a list of one to {MAIN_MAX_COMMANDS} command strings")
    parsed["commands"] = commands
    for field in MAIN_TEXT_FIELDS:
        if parsed.get(field) is not None and not isinstance(parsed[field], str):
            parsed[field] = json.dumps(parsed[field], ensure_ascii=False)


def parse_action_response(text: str, allow_flow=False) -> tuple[dict | None, dict]:
    """Return a validated response and metadata suitable for the decision log.

    The reply has to carry a list of discrete command strings under
    ``commands``; the free-text fields are coerced to strings. On failure,
    return ``None`` and a short error instead of executing a partial object.
    Numeric recovery is recorded with exact token offsets and values.
    """
    metadata = _metadata()
    try:
        parsed, metadata = _parse_object(text)
        _validate_main(parsed, allow_flow=allow_flow)
    except (ValueError, RecursionError) as exc:
        metadata["recovered"] = False
        metadata["error"] = str(exc) if isinstance(exc, ActionParseError) else "invalid JSON syntax"
        return None, metadata
    return parsed, metadata


def extract_json_object(text: str) -> dict | None:
    """Compatibility helper for callers that need an object without a schema."""
    try:
        parsed, _ = _parse_object(text)
        return parsed
    except (ValueError, RecursionError):
        return None
