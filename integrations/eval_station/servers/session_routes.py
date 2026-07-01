"""Shared HTTP route parsing for RoboDojo session control planes."""

from __future__ import annotations

from urllib.parse import unquote, urlparse


def parse_session_route(path: str) -> tuple[str, str, int | None] | None:
    parts = urlparse(path).path.strip("/").split("/")
    match parts:
        case ["sessions", raw_evaluation_id, "dispatch"]:
            evaluation_id = unquote(raw_evaluation_id)
            return (evaluation_id, "dispatch", None) if evaluation_id else None
        case ["sessions", raw_evaluation_id, "trials", raw_trial_index, action]:
            evaluation_id = unquote(raw_evaluation_id)
            if action not in ("start", "stop"):
                return None
            try:
                trial_index = int(raw_trial_index)
            except ValueError:
                return None
            if not evaluation_id or trial_index < 1:
                return None
            return evaluation_id, action, trial_index
    return None
