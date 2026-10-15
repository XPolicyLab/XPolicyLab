"""Replay a finished episode's conversation into the start of a new one.

An operator who reruns a layout after watching an attempt fail does not want
the model to rediscover the same dead end. Pointing
``L3_INSPECT_PRIOR_TRANSCRIPT`` at that attempt's transcript rebuilds its turns
as chat messages ahead of the first fresh observation.

The rerun resets the scene, so the replay is framed as history rather than as
something the model just did: the transcript ends in a pose and a set of object
positions that no longer exist, and a model that reads it as the present will
send a small delta from a pose the arm is nowhere near. The boundary note after
the replay says so explicitly.

Images are not replayed. Reasoning items are not either — they belong to the
Azure session that produced them, and ``ReasoningReplayStore`` has nothing
keyed to the replayed call ids, so the assistant turns go out chat-shaped.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

PREFACE = (
    "The transcript that follows is a PREVIOUS, FINISHED attempt at this same "
    "task and scene layout, replayed so that you do not repeat it. You are not "
    "in that episode any more. Read it, then read the boundary note after it "
    "before you act."
)


def _outcome(transcript: Mapping[str, Any]) -> str:
    success = transcript.get("official_success") or []
    if success and success[0]:
        return "It succeeded."
    reason = transcript.get("termination_reason")
    if reason:
        return f"It ended in failure: {reason}."
    return (
        "It ended in failure: the environment terminated the episode on its own; "
        "the agent never chose to stop."
    )


def _boundary(transcript: Mapping[str, Any], note: str | None) -> str:
    lines = [
        "End of the previous attempt.",
        _outcome(transcript),
        "The scene has since been RESET to its initial state: the robot is back "
        "at its home configuration and every object is back at its starting "
        "pose. Nothing the previous attempt did to the world persists.",
        "The joint state and camera images you receive from here on are the "
        "fresh, authoritative state. Do not assume the pose the transcript ends "
        "in, and do not issue a small delta from it.",
        "Your LLM call budget is full again; the calls above do not count against it.",
    ]
    if note:
        lines.append(f"Operator note: {note}")
    return "\n\n".join(lines)


def _observation_message(
    turn: Mapping[str, Any],
    instruction: str | None,
    render_state: Callable[[Mapping[str, Any], str | None], str],
) -> dict[str, Any]:
    observation = turn.get("observation") or {}
    state = observation.get("state") or {}
    step = turn.get("policy_step")
    text = render_state(state, instruction)
    return {
        "role": "user",
        "content": (
            f"[previous attempt, step {step}]\n{text}\n"
            "(camera images from that step are not replayed)"
        ),
    }


def _call_messages(call: Mapping[str, Any]) -> list[dict[str, Any]]:
    response = call.get("response") or {}
    choices = response.get("choices") or [{}]
    message = choices[0].get("message") or {}
    tool_calls = message.get("tool_calls") or []
    content = message.get("content")
    if not tool_calls:
        # A turn the model answered with prose instead of a tool call. The
        # repair prompt that followed is not replayed; the model can see from
        # the next turn that the attempt carried on.
        return [{"role": "assistant", "content": content}] if content else []
    result = str(call.get("tool_result") or "")
    messages: list[dict[str, Any]] = [
        {"role": "assistant", "content": content, "tool_calls": tool_calls}
    ]
    messages.extend(
        {"role": "tool", "tool_call_id": str(tool_call["id"]), "content": result}
        for tool_call in tool_calls
    )
    return messages


def build_prior_messages(
    transcript: Mapping[str, Any],
    *,
    render_state: Callable[[Mapping[str, Any], str | None], str],
    note: str | None = None,
) -> list[dict[str, Any]]:
    """The previous attempt as chat messages, wrapped in its own framing."""
    turns = transcript.get("turns") or []
    if not turns:
        raise ValueError("prior transcript has no turns to replay")
    instruction = transcript.get("instruction")
    messages: list[dict[str, Any]] = [{"role": "user", "content": PREFACE}]
    for turn in turns:
        messages.append(_observation_message(turn, instruction, render_state))
        for call in turn.get("llm_calls") or []:
            messages.extend(_call_messages(call))
    messages.append({"role": "user", "content": _boundary(transcript, note)})
    return messages


def load_prior_messages(
    path: str | Path,
    *,
    render_state: Callable[[Mapping[str, Any], str | None], str],
    note: str | None = None,
) -> list[dict[str, Any]]:
    """``build_prior_messages`` for a transcript on disk."""
    try:
        transcript = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read prior transcript {path}: {error}") from error
    if not isinstance(transcript, dict):
        raise ValueError(f"prior transcript {path} is not a JSON object")
    try:
        return build_prior_messages(transcript, render_state=render_state, note=note)
    except ValueError as error:
        raise ValueError(f"{error} ({path})") from error
