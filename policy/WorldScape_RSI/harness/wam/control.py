"""Run one continuous WAM control segment inside an existing RoboDojo episode."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

if __package__:
    from .robodojo_adapter import action_to_robodojo, build_inputs, capture_frame
else:
    from robodojo_adapter import action_to_robodojo, build_inputs, capture_frame


@dataclass(frozen=True)
class SegmentResult:
    reason: str
    blocks_requested: int
    steps_executed: int


def run_wam_segment(task_env: Any, client: Any, *, max_blocks: int,
                    task: str | None = None, seed: int = 0,
                    should_yield: Callable[[], bool] | None = None) -> SegmentResult:
    """Execute at most max_blocks WAM chunks, retaining history within this call.

    Unlike the request-only examples, this function calls take_action(). The
    caller supplies an initialized single environment and an open client, and
    must not run another controller concurrently. Every invocation starts a
    fresh WAM segment; the environment itself is never reset.

    should_yield is checked before each request and action. Yielding discards
    unexecuted actions. Exceptions propagate without retries; start a new
    segment after recovery (and reconnect after transport failures).
    """
    if isinstance(max_blocks, bool) or not isinstance(max_blocks, int) or max_blocks < 1:
        raise ValueError("max_blocks must be a positive integer")
    blocks, steps = 0, 0

    def stop_reason() -> str | None:
        if task_env.is_episode_end():
            return "episode_end"
        if should_yield is not None and should_yield():
            return "yield_to_agent"
        return None

    reason = stop_reason()
    if reason:
        return SegmentResult(reason, blocks, steps)
    anchor = capture_frame(task_env.get_obs())
    frames = [anchor]
    for block in range(max_blocks):
        reason = stop_reason()
        if reason:
            return SegmentResult(reason, blocks, steps)
        images, state = build_inputs(frames)
        if block == 0:
            reply = client.predict_current(images, state, instruction=anchor.instruction,
                                           task=task, seed=seed, source_color_order="rgb")
        else:
            reply = client.predict(images, state)
        blocks += 1
        actions = reply["result"]["action"]
        if actions.shape != (24, 14):
            raise ValueError("Expected a complete [24,14] action chunk")
        # Validate all targets before executing any of the returned chunk.
        targets = [action_to_robodojo(action) for action in actions]
        frames = []
        for target in targets:
            reason = stop_reason()
            if reason:
                return SegmentResult(reason, blocks, steps)
            task_env.take_action(target)
            steps += 1
            reason = stop_reason()
            if reason:
                return SegmentResult(reason, blocks, steps)
            frame = capture_frame(task_env.get_obs())
            if frame.env_idx != anchor.env_idx or frame.instruction != anchor.instruction:
                raise ValueError("Environment or instruction changed during the WAM segment")
            frames.append(frame)
    return SegmentResult("block_limit", blocks, steps)
