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

"""Drive one RoboRSI Agent episode with EmbodiedAgent's deferred tool loop."""

from __future__ import annotations

import base64
import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jsonschema import Draft202012Validator

from roborsi.evaluation.media import CaseMediaChannel
from roborsi.evaluation.messages import Action, Observation, StepResult, Task
from roborsi.tools.toolkit import ToolResult
from roborsi.utils.logging import get_logger

if TYPE_CHECKING:
    from roborsi.data_convert import TextDocument
    from roborsi.embodied_agent import (
        EmbodiedAgent,
        EmbodiedEpisode,
        LocalToolSpec,
        PendingToolCall,
    )

logger = get_logger("evaluation.embodied_bridge")


def roborsi_action_tool(
    schema: dict[str, Any], *, name: str = "roborsi_action"
) -> LocalToolSpec:
    """Declare one deferred tool matching the Benchmark's native action schema."""
    from roborsi.embodied_agent import LocalToolSpec

    return LocalToolSpec(
        name=name,
        description="Submit one native action for the current observation.",
        input_schema=schema,
    )


class _EmbodiedToolConnector:
    """Connect a persistent EmbodiedAgent dialogue to the evaluation protocol.

    The Agent runs in its own service process. It receives observations and
    proposes actions; only the Benchmark Adapter executes and scores them.
    """

    def __init__(
        self,
        planner: EmbodiedAgent | Callable[[Task], EmbodiedAgent],
        *,
        media_dir: str | Path,
        system_prompt: str,
        output_dir: str | Path,
        action_tool_name: str = "roborsi_action",
        skills: Sequence[str | Path] = (),
        skill_paths: Sequence[str | Path] = (),
        memory: Sequence[TextDocument] = (),
        initial_context: Sequence[Any] = (),
        action_timeout_s: float | None = 120.0,
        cleanup_timeout_s: float | None = None,
    ) -> None:
        if not system_prompt.strip():
            raise ValueError("system_prompt must be non-empty")
        if action_timeout_s is not None and (
            not math.isfinite(action_timeout_s) or action_timeout_s <= 0
        ):
            raise ValueError("action_timeout_s must be positive")
        if cleanup_timeout_s is not None and (
            not math.isfinite(cleanup_timeout_s) or cleanup_timeout_s <= 0
        ):
            raise ValueError("cleanup_timeout_s must be positive")
        self._planner_source = planner
        self._planner: EmbodiedAgent | None = None
        self._media = CaseMediaChannel(media_dir)
        self._system_prompt = system_prompt
        self._output_dir = Path(output_dir)
        self._action_tool_name = action_tool_name
        self._skills = tuple(skills)
        self._skill_paths = tuple(skill_paths)
        self._memory = tuple(memory)
        self._initial_context = tuple(initial_context)
        self._action_timeout_s = action_timeout_s
        self._close_timeout_s = cleanup_timeout_s
        self._cleanup_timeout_s = (
            cleanup_timeout_s
            if cleanup_timeout_s is not None
            else action_timeout_s
            if action_timeout_s is not None
            else 30.0
        )
        self._task: Task | None = None
        self._episode: EmbodiedEpisode | None = None
        self._pending: PendingToolCall | None = None
        self._episode_id: str | None = None
        self._last_step = -1
        self._closed = False
        self._ended = False

    def reset(self, task: Task) -> None:
        """Bind the task; the first observation starts the model dialogue."""
        if self._task is not None or self._closed:
            raise ValueError("Embodied RoboRSI Agent owns only one episode")
        schema = task.capabilities.get("action_schema")
        if not isinstance(schema, dict):
            raise ValueError("RoboRSI task needs an action_schema")
        Draft202012Validator.check_schema(schema)
        planner = (
            self._planner_source(task)
            if callable(self._planner_source)
            else self._planner_source
        )
        matching = [
            tool for tool in planner.local_tools if tool.name == self._action_tool_name
        ]
        if (
            len(matching) != 1
            or matching[0].handler is not None
            or matching[0].input_schema != schema
        ):
            raise ValueError("Embodied Agent needs a matching deferred action tool")
        self._planner = planner
        self._task = task

    def _context(self, observation: Observation) -> list[Any]:
        document = {
            "step_index": observation.step_index,
            "observation": observation.data,
            "media": [
                {"index": index, "mime_type": reference.mime_type}
                for index, reference in enumerate(observation.media)
            ],
        }
        parts: list[Any] = [json.dumps(document, ensure_ascii=False, allow_nan=False)]
        if observation.media:
            from pydantic_ai import BinaryContent

            for reference in observation.media:
                parts.append(
                    BinaryContent(
                        data=self._media.read(
                            reference, observation.episode_id, observation.step_index
                        ),
                        media_type=reference.mime_type,
                    )
                )
        return parts

    def _feedback(self, observation: Observation) -> ToolResult:
        assert self._pending is not None
        parts = self._context(observation)
        result = ToolResult(
            name=self._pending.name,
            result={
                "step_index": observation.step_index,
                "observation": observation.data,
            },
        )
        result.content_blocks = [{"type": "text", "text": parts[0]}]
        for part in parts[1:]:
            result.content_blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": part.media_type,
                        "data": base64.b64encode(part.data).decode("ascii"),
                    },
                }
            )
        return result

    def act(self, observation: Observation) -> Action:
        """Return the next deferred native action after feeding the observation."""
        if self._task is None or self._closed or self._ended:
            raise ValueError("Embodied RoboRSI Agent has no active task")
        if observation.benchmark != self._task.benchmark:
            raise ValueError("observation Benchmark differs from task")
        if observation.step_index != self._last_step + 1:
            raise ValueError("Embodied RoboRSI observation step is out of order")
        if self._episode_id is not None and observation.episode_id != self._episode_id:
            raise ValueError("Embodied RoboRSI episode identity mismatch")
        if self._episode is None:
            assert self._planner is not None
            self._episode_id = observation.episode_id
            self._episode = self._planner.start_episode(
                self._task.instruction,
                system_prompt=self._system_prompt,
                deferred_tools=[self._action_tool_name],
                skills=self._skills,
                skill_paths=self._skill_paths,
                memory=self._memory,
                initial_context=[*self._initial_context, *self._context(observation)],
                output_dir=self._output_dir,
                include_finish=False,
            )
        else:
            if self._pending is None:
                raise RuntimeError("Embodied RoboRSI Agent has no pending action")
            self._episode.complete(self._pending, self._feedback(observation))
            self._pending = None
        call = self._episode.next_call(timeout=self._action_timeout_s)
        if call is None:
            result = self._episode.wait(timeout=self._action_timeout_s)
            if result.error:
                raise RuntimeError(f"Embodied Agent failed: {result.error}")
            raise RuntimeError("Embodied Agent ended before the Benchmark episode")
        if call.name != self._action_tool_name:
            raise ValueError("Embodied Agent submitted an unexpected tool call")
        schema = self._task.capabilities["action_schema"]
        Draft202012Validator(schema).validate(call.arguments)
        action = Action(
            benchmark=observation.benchmark,
            episode_id=observation.episode_id,
            step_index=observation.step_index,
            data=call.arguments,
        )
        self._pending = call
        self._last_step = observation.step_index
        return action

    def observe_end(self, step: StepResult) -> None:
        """Complete the final action with native feedback without another action."""
        observation = step.observation
        if (
            self._task is None
            or self._closed
            or self._ended
            or self._episode is None
            or self._pending is None
        ):
            raise ValueError("Embodied Agent has no pending action")
        if not (step.terminated or step.truncated):
            raise ValueError("final feedback requires an ended Benchmark episode")
        if (
            observation.benchmark != self._task.benchmark
            or observation.episode_id != self._episode_id
            or observation.step_index != self._last_step + 1
        ):
            raise ValueError("final feedback does not match the pending action")
        feedback = self._feedback(observation)
        feedback.result["media"] = json.loads(feedback.content_blocks[0]["text"])[
            "media"
        ]
        feedback.result.update(
            _finish=True,
            terminated=step.terminated,
            truncated=step.truncated,
            termination_reason=step.termination_reason,
            reward=step.reward,
            native=step.native,
        )
        feedback.is_finish = True
        feedback.content_blocks[0] = {
            "type": "text",
            "text": json.dumps(feedback.result, ensure_ascii=False, allow_nan=False),
        }
        self._episode.complete(self._pending, feedback)
        self._pending = None
        self._last_step = observation.step_index
        self._ended = True
        # A successful acknowledgement means the planner recorded the final
        # tool response and released its connections before service shutdown.
        result = self._episode.wait(timeout=self._cleanup_timeout_s)
        if result.error:
            raise RuntimeError(f"Embodied Agent failed: {result.error}")

    def close(self) -> None:
        """Finish the last pending tool call and release planner resources."""
        if self._closed:
            # A previous close may have timed out while a physical handler was
            # finishing. Retrying must still wait for its resource cleanup.
            if self._episode is not None:
                self._episode.close(
                    **(
                        {"timeout": self._close_timeout_s}
                        if self._close_timeout_s is not None
                        else {}
                    )
                )
            return
        self._closed = True
        if self._episode is None:
            return
        try:
            if self._pending is not None:
                try:
                    self._episode.complete(
                        self._pending,
                        ToolResult(
                            self._pending.name,
                            {"_finish": True, "status": "episode_ended"},
                        ),
                    )
                except (RuntimeError, ValueError) as exc:
                    logger.warning(
                        "Embodied RoboRSI final tool response failed: %s", exc
                    )
        finally:
            self._pending = None
            self._episode.close(
                **(
                    {"timeout": self._close_timeout_s}
                    if self._close_timeout_s is not None
                    else {}
                )
            )
