from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from worldscape_policy.data.schema import EventSample, VisualPromptMetadata


@dataclass(frozen=True)
class LegacyContextAdapter:
    """Translate the old context encoding into explicit visual prompts.

    This is deliberately the only native data component that understands
    ``context_sampling_mode``, ``video_ctx`` or ``is_exec``.
    """

    context_key: str = "video_ctx"
    history_key: str = "history_images"
    context_sampling_mode: str | None = None

    def adapt(
        self,
        record: Mapping[str, Any],
        *,
        context_sampling_mode: str | None = None,
    ) -> EventSample:
        context_sampling_mode = (
            context_sampling_mode
            or self.context_sampling_mode
            or record.get("context_sampling_mode")
        )
        if not isinstance(context_sampling_mode, str):
            raise TypeError("legacy context_sampling_mode must be a string")
        mode = context_sampling_mode.strip().lower()
        if mode not in {"uniform", "last"}:
            raise ValueError(
                "context_sampling_mode must be 'uniform' or 'last', "
                f"got {context_sampling_mode!r}"
            )
        context = record.get(self.context_key)
        if context is None:
            raise ValueError(f"legacy record is missing {self.context_key!r}")
        context_array = np.asarray(context)
        if context_array.ndim < 4 or context_array.shape[0] == 0:
            raise ValueError("legacy context must be a non-empty image sequence")

        goal_image = None
        demo_video = None
        if mode == "uniform":
            demo_video = context_array.copy()
        else:
            goal_image = context_array[-1].copy()

        observations = record.get("observations")
        if observations is None:
            video = record.get("video")
            if video is None:
                raise ValueError("legacy record needs 'observations' or 'video'")
            observations = {"video": np.asarray(video)}
        else:
            observations = {
                str(name): np.asarray(value) for name, value in observations.items()
            }

        episode_id = str(record.get("episode_id", record.get("trajectory_id", "")))
        session_id = str(record.get("session_id", episode_id))
        high_level_instruction = _optional_string(
            record.get("high_level_instruction", record.get("language"))
        )
        event_instruction = _optional_string(
            record.get("event_instruction", record.get("subtask"))
        )
        task_id = str(
            record.get(
                "task_id",
                record.get(
                    "task_index",
                    high_level_instruction
                    or event_instruction
                    or f"episode:{episode_id}",
                ),
            )
        )
        prompt_metadata = VisualPromptMetadata(
            task_id=task_id,
            embodiment=str(record.get("embodiment", record.get("embodiment_tag", ""))),
            source_episode_id=episode_id,
            source_session_id=session_id,
            trusted_same_sample=True,
        )
        sample = EventSample(
            episode_id=episode_id,
            event_id=str(record.get("event_id", record.get("chunk_id", ""))),
            observations=observations,
            actions=np.asarray(record.get("actions", record.get("action"))),
            robot_state=np.asarray(record.get("robot_state", record.get("state"))),
            high_level_instruction=high_level_instruction,
            event_instruction=event_instruction,
            goal_image=goal_image,
            demo_video=demo_video,
            history_head_frames=_optional_array(record.get(self.history_key)),
            embodiment=prompt_metadata.embodiment,
            task_id=task_id,
            session_id=session_id,
            goal_prompt_metadata=prompt_metadata if goal_image is not None else None,
            demo_prompt_metadata=prompt_metadata if demo_video is not None else None,
        )
        sample.validate()
        return sample

    def __call__(
        self,
        record: Mapping[str, Any],
        *,
        context_sampling_mode: str | None = None,
    ) -> EventSample:
        return self.adapt(record, context_sampling_mode=context_sampling_mode)


def _optional_array(value: object) -> np.ndarray | None:
    return None if value is None else np.asarray(value)


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        value = value.reshape(-1)[0]
    return str(value)
