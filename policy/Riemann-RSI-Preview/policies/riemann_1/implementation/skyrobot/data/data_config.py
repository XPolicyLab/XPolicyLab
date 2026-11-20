"""Checkpoint-compatible inference constants and the original prompt format."""
from skyrobot.data.robot_config.base_config import (
    base_data_config, EmbodimentId, Embodiment_DIM, MAX_ACTION_DIM, MAX_STATE_DIM,
)

DEFAULT_PROMPT = (
    "Video recorded from embodiment {embodiment}, showing the {views} camera views, "
    "while performing the task: {task}"
)
