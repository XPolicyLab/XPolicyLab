"""Small, dependency-free definitions shared by training and inference."""

import math
import re

UPSTREAM_COMMIT = "fbab441b63c789e5c37f7293e61fea9ed356c6c5"
BASE_REVISION = "main"
POLICY_REVISION = "main"
SOURCE_REVISION = "main"
IMAGE_KEYS = ["images_1", "images_2", "images_3"]
IMAGE_PROMPTS = ["Head", "Left wrist", "Right wrist"]
CAMERAS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
RELEASE_STEPS = 100_000
WARMUP_STEPS = 1_000
BASE_LR = 5e-5
FINAL_LR = 2.5e-5
MIN_LR_RATE = FINAL_LR / BASE_LR


def accumulation_steps(world_size, per_device_batch, global_batch=1024):
    micro_batch = world_size * per_device_batch
    if global_batch <= 0 or micro_batch <= 0 or global_batch % micro_batch:
        raise ValueError("global batch must be a positive multiple of GPUs × per-device batch")
    return global_batch // micro_batch


def scheduler_kwargs():
    return {"min_lr_rate": MIN_LR_RATE}


def learning_rate(step, total_steps=RELEASE_STEPS):
    if step < WARMUP_STEPS:
        return BASE_LR * step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
    return BASE_LR * (MIN_LR_RATE + (1 - MIN_LR_RATE) * (1 + math.cos(math.pi * progress)) / 2)


def task_text(prompt):
    text = str(prompt).strip()
    if text and text[-1] not in ".!?。！？…":
        text += "."
    if text and not text.lower().startswith(("task:", "subtask:")):
        text = "Task: " + text
    return text


class CheckpointPromptProcessor:
    """Keep released task punctuation/prefixes with upstream tokenization.

    Upstream wraps every prompt as ``Task: {prompt}.``. Preserve the released
    policy's already-punctuated / Task-prefixed instructions outside its source.
    """

    def __init__(self, processor):
        self.processor = processor

    def __getattr__(self, name):
        return getattr(self.processor, name)

    def apply_chat_template(self, messages, **kwargs):
        messages = [
            {**message, "content": [dict(part) for part in message["content"]]}
            for message in messages
        ]
        content = messages[0]["content"][0]
        content["text"] = re.sub(
            r"(?ms)^Task: (.*?)\.\n(?=History images:)",
            lambda match: task_text(match.group(1)) + "\n",
            content["text"],
        )
        return self.processor.apply_chat_template(messages, **kwargs)
