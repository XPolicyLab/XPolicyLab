"""Task prompts with optional embodiment, control-rate and action-frame headers."""

from __future__ import annotations

import random
from dataclasses import dataclass

PROMPT_FIELDS = ("embodiment", "fps", "action", "frame")


@dataclass(frozen=True)
class PromptSpec:
    embodiment: str
    fps: float
    action_type: str
    reference_frame: str
    instruction: str


def build_prompt(
    spec: PromptSpec,
    *,
    dropout: float = 0.0,
    rng: random.Random | None = None,
    header: bool = False,
) -> str:
    """Render the prompt.

    ``header=False`` (from-scratch runs): the task instruction only. ``header=True`` (finetuning the
    MM-ABC pretrain): the ``embodiment | fps | action | frame`` header the
    pretrained backbone was trained with, each field dropped with ``dropout``.
    """
    instruction = spec.instruction.strip() or "perform the demonstrated task"
    if not header:
        return f"task: {instruction}"
    r = rng or random
    values = {
        "embodiment": spec.embodiment,
        "fps": f"{spec.fps:g}",
        "action": spec.action_type,
        "frame": spec.reference_frame,
    }
    parts = [f"{f}: {values[f]}" for f in PROMPT_FIELDS if not (dropout > 0.0 and r.random() < dropout)]
    joined = " | ".join(parts)
    return f"{joined}\ntask: {instruction}" if joined else f"task: {instruction}"


def wrap_chat(prompt: str, processor) -> str:
    """Apply the backbone's chat template, with image placeholders left to the caller.

    Kept separate from :func:`build_prompt` so statistics passes and tests can
    build prompts without loading a tokenizer.
    """
    messages = [{"role": "user", "content": prompt}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
