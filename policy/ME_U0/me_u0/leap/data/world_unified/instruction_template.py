"""Lance chat-template text tokenization for MachEmbodiedUnifiedModel (shared train<->serve).

Lance's understanding expert was pretrained on the Qwen chat template
(``<|im_start|>system\\n{sys}<|im_end|>\\n<|im_start|>user\\n{instr}<|im_end|>\\n``)
plus a task-type system prompt (``data/common.generate_system_prompt``). Our v1
port previously fed a BARE ``tokenizer(instruction)``, which is out-of-distribution
for the FROZEN understanding expert. This helper reproduces Lance's wrapping so
the frozen expert sees in-distribution text. Used by BOTH the dataset
(:mod:`robot_policy_dataset`) and the serving policy (:mod:`leap.serving.ME_U0_policy`)
so train and inference are byte-identical.
"""

from __future__ import annotations

import torch

from leap.data.world_unified.metadata import Metadata, render_instruction

# Lance t2v/i2v system prompt (data/common.generate_system_prompt("t2v")). The
# understanding expert saw this exact structure during video generation training.
WORLD_UNIFIED_T2V_SYSTEM_PROMPT = (
    "Describe the video by detailing the color, quantity, visible text, shape, "
    "size, texture, spatial relationships and motion/camera movements of the "
    "objects and background:"
)


def build_instruction_text_ids(
    tokenizer,
    instruction: str,
    max_text_len: int = 512,
    system_prompt: str = WORLD_UNIFIED_T2V_SYSTEM_PROMPT,
    metadata: Metadata | None = None,
) -> torch.Tensor:
    """Return chat-template-wrapped ``input_ids`` (1-D long tensor).

    Layout (matches Lance ``system_prompt_render`` + ``dataset_base_train``)::

        <|im_start|>system\\n{system_prompt}<|im_end|>\\n
        <|im_start|>user\\n{instruction}<|im_end|>\\n

    The instruction body is truncated to ``max_text_len`` tokens; the wrapper
    tokens are always kept so the sequence is never empty / never a degenerate
    0-length split. Default 512 is effectively no-truncation for robot
    instructions (native Lance does not truncate); it is only a safety ceiling.

    The shared renderer adds the task and dataset action representation.
    """
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")

    def enc(text: str, limit: int) -> list:
        return tokenizer(text, add_special_tokens=False, truncation=True,
                         max_length=limit)["input_ids"]

    sys_ids = enc("system\n" + system_prompt, max_text_len)
    instruction = render_instruction(instruction, metadata)
    usr_ids = enc("user\n" + instruction, max_text_len)
    ids = [im_start] + sys_ids + [im_end] + [im_start] + usr_ids + [im_end]
    return torch.tensor(ids, dtype=torch.long)
