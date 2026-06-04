# Project RoboOrchard
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

from typing import Any, Dict, List, Optional, Sequence

import torch
import transformers

from robo_orchard_lab.models.monodream.multimodal_encoder.mm_utils import (
    tokenizer_image_token,
)
from robo_orchard_lab.models.monodream.utils.constants import (
    IGNORE_INDEX,
    SENTINEL_TOKEN,
)

__all__ = [
    "tokenize_conversation",
    "preprocess_conversation",
    "infer_stop_tokens",
]

DUMMY_CONVERSATION = [
    {"from": "human", "value": "question"},
    {"from": "gpt", "value": "answer"},
] * 10


def tokenize_conversation(
    messages: Sequence[Dict[str, str]],
    tokenizer: transformers.PreTrainedTokenizer,
    add_generation_prompt: bool = False,
    overrides: Optional[Dict[str, str]] = None,
    no_system_prompt: bool = False,
) -> torch.Tensor:
    # Normalize the conversation before tokenization
    for message in messages:
        message["value"] = message["value"].strip()

    conversation = []
    for m in messages:
        message = {}
        if m["from"] == "human":
            message["role"] = "user"
        elif m["from"] == "gpt":
            message["role"] = "assistant"
        else:
            raise ValueError(
                f"Unexpected sender '{m['from']}' in conversation entry."
            )

        message["content"] = m["value"]
        if overrides is not None and m["from"] in overrides:
            message["content"] = overrides[m["from"]]
        conversation.append(message)

    if no_system_prompt:
        conversation = [{"role": "system", "content": None}] + conversation

    text = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
    )
    return tokenizer_image_token(text, tokenizer, return_tensors="pt")


def _maybe_add_sentinel_token(
    tokenizer: transformers.PreTrainedTokenizer,
) -> None:
    if not hasattr(tokenizer, "sentinel_token"):
        tokenizer.add_tokens([SENTINEL_TOKEN], special_tokens=True)
        tokenizer.sentinel_token = SENTINEL_TOKEN
        tokenizer.sentinel_token_id = tokenizer.convert_tokens_to_ids(
            SENTINEL_TOKEN
        )


def preprocess_conversation(
    conversation: Sequence[Dict[str, str]],
    tokenizer: transformers.PreTrainedTokenizer,
    no_system_prompt: bool = False,
    retried: bool = False,
) -> Dict[str, Any]:
    inputs = tokenize_conversation(
        conversation, tokenizer, no_system_prompt=no_system_prompt
    )
    labels = torch.ones_like(inputs) * IGNORE_INDEX

    # Generate the template by replacing the assistant's response
    # with a sentinel.
    _maybe_add_sentinel_token(tokenizer)
    template = tokenize_conversation(
        conversation,
        tokenizer,
        overrides={"gpt": SENTINEL_TOKEN},
        no_system_prompt=no_system_prompt,
    )

    # Remove sentinel tokens from the template.
    mask = torch.ones_like(template, dtype=torch.bool)
    for k in range(template.size(0) - 1):
        if template[k] == tokenizer.sentinel_token_id:
            mask[k : k + 2] = False
            if k > 0 and retried:
                mask[k - 1] = False
    template = template[mask]

    # Match the tokenized conversation with the template.
    # Every token that is not matched will be included in the label.
    p = 0
    for k in range(inputs.size(0)):
        if p < template.size(0) and inputs[k] == template[p]:
            p += 1
        else:
            labels[k] = inputs[k]

    # Mask all tokens in the label if the template is not fully matched.
    if p < template.size(0):
        if not retried:
            return preprocess_conversation(
                conversation,
                tokenizer,
                no_system_prompt=no_system_prompt,
                retried=True,
            )
        print(
            f"Failed to process the conversation: '{conversation}'.",
            "All tokens will be masked in the label.",
        )
        labels[:] = IGNORE_INDEX

    return {"input_ids": inputs, "labels": labels}


def infer_stop_tokens(
    tokenizer: transformers.PreTrainedTokenizer,
) -> List[str]:
    _maybe_add_sentinel_token(tokenizer)
    template = tokenize_conversation(
        DUMMY_CONVERSATION, tokenizer, overrides={"gpt": SENTINEL_TOKEN}
    )

    stop_tokens = {tokenizer.eos_token}
    for k in range(template.size(0) - 1):
        if template[k] == tokenizer.sentinel_token_id:
            stop_token = tokenizer.decode(template[k + 1])
            stop_tokens.add(stop_token)
    return list(stop_tokens)
