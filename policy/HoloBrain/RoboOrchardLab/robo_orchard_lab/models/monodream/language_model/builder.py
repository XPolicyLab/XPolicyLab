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

import math
import os
import os.path as osp
from typing import Tuple

from huggingface_hub import file_exists, repo_exists
from huggingface_hub.utils import HFValidationError
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from robo_orchard_lab.models.monodream.utils.constants import MEDIA_TOKENS
from robo_orchard_lab.models.monodream.utils.tokenizer import infer_stop_tokens


def has_tokenizer(repo_id_or_path: str) -> bool:
    # Check if the tokenizer is in a local directory
    if osp.exists(osp.join(repo_id_or_path, "tokenizer_config.json")):
        return True

    # Check if the tokenizer is in a Hugging Face Hub repo
    try:
        return repo_exists(repo_id_or_path) and file_exists(
            repo_id_or_path, "tokenizer_config.json"
        )
    except HFValidationError:
        return False


def context_length_extension(config):
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    model_max_length = getattr(config, "model_max_length", None)
    if orig_ctx_len and model_max_length > orig_ctx_len:
        print(f"Scaling RoPE from {orig_ctx_len} to {model_max_length}")
        scaling_factor = float(math.ceil(model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    return config


def build_llm_and_tokenizer(
    model_name_or_path: str,
    config: PretrainedConfig,
    attn_implementation=None,
    model_max_length=None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    llm_cfg = AutoConfig.from_pretrained(model_name_or_path)
    llm_cfg._attn_implementation = attn_implementation
    llm_cfg.model_max_length = model_max_length
    if model_max_length is not None:
        context_length_extension(llm_cfg)

    llm = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        config=llm_cfg,
        torch_dtype=eval(config.model_dtype),
    )

    # Locate and build the tokenizer
    llm_path = model_name_or_path
    if not has_tokenizer(llm_path):
        llm_path = osp.join(llm_path, "llm")
    if not has_tokenizer(llm_path):
        raise ValueError(f"Cannot find tokenizer in {llm_path}.")

    tokenizer = AutoTokenizer.from_pretrained(
        llm_path, padding_side="right", use_fast=True, legacy=False
    )
    if model_max_length is not None:
        tokenizer.model_max_length = model_max_length

    # Load chat template if specified.
    if getattr(config, "chat_template", None) is not None:
        print(f"Using chat template: {config.chat_template}")
        fpath = os.path.join(
            os.path.dirname(__file__),
            "chat_templates",
            f"{config.chat_template}.jinja",
        )
        with open(fpath) as fd:
            chat_template = fd.read()
        tokenizer.chat_template = chat_template.replace("    ", "").replace(
            "\n", ""
        )

    if (
        not hasattr(tokenizer, "chat_template")
        or tokenizer.chat_template is None
    ):
        print("✅ Injecting chat_template for low-version transformers...")
        tokenizer.chat_template = (
            "{% if messages[0]['role'] != 'system' %}"
            "{{ '<|im_start|>system\\nYou are a helpful assistant<|im_end|>\\n' }}"  # noqa: E501
            "{% endif %}"
            "{% for message in messages if message['content'] is not none %}"
            "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}"  # noqa: E501
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\\n' }}"
            "{% endif %}"
        )

    # Set stop tokens for the tokenizer
    tokenizer.stop_tokens = infer_stop_tokens(tokenizer)
    tokenizer.stop_token_ids = tokenizer.convert_tokens_to_ids(
        tokenizer.stop_tokens
    )

    # Add media tokens to the tokenizer
    tokenizer.media_tokens = MEDIA_TOKENS
    tokenizer.media_token_ids = {}
    for name, token in MEDIA_TOKENS.items():
        tokenizer.add_tokens([token], special_tokens=True)
        tokenizer.media_token_ids[name] = tokenizer.convert_tokens_to_ids(
            token
        )

    config.hidden_size = llm.config.hidden_size
    return llm, tokenizer


def init_tokenizer(llm_path, config, model_max_length=4096):
    tokenizer = AutoTokenizer.from_pretrained(
        llm_path, padding_side="right", use_fast=True, legacy=False
    )
    if model_max_length is not None:
        tokenizer.model_max_length = model_max_length

    # Load chat template if specified.
    if getattr(config, "chat_template", None) is not None:
        print(f"Using chat template: {config.chat_template}")
        fpath = os.path.join(
            os.path.dirname(__file__),
            "chat_templates",
            f"{config.chat_template}.jinja",
        )
        with open(fpath) as fd:
            chat_template = fd.read()
        tokenizer.chat_template = chat_template.replace("    ", "").replace(
            "\n", ""
        )

    if (
        not hasattr(tokenizer, "chat_template")
        or tokenizer.chat_template is None
    ):
        print("Injecting chat_template for low-version transformers")
        tokenizer.chat_template = (
            "{% if messages[0]['role'] != 'system' %}"
            "{{ '<|im_start|>system\\nYou are a helpful assistant<|im_end|>\\n' }}"  # noqa: E501
            "{% endif %}"
            "{% for message in messages if message['content'] is not none %}"
            "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}"  # noqa: E501
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\\n' }}"
            "{% endif %}"
        )

    # Set stop tokens for the tokenizer
    tokenizer.stop_tokens = infer_stop_tokens(tokenizer)
    tokenizer.stop_token_ids = tokenizer.convert_tokens_to_ids(
        tokenizer.stop_tokens
    )

    # Add media tokens to the tokenizer
    tokenizer.media_tokens = MEDIA_TOKENS
    tokenizer.media_token_ids = {}
    for name, token in MEDIA_TOKENS.items():
        tokenizer.add_tokens([token], special_tokens=True)
        tokenizer.media_token_ids[name] = tokenizer.convert_tokens_to_ids(
            token
        )

    return tokenizer
