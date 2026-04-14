# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import TYPE_CHECKING, Any, Dict, Literal, Optional

import torch
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedModel,
)
from lerobot.configs.policies import PreTrainedConfig
from ..distributed.parallel_state import get_parallel_state
from ..utils import logging
from .loader import BaseModelLoader, get_loader

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

logger = logging.get_logger(__name__)


def build_tokenizer(tokenizer_path: str) -> "PreTrainedTokenizer":
    """
    Builds the tokenizer.
    """
    return AutoTokenizer.from_pretrained(tokenizer_path, padding_side="right", trust_remote_code=True)


def build_processor(processor_path: str) -> "ProcessorMixin":
    """
    Builds the processor.
    """
    return AutoProcessor.from_pretrained(processor_path, padding_side="right", trust_remote_code=True)


def build_foundation_model(
    config_path: str,
    weights_path: Optional[str] = None,
    torch_dtype: Literal["float16", "bfloat16", "float32"] = "bfloat16",
    attn_implementation: Optional[Literal["eager", "sdpa", "flash_attention_2", "flex"]] = "flash_attention_2",
    moe_implementation: Optional[Literal["eager", "fused"]] = None,
    init_device: Literal["cpu", "cuda", "meta"] = "cuda",
    freeze_vision_encoder: Optional[bool] = False,
    tokenizer_max_length: Optional[int] = 48,
    vocab_size: Optional[int] = 0,
    use_lm_head: Optional[bool] = False,
    config_kwargs: Optional[Dict[str, Any]] = None,
    force_use_huggingface: bool = False,
) -> "PreTrainedModel":
    """
    Builds the foundation model.

    If weights_path is provided, it loads the pre-trained weights, otherwise it initializes weights.
    """
    if config_kwargs is None:
        config_kwargs = {}
    vlm_repo_id = config_kwargs['vlm_repo_id'] if 'vlm_repo_id' in config_kwargs else None
    expert_vision_path = config_kwargs['expert_vision_path'] if 'expert_vision_path' in config_kwargs else None
    tokenizer_path = config_kwargs['tokenizer_path'] if 'tokenizer_path' in config_kwargs else None
    post_training = config_kwargs['post_training']
    adanorm_time = config_kwargs['adanorm_time']
    assert not (config_kwargs['split_gate_liner'] and config_kwargs['nosplit_gate_liner']), 'split_gate_liner and nosplit_gate_liner can not be both True.'
    enable_expert_vision = config_kwargs['enable_expert_vision']
    incremental_training = config_kwargs['incremental_training']
    depth_incremental_training = config_kwargs['depth_incremental_training']
    norm_qkv =  config_kwargs['norm_qkv']
    loss_type = config_kwargs['loss_type']
    config = PreTrainedConfig.from_pretrained(config_path)
    config.train_state_proj = True
    config.adanorm_time = adanorm_time
    config.split_gate_liner = config_kwargs['split_gate_liner']
    config.nosplit_gate_liner = config_kwargs['nosplit_gate_liner']
    config.separate_time_proj = config_kwargs['separate_time_proj']
    config.old_adanorm = config_kwargs['old_adanorm']
    config.final_norm_adanorm = config_kwargs['final_norm_adanorm']
    config.freeze_vision_encoder = freeze_vision_encoder
    config.tokenizer_max_length = tokenizer_max_length
    config.attention_implementation = 'flex' # TODO
    config.enable_expert_vision = config_kwargs['enable_expert_vision']
    config.expert_vision_type = config_kwargs['expert_vision_type']
    config.action_dim = config_kwargs['action_dim']
    config.max_action_dim = config_kwargs['max_action_dim']
    config.max_state_dim = config_kwargs['max_state_dim']
    config.n_action_steps = config_kwargs['chunk_size']
    config.vlm_repo_id = vlm_repo_id
    config.expert_vision_path = expert_vision_path
    config.tokenizer_path = tokenizer_path
    config.loss_type = loss_type
    config.align_params = config_kwargs['align_params']
    config.norm_qkv = config_kwargs['norm_qkv']
    config.use_lm_head = use_lm_head
    if vocab_size == 0:
        if vlm_repo_id and 'paligemma' in vlm_repo_id:
            config.vocab_size = 257216
        # elif vlm_repo_id and 'qwen' in vlm_repo_id.lower() and 'fast' in vlm_repo_id.lower():
        #     config.vocab_size = 153715
        elif vlm_repo_id and 'qwen' in vlm_repo_id.lower():
            config.vocab_size = 151936
        else:
            config.vocab_size = 257152
    else:
        config.vocab_size = vocab_size

    if moe_implementation is not None:
        if moe_implementation not in ["eager", "fused"]:
            raise ValueError(f"Invalid moe_implementation: {moe_implementation}")
        config._moe_implementation = moe_implementation
        logger.info_rank0(f"Moe implementation: {moe_implementation}")

    loader: Optional[BaseModelLoader] = get_loader(config, force_use_huggingface)
    init_kwargs = {
        "config": config,
        "torch_dtype": getattr(torch, torch_dtype),
        "attn_implementation": attn_implementation,
        "ckpt_path": weights_path,
        "trust_remote_code": True,
    }

    if (init_device == "cpu" and get_parallel_state().global_rank != 0) or init_device == "meta":
        empty_init = True
    else:
        empty_init = False
    weights_path = vlm_repo_id if vlm_repo_id else weights_path
    model = loader.load_model(
        init_kwargs=init_kwargs,
        weights_path=weights_path,
        empty_init=empty_init,
        init_device=init_device,
        vlm_repo_id=vlm_repo_id,
        expert_vision_path=expert_vision_path,
        post_training=post_training,
        adanorm_time=adanorm_time,
        incremental_training=incremental_training,
        depth_incremental_training=depth_incremental_training,
        norm_qkv=norm_qkv,
        enable_expert_vision=enable_expert_vision,
    )
    return model
