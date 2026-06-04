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

import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.nn.functional import cross_entropy
from transformers import (
    AutoConfig,
    AutoModel,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from robo_orchard_lab.models.monodream.configuration_llava import LlavaConfig
from robo_orchard_lab.models.monodream.navigation_arch import (
    LlavaMetaForCausalLM,
    LlavaMetaModel,
)
from robo_orchard_lab.models.monodream.utils.constants import IGNORE_INDEX


class LlavaLlamaConfig(LlavaConfig):
    """Language model config.

    Initialized from LlavaConfig.
    """

    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlavaMetaForCausalLM, PreTrainedModel):
    """Language model define.

    Inherits from LlavaMetaModel and LlavaMetaForCausalLM.
    Main function is to handle multimodal input and forward pass.
    """

    config_class = LlavaLlamaConfig
    main_input_name = "input_embeds"
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True

    def __init__(
        self, config: LlavaLlamaConfig = None, preload=True, *args, **kwargs
    ) -> None:
        super().__init__(config)
        if preload:
            self.init_vlm(config=config)
        else:
            pass

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: bool = None,
        **kwargs,
    ):
        """Load a pretrained LlavaLlamaModel from a pre-trained model.

        Inherits from PreTrainedModel.
        """

        if hasattr(cls, "load_pretrained"):
            return cls.load_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                config=config,
                cache_dir=cache_dir,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                use_safetensors=use_safetensors,
                **kwargs,
            )
        return super(LlavaLlamaModel).from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            use_safetensors=use_safetensors,
            **kwargs,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        media: Optional[Dict[str, List[torch.Tensor]]] = None,
        images: Optional[torch.FloatTensor] = None,
        media_config: Optional[List] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """Main forward pass of MonoDream.

        Main function is to handle multimodal input and forward pass.
        """
        self.freezed_module_patch()

        if images is not None:
            media = {"image": images}

        if media_config is None:
            media_config = defaultdict(dict)

        is_image = []
        loss_begin_index = []
        if self.training:
            batch_size = labels.size(0)
            for i in range(batch_size):
                label_i = labels[i]
                input_i = input_ids[i]
                if (label_i == 151654).any():
                    is_image.append(1)
                    image_indices = (
                        torch.nonzero(input_i == 151654, as_tuple=False)
                        .squeeze()
                        .tolist()
                    )
                    label_indices = (
                        torch.nonzero(label_i != -100, as_tuple=False)
                        .squeeze()
                        .tolist()
                    )
                    loss_begin_index.append(
                        label_indices[0] - (len(image_indices) - 1)
                    )
                else:
                    is_image.append(0)
                    loss_begin_index.append(-1)

        if inputs_embeds is None:
            inputs_embeds, labels, attention_mask = self._embed(
                input_ids, media, media_config, labels, attention_mask
            )

        outputs = self.llm.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=None,
            output_hidden_states=True,
        )

        hidden_states = outputs[0]
        logits = self.llm.lm_head(hidden_states)

        loss = 0.0
        if self.training:
            batch_size = labels.size(0)
            loss = 0.0
            for i in range(batch_size):
                input_i = input_ids[i]
                label_i = labels[i]
                hs_i = hidden_states[i]
                embed_i = inputs_embeds[i]
                label_index = loss_begin_index[i]
                image_token_indices = torch.nonzero(
                    (label_i == 151649) | (label_i == 151654), as_tuple=True
                )[0]

                if label_index != -1:
                    gt_all_image_tokens = embed_i[image_token_indices]
                    image_token_indices = image_token_indices + 1
                    pred_all_image_tokens = hs_i[image_token_indices]
                    loss_i = (
                        F.mse_loss(pred_all_image_tokens, gt_all_image_tokens)
                        * 0.1
                    )
                else:
                    logits_i = logits[i].unsqueeze(0)
                    labels_i = label_i.unsqueeze(0)

                    loss_i = soft_cross_entropy(
                        logits_i,
                        labels_i,
                        soft_tokens=[],
                        std=self.config.soft_ce_std,
                    )

                    loss += loss_i

        loss = loss / labels.size(0)
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )


def soft_cross_entropy(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    soft_tokens: Union[torch.Tensor, List[int]],
    std: float = 1,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    # Remove last token from outputs and first token from targets
    outputs = outputs[..., :-1, :].contiguous()
    targets = targets[..., 1:].contiguous()

    # Flatten outputs and targets
    targets = targets.view(-1)
    outputs = outputs.view(targets.size(0), -1)

    # Remove outputs and targets with ignore_index
    indices = targets != ignore_index
    outputs = outputs[indices]
    targets = targets[indices]

    # Convert soft token IDs to tensor
    if isinstance(soft_tokens, list):
        soft_tokens = torch.tensor(soft_tokens).to(targets)

    # Calculate loss for non-soft tokens
    indices = torch.isin(targets, soft_tokens, invert=True)
    loss = cross_entropy(outputs[indices], targets[indices], reduction="sum")

    # Calculate loss for soft tokens
    indices = torch.isin(targets, soft_tokens)
    targets_indices = torch.zeros_like(outputs[indices])
    for k, target in enumerate(targets[indices]):
        dist = torch.exp(-((target - soft_tokens) ** 2) / (2 * std**2))
        targets_indices[k][soft_tokens] = dist / dist.sum()
    loss += cross_entropy(outputs[indices], targets_indices, reduction="sum")

    # Return average loss
    return loss / targets.size(0)


AutoConfig.register("llava_llama", LlavaLlamaConfig)
AutoModel.register(LlavaLlamaConfig, LlavaLlamaModel)
