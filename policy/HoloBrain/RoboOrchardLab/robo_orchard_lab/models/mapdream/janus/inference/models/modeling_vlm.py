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


import torch
from easydict import EasyDict
from einops import rearrange
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedModel,
)
from transformers.configuration_utils import PretrainedConfig

from robo_orchard_lab.models.mapdream.janus.inference.models import (
    clip_encoder,
    vq_model,
)
from robo_orchard_lab.models.mapdream.janus.inference.models.projector import (
    MlpProjector,
)

CLIPVisionTower = clip_encoder.CLIPVisionTower
VQ_models = vq_model.VQ_models


class VisionHead(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.output_mlp_projector = torch.nn.Linear(
            params.n_embed, params.image_token_embed
        )
        self.vision_activation = torch.nn.GELU()
        self.vision_head = torch.nn.Linear(
            params.image_token_embed, params.image_token_size
        )

    def forward(self, x):
        x = self.output_mlp_projector(x)
        x = self.vision_activation(x)
        x = self.vision_head(x)
        return x


def model_name_to_cls(cls_name):
    if "MlpProjector" in cls_name:
        cls = MlpProjector

    elif "CLIPVisionTower" in cls_name:
        cls = CLIPVisionTower

    elif "VQ" in cls_name:
        cls = VQ_models[cls_name]
    elif "vision_head" in cls_name:
        cls = VisionHead
    else:
        raise ValueError(f"class_name {cls_name} is invalid.")

    return cls


class VisionConfig(PretrainedConfig):
    model_type = "vision"
    cls: str = ""
    params: EasyDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = EasyDict(kwargs.get("params", {}))


class AlignerConfig(PretrainedConfig):
    model_type = "aligner"
    cls: str = ""
    params: EasyDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = EasyDict(kwargs.get("params", {}))


class GenVisionConfig(PretrainedConfig):
    model_type = "gen_vision"
    cls: str = ""
    params: EasyDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = EasyDict(kwargs.get("params", {}))


class GenAlignerConfig(PretrainedConfig):
    model_type = "gen_aligner"
    cls: str = ""
    params: EasyDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = EasyDict(kwargs.get("params", {}))


class GenHeadConfig(PretrainedConfig):
    model_type = "gen_head"
    cls: str = ""
    params: EasyDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = EasyDict(kwargs.get("params", {}))


class MultiModalityConfig(PretrainedConfig):
    model_type = "multi_modality"
    vision_config: VisionConfig
    aligner_config: AlignerConfig

    gen_vision_config: GenVisionConfig
    gen_aligner_config: GenAlignerConfig
    gen_head_config: GenHeadConfig

    language_config: LlamaConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        vision_config = kwargs.get("vision_config", {})
        self.vision_config = VisionConfig(**vision_config)

        aligner_config = kwargs.get("aligner_config", {})
        self.aligner_config = AlignerConfig(**aligner_config)

        gen_vision_config = kwargs.get("gen_vision_config", {})
        self.gen_vision_config = GenVisionConfig(**gen_vision_config)

        gen_aligner_config = kwargs.get("gen_aligner_config", {})
        self.gen_aligner_config = GenAlignerConfig(**gen_aligner_config)

        gen_head_config = kwargs.get("gen_head_config", {})
        self.gen_head_config = GenHeadConfig(**gen_head_config)

        language_config = kwargs.get("language_config", {})
        if isinstance(language_config, LlamaConfig):
            self.language_config = language_config
        else:
            self.language_config = LlamaConfig(**language_config)


class MultiModalityPreTrainedModel(PreTrainedModel):
    config_class = MultiModalityConfig
    base_model_prefix = "multi_modality"
    _no_split_modules = []
    _skip_keys_device_placement = "past_key_values"


class MultiModalityCausalLM(MultiModalityPreTrainedModel):
    def __init__(self, config: MultiModalityConfig):
        super().__init__(config)

        vision_config = config.vision_config
        vision_cls = model_name_to_cls(vision_config.cls)
        self.vision_model = vision_cls(**vision_config.params)

        aligner_config = config.aligner_config
        aligner_cls = model_name_to_cls(aligner_config.cls)
        self.aligner = aligner_cls(aligner_config.params)

        gen_vision_config = config.gen_vision_config
        gen_vision_cls = model_name_to_cls(gen_vision_config.cls)
        self.gen_vision_model = gen_vision_cls()

        gen_aligner_config = config.gen_aligner_config
        gen_aligner_cls = model_name_to_cls(gen_aligner_config.cls)
        self.gen_aligner = gen_aligner_cls(gen_aligner_config.params)

        gen_head_config = config.gen_head_config
        gen_head_cls = model_name_to_cls(gen_head_config.cls)
        self.gen_head = gen_head_cls(gen_head_config.params)

        self.gen_embed = torch.nn.Embedding(
            gen_vision_config.params.image_token_size,
            gen_vision_config.params.n_embed,
        )
        self.loss_fct = torch.nn.CrossEntropyLoss()

        language_config = config.language_config
        # language_config._attn_implementation = 'flash_attention_2'
        self.language_model = LlamaForCausalLM(language_config)

        self.language_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        for _n, p in self.language_model.named_parameters():
            p.requires_grad = True

        for _n, p in self.vision_model.named_parameters():
            p.requires_grad = False
        self.vision_model.eval()

        for _n, p in self.gen_vision_model.named_parameters():
            p.requires_grad = False
        self.gen_vision_model.eval()

        for _n, p in self.aligner.named_parameters():
            p.requires_grad = True

        for _n, p in self.gen_aligner.named_parameters():
            p.requires_grad = True

        for _n, p in self.gen_embed.named_parameters():
            p.requires_grad = True

        for _n, p in self.gen_head.named_parameters():
            p.requires_grad = True

    def set_eval(self):
        self.gen_head.eval()
        self.gen_embed.eval()
        self.aligner.eval()
        self.gen_aligner.eval()
        self.language_model.eval()

    def set_train(self):
        self.gen_head.train()
        self.gen_embed.train()
        self.aligner.train()
        self.gen_aligner.train()
        self.language_model.train()

    def prepare_inputs_embeds(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        images_seq_mask: torch.LongTensor,
        images_emb_mask: torch.LongTensor,
        **kwargs,
    ):
        """Prepare multimodal input embeddings.

        Args:
            input_ids (torch.LongTensor): [b, T]
            pixel_values (torch.FloatTensor):   [b, n_images, 3, h, w]
            images_seq_mask (torch.BoolTensor): [b, T]
            images_emb_mask (torch.BoolTensor): [b, n_images, n_image_tokens]

            assert torch.sum(images_seq_mask) == torch.sum(images_emb_mask)

        Returns:
            input_embeds (torch.Tensor): [b, T, D]
        """

        bs, n = pixel_values.shape[0:2]
        images = rearrange(pixel_values, "b n c h w -> (b n) c h w")
        # [b x n, T2, D]
        images_embeds = self.aligner(self.vision_model(images))

        # [b x n, T2, D] -> [b, n x T2, D]
        images_embeds = rearrange(
            images_embeds,
            "(b n) t d -> b (n t) d",
            b=bs,
            n=n,
        )
        # [b, n, T2] -> [b, n x T2]
        images_emb_mask = rearrange(images_emb_mask, "b n t -> b (n t)")

        # [b, T, D]
        input_ids[input_ids < 0] = 0  # ignore the image embeddings
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # replace with the image embeddings
        inputs_embeds[images_seq_mask] = images_embeds[images_emb_mask]

        return inputs_embeds

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None,
        image1=None,
        image_seq_mask=None,
        image2=None,
        task_type=0,
    ):
        if task_type == 0 or task_type == 2:
            if task_type == 2:
                image_embeds, _ = self.prepare_embedding(image1)
                image_embeds2, labels = self.prepare_embedding(image2)
                input_ids[input_ids < 0] = 0
                input_embeds = self.language_model.get_input_embeddings()(
                    input_ids
                )
                for i in range(input_embeds.shape[0]):
                    input_embeds[i][image_seq_mask[i]] = image_embeds[i]
                input_embeds = torch.cat((input_embeds, image_embeds2), dim=1)
                b, seq_len = image_embeds2.shape[0], image_embeds2.shape[1]
            else:
                input_embeds = self.language_model.get_input_embeddings()(
                    input_ids
                )
                image_embeds, labels = self.prepare_embedding(image1)
                input_embeds = torch.cat((input_embeds, image_embeds), dim=1)
                b, seq_len = image_embeds.shape[0], image_embeds.shape[1]
            attention_mask = torch.cat(
                (
                    attention_mask,
                    torch.ones((b, seq_len)).long().to(attention_mask.device),
                ),
                dim=1,
            )
            label_len = labels.shape[-1]
            last_hidden_state = self.language_model.model(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
            ).last_hidden_state
            image_logits = self.gen_head(last_hidden_state)
            visual_vocab_size = image_logits.shape[-1]
            shift_logits = image_logits[
                ..., -1 - label_len : -1, :
            ].contiguous()
            loss = self.loss_fct(
                shift_logits.view(-1, visual_vocab_size),
                labels.view(-1),
            )
        elif task_type == 1:
            image_embeds, _ = self.prepare_embedding(image1, gen_image=False)
            input_ids[input_ids < 0] = 0
            input_embeds = self.language_model.get_input_embeddings()(
                input_ids
            )
            for i in range(input_embeds.shape[0]):
                input_embeds[i][image_seq_mask[i]] = image_embeds[i]
            label_len = labels.shape[-1]
            text_logits = self.language_model(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
            ).logits
            text_vocab_size = text_logits.shape[-1]
            shift_logits = text_logits[..., -label_len:-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            alpha = 0.8
            shift_labels_1 = torch.where(
                (shift_labels == 5661) | (shift_labels == 3233),
                shift_labels,
                -100,
            )
            shift_labels_2 = torch.where(
                (shift_labels == 5661) | (shift_labels == 3233),
                -100,
                shift_labels,
            )
            loss_1 = self.loss_fct(
                shift_logits.view(-1, text_vocab_size),
                shift_labels_1.view(-1),
            )
            loss_2 = self.loss_fct(
                shift_logits.view(-1, text_vocab_size),
                shift_labels_2.view(-1),
            )
            loss = alpha * loss_1 + (1 - alpha) * loss_2
        else:
            raise NotImplementedError

        return loss

    def prepare_embedding(
        self,
        pixel_values: torch.FloatTensor,
        gen_image=True,
        **kwargs,
    ):
        if gen_image:
            _, _, all_image_ids = self.gen_vision_model.encode(pixel_values)
            image_ids = all_image_ids[2]
            images_embeds = self.gen_aligner(self.gen_embed(image_ids))
        else:
            image_ids = None
            images_embeds = self.aligner(self.vision_model(pixel_values))

        return images_embeds, image_ids

    def prepare_gen_img_embeds(self, image_ids: torch.LongTensor):
        return self.gen_aligner(self.gen_embed(image_ids))


AutoConfig.register("vision", VisionConfig)
AutoConfig.register("aligner", AlignerConfig)
AutoConfig.register("gen_vision", GenVisionConfig)
AutoConfig.register("gen_aligner", GenAlignerConfig)
AutoConfig.register("gen_head", GenHeadConfig)
AutoConfig.register("multi_modality", MultiModalityConfig)
AutoModelForCausalLM.register(MultiModalityConfig, MultiModalityCausalLM)
