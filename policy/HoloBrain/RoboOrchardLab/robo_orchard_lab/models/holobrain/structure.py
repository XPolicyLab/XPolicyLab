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

import logging
import os
from typing import Optional

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import (
    AutoProcessor,
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
)

from robo_orchard_lab.models.mixin import (
    ClassType_co,
    ModelMixin,
    TorchModuleCfg,
    TorchModuleCfgType_co,
)
from robo_orchard_lab.utils.build import (
    DelayInitDictType,
    build,
)

__all__ = [
    "TextTemplate",
    "HoloBrain_Qwen2_5_VL",
    "HoloBrain_Qwen2_5_VLConfig",
]


logger = logging.getLogger(__name__)


class TextTemplate(nn.Module):
    def __init__(self, with_subtask=True):
        super().__init__()
        self.with_subtask = with_subtask
        self.template = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n{image_token}\nYou are a robot. "
            "{instruction}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def forward(self, data):
        batch_size, num_cams = data["imgs"].shape[:2]
        image_token = [
            "<|vision_start|><|image_pad|><|vision_end|>"
        ] * num_cams
        image_token = " ".join(image_token)
        instructions = data.get("text", [""] * batch_size)
        text = [
            self.template.format(
                image_token=image_token, instruction=instruction
            )
            for instruction in instructions
        ]
        if self.with_subtask and "subtask" in data:
            for i, subtask in enumerate(data["subtask"]):
                if subtask is not None and len(subtask) > 0:
                    text[i] += f"{subtask}\n"
        data["instruction"] = instructions
        data["text"] = text
        return data


class HoloBrain_Qwen2_5_VL(ModelMixin):  # noqa: N801
    cfg: "HoloBrain_Qwen2_5_VLConfig"

    def __init__(self, cfg: "HoloBrain_Qwen2_5_VLConfig"):
        super().__init__(cfg)
        self.decoder = build(self.cfg.decoder)
        self.spatial_enhancer = build(self.cfg.spatial_enhancer)
        self.data_preprocessor = build(self.cfg.data_preprocessor)
        self.backbone_3d = build(self.cfg.backbone_3d)
        self.neck_3d = build(self.cfg.neck_3d)
        self.input_2d = self.cfg.input_2d
        self.input_3d = self.cfg.input_3d
        self.use_state_dict_with_vlm = self.cfg.use_state_dict_with_vlm
        if not self.use_state_dict_with_vlm:
            assert self.cfg.freeze_vlm, (
                "The VLM's state_dict must be saved when it is not frozen."
            )
        self.with_cot = self.cfg.with_cot

        vlm_pretrain = self.cfg.vlm_pretrain
        if self.cfg.load_vlm_checkpoint:
            self.vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                vlm_pretrain,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
        else:
            config = Qwen2_5_VLConfig.from_pretrained(vlm_pretrain)
            self.vlm = Qwen2_5_VLForConditionalGeneration._from_config(
                config,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
            )
        if not hasattr(self.vlm, "language_model"):
            logger.warning("Deprecated, please use `transformers` >= 4.57.1.")
            self.vlm.language_model = self.vlm.model
            self.vlm.model.get_rope_index = self.vlm.get_rope_index

        if self.cfg.freeze_vlm:
            self.vlm.eval()
            self.vlm.requires_grad_(False)
        else:
            # self.vlm.language_model.gradient_checkpointing_enable()
            if self.cfg.freeze_vision:
                self.vlm.visual.eval()
                self.vlm.visual.requires_grad_(False)

        origin_num_layers = len(self.vlm.language_model.layers)
        if (
            self.cfg.num_vlm_layers is not None
            and self.cfg.num_vlm_layers >= 0
        ):
            self.vlm.language_model.layers = self.vlm.language_model.layers[
                : self.cfg.num_vlm_layers
            ]

        self.vlm_processor = AutoProcessor.from_pretrained(
            vlm_pretrain, use_fast=True
        )
        self.vlm_processor.tokenizer.padding_side = "left"
        self.feat_mapping = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    self.vlm.language_model.config.hidden_size,
                    self.decoder.embed_dims,
                    bias=True,
                    dtype=torch.bfloat16,
                )
                for _ in range(len(self.vlm.language_model.layers) + 1)
            ]
        )
        temperature = 3
        highlighted_layer = origin_num_layers // 2  # the highlighted layer
        weight = torch.cat(
            [
                torch.linspace(0.1, 1, highlighted_layer + 1),
                torch.linspace(1, 0.1, origin_num_layers - highlighted_layer),
            ]
        )[: len(self.vlm.language_model.layers) + 1]
        weight = weight.to(dtype=torch.bfloat16) * temperature
        self.weight = torch.nn.Parameter(weight, requires_grad=True)
        self.qwen_patch_size = (
            self.vlm.config.vision_config.patch_size
            * self.vlm.config.vision_config.spatial_merge_size
        )

    def save_model(
        self, directory: str, model_prefix: str = "model", **kwargs
    ):
        super().save_model(directory, model_prefix, **kwargs)
        directory = os.path.join(directory, self.cfg.vlm_pretrain)
        if self.cfg.save_model_with_vlm:
            if not os.path.exists(directory):
                os.makedirs(directory)
            self.vlm.save_pretrained(directory)
            self.vlm_processor.save_pretrained(directory)
        elif os.path.isdir(self.cfg.vlm_pretrain):
            if not os.path.exists(os.path.split(directory)[0]):
                os.makedirs(os.path.split(directory)[0])
            os.symlink(os.path.abspath(self.cfg.vlm_pretrain), directory)

    def forward(self, inputs):
        if self.data_preprocessor is not None:
            device = next(self.parameters()).device
            inputs = self.data_preprocessor(inputs, device)
        if self.training:
            return self.loss(inputs)
        else:
            return self.predict(inputs)

    def loss(self, inputs):
        model_outs, _, text_dict, loss_depth = self._forward(inputs)
        loss = self.decoder.loss(model_outs, inputs, text_dict=text_dict)
        if loss_depth is not None:
            loss["loss_depth"] = loss_depth
        return loss

    @torch.no_grad()
    def predict(self, inputs):
        model_outs, _, text_dict = self._forward(inputs)
        results = self.decoder.post_process(
            model_outs, inputs, text_dict=text_dict
        )
        return results

    def extract_feature_3d(self, inputs):
        input_3d = inputs.get(self.input_3d)
        if self.backbone_3d is not None and input_3d is not None:
            dtype = next(self.backbone_3d.parameters()).dtype
            input_3d = input_3d.to(dtype=dtype)
            if "depth" in self.input_3d and input_3d.dim() == 5:
                bs, num_cams = input_3d.shape[:2]
                input_3d = input_3d.flatten(end_dim=1)
            elif "depth" in self.input_3d:
                num_cams = 1
            feature_3d = self.backbone_3d(input_3d)
            if self.neck_3d is not None:
                feature_3d = self.neck_3d(feature_3d)
            if "depth" in self.input_3d:
                feature_3d = [
                    x.unflatten(0, (bs, num_cams)) for x in feature_3d
                ]
        else:
            feature_3d = None
        return feature_3d

    def batch_decode_ids(self, token_ids, skip_special_tokens=True):
        text = self.vlm_processor.batch_decode(
            token_ids, skip_special_tokens=skip_special_tokens
        )
        return text

    def foward_feat_mapping(self, x):
        if isinstance(x, (list, tuple)):
            x = torch.stack(x, dim=0)
        weight = torch.stack(
            [layer.weight for layer in self.feat_mapping], dim=0
        )
        weight = weight[:, None]
        x = x @ weight.mT

        bias = torch.stack([layer.bias for layer in self.feat_mapping], dim=0)
        bias = bias[:, None, None]
        x = x + bias
        x = x.permute(1, 2, 3, 0)
        x = x @ torch.nn.functional.softmax(self.weight, dim=0)
        return x

    def _vlm_outputs_handler(self, vlm_outputs, vlm_inputs, inputs):
        if self.with_cot:
            assert "sequences" in vlm_outputs
            vlm_input_ids = vlm_outputs.sequences[:, :-1]
        else:
            vlm_input_ids = vlm_inputs["input_ids"]

        batch_size, num_cams, _, h, w = inputs["imgs"].shape
        vlm_outputs = vlm_outputs["hidden_states"]

        vlm_outputs = checkpoint(
            self.foward_feat_mapping, vlm_outputs, use_reentrant=False
        )
        # vlm_outputs = self.foward_feat_mapping(vlm_outputs)
        vlm_outputs = vlm_outputs.to(torch.float32)

        img_feature_mask = self.vlm.config.image_token_id == vlm_input_ids
        img_feature = vlm_outputs[img_feature_mask].unflatten(
            0, (batch_size, -1)
        )
        h_, w_ = h // self.qwen_patch_size, w // self.qwen_patch_size
        feature_maps = [
            img_feature.reshape(batch_size, num_cams, h_, w_, -1).permute(
                0, 1, 4, 2, 3
            )
        ]

        text_feature_mask = ~img_feature_mask
        text_feature = vlm_outputs[text_feature_mask]
        text_feature = text_feature.unflatten(0, (batch_size, -1))
        not_pad_mask = (
            vlm_input_ids != self.vlm_processor.tokenizer.pad_token_id
        )
        text_feature_mask = text_feature_mask & not_pad_mask
        text_feature_mask = text_feature_mask[~img_feature_mask]
        text_feature_mask = text_feature_mask.unflatten(0, (batch_size, -1))

        text_dict = dict(
            embedded=text_feature, text_token_mask=text_feature_mask
        )
        return feature_maps, text_dict

    def _forward(self, inputs):
        batch_size, num_cams, _, h, w = inputs["imgs"].shape
        device = next(self.parameters()).device
        text = inputs["text"]

        vlm_inputs = self.vlm_processor(
            text=text,
            images=inputs["imgs"].permute(0, 1, 3, 4, 2).flatten(0, 1),
            padding=True,
            return_tensors="pt",
        )
        vlm_inputs = vlm_inputs.to(device)
        if not self.with_cot:
            vlm_outputs = self._forward_vlm(**vlm_inputs)
        else:
            vlm_outputs = self._generate_vlm(vlm_inputs)

        feature_maps, text_dict = self._vlm_outputs_handler(
            vlm_outputs, vlm_inputs, inputs
        )

        feature_3d = self.extract_feature_3d(inputs)

        if self.spatial_enhancer is not None:
            feature_maps, depth_prob, loss_depth = self.spatial_enhancer(
                feature_maps=feature_maps,
                feature_3d=feature_3d,
                text_dict=text_dict,
                inputs=inputs,
            )
        else:
            depth_prob = loss_depth = None
        model_outs = self.decoder(
            feature_maps=feature_maps,
            feature_3d=feature_3d,
            text_dict=text_dict,
            inputs=inputs,
            depth_prob=depth_prob,
        )
        if self.training:
            return model_outs, feature_maps, text_dict, loss_depth
        return model_outs, feature_maps, text_dict

    def _forward_vlm(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        image_grid_thw=None,
    ):
        inputs_embeds = self.vlm.language_model.embed_tokens(input_ids)
        if pixel_values is not None:
            pixel_values = pixel_values.type(self.vlm.visual.dtype)
            image_embeds = self.vlm.visual(
                pixel_values, grid_thw=image_grid_thw
            )
            n_image_tokens = (
                (input_ids == self.vlm.config.image_token_id).sum().item()
            )
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    "Image features and image tokens do not match: "
                    f"tokens: {n_image_tokens}, features {n_image_features}"
                )

            mask = input_ids == self.vlm.config.image_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            image_mask = mask_expanded.to(inputs_embeds.device)

            image_embeds = image_embeds.to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask, image_embeds
            )

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

        position_ids, _rope_deltas = self.vlm.model.get_rope_index(
            input_ids,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
        )

        outputs = self.vlm.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        return outputs

    @torch.no_grad()
    def _generate_vlm(self, inputs):
        outputs = self.vlm.generate(
            **inputs,
            max_new_tokens=256,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

        # (ar_times, num_layers + 1, batch_size, seq, hidden_size)
        hidden_states = outputs.hidden_states
        # + 1 for the input embeddings
        num_hidden_states = len(self.vlm.language_model.layers) + 1
        cated_hidden_states = []
        for i in range(num_hidden_states):
            hs_i = torch.cat([h[i] for h in hidden_states], dim=1)
            cated_hidden_states.append(hs_i)
        cated_hidden_states = tuple(cated_hidden_states)
        outputs.hidden_states = cated_hidden_states
        return outputs

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        state_dict = super().state_dict(*args, destination, prefix, keep_vars)
        if self.use_state_dict_with_vlm:
            return state_dict
        keys_to_remove = [k for k in state_dict.keys() if k.startswith("vlm.")]
        for k in keys_to_remove:
            del state_dict[k]
        return state_dict

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        if self.use_state_dict_with_vlm:
            return super().load_state_dict(state_dict, strict=strict, **kwargs)
        filtered_state_dict = {
            k: v for k, v in state_dict.items() if not k.startswith("vlm.")
        }
        incompatible_keys = super().load_state_dict(
            filtered_state_dict, strict=False, **kwargs
        )
        missing_keys = []
        for key in incompatible_keys.missing_keys:
            if not key.startswith("vlm."):
                missing_keys.append(key)
        incompatible_keys = type(incompatible_keys)(
            missing_keys, incompatible_keys.unexpected_keys
        )
        if strict:
            assert (
                len(incompatible_keys.missing_keys) == 0
                and len(incompatible_keys.unexpected_keys) == 0
            ), (
                "Unexpected key(s) in state_dict: {}. Missing key(s) in state_dict: {}.".format(  # noqa: E501
                    ", ".join(
                        f'"{k}"' for k in incompatible_keys.unexpected_keys
                    ),
                    ", ".join(
                        f'"{k}"' for k in incompatible_keys.missing_keys
                    ),
                )
            )
        return incompatible_keys


MODULE_TYPE = TorchModuleCfgType_co | DelayInitDictType  # noqa: E501


class HoloBrain_Qwen2_5_VLConfig(TorchModuleCfg[HoloBrain_Qwen2_5_VL]):  # noqa: N801
    class_type: ClassType_co[HoloBrain_Qwen2_5_VL] = HoloBrain_Qwen2_5_VL
    vlm_pretrain: str
    decoder: MODULE_TYPE  # type: ignore
    spatial_enhancer: MODULE_TYPE | None = None
    data_preprocessor: MODULE_TYPE | None = None
    backbone_3d: MODULE_TYPE | None = None
    neck_3d: MODULE_TYPE | None = None
    input_2d: str = "imgs"
    input_3d: str = "depths"
    freeze_vlm: bool = True
    freeze_vision: bool = True
    use_state_dict_with_vlm: bool = False
    load_vlm_checkpoint: bool = True
    with_cot: bool = False
    save_model_with_vlm: bool = False
    num_vlm_layers: Optional[int] = None
