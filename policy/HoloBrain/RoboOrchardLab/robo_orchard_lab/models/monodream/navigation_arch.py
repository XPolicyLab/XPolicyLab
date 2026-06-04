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

import copy
import logging
import os
import warnings
from abc import ABC
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple, Union

import PIL.Image
import torch
from transformers import GenerationConfig, PretrainedConfig

from robo_orchard_lab.models.monodream.language_model.builder import (
    build_llm_and_tokenizer,
)
from robo_orchard_lab.models.monodream.multimodal_encoder import (
    BasicImageEncoder,
)
from robo_orchard_lab.models.monodream.multimodal_encoder.builder import (
    build_vision_tower,
)
from robo_orchard_lab.models.monodream.multimodal_encoder.mm_utils import (
    process_image,
    process_images,
)
from robo_orchard_lab.models.monodream.multimodal_projector.builder import (
    build_mm_projector,
)
from robo_orchard_lab.models.monodream.utils.constants import (
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    MEDIA_TOKENS,
    NUM_EXTRA_TOKENS,
    Image,
)
from robo_orchard_lab.models.monodream.utils.tokenizer import (
    tokenize_conversation,
)
from robo_orchard_lab.utils import as_sequence


def extract_media(
    messages: List[Dict[str, Any]],
    draft: bool = False,
) -> Dict[str, List[Any]]:
    media = defaultdict(list)
    for message in messages:
        text = ""
        for part in as_sequence(message["value"]):
            if isinstance(part, str):
                for token in MEDIA_TOKENS.values():
                    if token in part:
                        part = part.strip()
                text += part
            elif isinstance(part, (Image, PIL.Image.Image)):
                if draft:
                    media["image"].append(part)
                else:
                    media["image"].append(PIL.Image.open(part))
            else:
                raise ValueError(f"Unsupported prompt part type: {type(part)}")
        message["value"] = text
    return media


def get_model_config(config):
    default_keys = ["llm_cfg", "vision_tower_cfg", "mm_projector_cfg"]

    if hasattr(config, "_name_or_path") and len(config._name_or_path) >= 2:
        root_path = config._name_or_path
    else:
        root_path = config.resume_path

    return_list = []
    for key in default_keys:
        cfg = getattr(config, key, None)
        if isinstance(cfg, dict):
            try:
                return_list.append(os.path.join(root_path, key[:-4]))
            except Exception as e:
                print(e)
                raise ValueError(
                    f"Cannot find resume path in config for {key}!"
                )
        elif isinstance(cfg, PretrainedConfig):
            return_list.append(os.path.join(root_path, key[:-4]))
        elif isinstance(cfg, str):
            return_list.append(cfg)

    return return_list


class LlavaMetaModel(ABC):  # noqa: B024
    """Language model define.

    Define the construction methods.
    """

    def init_vlm(self, config, *args, **kwargs):
        if (
            hasattr(self, "llm")
            or hasattr(self, "vision_tower")
            or hasattr(self, "mm_projector")
        ):
            return

        model_dtype = getattr(config, "model_dtype", "torch.float16")
        if not hasattr(config, "model_dtype"):
            warnings.warn(
                "model_dtype not found in config, defaulting to torch.float16."
            )
            config.model_dtype = model_dtype

        cfgs = get_model_config(config)
        if len(cfgs) == 3:
            llm_cfg, vision_tower_cfg, mm_projector_cfg = cfgs
        else:
            raise ValueError(
                "`llm_cfg` `mm_projector_cfg` `vision_tower_cfg`"
                "not found in the config."
            )

        self.llm, self.tokenizer = build_llm_and_tokenizer(
            llm_cfg, config, *args, **kwargs
        )
        self.vision_tower = build_vision_tower(vision_tower_cfg, config)
        self.mm_projector = build_mm_projector(mm_projector_cfg, config)

        self.vocab_size = config.llm_cfg["vocab_size"] + NUM_EXTRA_TOKENS

        self.encoders = {}
        self.encoders["image"] = BasicImageEncoder(parent=self.model).cuda()

        self.post_config()
        self.is_loaded = True

        assert (
            self.llm is not None
            or self.vision_tower is not None
            or self.mm_projector is not None
        ), "At least one of the components must be instantiated."

    def get_llm(self):
        llm = getattr(self, "llm", None)
        if type(llm) is list:
            llm = llm[0]
        return llm

    def get_lm_head(self):
        lm_head = getattr(self.get_llm(), "lm_head", None)
        return lm_head

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def get_mm_projector(self):
        mm_projector = getattr(self, "mm_projector", None)
        if type(mm_projector) is list:
            mm_projector = mm_projector[0]
        return mm_projector

    def post_config(self):
        self.training = self.get_llm().training
        ## configuration
        if getattr(self.config, "llm_cfg", None) is None:
            self.config.llm_cfg = self.llm.config
        if getattr(self.config, "vision_tower_cfg", None) is None:
            self.config.vision_tower_cfg = self.vision_tower.config
        if getattr(self.config, "mm_projector_cfg", None) is None:
            self.config.mm_projector_cfg = self.mm_projector.config

    def freezed_module_patch(self):
        if self.training:
            if self.get_llm() and not getattr(
                self.config, "tune_language_model", False
            ):
                pass
            if self.get_vision_tower() and not getattr(
                self.config, "tune_vision_tower", False
            ):
                self.get_vision_tower().eval()
            if self.get_mm_projector() and not getattr(
                self.config, "tune_mm_projector", False
            ):
                self.get_mm_projector().eval()

    def encode_images(
        self, images, block_sizes: Optional[Optional[Tuple[int, ...]]] = None
    ):
        if block_sizes is None:
            block_sizes = [None] * len(images)

        image_features = self.get_vision_tower()(images)
        image_features = self.get_mm_projector()(image_features)
        return image_features

    def get_input_embeddings(self):
        return self.get_llm().get_input_embeddings()

    def get_output_embeddings(self):
        return self.get_llm().get_output_embeddings()

    def resize_token_embeddings(self, embed_size):
        self.get_llm().resize_token_embeddings(embed_size)


class LlavaMetaForCausalLM(ABC):  # noqa: B024
    """Language model define.

    Define the forward and generation methods.
    """

    def _embed(
        self,
        input_ids: torch.Tensor,
        media: Dict[str, List[torch.Tensor]],
        media_config: Dict[str, Dict[str, Any]],
        labels: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        labels = (
            labels
            if labels is not None
            else torch.full_like(input_ids, IGNORE_INDEX)
        )
        attention_mask = (
            attention_mask
            if attention_mask is not None
            else torch.ones_like(input_ids, dtype=torch.bool)
        )

        # Extract text and media embeddings
        text_embeds = self.llm.model.embed_tokens(input_ids)
        media_embeds = self.__embed_media_tokens(media, media_config)

        # This is a workaround to make sure the dummy embeddings are consumed
        while media_embeds.get("dummy"):
            dummy_embed = media_embeds["dummy"].popleft()
            text_embeds += torch.sum(dummy_embed) * 0

        # Remove padding
        batch_size = labels.shape[0]
        text_embeds = [
            text_embeds[k][attention_mask[k]] for k in range(batch_size)
        ]
        labels = [labels[k][attention_mask[k]] for k in range(batch_size)]

        # Build inverse mapping from token ID to media name
        media_tokens = {}
        for name, token_id in self.tokenizer.media_token_ids.items():
            media_tokens[token_id] = name

        # Fuse text and media embeddings
        inputs_m, labels_m = [], []
        for k in range(batch_size):
            inputs_mk, labels_mk = [], []
            pos = 0
            while pos < len(labels[k]):
                if input_ids[k][pos].item() in media_tokens:
                    end = pos + 1
                    name = media_tokens[input_ids[k][pos].item()]
                    input = media_embeds[name].popleft()
                    label = torch.full(
                        [input.shape[0]],
                        IGNORE_INDEX,
                        device=labels[k].device,
                        dtype=labels[k].dtype,
                    )
                else:
                    end = pos
                    while (
                        end < len(labels[k])
                        and input_ids[k][end].item() not in media_tokens
                    ):
                        end += 1
                    input = text_embeds[k][pos:end]
                    label = labels[k][pos:end]
                inputs_mk.append(input)
                labels_mk.append(label)
                pos = end
            inputs_m.append(torch.cat(inputs_mk, dim=0))
            labels_m.append(torch.cat(labels_mk, dim=0))
        inputs, labels = inputs_m, labels_m

        # Check if all media embeddings are consumed
        for name in media_embeds:
            if media_embeds[name]:
                raise ValueError(f"Not all {name} embeddings are consumed!")

        # Truncate sequences to `model_max_length`
        # as media embeddings are inserted
        inputs, labels = self.__truncate_sequence(inputs, labels)

        # Pad sequences to the longest one in the batch
        return self.__batchify_sequence(inputs, labels)

    def __embed_media_tokens(
        self,
        media: Dict[str, List[torch.Tensor]],
        media_config: Dict[str, Dict[str, Any]],
    ) -> Dict[str, List[torch.Tensor]]:
        embeds = defaultdict(deque)
        for name in media:
            embeds[name] = deque(
                self.encoders[name](media[name], media_config[name])
            )
        return embeds

    def __truncate_sequence(
        self, inputs: List[torch.Tensor], labels: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if any(
            len(input) > self.tokenizer.model_max_length for input in inputs
        ):
            warnings.warn(
                f"Truncating sequences to `model_max_length`"
                f"({self.tokenizer.model_max_length})."
            )
            inputs = [
                input[: self.tokenizer.model_max_length] for input in inputs
            ]
            labels = [
                label[: self.tokenizer.model_max_length] for label in labels
            ]
        return inputs, labels

    def __batchify_sequence(
        self, inputs: List[torch.Tensor], labels: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = len(inputs)
        device = inputs[0].device
        hidden_size = inputs[0].shape[1]
        max_length = max(inputs[k].shape[0] for k in range(batch_size))
        attention_mask = torch.ones(
            (batch_size, max_length), dtype=torch.bool, device=device
        )

        inputs_p, labels_p = [], []
        for k in range(batch_size):
            size_pk = max_length - inputs[k].shape[0]
            inputs_pk = torch.zeros(
                (size_pk, hidden_size), dtype=inputs[k].dtype, device=device
            )
            labels_pk = torch.full(
                (size_pk,), IGNORE_INDEX, dtype=labels[k].dtype, device=device
            )
            if self.tokenizer.padding_side == "right":
                attention_mask[k, inputs[k].shape[0] :] = False
                inputs_pk = torch.cat([inputs[k], inputs_pk], dim=0)
                labels_pk = torch.cat([labels[k], labels_pk], dim=0)
            else:
                attention_mask[k, : -inputs[k].shape[0]] = False
                inputs_pk = torch.cat([inputs_pk, inputs[k]], dim=0)
                labels_pk = torch.cat([labels_pk, labels[k]], dim=0)
            inputs_p.append(inputs_pk)
            labels_p.append(labels_pk)

        inputs = torch.stack(inputs_p, dim=0)
        labels = torch.stack(labels_p, dim=0)
        return inputs, labels, attention_mask

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Optional[torch.FloatTensor] = None,
        media: Optional[Dict[str, List[torch.Tensor]]] = None,
        media_config: Dict[str, Dict[str, Any]] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **generation_kwargs,
    ):
        inputs_embeds, _, attention_mask = self._embed(
            input_ids, media, media_config, None, attention_mask
        )
        return self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generation_kwargs,
        )

    @torch.inference_mode()
    def generate_content(
        self,
        prompt: Union[str, List],
        generation_config: Optional[GenerationConfig] = None,
    ) -> str:
        conversation = [{"from": "human", "value": prompt}]

        # Extract media from the conversation
        media = extract_media(conversation, self.config)

        # Process media
        media_config = defaultdict(dict)
        for name in media:
            if name == "image":
                if len(
                    media["image"]
                ) == 1 and self.config.image_aspect_ratio in [
                    "dynamic",
                    "dynamic_s2",
                ]:
                    self.config.image_processor = (
                        self.vision_tower.image_processor
                    )
                    if self.config.image_aspect_ratio == "dynamic":
                        images = process_image(
                            media["image"][0],
                            self.config,
                            None,
                            enable_dynamic_res=True,
                        ).half()
                        conversation[0]["value"] = conversation[0][
                            "value"
                        ].replace(
                            DEFAULT_IMAGE_TOKEN,
                            f"{DEFAULT_IMAGE_TOKEN}\n" * images.shape[0],
                        )
                    else:
                        if type(self.config.s2_scales) is str:
                            self.config.s2_scales = list(
                                map(int, self.config.s2_scales.split(","))
                            )
                        images, block_sizes = process_image(
                            media["image"][0],
                            self.config,
                            None,
                            enable_dynamic_s2=True,
                        )
                        images = images.half()
                        media_config[name]["block_sizes"] = [block_sizes]
                else:
                    images = process_images(
                        media["image"],
                        self.vision_tower.image_processor,
                        self.config,
                    ).half()
                media[name] = [image for image in images]
            elif name == "video":
                media[name] = [
                    process_images(
                        images, self.vision_tower.image_processor, self.config
                    ).half()
                    for images in media[name]
                ]
            else:
                raise ValueError(f"Unsupported media type: {name}")

        # Tokenize the conversation
        input_ids = (
            tokenize_conversation(
                conversation, self.tokenizer, add_generation_prompt=True
            )
            .cuda()
            .unsqueeze(0)
        )

        # Set up the generation config
        generation_config = generation_config or self.default_generation_config

        # Generate the response
        try:
            output_ids = self.generate(
                input_ids=input_ids,
                media=media,
                media_config=media_config,
                generation_config=generation_config,
            )
        except ValueError:
            if not generation_config.do_sample:
                raise
            logging.warning(
                "Generation failed with samplingretrying with greedy decoding."
            )
            generation_config.do_sample = False
            output_ids = self.generate(
                input_ids=input_ids,
                media=media,
                media_config=media_config,
                generation_config=generation_config,
            )

        # Decode the response
        response = self.tokenizer.decode(
            output_ids[0], skip_special_tokens=True
        ).strip()
        return response

    @property
    def default_generation_config(self) -> GenerationConfig:
        generation_config = copy.deepcopy(
            self.generation_config or GenerationConfig()
        )
        if self.tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must have an EOS token")
        if generation_config.max_length == GenerationConfig().max_length:
            generation_config.max_length = self.tokenizer.model_max_length
        if generation_config.pad_token_id is None:
            generation_config.pad_token_id = (
                self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            )
        if generation_config.bos_token_id is None:
            generation_config.bos_token_id = (
                self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
            )
        if generation_config.eos_token_id is None:
            generation_config.eos_token_id = self.tokenizer.stop_token_ids
        return generation_config
