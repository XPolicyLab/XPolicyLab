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

from __future__ import annotations
import os
from typing import Any

import numpy as np
import PIL.Image
import torch
from robo_orchard_core.utils.config import load_config_class
from torchvision import transforms
from transformers import AutoModelForCausalLM

from robo_orchard_lab.models.mapdream.janus.inference.models import (
    VLChatProcessor,
)
from robo_orchard_lab.models.mixin import ClassType_co
from robo_orchard_lab.models.torch_model import TorchModelMixin, TorchModuleCfg

__all__ = ["ProgressModel", "ProgressModelConfig"]


def center_crop_arr(pil_image, image_size: int):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size),
            resample=PIL.Image.BOX,
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size),
        resample=PIL.Image.BICUBIC,
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return PIL.Image.fromarray(
        arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    )


@torch.no_grad()
def generate_with_refine(
    mmgpt,
    vl_chat_processor,
    input_dict,
    input_ids=None,
    attention_mask=None,
    temperature: float = 1,
    parallel_size: int = 1,
    image_token_num_per_image: int = 36,
    img_size: int = 96,
    patch_size: int = 16,
    img_top_k: int | None = None,
    img_top_p: float | None = None,
    txt_top_k: int | None = None,
    txt_top_p: float | None = None,
):
    # Behavior copied from Janus inference.py.
    input_ids = input_dict["prompt_ids"].cuda()
    attention_mask = input_dict["prompt_mask"].cuda()
    images = torch.stack(input_dict["images"], dim=0).to(torch.bfloat16).cuda()

    all_imgs_2 = []

    images_seq_mask = input_ids == vl_chat_processor.image_id
    inputs_embeds = mmgpt.language_model.get_input_embeddings()(input_ids)
    _, _, all_image_ids = mmgpt.gen_vision_model.encode(images)
    image_ids = all_image_ids[2]
    image_embeds = mmgpt.gen_aligner(mmgpt.gen_embed(image_ids))
    n, t, c = image_embeds.shape
    image_embeds = image_embeds.reshape(n * t, c)
    inputs_embeds[images_seq_mask] = image_embeds

    new_generated_tokens = torch.zeros(
        (parallel_size, image_token_num_per_image),
        dtype=torch.int,
    ).cuda()

    outputs = None

    for i in range(image_token_num_per_image):
        outputs = mmgpt.language_model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=True,
            past_key_values=(
                outputs.past_key_values if outputs is not None else None
            ),
        )
        hidden_states = outputs.last_hidden_state

        new_attn = torch.ones((parallel_size, 1), dtype=torch.int).cuda()
        attention_mask = torch.cat((attention_mask, new_attn), dim=1)

        logits = mmgpt.gen_head(hidden_states[:, -1, :])

        if img_top_k:
            v, _ = torch.topk(logits, min(img_top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
        probs = torch.softmax(logits / temperature, dim=-1)
        if img_top_p:
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > img_top_p
            probs_sort[mask] = 0.0
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
            next_token = torch.multinomial(probs_sort, num_samples=1)
            next_token = torch.gather(probs_idx, -1, next_token)
        else:
            next_token = torch.argmax(probs, dim=-1, keepdim=True)

        new_generated_tokens[:, i] = next_token.squeeze(dim=-1)

        img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds

    new_dec = mmgpt.gen_vision_model.decode_code(
        new_generated_tokens.to(dtype=torch.int),
        shape=[
            parallel_size,
            8,
            img_size // patch_size,
            img_size // patch_size,
        ],
    )
    new_dec = new_dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    new_dec = np.clip((new_dec + 1) / 2 * 255, 0, 255)
    new_visual_img = np.zeros(
        (parallel_size, img_size, img_size, 3), dtype=np.uint8
    )
    new_visual_img[:, :, :] = new_dec
    for i in range(parallel_size):
        all_imgs_2.append(PIL.Image.fromarray(new_visual_img[i]))

    return all_imgs_2


def construct_input(
    vl_chat_processor,
    tokenizer,
    prompt,
    images,
    base_dir=None,
):
    # Behavior copied from Janus inference.py.
    gen_transform = transforms.Compose(
        [
            transforms.Lambda(
                lambda pil_image: center_crop_arr(pil_image, 96)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
                inplace=True,
            ),
        ]
    )

    num_gen_image_tokens = 36
    prompt = prompt.replace(
        "<image>",
        f"{vl_chat_processor.image_start_tag}"
        f"{vl_chat_processor.image_tag * num_gen_image_tokens}"
        f"{vl_chat_processor.image_end_tag}",
    )

    conversation = [
        {"role": "User", "content": prompt},
        {"role": "Assistant", "content": ""},
    ]

    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
    )

    tokenized_input = tokenizer(
        sft_format,
        return_tensors="pt",
        max_length=6000,
        padding="longest",
        truncation=False,
    ).to("cuda")
    prompt_ids = tokenized_input["input_ids"]
    prompt_mask = tokenized_input["attention_mask"]

    foreground_images = images

    all_pixel_values = []
    for img in foreground_images:
        processed_img = gen_transform(img)
        all_pixel_values.append(processed_img)

    return {
        "prompt_ids": prompt_ids,
        "prompt_mask": prompt_mask,
        "images": all_pixel_values,
        "task_type": 3,
    }


class ProgressModel(TorchModelMixin):
    """ProgressModel wrapper implemented with Janus inference pipeline.

    This aligns with `run_api_nofile.py`'s `janus_inference`:
    - load `VLChatProcessor` from pretrained
    - load base LLM with `AutoModelForCausalLM.from_pretrained`
    - load checkpoint state_dict with `torch.load`
    - call `construct_input` + `generate_with_refine`
    """

    cfg: "ProgressModelConfig"

    def __init__(self, cfg: "ProgressModelConfig" = None):
        super().__init__(cfg)
        self.cfg = cfg
        self.model = None

    @classmethod
    def load_model(
        cls, directory: str, use_decrete: bool = True
    ) -> "ProgressModel":

        # 1️⃣ 创建实例

        config_file = os.path.join(directory, "model.config.json")
        with open(config_file, "r") as f:
            cfg: ProgressModelConfig = load_config_class(f.read())

        model = cls(cfg)

        # 2️⃣ 加载基础模型（Janus）
        model.vl_chat_processor = VLChatProcessor.from_pretrained(directory)

        model.vl_gpt = AutoModelForCausalLM.from_pretrained(
            directory, trust_remote_code=True
        )

        # 3️⃣ 加载你自己的 checkpoint（Progress 权重）
        state_dict = torch.load(
            directory + "/iter_3699.pth",
            map_location="cpu",
        )
        model.vl_gpt.load_state_dict(state_dict)

        # 4️⃣ 放到 GPU + 推理模式
        model.vl_gpt = model.vl_gpt.to(torch.bfloat16).cuda().eval()

        return model

    @torch.inference_mode()
    def forward(self, inputs: Any, is_training: bool = False):

        prompt = inputs.get("prompt")
        images = inputs.get("images")

        input_dict = construct_input(
            self.vl_chat_processor,
            self.vl_chat_processor.tokenizer,
            prompt,
            images,
        )
        gen_image = generate_with_refine(
            self.vl_gpt,
            self.vl_chat_processor,
            input_dict=input_dict,
        )

        return gen_image[0]


class ProgressModelConfig(TorchModuleCfg[ProgressModel]):
    class_type: ClassType_co[ProgressModel] = ProgressModel
