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


import base64
import io
import json
from typing import Dict, List

import PIL.Image
import torch
from janus.models import MultiModalityCausalLM, VLChatProcessor
from transformers import AutoModelForCausalLM


def load_pretrained_model(model_path: str):
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(
        model_path
    )
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True
    )
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

    return tokenizer, vl_chat_processor, vl_gpt


def load_pil_images(
    conversations: List[Dict[str, str]],
) -> List[PIL.Image.Image]:
    """Support file path or base64 images.

    Args:
        conversations (List[Dict[str, str]]): the conversations with a
            list of messages.

    Returns:
        pil_images (List[PIL.Image.Image]): the list of PIL images.
    """

    pil_images = []

    for message in conversations:
        if "images" not in message:
            continue

        for image_data in message["images"]:
            if image_data.startswith("data:image"):
                # Image data is in base64 format
                _, image_data = image_data.split(",", 1)
                image_bytes = base64.b64decode(image_data)
                pil_img = PIL.Image.open(io.BytesIO(image_bytes))
            else:
                # Image data is a file path
                pil_img = PIL.Image.open(image_data)
            pil_img = pil_img.convert("RGB")
            pil_images.append(pil_img)

    return pil_images


def load_json(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
        return data
