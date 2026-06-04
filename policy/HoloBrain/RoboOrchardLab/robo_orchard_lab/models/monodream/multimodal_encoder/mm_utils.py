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

# This file was originally copied from the [VILA] repository:
# https://github.com/NVlabs/VILA
# Modifications have been made to fit the needs of this project.


import base64
import os
from io import BytesIO

import torch
from PIL import Image as PILImage

from robo_orchard_lab.models.monodream.utils.constants import Image


def load_image_from_base64(image):
    return PILImage.open(BytesIO(base64.b64decode(image)))


def expand2square(pil_img, background_color):
    """Expand the given PIL image to a square shape by adding padding.

    Parameters:
    - pil_img: The PIL image to be expanded.
    - background_color: The color of the padding to be added.

    Returns:
    - The expanded PIL image.

    If the image is already square, it is returned as is.
    If the image is wider than it is tall,
    padding is added to the top and bottom.
    If the image is taller than it is wide,
    padding is added to the left and right.
    """
    width, height = pil_img.size
    if pil_img.mode == "L":
        background_color = background_color[0]
    if width == height:
        return pil_img
    elif width > height:
        result = PILImage.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = PILImage.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_image(
    image_file,
    data_args,
    image_folder,
):
    processor = data_args.image_processor
    if isinstance(image_file, str):
        if image_folder is not None:
            image = PILImage.open(
                os.path.join(image_folder, image_file)
            ).convert("RGB")
        else:
            image = PILImage.open(image_file).convert("RGB")
    elif isinstance(image_file, Image):
        image = PILImage.open(image_file.path).convert("RGB")
    else:
        # image is stored in bytearray
        image = image_file
    image = image.convert("RGB")
    if hasattr(data_args.image_processor, "crop_size"):
        crop_size = data_args.image_processor.crop_size
    else:
        assert hasattr(data_args.image_processor, "size")
        crop_size = data_args.image_processor.size

    if data_args.image_aspect_ratio == "resize":
        image = image.resize((crop_size["width"], crop_size["height"]))
    if data_args.image_aspect_ratio == "pad":

        def expand2square(pil_img, background_color):
            width, height = pil_img.size
            if width == height:
                return pil_img
            elif width > height:
                result = PILImage.new(
                    pil_img.mode, (width, width), background_color
                )
                result.paste(pil_img, (0, (width - height) // 2))
                return result
            else:
                result = PILImage.new(
                    pil_img.mode, (height, height), background_color
                )
                result.paste(pil_img, ((height - width) // 2, 0))
                return result

        image = expand2square(
            image, tuple(int(x * 255) for x in processor.image_mean)
        )
        image = processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]
    else:
        image = processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]
    return image


def process_images(
    images, image_processor, model_cfg, enable_dynamic_res=False
):
    model_cfg.image_processor = image_processor
    new_images = [process_image(image, model_cfg, None) for image in images]

    if all(x.shape == new_images[0].shape for x in new_images):
        if len(new_images[0].shape) == 4:
            new_images = torch.cat(new_images, dim=0)
        elif len(new_images[0].shape) == 3:
            new_images = torch.stack(new_images, dim=0)
        else:
            raise ValueError(
                f"new_images rank does not equal to 4,"
                f"rank: {len(new_images[0].shape)}"
            )
    else:
        raise ValueError("The shape of images in new_images is different!")
    return new_images


def tokenizer_image_token(prompt, tokenizer, return_tensors=None):
    return tokenizer(prompt, return_tensors=return_tensors).input_ids[0]


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith("checkpoint-"):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]
