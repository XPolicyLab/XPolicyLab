from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from transformers.models.qwen3_vl import Qwen3VLProcessor

from lerobot.transforms.core import DataDict, DataTransformFn
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE, OBS_STR


@DataTransformFn.register_subclass("qwenaction_processor")
@dataclass
class QwenActionProcessorTransformFn(DataTransformFn):
    pretrained_model_name_or_path: str = "Qwen/Qwen3-VL-2B-Instruct"
    max_length: int = 48
    task_key: str = "task"
    padding_side: str = "right"
    padding: str = "max_length"
    truncation: bool = True
    current_image_index: int = -1

    spatial_merge_size: int = 2

    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    image_token_id: int = 151655

    processor: Any = field(default=None, init=False, repr=False)
    _processor_source: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.processor = None
        self._processor_source = None

    def _ensure_processor(self) -> None:
        if self.processor is not None and self._processor_source == self.pretrained_model_name_or_path:
            return

        self.processor = Qwen3VLProcessor.from_pretrained(self.pretrained_model_name_or_path)
        self._processor_source = self.pretrained_model_name_or_path
        self.vision_start_token_id = self.processor.vision_start_token_id
        self.vision_end_token_id = self.processor.vision_end_token_id
        self.image_token_id = self.processor.image_token_id

    @staticmethod
    def _is_valid_mask(mask: Any) -> bool:
        if torch.is_tensor(mask):
            return bool(mask.detach().cpu().item())
        return bool(mask)

    def _select_current_image(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim == 4:
            return image[self.current_image_index]
        return image

    def __call__(self, data: DataDict) -> DataDict:
        self._ensure_processor()
        input_ids = []
        attention_mask = []
        pixel_values = []
        image_grid_thw = []

        for i in range(3):
            key = f"{OBS_IMAGES}.image{i}"
            image = self._select_current_image(data[key])
            img_inputs = self.processor.image_processor(image, do_rescale=False)
            pixel_values.append(img_inputs.pixel_values)
            image_grid_thw.append(img_inputs.image_grid_thw)

            num_img_token = int(torch.prod(image_grid_thw[-1]).item() // self.spatial_merge_size**2)
            input_ids += [self.vision_start_token_id] + [self.image_token_id] * num_img_token + [
                self.vision_end_token_id
            ]
            if self._is_valid_mask(data[f"{key}_mask"]):
                attention_mask += [1] * (num_img_token + 2)
            else:
                attention_mask += [0] * (num_img_token + 2)

        data[f"{OBS_STR}.pixel_values"] = torch.cat(pixel_values)
        data[f"{OBS_STR}.image_grid_thw"] = torch.cat(image_grid_thw)

        lang_inputs = self.processor.tokenizer(
            data[self.task_key],
            max_length=self.max_length,
            padding_side=self.padding_side,
            padding=self.padding,
            truncation=self.truncation,
        )

        input_ids += lang_inputs.input_ids
        attention_mask += lang_inputs.attention_mask
        data[f"{OBS_STR}.input_ids"] = torch.tensor(input_ids)
        data[f"{OBS_STR}.attention_mask"] = torch.tensor(attention_mask)

        return data


@DataTransformFn.register_subclass("unify_qwenaction_inputs")
@dataclass
class UnifyQwenActionInputsTransformFn(DataTransformFn):
    def __call__(self, data: DataDict) -> DataDict:
        return {
            OBS_STATE: data[OBS_STATE],
            ACTION: data[ACTION],
            f"{OBS_IMAGES}.image0": data[f"{OBS_IMAGES}.image0"],
            f"{OBS_IMAGES}.image1": data[f"{OBS_IMAGES}.image1"],
            f"{OBS_IMAGES}.image2": data[f"{OBS_IMAGES}.image2"],
            f"{OBS_IMAGES}.image0_mask": data[f"{OBS_IMAGES}.image0_mask"],
            f"{OBS_IMAGES}.image1_mask": data[f"{OBS_IMAGES}.image1_mask"],
            f"{OBS_IMAGES}.image2_mask": data[f"{OBS_IMAGES}.image2_mask"],
            f"{OBS_STR}.pixel_values": data[f"{OBS_STR}.pixel_values"],
            f"{OBS_STR}.image_grid_thw": data[f"{OBS_STR}.image_grid_thw"],
            f"{OBS_STR}.input_ids": data[f"{OBS_STR}.input_ids"],
            f"{OBS_STR}.attention_mask": data[f"{OBS_STR}.attention_mask"],
        }
