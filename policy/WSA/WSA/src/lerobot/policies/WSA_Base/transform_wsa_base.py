from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

from transformers.models.qwen3_vl import Qwen3VLProcessor

import torch

from lerobot.utils.constants import (
    ACTION,
    SAMPLE_ACTION_LOSS_MASK,
    OBS_IMAGE, OBS_IMAGES, OBS_STATE, OBS_STR,
)
from lerobot.transforms.core import DataTransformFn, DataDict


@DataTransformFn.register_subclass("wsa_base_processor")
@DataTransformFn.register_subclass("tbot_sa1_processor")
@DataTransformFn.register_subclass("magicbot_processor")
@DataTransformFn.register_subclass("cubev2_processor")
@dataclass
class Qwen3_VLProcessorTransformFn(DataTransformFn):
    pretrained_model_name_or_path: str = 'Qwen/Qwen3-VL-2B-Instruct'
    max_length: int = 48
    task_key: str = "task"
    padding_side: str = "right"
    padding: str = "max_length"
    truncation: bool = True

    spatial_merge_size: int = 2

    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    image_token_id: int = 151655

    processor: Any = field(default=None, init=False, repr=False)
    _processor_source: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        # Delay loading so config parsing can override pretrained_model_name_or_path
        # before we touch the filesystem / Hugging Face cache.
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

    def __call__(self, data: DataDict) -> DataDict:
        self._ensure_processor()
        input_ids = []
        attention_mask = []
        pixel_values = []
        image_grid_thw = []
        for i in range(3):
            k = f"{OBS_IMAGES}.image{i}"
            img_inputs = self.processor.image_processor(
                data[k][1],  # we only feed images at current time to vlm
                do_rescale=False,
            )
            num_img_token = torch.prod(img_inputs.image_grid_thw) // self.spatial_merge_size ** 2
            pixel_values.append(img_inputs.pixel_values)
            image_grid_thw.append(img_inputs.image_grid_thw)
            if data[f"{k}_mask"]:
                input_ids += [self.vision_start_token_id] + [self.image_token_id] * num_img_token + [self.vision_end_token_id]
                attention_mask += [1] * (num_img_token + 2)
                # attention_mask += [0] + [1] * num_img_token + [0]
            else:
                input_ids += [self.vision_start_token_id] + [self.image_token_id] * num_img_token + [self.vision_end_token_id]
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


@DataTransformFn.register_subclass("unify_wsa_base_inputs")
@DataTransformFn.register_subclass("unify_tbot_sa1_inputs")
@DataTransformFn.register_subclass("unify_magicbot_inputs")
@DataTransformFn.register_subclass("unify_cubev2_inputs")
@dataclass
class UnifyWSABaseInputsTransformFn(DataTransformFn):
    def __call__(self, data: DataDict) -> DataDict:
        default_action_loss_mask = 0.0 if data.get("robot_type") == "egodex_v" else 1.0
        data = {
            OBS_STATE: data[OBS_STATE],
            ACTION: data[ACTION],
            SAMPLE_ACTION_LOSS_MASK: data.get(
                SAMPLE_ACTION_LOSS_MASK,
                torch.tensor([default_action_loss_mask], dtype=torch.float32),
            ),
            f"{OBS_IMAGES}.image0": data[f"{OBS_IMAGES}.image0"],
            f"{OBS_IMAGES}.image1": data[f"{OBS_IMAGES}.image1"],
            f"{OBS_IMAGES}.image2": data[f"{OBS_IMAGES}.image2"],
            f"{OBS_IMAGES}.image0_mask": data[f"{OBS_IMAGES}.image0_mask"],
            f"{OBS_IMAGES}.image1_mask": data[f"{OBS_IMAGES}.image1_mask"],
            f"{OBS_IMAGES}.image2_mask": data[f"{OBS_IMAGES}.image2_mask"],
            f"{OBS_STR}.pixel_values": data[f"{OBS_STR}.pixel_values"],
            f"{OBS_STR}.image_grid_thw": data[f"{OBS_STR}.image_grid_thw"],
            # "task": data["task"],
            f"{OBS_STR}.input_ids": data[f"{OBS_STR}.input_ids"],
            f"{OBS_STR}.attention_mask": data[f"{OBS_STR}.attention_mask"],
        }
        return data


if __name__ == "__main__":
    sample = {
        f"{OBS_IMAGES}.image0": torch.rand((224, 224, 3)),
        f"{OBS_IMAGES}.image1": torch.rand((224, 224, 3)),
        f"{OBS_IMAGES}.image2": torch.rand((224, 224, 3)),
        "task": "This is a test sample.",
    }
    processor = Qwen3_VLProcessorTransformFn()
    output = processor(sample)
    print(output)
