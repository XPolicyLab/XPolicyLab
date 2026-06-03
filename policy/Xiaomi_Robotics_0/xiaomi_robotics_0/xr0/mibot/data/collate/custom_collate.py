# Copyright (C) 2026 Xiaomi Corporation.
from typing import Any, Dict, List

from torch.utils.data.dataloader import default_collate
from transformers import AutoProcessor


class CustomCollate:
    """Apply the VLM chat template and collate the remaining tensors."""

    def __init__(self) -> None:
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
        self.processor.tokenizer.padding_side = "right"

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        messages: List[Any] = [item["messages"] for item in batch]
        payload: List[Dict[str, Any]] = [{key: value for key, value in item.items() if key != "messages"} for item in batch]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            images_kwargs={"do_resize": False},
        )

        if payload[0]:
            inputs.update(default_collate(payload))

        return inputs
