import random
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from llava.mm_utils import dynamic_process_images_and_prompt, process_images
from llava.train.args import DataArguments
from llava.utils.logging import logger
from llava.utils.media import extract_media
from llava.utils.tokenizer import preprocess_conversation

__all__ = ["BaseDataset"]


def _process_image(image: List[Any], data_args: DataArguments) -> torch.Tensor:
    return process_images(image, data_args.image_processor, data_args)


class BaseDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_args: DataArguments,
        no_system_prompt: bool = False,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.no_system_prompt = no_system_prompt
        self.instances = []
        self.enable_dynamic_res = False

    def process(self, instance: Dict[str, Any]) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Dict[str, Any]:
        instance = self.instances[index]

        try:
            # Process instance to conversation
            conversation = self.process(instance)

            # Extract media from conversation
            media = extract_media(conversation, self.data_args)

            # Process media
            if "image" in media:
                if self.enable_dynamic_res and self.data_args.image_aspect_ratio == "dynamic":
                    processed_images, processed_prompt = dynamic_process_images_and_prompt(
                        media["image"], conversation[0]["value"], self.data_args
                    )
                    conversation[0]["value"] = processed_prompt
                else:
                    processed_images = _process_image(media["image"], self.data_args)

            # Prepare "input_ids" and "labels" for training
            data = preprocess_conversation(conversation, self.tokenizer, no_system_prompt=self.no_system_prompt)

            if "image" in media:
                data["image"] = processed_images

        except Exception as e:
            logger.exception(f"Error processing instance '{instance}': '{e}'. Resampling.")
            return self.__getitem__(random.randint(0, len(self.instances) - 1))

        return data

    def __len__(self) -> int:
        return len(self.instances)
