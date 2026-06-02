"""
This file is based on work from smollm (https://github.com/huggingface/smollm),
licensed under the MIT License.

Modifications:
   Copyright (c) 2025 Galaxea AI.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

from typing import Dict, Any, Optional, Literal, List

import torch
from transformers import AutoTokenizer

IGNORE_INDEX = -100


class SmolVLM2Tokenizer:
    def __init__(
        self,
        tokenizer_params: Dict[str, Any],
        pad_token_id: int,
        image_token_index: int,
        max_text_tokens: int,
        num_tokens_per_image: int,
        num_input_images: int,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(**tokenizer_params)
        self.tokenizer.pad_token_id = pad_token_id
        self.pad_token_id = pad_token_id
        self.image_token_index = image_token_index
        self.max_text_tokens = max_text_tokens
        self.num_input_images = num_input_images
        self.num_tokens_per_image = num_tokens_per_image
        self.total_image_tokens = self.num_input_images * (num_tokens_per_image + 3)
        self.max_image_text_tokens = self.total_image_tokens + self.max_text_tokens
        self.global_image_token_index = 49152
        self.fake_token_around_image_token_index = 49189

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample["input_ids"], sample["labels"], sample["attention_mask"] = self._tokenize(sample["instruction"])
        return sample

    def _prompt_images(self) -> List[int]:
        single_image_input_ids = [
            self.fake_token_around_image_token_index,
            self.global_image_token_index,
        ] + [self.image_token_index] * self.num_tokens_per_image + [self.fake_token_around_image_token_index]
        image_input_ids = single_image_input_ids * self.num_input_images

        return image_input_ids

    def _tokenize(self, instructions: List[str]) -> List[torch.Tensor]:
        if isinstance(instructions, str):
            instructions = [instructions]
        
        PROMPT_TEMPLATE = '{bos_token}User: {instruction}<end_of_utterance>\nAssistant:'
        instructions = [PROMPT_TEMPLATE.format(bos_token=self.tokenizer.bos_token, instruction=instruct) for instruct in instructions]
        input_text = self.tokenizer(
            instructions,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )

        # 1. tokenize text instruction
        text_input_ids = input_text.input_ids # [batch_size, text_seq_len]
        attention_mask = input_text.attention_mask # [batch_size, text_seq_len]
        labels = torch.full_like(text_input_ids, fill_value=IGNORE_INDEX) # [batch_size, text_seq_len]

        batch_size, current_length = text_input_ids.shape
        # pad text_input_ids to max_text_tokens
        if current_length < self.max_text_tokens:
            padding_length = self.max_text_tokens - current_length
            text_input_ids = torch.nn.functional.pad(text_input_ids, (0, padding_length), value=self.pad_token_id)
            labels = torch.nn.functional.pad(labels, (0, padding_length), value=IGNORE_INDEX)
        else:
            text_input_ids = text_input_ids[..., :self.max_text_tokens]
            labels = labels[..., :self.max_text_tokens]
        
        # 2. fill in image tokens
        image_input_ids = self._prompt_images()
        image_input_ids = torch.tensor(image_input_ids)
        image_input_ids = image_input_ids.unsqueeze(0).repeat(batch_size, 1)

        # 3. merge text_input_ids and image_input_ids
        input_ids = torch.cat([text_input_ids[:, :3], image_input_ids, text_input_ids[:, 3:]], dim=1) # [batch_size, max_imag_text_tokens]
        labels = torch.full_like(input_ids, fill_value=IGNORE_INDEX)
        attention_mask = input_ids.ne(self.pad_token_id)

        assert input_ids.shape[1] == self.max_image_text_tokens, \
            f"Input_ids length {input_ids.shape[1]} does not match max_image_text_tokens {self.max_image_text_tokens}"
        
        return input_ids.squeeze(0), labels.squeeze(0), attention_mask.squeeze(0)
