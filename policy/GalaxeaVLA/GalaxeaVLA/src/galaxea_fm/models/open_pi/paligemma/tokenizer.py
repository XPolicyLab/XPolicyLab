"""Tokenizers for OpenPI models (Pi0, Pi0-FAST).

This module contains:
- Pi0Tokenizer: Text tokenizer for Pi0 (instruction only)

Designed for single-sample processing. Returns 1D tensors [seq_len].
DataLoader collation will stack them into [batch, seq_len].
"""

from typing import Dict, Any, List, Tuple

import torch

from galaxea_fm.models.open_pi.paligemma.modules import PaligemmaTokenizer

IGNORE_INDEX = -100


class Pi0Tokenizer:
    """
    Pi0 tokenizer without state discretization.
    Only tokenizes the instruction text.
    Processes single sample, returns 1D tensors.
    """
    def __init__(
        self,
        tokenizer_params: PaligemmaTokenizer,
        pad_token_id: int,
        image_token_index: int,
        max_text_tokens: int,
        num_tokens_per_image: int,
        num_input_images: int,
    ):
        self.tokenizer = PaligemmaTokenizer(**tokenizer_params)
        self.pad_token_id = pad_token_id
        self.image_token_index = image_token_index
        self.max_text_tokens = max_text_tokens
        self.num_input_images = num_input_images
        self.total_image_tokens = self.num_input_images * num_tokens_per_image
        self.max_image_text_tokens = self.total_image_tokens + self.max_text_tokens

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample["input_ids"], sample["labels"], sample["attention_mask"] = self._tokenize(sample["instruction"])
        return sample

    def _tokenize(self, instruction: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tokenize single instruction, returns 1D tensors [seq_len]."""
        # 1. tokenize text
        cleaned_text = instruction.strip().replace("_", " ").replace("\n", " ")
        full_prompt = f"Task: {cleaned_text};\\nAction: "
        tokens = self.tokenizer._tokenizer.encode(full_prompt, add_bos=True)
        tokens_len = len(tokens)
        if tokens_len < self.max_text_tokens:
            padding = [0] * (self.max_text_tokens - tokens_len)
            tokens = tokens + padding
        else:
            tokens = tokens[:self.max_text_tokens]

        input_ids = torch.as_tensor(tokens, dtype=torch.int32)

        # 2. tokenize image tokens
        image_input_ids = torch.tensor([self.image_token_index] * self.total_image_tokens, dtype=torch.int32)

        # 3. merge image_input_ids and text_input_ids
        input_ids = torch.cat([image_input_ids, input_ids])  # 1D: [seq_len]
        labels = torch.full_like(input_ids, fill_value=IGNORE_INDEX)
        attention_mask = input_ids.ne(self.pad_token_id)

        assert input_ids.shape[0] == self.max_image_text_tokens, \
            f"input_ids length {input_ids.shape[0]} does not match max_image_text_tokens {self.max_image_text_tokens}"

        return input_ids, labels, attention_mask
