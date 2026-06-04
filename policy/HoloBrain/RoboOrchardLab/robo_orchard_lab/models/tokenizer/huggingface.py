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

from torch import Tensor
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from robo_orchard_lab.models.codec import (
    ClassType,
    Codec,
    CodecConfig,
)

__all__ = ["HgTokenizer", "HgTokenizerConfig"]


class HgTokenizer(Codec):
    cfg: HgTokenizerConfig

    def __init__(self, cfg: HgTokenizerConfig):
        self.cfg = cfg
        self._tokenizer = cfg.get_tokenizer()

    def __repr__(self):
        return f"HgTokenizer(tokenizer={self._tokenizer})"

    def encode(self, data: str | list[str]) -> Tensor:
        """Encode data into a list of token IDs.

        Args:
            data (str | list[str]): The input text, pretokenized (list[str])
                strings to encode. It must represent a single sample.

        Returns:
            Tensor: A tensor of token IDs.

        """
        ret = self._tokenizer.encode(text=data, return_tensors="pt")
        return ret[0]  # type: ignore

    def decode(self, data: Tensor) -> str | list[str]:
        """Decode a list of token IDs into a string."""
        if data.dim() == 1:
            return self._tokenizer.decode(data)
        else:
            flattened = data.flatten(end_dim=-2)
            return self._tokenizer.batch_decode(flattened)


class HgTokenizerConfig(CodecConfig[HgTokenizer]):
    """Configuration for the Huggingface tokenizer codec."""

    class_type: ClassType[HgTokenizer] = HgTokenizer

    tokenizer: PreTrainedTokenizerFast | PreTrainedTokenizer | str

    def get_tokenizer(self) -> PreTrainedTokenizerFast | PreTrainedTokenizer:
        if isinstance(
            self.tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)
        ):
            return self.tokenizer
        else:
            return AutoTokenizer.from_pretrained(self.tokenizer)
