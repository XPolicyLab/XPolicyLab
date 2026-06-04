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
import warnings
from abc import abstractmethod
from typing import TypeVar

import numpy as np
import torch
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

from robo_orchard_lab.models.codec import (
    ClassType,
    Codec,
    CodecConfig,
)

__all__ = [
    "ToTextCodec",
    "ToTextCodecT_co",
    "ToTextCodecConfig",
    "ToTextCodecConfigT_co",
    "DCTAction2Text",
    "DCTAction2TextConfig",
    "train_bbpe",
]


class ToTextCodec(Codec):
    @property
    @abstractmethod
    def special_tokens(self) -> list[str]:
        """List of special tokens used in the encoding/decoding process."""
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def get_alphabet_wo_special_tokens(self) -> list[str]:
        """Get the alphabet used for encoding, excluding special tokens."""
        raise NotImplementedError("Subclasses must implement this method.")


ToTextCodecT_co = TypeVar("ToTextCodecT_co", bound=ToTextCodec, covariant=True)


class ToTextCodecConfig(CodecConfig[ToTextCodecT_co]):
    """Configuration for the ToTextCodecMixin codec."""

    pass


ToTextCodecConfigT_co = TypeVar(
    "ToTextCodecConfigT_co",
    bound=ToTextCodecConfig[ToTextCodec],
    covariant=True,
)


class DCTAction2Text(ToTextCodec):
    cfg: DCTAction2TextConfig

    def __init__(self, cfg: DCTAction2TextConfig):
        self.cfg = cfg

    def get_alphabet_wo_special_tokens(self) -> list[str]:
        """Get the alphabet used for encoding, excluding special tokens."""
        return [chr(i) for i in range(self.cfg.max_token + 1)]

    @property
    def special_tokens(self) -> list[str]:
        return [self.cfg.arm_token, self.cfg.gripper_token]

    @property
    def arm_token(self) -> str:
        return self.cfg.arm_token

    @property
    def gripper_token(self) -> str:
        return self.cfg.gripper_token

    def encode(self, data: np.ndarray | torch.Tensor) -> str:
        """Encode a 2D array of actions into a string.

        Args:
            data (np.ndarray): A 2D array of shape (timestamps, action_dim).
                The input data should be of type int32, and should be in
                the range of [0, max_token].

        Returns:
            str: A string representation of the actions, with arm and gripper
                tokens separating the arm and gripper actions. For example,
                the action data [T]x[D] after DCT on time dimension is
                [[1, 2, 3], [5, 6, 7]],  and if `gripper_dim` is 1, the output
                will be flattened as column-first string like
                "<arm>chr(1)chr(2)chr(5)chr(6)<gripper>chr(3)chr(7)".
                The arm actions are encoded as a string of characters, and the
                gripper actions are encoded as a string of characters after
                the gripper.

        """

        if isinstance(data, torch.Tensor):
            data = data.numpy()

        if data.dtype != np.int32:
            raise ValueError("Input data must be of type int32")

        if data.ndim != 2:
            raise ValueError(
                "Input data must be a 2D array (timestamps, action_dim)"
            )

        ret = []
        ret.append(self.arm_token)
        arm_token = "".join(
            map(chr, data[:, : -self.cfg.gripper_dim].flatten())
        )
        ret.append(arm_token)
        ret.append(self.gripper_token)
        gripper_token = "".join(
            map(chr, data[:, -self.cfg.gripper_dim :].flatten())
        )
        ret.append(gripper_token)
        return "".join(ret)

    def decode_sample(self, data: str) -> torch.Tensor:
        """Decode a string representation of actions into a 2D array.

        Args:
            data (str): A string representation of the actions, with arm and
                gripper tokens separating the arm and gripper actions. For
                example, "<arm>chr(1)chr(2)chr(5)chr(6)<gripper>chr(3)chr(7>".

        Returns:
            torch.Tensor: A 2D array of shape (timestamps, action_dim), where
                `action_dim` is the sum of arm and gripper dimensions.
        """
        if not isinstance(data, str):
            raise ValueError(
                f"Input data must be a string but got {type(data)}"
            )

        if self.arm_token not in data or self.gripper_token not in data:
            raise ValueError("Input data must contain arm and gripper tokens")

        parts = data.split(self.arm_token)
        if len(parts) != 2:
            raise ValueError("Input data must contain exactly one arm token")

        arm_part = parts[1].split(self.gripper_token)
        if len(arm_part) != 2:
            raise ValueError(
                "Input data must contain exactly one gripper token"
            )

        arm_data = np.array([ord(c) for c in arm_part[0]], dtype=np.int32)
        gripper_data = np.array([ord(c) for c in arm_part[1]], dtype=np.int32)
        action_data = np.concatenate(
            [
                arm_data.reshape(
                    -1, self.cfg.action_dim - self.cfg.gripper_dim
                ),
                gripper_data.reshape(-1, self.cfg.gripper_dim),
            ],
            axis=1,
        )

        return torch.from_numpy(action_data)

    def decode(self, data: str | list[str]) -> torch.Tensor:
        """Decode a string or list of strings into a 2D or 3D array.

        Args:
            data (str | list[str]): A string or list of strings representing
                the actions.

        Returns:
            torch.Tensor: A 2D array of shape (timestamps, action_dim) if
                input is a single string, or a 3D array of shape
                (batch, timestamps, action_dim) if input is a list of strings.
        """
        if isinstance(data, str):
            return self.decode_sample(data)
        elif isinstance(data, list):
            decoded = [self.decode_sample(d) for d in data]
            return torch.stack(decoded, dim=0)
        else:
            raise ValueError(
                f"Input data must be a string or list of strings but got "
                f"{type(data)}"
            )


class DCTAction2TextConfig(ToTextCodecConfig[DCTAction2Text]):
    """Configuration for the DCTAction2Text codec."""

    class_type: ClassType[DCTAction2Text] = DCTAction2Text

    action_dim: int
    """The total dimension of the action data, including arm and gripper."""
    gripper_dim: int
    """The dimension of the gripper action data."""
    max_token: int
    """The maximum token value for encoding the action data.

    This should be set to the maximum value of the action data after DCT."""

    arm_token: str = "<arm>"
    """Token to separate arm actions in the encoded string."""
    gripper_token: str = "<gripper>"
    """Token to separate gripper actions in the encoded string."""

    def __post_init__(self):
        if self.gripper_dim < 0:
            raise ValueError("Gripper dimension must be non-negative.")
        if self.action_dim <= 0:
            raise ValueError("Action dimension must be a positive integer.")
        if self.gripper_dim >= self.action_dim:
            raise ValueError(
                "Gripper dimension must be less than action dimension."
            )


def train_bbpe(
    action_data: np.ndarray | list[np.ndarray],
    action_codec_cfg: CodecConfig,
    to_text_codec_cfg: ToTextCodecConfig,
    vocab_size: int = 1024,
    min_frequency=2,
    max_token_length=10000,
    show_progress=True,
) -> PreTrainedTokenizerFast:
    """Train a Byte-Pair Encoding (BPE) tokenizer on the provided action data.

    Args:
        action_data (np.ndarray | list[np.ndarray]): The action data to train
            the tokenizer on (3D array or list of 2D arrays). The data should
            be in the shape of (batch, timestamps, action_dim) for 3D.
        action_codec (CodecMixin): The codec to encode the action data into
            a new format that `to_text_codec` can handle.
        to_text_codec (ToTextCodecMixin): The codec to convert the encoded
            action data into a text format suitable for BPE training.
        vocab_size (int, optional): The size of the vocabulary to be created.
            Defaults to 1024.
        min_frequency (int, optional): The minimum frequency of tokens to be
            included in the vocabulary. Defaults to 2.
        max_token_length (int, optional): The maximum length of tokens to be
            considered during training. Defaults to 10000.
        show_progress (bool, optional): Whether to show the training progress.
            Defaults to True.

    """
    to_text_codec = to_text_codec_cfg()
    action_codec = action_codec_cfg()

    def make_data_iter():
        for _, inst in enumerate(action_data):
            inst = torch.asarray(inst, dtype=torch.int32)
            yield to_text_codec.encode(action_codec.encode(inst))

    initial_alphabet = to_text_codec.get_alphabet_wo_special_tokens()
    special_tokens = to_text_codec.special_tokens

    init_vocab_size = len(initial_alphabet) + len(special_tokens)
    if init_vocab_size >= vocab_size:
        raise ValueError(
            f"Initial alphabet size {init_vocab_size} is too large for "
            f"vocab size {vocab_size}. Consider reducing the number of "
            f"special tokens or the max token value."
        )
    warning_thresh_vocab_size = int(0.8 * vocab_size)
    if init_vocab_size > warning_thresh_vocab_size:
        warnings.warn(
            f"Initial alphabet size {init_vocab_size} is greater than 80% "
            f"of the target vocab size {vocab_size}. This may lead to "
            f"suboptimal tokenization results."
        )

    trainer = BpeTrainer(
        vocab_size=vocab_size,  # type: ignore
        min_frequency=min_frequency,  # type: ignore
        initial_alphabet=initial_alphabet,  # type: ignore
        show_progress=show_progress,  # type: ignore
        special_tokens=special_tokens,  # type: ignore
        max_token_length=max_token_length,  # type: ignore
    )
    bpe = ByteLevelBPETokenizer()

    bpe._tokenizer.train_from_iterator(make_data_iter(), trainer=trainer)
    return PreTrainedTokenizerFast(
        tokenizer_object=bpe, clean_up_tokenization_spaces=False
    )
