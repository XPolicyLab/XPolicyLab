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

import logging
import os
from typing import Literal

import torch
from safetensors.torch import save_file

logger = logging.getLogger(__name__)

__all__ = ["checkpoint2pretrain", "UnknownCheckpointFormatError"]


class UnknownCheckpointFormatError(Exception):
    pass


def _get_unique_state_dict(model_state_dict):
    """Get a state dictionary with unique tensors."""
    unique_model_state_dict = dict()
    data_ptr_to_key = {}
    for key, tensor in model_state_dict.items():
        data_ptr = tensor.data_ptr()
        if data_ptr not in data_ptr_to_key:
            data_ptr_to_key[data_ptr] = key
            unique_model_state_dict[key] = tensor

    return unique_model_state_dict


def _deepspeed_convert(
    ckpt_path: str,
    target_path: str,
) -> None:
    """Convert a DeepSpeed checkpoint to a standard PyTorch checkpoint.

    Args:
        ckpt_path (str): The path to the DeepSpeed checkpoint.
        target_path (str): The folder path to save the converted checkpoint.

    """

    if os.path.isfile(ckpt_path):
        logger.info(f"Loading model from file: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location="cpu")
    else:
        model_path = os.path.join(
            ckpt_path, "pytorch_model", "mp_rank_00_model_states.pt"
        )
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        logger.info(f"Loading model from file: {model_path}")
        state_dict = torch.load(model_path, map_location="cpu")

    if "module" in state_dict:
        state_dict = state_dict["module"]
    else:
        raise UnknownCheckpointFormatError(
            f"Unknown DeepSpeed checkpoint format: {model_path}. "
            f"Expected 'module' key in state dict."
        )
    state_dict = _get_unique_state_dict(state_dict)
    target_file = os.path.join(target_path, "model.safetensors")
    logger.info(f"Saving the converted pretrain model to {target_file} ...")
    save_file(
        state_dict,
        filename=target_file,
        metadata={"format": "pt"},
    )


def checkpoint2pretrain(
    ckpt_path: str, target_path: str, ckpt_format: Literal["deepspeed"]
) -> None:
    """Convert a checkpoint to a pretrain model.

    Args:
        ckpt_path (str): The path to the checkpoint.
        target_path (str): The target folder to save the pretrain model.
        ckpt_format (Literal["deepspeed"]): The format of the checkpoint.
            Currently only "deepspeed" is supported.
    """
    os.makedirs(target_path, exist_ok=True)

    if ckpt_format == "deepspeed":
        _deepspeed_convert(ckpt_path, target_path)
    else:
        raise UnknownCheckpointFormatError(
            f"Unknown checkpoint format: {ckpt_format}. "
            f"Supported formats: ['deepspeed']"
        )
