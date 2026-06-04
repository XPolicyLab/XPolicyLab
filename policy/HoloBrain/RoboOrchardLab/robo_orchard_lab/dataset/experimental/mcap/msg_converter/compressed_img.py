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
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from foxglove_schemas_protobuf.CompressedImage_pb2 import CompressedImage
from google.protobuf.timestamp import from_seconds
from google.protobuf.timestamp_pb2 import Timestamp

from robo_orchard_lab.dataset.experimental.mcap.msg_converter.base import (
    MessageConverterConfig,
    MessageConverterStateless,
)

__all__ = [
    "NumpyImageMsg",
    "Numpy2CompressedImage",
    "Numpy2CompressedImageConfig",
    "CompressedImage2Numpy",
    "CompressedImage2NumpyConfig",
]


@dataclass
class NumpyImageMsg:
    data: np.ndarray
    """Message class for storing numpy image data."""

    target_format: Literal["jpeg", "png"]
    """Target format for the image, e.g., 'jpg' or 'png'."""

    frame_id: str = ""
    """Frame ID for the image."""

    timestamp: Timestamp | None = None
    """Timestamp for the image, defaults to zero if not provided."""


class Numpy2CompressedImage(
    MessageConverterStateless[NumpyImageMsg, CompressedImage]
):
    """Convert numpy array to CompressedImage."""

    def __init__(self, cfg: Numpy2CompressedImageConfig | None = None):
        """Initialize the converter with optional quality parameters."""
        if cfg is None:
            cfg = Numpy2CompressedImageConfig()

        self.jpeg_quality = cfg.jpeg_quality
        self.png_compression = cfg.png_compression

    def convert(self, data: NumpyImageMsg) -> CompressedImage:
        """Convert numpy array to CompressedImage."""

        if data.timestamp is None:
            data.timestamp = from_seconds(0)

        encode_param = (
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            if data.target_format == "jpg"
            else [int(cv2.IMWRITE_PNG_COMPRESSION), self.png_compression]
        )

        cv2_encode_success, encoded_img = cv2.imencode(
            f".{data.target_format}", data.data, encode_param
        )
        if not cv2_encode_success:
            raise ValueError(
                f"Failed to encode image to {data.target_format} format."
            )
        compressed_image = CompressedImage(
            data=encoded_img.tobytes(),
            format=data.target_format,
            frame_id=data.frame_id,
            timestamp=data.timestamp,
        )
        return compressed_image


class Numpy2CompressedImageConfig(
    MessageConverterConfig[Numpy2CompressedImage]
):
    """Configuration class for Numpy2CompressedImage."""

    class_type: type[Numpy2CompressedImage] = Numpy2CompressedImage

    jpeg_quality: int = 90
    """JPEG quality for encoding."""

    png_compression: int = 3
    """PNG compression level for encoding."""


class CompressedImage2Numpy(
    MessageConverterStateless[CompressedImage, NumpyImageMsg]
):
    """Convert CompressedImage to numpy array."""

    def __init__(self, cfg: CompressedImage2NumpyConfig | None = None):
        """Initialize the converter."""
        if cfg is None:
            cfg = CompressedImage2NumpyConfig()

    def convert(self, data: CompressedImage) -> NumpyImageMsg:
        """Convert CompressedImage to numpy array."""

        img = cv2.imdecode(
            np.frombuffer(data.data, np.uint8), cv2.IMREAD_UNCHANGED
        )
        return NumpyImageMsg(
            data=img,  # type: ignore
            target_format=data.format,  # type: ignore
            frame_id=data.frame_id,
            timestamp=data.timestamp if data.HasField("timestamp") else None,
        )


class CompressedImage2NumpyConfig(
    MessageConverterConfig[CompressedImage2Numpy]
):
    """Configuration class for CompressedImage2Numpy."""

    class_type: type[CompressedImage2Numpy] = CompressedImage2Numpy
