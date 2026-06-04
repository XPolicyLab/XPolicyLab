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
import io
from typing import Literal, Sequence

import numpy as np
import torch
from pydantic import AliasChoices, Field

from robo_orchard_lab.dataset.datatypes import (
    BatchCameraData,
    BatchCameraDataEncoded,
    BatchImageData,
    ImageMode,
)
from robo_orchard_lab.transforms.base import (
    ClassType,
    DictTransform,
    DictTransformConfig,
)

__all__ = [
    "ImageDecode",
    "ImageDecodeConfig",
]


class ImageDecode(DictTransform[dict[str, BatchCameraData]]):
    """A transform to decode BatchCameraDataEncoded to BatchCameraData."""

    cfg: ImageDecodeConfig
    is_variadic: bool = True

    def __init__(self, cfg: ImageDecodeConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def transform(self, **kwargs) -> dict[str, BatchCameraData]:
        """Decode the image data from bytes to PIL images."""
        if self.cfg.backend == "pil":
            # return img.decode(self._decode_impl_pil)
            impl = self._decode_impl_pil
        elif self.cfg.backend == "cv2":
            # return img.decode(self._decode_impl_cv2)
            impl = self._decode_impl_cv2
        else:
            raise ValueError(f"Unsupported backend: {self.cfg.backend}")

        ret = {}
        for key, value in kwargs.items():
            if isinstance(value, BatchCameraDataEncoded):
                # Decode the image data
                assert isinstance(value, BatchCameraDataEncoded)
                out = value.decode(impl)
                # handle RGB/BGR inversion if needed
                if self.cfg.invert_rgb and out.pix_fmt == ImageMode.RGB:
                    out.pix_fmt = ImageMode.BGR
                elif self.cfg.invert_rgb and out.pix_fmt == ImageMode.BGR:
                    out.pix_fmt = ImageMode.RGB
                ret[key] = out
            else:
                raise TypeError(
                    f"Expected BatchCameraDataEncoded for key '{key}', "
                    f"but got {type(value)}."
                )
        return ret

    def _decode_impl_pil(
        self, list_data: list[bytes], format: str
    ) -> BatchImageData:
        """Decode the image data from bytes to PIL images."""
        from PIL import Image as PILImage

        img_mode: ImageMode | None = None
        sensor_data = []
        for data in list_data:
            img = PILImage.open(io.BytesIO(data))
            # # convert img to numpy
            img_tensor = torch.asarray(np.array(img))
            if img_mode is None:
                img_mode = ImageMode(img.mode)
            else:
                assert img_mode == ImageMode(img.mode), (
                    "All images must have the same mode"
                )

            if img_tensor.ndim == 2:
                img_tensor = img_tensor.unsqueeze(-1)
            assert img_tensor.ndim == 3, "Image tensor must be 3D (H, W, C)"
            sensor_data.append(img_tensor)

        return BatchImageData(
            sensor_data=torch.stack(sensor_data, dim=0),
            pix_fmt=img_mode,
        )

    def _decode_impl_cv2(
        self, data_list: list[bytes], format: str
    ) -> BatchImageData:
        """Decode the image data from bytes to OpenCV images."""
        import cv2

        def get_image_mode(img_tensor: torch.Tensor) -> ImageMode:
            if img_tensor.shape[-1] == 3 and img_tensor.dtype == torch.uint8:
                ret_mode = ImageMode.BGR
            elif img_tensor.shape[-1] == 1 and img_tensor.dtype == torch.uint8:
                ret_mode = ImageMode.L
            elif (
                img_tensor.shape[-1] == 1 and img_tensor.dtype == torch.uint16
            ):
                ret_mode = ImageMode.I16
            else:
                raise ValueError(
                    f"Unsupported image format: {img_tensor.shape} "
                    f"with dtype {img_tensor.dtype}"
                )
            return ret_mode

        img_mode: ImageMode | None = None
        sensor_data = []

        for data in data_list:
            nparr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            img_tensor = torch.from_numpy(img)

            if img_tensor.ndim == 2:
                img_tensor = img_tensor.unsqueeze(-1)

            if img_mode is None:
                img_mode = get_image_mode(img_tensor)
            else:
                assert img_mode == get_image_mode(img_tensor), (
                    "All images must have the same mode"
                )
            assert img_tensor.ndim == 3, "Image tensor must be 3D (H, W, C)"
            sensor_data.append(img_tensor)

        return BatchImageData(
            sensor_data=torch.stack(sensor_data, dim=0),
            pix_fmt=img_mode,
        )


class ImageDecodeConfig(DictTransformConfig[ImageDecode]):
    class_type: ClassType[ImageDecode] = ImageDecode

    backend: Literal["pil", "cv2"] = "pil"

    input_columns: Sequence[str] | None = Field(
        default=None,
        validation_alias=AliasChoices("input_column_mapping", "input_columns"),
        description="The columns to decode.",
    )

    invert_rgb: bool = False
    """Whether to invert RGB to BGR.

    For some datasets, the actual image data may be incorrectly stored as RGB
    but should be treated as BGR. This flag allows to invert the channels
    accordingly.
    """

    def __post_init__(self):
        super().__post_init__()
        if self.input_columns is None:
            raise ValueError("input_columns must be provided.")
