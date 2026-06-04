# Project RoboOrchard
#
# Copyright (c) 2025 Horizon Robotics. All Rights Reserved.
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
from typing import Literal, TypeVar

import torch
from foxglove_schemas_protobuf.CameraCalibration_pb2 import (
    CameraCalibration as FgCameraCalibration,
)
from foxglove_schemas_protobuf.CompressedImage_pb2 import (
    CompressedImage as FgCompressedImage,
)
from foxglove_schemas_protobuf.FrameTransform_pb2 import (
    FrameTransform as FgFrameTransform,
)
from foxglove_schemas_protobuf.RawImage_pb2 import RawImage as FgRawImage
from google.protobuf.timestamp import from_nanoseconds

from robo_orchard_lab.dataset.datatypes.camera import (
    BatchCameraData,
    BatchCameraDataEncoded,
    BatchImageData,
    Distortion,
    ImageChannelLayout,
    ImageMode,
)
from robo_orchard_lab.dataset.experimental.mcap.msg_converter.base import (
    MessageConverterConfig,
    MessageConverterStateless,
)
from robo_orchard_lab.dataset.experimental.mcap.msg_converter.frame_transform import (  # noqa: E501
    ToBatchFrameTransformConfig,
)
from robo_orchard_lab.transforms.image import ImageDecodeConfig
from robo_orchard_lab.transforms.image.formatter import (
    ImageLayoutFormatterConfig,
)

__all__ = [
    "FgCameraCompressedImages",
    "ToBatchCameraData",
    "ToBatchCameraDataEncoded",
    "ToBatchCameraDataConfig",
    "ToBatchCameraDataEncodedConfig",
    "FromBatchImageData",
    "FromBatchImageDataConfig",
]


@dataclass
class FgCameraCompressedImages:
    images: list[FgCompressedImage]
    """List of compressed images."""

    calib: list[FgCameraCalibration] | FgCameraCalibration | None = None
    """Calibration data associated with the images."""

    tf: FgFrameTransform | list[FgFrameTransform] | None = None
    """Frame transform associated with the images."""

    def __post_init__(self):
        if self.tf is not None and not isinstance(self.tf, list):
            for img in self.images:
                if img.frame_id != self.tf.child_frame_id:
                    raise ValueError(
                        "All images must have the same frame_id as the "
                        "child_frame_id of the FrameTransform."
                    )
        if isinstance(self.tf, list):
            for img, tf in zip(self.images, self.tf, strict=True):
                if img.frame_id != tf.child_frame_id:
                    raise ValueError(
                        "All images must have the same frame_id as the "
                        "child_frame_id of the FrameTransform."
                    )
        if isinstance(self.calib, list) and len(self.calib) == 1:
            self.calib = self.calib[0]
        if isinstance(self.calib, list) and len(self.calib) != len(
            self.images
        ):
            raise ValueError(
                "If calib is a list, it must have the same length as images."
            )


class ToBatchCameraDataEncoded(
    MessageConverterStateless[FgCameraCompressedImages, BatchCameraDataEncoded]
):
    def __init__(self, cfg: ToBatchCameraDataEncodedConfig | None = None):
        """Initialize the converter."""
        if cfg is None:
            cfg = ToBatchCameraDataEncodedConfig()

        self._cfg = cfg
        self._to_tf = ToBatchFrameTransformConfig()()

    def convert(self, src: FgCameraCompressedImages) -> BatchCameraDataEncoded:
        # img_byte_list = [img.data for img in src.images]
        timestamps: list[int] | None = []
        for img in src.images:
            if img.frame_id != src.images[0].frame_id:
                raise ValueError(
                    "All images must have the same frame_id, "
                    f"but got {img.frame_id} and {src.images[0].frame_id}."
                )
            if img.format != src.images[0].format:
                raise ValueError(
                    "All images must have the same format, "
                    f"but got {img.format} and {src.images[0].format}."
                )
            timestamps.append(img.timestamp.ToNanoseconds())

        if all(timestamp is None for timestamp in timestamps):
            timestamps = None

        ret = BatchCameraDataEncoded(
            sensor_data=[img.data for img in src.images],
            frame_id=src.images[0].frame_id,
            format=src.images[0].format,  # type: ignore
            image_shape=None,
            timestamps=timestamps,
        )
        self._set_camera_calib(calib=src.calib, target=ret)
        self._set_tf(tf=src.tf, target=ret, timestamps=timestamps)
        return ret

    def _set_tf(
        self,
        tf: FgFrameTransform | list[FgFrameTransform] | None,
        timestamps: list[int] | None,
        target: BatchCameraData | BatchCameraDataEncoded,
    ):
        """Set the frame transform."""
        if tf is None:
            target.pose = None
            return
        target.pose = self._to_tf.convert(tf)

        if target.pose.batch_size == 1 and target.batch_size > 1:
            target.pose = target.pose.repeat(
                target.batch_size, timestamps=timestamps
            )

    def _set_camera_calib(
        self,
        calib: FgCameraCalibration | list[FgCameraCalibration] | None,
        target: BatchCameraData | BatchCameraDataEncoded,
    ) -> None:
        """Set the camera calibration."""
        if calib is None:
            return
        if isinstance(calib, list):
            if len(calib) == 0:
                return
            if len(calib) == 1:
                calib = [calib[0]] * len(target.sensor_data)
            if len(calib) != len(target.sensor_data):
                raise ValueError(
                    "If calib is a list, it must have the same length as "
                    "sensor_data."
                    f" Got {len(calib)} and {len(target.sensor_data)}."
                )
            for c in calib:
                if c.height != calib[0].height:
                    raise ValueError(
                        "All camera calibrations must have the same height, "
                        f"but got {c.height} and {calib[0].height}."
                    )
                if c.width != calib[0].width:
                    raise ValueError(
                        "All camera calibrations must have the same width, "
                        f"but got {c.width} and {calib[0].width}."
                    )
                if c.distortion_model != calib[0].distortion_model:
                    raise ValueError(
                        "All camera calibrations must have the same "
                        f"distortion model, but got {c.distortion_model} "
                        f"and {calib[0].distortion_model}."
                    )
                if c.K != calib[0].K:
                    raise ValueError(
                        "All camera calibrations must have the same "
                        f"intrinsic matrix, but got {c.K} and {calib[0].K}."
                    )
            calibs = calib
        else:
            calibs = [calib] * len(target.sensor_data)

        if target.image_shape is None:
            target.image_shape = (
                calibs[0].height,
                calibs[0].width,
            )
        else:
            if target.image_shape != (calibs[0].height, calibs[0].width):
                raise ValueError(
                    f"Image shape {target.image_shape} does not match "
                    f"calibration shape {(calibs[0].height, calibs[0].width)}."
                )
        target.intrinsic_matrices = torch.stack(
            [
                torch.tensor(c.K, dtype=torch.float32).reshape(3, 3)
                for c in calibs
            ]
        )

        target.distortion = Distortion(
            model=calibs[0].distortion_model,  # type: ignore
            coefficients=torch.tensor(calibs[0].D, dtype=torch.float32),
        )


class ToBatchCameraData(
    MessageConverterStateless[FgCameraCompressedImages, BatchCameraData]
):
    """Convert FgCameraCompressedImages to BatchCameraDataStamped."""

    def __init__(self, cfg: ToBatchCameraDataConfig | None = None):
        """Initialize the converter."""
        if cfg is None:
            cfg = ToBatchCameraDataConfig()

        self._cfg = cfg
        self._to_encoded = ToBatchCameraDataEncodedConfig()()

        self._decode_transform = ImageDecodeConfig(
            input_columns=["image"], backend=self._cfg.backend
        )()

    def convert(self, src: FgCameraCompressedImages) -> BatchCameraData:
        ret = self._to_encoded.convert(src)

        return self._decode_transform.transform(image=ret)["image"]


T = TypeVar("T")


class ToBatchCameraDataEncodedConfig(
    MessageConverterConfig[ToBatchCameraDataEncoded],
):
    """Configuration class for CameraMsgs2BatchCameraData."""

    class_type: type[ToBatchCameraDataEncoded] = ToBatchCameraDataEncoded


class ToBatchCameraDataConfig(
    MessageConverterConfig[ToBatchCameraData],
):
    """Configuration class for CameraMsgs2BatchCameraData."""

    class_type: type[ToBatchCameraData] = ToBatchCameraData

    backend: Literal["pil", "cv2"] = "cv2"


class FromBatchImageData(
    MessageConverterStateless[BatchImageData, list[FgRawImage]]
):
    _cfg: FromBatchImageDataConfig

    def __init__(self, cfg: FromBatchImageDataConfig):
        self._cfg = cfg

        self._layout_formatter = ImageLayoutFormatterConfig(
            output_layout=ImageChannelLayout.HWC,
            input_columns=["image"],
        )()

    def convert(self, data: BatchImageData) -> list[FgRawImage]:
        cur_data = data
        if cur_data.pix_fmt is None:
            raise ValueError("BatchImageData must have a pixel format.")
        # check channel layout
        if cur_data.channel_layout != ImageChannelLayout.HWC:
            cur_data: BatchImageData = self._layout_formatter.__call__(
                {"image": cur_data}
            )["image"]
        if cur_data.pix_fmt is None:
            raise ValueError("BatchImageData must have a pixel format.")
        batch_size = cur_data.sensor_data.shape[0]
        height, width = cur_data.sensor_data.shape[1:3]

        ret = []
        if cur_data.timestamps is None:
            raise ValueError("BatchCameraDataEncoded must have timestamps.")
        for i in range(batch_size):
            # convert the i-th image to bytes
            img_data = cur_data.sensor_data[i].contiguous().view(-1)
            encoding, data_width, data_bytes = self._pix_fmt2raw_encoding(
                cur_data.pix_fmt, img_data
            )
            img = FgRawImage(
                timestamp=from_nanoseconds(cur_data.timestamps[i]),
                frame_id="",
                width=width,
                height=height,
                encoding=encoding,
                step=width * data_width,
                data=data_bytes,
            )
            ret.append(img)

        return ret

    def _pix_fmt2raw_encoding(
        self, pix_fmt: ImageMode, data: torch.Tensor
    ) -> tuple[str, int, bytes]:
        """Convert ImageMode to raw image encoding string and data width in bytes.

        If the pixel format is not supported, raise a ValueError.
        """  # noqa: E501
        if pix_fmt == ImageMode.RGB:
            return "rgb8", 3, data.to(dtype=torch.uint8).numpy().tobytes()
        elif pix_fmt == ImageMode.BGR:
            return "bgr8", 3, data.to(dtype=torch.uint8).numpy().tobytes()
        elif pix_fmt == ImageMode.F:
            return "32FC1", 4, data.to(dtype=torch.float32).numpy().tobytes()
        elif pix_fmt == ImageMode.L:
            return "mono8", 1, data.to(dtype=torch.uint8).numpy().tobytes()
        elif pix_fmt == ImageMode.I16:
            return "mono16", 2, data.to(dtype=torch.uint16).numpy().tobytes()
        else:
            raise ValueError(f"Unsupported pixel format: {pix_fmt}.")


class FromBatchImageDataConfig(MessageConverterConfig[FromBatchImageData]):
    class_type: type[FromBatchImageData] = FromBatchImageData
