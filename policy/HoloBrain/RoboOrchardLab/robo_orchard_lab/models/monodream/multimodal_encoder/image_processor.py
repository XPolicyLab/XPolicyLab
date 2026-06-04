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

# This file was originally copied from the [VILA] repository:
# https://github.com/NVlabs/VILA
# Modifications have been made to fit the needs of this project.
import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL.Image import Image
from transformers.image_processing_utils import (
    BaseImageProcessor,
    BatchFeature,
    get_size_dict,
)
from transformers.image_transforms import (
    convert_to_rgb,
    pad,
    resize,
    to_channel_dimension_format,
)
from transformers.image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
)
from transformers.utils import (
    TensorType,
)


class ImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_pad: bool = True,
        pad_size: int = None,
        pad_multiple: int = None,
        pad_value: Optional[Union[float, List[float]]] = 0.0,
        do_convert_rgb: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        size = size if size is not None else {"longest_edge": 1024}
        size = (
            get_size_dict(max_size=size, default_to_square=False)
            if not isinstance(size, dict)
            else size
        )

        if pad_size is not None and pad_multiple is not None:
            raise ValueError(
                "pad_size and pad_multiple should not be set at the same time."
            )

        pad_size = (
            pad_size
            if pad_size is not None
            else {"height": 1024, "width": 1024}
            if pad_multiple is not None
            else None
        )
        if do_pad:
            pad_size = get_size_dict(pad_size, default_to_square=True)

        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = (
            image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN
        )
        self.image_std = (
            image_std if image_std is not None else IMAGENET_DEFAULT_STD
        )
        self.do_pad = do_pad
        self.pad_multiple = pad_multiple
        self.pad_size = pad_size
        self.pad_value = (
            tuple(pad_value) if isinstance(pad_value, list) else pad_value
        )
        self.do_convert_rgb = do_convert_rgb
        self._valid_processor_keys = [
            "images",
            "segmentation_maps",
            "do_resize",
            "size",
            "resample",
            "do_rescale",
            "rescale_factor",
            "do_normalize",
            "image_mean",
            "image_std",
            "do_pad",
            "pad_size",
            "do_convert_rgb",
            "return_tensors",
            "data_format",
            "input_data_format",
        ]

    def pad_image(
        self,
        image: np.ndarray,
        pad_size: Dict[str, int],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """Padding function.

        Pad an image to `(pad_size["height"], pad_size["width"])` to
        the right and bottom.


        Args:
            image (`np.ndarray`):
                Image to pad.
            pad_size (`Dict[str, int]`):
                Size of the output image after padding.
            data_format (`str` or `ChannelDimension`, *optional*):
                The data format of the image.
                Can be either "channels_first" or "channels_last". If `None`,
                the `data_format` of the `image` will be used.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the input image.
                If not provided, it will be inferred.
        """
        output_height, output_width = pad_size["height"], pad_size["width"]
        input_height, input_width = get_image_size(
            image, channel_dim=input_data_format
        )

        pad_width = output_width - input_width
        pad_height = output_height - input_height

        padded_image = pad(
            image,
            ((0, pad_height), (0, pad_width)),
            data_format=data_format,
            input_data_format=input_data_format,
            constant_values=self.pad_value,
            **kwargs,
        )
        return padded_image

    def _get_preprocess_shape(
        self, old_shape: Tuple[int, int], longest_edge: int
    ):
        oldh, oldw = old_shape
        scale = longest_edge * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        newh = int(newh + 0.5)
        neww = int(neww + 0.5)
        return (newh, neww)

    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        size = get_size_dict(size)
        if "longest_edge" not in size:
            if "width" not in size or "height" not in size:
                raise ValueError(
                    f"The `size` dictionary must contain the key"
                    f"`longest_edge`, or `width` and `height`."
                    f"Got {size.keys()}"
                )
        input_size = get_image_size(image, channel_dim=input_data_format)
        if "longest_edge" in size:
            output_height, output_width = self._get_preprocess_shape(
                input_size, size["longest_edge"]
            )
        else:
            output_height, output_width = size["height"], size["width"]
        return resize(
            image,
            size=(output_height, output_width),
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    def _preprocess(
        self,
        image: ImageInput,
        do_resize: bool,
        do_rescale: bool,
        do_normalize: bool,
        size: Optional[Dict[str, int]] = None,
        resample: PILImageResampling = None,
        rescale_factor: Optional[float] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_pad: Optional[bool] = None,
        pad_size: Optional[Dict[str, int]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        if do_resize:
            image = self.resize(
                image=image,
                size=size,
                resample=resample,
                input_data_format=input_data_format,
            )
        reshaped_input_size = get_image_size(
            image, channel_dim=input_data_format
        )

        if do_rescale:
            image = self.rescale(
                image=image,
                scale=rescale_factor,
                input_data_format=input_data_format,
            )

        if do_normalize:
            image = self.normalize(
                image=image,
                mean=image_mean,
                std=image_std,
                input_data_format=input_data_format,
            )

        if do_pad:
            if self.pad_multiple:
                h, w = get_image_size(image, channel_dim=input_data_format)
                pad_size = {
                    "height": math.ceil(h / self.pad_multiple)
                    * self.pad_multiple,
                    "width": math.ceil(w / self.pad_multiple)
                    * self.pad_multiple,
                }

            image = self.pad_image(
                image=image,
                pad_size=pad_size,
                input_data_format=input_data_format,
            )

        return image, reshaped_input_size

    def _preprocess_image(
        self,
        image: ImageInput,
        do_resize: Optional[bool] = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_rescale: bool = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_pad: Optional[bool] = None,
        pad_size: Optional[Dict[str, int]] = None,
        do_convert_rgb: Optional[bool] = None,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
        if isinstance(image, Image):
            # PIL always uses Channels Last.
            input_data_format = ChannelDimension.LAST

        # PIL RGBA images are converted to RGB
        # mode_before = image.mode
        if do_convert_rgb:
            image = convert_to_rgb(image)

        # All transformations expect numpy arrays.
        image = to_numpy_array(image)

        if len(image.shape) == 2:
            h, w = image.shape
            ret = np.empty((h, w, 3), dtype=np.uint8)
            ret[:, :, 0] = image
            ret[:, :, 1] = image
            ret[:, :, 2] = image
            image = ret
            print(f"preprocess new image shape={image.shape}")
        elif len(image.shape) == 3 and image.shape[-1] == 1:
            ret = np.empty((h, w, 3), dtype=np.uint8)
            ret[:, :, 0] = image[:, :, 0]
            ret[:, :, 1] = image[:, :, 0]
            ret[:, :, 2] = image[:, :, 0]
            image = ret
            print(f"preprocess new image shape={image.shape}")

        if is_scaled_image(image) and do_rescale:
            print(
                "It looks like you are trying to rescale already"
                "rescaled images. If the input images"
                "have pixel values between 0 and 1"
                ",set `do_rescale=False` to avoid rescaling them again."
            )

        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(image)

        original_size = get_image_size(image, channel_dim=input_data_format)

        image, reshaped_input_size = self._preprocess(
            image=image,
            do_resize=do_resize,
            size=size,
            resample=resample,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_pad=do_pad,
            pad_size=pad_size,
            input_data_format=input_data_format,
        )

        if data_format is not None:
            image = to_channel_dimension_format(
                image, data_format, input_channel_dim=input_data_format
            )

        # if image is a single channel convert to rgb
        if do_convert_rgb and image.shape[0] == 1:
            c, h, w = image.shape
            ret = np.empty((3, h, w), dtype=np.uint8)
            ret[0, :, :] = image[0, :, :]
            ret[1, :, :] = image[0, :, :]
            ret[2, :, :] = image[0, :, :]
            image = ret
            print(f"preprocess final: {image.shape}")

        return image, original_size, reshaped_input_size

    def preprocess(
        self,
        images: ImageInput,
        do_resize: Optional[bool] = None,
        size: Optional[Dict[str, int]] = None,
        resample: Optional["PILImageResampling"] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[Union[int, float]] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_pad: Optional[bool] = None,
        pad_size: Optional[Dict[str, int]] = None,
        do_convert_rgb: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ):
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        size = (
            get_size_dict(max_size=size, default_to_square=False)
            if not isinstance(size, dict)
            else size
        )
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = (
            rescale_factor
            if rescale_factor is not None
            else self.rescale_factor
        )
        do_normalize = (
            do_normalize if do_normalize is not None else self.do_normalize
        )
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_pad = do_pad if do_pad is not None else self.do_pad
        pad_size = pad_size if pad_size is not None else self.pad_size
        if do_pad:
            pad_size = get_size_dict(pad_size, default_to_square=True)
        do_convert_rgb = (
            do_convert_rgb
            if do_convert_rgb is not None
            else self.do_convert_rgb
        )

        images = make_list_of_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type"
                "PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        images, original_sizes, reshaped_input_sizes = zip(
            *(
                self._preprocess_image(
                    image=img,
                    do_resize=do_resize,
                    size=size,
                    resample=resample,
                    do_rescale=do_rescale,
                    rescale_factor=rescale_factor,
                    do_normalize=do_normalize,
                    image_mean=image_mean,
                    image_std=image_std,
                    do_pad=do_pad,
                    pad_size=pad_size,
                    do_convert_rgb=do_convert_rgb,
                    data_format=data_format,
                    input_data_format=input_data_format,
                )
                for img in images
            ),
            strict=False,
        )

        data = {
            "pixel_values": images,
            "original_sizes": original_sizes,
            "reshaped_input_sizes": reshaped_input_sizes,
        }

        return BatchFeature(data=data, tensor_type=return_tensors)
