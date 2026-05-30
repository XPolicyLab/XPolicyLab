"""
Visualizations
"""

from __future__ import annotations

import os
import time
from typing import Literal
import warnings

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch

from .array_tensor_utils import any_describe
from .misc_utils import global_once
from .torch_utils import torch_normalize


def to_image(img, channel_order="auto"):
    """
    Returns:
        numpy image of shape [H, W, C]
        in "auto" mode, we assume C == 3
    """
    assert channel_order in ["hwc", "chw", "auto"]
    if torch.is_tensor(img):
        img = img.cpu().numpy()
    assert isinstance(img, np.ndarray)
    if img.ndim == 4:
        assert img.shape[0] == 1
        img = img[0]
    assert img.ndim == 3
    if channel_order == "auto":
        # use C==3 to detect order
        if img.shape[0] == 3:
            channel_order = "chw"
        else:
            assert img.shape[-1] == 3, "image should either have [3,H,W] or [H,W,3]"
            channel_order = "hwc"
    img = img.astype(np.uint8)
    if channel_order == "chw":
        return np.transpose(img, (1, 2, 0))
    else:
        return img


def imshow(img):
    plt.imshow(to_image(img))


def imsave(img, path):
    imageio.imsave(os.path.expanduser(path), to_image(img))


def imread(path, channel_order="chw", format="torch"):
    assert channel_order in ["hwc", "chw"]
    assert format in ["numpy", "torch"]
    img = imageio.imread(path)
    if channel_order == "chw":
        img = np.transpose(img, (2, 0, 1))  # hwc -> chw
    if format == "torch":
        return torch.from_numpy(img)
    else:
        return img


class Cv2Display:
    def __init__(
        self,
        window_name="display",
        image_size=None,
        channel_order="auto",
        bgr2rgb=True,
        step_sleep=0,
        enabled=True,
    ):
        """
        Use cv2.imshow() to pop a window, requires virtual desktop GUI

        Args:
            channel_order: auto, hwc, or chw
            image_size: None to use the original image size, otherwise resize
            step_sleep: sleep for a few seconds
        """
        self._window_name = window_name
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        else:
            assert image_size is None or len(image_size) == 2
        self._image_size = image_size
        assert channel_order in ["auto", "chw", "hwc"]
        self._channel_order = channel_order
        self._bgr2rgb = bgr2rgb
        self._step_sleep = step_sleep
        self._enabled = enabled

    def _resize(self, img):
        if self._image_size is None:
            return img
        H, W = img.shape[:2]
        Ht, Wt = self._image_size  # target
        return cv2.resize(
            img,
            self._image_size,
            interpolation=cv2.INTER_AREA if Ht < H else cv2.INTER_LINEAR,
        )

    def _reorder(self, img):
        if self._channel_order == "chw":
            return np.transpose(img, (1, 2, 0))
        elif self._channel_order == "hwc":
            return img
        else:
            if img.shape[0] in [1, 3]:  # chw
                return np.transpose(img, (1, 2, 0))
            else:
                return img

    def __call__(self, img):
        if not self._enabled:
            return
        import torch

        # prevent segfault in IsaacGym
        display_var = os.environ.get("DISPLAY", None)
        if not display_var:
            os.environ["DISPLAY"] = ":0.0"

        if torch.is_tensor(img):
            img = img.detach().cpu().numpy()

        img = self._resize(self._reorder(img))
        if self._bgr2rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        time.sleep(self._step_sleep)
        cv2.imshow(self._window_name, img)
        cv2.waitKey(1)

        if display_var is not None:
            os.environ["DISPLAY"] = display_var

    def close(self):
        if not self._enabled:
            return
        cv2.destroyWindow(self._window_name)


# ---------------- Image tensor handling -----------------
def sanity_check_image_tensor(
    img: torch.Tensor, on_error: Literal["raise", "warn", "ignore"] = "raise"
):
    """
    Check if the input image tensor is all integers, which is wrong for any NN input.
    This is a common case if the user forgets to normalize the image first
    """
    assert on_error in [
        "raise",
        "warn",
        "ignore",
    ], 'on_error must be "raise", "warn", or "ignore"'
    if not img.dtype.is_floating_point:
        msg = f"Image tensor is not floating point format, but {img.dtype}!"
        if on_error == "raise":
            raise ValueError(msg)
        elif on_error == "warn":
            warnings.warn(msg)
        else:
            return False
    # check if all values in the image are close to an integer
    if (img - torch.round(img)).abs().max() < 1e-5:
        msg = (
            "Input image is all close to integers, "
            "are you sure you have normalized it before passing it to a NN?"
        )
        if on_error == "raise":
            raise ValueError(msg)
        elif on_error == "warn":
            warnings.warn(msg)
        else:
            return False
    return True


@torch.no_grad()
def basic_image_tensor_preprocess(
    img,
    mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
    std: tuple[float, float, float] = (0.5, 0.5, 0.5),
    shape: tuple[int, int] | None = None,
):
    """
    Check for resize, and divide by 255
    """
    import kornia

    assert torch.is_tensor(img)
    assert img.dim() >= 4, any_describe(img)
    original_shape = list(img.size())
    img = img.float()
    img = img.flatten(0, img.dim() - 4)
    assert img.dim() == 4

    input_size = img.size()[-2:]
    if global_once("groot.vla.common.utils.image_utils.basic_image_preprocess:input_size"):
        assert img.max() > 2, "img should be between [0, 255] before normalize"

    if shape and input_size != shape:
        if global_once("groot.vla.common.utils.image_utils.basic_image_preprocess:transform"):
            warnings.warn(
                f'{"Down" if shape < input_size else "Up"}sampling image'
                f" from original resolution {input_size}x{input_size}"
                f" to {shape}x{shape}"
            )
        img = kornia.geometry.transform.resize(img, shape).clamp(0.0, 255.0)

    B, C, H, W = img.size()
    assert C % 3 == 0, "channel must divide 3"
    img = img.view(B * C // 3, 3, H, W)
    img = torch_normalize(img / 255.0, mean=mean, std=std)
    original_shape[-2:] = H, W
    return img.view(original_shape)
