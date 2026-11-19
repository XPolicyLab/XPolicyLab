import numpy as np
from PIL import Image


def convert_to_uint8(img: np.ndarray) -> np.ndarray:
    """Converts an image to uint8 if it is a float image.

    This is important for reducing the size of the image when sending it over the network.
    """
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    return img


def resize_with_pad(images: np.ndarray, intrinsics: np.ndarray, height: int, width: int, method=Image.BILINEAR) -> np.ndarray:
    """Replicates tf.image.resize_with_pad for multiple images using PIL. Resizes a batch of images to a target height.
    Besides, as images are resized, the corresponding camera intrinsics are also adjusted accordingly.

    Args:
        images: A batch of images in [..., height, width, channel] format.
        intrinsics: A batch of camera intrinsics in [..., 3, 3] format.
        height: The target height of the image.
        width: The target width of the image.
        method: The interpolation method to use. Default is bilinear.

    Returns:
        The resized images in [..., height, width, channel].
        The adjusted camera intrinsics in [..., 3, 3].
    """
    # If the images are already the correct size, return them as is.
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape
    original_intrinsics_shape = intrinsics.shape

    images = images.reshape(-1, *original_shape[-3:])
    intrinsics = intrinsics.reshape(-1, *original_intrinsics_shape[-2:])

    resized_results = [
        _resize_with_pad_pil(Image.fromarray(im), intri, height, width, method=method)
        for im, intri in zip(images, intrinsics)
    ]

    resized_images, img_padding_mask, resized_intrinsics = zip(*resized_results)
    resized_images = np.stack(resized_images)
    img_padding_mask = np.stack(img_padding_mask)
    resized_intrinsics = np.stack(resized_intrinsics)

    return (
        resized_images.reshape(*original_shape[:-3], *resized_images.shape[-3:]), 
        img_padding_mask.reshape(*original_intrinsics_shape[:-2], *img_padding_mask.shape[-2:]),
        resized_intrinsics.reshape(*original_intrinsics_shape[:-2], *resized_intrinsics.shape[-2:])
    )


def _resize_with_pad_pil(image: Image.Image, intri: np.ndarray, height: int, width: int, method: int):
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return image  # No need to resize if the image is already the correct size.

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)

    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    assert zero_image.size == (width, height)

    img_padding_mask = np.zeros((height, width), dtype=bool)
    img_padding_mask[pad_height:pad_height+resized_height, pad_width:pad_width+resized_width] = True

    intri[:2, :3] /= ratio
    intri[0, 2] += pad_width
    intri[1, 2] += pad_height

    return zero_image, img_padding_mask, intri
