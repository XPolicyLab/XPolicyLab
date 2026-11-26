"""Bound outbound vision images without changing calibrated pixel coordinates."""

import io

from .state import RGBImage, image_data_url, image_media_type

MAX_IMAGE_BYTES = 300_000
MAX_REQUEST_BYTES = 2_000_000


def bounded_image_data_url(image, *, max_bytes=MAX_IMAGE_BYTES):
    """Compress only the API copy; never resize, crop, or mutate observations."""
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("A positive image byte limit is required")
    if isinstance(image, bytes):
        image_media_type(image)
        if len(image) <= max_bytes:
            return image_data_url(image)
    elif not isinstance(image, RGBImage):
        raise ValueError("Images must be RGBImage or encoded PNG/JPEG bytes")

    from PIL import Image, ImageFile

    if isinstance(image, RGBImage):
        pixels = Image.frombytes("RGB", (image.width, image.height), image.pixels)
    else:
        parser = ImageFile.Parser()
        parser.feed(image)
        with parser.close() as decoded:
            pixels = decoded.convert("RGB")
    try:
        # Keep full chroma resolution as well as width/height for small landmarks.
        # Do not sacrifice pixel geometry or use arbitrarily low JPEG quality.
        for quality in (90, 80, 70, 60, 50):
            output = io.BytesIO()
            pixels.save(
                output, format="JPEG", quality=quality, subsampling=0, optimize=False
            )
            encoded = output.getvalue()
            if len(encoded) <= max_bytes:
                return image_data_url(encoded)
    finally:
        pixels.close()
    raise ValueError(
        f"Image cannot fit within {max_bytes} bytes without resizing or excessive "
        "quality loss; select an explicit calibrated view or mapped crop"
    )
