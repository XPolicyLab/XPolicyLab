"""Image geometry for standalone human/robot video conditioning (sizes are W,H)."""

from pathlib import Path
import re

from PIL import Image, ImageOps


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def load_views(source):
    """Read one image or a directory containing view_0 and optional view_1/view_2."""
    source = Path(source)
    paths = [None, None, None]
    if source.is_dir():
        for path in sorted(source.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            match = re.fullmatch(r"view_([0-9]+)", path.stem)
            if match is None or int(match[1]) > 2:
                raise ValueError(f"Expected only view_0, view_1, view_2 images: {path}")
            index = int(match[1])
            if paths[index] is not None:
                raise ValueError(f"Duplicate view_{index}: {paths[index]} and {path}")
            paths[index] = path
        if paths[0] is None:
            raise ValueError(f"Missing view_0 image in {source}")
    else:
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Expected an image or view directory: {source}")
        paths[0] = source
    views = []
    for path in paths:
        if path is None:
            views.append(None)
        else:
            with Image.open(path) as image:
                views.append(ImageOps.exif_transpose(image).convert("RGB"))
    return views, paths


def crop_box(size, aspect):
    """Largest centered crop with the requested width / height ratio."""
    width, height = size
    crop_width, crop_height = min(width, height * aspect), min(height, width / aspect)
    left, top = (width - crop_width) / 2, (height - crop_height) / 2
    return (left, top, left + crop_width, top + crop_height)


def prepare_image(views, input_type):
    """Crop the main image, resize whole wrist views, and pad missing views black."""
    if not 1 <= len(views) <= 3 or views[0] is None:
        raise ValueError("Require view_0 and at most three views")
    views = list(views) + [None] * (3 - len(views))
    boxes = [None, None, None]
    if input_type == "human":
        if any(view is not None for view in views[1:]):
            raise ValueError("Human ego-view input must contain only one image")
        boxes[0] = crop_box(views[0].size, 16 / 9)
        result = views[0].crop(boxes[0]).resize((416, 240), Image.Resampling.BICUBIC)
    elif input_type == "robot":
        canvas = Image.new("RGB", (440, 240), "black")
        for index, (view, size, position) in enumerate(zip(
            views, [(320, 240), (120, 120), (120, 120)], [(0, 0), (320, 0), (320, 120)]
        )):
            if view is not None:
                boxes[index] = (crop_box(view.size, 4 / 3) if index == 0 else
                                (0, 0, view.width, view.height))
                canvas.paste(view.crop(boxes[index]).resize(size, Image.Resampling.BICUBIC), position)
        result = canvas.resize((416, 240), Image.Resampling.BICUBIC)
    else:
        raise ValueError(f"Unknown input type: {input_type}")
    return result, {
        "input_type": input_type,
        "original_sizes_wh": [list(v.size) if v is not None else None for v in views],
        "crop_boxes_xyxy": boxes,
        "missing_views": [i for i, v in enumerate(views) if v is None],
        "output_size_wh": list(result.size),
    }
