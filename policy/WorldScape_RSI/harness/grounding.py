"""Visual grounding from camera calibration and robot proprioception only."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


def tool_forward(quat: Any) -> np.ndarray:
    """World direction of the X5 fingers: link6 local +X, quaternion wxyz."""
    q = np.asarray(quat, dtype=float).reshape(4)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(q).all() or norm < 1e-12:
        raise ValueError("tool quaternion must be finite and nonzero")
    w, x, y, z = q / norm
    return np.array([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)])


def tool_tip(position: Any, quat: Any, offset: float = 0.145) -> np.ndarray:
    """Approximate finger tip center; offset is robot geometry, not perception."""
    position = np.asarray(position, dtype=float).reshape(3)
    if not np.isfinite(position).all() or not np.isfinite(offset):
        raise ValueError("tool position and offset must be finite")
    return position + float(offset) * tool_forward(quat)


def project_points(
    points: Any, intrinsic_matrix: Any, camera_to_env: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Project env points through a USD/OpenGL camera (-Z forward, +Y up).

    Returns Nx2 pixels and an N-element validity mask. Points behind or on the
    camera plane have NaN pixels. Visibility in an image is checked by callers.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    k = np.asarray(intrinsic_matrix, dtype=float)
    transform = np.asarray(camera_to_env, dtype=float)
    if k.shape != (3, 3) or transform.shape != (4, 4):
        raise ValueError("camera calibration requires 3x3 intrinsics and a 4x4 pose")
    if not np.isfinite(k).all() or not np.isfinite(transform).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError("camera calibration must be finite with positive focal lengths")
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("camera pose must be a homogeneous transform")
    homogeneous = np.column_stack((points, np.ones(len(points))))
    camera_points = (np.linalg.inv(transform) @ homogeneous.T).T[:, :3]
    optical_points = camera_points * np.array([1.0, -1.0, -1.0])
    valid = np.isfinite(optical_points).all(axis=1) & (optical_points[:, 2] > 1e-5)
    pixels = np.full((len(points), 2), np.nan)
    projected = (k @ optical_points[valid].T).T
    pixels[valid] = projected[:, :2] / projected[:, 2:3]
    return pixels, valid


def annotate_head(
    color: Any,
    calibration: Mapping[str, Any],
    arm_states: Mapping[str, Any],
    *,
    table_z: float = 0.765,
    grid: bool = True,
    markers: bool = True,
    units: str = "m",
) -> np.ndarray:
    """Copy head RGB and draw a 10 cm table grid (5 cm minor lines) plus wrist/tip markers.

    ``units`` selects the grid labels: ``"m"`` writes ``X+0.2``, ``"cm"`` writes
    ``x+20`` so that a prompt which talks in centimetres and the image agree.

    ``grid`` and ``markers`` are separate because they answer separate
    questions - where a point on the table is, and where the hand is - and an
    ablation that turns both off at once cannot say which one carried the run.

    Calibration keys are ``intrinsic_matrix`` and ``extrinsic_matrix`` (camera
    to world in USD axes). Optional ``env_origin`` translates that pose into the
    frame of arm states. ``image_size`` is the calibration's [width, height],
    allowing a resized image. Arm states expose ``pos`` and ``quat`` attributes.
    """
    array = np.asarray(color)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError("head image must have at least three color channels")
    if np.issubdtype(array.dtype, np.floating) and array.size and np.nanmax(array) <= 1.0:
        array = array * 255
    image = Image.fromarray(np.clip(array[:, :, :3], 0, 255).astype(np.uint8))
    width, height = image.size
    k = np.asarray(calibration["intrinsic_matrix"], dtype=float).copy()
    transform = np.asarray(calibration["extrinsic_matrix"], dtype=float).copy()
    if transform.shape != (4, 4):
        raise ValueError("camera extrinsic_matrix must be 4x4")
    transform[:3, 3] -= np.asarray(calibration.get("env_origin", [0, 0, 0]), dtype=float).reshape(3)
    if "image_size" in calibration:
        source_width, source_height = calibration["image_size"]
        if source_width <= 0 or source_height <= 0:
            raise ValueError("calibration image size must be positive")
        k[0] *= width / source_width
        k[1] *= height / source_height
    # Validate calibration even if no grid or robot points fall inside the image.
    project_points([], k, transform)
    overlay = Image.new("RGBA", image.size)
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default(size=11)

    def project(point):
        pixels, valid = project_points([point], k, transform)
        return tuple(np.round(pixels[0], 6)) if valid[0] else None

    def visible(point):
        return point is not None and 0 <= point[0] < width and 0 <= point[1] < height

    def line(a, b, fill, line_width=1):
        start, end = project(a), project(b)
        if start is not None and end is not None:
            draw.line([start, end], fill=fill, width=line_width)

    def label(point, text, fill):
        if not visible(point):
            return
        bounds = draw.textbbox((0, 0), text, font=font, stroke_width=1)
        text_width, text_height = bounds[2] - bounds[0], bounds[3] - bounds[1]
        x = min(max(point[0] + 5, 1), max(1, width - text_width - 2))
        y = min(max(point[1] + 3, 1), max(1, height - text_height - bounds[1] - 2))
        draw.text((x, y), text, fill=fill, font=font, stroke_width=1, stroke_fill=(15, 20, 20, 200))

    # 10 cm lines with a fainter 5 cm line between them: with 10 cm cells the models' first grasp attempts
    # landed a median 6 cm from the object, i.e. they could not interpolate inside a cell.
    grid_color = (140, 225, 220, 90)
    minor_color = (140, 225, 220, 65)
    for x in (np.arange(-12, 13) / 20 if grid else []):
        line([x, -0.5, table_z], [x, 0.35, table_z], grid_color if round(x * 20) % 2 == 0 else minor_color)
    for y in (np.arange(-10, 8) / 20 if grid else []):
        line([-0.6, y, table_z], [0.6, y, table_z], grid_color if round(y * 20) % 2 == 0 else minor_color)
    if units not in ("m", "cm"):
        raise ValueError("units must be m or cm")

    def coordinate(prefix: str, value: float) -> str:
        return f"{prefix.lower()}{value * 100:+.0f}" if units == "cm" else f"{prefix}{value:+.1f}"

    for x in (np.arange(-6, 7) / 10 if grid else []):
        label(project([x, -0.3, table_z]), coordinate("X", x), (215, 245, 245, 235))
    for y in (np.arange(-4, 4) / 10 if grid else []):
        label(project([-0.55, y, table_z]), coordinate("Y", y), (215, 245, 245, 235))

    origin = [0, 0, table_z]
    for axis, end, fill in (() if not grid else (
        ("+X", [0.16, 0, table_z], (255, 205, 90, 230)),
        ("+Y", [0, 0.16, table_z], (125, 240, 225, 230)),
    )):
        line(origin, end, fill, 2)
        point, start = project(end), project(origin)
        if point is not None and start is not None:
            vector = np.asarray(point) - start
            length = np.linalg.norm(vector)
            if length > 1:
                direction = vector / length
                normal = np.array([-direction[1], direction[0]])
                for sign in (-1, 1):
                    draw.line([point, tuple(np.asarray(point) - direction * 7 + normal * sign * 3)], fill=fill, width=2)
        label(point, axis, fill)

    for prefix, fill in (() if not markers else
                         (("left", (255, 160, 105, 255)), ("right", (110, 195, 255, 255)))):
        state = arm_states.get(prefix)
        if state is None:
            continue
        position, quat = state.pos, state.quat
        tip = tool_tip(position, quat)
        line(position, tip, fill, 2)
        for name, location in (("wrist", position), ("tip", tip)):
            point = project(location)
            if not visible(point):
                continue
            x, y = point
            if name == "wrist":
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline=fill, width=2)
            else:
                draw.line((x - 5, y, x + 5, y), fill=fill, width=2)
                draw.line((x, y - 5, x, y + 5), fill=fill, width=2)
            label(point, f"{prefix[0].upper()} {name}", fill)
    return np.asarray(Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")).copy()


def autocontrast(color: Any, cutoff: float = 1.0) -> np.ndarray:
    """Percentile-stretch an RGB frame.

    RoboDojo's table scenes are bright and low-contrast, which costs detail
    exactly where grasping is decided. This is a display change only: it never
    moves a pixel, so a coordinate read off the image still means what it did.
    """
    array = np.asarray(color)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError("head image must have at least three color channels")
    if np.issubdtype(array.dtype, np.floating) and array.size and np.nanmax(array) <= 1.0:
        array = array * 255
    image = Image.fromarray(np.clip(array[:, :, :3], 0, 255).astype(np.uint8))
    try:
        stretched = ImageOps.autocontrast(image, cutoff=cutoff, preserve_tone=True)
    except TypeError:  # Pillow older than 9.5 has no preserve_tone
        stretched = ImageOps.autocontrast(image, cutoff=cutoff)
    return np.asarray(stretched).copy()


WRIST_AXIS_STYLE = (
    ("+x", (1.0, 0.0, 0.0), (255, 205, 90, 240)),
    ("+y", (0.0, 1.0, 0.0), (125, 240, 225, 240)),
    ("+z", (0.0, 0.0, 1.0), (245, 140, 245, 240)),
)


def annotate_wrist(
    color: Any,
    calibration: Mapping[str, Any],
    position: Any,
    quat: Any,
    *,
    offset: float = 0.145,
    axis_length_m: float = 0.05,
    label_text: str = "tip",
) -> np.ndarray:
    """Copy a wrist RGB frame and mark the fingertip centre plus world-axis arrows.

    The wrist camera rides on the hand, so the fingertip centre projects to
    (nearly) the same pixel in every frame: the cross says where the fingers
    will close, which a naked wrist image does not - an object seen between the
    two finger silhouettes can still be beside or below the fingertips. The
    arrows start at the cross and show the WORLD +x / +y / +z directions, each
    ``axis_length_m`` long, so an offset read off this image converts into a
    ``move`` command along a world axis. An axis pointing along the line of
    sight has no direction on the image and is left out.
    """
    array = np.asarray(color)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError("wrist image must have at least three color channels")
    if np.issubdtype(array.dtype, np.floating) and array.size and np.nanmax(array) <= 1.0:
        array = array * 255
    image = Image.fromarray(np.clip(array[:, :, :3], 0, 255).astype(np.uint8))
    width, height = image.size
    k = np.asarray(calibration["intrinsic_matrix"], dtype=float).copy()
    transform = np.asarray(calibration["extrinsic_matrix"], dtype=float).copy()
    if transform.shape != (4, 4):
        raise ValueError("camera extrinsic_matrix must be 4x4")
    transform[:3, 3] -= np.asarray(calibration.get("env_origin", [0, 0, 0]), dtype=float).reshape(3)
    if "image_size" in calibration:
        source_width, source_height = calibration["image_size"]
        if source_width <= 0 or source_height <= 0:
            raise ValueError("calibration image size must be positive")
        k[0] *= width / source_width
        k[1] *= height / source_height
    tip = tool_tip(position, quat, offset)
    points = [tip] + [tip + float(axis_length_m) * np.asarray(direction, dtype=float)
                      for _, direction, _ in WRIST_AXIS_STYLE]
    pixels, valid = project_points(points, k, transform)
    # The two jaw-contact directions are distinct from the world axes.
    w, x, y, z = np.asarray(quat, dtype=float) / np.linalg.norm(quat)
    jaw_axis = np.array([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)])
    jaw_pixels, jaw_valid = project_points([tip - axis_length_m * 0.5 * jaw_axis,
                                          tip + axis_length_m * 0.5 * jaw_axis], k, transform)
    overlay = Image.new("RGBA", image.size)
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default(size=12)

    def label(point, text, fill):
        bounds = draw.textbbox((0, 0), text, font=font, stroke_width=1)
        text_width, text_height = bounds[2] - bounds[0], bounds[3] - bounds[1]
        x = min(max(point[0] + 4, 1), max(1, width - text_width - 2))
        y = min(max(point[1] + 2, 1), max(1, height - text_height - bounds[1] - 2))
        draw.text((x, y), text, fill=fill, font=font, stroke_width=1, stroke_fill=(15, 20, 20, 210))

    if valid[0]:
        centre = tuple(float(v) for v in pixels[0])
        cx, cy = centre
        if -20 <= cx < width + 20 and -20 <= cy < height + 20:
            fill = (255, 255, 255, 245)
            draw.line((cx - 9, cy, cx + 9, cy), fill=fill, width=2)
            draw.line((cx, cy - 9, cx, cy + 9), fill=fill, width=2)
            draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), outline=fill, width=1)
            label(centre, label_text, fill)
        if all(jaw_valid):
            draw.line([tuple(point) for point in jaw_pixels], fill=(80, 240, 240, 220), width=2)
            label(tuple(jaw_pixels[1]), "jaw axis", (80, 240, 240, 240))
        for index, (name, _, fill) in enumerate(WRIST_AXIS_STYLE, start=1):
            if not valid[index]:
                continue
            end = np.asarray(pixels[index], dtype=float)
            vector = end - np.asarray(centre)
            length = float(np.linalg.norm(vector))
            if not np.isfinite(length) or length < 8:
                continue  # the axis points along the line of sight
            direction = vector / length
            normal = np.array([-direction[1], direction[0]])
            draw.line([centre, tuple(end)], fill=fill, width=2)
            for sign in (-1, 1):
                draw.line([tuple(end), tuple(end - direction * 7 + normal * sign * 3)], fill=fill, width=2)
            # 1 cm ticks along the arrow (image-space interpolation of the projected 5 cm), so an offset of
            # 2-3 cm is read off the ticks instead of guessed as a fraction of the arrow
            ticks = int(round(axis_length_m * 100))
            for step in range(1, ticks):
                at = np.asarray(centre) + vector * (step / ticks)
                half = 5 if step % 5 == 0 else 3
                draw.line([tuple(at - normal * half), tuple(at + normal * half)], fill=fill, width=2)
            label(tuple(end), f"{name} {axis_length_m * 100:g}cm", fill)
    return np.asarray(Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")).copy()
