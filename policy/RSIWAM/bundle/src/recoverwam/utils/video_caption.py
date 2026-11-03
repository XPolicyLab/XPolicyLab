"""Render a full instruction below video frames without covering the prediction."""

import math

from PIL import Image, ImageDraw, ImageFont


def make_caption_panel(width, prompt, font_size=14, font_path=None):
    if width < 64 or font_size < 8:
        raise ValueError("Caption rendering requires width >= 64 and font size >= 8")
    try:
        font = ImageFont.truetype(str(font_path or "DejaVuSans.ttf"), font_size)
    except OSError:
        if font_path is not None:
            raise
        font = ImageFont.load_default(size=font_size)
    padding = 12
    available = width - padding * 2
    lines, current = [], ""
    for word in prompt.split():
        proposed = f"{current} {word}" if current else word
        if font.getlength(proposed) <= available:
            current = proposed
            continue
        if current:
            lines.append(current)
        # Preserve long tokens rather than letting them run off the right edge.
        current = ""
        for character in word:
            if current and font.getlength(current + character) > available:
                lines.append(current)
                current = ""
            current += character
    if current:
        lines.append(current)
    ascent, descent = font.getmetrics()
    spacing = ascent + descent + 3
    height = math.ceil((padding * 2 + (len(lines) + 1) * spacing) / 16) * 16
    panel = Image.new("RGB", (width, height), (20, 20, 20))
    draw = ImageDraw.Draw(panel)
    draw.text((padding, padding), "Instruction", font=font, fill=(155, 199, 255), anchor="lt")
    for index, line in enumerate(lines, 1):
        draw.text((padding, padding + index * spacing), line, font=font,
                  fill=(245, 245, 245), anchor="lt")
    return panel


def append_caption(frames, panel):
    for frame in frames:
        if frame.width != panel.width:
            raise ValueError("Video and caption widths differ")
        output = Image.new("RGB", (frame.width, frame.height + panel.height))
        output.paste(frame, (0, 0))
        output.paste(panel, (0, frame.height))
        yield output
