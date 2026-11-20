import cv2
import numpy as np


def normalize_glyph(mask, size=64):
    mask = np.asarray(mask, dtype=np.uint8)
    rows, columns = np.where(mask > 0)
    if len(rows) < 20:
        raise RuntimeError('The public crop has too little glyph foreground')
    cropped = mask[rows.min():rows.max() + 1, columns.min():columns.max() + 1]
    height, width = cropped.shape
    scale = min((size - 8) / max(width, 1), (size - 8) / max(height, 1))
    resized = cv2.resize(cropped, (max(1, int(round(width * scale))),
        max(1, int(round(height * scale)))), interpolation=cv2.INTER_AREA)
    resized = (resized >= 96).astype(np.uint8) * 255
    canvas = np.zeros((size, size), dtype=np.uint8)
    row_offset = (size - resized.shape[0]) // 2
    column_offset = (size - resized.shape[1]) // 2
    canvas[row_offset:row_offset + resized.shape[0],
           column_offset:column_offset + resized.shape[1]] = resized
    return canvas
