"""Decoded RGB observation contracts for the historical inference adapter."""
import cv2
import numpy as np

CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")


def build_tshape(head, left, right):
    images = [np.asarray(x) for x in (head, left, right)]
    for image in images:
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8 or not min(image.shape[:2]):
            raise ValueError("Expected decoded uint8 HWC RGB frames")
    top = cv2.resize(images[0], (320, 256), interpolation=cv2.INTER_AREA)
    wrists = [cv2.resize(x, (160, 128), interpolation=cv2.INTER_AREA) for x in images[1:]]
    return np.concatenate((top, np.concatenate(wrists, axis=1)), axis=0)


def instruction_text(obs):
    value = obs.get("task_instruction")
    if value is None:
        value = obs.get("instruction", obs.get("instructions"))
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("A nonempty instruction is required")
    return value.strip()
