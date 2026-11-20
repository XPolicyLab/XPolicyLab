import cv2
import numpy as np


def material_mask(image, extended=False):
    image = np.asarray(image)
    if (image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8
            or not 1 <= image.shape[0] * image.shape[1] <= 307200 or type(extended) is not bool):
        raise ValueError('Segment only actual public RGB with an explicit appearance preset')
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hue, saturation, value = np.moveaxis(hsv, -1, 0)
    blue = (hue >= 60) & (hue <= 140) & (saturation >= 35) & (value >= 30)
    purple = (hue >= 135) & (hue <= 179) & (saturation >= 40) & (value >= 130)
    red = ((hue <= 11) & (saturation >= 150) & (value >= 220)) | (
        (hue <= 18) & (saturation >= 50) & (value >= 248))
    salmon = (hue <= 19) & (saturation >= 30) & (saturation < 150) & (value >= 245)
    orange = ((hue >= 23) & (hue <= 36) & (saturation >= 30) & (value >= 245)) | (
        (hue >= 23) & (hue <= 36) & (saturation >= 50) & (value >= 235)) | (
        (hue >= 20) & (hue <= 22) & (saturation >= 160) & (value >= 230))
    mask = blue | red | orange
    if extended:
        mask |= purple | salmon
    return mask.astype(np.uint8)


def run(image, extended=False):
    primary = material_mask(image, extended).astype(bool)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hue, saturation, value = np.moveaxis(hsv, -1, 0)
    rgb = image.astype(float)
    proposals = [primary]
    for delta in (-2., 0., 3., 6., 9.):
        proposals.append(((rgb[:, :, 0] - rgb[:, :, 1]) > delta)
                         & (rgb[:, :, 0] > rgb[:, :, 2] * .90) & (value > 45) & (value < 245))
    proposals.append((hue >= 125) & (hue <= 179) & (saturation >= 18) & (value > 40))
    for floor in (140, 180, 220):
        proposals.append((hue >= 125) & (saturation >= 60) & (value >= floor))
        proposals.append(((hue <= 10) | (hue >= 170)) & (saturation >= 60) & (value >= floor))
    unique = []
    for proposal in proposals:
        if proposal.any() and not any(np.array_equal(proposal, prior) for prior in unique):
            proposal.flags.writeable = False
            unique.append(proposal)
    return tuple(unique)
