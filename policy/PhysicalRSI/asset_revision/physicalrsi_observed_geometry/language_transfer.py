"""Transfer heights from a currently segmented target and public robot reach."""
import cv2
import numpy as np
from contextlib import nullcontext
from types import SimpleNamespace
from PIL import Image
from . import GeometryUnavailable
from .pickup_target import select_target_region


def observed_basket_region(episode, box):
    """Use the current grounded RGB box, as the original inventory perceptor does."""
    image = np.asarray(episode.rgb)
    box = np.asarray(box, float)
    if image.ndim != 3 or image.shape[2] != 3 or box.shape != (4,) or not np.isfinite(box).all():
        raise GeometryUnavailable('Current RGB and finite basket box required')
    h,w = image.shape[:2]
    x0,y0,x1,y1 = np.clip(box, [0,0,0,0], [w,h,w,h])
    if x1<=x0 or y1<=y0:
        raise GeometryUnavailable('Visible basket box required')
    sam = episode.inventory.sam
    torch = sam._torch
    mixed = torch.autocast('cuda', dtype=torch.bfloat16) if str(sam.device).startswith('cuda') else nullcontext()
    with torch.inference_mode(), mixed:
        state = sam.processor.set_image(Image.fromarray(image))
        sam.processor.reset_all_prompts(state)
        result = sam.processor.add_geometric_prompt(
            [(x0+x1)/2/w, (y0+y1)/2/h, (x1-x0)/w, (y1-y0)/h], True, state)
        masks = result['masks'].cpu().numpy().astype(bool).reshape(-1,h,w)
        scores = result['scores'].float().cpu().numpy()
    return select_target_region([SimpleNamespace(masks=masks,scores=scores)], box, image.shape[:2])


def observed_basket_top(episode, box):
    image = np.asarray(episode.rgb)
    heights = np.asarray(episode.inventory_height, float)
    if heights.shape != image.shape[:2]:
        raise GeometryUnavailable('Current calibrated height image required for transfer')
    mask = observed_basket_region(episode, box)
    mask = cv2.erode(mask.astype('uint8'), np.ones((3,3),'uint8')).astype(bool)
    values = heights[mask]
    if values.size < 8 or not np.isfinite(values).all():
        raise GeometryUnavailable('Finite observed basket surface required')
    # Estimate the visible upper surface, including any observed contents.
    # This is a measured height, not a supplied basket dimension or CAD rim.
    return float(np.quantile(values, .95))


def observed_transfer_heights(contact_z, target_top, tilt_degrees):
    if not np.isfinite([contact_z,target_top,tilt_degrees]).all() or not 0 <= tilt_degrees < 90:
        raise GeometryUnavailable('Finite transfer measurements and downward tool orientation required')
    # Public X5 fingertip reach; the margins retain generic approach/release
    # clearances rather than any task's absolute world Z.
    lower_z = target_top + .15757000232458113 * np.cos(np.deg2rad(tilt_degrees)) + .015
    carry_z = max(contact_z + .10, lower_z + .04)
    return float(carry_z), float(lower_z)
