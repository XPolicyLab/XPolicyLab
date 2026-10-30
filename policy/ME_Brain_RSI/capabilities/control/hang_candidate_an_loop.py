"""Keep loop statistics defined when the current background ring is outside RGB."""
import math
import cv2
import numpy as np
from coap.tasks.hang_candidate_s import finger_mask, project_points


def loop_fit(ctx, xy, radius, height, seed, camera='cam_head'):
    """Use HexaShell's non-table ring profile to find the exposed loop direction."""
    image = np.asarray(ctx.vision[camera]['color'], np.uint8)[..., :3]
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(float)
    fingers = finger_mask(image)
    angles = np.arange(120)*math.pi/60
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    reference = np.asarray(xy)+directions[:, None, :]*np.array([radius+.060, radius+.075])[None, :, None]

    def sample(points):
        pixels = np.rint(project_points(ctx, points, camera)).astype(int)
        valid = ((pixels[:, 0] >= 0)&(pixels[:, 0] < image.shape[1])&
                 (pixels[:, 1] >= 0)&(pixels[:, 1] < image.shape[0]))
        valid[valid] &= ~fingers[pixels[valid, 1], pixels[valid, 0]]
        values = np.zeros((len(points), 3))
        values[valid] = lab[pixels[valid, 1], pixels[valid, 0]]
        return values, valid

    values, valid = sample(np.column_stack((reference.reshape(-1, 2),
                                            np.full(reference.size//2, ctx.table_z))))
    reference_values = values[valid]
    reference_count = int(len(reference_values))
    if reference_count == 0:
        palette = ctx.memory.get('hang_s_association', {}).get('current_table_palette')
        if palette is None:
            palette = np.median(image.reshape(-1, 3), axis=0)
        reference_values = cv2.cvtColor(np.asarray(palette, np.uint8).reshape(1, 1, 3),
            cv2.COLOR_RGB2LAB).reshape(1, 3).astype(float)
    background = np.median(reference_values, axis=0)
    difference = reference_values-background
    chroma = max(8., 2.5*np.percentile(np.linalg.norm(difference[:, 1:], axis=1), 80))
    light = max(30., 2.5*np.percentile(np.abs(difference[:, 0]), 80))
    radii = np.linspace(radius+.004, radius+.035, 8)
    points = np.asarray(xy)+directions[:, None, :]*radii[None, :, None]
    world = np.column_stack((points.reshape(-1, 2),
                             np.full(points.size//2, ctx.table_z+height-.008)))
    values, valid = sample(world)
    difference = values-background
    outside = ((np.linalg.norm(difference[:, 1:], axis=1) > chroma)|
               (np.abs(difference[:, 0]) > light))&valid
    profile = outside.reshape(len(angles), -1).mean(axis=1)
    smooth = np.convolve(np.r_[profile[-2:], profile, profile[:2]], np.ones(5)/5, 'valid')
    prior = directions @ np.asarray(seed)[:2]
    score = smooth+.025*prior
    index = int(np.argmax(score))
    return np.r_[directions[index], 0.], dict(strength=float(smooth[index]),
        background=float(np.median(profile)), loop_yaw=float(angles[index]),
        observed_reference_samples=reference_count,
        background_source='observed_reference_ring' if reference_count else 'previously_observed_table_palette')
