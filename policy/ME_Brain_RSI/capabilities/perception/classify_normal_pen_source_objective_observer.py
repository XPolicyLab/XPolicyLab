"""Refine pen grasp points after the retained complete observer."""
import numpy as np

from coap.tasks.classify_r07_n2_material_centroid import observe as parent_observe
from coap.tasks.classify_r29_pen_stability import refine_pen_observation


def observe(ctx):
    """Preserve source anchors and mapping while recording the pen-only overlay."""
    original = parent_observe(ctx)
    scene = dict(original)
    objects = []
    updates = []
    for item in original['objects']:
        refined, evidence = item, None
        if item['category'] == 'pen':
            camera = item.get('camera', 'cam_head')
            shape = np.asarray(ctx.vision[camera]['color']).shape
            refined, evidence = refine_pen_observation(ctx, item, camera=camera, image_shape=shape)
        if evidence is not None:
            refined['pen_stability'] = evidence
            updates.append(evidence)
        objects.append(refined)
    scene['objects'] = objects
    ctx.memory.setdefault('classify_pen_stability', []).append(dict(
        step=ctx.steps, observed_pens=len(updates), updates=updates))
    return scene
