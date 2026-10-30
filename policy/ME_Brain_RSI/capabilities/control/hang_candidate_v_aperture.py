"""Refresh a selected native aperture from the real pregrasp wrist observation."""
import math

import numpy as np

from coap.tasks.hang_candidate_h_cad import _features
from coap.tasks.hang_candidate_s import loop_fit, refresh_rim as previous_refresh, yaw_matrix
from coap.tasks.mug_scene import public_models


def refresh_aperture(ctx, source, arm):
    """Keep the pinch geometry while updating only the handle aperture geometry."""
    camera = f'cam_{arm}_wrist'
    index = int(source['asset_index'])
    if (index >= 10 or camera not in ctx.vision or
            not source['source'].get('supplemental_standard_CAD')):
        return previous_refresh(ctx, source, arm)
    models, _ = _features()
    native = models[str(index)]
    public = public_models()['mugs'][str(index)]
    body_local = np.asarray(public['body_center_local'], float)
    handle_local = np.asarray(public['handle_frames'][0][:3], float)
    radial_local = handle_local - body_local
    radial, review = loop_fit(ctx, source['body'][:2], source['radius'],
                              source['height'], source['radial'], camera)
    yaw = math.atan2(radial[1], radial[0]) - math.atan2(radial_local[1], radial_local[0])
    rotation = yaw_matrix(yaw)
    root = np.asarray(source['body'], float) - rotation @ body_local
    root[2] = ctx.table_z - float(public['floor_local_z'])
    hole = root + rotation @ np.asarray(native['thread_local'], float)
    updated = dict(source, hole=hole, hole_yaw=float(yaw), hole_radial=rotation[:, 0].copy(),
        wrist_aperture=dict(camera=camera, observation_step=int(ctx.steps),
            asset_index=index, loop=review, previous_hole=np.asarray(source['hole']).tolist(),
            hole_delta_m=float(np.linalg.norm(hole-source['hole'])),
            previous_yaw=float(source['yaw']), hole_yaw=float(yaw),
            body_preserved=True, grasp_preserved=True, pinch_radial_preserved=True,
            source='current_pregrasp_wrist_RGB_and_selected_public_native_model'))
    return updated
