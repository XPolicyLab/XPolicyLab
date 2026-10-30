"""Propose visible single-wall pinches from exact public cup triangle slices."""
import json
import math
from pathlib import Path
from functools import lru_cache
import numpy as np


@lru_cache(maxsize=1)
def wall_slices():
    return json.loads((Path(__file__).parents[1]/'data'/'mug_wall_slices.json').read_text())


def near_wall_candidates(ctx, mug, center, yaw, arm):
    """Keep a visible same-material cup wall between one inner and one outer jaw."""
    from .precision import _rotate, _yaw_quaternion
    from .mug_scene import _projector
    data = wall_slices()
    if mug.get('asset_index') != data['asset_index']:
        return []
    if mug.get('camera','cam_head') != 'cam_head':
        ctx.memory['mug_near_wall_measurement'] = {'asset_index':8,'accepted_candidates':0,
            'reason':'Current head source bounding box is required','private_task_state_used':False}
        return []
    image = np.asarray(ctx.vision['cam_head']['color'],np.uint8)[...,:3]
    source_color = np.asarray(mug.get('mean_rgb',[]),float)
    if source_color.shape != (3,):
        return []
    body = np.asarray(center)[:2]+_rotate(_yaw_quaternion(yaw),
                                        np.r_[data['metadata_body_center'],0.])[:2]
    base = ctx.kinematics.base_transforms[arm][:2,3]
    normal = base-body
    if np.linalg.norm(normal)<.08:
        return []
    normal /= np.linalg.norm(normal)
    heading = math.atan2(normal[1],normal[0])
    local_heading = heading-yaw
    project = _projector(ctx,image.shape)
    results, reviews = [],[]
    for slice_data in data['wall_slices']:
        if not .013<=slice_data['local_z']<=.019:
            continue
        selected = sorted(slice_data['candidates'],key=lambda candidate:abs(math.atan2(
            math.sin(candidate['angle']-local_heading),math.cos(candidate['angle']-local_heading))))[:2]
        for candidate in selected:
            normal_error = abs(math.atan2(math.sin(candidate['angle']-local_heading),
                                          math.cos(candidate['angle']-local_heading)))
            if normal_error>math.radians(15):
                continue
            local = np.asarray(candidate['local_center'],float)
            world = np.asarray(center)+_rotate(_yaw_quaternion(yaw),local)
            pixel = project(world[None,:])[0]
            u,v = np.round(pixel).astype(int)
            x,y,width,height = mug['bbox']
            bounded = x-5<=u<=x+width+5 and y-5<=v<=y+height+5
            visible = 3<=u<image.shape[1]-3 and 3<=v<image.shape[0]-3
            record = {'camera':'cam_head','pixel':pixel.tolist(),'local_point_m':local.tolist(),
                'wall_thickness_m':candidate['wall_thickness'],'inside_source_bbox':bool(bounded),
                'near_arm_normal_error_rad':normal_error,
                'accepted':False,'private_task_state_used':False,'observation_step':int(ctx.steps)}
            reviews.append(record)
            if not bounded or not visible or ctx.perception.on_robot(pixel,ctx.state,image.shape):
                continue
            patch = image[v-2:v+3,u-2:u+3].astype(float)
            color_error = np.linalg.norm(patch-source_color,axis=2)/math.sqrt(3)
            same_material = float(np.mean(color_error<45.))
            record['same_material_fraction'] = same_material
            if same_material<.72:
                continue
            world_normal = _rotate(_yaw_quaternion(yaw),[math.cos(candidate['angle']),
                                                         math.sin(candidate['angle']),0.])
            closing_heading = math.atan2(world_normal[1],world_normal[0])-math.pi/2
            record.update(accepted=True,source='rgb_same_material_and_public_triangle_wall_slice',
                source_asset=data['source_asset'],source_usdz_sha256=data['source_sha256'],
                upright_table_prior=True,height_independently_measured=False,
                metric_position_error_bound_m=None,retention_verified=False,
                limits='A projected same-material wall with a known upright CAD slice is a grasp proposal, not a metric pose or holding witness.')
            results.append({'grasp_local':local,'grasp_heading':closing_heading,'evidence':record})
    ctx.memory['mug_near_wall_measurement'] = {'asset_index':8,'reviews':reviews,
                                             'accepted_candidates':len(results),
                                             'private_task_state_used':False}
    return results
