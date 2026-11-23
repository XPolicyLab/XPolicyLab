# Copyright © 2026 Martingale. All Rights Reserved.
import cv2
import numpy as np
from martingale.hosts import robodojo_tools as tools, robodojo_sight as sight
from martingale.inference.extensions.mugs.hosts.hanging_peg import fit_peg, peg_mask

def peg(primitives, selections, peg_id='target', reset=False):
    """Fit one static peg; each selection contains current-image root/tip pixels."""
    if not selections:
        return dict(valid=False, reason='Select the root and tip of one peg in current images')
    obs = tools._obs(primitives)
    state = tools._shown(obs)
    cache = getattr(primitives, '_hanging_pegs', {})
    primitives._hanging_pegs = cache
    if reset:
        cache.pop(peg_id, None)
    history = list(cache.get(peg_id, []))
    try:
        cameras = [selection['camera'] for selection in selections]
        if len(set(cameras)) != len(cameras):
            raise ValueError('Select each current camera at most once')
        for selection in selections:
            camera = selection['camera']
            if camera not in ('cam_head', 'cam_left_wrist', 'cam_right_wrist'):
                raise ValueError('Unknown camera')
            view = tools._view(camera, state)
            rgb = tools._camera(obs, camera)['color']
            seed = np.asarray([selection['root_rc'], selection['tip_rc']], float)
            if seed.shape != (2, 2) or not np.isfinite(seed).all() or (seed < 0).any() or (seed >= np.array(rgb.shape[:2])).any() or (np.linalg.norm(seed[1] - seed[0]) < 12):
                raise ValueError('Select in-image root/tip row,column pairs at least 12 pixels apart')
            entry = (view, peg_mask(rgb, seed), seed, sight.robot_pixels(view, state, own=tools._WRISTS.get(camera)))
            history = [x for x in history if np.linalg.norm(x[0].pose - view.pose) > 1e-05]
            history.append(entry)
        history = history[-3:]
        cache[peg_id] = history
        if len(history) < 2:
            return dict(valid=False, stored_views=len(history), reason='Stored one view. Move an EMPTY wrist for parallax and select the same peg again. Use reset=True for a different peg or if the rack moved.')
        result = fit_peg(*map(list, zip(*history)))
    except (ValueError, TypeError, KeyError, cv2.error, np.linalg.LinAlgError) as exc:
        return dict(valid=False, reason=str(exc))
    result.update(step=primitives.env.get_status().get('step'), stored_views=len(history), peg_id=peg_id)
    if not result['valid']:
        result['next_step'] = 'Do not use this rejected fit for insertion. Obtain clear views of the same peg; reset=True replaces old overlapping or occluded observations.'
    return result
peg._readonly = True
