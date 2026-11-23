# Copyright © 2026 Martingale. All Rights Reserved.
import hashlib
import numpy as np

CAMERAS=('cam_head','cam_left_wrist','cam_right_wrist')

def observation_bundle(robo, *, cameras=CAMERAS):
    """No actions. Reject views spanning an observation step.

    Returns state and views={camera:{rgb,calibration,rgb_sha256,step}}. Original
    pixels and calibration are preserved; images are not automatically displayed.
    The caller must explicitly display/review the views relevant to its decision.
    """
    if (not isinstance(cameras,(tuple,list)) or not cameras or len(cameras)>3
            or len(set(cameras))!=len(cameras) or any(c not in CAMERAS for c in cameras)):
        raise ValueError('Choose one to three distinct cameras')
    state=robo.state();step=state.step;views={}
    for camera in cameras:
        rgb=np.asarray(robo.image(camera,step=step))
        if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3:
            raise ValueError('Expected original uint8 RGB')
        views[camera]={'rgb':rgb,'calibration':robo.camera_meta(camera,step=step),
                       'rgb_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),'step':step}
    if robo.state().step!=step:raise ValueError('State changed during observation bundle')
    return {'state':state,'step':step,'views':views}
