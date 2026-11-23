import numpy as np
CAMERAS = ('cam_head', 'cam_left_wrist', 'cam_right_wrist')

def locate_from_available_camera(locate, query):
    attempts = []
    for camera in CAMERAS:
        result = locate(query, camera=camera)
        attempts.append(result)
        center = result.get('center_xyz')
        if center is not None and np.asarray(center).shape == (3,) and np.isfinite(center).all():
            return dict(result, localization_attempts=attempts)
    return dict(found=False, text_prompt=query, localization_attempts=attempts, geometry_error='No camera supplied a finite coarse position.')
