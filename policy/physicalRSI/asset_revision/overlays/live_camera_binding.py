"""Validate current sensor calibration without replacing it with recorded poses."""
from physicalrsi_observed_geometry import Camera


def bind_camera_calibration(observation):
    for name in ('cam_head', 'cam_left_wrist', 'cam_right_wrist'):
        Camera.from_observation(observation, name)
    return observation
