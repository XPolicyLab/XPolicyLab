"""Current RGB slot landmark reconstruction, without a stored toolbox point cloud."""
from physicalrsi_observed_geometry import GeometryUnavailable
from physicalrsi_observed_geometry.planar_landmark import observe_planar_landmark


def register(mask,guess_xy,yaw,*,observation=None,target_uv=None):
    if observation is None or target_uv is None:
        raise GeometryUnavailable('Current RGB calibration and visible slot landmark required')
    return observe_planar_landmark(observation,mask,target_uv)
