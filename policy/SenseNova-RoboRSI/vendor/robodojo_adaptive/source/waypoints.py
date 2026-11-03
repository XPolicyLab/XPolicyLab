"""Select consecutive policy waypoints and clip only the two gripper channels."""
import numpy as np

def clip_grippers(actions):
    clipped = actions.copy()
    clipped[:, [6, 13]] = np.clip(clipped[:, [6, 13]], 0., 1.)
    return clipped
