"""The checkpoint's deployment-known physical timing, with no row-shift corrections."""
import numpy as np


def inference_time_grids(protocol):
    shapes = dict(action=32, video=3, pointmap=3, dino=3)
    grids = {name: np.zeros((n, 8), np.float32) for name, n in shapes.items()}
    valid = {name: np.zeros(n, bool) for name, n in shapes.items()}
    if not protocol or not protocol.get("verified", False):
        return grids, valid
    if not protocol.get("evidence") or not protocol.get("deployment_known", False):
        raise ValueError("Verified timing requires deployment-known provenance")
    dt = float(protocol["period_seconds"])
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Invalid checkpoint physical period")
    action_offsets = np.arange(32) + int(protocol.get("action_target_offset_frames", 0))
    grids["action"][:, :2] = action_offsets[:, None] * dt
    valid["action"][:] = True
    frames = np.arange(9) * 4
    for i, group in enumerate((frames[:1], frames[1:5], frames[5:9])):
        pairs = np.repeat(group, 2) * dt
        grids["video"][i, :len(pairs)] = pairs
    valid["video"][:] = True
    grids["pointmap"] = grids["video"].copy()
    valid["pointmap"][:] = True
    for i, frame in enumerate(frames[[0, 4, 8]]):
        grids["dino"][i, :2] = frame * dt
    valid["dino"][:] = True
    return grids, valid
