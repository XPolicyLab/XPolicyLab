"""Choose a continuous valid IK branch from the solver's returned solutions."""

import numpy as np


def nearest_solution(positions, success, joint_names, active_names, current, lower, upper):
    values = np.asarray(positions, dtype=float)
    candidates = values.reshape(-1, values.shape[-1]).copy()
    columns = [joint_names.index(name) for name in active_names]
    active = candidates[:, columns]
    valid = np.asarray(success, dtype=bool).reshape(-1) & np.isfinite(active).all(
        axis=1
    )
    # All six controlled X5 joints are revolute. Choose equivalent angles
    # within the model limits before comparing candidate displacement.
    tau = 2 * np.pi
    safe = np.where(np.isfinite(active), active, 0)
    minimum_turn = np.ceil((np.asarray(lower) - safe) / tau)
    maximum_turn = np.floor((np.asarray(upper) - safe) / tau)
    valid &= (minimum_turn <= maximum_turn).all(axis=1)
    turns = np.clip(np.rint((np.asarray(current) - safe) / tau), minimum_turn, maximum_turn)
    active = safe + tau * turns
    candidates[:, columns] = active
    if not valid.any():
        raise ValueError("No valid IK solution")
    distances = np.max(np.abs(active - np.asarray(current)), axis=1)
    distances[~valid] = np.inf
    return candidates[int(np.argmin(distances))]


def x5_tcp_transform(gripper_bias):
    # X5 URDF joint7/8 originate along link6 X and move along +/-Y.
    # TCP X opens the fingers, TCP Z points towards their tips.
    result = np.eye(4)
    result[:3, :3] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    result[0, 3] = gripper_bias
    return result
