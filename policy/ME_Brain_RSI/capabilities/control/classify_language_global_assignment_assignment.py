"""Assign one frame using only the retained eligible source-material soft cost."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def assignment_indices(costs):
    """Maximize eligible cardinality, then minimize the unchanged total edge cost."""
    costs = np.asarray(costs, float)
    count, tracks = costs.shape
    if not count:
        return []
    if not tracks:
        return [None] * count
    eligible = costs[np.isfinite(costs)]
    scale = max(1., float(np.max(np.abs(eligible)))) if eligible.size else 1.
    # This algebraic bound encodes cardinality priority, not a matching threshold.
    # One more real edge saves more than the possible change of all real costs.
    birth_cost = (2 * min(count, tracks) + 1) * scale
    augmented = np.column_stack((costs, np.full((count, count), birth_cost)))
    rows, columns = linear_sum_assignment(augmented)
    result = [None] * count
    for row, column in zip(rows, columns):
        if column < tracks:
            result[int(row)] = int(column)
    return result


def assign_sources(ctx, tracks, objects, overlap, pair_bonus):
    """Score every old-track edge before updating any current-frame semantics."""
    costs = np.full((len(objects), len(tracks)), np.inf)
    for row, obj in enumerate(objects):
        point = np.asarray(obj['position'], float)
        for column, track in enumerate(tracks):
            distance = float(np.linalg.norm(point[:2] - np.asarray(track['object']['position'])[:2]))
            common = overlap(obj, track['object'])
            appearance = float(np.linalg.norm(np.asarray(obj['mean_rgb'], float) -
                np.asarray(track['object']['mean_rgb'], float)) / 255.)
            if distance < .060 and common > .25 and appearance < .50:
                costs[row, column] = distance + .020 * appearance - .010 * common - pair_bonus(ctx, track, obj)
    return [None if column is None else tracks[column] for column in assignment_indices(costs)]
