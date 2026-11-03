"""Extract successful, named IK solutions without collapsing the seed dimension."""
import numpy as np


def active_solutions(position, success, names, active_names, cspace_names, active_indices):
    values = np.asarray(position, dtype=float)
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    elif values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2:
        raise ValueError('IK solutions must describe one problem, with shape (seeds,joints)')
    mask = np.asarray(success, dtype=bool).reshape(-1)
    if len(mask) != len(values):
        raise ValueError('IK success mask does not match the solution seed dimension')
    dof = values.shape[1]
    if names:
        if len(names) != dof or len(set(names)) != len(names):
            raise ValueError('IK joint names do not match the returned joint dimension')
        try:
            indices = [names.index(name) for name in active_names]
        except ValueError as exc:
            raise ValueError('IK joint names do not contain all active joints') from exc
    elif dof == len(cspace_names):
        indices = active_indices
    elif dof == len(active_names):
        indices = list(range(dof))
    else:
        raise ValueError('IK solution joint dimension does not match active joints or cspace')
    selected = values[mask][:, indices]
    return selected[np.isfinite(selected).all(axis=1)]
