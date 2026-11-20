import numpy as np


def run(digits, targets):
    if not isinstance(digits, (list, tuple)) or not isinstance(targets, (list, tuple)) or not 1 <= len(digits) == len(targets) <= 10:
        raise ValueError('Assign only equally sized current observed digit and target families')
    labels = []
    for record in digits:
        if not isinstance(record, dict) or set(record) != {'digit', 'surface_xyz'}:
            raise ValueError('Digit assignment accepts only independent labels and current measured surfaces')
        digit = record['digit']
        point = np.asarray(record['surface_xyz'], dtype=float)
        if type(digit) is not int or not 0 <= digit <= 9 or point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError('Every digit needs its own independently recognized identity and finite public surface')
        labels.append(digit)
    if len(set(labels)) != len(labels):
        raise ValueError('Do not invent a missing identity to resolve duplicate recognitions')
    surfaces = np.asarray(targets, dtype=float)
    if surfaces.shape != (len(targets), 3) or not np.isfinite(surfaces).all():
        raise ValueError('Target surfaces must come from the current publicly observed family')
    ordering = sorted(range(len(targets)), key=lambda index: float(surfaces[index, 0]))
    if len(ordering) > 1 and np.any(np.diff(surfaces[ordering, 0]) <= 1e-8):
        raise ValueError('A left-to-right target order needs distinct measured horizontal positions')
    ranked = sorted(digits, key=lambda record: -record['digit'])
    return [{'digit': record['digit'], 'source_surface_xyz': list(record['surface_xyz']),
             'target_surface_xyz': surfaces[target].tolist(), 'target_index': target}
            for record, target in zip(ranked, ordering)]
