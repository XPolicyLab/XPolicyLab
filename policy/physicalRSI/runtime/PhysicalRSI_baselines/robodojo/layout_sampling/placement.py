"""Per-field independent sampling extracted from rdj_master."""

import numpy as np


def _unit(seed, task, layout_id, field, attempt=0):
    task_bytes = list(task.encode("utf-8"))
    field_bytes = list(field.encode("utf-8"))
    signed_id = int(layout_id)
    stream_id = 2 * signed_id if signed_id >= 0 else -2 * signed_id - 1
    entropy = [
        int(seed),
        stream_id,
        int(attempt),
        len(task_bytes),
        *task_bytes,
        len(field_bytes),
        *field_bytes,
    ]
    return float(np.random.default_rng(np.random.SeedSequence(entropy)).random())


def _input_record(path, source_root, role):
    return {"path": str(path.resolve()), "role": role}
