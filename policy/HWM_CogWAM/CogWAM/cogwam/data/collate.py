"""Batch collation for the CogWAM data path.

Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
Modified by [Fangjing Wang / SUST University] in [2025].
Modification: [return raw data and support multi-dataset mixture].
Modified by [Jinhui Ye / HKUST University] in [2025].
Modification: [support top-down processing, support param from config].

The framework consumes a list of per-sample dicts rather than stacked tensors:
samples carry ragged fields (PIL-convertible RGB arrays of differing view
counts, free-form text, per-sample semantic labels) that ``CogWAM.forward``
batches itself.  Collating here would force those fields into tensors the
model would immediately have to undo.
"""

from __future__ import annotations


def collate_fn(batch):
    return batch


__all__ = ["collate_fn"]
