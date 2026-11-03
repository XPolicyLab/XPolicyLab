"""Apply generic font identity refinement; metric object size is not validated."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


def refine_color_identity(
    reader: Any,
    mask: Any,
    record: Mapping[str, Any],
    metric_validator: Callable[[Any, Any, Mapping[str, Any]], Mapping[str, Any]],
) -> dict[str, Any]:
    """Keep original identities unless generic glyph evidence supports a change."""
    original = {key: record[key] for key in ("digit", "yaw_deg", "confidence")}
    fitted = reader.projected_fit(mask, record["xy"], original)
    if fitted["digit"] == original["digit"] or not fitted["view_conditioned_fit"]["used"]:
        return dict(record)
    candidate = {**record, **fitted}
    metric = dict(metric_validator(reader.projected, mask, candidate))
    if not metric.get("valid", False):
        return dict(record)
    candidate["metric_support"] = metric
    candidate["color_identity_correction"] = {
        "original": original,
        "source": "generic_font_normalized_identity_evidence",
        "grasp_support": "recomputed_from_current_RGB_after_identity_correction",
    }
    return candidate
