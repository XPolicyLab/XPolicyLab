"""Camera labels for metrics; these do not change model inputs or camera order."""
from typing import Sequence


LEGACY_CAMERA_NAMES = ("global", "left_wrist", "right_wrist")


def configure_camera_metrics(model, data_config) -> None:
    """Use the same ordered image specs as independent multiview data loading."""
    if data_config.get("concat_multi_camera") != "independent":
        return
    images = data_config["shape_meta"]["images"]
    names = tuple(str(image["key"]) for image in images)
    if len(set(names)) != len(names):
        raise ValueError(f"Camera metric names must be unique, got {names}.")
    model.camera_metric_names = names


def resolve_camera_metric_names(num_views: int, names: Sequence[str] | None = None) -> tuple[str, ...]:
    if names is None:
        # Preserve names for legacy callers without dataset metadata.
        names = LEGACY_CAMERA_NAMES
    elif len(names) != num_views:
        raise ValueError(f"Camera metric name count {len(names)} does not match {num_views} views.")
    return tuple(str(names[i]) if i < len(names) else f"view{i}" for i in range(num_views))
