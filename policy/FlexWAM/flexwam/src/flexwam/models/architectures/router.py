"""Action-camera router head."""

from flexwam.models.architectures.view_scoring_head import SpatialQueryHead


class CameraRouter(SpatialQueryHead):
    """Independent router parameters over first-frame visual tokens."""
