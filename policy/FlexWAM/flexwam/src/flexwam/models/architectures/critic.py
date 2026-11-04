"""Action-camera critic head."""

from flexwam.models.architectures.view_scoring_head import SpatialQueryHead


class CameraCritic(SpatialQueryHead):
    """Independent critic parameters over first-frame visual tokens."""
