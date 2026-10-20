"""RoboTwin observation encoding and native joint command decoding."""

from .normalization import Joint14Normalizer
from .eef import JointEefNormalizer


def __getattr__(name):
    if name == "prepare_observation":
        from .observation import prepare_observation
        return prepare_observation
    raise AttributeError(name)

__all__ = ["Joint14Normalizer", "JointEefNormalizer", "prepare_observation"]
