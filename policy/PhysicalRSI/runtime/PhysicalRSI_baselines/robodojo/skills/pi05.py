"""Index-correct wrapper for the separately installed trained pi05 adapter."""

from .pi05_backend import Model as UpstreamModel

from .metadata import ExternalPolicyMetadata
from .order import EnvironmentOrder


class Model(ExternalPolicyMetadata, EnvironmentOrder, UpstreamModel):
    policy_name = "pi05"
