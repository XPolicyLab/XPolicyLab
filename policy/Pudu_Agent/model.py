"""No-op policy-server model for the environment-side Inspect EEF agent."""

from XPolicyLab.policy.Pudu_Agent.inspect.model import Model as InspectModel


class Model(InspectModel):
    """The LLM and Cartesian executor run in the RoboDojo client process."""
