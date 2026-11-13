"""Protocol server; Cartesian planning remains in the upstream EEF deploy loop."""
from XPolicyLab.policy.GPT_6_Astra_Direct_EEF.inspect.model import Model as InspectModel


class Model(InspectModel):
    """The agent runs through the environment-side EEF action interface."""
