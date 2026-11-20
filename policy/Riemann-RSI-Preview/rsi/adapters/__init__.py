"""Register protocol plugins; policy implementations remain under policies/."""
from pathlib import Path


def register_policy():
    """Expose our named plugin through the SDK's normal policy search path."""
    import XPolicyLab.policy
    directory = str(Path(__file__).resolve().parent)
    if directory not in XPolicyLab.policy.__path__:
        XPolicyLab.policy.__path__.insert(0, directory)
