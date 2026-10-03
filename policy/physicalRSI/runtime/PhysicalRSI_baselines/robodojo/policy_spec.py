"""Public builder for a versioned code-policy skill assembly.

The compatibility builder keeps accepting the historical specification shape,
but newly created specifications use the neutral ``code-policy`` schema.  This
module is the preferred entry point for new task integrations.
"""

from copy import deepcopy

from .code_policy_spec import build as _legacy_build
from .code_policy_spec import identity as _legacy_identity


def build(**kwargs):
    """Build a policy assembly while preserving the existing validation path."""
    specification = _legacy_build(**kwargs)
    neutral = deepcopy(specification)
    neutral["schema"] = neutral["schema"].replace(
        "physicalrsi.robodojo.code-policy/", "physicalrsi.robodojo.code-policy/"
    )
    return neutral


def identity(specification):
    """Return the digest of a neutral or legacy policy assembly."""
    return _legacy_identity(specification)
