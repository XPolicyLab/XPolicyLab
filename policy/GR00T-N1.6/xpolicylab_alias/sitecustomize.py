"""Import alias for the GR00T-N1.6 policy directory.

XPolicyLab imports policies as Python modules.  The on-disk directory name
``GR00T-N1.6`` is not importable, so eval.sh exposes it as ``GR00T_N1_6``.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path


def _register_alias() -> None:
    policy_dir = os.environ.get("GR00T_XPOLICYLAB_POLICY_DIR")
    if not policy_dir:
        return
    policy_path = Path(policy_dir)
    if not policy_path.exists():
        return
    alias = os.environ.get("GR00T_XPOLICYLAB_ALIAS", "GR00T_N1_6")
    package_name = f"XPolicyLab.policy.{alias}"

    try:
        import XPolicyLab.policy as policy_pkg
    except Exception:
        return

    package = types.ModuleType(package_name)
    package.__path__ = [str(policy_path)]
    package.__package__ = package_name
    sys.modules[package_name] = package
    setattr(policy_pkg, alias, package)


_register_alias()
