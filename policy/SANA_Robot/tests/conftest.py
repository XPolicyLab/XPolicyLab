"""Test setup: the workspace that holds this XPolicyLab checkout and the checkout of the SANA repository on the path."""

import os
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[4]
SANA_ROOT = Path(os.environ.get("SANA_ROBOT_ROOT", Path(__file__).resolve().parents[1] / "Sana"))
for root in (WORKSPACE, SANA_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
