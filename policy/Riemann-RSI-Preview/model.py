"""Standard XPolicyLab entry for the declared Rule and WAM backends."""
from pathlib import Path
import sys

_root = str(Path(__file__).resolve().parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from rsi.official_adapter import Model

__all__ = ['Model']
