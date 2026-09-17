"""XPolicyLab data conversion command for the installed MoPA package."""

from pathlib import Path
import sys

XPL_ROOT = Path(__file__).resolve().parents[2]
if str(XPL_ROOT) not in sys.path:
    sys.path.insert(0, str(XPL_ROOT))

from mopa.integrations.xpolicylab import process_data_main


if __name__ == "__main__":
    process_data_main(policy_dir=Path(__file__).resolve().parent, xpl_root=XPL_ROOT)
