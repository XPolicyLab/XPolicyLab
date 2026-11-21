"""WorldScape RSI Harness integration; external WAM is unchanged."""
import os
import sys

# Enforce for package imports and child interpreters, not just shell launchers.
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
