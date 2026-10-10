#!/usr/bin/env python3
"""Run RoboDojo's current main.py with unambiguous AppLauncher arguments.

RoboDojo declares ``--device_id`` before IsaacLab's AppLauncher registers
``--device``.  With argparse abbreviation enabled, AppLauncher's preliminary
``parse_known_args`` consumes ``--device cuda:N`` as ``--device_id cuda:N``
and exits before registering its own option.  This shim changes only the
parser default, then executes the untouched RoboDojo entry point.
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


_ORIGINAL_ARGUMENT_PARSER_INIT = argparse.ArgumentParser.__init__


def _no_abbrev_init(self, *args, **kwargs):
    kwargs.setdefault("allow_abbrev", False)
    _ORIGINAL_ARGUMENT_PARSER_INIT(self, *args, **kwargs)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: run_robodojo_main.py /path/to/main.py [args ...]")
    target = Path(sys.argv[1]).expanduser().resolve()
    if not target.is_file():
        raise FileNotFoundError(f"RoboDojo main.py was not found: {target}")

    sys.argv = [str(target), *sys.argv[2:]]
    argparse.ArgumentParser.__init__ = _no_abbrev_init
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
