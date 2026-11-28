"""Install the pinned Focus-VLWA transformers compatibility patch."""

from __future__ import annotations

import argparse
import shutil
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path


def install_transformers_patch() -> None:
    """Copy the versioned patch into the active transformers installation."""
    spec = find_spec("transformers")
    if spec is None or spec.submodule_search_locations is None:
        raise RuntimeError("transformers is not installed in the active environment")
    if version("transformers") != "4.53.2":
        raise RuntimeError("This compatibility patch requires transformers==4.53.2")
    destination = Path(next(iter(spec.submodule_search_locations)))
    source = Path(__file__).parents[1] / "model" / "transformers_patch"
    for path in source.rglob("*.py"):
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    print(f"Installed Focus-VLWA transformers patch into {destination}")


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    install_transformers_patch()


if __name__ == "__main__":
    main()
