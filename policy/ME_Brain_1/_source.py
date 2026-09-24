"""Resolve the visible source checkout without duplicating model code."""

import sys
from pathlib import Path


def configure_source() -> Path:
    repository = Path(__file__).resolve().parent / "focus-vlwa"
    if not (repository / "src/focus_vlwa").is_dir():
        raise FileNotFoundError(f"Missing source checkout link: {repository}")
    repository = repository.resolve()
    for entry in (repository, repository / "src"):
        if str(entry) not in sys.path:
            sys.path.insert(0, str(entry))
    return repository
